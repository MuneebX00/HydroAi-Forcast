import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import pickle

FEATURE_COLUMNS: List[str] = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]
TARGET = "FloodProbability"


@st.cache_data(show_spinner=False)
def load_flood_data(csv_path: str = "flood.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


@st.cache_resource(show_spinner=False)
def train_flood_regressor(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model


def evaluate(model: XGBRegressor, X_te: pd.DataFrame, y_te: pd.Series) -> Dict[str, float]:
    y_pred = np.clip(model.predict(X_te), 0.0, 1.0)
    mse = float(mean_squared_error(y_te, y_pred))
    rmse = float(np.sqrt(mse))
    return {
        "R2": float(r2_score(y_te, y_pred)),
        "MAE": float(mean_absolute_error(y_te, y_pred)),
        "RMSE": rmse,
    }


def cross_validate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s: List[float] = []
    rmses: List[float] = []
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        m = XGBRegressor(
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            reg_alpha=model.reg_alpha,
            reg_lambda=model.reg_lambda,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        m.fit(X_tr, y_tr)
        pred = np.clip(m.predict(X_te), 0.0, 1.0)
        r2s.append(float(r2_score(y_te, pred)))
        mse = float(mean_squared_error(y_te, pred))
        rmses.append(float(np.sqrt(mse)))
    return {"CV_R2": float(np.mean(r2s)), "CV_RMSE": float(np.mean(rmses))}


def risk_label(prob: float) -> Tuple[str, str]:
    if prob < 0.3:
        return "Low", "#00c853"
    if prob < 0.6:
        return "Moderate", "#ffab00"
    return "High", "#d50000"


def manual_inputs_ui(scale_max: float = 12.0) -> pd.DataFrame:
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(FEATURE_COLUMNS):
        with cols[i % 2]:
            values[f] = st.slider(f, 0.0, scale_max, scale_max/3, 0.5)
    return pd.DataFrame([values])


def infer_features_from_weather(current_weather: Dict | None, forecast_df: pd.DataFrame | None, drought_mode: bool = False) -> pd.DataFrame | None:
    if current_weather is None:
        return None
    desc = str(current_weather.get("description", "")).lower()
    humidity = float(current_weather.get("humidity", np.nan))
    temp_c = float(current_weather.get("current_temp", np.nan))
    # Binary/rule flags
    rain_flag = 1.0 if ("rain" in desc or (isinstance(forecast_df, pd.DataFrame) and "rain_prediction" in forecast_df.columns and (forecast_df["rain_prediction"].astype(str).str.lower() == "yes").any())) else 0.0

    # Map to 0–12 scale (dataset scale). Keep values conservative when dry or drought_mode.
    base_hum = humidity if np.isfinite(humidity) else 60.0
    monsoon_raw = (base_hum / 100.0) * 6.0 + rain_flag * 6.0
    if drought_mode:
        monsoon_raw *= 0.4
    monsoon_intensity = float(np.clip(monsoon_raw, 0.0, 12.0))

    climate_change = float(np.interp(temp_c if np.isfinite(temp_c) else 28.0,
                                    [-5, 0, 15, 25, 35, 45],
                                    [2, 3, 5, 8, 10, 12]))
    drainage_systems = float(np.clip(8.0 - rain_flag * 2.5 - (base_hum/100.0) * 2.0, 0.0, 12.0))
    urbanization = 7.0
    river_management = float(np.clip(6.0 + rain_flag * 1.5, 0.0, 12.0))
    ineffective_prep = float(np.clip(6.0 + rain_flag * 1.5, 0.0, 12.0))
    siltation = 6.0
    dams_quality = float(np.clip(6.0 - rain_flag * 1.0, 0.0, 12.0))
    coastal_vulnerability = float(np.clip(6.0 + (1.5 if "storm" in desc else 0.0), 0.0, 12.0))
    population_score = 6.5
    inadequate_planning = 6.5
    deteriorating_infra = float(np.clip(6.0 + rain_flag * 1.5, 0.0, 12.0))
    landslides = float(np.clip(5.0 + (1.0 if ("haze" in desc or "storm" in desc) else 0.0), 0.0, 12.0))
    watersheds = 6.0
    wetland_loss = 6.0
    deforestation = 6.0
    encroachments = 6.0
    agricultural_practices = 6.0
    base = {
        "MonsoonIntensity": monsoon_intensity,
        "TopographyDrainage": 6.5,
        "RiverManagement": float(np.clip(river_management, 0, 12)),
        "Deforestation": deforestation,
        "Urbanization": urbanization,
        "ClimateChange": float(np.clip(climate_change, 0, 12)),
        "DamsQuality": float(np.clip(dams_quality, 0, 12)),
        "Siltation": siltation,
        "AgriculturalPractices": agricultural_practices,
        "Encroachments": encroachments,
        "IneffectiveDisasterPreparedness": float(np.clip(ineffective_prep, 0, 12)),
        "DrainageSystems": drainage_systems,
        "CoastalVulnerability": float(np.clip(coastal_vulnerability, 0, 12)),
        "Landslides": float(np.clip(landslides, 0, 12)),
        "Watersheds": 6.0,
        "DeterioratingInfrastructure": float(np.clip(deteriorating_infra, 0, 12)),
        "PopulationScore": population_score,
        "WetlandLoss": wetland_loss,
        "InadequatePlanning": inadequate_planning,
        "PoliticalFactors": 6.0,
    }
    return pd.DataFrame([base])


def feature_importance_df(model: XGBRegressor) -> pd.DataFrame:
    imps = getattr(model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS)))
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imps}).sort_values("importance", ascending=False)


def render_progress(prob: float):
    label, color = risk_label(prob)
    st.progress(min(max(prob, 0.0), 1.0))
    st.metric("Flood Probability", f"{prob:.2f}", help=f"Risk: {label}")
    if label == "Low":
        st.success("Low flood risk")
    elif label == "Moderate":
        st.warning("Moderate flood risk")
    else:
        st.error("High flood risk")


def build_flood_tab(enabled: bool, current_weather: Dict | None, forecast_df: pd.DataFrame | None):
    st.header("Smart Flood Risk Assessment")
    if not enabled:
        st.info("Enable the flood module from the sidebar to use this tab.")
        return

    try:
        with st.spinner("Loading flood dataset..."):
            df = load_flood_data("flood.csv")
    except Exception as e:
        st.error(f"Failed to load flood dataset: {e}")
        return

    with st.expander("Data Overview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)
        desc = df[FEATURE_COLUMNS + [TARGET]].describe().T
        st.dataframe(desc, use_container_width=True)
        st.caption(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} cols")
        corr = df[FEATURE_COLUMNS + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig, clear_figure=True)

    with st.spinner("Training flood model..."):
        X = df[FEATURE_COLUMNS]
        y = df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_flood_regressor(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        cv_metrics = cross_validate(model, X, y, n_splits=5)

    with st.expander("Model Performance", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['R2']:.4f}")
        c2.metric("MAE", f"{metrics['MAE']:.4f}")
        c3.metric("RMSE", f"{metrics['RMSE']:.4f}")
        c1a, c2a = st.columns(2)
        c1a.metric("CV R² (5-fold)", f"{cv_metrics['CV_R2']:.4f}")
        c2a.metric("CV RMSE (5-fold)", f"{cv_metrics['CV_RMSE']:.4f}")
        imp_df = feature_importance_df(model)
        fig_imp = px.bar(imp_df, x="feature", y="importance", color="importance", color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)
        buf = pickle.dumps(model)
        st.download_button("Download Flood Model", data=buf, file_name="flood_model.pkl", mime="application/octet-stream")

    st.subheader("Manual Prediction")
    manual_df = manual_inputs_ui(scale_max=12.0)
    if st.button("Predict Flood Risk", type="primary"):
        prob = float(np.clip(model.predict(manual_df)[0], 0.0, 1.0))
        render_progress(prob)

    st.subheader("Smart Automatic Flood Risk")
    drought_mode = st.toggle("Drought/Recovery Watch (down-weights monsoon today)", value=False)
    auto_feats = infer_features_from_weather(current_weather, forecast_df, drought_mode=drought_mode)
    if auto_feats is None:
        st.info("Fetch current weather to enable automatic flood risk prediction.")
    else:
        prob_auto = float(np.clip(model.predict(auto_feats)[0], 0.0, 1.0))
        render_progress(prob_auto)
        st.caption("Based on current weather conditions.")

    st.subheader("Visualization")
    if forecast_df is not None and not forecast_df.empty:
        horizon = min(len(forecast_df), 24)
        probs = []
        for i in range(horizon):
            feats = infer_features_from_weather(current_weather, forecast_df.iloc[[i]])
            if feats is None:
                feats = auto_feats
            if feats is None:
                break
            probs.append(float(np.clip(model.predict(feats)[0], 0.0, 1.0)))
        if probs:
            ts_df = pd.DataFrame({"t": list(range(len(probs))), "prob": probs})
            fig_ts = px.line(ts_df, x="t", y="prob", markers=True, range_y=[0, 1], title="Forecasted Flood Risk Trend")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No forecast available to render trend.")
