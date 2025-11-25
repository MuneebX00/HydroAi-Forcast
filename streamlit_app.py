import os
import io
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    HAVE_IMBLEARN = True
except Exception:
    SMOTE = None  # type: ignore
    ImbPipeline = None  # type: ignore
    HAVE_IMBLEARN = False
from flood_module import build_flood_tab


# =========================
# Page config
# =========================
st.set_page_config(
    page_title=" Weather Prediction Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Constants / Defaults
# =========================
DEFAULT_API_KEY = "8414dd804d65b13ee003e4f202cc57a6"
BASE_URL = "https://api.openweathermap.org/data/2.5/"
DEFAULT_DATASET = "weatherdataset.csv"


# =========================
# Helpers
# =========================
def c_to_f(celsius: float) -> float:
    return (celsius * 9.0 / 5.0) + 32.0


def ensure_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = [
        "MinTemp",
        "MaxTemp",
        "WindGustDir",
        "WindGustSpeed",
        "Humidity",
        "Pressure",
        "Temp",
        "RainTomorrow",
    ]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


@st.cache_data(show_spinner=False)
def load_historical_data(file_buffer: io.BytesIO | None, fallback_path: str) -> pd.DataFrame:
    try:
        if file_buffer is not None:
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_csv(fallback_path)
        # Cleaning: drop NA and duplicates to follow base script behavior
        df = df.dropna().drop_duplicates()
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def get_current_weather(city: str, api_key: str, units: str = "metric") -> Dict:
    url = f"{BASE_URL}weather?q={city}&appid={api_key}&units={units}"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Defensive parsing
    wind = data.get("wind", {})
    main = data.get("main", {})
    sys = data.get("sys", {})
    weather_list = data.get("weather", [{"description": "unknown"}])
    coord = data.get("coord", {})

    return {
        "city": data.get("name", city),
        "current_temp": float(main.get("temp", np.nan)),
        "feels_like": float(main.get("feels_like", np.nan)),
        "temp_min": float(main.get("temp_min", np.nan)),
        "temp_max": float(main.get("temp_max", np.nan)),
        "humidity": float(main.get("humidity", np.nan)),
        "description": weather_list[0].get("description", "unknown"),
        "country": sys.get("country", ""),
        "wind_deg": float(wind.get("deg", 0.0)),
        "pressure": float(main.get("pressure", np.nan)),
        "Wind_Gust_Speed": float(wind.get("speed", np.nan)),
        "lat": coord.get("lat"),
        "lon": coord.get("lon"),
    }


def get_daily_forecast(lat: float, lon: float, api_key: str, cnt: int = 15, units: str = "metric") -> pd.DataFrame:
    """Fetch 15-day daily forecast from OpenWeather if available.
    Uses the 16-day daily forecast endpoint (may require appropriate plan). Falls back gracefully on error.
    """
    try:
        url = (
            f"{BASE_URL}forecast/daily?lat={lat}&lon={lon}&cnt={cnt}&appid={api_key}&units={units}"
        )
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
        payload = resp.json()
        days = payload.get("list", [])
        rows: List[Dict] = []
        for d in days:
            ts = d.get("dt")
            date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            temp_obj = d.get("temp", {})
            humidity = d.get("humidity")
            pop = d.get("pop", 0.0)
            weather = d.get("weather", [{}])
            desc = weather[0].get("description", "") if weather else ""
            rows.append({
                "time": date_str,
                "temperature": float(temp_obj.get("day", np.nan)),
                "temp_min": float(temp_obj.get("min", np.nan)),
                "temp_max": float(temp_obj.get("max", np.nan)),
                "humidity": float(humidity) if humidity is not None else np.nan,
                "rain_prediction": "Yes" if float(pop or 0.0) >= 0.5 else ("Yes" if "rain" in desc.lower() else "No"),
                "description": desc,
                "pop": float(pop or 0.0),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch daily forecast: {e}")


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, LabelEncoder]:
    # Follow base script logic: separate encoders for wind direction and rain label
    df = df.copy()
    le_wind = LabelEncoder()
    le_rain = LabelEncoder()

    # Ensure WindGustDir as string then encode
    df["WindGustDir"] = df["WindGustDir"].astype(str)
    le_wind.fit(df["WindGustDir"].unique())
    df["WindGustDir"] = le_wind.transform(df["WindGustDir"])
    df["RainTomorrow"] = le_rain.fit_transform(df["RainTomorrow"])

    X = df[[
        "MinTemp",
        "MaxTemp",
        "WindGustDir",
        "WindGustSpeed",
        "Humidity",
        "Pressure",
        "Temp",
    ]]
    y = df["RainTomorrow"]
    return X, y, le_wind, le_rain


@st.cache_resource(show_spinner=False)
def train_rain_model(X: pd.DataFrame, y: pd.Series) -> Tuple[object, Dict[str, float], np.ndarray]:
    # Hold-out split for confusion matrix and quick metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if HAVE_IMBLEARN and SMOTE is not None and ImbPipeline is not None:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ])
    else:
        pipeline = SkPipeline([
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc}
    cm = confusion_matrix(y_test, y_pred)
    return pipeline, metrics, cm


def cross_validate_rain_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    if HAVE_IMBLEARN and SMOTE is not None and ImbPipeline is not None:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ])
    else:
        pipeline = SkPipeline([
            ("rf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    metrics = {
        "CV_Accuracy": float(results["test_accuracy"].mean()),
        "CV_Precision": float(results["test_precision"].mean()),
        "CV_Recall": float(results["test_recall"].mean()),
        "CV_F1": float(results["test_f1"].mean()),
        "CV_ROC_AUC": float(results["test_roc_auc"].mean()),
    }
    return metrics


def prepare_regression_data(df: pd.DataFrame, feature: str) -> Tuple[np.ndarray, np.ndarray]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    for i in range(len(df) - 1):
        x_vals.append(df[feature].iloc[i])
        y_vals.append(df[feature].iloc[i + 1])
    X = np.array(x_vals).reshape(-1, 1)
    y = np.array(y_vals)
    return X, y


@st.cache_resource(show_spinner=False)
def train_regression_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def predict_future(model: RandomForestRegressor, current_value: float, horizon: int = 5) -> List[float]:
    predictions: List[float] = [current_value]
    for _ in range(horizon):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(float(next_value[0]))
    return predictions[1:]


def compute_feature_importance(model: object | RandomForestClassifier, feature_names: List[str]) -> pd.DataFrame:
    rf = None
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
    elif hasattr(model, "feature_importances_"):
        rf = model
    if rf is not None and hasattr(rf, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})


def build_overview_tab(current_weather: Dict, unit: str):
    if current_weather is None:
        st.info("Use 'Fetch Current Weather' to load live data.")
        return

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    temp = current_weather["current_temp"]
    feels = current_weather["feels_like"]

    if unit == "Fahrenheit":
        temp = c_to_f(temp)
        feels = c_to_f(feels)

    col1.metric("City", current_weather.get("city", "-"))
    col2.metric("Country", current_weather.get("country", "-"))
    col3.metric("Current Temp", f"{round(temp, 1)}°{'F' if unit=='Fahrenheit' else 'C'}")
    col4.metric("Feels Like", f"{round(feels, 1)}°{'F' if unit=='Fahrenheit' else 'C'}")
    col5.metric("Humidity", f"{int(current_weather.get('humidity', 0))}%")
    col6.metric("Condition", current_weather.get("description", "-"))

    if current_weather.get("lat") is not None and current_weather.get("lon") is not None:
        st.map(pd.DataFrame({"lat": [current_weather["lat"]], "lon": [current_weather["lon"]]}))


def build_eda_tab(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Upload or load a dataset to view EDA.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.subheader("Distributions")
    cols = st.columns(3)
    for i, col in enumerate(numeric_cols[:6]):
        with cols[i % 3]:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30)),
                y="count()",
                tooltip=[col],
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

    st.subheader("Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig, clear_figure=True)
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

    st.subheader("Boxplots")
    if numeric_cols:
        selected_box_cols = st.multiselect(
            "Select columns for boxplots",
            options=numeric_cols,
            default=[c for c in ["Temp", "Humidity"] if c in numeric_cols] or numeric_cols[:2],
        )
        cols_box = st.columns(2)
        for i, col in enumerate(selected_box_cols):
            with cols_box[i % 2]:
                fig = px.box(df, y=col, points="outliers", color_discrete_sequence=["#1f77b4"])
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection (IQR)")
    if numeric_cols:
        target_col = st.selectbox("Column", options=numeric_cols, index=0)
        series = df[target_col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[target_col] < lower) | (df[target_col] > upper)
        n_outliers = int(mask.sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1", f"{q1:.3f}")
        c2.metric("Q3", f"{q3:.3f}")
        c3.metric("Lower Fence", f"{lower:.3f}")
        c4.metric("Upper Fence", f"{upper:.3f}")
        st.write(f"Outliers in `{target_col}`: {n_outliers}")
        if n_outliers > 0:
            st.dataframe(df.loc[mask].head(200), use_container_width=True)


def build_performance_tab(rain_metrics: Dict[str, float], rain_model: RandomForestClassifier, feature_names: List[str], cm: np.ndarray | None):
    if rain_metrics is None or rain_model is None:
        st.info("Train the models to view performance.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{rain_metrics['Accuracy']:.4f}")
    c2.metric("Precision", f"{rain_metrics['Precision']:.4f}")
    c3.metric("Recall", f"{rain_metrics['Recall']:.4f}")
    c4.metric("F1", f"{rain_metrics['F1']:.4f}")
    c5.metric("ROC-AUC", f"{rain_metrics['ROC_AUC']:.4f}")

    cvm = st.session_state.get("rain_cv_metrics")
    if cvm is not None:
        st.subheader("Cross-Validated Metrics (SMOTE + RF, 5-fold)")
        cv1, cv2, cv3, cv4, cv5 = st.columns(5)
        cv1.metric("CV Acc", f"{cvm['CV_Accuracy']:.4f}")
        cv2.metric("CV Prec", f"{cvm['CV_Precision']:.4f}")
        cv3.metric("CV Recall", f"{cvm['CV_Recall']:.4f}")
        cv4.metric("CV F1", f"{cvm['CV_F1']:.4f}")
        cv5.metric("CV ROC-AUC", f"{cvm['CV_ROC_AUC']:.4f}")

    st.subheader("Feature Importance")
    imp_df = compute_feature_importance(rain_model, feature_names)
    fig = px.bar(imp_df, x="feature", y="importance", color="importance", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    if cm is not None and cm.size == 4:
        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm, clear_figure=True)

    # Regression validation plots
    if st.session_state.get("temp_valid") is not None or st.session_state.get("hum_valid") is not None:
        st.subheader("Regression Validation: Actual vs Predicted")
        c1, c2 = st.columns(2)
        if st.session_state.get("temp_valid") is not None:
            tv = st.session_state.temp_valid
            df_tv = pd.DataFrame({"Actual": tv["y"], "Predicted": tv["y_pred"]})
            with c1:
                fig1 = px.scatter(df_tv, x="Actual", y="Predicted", title="Temperature")
                # Add y=x reference line
                min_v = float(min(df_tv.min()))
                max_v = float(max(df_tv.max()))
                fig1.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="red", dash="dash"))
                st.plotly_chart(fig1, use_container_width=True)
        if st.session_state.get("hum_valid") is not None:
            hv = st.session_state.hum_valid
            df_hv = pd.DataFrame({"Actual": hv["y"], "Predicted": hv["y_pred"]})
            with c2:
                fig2 = px.scatter(df_hv, x="Actual", y="Predicted", title="Humidity")
                min_vh = float(min(df_hv.min()))
                max_vh = float(max(df_hv.max()))
                fig2.add_shape(type="line", x0=min_vh, y0=min_vh, x1=max_vh, y1=max_vh, line=dict(color="red", dash="dash"))
                st.plotly_chart(fig2, use_container_width=True)


def build_forecast_tab(forecast_df: pd.DataFrame):
    if forecast_df is None or forecast_df.empty:
        st.info("Generate forecast to view results.")
        return

    st.subheader("Forecast Time Series")
    melted = forecast_df.melt(id_vars=["time"], value_vars=["temperature", "humidity"], var_name="metric", value_name="value")
    fig = px.line(melted, x="time", y="value", color="metric", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast (CSV)", data=csv, file_name="forecast.csv", mime="text/csv")


def main():
    # Sidebar
    with st.sidebar:
        st.title("Controls")
        city = st.text_input("City", value="Karachi")
        api_key = st.text_input("OpenWeather API Key", value=DEFAULT_API_KEY, type="password")
        unit = st.radio("Unit", ["Celsius", "Fahrenheit"], index=0, horizontal=True)
        uploaded = st.file_uploader("Upload historical CSV (optional)", type=["csv"]) 
        st.caption("If not provided, the app will use weatherdataset.csv from the project folder.")
        st.divider()
        st.write("Model: RandomForest")
        forecast_mode = st.radio("Forecast mode", ["Short-term (hours)", "Long-term (days)"], index=0)
        if forecast_mode == "Short-term (hours)":
            forecast_horizon = st.slider("Forecast horizon (hours)", min_value=1, max_value=48, value=6)
            day_horizon = None
        else:
            day_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=15, value=15)
            forecast_horizon = None
        decision_threshold = st.slider("Rain decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        fetch_btn = st.button("Fetch Current Weather")
        train_btn = st.button("Train Models")
        forecast_btn = st.button("Generate Forecast")
        st.divider()
        flood_enabled = st.toggle("Enable Flood Module", value=True)

    st.title("🌦️ Weather Prediction Dashboard")

    # Session state
    if "historical" not in st.session_state:
        st.session_state.historical = None
    if "current_weather" not in st.session_state:
        st.session_state.current_weather = None
    if "le_wind" not in st.session_state:
        st.session_state.le_wind = None
    if "le_rain" not in st.session_state:
        st.session_state.le_rain = None
    if "rain_model" not in st.session_state:
        st.session_state.rain_model = None
    if "rain_metrics" not in st.session_state:
        st.session_state.rain_metrics = None
    if "rain_cm" not in st.session_state:
        st.session_state.rain_cm = None
    if "rain_cv_metrics" not in st.session_state:
        st.session_state.rain_cv_metrics = None
    if "temp_model" not in st.session_state:
        st.session_state.temp_model = None
    if "hum_model" not in st.session_state:
        st.session_state.hum_model = None
    if "forecast_df" not in st.session_state:
        st.session_state.forecast_df = None
    if "temp_valid" not in st.session_state:
        st.session_state.temp_valid = None
    if "hum_valid" not in st.session_state:
        st.session_state.hum_valid = None

    # Load data
    with st.spinner("Loading historical data..."):
        try:
            df = load_historical_data(uploaded, DEFAULT_DATASET)
            valid, missing = ensure_required_columns(df)
            if not valid:
                st.error(f"Missing required columns: {missing}")
            else:
                st.session_state.historical = df
        except Exception as e:
            st.warning(str(e))

    # Handle actions
    if fetch_btn:
        try:
            units_param = "metric" if unit == "Celsius" else "imperial"
            with st.spinner("Fetching current weather..."):
                st.session_state.current_weather = get_current_weather(city, api_key, units=units_param)
            st.success("Fetched current weather.")
        except Exception as e:
            st.error(f"Failed to fetch weather: {e}")

    if train_btn:
        if st.session_state.historical is None:
            st.error("No historical data loaded.")
        else:
            try:
                with st.spinner("Preparing data and training models..."):
                    X, y, le_wind, le_rain = prepare_data(st.session_state.historical)
                    rain_model, rain_metrics, cm = train_rain_model(X, y)
                    cv_metrics = cross_validate_rain_model(X, y)

                    # Regression data for Temp and Humidity with validation split
                    X_temp_all, y_temp_all = prepare_regression_data(st.session_state.historical, "Temp")
                    X_hum_all, y_hum_all = prepare_regression_data(st.session_state.historical, "Humidity")

                    X_temp_tr, X_temp_te, y_temp_tr, y_temp_te = train_test_split(
                        X_temp_all, y_temp_all, test_size=0.2, random_state=42
                    )
                    X_hum_tr, X_hum_te, y_hum_tr, y_hum_te = train_test_split(
                        X_hum_all, y_hum_all, test_size=0.2, random_state=42
                    )

                    temp_model = train_regression_model(X_temp_tr, y_temp_tr)
                    hum_model = train_regression_model(X_hum_tr, y_hum_tr)

                    y_temp_pred = temp_model.predict(X_temp_te)
                    y_hum_pred = hum_model.predict(X_hum_te)

                st.session_state.le_wind = le_wind
                st.session_state.le_rain = le_rain
                st.session_state.rain_model = rain_model
                st.session_state.rain_metrics = rain_metrics
                st.session_state.rain_cm = cm
                st.session_state.rain_cv_metrics = cv_metrics
                st.session_state.temp_model = temp_model
                st.session_state.hum_model = hum_model
                st.session_state.temp_valid = {"y": y_temp_te, "y_pred": y_temp_pred}
                st.session_state.hum_valid = {"y": y_hum_te, "y_pred": y_hum_pred}
                st.success("Models trained.")
            except Exception as e:
                st.error(f"Failed to train models: {e}")

    if forecast_btn:
        cw = st.session_state.current_weather
        df_hist = st.session_state.historical
        rain_model = st.session_state.rain_model
        le_wind = st.session_state.le_wind
        temp_model = st.session_state.temp_model
        hum_model = st.session_state.hum_model

        # Avoid DataFrame equality comparison; use identity checks
        if forecast_mode == "Short-term (hours)":
            if any(x is None for x in (cw, df_hist, rain_model, le_wind, temp_model, hum_model)):
                st.error("Please fetch current weather and train models first.")
            else:
                try:
                    with st.spinner("Generating forecast..."):
                        wind_deg = cw["wind_deg"] % 360
                        compass_points = [
                            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                            ("NNW", 326.25, 348.75)
                        ]
                        compass_direction = next((p for p, start, end in compass_points if start <= wind_deg < end), None)
                        if compass_direction and compass_direction in le_wind.classes_:
                            compass_direction_encoded = int(le_wind.transform([compass_direction])[0])
                        else:
                            compass_direction_encoded = -1

                        # Prepare current data point to match training features
                        current_data = pd.DataFrame([{
                            "MinTemp": float(cw["temp_min"]),
                            "MaxTemp": float(cw["temp_max"]),
                            "WindGustDir": compass_direction_encoded,
                            "WindGustSpeed": float(cw["Wind_Gust_Speed"]),
                            "Humidity": float(cw["humidity"]),
                            "Pressure": float(cw["pressure"]),
                            "Temp": float(cw["current_temp"]),
                        }])

                        # Rain prediction with decision threshold
                        prob = float(rain_model.predict_proba(current_data)[0, 1])
                        rain_pred = 1 if prob >= decision_threshold else 0

                        # Build horizon timestamps (next rounded hour)
                        now = datetime.now()
                        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                        times = [next_hour + timedelta(hours=i) for i in range(forecast_horizon or 6)]

                        # Regression forecasts starting from current observed values
                        future_temp_raw = predict_future(temp_model, float(cw["temp_min"]), horizon=(forecast_horizon or 6))
                        future_hum_raw = predict_future(hum_model, float(cw["humidity"]), horizon=(forecast_horizon or 6))

                        # Unit conversion for display if needed
                        future_temp_display = list(future_temp_raw)
                        if unit == "Fahrenheit":
                            future_temp_display = [c_to_f(t) for t in future_temp_raw]

                        st.session_state.forecast_df = pd.DataFrame({
                            "time": [t.strftime("%H:00") for t in times],
                            "temperature": [round(t, 2) for t in future_temp_display],
                            "humidity": [round(h, 2) for h in future_hum_raw],
                            "rain_prediction": ["Yes" if rain_pred else "No"] * len(times),
                        })
                        with st.expander("Debug: Forecast internals", expanded=False):
                            st.write("Raw temperature predictions (C):", [round(v, 3) for v in future_temp_raw])
                            st.write("Displayed temperature predictions:", [round(v, 3) for v in future_temp_display])
                            st.write("Raw humidity predictions:", [round(v, 3) for v in future_hum_raw])
                    st.success("Forecast generated.")
                except Exception as e:
                    st.error(f"Failed to generate forecast: {e}")
        else:
            # Long-term daily forecast path (no ML required)
            if cw is None:
                st.error("Please fetch current weather first (to get location).")
            else:
                try:
                    with st.spinner("Fetching 15-day daily forecast..."):
                        units_param = "metric" if unit == "Celsius" else "imperial"
                        daily_df = get_daily_forecast(cw.get("lat"), cw.get("lon"), api_key, cnt=(day_horizon or 15), units=units_param)
                        # Convert temperature to Fahrenheit if needed (API respects units, so only ensure rounding)
                        daily_df["temperature"] = daily_df["temperature"].astype(float)
                        daily_df["humidity"] = daily_df["humidity"].astype(float)
                        st.session_state.forecast_df = daily_df[["time", "temperature", "humidity", "rain_prediction"]]
                    st.success("Daily forecast generated.")
                except Exception as e:
                    st.error(f"Failed to fetch daily forecast: {e}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "EDA", "Model Performance", "Forecast Results", "Flood Risk Assessment", "About"])
    with tab1:
        build_overview_tab(st.session_state.current_weather, unit)
    with tab2:
        build_eda_tab(st.session_state.historical)
    with tab3:
        feature_names = [
            "MinTemp", "MaxTemp", "WindGustDir", "WindGustSpeed", "Humidity", "Pressure", "Temp"
        ]
        build_performance_tab(st.session_state.rain_metrics, st.session_state.rain_model, feature_names, st.session_state.rain_cm)
    with tab4:
        build_forecast_tab(st.session_state.forecast_df)
    with tab5:
        build_flood_tab(
            flood_enabled,
            st.session_state.current_weather,
            st.session_state.forecast_df,
        )
    with tab6:
        st.markdown(
            """
            **About**

            This dashboard fetches real-time weather via OpenWeather, cleans a historical dataset,
            trains a RandomForest classifier for rain prediction and RandomForest regressors for
            temperature and humidity, and produces:

            - Short-term forecasts (up to 48 hours) using an autoregressive ML approach.
            - Long-term daily forecasts (up to 15 days) fetched from the OpenWeather daily API.

            - Historical data columns required: `MinTemp`, `MaxTemp`, `WindGustDir`, `WindGustSpeed`, `Humidity`, `Pressure`, `Temp`, `RainTomorrow`.
            - Use sidebar controls to fetch weather, train models, and generate forecasts.
            - Metrics shown include classification metrics for the rain model.
            - Forecasts are generated autoregressively from the current observation.
            - API usage may be rate-limited; use your own API key if you encounter errors.
            """
        )


if __name__ == "__main__":
    main()




