# Weather Prediction Dashboard

A Streamlit app that combines live OpenWeather data with a historical dataset to:
- **Classify rain (Yes/No)** using a SMOTE-balanced RandomForest with cross-validated metrics.
- **Forecast temperature and humidity** with RandomForest regressors.
- **Explore data** via distributions, correlations, true boxplots, and IQR-based outlier detection.
 - **Assess flood risk** via an XGBoost regressor trained on engineered risk factors (0–12 scale), with automatic weather-informed features and a drought/recovery toggle.

## Project Structure
- **`streamlit_app.py`**: Main Streamlit dashboard.
- **`flood_module.py`**: Flood Risk tab (XGBoost model, CV metrics, feature importance, manual + automatic prediction).
- **`flood.csv`**: Sample flood dataset used by the Flood tab (features on a 0–12 scale, target in [0, 1]).
- **`weatherforcastingapp.py`**: Legacy console version (not required for the dashboard).
- **`weatherdataset.csv`**: Default historical dataset used if no upload is provided.

## Requirements
- Python 3.9+
- Install dependencies:
```bash
pip install -U streamlit pandas numpy scikit-learn seaborn matplotlib plotly altair requests xgboost
# Optional (enables SMOTE in rain classifier; app falls back gracefully if missing):
pip install -U imbalanced-learn
```

## Run
```bash
streamlit run streamlit_app.py
```
Then open the local URL shown in the terminal.

## Data
- By default, the app loads `weatherdataset.csv`. You can also upload a CSV via the sidebar.
- Required columns (must match exactly):
  - `MinTemp`, `MaxTemp`, `WindGustDir`, `WindGustSpeed`, `Humidity`, `Pressure`, `Temp`, `RainTomorrow`
- Notes:
  - `WindGustDir` should be compass labels like `N`, `NE`, `SW`, ...
  - `RainTomorrow` must be a categorical Yes/No label.
 - Flood tab uses `flood.csv` where engineered features (e.g., `MonsoonIntensity`, `DrainageSystems`, `Urbanization`, etc.) are on a **0–12 scale** and the target `FloodProbability` is continuous in [0, 1].

## How to Use
1. **Controls (Sidebar)**
   - Enter City, API Key (from OpenWeather), choose units (°C/°F).
   - Optionally upload a historical CSV.
   - Buttons: Fetch Current Weather, Train Models, Generate Forecast.
2. **Overview tab**
   - Shows current weather and map (when available).
3. **EDA tab**
   - Distributions (histograms), correlation heatmap.
   - True **boxplots** (pick columns) and **IQR outlier detection** with fences and outlier table.
4. **Model Performance tab**
   - Classification metrics (hold-out): Accuracy, Precision, Recall, F1, ROC-AUC.
   - Confusion matrix.
   - Cross-validated metrics (SMOTE + RF, 5-fold when SMOTE available; RF-only otherwise): CV Acc/Prec/Recall/F1/ROC-AUC.
   - Regression validation scatter plots: Actual vs Predicted (Temp/Humidity) with y=x line.
5. **Forecast Results tab**
   - Multi-hour projections for temperature and humidity.
   - Rain prediction label using a decision threshold (configurable in sidebar).
   - Expander: **Debug: Forecast internals** showing raw predictions pre/post unit conversion.
6. **Flood Risk Assessment tab**
   - Trains an XGBoost regressor on `flood.csv` with **L1/L2 regularization** and shows hold-out and **5-fold CV** metrics.
   - Manual inputs via sliders on a **0–12 scale** and Automatic mode inferred from current weather and forecast.
   - Toggle: **Drought/Recovery Watch** to down-weight monsoon intensity on dry days.
   - Feature importance and a downloadable model artifact.

## Models
- **Rain (classification)**
  - Pipeline: `SMOTE -> RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)` when `imbalanced-learn` is installed; otherwise RF-only pipeline.
  - Evaluation:
    - Hold-out split (20%) for quick metrics and confusion matrix.
    - 5-fold **Stratified** cross-validation for robust CV metrics.
- **Regression (Temp/Humidity)**
  - `RandomForestRegressor(n_estimators=100, random_state=42)`
  - Validation: 80/20 split; plots of Actual vs Predicted.
- **Flood Risk (regression)**
  - `XGBRegressor` with `reg_alpha=0.5`, `reg_lambda=1.0`, tuned for stability.
  - Features in 0–12 scale; target clipped to [0, 1] at inference for safety.
  - Metrics: Hold-out R²/MAE/RMSE and 5-fold CV R²/RMSE.

## OpenWeather API Key
- Get a key from https://openweathermap.org/ and paste it in the sidebar.
- The app uses the current weather endpoint.

## Troubleshooting
- **Missing columns error**: Ensure your CSV has the exact required columns listed above.
- **Boxplots/Outliers not showing**: Confirm you selected numeric columns in the EDA controls.
- **SMOTE not installed**: The rain classifier automatically falls back to RF-only. To enable SMOTE, install `imbalanced-learn`.
- **XGBoost missing**: Install `xgboost` (see Requirements) to enable the Flood tab.
- **API errors**: Provide your own API key; free tiers are rate-limited.

## Notes
- Forecast temperature values are continuous; rain is a categorical label derived from predicted probability and a decision threshold.
- Humidity forecasts are continuous and reported as numeric values.
- Flood features and sliders use a 0–12 scale to match the dataset; automatic inference is normalized to the same scale.

## License
Specify a license if needed (e.g., MIT).
