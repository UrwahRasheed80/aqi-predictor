# src/dashboard.py (Simplified, no SHAP)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
from hopsworks_utils import get_latest_features, load_model_from_registry

st.set_page_config(page_title="AQI Predictor", layout="wide")
st.title("🌍 Pearls AQI Predictor – Next 3 Days Forecast")

@st.cache_resource
def load_resources():
    """Load model and feature columns"""
    model = load_model_from_registry("aqi_forecast_model", version=1)
    feature_cols = ['hour', 'day_of_week', 'pm25', 'temperature', 
                    'humidity', 'wind_speed', 'aqi_lag1', 'aqi_lag6', 'aqi_rolling_mean_6']
    return model, feature_cols

def predict_next_3_days(model, latest, feature_cols):
    """Iteratively predict next 72 hours using last known values"""
    predictions = []
    current = latest.copy()
    for h in range(1, 73):  # 72 hours = 3 days
        hour_of_day = (latest['hour'] + h) % 24
        dow = (latest['day_of_week'] + (h // 24)) % 7
        # Build feature vector using available keys
        features = []
        for col in feature_cols:
            if col == 'hour':
                features.append(hour_of_day)
            elif col == 'day_of_week':
                features.append(dow)
            else:
                features.append(current.get(col, 100))
        pred = model.predict([features])[0]
        predictions.append(pred)
        # Update current for next iteration
        current['aqi'] = pred
        current['hour'] = hour_of_day
        current['day_of_week'] = dow
    return predictions

st.sidebar.header("Settings")
city = st.sidebar.text_input("City", value=os.getenv("CITY", "Delhi"))
if st.sidebar.button("Refresh Predictions"):
    st.cache_resource.clear()
    st.rerun()

# Load resources
model, feature_cols = load_resources()

# Get latest features (from local CSV)
try:
    latest = get_latest_features()
except:
    st.error("No feature data found. Run feature_pipeline.py first.")
    st.stop()

# Display current AQI
current_aqi = latest.get('aqi', 'N/A')
st.metric("Current AQI", int(current_aqi) if current_aqi != 'N/A' else 'N/A')

col1, col2 = st.columns(2)

# Historical plot (if data exists)
hist_path = "data/aqi_history.csv"
if os.path.exists(hist_path):
    hist_df = pd.read_csv(hist_path)
    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
    fig_hist = px.line(hist_df.tail(168), x='timestamp', y='aqi', title="Last 7 Days AQI")
    col1.plotly_chart(fig_hist, use_container_width=True)
else:
    col1.info("No historical data yet. Run feature pipeline to collect data.")

# Forecast plot
try:
    future_preds = predict_next_3_days(model, latest, feature_cols)
    future_times = [datetime.now() + timedelta(hours=i) for i in range(1, 73)]
    forecast_df = pd.DataFrame({'timestamp': future_times, 'predicted_aqi': future_preds})
    fig_fore = px.line(forecast_df, x='timestamp', y='predicted_aqi', title="Forecast Next 3 Days")
    col2.plotly_chart(fig_fore, use_container_width=True)
    
    # Alert if hazardous
    max_pred = max(future_preds)
    if max_pred > 300:
        st.error("🚨 ALERT: Hazardous AQI levels predicted in the next 3 days! Take precautions.")
    elif max_pred > 200:
        st.warning("⚠️ Very Unhealthy AQI expected. Limit outdoor activities.")
    elif max_pred > 150:
        st.info("Unhealthy for sensitive groups.")
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.caption("Model trained on historical AQI data. Predictions improve with more data.")