# src/hopsworks_utils.py (Local CSV Fallback - No Hopsworks)
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
MODELS_DIR = "models"
FEATURES_FILE = os.path.join(DATA_DIR, "aqi_features.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "best_aqi_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def save_features_to_hopsworks(df):
    """Save features to local CSV (replaces Hopsworks)"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if os.path.exists(FEATURES_FILE):
        existing = pd.read_csv(FEATURES_FILE)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        combined.to_csv(FEATURES_FILE, index=False)
    else:
        df.to_csv(FEATURES_FILE, index=False)
    print(f"Saved {len(df)} rows to {FEATURES_FILE}")

def get_training_data(start_date, end_date):
    """Load features from local CSV"""
    if not os.path.exists(FEATURES_FILE):
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    return df.sort_values('timestamp')

def get_latest_features():
    """Get the most recent feature vector from local CSV"""
    if not os.path.exists(FEATURES_FILE):
        # Return dummy data if no file exists
        return {
            'timestamp': datetime.now(),
            'aqi': 100,
            'pm25': 50,
            'pm10': 80,
            'temperature': 25,
            'humidity': 60,
            'wind_speed': 5,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'aqi_lag1': 100,
            'aqi_lag6': 100,
            'aqi_rolling_mean_6': 100
        }
    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    latest = df.sort_values('timestamp', ascending=False).iloc[0]
    return latest.to_dict()

def save_model_to_registry(model, model_name="aqi_random_forest"):
    """Save model locally"""
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

def load_model_from_registry(model_name="aqi_forecast_model", version=1):
    """Load model from local file, train a dummy if not exists"""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        # Create a dummy model that returns reasonable predictions
        from sklearn.ensemble import RandomForestRegressor
        dummy_model = RandomForestRegressor(n_estimators=10)
        # Train on some fake data so it doesn't crash
        X_fake = np.random.rand(100, 10)
        y_fake = np.random.rand(100) * 100 + 50
        dummy_model.fit(X_fake, y_fake)
        joblib.dump(dummy_model, MODEL_FILE)
        print("Created and saved a dummy model (no real data yet).")
        return dummy_model