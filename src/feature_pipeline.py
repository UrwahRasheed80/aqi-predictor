# src/feature_pipeline.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from hopsworks_utils import save_features_to_hopsworks

load_dotenv()

API_KEY = os.getenv("AQICN_API_KEY")
CITY = os.getenv("CITY", "Islamabad")

def fetch_current_aqi():
    """Get current air quality and weather from AQICN"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data['status'] != 'ok':
        raise Exception(f"API error: {data.get('data')}")
    
    obs = data['data']
    iaqi = obs.get('iaqi', {})
    
    return {
        'timestamp': datetime.now(),
        'aqi': obs['aqi'],
        'pm25': iaqi.get('pm25', {}).get('v', np.nan),
        'pm10': iaqi.get('pm10', {}).get('v', np.nan),
        'temperature': iaqi.get('t', {}).get('v', np.nan),
        'humidity': iaqi.get('h', {}).get('v', np.nan),
        'wind_speed': iaqi.get('w', {}).get('v', np.nan),
    }

def compute_features(df):
    """Add time-based and lag features, fill missing lags with defaults"""
    df = df.sort_values('timestamp').copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['aqi_lag1'] = df['aqi'].shift(1)
    df['aqi_lag6'] = df['aqi'].shift(6)
    df['aqi_rolling_mean_6'] = df['aqi'].rolling(6).mean()
    
    # Fill NaN values (first rows) with sensible defaults
    df['aqi_lag1'].fillna(df['aqi'], inplace=True)
    df['aqi_lag6'].fillna(df['aqi'], inplace=True)
    df['aqi_rolling_mean_6'].fillna(df['aqi'], inplace=True)
    
    return df

def save_local_backup(df):
    """Save to CSV as backup"""
    os.makedirs("data", exist_ok=True)
    backup_path = "data/aqi_history.csv"
    if os.path.exists(backup_path):
        existing = pd.read_csv(backup_path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        combined.to_csv(backup_path, index=False)
    else:
        df.to_csv(backup_path, index=False)
    print(f"Backup saved to {backup_path}")

def main():
    print(f"Fetching AQI for {CITY} at {datetime.now()}")
    raw = fetch_current_aqi()
    new_row = pd.DataFrame([raw])
    
    # Load existing history to compute features on full dataset
    backup_path = "data/aqi_history.csv"
    if os.path.exists(backup_path):
        hist = pd.read_csv(backup_path)
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        df_combined = pd.concat([hist, new_row], ignore_index=True)
    else:
        df_combined = new_row
    
    # Compute features on the whole combined dataframe
    df_feat = compute_features(df_combined)
    
    # Save ALL rows (not just latest) to Hopsworks and backup
    save_features_to_hopsworks(df_feat)   # This now saves the entire feature set
    save_local_backup(df_feat)            # Save full history with features
    print(f"Saved {len(df_feat)} rows to feature store.")
    print("Feature pipeline completed successfully.")

if __name__ == "__main__":
    main()