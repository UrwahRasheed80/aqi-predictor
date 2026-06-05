# src/training_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os
from hopsworks_utils import get_training_data, save_model_to_registry

def load_and_prepare_data():
    """Load data, handle missing values, drop useless columns"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = get_training_data(start_date, end_date)
    
    if df is None or len(df) < 5:
        print(f"Warning: Only {len(df) if df is not None else 0} rows available. Need at least 5.")
        return None
    
    feature_cols = ['hour', 'day_of_week', 'pm25', 'pm10', 'temperature', 
                    'humidity', 'wind_speed', 'aqi_lag1', 'aqi_lag6', 'aqi_rolling_mean_6']
    
    # Only keep columns that exist in the dataframe
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].copy()
    y = df['aqi']
    
    # Drop columns that are ALL NaN (e.g., pm10 if completely missing)
    cols_before = X.columns.tolist()
    X = X.dropna(axis=1, how='all')
    dropped_cols = set(cols_before) - set(X.columns)
    if dropped_cols:
        print(f"Dropped columns that were all NaN: {dropped_cols}")
    
    # For remaining columns, drop rows with any NaN
    initial_rows = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    rows_dropped = initial_rows - len(X)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to remaining NaN values. {len(X)} rows left.")
    
    if len(X) < 5:
        print("Too few rows after cleaning. Need at least 5.")
        return None
    
    # Impute any remaining NaNs (should be none, but just in case)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print(f"Final dataset: {len(X_imputed)} rows, {len(X_imputed.columns)} features.")
    return X_imputed, y, X_imputed.columns.tolist()

def train_models(X, y, feature_cols):
    """Train models and return best"""
    if len(X) < 5:
        print("Not enough data for training.")
        return None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    results = {}
    best_model = None
    best_mae = float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        if mae < best_mae:
            best_mae = mae
            best_model = model
    
    # Save model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_aqi_model.pkl")
    print(f"Best model saved to models/best_aqi_model.pkl (MAE: {best_mae:.2f})")
    return best_model, results

def main():
    print("Starting training pipeline...")
    result = load_and_prepare_data()
    if result is None:
        print("Insufficient data. Skipping training.")
        return
    X, y, feature_cols = result
    best_model, metrics = train_models(X, y, feature_cols)
    if best_model:
        print("Training completed successfully.")
    else:
        print("Training failed due to insufficient data after cleaning.")

if __name__ == "__main__":
    main()