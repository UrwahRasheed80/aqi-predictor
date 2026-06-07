#  Pearls AQI Predictor – End‑to‑End Air Quality Forecasting System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red.svg)](https://streamlit.io)
[![Scikit‑learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Executive Summary

This project implements a **fully serverless machine learning pipeline** that predicts the Air Quality Index (AQI) for the next **three days** using real‑time data from the AQICN API. It includes:

- Automated **feature pipeline** (hourly data collection & feature engineering)
- **Historical backfill** (30 days of realistic synthetic AQI data)
- **Multi‑model training** (Random Forest vs. Ridge Regression) with MAE/RMSE/R² evaluation
- **Model registry** (local storage for the best performing model)
- **Interactive Streamlit dashboard** featuring:
  - Exploratory Data Analysis (time trends, correlations, hourly patterns)
  - 3‑day hourly forecast with health alerts
  - SHAP model explanations (or coefficient bar chart for linear models)

The entire system runs in **GitHub Codespaces** (or any Linux environment) and is designed for full automation via **GitHub Actions** (cron jobs). The dashboard is publicly accessible via a forwarded port URL, demonstrating a production‑ready, zero‑server‑management solution.

---

##  Live Dashboard

After following the setup instructions, the Streamlit dashboard becomes available at a public URL similar to:
## https://verbose-space-carnival-r4r6476vw79v2xpjv-8501.app.github.dev/


> The dashboard includes real‑time AQI, 72‑hour forecasts, health alerts, and SHAP explanations.

---

##  Key Features

| Feature | Implementation |
|---------|----------------|
| **Real‑time data ingestion** | AQICN API – current AQI, PM₂.₅, PM₁₀, NO₂, O₃, temperature, humidity, wind speed |
| **Feature engineering** | Hour, day_of_week, lag features (1h, 6h), 6‑hour rolling mean of AQI |
| **Historical backfill** | 30 days of synthetic AQI data with diurnal/weekend patterns + random noise |
| **Model training** | Random Forest Regressor & Ridge Regression (scikit‑learn) |
| **Model evaluation** | Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² score |
| **Model registry** | Best model saved as `models/best_aqi_model.pkl` |
| **Forecast (72h)** | Uses synthetic weather forecast (temperature, humidity, wind) + last known AQI |
| **Dashboard** | Streamlit – EDA plots, forecast line chart, health alerts, SHAP explanation button |
| **Explainability** | SHAP summary plot (Random Forest) or coefficient bar chart (Ridge) |
| **Serverless runtime** | GitHub Codespaces (cloud‑based, no local setup needed) |
| **Automation ready** | GitHub Actions workflows (cron: hourly feature pipeline, daily training) |

---

##  Technology Stack

| Category | Tools & Libraries |
|----------|-------------------|
| Language | Python 3.12 |
| Data handling | Pandas, NumPy |
| Machine learning | Scikit‑learn (Random Forest, Ridge) |
| Visualisation | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Model explanation | SHAP |
| API | AQICN (air quality data) |
| Environment | GitHub Codespaces (Linux) |
| Automation | GitHub Actions (planned) |

---

##  Project Structure

aqi-predictor/
├── .github/workflows/ # CI/CD automation (cron jobs) – ready to use
├── src/
│ ├── feature_pipeline.py # Fetches live AQI, computes features, saves to CSV
│ ├── backfill_historical.py # Generates 30 days of synthetic AQI data
│ ├── training_pipeline.py # Trains Random Forest & Ridge, evaluates, saves best model
│ ├── dashboard.py # Streamlit app (EDA, forecast, SHAP, alerts)
│ └── hopsworks_utils.py # Local feature store / model registry (CSV + pickle)
├── data/ # Stores aqi_features.csv, aqi_history.csv, forecast.csv
├── models/ # Stores best_aqi_model.pkl
├── requirements.txt # Python dependencies
├── .env # API keys (not committed – add your own)
└── README.md # This file


---

## 
How to Run the Project (Step‑by‑Step)

### 1. Prerequisites

- A **GitHub account** (to use Codespaces).
- An **AQICN API key** – free from [aqicn.org/data‑platform/token/](https://aqicn.org/data-platform/token/).

### 2. Open the Repository in GitHub Codespaces

1. Go to your repository on GitHub (e.g., `https://github.com/UrwahRasheed80/aqi-predictor`).
2. Click the **Code** button → **Codespaces** → **Create codespace on main**.
3. Wait a minute – a VS Code in the browser will open with the project loaded.

### 3. Set Up Environment Variables

In the Codespace terminal, create a `.env` file:

```bash
cat > .env <<EOF
AQICN_API_KEY=your_api_key_here
CITY=Islamabad
EOF

Acknowledgements
AQICN for providing free air quality data.

Streamlit for the amazing dashboard framework.

Scikit‑learn and SHAP communities.
