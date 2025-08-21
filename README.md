# 🚦 MLOps Project – Traffic Forecasting & Incident Analysis

## 🧠 Objective

Build a full **MLOps pipeline** to forecast **road traffic congestion** and analyze **traffic incidents** using **live data** from the TomTom API and weather conditions from OpenWeather.

---

## 🧩 Project Overview

This repository includes:

- 📥 **Live data collection** from APIs (TomTom + OpenWeather)
- 🧼 **Data cleaning and preparation**
- 🤖 **Model training** for multiple incident-related targets
- 📈 **Evaluation and metrics logging**
- 📦 **Model deployment** via **FastAPI**
- 📊 **Strategy comparison**
- 📊 **Drift monitoring** (optional)
- 🔁 **Versioning with DVC**

---

## 📁 Project Structure

```
.
├── main.py                  # Main script entrypoint
├── requirements.txt         # All required packages
├── pytest.ini               # Pytest config
├── README.md                # This file
├── .env                     # API keys (not tracked)
│
├── data/                    # All data (raw, processed, live)
│   ├── raw/
│   ├── processed/
│   └── live/
│
├── model/                   # Trained models (.joblib)
├── metrics/                 # Output metrics
├── logs/                    # Log files
├── eda/                     # Exploratory Data Analysis
│
├── docker/                  # (Optional) Docker structure
│
├── src/                     # Source code
│   ├── auth/                # JWT Auth
│   ├── core/                # Constants, logger, API service
│   ├── data/                # Data acquisition & preparation
│   ├── eda/                 # EDA generation
│   ├── model/               # Model training
│   ├── monitoring/          # Drift monitoring (optional)
│   └── utils/               # Utility functions
│
└── tests/                   # API and logic tests
```

---

## 🔑 Setup Instructions

```bash
# 1. Clone the repository
git clone <repo_url>
cd traffic_prediction

# 2. Create and activate virtual env
python -m venv venv
venv\Scripts\activate.bat  # (Windows)
# or
source venv/bin/activate   # (Linux/Mac)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys in a `.env` file
echo TOMTOM_API_KEY=your_tomtom_key >> .env
echo WEATHER_KEY=your_weather_key >> .env

# 5. Run full pipeline
python main.py
```

---

## Data Sources

- **Incidents and traffic**: [TomTom Traffic API](https://developer.tomtom.com/traffic-api/api-explorer)
- **Weather**: [OpenWeatherMap](https://openweathermap.org/api)

## Data Acquisition Strategy

We focus on **live data extraction** instead of historical data for the following reasons:

- The historical `Traffic Stats API` from TomTom is **paid and limited with partial data** during the free trial period.
- With live data, we can automate calls and build a consistent time-series dataset.
  - **Daily limits**: 2500 calls (TomTom), 1000 calls (Weather).

We use a `.csv` file with **bounding boxes (BBOX)** of Paris arrondissements, applied fully or split.


## 📊 Strategy Comparison

- Multiple strategies can be tested and compared by changing configuration values.
- Models support different targets like:
  - `jam_factor`
  - `congestion_label`
  - `incident_duration_min`
- Results are stored in `metrics/metrics.csv` and exported as visual plots.


The figure below illustrates the two possible strategies for data extraction:

![Traffic vs Incident Analysis](images/data_strategies.png)

| Strategy              | Description                                          | Pros                                              | Cons                                                  |
|-----------------------|------------------------------------------------------|---------------------------------------------------|--------------------------------------------------------|
| **Traffic Analysis**  | Record traffic at fixed points                      | Easy to set up, real-time, no need for incidents  | Many empty calls, low incident yield or non realistic call number to detect               |
| **Incident Analysis** | Get traffic where incidents occurred (via BBOX)     | Targeted, efficient, good incident coverage       | Misses pre-incident flow, no normal traffic context   |

---

## Data Synchronization and DVC/Dagshub file versioning

This repository uses **DVC** to track different files important for our project

### Tracked files 
- **data/raw** -> .csv files storing collected data for each Paris district.
- **data/processed** -> .csv containing features for training
- **model/** -> model artifacts

### Setup
The following steps allow to use DVC tracking with remote storage on Dagshub server:

```
pip install "dvc[s3]"
dvc init

# Configure DagsHub remote
dvc remote add -d origin s3://dvc/mateovillaarias/traffic_prediction
dvc remote modify origin endpointurl https://dagshub.com

# Local-only credentials (never commit these)
export DAGSHUB_USER="your_username"
export DAGSHUB_TOKEN="your_personal_access_token"
dvc remote modify origin --local access_key_id $DAGSHUB_USER
dvc remote modify origin --local secret_access_key $DAGSHUB_TOKEN

```
**MANDATORY**: Once the remote access is set, download the latest data:

```
dvc pull
```
### File synchronisation and push to remote

Collected live data is stores at data/live in form of .csv files per paris district.
To synchornize files in data/live to the main dataset located in data/raw, you must execute option 2 from the main.py script

```
python main.py 
=== TRAFFIC LIVE DATA MENU ===
1. Run live data collection
2. Sync live files into raw directory
3. Run feature engineering on collected data
4. Run EDA analysis on data
5. Run train model on data
6. Launch FastAPI service with uvicorn
7. Run tests avec pytest
8. Push raw data to DVC remote
9. Push pipeline results to DVC remote
q. Quit
==============================
Select an option: 2
Delete live files after sync? (y/n): n
```

After syncrhonization you can push the raw data to the dvc server using the option 8 from main.py



## 🛠 Tech Stack

- **Python 3.11**
- **FastAPI** for the prediction service
- **TomTom** and **OpenWeatherMap** APIs
- **Pandas**, **Scikit-learn**, **Joblib**
- **DVC** for data and model versioning
- **Dagshub** for data storage and DVC integration with Github repo
- *(Planned)*: uv, authorisation, monitoring, CI tools


---

## 📞 Contact

> Email: georges.nassopoulos@gmail.com, ingmatvillaa@gmail.com, elqounss.karim@gmail.com
