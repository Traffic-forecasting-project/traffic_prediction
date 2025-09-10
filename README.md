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
- 🔑 **Authentication & Authorization** with **JWT tokens** (role-based access: admin, user, guest)
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
├── src/                     # Source code for the entire pipeline and API
│   ├── auth/                # JWT-based authentication & role-based authorization
│   ├── core/                # Core logic: API service, constants, centralized logging
│   ├── data/                # Data acquisition, synchronization and preprocessing workflows
│   ├── eda/                 # Exploratory Data Analysis (EDA) scripts and visualizations
│   ├── model/               # Model training, evaluation and serialization
│   ├── monitoring/          # Drift detection and monitoring modules (optional)
│   └── utils/               # General-purpose helper functions shared across modules
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

## 🔑 Authentication & Authorization

The API is secured with **JWT tokens**:
- **Admin**: full access (metrics, prediction, monitoring)
- **User**: limited access (prediction, test-token)
- **Guest**: minimal access (healthcheck only)

- Tokens are issued via the `/login` endpoint.
- Protected endpoints validate tokens with role-based dependencies.
- Invalid or expired tokens return `401 Unauthorized`.
- Access to forbidden routes returns `403 Forbidden`.

Example:
```bash
# Obtain a token
curl -X POST http://localhost:8000/login \
  -d "username=admin&password=adminpass" \
  -H "Content-Type: application/x-www-form-urlencoded"

# Use token to call a protected endpoint
curl -X GET http://localhost:8000/metrics \
  -H "Authorization: Bearer <token>"
```
---

## Data Synchronization and DVC/Dagshub File Versioning

This repository uses **DVC** to track data and models important for the project.

### Tracked files
- **data/raw/** → raw `.csv` files storing collected data for each Paris district
- **data/processed/** → processed features for training
- **model/** → model artifacts

---

### Setup

Install and initialize DVC with S3 support:
```
pip install "dvc[s3]"
dvc init
```
Configure the DagsHub remote:
```
dvc remote add -d origin s3://dvc/mateovillaarias/traffic_prediction
dvc remote modify origin endpointurl https://dagshub.com
```
Local-only credentials (never commit these)
```
export DAGSHUB_USER="your_username"
export DAGSHUB_TOKEN="your_personal_access_token"
dvc remote modify origin --local access_key_id $DAGSHUB_USER
dvc remote modify origin --local secret_access_key $DAGSHUB_TOKEN
```
**Mandatory:** once the remote is set, pull the latest data:
```
dvc pull
```
---

### Usage

Collected live data is stored in `data/live/` (per district `.csv` files).  

- To synchronize `data/live` into the main dataset in `data/raw`, run option 2 in `main.py`:
```
python main.py
```
```
=== TRAFFIC LIVE DATA MENU ===
...
2. Sync live files into raw directory
...
Select an option: 2
Delete live files after sync? (y/n): n
```

- After synchronization, you can push data and models to DVC using option 8:
```
python main.py

=== TRAFFIC LIVE DATA MENU ===
...
8. Push data to DVC remote
...
Select an option: 8
DVC remote name, leave blank for default:
Git commit message, leave blank for default:
Select what to push [raw/processed/train/all] (default: raw): raw
```
- **This will:**
  - Commit changes (dvc commit, git add, git commit)
  - Push to the configured DagsHub remote


- **Option 8** works with:
  - raw → pushes data/raw
  - processed → pushes data/processed
  - train → pushes model/
  - all → pushes all of the above



## 🛠 Tech Stack

- **Python 3.11**
- **FastAPI** for the prediction service
- **TomTom** and **OpenWeatherMap** APIs
- **Pandas**, **Scikit-learn**, **Joblib**
- **DVC** for data and model versioning
- **Dagshub** for data storage and DVC integration with Github repo
- *(Planned)*: uv, airflow, monitoring


---

## 📞 Contact

> Email: georges.nassopoulos@gmail.com, ingmatvillaa@gmail.com, elqounss.karim@gmail.com
