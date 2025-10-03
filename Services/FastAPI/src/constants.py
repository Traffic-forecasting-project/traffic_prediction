'''
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Global constants for live data collection and model pipeline
'''

import os
from dotenv import load_dotenv
from typing import List

## Load environment variables from .env file
load_dotenv()

## ========================
## SECURITY CONSTANTS
## ========================
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

## ========================
## BASE DIRECTORIES
## ========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "model")
EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "eda")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
LIVE_DATA_DIR = os.path.join(DATA_DIR, "live")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

## ========================
## AUTHORIZATION DUMMY USERS
## ========================
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "adminpass",
        "role": "admin"
    },
    "user": {
        "username": "user",
        "password": "userpass",
        "role": "user"
    },
    "guest": {
        "username": "guest",
        "password": "guestpass",
        "role": "guest"
    }
}

## ========================
## FILE PATHS
## ========================
ARRONDISSEMENTS_PATH = os.path.join(DATA_DIR, "arrondissements.csv")
CSV_PATH = os.path.join(LIVE_DATA_DIR, "live_data.csv")
TOP_FEATURES_FILE = os.path.join(MODELS_DIR, "feature_importances.json" )
MODEL_PATH = os.path.join(MODELS_DIR, "model_incident_analysis_incident_duration_min.joblib" )

## ========================
## API KEYS
## ========================
# These depend on arrondissement, dynamically built in main code
# e.g. os.getenv(f"TOMTOM_KEY_{arrondissement}")
# Example fallback default (if needed):
DEFAULT_TOMTOM_KEY = os.getenv("TOMTOM_KEY")
DEFAULT_WEATHER_KEY = os.getenv("WEATHER_KEY")
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
WEATHER_KEY = os.getenv("WEATHER_KEY")

## ========================
## API LIMITATIONS & OPTIONS
## ========================
CALL_DELAY_SECONDS = int(os.getenv("CALL_DELAY_SECONDS", 10))  ## default delay 10 seconds
MAX_CALLS_PER_DAY = int(os.getenv("MAX_CALLS_PER_DAY", 2500))  ## default max calls
MULTIPLE_WEATHER_CALLS = os.getenv("MULTIPLE_WEATHER_CALLS", "false").lower() == "true"
WEATHER_REFRESH_DELAY = int(os.getenv("WEATHER_REFRESH_DELAY", 360))  ## default: 30 min

## ========================
## STRATEGY PARAMETERS
## ========================
STRATEGY = os.getenv("STRATEGY", "incident_analysis")  ## for traffic_analysis
NB_POINTS_TO_COLLECT = int(os.getenv("NB_POINTS_TO_COLLECT", 20))  ## for traffic_analysis
BBOX_SPLIT_COUNT = int(os.getenv("BBOX_SPLIT_COUNT", 1))           ## for incident_analysis
DELTA_BBOX = int(os.getenv("DELTA_BBOX", 0.01))                    ## for traffic_analysis
SAMPLE_ALL_INCIDENT_POINTS = os.getenv("SAMPLE_ALL_INCIDENT_POINTS", "false").lower() == "true"

## ========================
## DATA ENGINEERING & TRAIN PARAMETERS
## ========================
FILTER_STANDARD_INCIDENTS = os.getenv("FILTER_STANDARD_INCIDENTS", "false").lower() == "true"   ## Filter out extreme outliers for regression targets
USE_TOP_FEATURES_ONLY = os.getenv("USE_TOP_FEATURES_ONLY", "false").lower() == "true"       ## Restrict to top features if previous importances exist
ENABLE_EXTRA_FEATURES =  os.getenv("ENABLE_EXTRA_FEATURES", "false").lower() == "true"       ## Enable creation of extra feature to make more descriptive model
EDA_ENABLED = os.getenv("EDA_ENABLED", "false").lower() == "true"  
TOP_N_FEATURES = int(os.getenv("TOP_N_FEATURES", 15))  
TARGETS: List[str] = [ ## List of target columns
    "jam_factor",
    "incident_duration_min",
    "mean_magnitude"
]
TARGET_METADATA = { ## Mapping of all supported targets with allowed strategies
    "jam_factor": {"strategies": ["traffic_analysis", "incident_analysis"]},
    "incident_duration_min": {"strategies": ["incident_analysis"]},
    "mean_magnitude": {"strategies": ["incident_analysis"]},
}

TIMESTAMP_COLUMN = "timestamp"


## ========================
## EXPERIMENT LOGGING
## ========================

MLFLOW_ENABLE_REMOTE = os.getenv("MLFLOW_REMOTE", "true").lower() == "true"
MLFLOW_REMOTE_URL = "https://dagshub.com/mateovillaarias/traffic_prediction.mlflow"
MLFLOW_LOCAL_URI = "http://127.0.0.1:8050/"
MLFLOW_DEFAULT_EXPERIMENT_NAME = "Default"
DAGSHUB_REPO_OWNER = "mateovillaarias"
DAGSHUB_REPO_NAME = "traffic_prediction"