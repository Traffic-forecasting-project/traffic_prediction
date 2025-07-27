'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Global constants for live data collection and model pipeline
'''

import os
from dotenv import load_dotenv

## Load environment variables from .env file
load_dotenv()

## ========================
## BASE DIRECTORIES
## ========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
LIVE_DATA_DIR = os.path.join(DATA_DIR, "live")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

## ========================
## FILE PATHS
## ========================
ARRONDISSEMENTS_PATH = os.path.join(RAW_DATA_DIR, "arrondissements.csv")
CSV_PATH = os.path.join(LIVE_DATA_DIR, "live_data.csv")

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
CALL_DELAY_SECONDS = int(os.getenv("CALL_DELAY_SECONDS", 10))  # default delay 10 seconds
MAX_CALLS_PER_DAY = int(os.getenv("MAX_CALLS_PER_DAY", 2500))  # default max calls
MULTIPLE_WEATHER_CALLS = os.getenv("MULTIPLE_WEATHER_CALLS", "false").lower() == "true"
WEATHER_REFRESH_DELAY = int(os.getenv("WEATHER_REFRESH_DELAY", 360))  # default: 30 min

## ========================
## STRATEGY PARAMETERS
## ========================
STRATEGY = os.getenv("NB_POINTS_TO_COLLECT", "incident_analysis")  # for traffic_analysis
NB_POINTS_TO_COLLECT = int(os.getenv("NB_POINTS_TO_COLLECT", 20))  # for traffic_analysis
BBOX_SPLIT_COUNT = int(os.getenv("BBOX_SPLIT_COUNT", 1))           # for incident_analysis
DELTA_BBOX = int(os.getenv("DELTA_BBOX", 0.01))                    # for traffic_analysis
SAMPLE_ALL_INCIDENT_POINTS = os.getenv("SAMPLE_ALL_INCIDENT_POINTS", "false").lower() == "true"

def load_api_keys(arrondissement: int) -> tuple:
    """
    Load TOMTOM and WEATHER API keys from environment variables.

    Args:
        arrondissement (int): Arrondissement number.

    Returns:
        tuple: (TOMTOM_KEY, WEATHER_KEY)
    """
    
    tomtom_key = os.getenv(f"TOMTOM_KEY_{arrondissement}")
    weather_key = os.getenv(f"WEATHER_KEY_{arrondissement}")
    
    return tomtom_key, weather_key
