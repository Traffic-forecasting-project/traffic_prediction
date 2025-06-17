''' 
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Live traffic, weather, and incident data collection for Paris 8th district, repeated for 1000 rows with API limits and delay handling."

## This code : 
## 1) Combines traffic, weather & incidents
## 2) Extracts same data types (real-time snapshot)
## 3) Targets same district (Paris 8th)
## 4) Uses periodicity (on demand)
## 5) Stops on limit imposed by weather API
'''

import requests
import datetime
import pandas as pd
import os
import time
import logging
import sys
import io
import json
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
from dotenv import load_dotenv
import argparse
import random

##  Load API keys from .env file
load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
WEATHER_KEY = os.getenv("WEATHER_KEY")

## Define directories and ensure they exist
os.makedirs('live', exist_ok=True)
os.makedirs('logs', exist_ok=True)

CSV_PATH = "live/live_data.csv"

## Call limit parameters (half for historcal half for live)
## API call tracking
MAX_CALLS_PER_DAY = 1000 # ==> limit for weather
CALL_DELAY_SECONDS = 20
calls_today = 0

## Setup logging to file + console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

## File handler
file_handler = logging.FileHandler("logs/live_data_collector.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

## Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

def safe_request(url, params):

    global calls_today
    
    if calls_today >= MAX_CALLS_PER_DAY:
        logging.warning("Max daily API calls reached.")
        raise Exception("API limit reached")
        
    time.sleep(CALL_DELAY_SECONDS)
    r = requests.get(url, params=params)
    
    calls_today += 1
    
    return r

def get_weather(lat,lon):

    url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_KEY,
        'units': 'metric'
    }
    
    r = safe_request(url, params)
    data = r.json()
    m = data["main"]
    
    return {
        "temp": m["temp"],
        "wind": data["wind"]["speed"],
        "rain": data.get("rain", {}).get("1h", 0)
    }

def get_traffic_flow(lat, lon):   
    # # Proceed to fetch traffic flow data using the snapped coordinates
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    
    params = {
        'point': f"{lat},{lon}",
        'key': TOMTOM_KEY
    }
    
    r = safe_request(url, params)
    data = r.json()
    
    if "flowSegmentData" not in data:
        logging.warning(f"Empty or unexpected response from TomTom Traffic API:\n{data}")
        raise Exception("Missing flowSegmentData in TomTom response")
    
    return {
        "avg_speed": data["flowSegmentData"]["currentSpeed"],
        "free_flow_speed": data["flowSegmentData"]["freeFlowSpeed"],
        "jam_factor": data["flowSegmentData"]["confidence"]
    }


def get_incidents(lat, lon):

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    
    params = {
        'bbox': f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}",
        'key': TOMTOM_KEY,
        'fields': 'id',
        'language': 'en'
    }
    
    r = safe_request(url, params)
    data = r.json()
    
    return {
        "incident_count": len(data.get("incidents", []))
    }

def get_last_timestamp():

    if os.path.exists(CSV_PATH):
        
        df = pd.read_csv(CSV_PATH)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df["timestamp"].max()
            
    return None

def extract_point_list_from_geometry(geometry_str):
    geom = json.loads(geometry_str)
    coords = geom["coordinates"][0]
    return coords

def collect(lat, lon):
    try:
        ts = datetime.datetime.now().replace(microsecond=0)
        traffic = get_traffic_flow(lat, lon)
        weather = get_weather(lat, lon)
        incidents = get_incidents(lat, lon)
        row = {
            "timestamp": ts.isoformat(),
            "lat": lat,         # added latitude
            "lon": lon,         # added longitude
            **incidents,
            **traffic,
            **weather
        }
        return pd.DataFrame([row])
    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        return None


def save_csv(df,arrondissement):
    new_csv_path = CSV_PATH.replace(".csv",f'_{arrondissement}.csv')
    if os.path.exists(new_csv_path):
        df.to_csv(new_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--arrondissement", help="Paris arrondissement from 1 to 20", type=int, default=8)

    args = parser.parse_args()

    arrondissement = args.arrondissement

    point_list = extract_point_list_from_geometry (
        pd.read_csv("src/data/arrondissements.csv").iloc[arrondissement-1]["Geometry"])
    print("Total of points at arrondissement {} : {}".format(arrondissement,len(point_list)),flush=True)
    
    success_count = 0
    MAX_ROWS = 1000 ## successfull calls per day

    while success_count < MAX_ROWS and calls_today + 3 <= MAX_CALLS_PER_DAY:
        lon,lat = random.choice(point_list) ## Will always be random inside the point list
        df = collect(lat,lon)
        print("Extracting data from point {},{}".format(lat,lon),flush=True)
        if df is not None:
            save_csv(df,arrondissement)
            success_count += 1
            logging.info(f"Progress: {success_count}/{MAX_ROWS} rows collected.")
        else:
            logging.warning("No data collected, retrying after delay.")

        time.sleep(CALL_DELAY_SECONDS)

    logging.info(f"Live data collection completed: {success_count} rows collected.")
    print(f"Finished collecting {success_count} live entries.")