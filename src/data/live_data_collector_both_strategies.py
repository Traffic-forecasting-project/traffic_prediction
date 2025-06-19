''' 
__author__ = "-"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Live traffic, weather, and incident data collection for Paris 8th district, repeated for 1000 rows with API limits and delay handling."

## This script:
## 1) Collects real-time traffic, incident and weather data
## 2) Supports two modes: 'traffic_analysis' and 'incident_analysis'
## 3) Targets a specific Paris district ('arrondissement')
## 4) Handles API limits and delays gracefully
## 5) Writes results in CSV files named according to the selected strategy and district
## 6) Supports single or multiple weather calls based on global settings
## 7) Allows for bounding box (bbox) splitting for more granular incident collection
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
from dotenv import load_dotenv
import argparse
import random

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
load_dotenv()

# TOMTOM_KEY = os.getenv("TOMTOM_KEY")
# WEATHER_KEY = os.getenv("WEATHER_KEY")

os.makedirs('live', exist_ok=True)
os.makedirs('logs', exist_ok=True)

CSV_PATH = "live/live_data.csv"  # Output path for collected data in CSV format

ARRONDISSEMENTS_PATH = "src/data/arrondissements.csv"  ## (Old path if needed)
#ARRONDISSEMENTS_PATH = "arrondissements.csv"  ## Path to CSV containing arrondissement boundaries (polygon coordinates)

NB_POINTS_TO_COLLECT = 5  ## Number of random points to sample per run (used in 'traffic_analysis' strategy)

MULTIPLE_WEATHER_CALLS = True  ## If True: call weather API once per point; if False: reuse same weather data for all
WEATHER_REFRESH_DELAY = 20     ## Delay (in seconds) between weather refreshes in shared mode (e.g., 600 = 10 min)

# STRATEGY = "traffic_analysis"  ## Collect data starting from random coordinates
STRATEGY = "incident_analysis"   ## Collect data starting from incident bounding boxes ("traffic_analysis" or "incident_analysis")

# BBOX_SPLIT_COUNT = 4  ## (Optional: how many sub-bounding boxes to split arrondissement into)
BBOX_SPLIT_COUNT = 1  ## Number of sub-bboxes to use in 'incident_analysis' strategy (1 = full arrondissement)

MAX_CALLS_PER_DAY = 1000  ## Limit for weather API calls per day
CALL_DELAY_SECONDS = 20   ## Delay between two API calls (can be adjusted based on API rules)

calls_today = 0  ## Counter for the number of API calls made today
last_weather_time = None  ## Timestamp of the last weather call (for shared mode)
last_weather_data = None  ## Cached weather response (for shared mode)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/live_data_collector.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

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

def get_weather(lat, lon):

    url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_KEY,
        'units': 'metric'
    }
    
    r = safe_request(url, params)
    data = r.json()
    
    return {
        "temp": data.get("main", {}).get("temp", None),
        "wind": data.get("wind", {}).get("speed", None),
        "rain": data.get("rain", {}).get("1h", 0)
    }

def get_traffic_flow(lat, lon):

    ## Proceed to fetch traffic flow data using the snapped coordinates
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
        
    fsd = data["flowSegmentData"]
    
    return {
        "avg_speed": fsd.get("currentSpeed", None),
        "free_flow_speed": fsd.get("freeFlowSpeed", None),
        "jam_factor": fsd.get("confidence", None)
    }

def get_incidents(lat1, lon1, lat2, lon2):

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    
    ## the fourt points of a bbox are passed as parammeters
    ## the totality of properties are extracted
    params = {
        "key": TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,tmc{countryCode,tableNumber,tableVersion,direction,points{location,offset}}}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }
    
    r = safe_request(url, params)
    data = r.json()
    
    incidents = data.get("incidents", [])
    
    return {
        "incident_count": len(incidents),
        "incident_magnitudes": [i["properties"].get("magnitudeOfDelay", None) for i in incidents],
        "incident_delays": [i["properties"].get("delay", None) for i in incidents],
        "incident_roads": [i["properties"].get("roadNumbers", []) for i in incidents]
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

def collect(points):

    global last_weather_time, last_weather_data
    
    collected_rows = []

    for lon, lat in points:
        try:
            ts = datetime.datetime.now().replace(microsecond=0)

            ## Get weather (either from each specific point and each time)
            if MULTIPLE_WEATHER_CALLS:
                weather = get_weather(lat, lon)
            ## Or single point with a delay between calls    
            else:
                if last_weather_time is None or (datetime.datetime.now() - last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                    last_weather_data = get_weather(lat, lon)
                    last_weather_time = datetime.datetime.now()
                weather = last_weather_data

            traffic = get_traffic_flow(lat, lon)
            incidents = get_incidents(lat - 0.01, lon - 0.01, lat + 0.01, lon + 0.01)

            row = {
                "timestamp": ts.isoformat(),
                "lat": lat,
                "lon": lon,
                **incidents,
                **traffic,
                **weather
            }
            collected_rows.append(row)
            logging.info(f"Data collected at {lat},{lon}")
        except Exception as e:
            logging.warning(f"Failed at {lat},{lon}: {e}")

    df = pd.DataFrame(collected_rows)
    
    ## Reorganise collumns in specific order for extracted data
    columns_order = [
        "timestamp", "lat", "lon", 
        "incident_count", "incident_magnitudes", "incident_delays", "incident_roads",
        "avg_speed", "free_flow_speed", "jam_factor",
        "temp", "wind", "rain"
    ]
    
    ## Keep only present collumns in the DF
    columns_order = [col for col in columns_order if col in df.columns]

    return df[columns_order]

def save_csv(df, arrondissement):

    ## save DF with strategy and arrondissement used
    new_csv_path = CSV_PATH.replace(".csv", f'_{STRATEGY}.{arrondissement}.csv')
    
    if os.path.exists(new_csv_path):
        df.to_csv(new_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(new_csv_path, index=False)

def split_bbox(points, n_splits):

    lats = [p[1] for p in points]
    lons = [p[0] for p in points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_step = (max_lat - min_lat) / n_splits
    lon_step = (max_lon - min_lon) / n_splits

    bboxes = []
    for i in range(n_splits):
    
        for j in range(n_splits):
        
            lat1 = min_lat + i * lat_step
            lat2 = lat1 + lat_step
            lon1 = min_lon + j * lon_step
            lon2 = lon1 + lon_step
            bboxes.append((lat1, lon1, lat2, lon2))
    
    return bboxes

def collect_from_bbox(lat1, lon1, lat2, lon2):

    global last_weather_time, last_weather_data
    collected_rows = []

    try:
        incidents_raw = get_incidents(lat1, lon1, lat2, lon2)
        incident_coords = []

        ## Re extract incidents (we reconstruct to have coordinates of incidents)
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            "key": TOMTOM_KEY,
            "bbox": f"{lon1},{lat1},{lon2},{lat2}",
            "language": "en-GB",
            "timeValidityFilter": "present"
        }
        
        r = safe_request(url, params)
        data = r.json()
        
        for i in data.get("incidents", []):
        
            coords = i.get("geometry", {}).get("coordinates", [])
            
            if coords:
                for coord in coords:
                    if isinstance(coord, list) and len(coord) >= 2:
                        lon_i, lat_i = coord[:2]
                        incident_coords.append((lat_i, lon_i))

        for lat, lon in incident_coords:
        
            ts = datetime.datetime.now().replace(microsecond=0)

            if MULTIPLE_WEATHER_CALLS:
                weather = get_weather(lat, lon)
            else:
                if last_weather_time is None or (datetime.datetime.now() - last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                    last_weather_data = get_weather(lat, lon)
                    last_weather_time = datetime.datetime.now()
                weather = last_weather_data

            traffic = get_traffic_flow(lat, lon)

            row = {
                "timestamp": ts.isoformat(),
                "lat": lat,
                "lon": lon,
                **incidents_raw,
                **traffic,
                **weather
            }
            
            collected_rows.append(row)
            logging.info(f"Data collected from incident at {lat},{lon}")

    except Exception as e:
        logging.warning(f"Failed bbox {lat1},{lon1},{lat2},{lon2} : {e}")

    df = pd.DataFrame(collected_rows)
    columns_order = [
        "timestamp", "lat", "lon", 
        "incident_count", "incident_magnitudes", "incident_delays", "incident_roads",
        "avg_speed", "free_flow_speed", "jam_factor",
        "temp", "wind", "rain"
    ]
    
    columns_order = [col for col in columns_order if col in df.columns]
    
    return df[columns_order]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrondissement", help="Paris arrondissement from 1 to 20", type=int, default=8)
    parser.add_argument("--strategy", help="Data collection strategy: 'traffic_analysis' or 'incident_analysis'", type=str, choices=["traffic_analysis", "incident_analysis"], default="traffic_analysis")
    args = parser.parse_args()

    arrondissement = args.arrondissement
    STRATEGY = args.strategy
     
    # point_list = extract_point_list_from_geometry(
        # pd.read_csv(ARRONDISSEMENTS_PATH).iloc[arrondissement - 1]["Geometry"]
    # )
    points = extract_point_list_from_geometry(
        pd.read_csv(ARRONDISSEMENTS_PATH).iloc[arrondissement - 1]["Geometry"]
    )
        
    logging.info(f"STRATEGY = {STRATEGY}")
    
    if STRATEGY == "traffic_analysis":
        logging.info(f"NB_POINTS_TO_COLLECT = {NB_POINTS_TO_COLLECT}")
        #logging.info(f"Total of points at arrondissement {arrondissement} : {len(points)}", flush=True)        
    elif STRATEGY == "incident_analysis":
        logging.info(f"BBOX_SPLIT_COUNT = {BBOX_SPLIT_COUNT}")
 
    logging.info(f"MULTIPLE_WEATHER_CALLS = {MULTIPLE_WEATHER_CALLS}")
    logging.info(f"WEATHER_REFRESH_DELAY = {WEATHER_REFRESH_DELAY}")
    #logging.info(f"Total of points at arrondissement {arrondissement} : {len(point_list)}", flush=True) 
    #print(f"Total of points at arrondissement {arrondissement} : {len(point_list)}", flush=True) 

    success_count = 0
    MAX_ROWS = 10000

    while success_count < MAX_ROWS and calls_today + 3 <= MAX_CALLS_PER_DAY:
        if STRATEGY == "traffic_analysis":
        
            #lon, lat = random.choice(point_list)
            #df = collect(lat, lon)
            #print(f"Extracting data from point {lat},{lon}", flush=True)         
            sampled_points = random.sample(points, min(NB_POINTS_TO_COLLECT, len(points)))
            df = collect(sampled_points)
            logging.info(f"Extracting data from sampled points: {sampled_points}")
            #print(f"Extracting data from sampled points: {sampled_points}", flush=True)

        elif STRATEGY == "incident_analysis":
            bboxes = split_bbox(points, BBOX_SPLIT_COUNT)
            bbox = random.choice(bboxes)
            df = collect_from_bbox(*bbox)
            logging.info(f"Extracting data from bbox: {bbox}")           
            print(f"Extracting data from bbox: {bbox}")

        else:
            logging.error("Unknown strategy.")
            break

        if df is not None and not df.empty:
            save_csv(df, arrondissement)
            success_count += len(df)
            #success_count += 1            
            logging.info(f"Progress: {success_count}/{MAX_ROWS} rows collected.")
        else:
            logging.warning("No data collected, retrying after delay.")

        time.sleep(CALL_DELAY_SECONDS)

    logging.info(f"Live data collection completed: {success_count} rows collected.")
    print(f"Finished collecting {success_count} live entries.")
