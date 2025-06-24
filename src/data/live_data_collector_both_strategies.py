''' 
__author__ = "-"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com", "ingmatvillaa@gmail.com", "elqounss.karim@gmail.com"
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

## Ensure UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
load_dotenv()

## Load API keys (can be hardcoded or pulled from .env)
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
WEATHER_KEY = os.getenv("WEATHER_KEY")

## Create folders for data and logs
os.makedirs('live', exist_ok=True)
os.makedirs('logs', exist_ok=True)

CSV_PATH = "live/live_data.csv"  ## Output path for collected data

## Path to the arrondissement coordinates (polygon)
ARRONDISSEMENTS_PATH = "src/data/arrondissements.csv"  ## (Old path if needed)
#ARRONDISSEMENTS_PATH = "arrondissements.csv"

NB_POINTS_TO_COLLECT = 5  ## Number of points to collect per arrondissement if strategy is 'traffic_analysis'
SAMPLE_ALL_INCIDENTS = False ## Sample just one point per incident, or all points, useful if strategy is 'incident_analysis'

MULTIPLE_WEATHER_CALLS = False  ## If True, call weather API for every point; otherwise reuse the latest result
WEATHER_REFRESH_DELAY = 360  ## Refresh delay in seconds if weather is shared

# STRATEGY = "traffic_analysis"  ## Strategy that samples points randomly inside the polygon
STRATEGY = "incident_analysis"  ## Strategy based on incidents inside a bounding box


BBOX_SPLIT_COUNT = 1  ## How many parts to split the bounding box into in 'incident_analysis' mode
DELTA_BBOX = 0.01 ## range value for bbox over a random point of the original 'arrnondissement' box
MAX_CALLS_PER_DAY = 2500  ## Daily limit for weather API
CALL_DELAY_SECONDS = 10   ## Delay between any API calls to avoid rate limits

calls_today = 0  ## Counter for total API calls (weather)
last_weather_time = None  ## Last time weather was fetched
last_weather_data = None  ## Cached result if shared

## Configure logger to file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/live_data_collector.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

def safe_request(url, params):
    """
        Makes a safe API request with delay and call counter

        Args:
            url (str): The API endpoint
            params (dict): The query parameters

        Returns:
            Response: The HTTP response from requests.get()
    """
    
    global calls_today

    if calls_today >= MAX_CALLS_PER_DAY:
        logging.warning("Max daily API calls reached.")
        raise Exception("API limit reached")

    ## Respect the delay between calls to avoid bans
    time.sleep(CALL_DELAY_SECONDS)

    r = requests.get(url, params=params)
    calls_today += 1

    return r

def load_arrondissement_data(path):
    """
        [DEPRECATED] Load the polygon definitions for each Paris arrondissement from a CSV file

        Args:
            path (str): Path to the CSV file containing the 'arrondissement' and 'polygon' columns

        Returns:
            dict: A dictionary where keys are arrondissement numbers and values are lists of [lon, lat] points
    """
    
    df = pd.read_csv(path)
    arrondissement_polygons = {}

    for _, row in df.iterrows():
        arr = int(row["arrondissement"])  ## Convert arrondissement number to int
        points = json.loads(row["polygon"])  ## Parse the polygon points from JSON string
        arrondissement_polygons[arr] = points  ## Store in dictionary

    return arrondissement_polygons

def load_arrondissement_polygons(path):
    """
        Loads the polygon coordinates of each Paris arrondissement from a CSV file

        Args:
            path (str): Path to the CSV file containing arrondissement boundaries

        Returns:
            dict: A dictionary mapping arrondissement number to its polygon coordinates
    """
    
    df = pd.read_csv(path)

    ## Create a dictionary mapping each arrondissement number to a list of coordinates
    arr_dict = {}
    for _, row in df.iterrows():
        num = row["num"]
        coords_str = row["coordinates"]
        try:
            coords = json.loads(coords_str)
        except Exception:
            coords = eval(coords_str)  ## fallback in case JSON fails

        arr_dict[num] = coords

    return arr_dict

def extract_point_list_from_geometry(geometry_str: str) -> list:
    """
        Convert a GeoJSON-like geometry string into a list of [lon, lat] coordinates
        
        Args:
            geometry_str (str): A string representation of the geometry (e.g., GeoJSON Polygon)
        
        Returns:
            list: A list of [lon, lat] coordinate pairs
    """
    
    try:
        geometry = json.loads(geometry_str)
        return geometry.get("coordinates", [])[0]  # Assuming a single polygon
    except Exception as e:
        logging.warning(f"Could not parse geometry: {e}")
        return []

def point_inside_polygon(x, y, polygon):
    """
        Check whether a point is inside a polygon using the ray casting algorithm

        Args:
            x (float): Longitude of the point
            y (float): Latitude of the point
            polygon (list): List of [lon, lat] pairs defining the polygon

        Returns:
            bool: True if the point is inside the polygon, otherwise False
    """
    
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]  ## Initialize with the first point

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]  ## Loop back to first point after last

        ## Check if y is within the vertical range of edge (p1, p2)
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
            
                ## Check if x is to the left of or on the edge
                if x <= max(p1x, p2x):
                
                    ## Avoid division by zero if line is horizontal
                    if p1y != p2y:
                        xinters = ((y - p1y) * (p2x - p1x)) / (p2y - p1y + 1e-9) + p1x
                    else:
                        xinters = p1x
                        
                    ## Toggle the inside status if x is less than intersection
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y  ## Move to next edge

    return inside

def get_coordinates(polygon, nb_points):
    """
        Generate random coordinates that lie within a given polygon

        Args:
            polygon (list): List of [lon, lat] pairs forming the polygon boundary
            nb_points (int): Number of valid random points to generate inside the polygon

        Returns:
            list: List of (lon, lat) tuples contained in the polygon
    """
    
    ## Extract the bounding box from the polygon
    longitudes = [p[0] for p in polygon]
    latitudes = [p[1] for p in polygon]

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    coordinates = []

    ## Loop until enough points are found inside the polygon
    while len(coordinates) < nb_points:
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)

        if point_inside_polygon(lon, lat, polygon):
            coordinates.append((lon, lat))  ## Add valid point

    return coordinates

def get_weather(lat, lon):
    """
        Retrieves current weather data from OpenWeatherMap API for a given location

        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location

        Returns:
            dict: Dictionary containing temperature, wind speed, and rain volume (if any)
    """
    
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_KEY,
        "units": "metric"
    }

    ## Send request and parse JSON
    r = safe_request(url, params)
    data = r.json()

    ## Extract basic weather elements
    temp = data.get("main", {}).get("temp")
    wind = data.get("wind", {}).get("speed")
    rain = data.get("rain", {}).get("1h") if "rain" in data else 0.0

    ## Return extracted metrics
    return {
        "temp": temp,
        "wind": wind,
        "rain": rain
    }

def get_traffic_flow(lat, lon):
    """
        Retrieves real-time traffic flow data from the TomTom Traffic API for a specific point

        Args:
            lat (float): Latitude of the point
            lon (float): Longitude of the point

        Returns:
            dict: Dictionary containing average speed, free-flow speed, and jam factor
    """
    
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    params = {
        "point": f"{lat},{lon}",
        "key": TOMTOM_KEY
    }

    ## Make safe request to TomTom Traffic API
    r = safe_request(url, params)
    data = r.json()

    ## Navigate through the response structure
    flow_data = data.get("flowSegmentData", {})

    ## Extract desired traffic metrics
    avg_speed = flow_data.get("currentSpeed")
    free_flow_speed = flow_data.get("freeFlowSpeed")
    jam_factor = flow_data.get("currentTravelTime") / flow_data.get("freeFlowTravelTime") if (
        flow_data.get("freeFlowTravelTime") and flow_data.get("currentTravelTime")
    ) else None

    ## Return traffic info
    return {
        "avg_speed": avg_speed,
        "free_flow_speed": free_flow_speed,
        "jam_factor": jam_factor
    }

def get_incidents(lat1, lon1, lat2, lon2):
    """
        Fetches incident data from TomTom API within a given bounding box

        Args:
            lat1 (float): Southern latitude of the bounding box
            lon1 (float): Western longitude of the bounding box
            lat2 (float): Northern latitude of the bounding box
            lon2 (float): Eastern longitude of the bounding box

        Returns:
            List[dict]: List of incident dictionaries with extracted and cleaned attributes
    """
    
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,tmc{countryCode,tableNumber,tableVersion,direction,points{location,offset}}}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    ## Perform the API request safely (with rate limiting and tracking)
    r = safe_request(url, params)
    data = r.json()

    ## Extract incidents from response
    incidents = data.get("incidents", [])
    logging.info(f"{len(incidents)} incidents (uniques) extracted.")

    processed_incidents = []

    for inc in incidents:
        props = inc.get("properties", {})
        coords = inc.get("geometry", {}).get("coordinates", [])

        ## Fallback if no coordinates are found
        if not coords or not isinstance(coords, list):
            continue
        if SAMPLE_ALL_INCIDENTS == False:
            coords = coords[0:1] ## Just get the first point in the road (to modify, ideally the point in the center)
        ## Loop over coordinates involved in the incident
        for coord in coords[0:1]:
            if not isinstance(coord, list) or len(coord) != 2:
                continue

            lon_val, lat_val = coord[0], coord[1]

            ## Compose a clean and enriched incident dictionary
            incident_data = {
                "incident_count": 1,
                "incident_magnitudes": [props.get("magnitudeOfDelay")],
                "incident_delays": [props.get("delay")],
                "incident_roads": [props.get("roadNumbers", [])],
                "incident_id": props.get("id"),
                "icon_category": props.get("iconCategory"),
                "start_time": props.get("startTime"),
                "end_time": props.get("endTime"),
                "from_location": props.get("from"),
                "to_location": props.get("to"),
                "length": props.get("length"),
                "time_validity": props.get("timeValidity"),
                "probability": props.get("probabilityOfOccurrence"),
                "num_reports": props.get("numberOfReports"),
                "last_report": props.get("lastReportTime"),
                "tmc_countryCode": props.get("tmc", {}).get("countryCode") if props.get("tmc") else None,
                "tmc_tableNumber": props.get("tmc", {}).get("tableNumber") if props.get("tmc") else None,
                "tmc_tableVersion": props.get("tmc", {}).get("tableVersion") if props.get("tmc") else None,
                "tmc_direction": props.get("tmc", {}).get("direction") if props.get("tmc") else None,
                "event_descriptions": [e.get("description") for e in props.get("events", []) if "description" in e],
                "incident_coords": coords,  ## All coordinates related to the incident
                "lat": lat_val,
                "lon": lon_val
            }

            processed_incidents.append(incident_data)

    return processed_incidents

def get_incidents_per_coordinate(lat1, lon1, lat2, lon2):
    """
        [DEPRECATED] Retrieve detailed incidents from the TomTom API for each coordinate
        within the specified bounding box, and return individual records per coordinate

        This function provides very granular incident data, mapping each coordinate inside
        an incident geometry as a separate row. It is no longer used because the newer
        'get_incidents' function provides aggregated incident objects with coordinates grouped

        Args:
            lat1 (float): Minimum latitude of the bounding box
            lon1 (float): Minimum longitude of the bounding box
            lat2 (float): Maximum latitude of the bounding box
            lon2 (float): Maximum longitude of the bounding box

        Returns:
            list of dict: List of incidents, each associated with one specific coordinate
    """

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,tmc{countryCode,tableNumber,tableVersion,direction,points{location,offset}}}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    r = safe_request(url, params)
    data = r.json()

    if not isinstance(data, dict):
        logging.warning("Invalid response from TomTom API.")
        return []

    incidents = data.get("incidents", [])

    result_rows = []

    for inc in incidents:
        geometry = inc.get("geometry", {})
        props = inc.get("properties", {})
        coords = geometry.get("coordinates", [])

        ## Each coordinate in the incident geometry becomes a separate row
        for coord in coords:
            if isinstance(coord, list) and len(coord) >= 2:
                lat, lon = coord[1], coord[0]

                row = {
                    "lat": lat,
                    "lon": lon,
                    "incident_count": 1,
                    "incident_magnitudes": [props.get("magnitudeOfDelay")],
                    "incident_delays": [props.get("delay")],
                    "incident_roads": [props.get("roadNumbers", [])],
                    "incident_id": props.get("id"),
                    "icon_category": props.get("iconCategory"),
                    "start_time": props.get("startTime"),
                    "end_time": props.get("endTime"),
                    "from_location": props.get("from"),
                    "to_location": props.get("to"),
                    "length": props.get("length"),
                    "time_validity": props.get("timeValidity"),
                    "probability": props.get("probabilityOfOccurrence"),
                    "num_reports": props.get("numberOfReports"),
                    "last_report": props.get("lastReportTime"),
                    "tmc_countryCode": props.get("tmc", {}).get("countryCode") if props.get("tmc") else None,
                    "tmc_tableNumber": props.get("tmc", {}).get("tableNumber") if props.get("tmc") else None,
                    "tmc_tableVersion": props.get("tmc", {}).get("tableVersion") if props.get("tmc") else None,
                    "tmc_direction": props.get("tmc", {}).get("direction") if props.get("tmc") else None,
                    "event_descriptions": [
                        e.get("description") for e in props.get("events", [])
                        if "description" in e
                    ],
                }

                result_rows.append(row)

    logging.info(f"Number of incident coordinates found: {len(result_rows)}")
    
    return result_rows

def split_bbox(lat1, lon1, lat2, lon2, count):
    """
        Split a bounding box into smaller sub-bounding boxes

        Args:
            lat1 (float): South latitude of original bbox
            lon1 (float): West longitude of original bbox
            lat2 (float): North latitude of original bbox
            lon2 (float): East longitude of original bbox
            count (int): Number of parts to split the box into (1 = no split)

        Returns:
            list: List of sub-bboxes as (lat1, lon1, lat2, lon2) tuples
    """
    
    if count <= 1:
        return [(lat1, lon1, lat2, lon2)]

    lat_step = (lat2 - lat1) / count
    lon_step = (lon2 - lon1) / count

    bboxes = []

    for i in range(count):
        for j in range(count):
            sub_lat1 = lat1 + i * lat_step
            sub_lat2 = sub_lat1 + lat_step
            sub_lon1 = lon1 + j * lon_step
            sub_lon2 = sub_lon1 + lon_step
            bboxes.append((sub_lat1, sub_lon1, sub_lat2, sub_lon2))

    return bboxes

def get_bbox_from_coords(coords):
    """
        Get the bounding box (min/max lat/lon) from a list of coordinates

        Args:
            coords (list): List of [lon, lat] pairs

        Returns:
            tuple: (min_lat, min_lon, max_lat, max_lon)
    """
    
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    
    return min(lats), min(lons), max(lats), max(lons)

def save_csv(df_row, arrondissement, strategy):
    """
        Save a single row of collected data into a CSV file named after strategy and district

        Args:
            df_row (DataFrame): Data row to append
            arrondissement (int): Arrondissement number
    """

    ## Reorder columns based on strategy
    if strategy == "incident_analysis":
        ordered_columns = [
            "timestamp", "lat", "lon", "incident_coords", "incident_count", "incident_magnitudes",
            "incident_delays", "incident_roads", "incident_id", "icon_category", "start_time",
            "end_time", "from_location", "to_location", "length", "time_validity", "probability",
            "num_reports", "last_report", "tmc_countryCode", "tmc_tableNumber", "tmc_tableVersion",
            "tmc_direction", "event_descriptions", "avg_speed", "free_flow_speed", "jam_factor",
            "temp", "wind", "rain"
        ]
    elif strategy == "traffic_analysis":
        ordered_columns = [
            "timestamp", "lat", "lon", "center_lat", "center_lon", "incident_coords", "incident_count",
            "incident_magnitudes", "incident_delays", "incident_roads", "incident_id", "icon_category",
            "start_time", "end_time", "from_location", "to_location", "length", "time_validity",
            "probability", "num_reports", "last_report", "tmc_countryCode", "tmc_tableNumber",
            "tmc_tableVersion", "tmc_direction", "event_descriptions", "avg_speed", "free_flow_speed",
            "jam_factor", "temp", "wind", "rain"
        ]
    else:
        logging.warning("Unknown strategy, column order not enforced.")
        ordered_columns = df_row.columns.tolist()

    ## Determine output file based on strategy
    output_file = f"live/live_data_{STRATEGY}.{arrondissement}.csv"

    ## Write header only if file doesn't exist
    header = not os.path.exists(output_file)

    ## Ensure all columns are present and reorder
    df_row = df_row[[col for col in ordered_columns if col in df_row.columns]]

    ## Append row
    df_row.to_csv(output_file, mode='a', index=False, header=header, encoding='utf-8-sig')
    
def collect(points):
    """
        Collect weather, traffic, and incident data for each point in a list of coordinates

        Args:
            points (list of tuples): List of (lon, lat) coordinates to analyze

        Returns:
            pd.DataFrame: DataFrame containing the collected information for each point or incident
    """
    
    global last_weather_time, last_weather_data

    collected_rows = []

    for lon, lat in points:
        try:
            ts = datetime.datetime.now().replace(microsecond=0)

            ## Weather data
            if MULTIPLE_WEATHER_CALLS:
                weather = get_weather(lat, lon)
            else:
                if last_weather_time is None or (datetime.datetime.now() - last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                    last_weather_data = get_weather(lat, lon)
                    last_weather_time = datetime.datetime.now()
                weather = last_weather_data

            ## Traffic data
            traffic = get_traffic_flow(lat, lon)

            ## Incidents list
            incidents_list = get_incidents(lat - DELTA_BBOX, lon - DELTA_BBOX, lat + DELTA_BBOX, lon + DELTA_BBOX)

            if not incidents_list:
                ## No incident found, create base row
                row = {
                    "timestamp": ts.isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "center_lat": lat,
                    "center_lon": lon,                    
                    "incident_count": 0,
                    "incident_magnitudes": [],
                    "incident_delays": [],
                    "incident_roads": [],
                    **traffic,
                    **weather
                }
                df_row = pd.DataFrame([row])
                save_csv(df_row, arrondissement, strategy = "traffic_analysis")
                collected_rows.append(row)
                logging.info(f"No incidents at {lat},{lon}, base row saved.")
            else:
                ## When multiple incidents are found
                for i, inc in enumerate(incidents_list, start=1):
                    coords = inc.get("incident_coords", [[lon, lat]])
                    lon_ref, lat_ref = coords[0][0], coords[0][1] if coords and len(coords[0]) == 2 else (lon, lat)

                    row = {
                        "timestamp": ts.isoformat(),
                        "lat": lat_ref,
                        "lon": lon_ref,
                        "center_lat": lat,
                        "center_lon": lon,                        
                        **inc,
                        **traffic,
                        **weather
                    }
                    df_row = pd.DataFrame([row])
                    save_csv(df_row, arrondissement, strategy = "traffic_analysis")
                    collected_rows.append(row)
                    logging.info(f"[{i}/{len(incidents_list)}] Incident at {lat_ref},{lon_ref} collected.")

        except Exception as e:
            logging.warning(f"Failed at {lat},{lon}: {e}")

    df = pd.DataFrame(collected_rows)

    ## Reorder columns for readability
    columns_order = [
            "timestamp", "lat", "lon", "center_lat", "center_lon", "incident_coords", "incident_count",
            "incident_magnitudes", "incident_delays", "incident_roads", "incident_id", "icon_category",
            "start_time", "end_time", "from_location", "to_location", "length", "time_validity",
            "probability", "num_reports", "last_report", "tmc_countryCode", "tmc_tableNumber",
            "tmc_tableVersion", "tmc_direction", "event_descriptions", "avg_speed", "free_flow_speed",
            "jam_factor", "temp", "wind", "rain"
    ]

    ## Keep only columns that exist in the final DataFrame
    columns_order = [col for col in columns_order if col in df.columns]
    return df[columns_order]

def collect_from_bbox(lat1, lon1, lat2, lon2):
    """
        Collects incidents within a bounding box and enriches them with traffic and weather data

        Args:
            lat1 (float): Southern latitude of the bounding box
            lon1 (float): Western longitude of the bounding box
            lat2 (float): Northern latitude of the bounding box
            lon2 (float): Eastern longitude of the bounding box

        Returns:
            pd.DataFrame: DataFrame with one row per incident enriched with traffic and weather data
    """
    
    global last_weather_time, last_weather_data
    collected_rows = []

    try:
        ## Unified call to get all incidents in the bounding box
        incidents_list = get_incidents(lat1, lon1, lat2, lon2)
        total = len(incidents_list)
        logging.info(f"{total} incident features found in this bbox. Beginning data collection...")

        for i, inc in enumerate(incidents_list, start=1):
            lat, lon = inc["lat"], inc["lon"]
            logging.info(f"[{i}/{total}] Collecting incident at {lat},{lon}")
            ts = datetime.datetime.now().replace(microsecond=0)

            try:
                ## Weather
                if MULTIPLE_WEATHER_CALLS:
                    weather = get_weather(lat, lon)
                else:
                    if last_weather_time is None or (ts- last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                        last_weather_data = get_weather(lat, lon)
                        last_weather_time = ts 
                    weather = last_weather_data

                ## Traffic
                traffic = get_traffic_flow(lat, lon)

                ## Assemble one row per incident
                row = {
                    "timestamp": ts.isoformat(),
                    "lat": lat,
                    "lon": lon,
                    **inc,
                    **traffic,
                    **weather
                }

                ## Save the row to disk and memory
                df_row = pd.DataFrame([row])
                save_csv(df_row, arrondissement, strategy = "incident_analysis")
                collected_rows.append(row)

                logging.info(f"[{i}/{total}] Data collected and saved.")

            except Exception as e:
                logging.warning(f"[{i}/{total}] Failed at {lat},{lon}: {e}")

    except Exception as e:
        logging.warning(f"Failed bbox {lat1},{lon1},{lat2},{lon2} : {e}")

    df = pd.DataFrame(collected_rows)

    ## Final column order for readability and consistency
    columns_order = [
            "timestamp", "lat", "lon", "incident_coords", "incident_count", "incident_magnitudes",
            "incident_delays", "incident_roads", "incident_id", "icon_category", "start_time",
            "end_time", "from_location", "to_location", "length", "time_validity", "probability",
            "num_reports", "last_report", "tmc_countryCode", "tmc_tableNumber", "tmc_tableVersion",
            "tmc_direction", "event_descriptions", "avg_speed", "free_flow_speed", "jam_factor",
            "temp", "wind", "rain"
    ]

    ## Filter out missing columns to avoid errors
    columns_order = [col for col in columns_order if col in df.columns]
    return df[columns_order]

def parse_arguments():
    """
        Parse command-line arguments to extract arrondissement number

        Returns:
            argparse.Namespace: The parsed arguments including 'arrondissement'
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arrondissement", type=int, default=17,
                        help="The Paris arrondissement number (1 to 20). Default is 8.")
    parser.add_argument("-s", "--strategy", type=str, default="incident_analysis",
                        help="Choose strategy: 'traffic_analysis' or 'incident_analysis'. Default is 'incident_analysis'.")
                        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    arrondissement = args.arrondissement
    
    STRATEGY = args.strategy

    ## Load list of coordinates from the specified arrondissement
    df_arr = pd.read_csv(ARRONDISSEMENTS_PATH)

    points = extract_point_list_from_geometry(
        pd.read_csv(ARRONDISSEMENTS_PATH).iloc[arrondissement - 1]["Geometry"]
    )

    logging.info(f"=== STARTING LIVE DATA COLLECTION ===")
    logging.info(f"Arrondissement selected: {arrondissement}")
    logging.info(f"Strategy selected: {STRATEGY}")
    logging.info(f"CSV output path: {CSV_PATH}")
    logging.info(f"Weather mode: {'MULTIPLE' if MULTIPLE_WEATHER_CALLS else 'SHARED'}")
    logging.info(f"Weather refresh delay: {WEATHER_REFRESH_DELAY} seconds")
    logging.info(f"Max API calls allowed: {MAX_CALLS_PER_DAY}")

    if STRATEGY == "traffic_analysis":
        logging.info(f"Number of points to sample: {NB_POINTS_TO_COLLECT}")
    elif STRATEGY == "incident_analysis":
        logging.info(f"BBox split count: {BBOX_SPLIT_COUNT}")

    success_count = 0
    MAX_ROWS = 10000

    while success_count < MAX_ROWS and calls_today + 3 <= MAX_CALLS_PER_DAY:

        if STRATEGY == "traffic_analysis":
            sampled_points = random.sample(points, min(NB_POINTS_TO_COLLECT, len(points)))
            logging.info(f"Sampling {len(sampled_points)} points for traffic analysis.")
            df = collect(sampled_points)

        elif STRATEGY == "incident_analysis":
            bboxes = split_bbox(*get_bbox_from_coords(points), BBOX_SPLIT_COUNT)
            for i, bbox in enumerate(bboxes, start=1):
                logging.info(f"[{i}/{len(bboxes)}] Processing bbox: {bbox}")
                df = collect_from_bbox(*bbox)
                logging.info(f"Collected {len(df)} rows from bbox.")

                if df is not None and not df.empty:
                    success_count += len(df)
                    logging.info(f"Progress: {success_count}/{MAX_ROWS} rows collected.")
                else:
                    logging.warning("No data collected from this bbox.")

                if success_count >= MAX_ROWS or calls_today + 3 > MAX_CALLS_PER_DAY:
                    break

                logging.info("Sleeping to respect API delay...")
                time.sleep(CALL_DELAY_SECONDS)

        else:
            logging.error("Unknown strategy selected. Exiting.")
            break

        if df is not None and not df.empty:
            logging.info(f"Progress updated: {success_count}/{MAX_ROWS} rows total.")
        else:
            logging.warning("Empty dataframe collected. Retrying after short delay.")

        time.sleep(CALL_DELAY_SECONDS)

    logging.info(f"=== FINISHED: {success_count} rows collected in total ===")
    print(f"Finished collecting {success_count} live entries.")