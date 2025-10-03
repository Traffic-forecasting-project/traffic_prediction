'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Utility functions for live data collection
'''


import sys
import time
import requests
import json
import ast
import logging
import pandas as pd
from shapely.geometry import shape

from Services.FastAPI.src.constants import MAX_CALLS_PER_DAY, CALL_DELAY_SECONDS, WEATHER_KEY, TOMTOM_KEY


calls_today = 0 ## Global variable that tracks how many API calls were made today
last_weather_time = None
last_weather_data = None

## ========================
## Basic utilities
## ========================
def convert_to_local_timezone(api_time):
    """
    Convert a UTC time string from an API to Paris local timezone.

    Args:
        api_time (str): ISO-format UTC time string (e.g., '2023-01-01T12:00:00Z')

    Returns:
        datetime.datetime: Datetime object converted to Europe/Paris timezone
    """
    if api_time is None:
        return api_time

    ## Parse ISO string as UTC datetime
    time_utc = datetime.datetime.fromisoformat(api_time.replace("Z", "+00:00"))

    ## Convert to Paris timezone
    dt_paris = time_utc.astimezone(ZoneInfo("Europe/Paris"))
    return dt_paris


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
    
def safe_request(url: str, params: dict) -> requests.Response:
    """
    Make a safe API request with retry protection, logging, and call counting.

    Args:
        url (str): The API endpoint URL.
        params (dict): Parameters to include in the API request.

    Returns:
        requests.Response: The response object from the API.

    Raises:
        SystemExit: Exits the program if any request error occurs.
    """
    
    global calls_today

    ## Check if daily limit has been reached
    if calls_today >= MAX_CALLS_PER_DAY:
        logging.error(f"Max daily API calls reached ({calls_today}/{MAX_CALLS_PER_DAY}). Exiting.")
        sys.exit(1)

    ## Respect API rate limit with delay
    time.sleep(CALL_DELAY_SECONDS)

    try:
        ## Send GET request
        response = requests.get(url, params=params, timeout=15)

        ## Raise an error for HTTP 4xx or 5xx
        response.raise_for_status()

        ## Increment call counter
        calls_today += 1

        return response

    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP error: {errh}. Exiting.")
        sys.exit(1)

    except requests.exceptions.RequestException as err:
        logging.error(f"Request failed: {err}. Exiting.")
        sys.exit(1)

## ========================
## API Call utilities
## ========================
def get_weather(lat, lon):
    """
    Retrieves current weather data from OpenWeatherMap API for a given location.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        dict: Dictionary containing temperature, wind speed, and rain volume (if available).
    """
    ## OpenWeatherMap endpoint
    url = "https://api.openweathermap.org/data/2.5/weather"

    ## Request parameters
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_KEY,
        "units": "metric"
    }

    ## Send request and parse response
    r = safe_request(url, params)
    data = r.json()

    ## Extract useful weather data
    temp = data.get("main", {}).get("temp")
    wind = data.get("wind", {}).get("speed")
    rain = data.get("rain", {}).get("1h") if "rain" in data else 0.0

    return {
        "temp": temp,
        "wind": wind,
        "rain": rain
    }

def get_traffic_flow(lat, lon):
    """
    Retrieves real-time traffic flow data from the TomTom Traffic API for a specific location.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.

    Returns:
        dict: Dictionary containing average speed, free-flow speed, and jam factor.
    """
    ## TomTom Traffic Flow API endpoint
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    ## Parameters for the API call
    params = {
        "point": f"{lat},{lon}",
        "key": TOMTOM_KEY
    }

    ## Perform API request safely
    r = safe_request(url, params)
    data = r.json()

    ## Navigate into the API response
    flow_data = data.get("flowSegmentData", {})

    ## Compute traffic-related metrics
    avg_speed = flow_data.get("currentSpeed")
    free_flow_speed = flow_data.get("freeFlowSpeed")
    jam_factor = (
        flow_data.get("currentTravelTime") / flow_data.get("freeFlowTravelTime")
        if flow_data.get("freeFlowTravelTime") and flow_data.get("currentTravelTime")
        else None
    )

    return {
        "avg_speed": avg_speed,
        "free_flow_speed": free_flow_speed,
        "jam_factor": jam_factor
    }

def get_incidents(lat1, lon1, lat2, lon2):
    """
    Fetches incident data from TomTom API within a specified bounding box.

    Args:
        lat1 (float): Southern latitude of the bounding box.
        lon1 (float): Western longitude of the bounding box.
        lat2 (float): Northern latitude of the bounding box.
        lon2 (float): Eastern longitude of the bounding box.

    Returns:
        List[dict]: List of cleaned incident dictionaries containing metadata and coordinates.
    """
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{...}}}",  ## Long query structure
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    ## Perform API request safely
    r = safe_request(url, params)
    data = r.json()

    incidents = data.get("incidents", [])
    logging.info(f"{len(incidents)} incidents (uniques) extracted.")

    processed_incidents = []

    for inc in incidents:
        props = inc.get("properties", {})
        coords = inc.get("geometry", {}).get("coordinates", [])

        ## Skip if no coordinates
        if not coords or not isinstance(coords, list):
            continue

        ## Keep only first point if sampling is limited
        if not SAMPLE_ALL_INCIDENT_POINTS:
            coords = coords[0:1]

        ## Iterate over incident points
        for coord in coords:
            if not isinstance(coord, list) or len(coord) != 2:
                continue

            lon_val, lat_val = coord[0], coord[1]

            ## Build structured incident dictionary
            incident_data = {
                "incident_count": 1,
                "incident_magnitudes": [props.get("magnitudeOfDelay")],
                "incident_delays": [props.get("delay")],
                "incident_roads": [props.get("roadNumbers", [])],
                "incident_id": props.get("id"),
                "icon_category": props.get("iconCategory"),
                "start_time": convert_to_local_timezone(props.get("startTime")),
                "end_time": convert_to_local_timezone(props.get("endTime")),
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
                "incident_coords": coords,
                "lat": lat_val,
                "lon": lon_val
            }

            processed_incidents.append(incident_data)

    return processed_incidents

def get_incidents_per_coordinate(lat1, lon1, lat2, lon2, ts=None):
    """
    [DEPRECATED] Retrieve granular incident data from TomTom API,
    returning one row per coordinate involved in the incident geometry.

    Args:
        lat1 (float): Minimum latitude of the bounding box.
        lon1 (float): Minimum longitude of the bounding box.
        lat2 (float): Maximum latitude of the bounding box.
        lon2 (float): Maximum longitude of the bounding box.

    Returns:
        list of dict: One entry per coordinate with detailed incident properties.
    """

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{...}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    ## Request API and parse response
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

        ## Each coordinate becomes its own row
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
                    "event_descriptions": [e.get("description") for e in props.get("events", []) if "description" in e],
                }

                result_rows.append(row)

    logging.info(f"Number of incident coordinates found: {len(result_rows)}")
    return result_rows


## ========================
## Coordinate utilities
## ========================
def load_arrondissement_data(path: str) -> dict:
    """
    [DEPRECATED] Load polygon definitions for each Paris arrondissement from a CSV file.

    Args:
        path (str): Path to the CSV file containing 'arrondissement' and 'polygon' columns.

    Returns:
        dict: Dictionary mapping arrondissement numbers to lists of [lon, lat] points.
    """
    ## Read CSV containing arrondissement and polygon columns
    df = pd.read_csv(path)
    arrondissement_polygons = {}

    ## Iterate over rows to extract polygon data
    for _, row in df.iterrows():
        arr = int(row["arrondissement"])  ## Convert arrondissement to int
        points = json.loads(row["polygon"])  ## Parse polygon string into list of points
        arrondissement_polygons[arr] = points  ## Store mapping in dictionary

    return arrondissement_polygons

def load_arrondissement_polygons(path):
    """
    Loads the polygon coordinates of each Paris arrondissement from a CSV file.

    Args:
        path (str): Path to the CSV file containing arrondissement boundaries.

    Returns:
        dict: A dictionary mapping arrondissement number to its polygon coordinates.
    """
    ## Load the CSV containing arrondissement numbers and their coordinates
    df = pd.read_csv(path)

    ## Initialize empty dictionary for mapping arrondissement to coordinates
    arr_dict = {}

    ## Iterate over each row to extract and parse the coordinates
    for _, row in df.iterrows():
        num = row["num"]  ## Arrondissement number
        coords_str = row["coordinates"]  ## Coordinate string (likely JSON)

        try:
            coords = json.loads(coords_str)  ## Try parsing as JSON
        except Exception:
            coords = eval(coords_str)  ## Fallback to eval if JSON parsing fails

        arr_dict[num] = coords  ## Store in dictionary

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
    Check whether a point is inside a polygon using the ray casting algorithm.

    Args:
        x (float): Longitude of the point.
        y (float): Latitude of the point.
        polygon (list): List of [lon, lat] pairs defining the polygon.

    Returns:
        bool: True if the point is inside the polygon, otherwise False.
    """
    n = len(polygon)
    inside = False

    ## Start with the first point of the polygon
    p1x, p1y = polygon[0]

    ## Iterate over each edge in the polygon
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]  ## Loop back to the start at the end

        ## Check if the horizontal ray crosses the edge vertically
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):

                    ## Compute the x-intersection with the edge
                    if p1y != p2y:
                        xinters = ((y - p1y) * (p2x - p1x)) / (p2y - p1y + 1e-9) + p1x
                    else:
                        xinters = p1x

                    ## Flip 'inside' if the point is to the left of the edge
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        ## Move to next edge
        p1x, p1y = p2x, p2y

    return inside

def get_coordinates(polygon, nb_points):
    """
    Generate random coordinates that lie within a given polygon.

    Args:
        polygon (list): List of [lon, lat] pairs forming the polygon boundary.
        nb_points (int): Number of valid random points to generate inside the polygon.

    Returns:
        list: List of (lon, lat) tuples contained in the polygon.
    """
    ## Compute bounding box around the polygon
    longitudes = [p[0] for p in polygon]
    latitudes = [p[1] for p in polygon]

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    coordinates = []

    ## Randomly sample points within the bounding box until they fall inside the polygon
    while len(coordinates) < nb_points:
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)

        if point_inside_polygon(lon, lat, polygon):
            coordinates.append((lon, lat))

    return coordinates


def split_bbox(lat1, lon1, lat2, lon2, count):
    """
    Split a bounding box into smaller sub-bounding boxes.

    Args:
        lat1 (float): South latitude of the original bbox.
        lon1 (float): West longitude of the original bbox.
        lat2 (float): North latitude of the original bbox.
        lon2 (float): East longitude of the original bbox.
        count (int): Number of parts to split each side into (1 = no split).

    Returns:
        list: List of sub-bounding boxes as (lat1, lon1, lat2, lon2) tuples.
    """

    ## If no split is requested, return the full bounding box
    if count <= 1:
        return [(lat1, lon1, lat2, lon2)]

    ## Compute latitude and longitude step sizes
    lat_step = (lat2 - lat1) / count
    lon_step = (lon2 - lon1) / count

    bboxes = []

    ## Loop to generate each sub-bounding box
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
    Get the bounding box (min/max lat/lon) from a list of coordinates.

    Args:
        coords (list): List of [lon, lat] pairs.

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon)
    """

    ## Extract all longitude and latitude values from the coordinate list
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]

    ## Return the bounding box: (min_latitude, min_longitude, max_latitude, max_longitude)
    return min(lats), min(lons), max(lats), max(lons)
