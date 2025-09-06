'''
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Live data collection script for traffic or incident analysis"
'''

import datetime
import time
import random
import logging
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

# check if loading the .env file works
load_dotenv(find_dotenv())

# check if loading the .env file works
if not os.getenv("PYTHONPATH"):
    print("PYTHONPATH not set")
else:
    print(f"PYTHONPATH is set to: {os.getenv('PYTHONPATH')}")

# Add repo root (/workspace) so `from Services...` works both locally and in Docker
# import sys
# from pathlib import Path
# REPO_ROOT = Path(__file__).resolve().parents[2]  # .../traffic_prediction
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

# print(f"Repo root added to PYTHONPATH: {REPO_ROOT}")

from Services.utils.utils import save_csv
from Services.utils.utils_api_calls import (
    get_weather,
    get_traffic_flow,
    get_incidents
)
from Services.utils.utils_coordinates import (
    extract_point_list_from_geometry,
    get_bbox_from_coords,
    split_bbox
)
from Services.FastAPI.src.constants import (
    DELTA_BBOX,
    ARRONDISSEMENTS_PATH,
    CSV_PATH,
    MULTIPLE_WEATHER_CALLS,
    WEATHER_REFRESH_DELAY,
    MAX_CALLS_PER_DAY,
    CALL_DELAY_SECONDS,
    NB_POINTS_TO_COLLECT,
    BBOX_SPLIT_COUNT,
)

calls_today = 0  ## Counter for total API calls (weather)
last_weather_time = None  ## Last time weather was fetched
last_weather_data = None  ## Cached result if shared
tomtom_key = None
weather_key = None

def log_collection_config(arrondissement: int, strategy: str) -> None:
    """
        Log initial configuration for the data collection

        Args:
            arrondissement (int): Selected arrondissement
            strategy (str): Strategy used for data collection
    """
    
    logging.info("=== STARTING LIVE DATA COLLECTION ===")
    logging.info(f"Arrondissement selected: {arrondissement}")
    logging.info(f"Strategy selected: {strategy}")
    logging.info(f"CSV output path: {CSV_PATH}")
    logging.info(f"Weather mode: {'MULTIPLE' if MULTIPLE_WEATHER_CALLS else 'SHARED'}")
    logging.info(f"Weather refresh delay: {WEATHER_REFRESH_DELAY} seconds")
    logging.info(f"Max API calls allowed: {MAX_CALLS_PER_DAY}")

    if strategy == "traffic_analysis":
        logging.info(f"Number of points to sample: {NB_POINTS_TO_COLLECT}")
    elif strategy == "incident_analysis":
        logging.info(f"BBox split count: {BBOX_SPLIT_COUNT}")

def collect(
    points: list[tuple[float, float]],
    arrondissement: int
) -> pd.DataFrame:
    """
        Collect weather, traffic, and incident data for each point in a list of coordinates

        Args:
            points (list of tuples): List of (lon, lat) coordinates to analyze
            arrondissement (int): Selected arrondissement

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
                weather = get_weather(weather_key, lat, lon)
            else:
                if last_weather_time is None or (datetime.datetime.now() - last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                    last_weather_data = get_weather(weather_key, lat, lon)
                    last_weather_time = datetime.datetime.now()
                weather = last_weather_data

            ## Traffic data
            traffic = get_traffic_flow(tomtom_key, lat, lon)

            ## Incidents list
            incidents_list = get_incidents(
                tomtom_key,
                lat - DELTA_BBOX,
                lon - DELTA_BBOX,
                lat + DELTA_BBOX,
                lon + DELTA_BBOX
            )

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
                save_csv(df_row, arrondissement, strategy="traffic_analysis")
                collected_rows.append(row)
                logging.info(f"No incidents at {lat},{lon}, base row saved.")
            else:
                ## When multiple incidents are found
                for i, inc in enumerate(incidents_list, start=1):
                    coords = inc.get("incident_coords", [[lon, lat]])
                    lon_ref, lat_ref = coords[0] if coords and len(coords[0]) == 2 else (lon, lat)
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
                    save_csv(df_row, arrondissement, strategy="traffic_analysis")
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

def collect_from_bbox(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    arrondissement: int
) -> pd.DataFrame:
    """
        Collects incidents within a bounding box and enriches them with traffic and weather data

        Args:
            lat1 (float): Southern latitude of the bounding box
            lon1 (float): Western longitude of the bounding box
            lat2 (float): Northern latitude of the bounding box
            lon2 (float): Eastern longitude of the bounding box
            arrondissement (int): Selected arrondissement

        Returns:
            pd.DataFrame: DataFrame with one row per incident enriched with traffic and weather data
    """
    
    global last_weather_time, last_weather_data
    collected_rows = []

    try:
        ## Unified call to get all incidents in the bounding box
        incidents_list = get_incidents(tomtom_key, lat1, lon1, lat2, lon2)
        total = len(incidents_list)
        logging.info(f"{total} incident features found in this bbox. Beginning data collection...")

        for i, inc in enumerate(incidents_list, start=1):
            lat, lon = inc["lat"], inc["lon"]
            logging.info(f"[{i}/{total}] Collecting incident at {lat},{lon}")
            ts = datetime.datetime.now().replace(microsecond=0)

            try:
                ## Weather
                if MULTIPLE_WEATHER_CALLS:
                    weather = get_weather(weather_key, lat, lon)
                else:
                    if last_weather_time is None or (ts - last_weather_time).total_seconds() > WEATHER_REFRESH_DELAY:
                        last_weather_data = get_weather(weather_key, lat, lon)
                        last_weather_time = ts
                    weather = last_weather_data

                ## Traffic
                traffic = get_traffic_flow(tomtom_key, lat, lon)

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
                save_csv(df_row, arrondissement, strategy="incident_analysis")
                collected_rows.append(row)
                logging.info(f"[{i}/{total}] Data collected and saved.")

            except Exception as e:
                logging.warning(f"[{i}/{total}] Failed at {lat},{lon}: {e}")

    except Exception as e:
        logging.warning(f"Failed bbox {lat1},{lon1},{lat2},{lon2} : {e}")

    df = pd.DataFrame(collected_rows)

    ## Final column order for readability and consistency
    columns_order = [
        "timestamp","lat","lon","incident_coords","incident_count",
        "incident_magnitudes","incident_delays","incident_roads","incident_id",
        "icon_category","start_time","end_time","from_location","to_location",
        "length","time_validity","probability","num_reports","last_report",
        "tmc_countryCode","tmc_tableNumber","tmc_tableVersion","tmc_direction",
        "event_descriptions","avg_speed","free_flow_speed","jam_factor","temp",
        "wind","rain"
    ]

    ## Filter out missing columns to avoid errors
    columns_order = [col for col in columns_order if col in df.columns]
    return df[columns_order]

def run_live_data_pipeline(arrondissement: int, strategy: str) -> None:
    """
        Run the live data collection loop using the specified strategy

        Args:
            arrondissement (int): Paris arrondissement number
            strategy (str): 'traffic_analysis' or 'incident_analysis'
    """

    global tomtom_key, weather_key, calls_today
    ## Load API keys from .env or config
    tomtom_key, weather_key = load_api_keys(arrondissement)

    ## Load the polygon geometry of the selected arrondissement
    df_arr = pd.read_csv(ARRONDISSEMENTS_PATH)
    geometry = df_arr.iloc[arrondissement - 1]["Geometry"]

    ## Convert geometry string into a list of coordinate points
    points = extract_point_list_from_geometry(geometry)

    ## Log collection config (strategy and arrondissement)
    log_collection_config(arrondissement, strategy)

    ## Initialize tracking variables
    success_count = 0
    max_rows = 10000  ## Max number of rows to collect
    calls_today = 0   ## Tracks API calls made today (should be centralized globally)

    ## Main data collection loop
    while success_count < max_rows and calls_today + 3 <= MAX_CALLS_PER_DAY:

        ## Strategy 1: Random sampling inside polygon
        if strategy == "traffic_analysis":
            sampled_points = random.sample(points, min(NB_POINTS_TO_COLLECT, len(points)))
            logging.info(f"Sampling {len(sampled_points)} points for traffic analysis.")

            ## Collect traffic and weather data for these sampled points
            df = collect(sampled_points, arrondissement)

        ## Strategy 2: Use incidents within bounding boxes
        elif strategy == "incident_analysis":
            ## Split bounding box into smaller ones to avoid overloading
            bboxes = split_bbox(*get_bbox_from_coords(points), BBOX_SPLIT_COUNT)

            for i, bbox in enumerate(bboxes, start=1):
                logging.info(f"[{i}/{len(bboxes)}] Processing bbox: {bbox}")

                ## Collect incident and weather data from this sub-bbox
                df = collect_from_bbox(*bbox, arrondissement)
                logging.info(f"Collected {len(df)} rows from bbox.")

                ## Update success counter only if data was returned
                if df is not None and not df.empty:
                    success_count += len(df)
                    logging.info(f"Progress: {success_count}/{max_rows} rows collected.")
                else:
                    logging.warning("No data collected from this bbox.")

                ## Break early if we’ve reached our limits
                if success_count >= max_rows or calls_today + 3 > MAX_CALLS_PER_DAY:
                    break

                logging.info("Sleeping to respect API delay...")
                time.sleep(CALL_DELAY_SECONDS)

        ## Unknown strategy
        else:
            logging.error("Unknown strategy selected. Exiting.")
            return

        ## Handle empty DataFrame if no data was collected
        if df is not None and not df.empty:
            logging.info(f"Progress updated: {success_count}/{max_rows} rows total.")
        else:
            logging.warning("Empty dataframe collected. Retrying after short delay.")

        ## Respect delay between API calls
        time.sleep(CALL_DELAY_SECONDS)

    ## End of collection process
    logging.info(f"=== FINISHED: {success_count} rows collected in total ===")
    logging.info(f"Finished collecting {success_count} live entries.")

def load_api_keys(arrondissement: int) -> tuple:
    """
        Load TOMTOM and WEATHER API keys from environment variables

        Args:
            arrondissement (int): Arrondissement number

        Returns:
            tuple: (TOMTOM_KEY, WEATHER_KEY)
    """
    
    tomtom_key = os.getenv(f"TOMTOM_KEY_{arrondissement}")
    weather_key = os.getenv(f"WEATHER_KEY_{arrondissement}")
    return tomtom_key, weather_key

# Add main with run_live_data_pipeline
if __name__ == "__main__":
    import argparse
    import sys
    import io

    ## Ensure UTF-8 encoding for console output
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

    ## Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/live_data_collector.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    #Add print to see if the logging is working
    logging.info("Logging is set up.")

    ## Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/live", exist_ok=True)

    ## Valid parameters
    VALID_STRATEGIES = ["traffic_analysis", "incident_analysis"]
    VALID_ARRONDISSEMENTS = list(range(1, 21))

    def parse_arguments() -> argparse.Namespace:
        """
            Parse optional arguments (used only for validation)
        """
        
        parser = argparse.ArgumentParser(description="Live Data Collector CLI")
        parser.add_argument("-a", "--arrondissement", type=int, default=17, help="Paris arrondissement (1–20)")
        parser.add_argument("-s", "--strategy", type=str, default="incident_analysis", choices=["incident_analysis", "traffic_analysis"], help="Data collection strategy")
        return parser.parse_args()

    args = parse_arguments()

    if args.arrondissement is not None and args.arrondissement not in VALID_ARRONDISSEMENTS:
        logging.error(f"Invalid arrondissement '{args.arrondissement}'. Must be between 1 and 20.")
        sys.exit(1)

    if args.strategy is not None and args.strategy not in VALID_STRATEGIES:
        logging.error(f"Invalid strategy '{args.strategy}'. Choose from {VALID_STRATEGIES}.")
        sys.exit(1)

    # print 
    print("Starting live data collection...")
    logging.info(f"Starting live data collection for arrondissement {args.arrondissement} using strategy '{args.strategy}'")
    run_live_data_pipeline(args.arrondissement, args.strategy)
