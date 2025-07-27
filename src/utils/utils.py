'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Utility functions for live data collection
'''

import os
import sys
import time
import requests
import logging
import datetime
from zoneinfo import ZoneInfo   
from src.core.constants import STRATEGY, MAX_CALLS_PER_DAY, CALL_DELAY_SECONDS

calls_today = 0  ## Counter for total API calls (weather)
last_weather_time = None  ## Last time weather was fetched
last_weather_data = None  ## Cached result if shared

## ========================
## Basic utilities
## ========================
def convert_to_local_timezone(api_time):
    """
        Convert a UTC time string from an API to Paris local timezone

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
        Make a safe API request with retry protection, logging, and call counting

        Args:
            url (str): The API endpoint URL
            params (dict): Parameters to include in the API request

        Returns:
            requests.Response: The response object from the API

        Raises:
            SystemExit: Exits the program if any request error occurs
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