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
import pandas as pd
import time
import json
import glob
import requests
import datetime
from zoneinfo import ZoneInfo

from src.core.constants import STRATEGY, MAX_CALLS_PER_DAY, CALL_DELAY_SECONDS
from src.core.logging_utils import get_logger, log_execution_time_and_path

calls_today = 0  ## Counter for total API calls (weather)
last_weather_time = None  ## Last time weather was fetched
last_weather_data = None  ## Cached result if shared

## Setup logging
logger = get_logger(__name__)

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

def save_summary_stats(df: pd.DataFrame, label: str, output_path: str) -> None:
    """
        Save summary statistics (mean, std, missing values, etc.) of a DataFrame to a CSV file

        Args:
            df (pd.DataFrame): Input data
            label (str): Dataset label for identification
            output_path (str): Path to output CSV file
    """
    
    ## Compute basic descriptive statistics
    desc = df.describe(include='all').transpose()

    ## Add missing value metrics
    desc["missing_count"] = df.isnull().sum()
    desc["missing_ratio"] = df.isnull().mean()
    desc["dataset"] = label

    ## Save to CSV, append if already exists
    desc.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

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
        logger.warning("Unknown strategy, column order not enforced.")
        ordered_columns = df_row.columns.tolist()

    ## Determine output file based on strategy
    output_file = f"live/live_data_{STRATEGY}.{arrondissement}.csv"

    ## Write header only if file doesn't exist
    header = not os.path.exists(output_file)

    ## Ensure all columns are present and reorder
    df_row = df_row[[col for col in ordered_columns if col in df_row.columns]]

    ## Append row
    df_row.to_csv(output_file, mode='a', index=False, header=header, encoding='utf-8-sig')  

@log_execution_time_and_path
def load_and_merge_files(strategy: str, data_dir: str) -> pd.DataFrame:
    """
        Load and merge all CSV and JSON files matching the strategy from a given directory

        Args:
            strategy (str): Strategy name ('traffic_analysis' or 'incident_analysis')
            data_dir (str): Path to the folder containing live data files

        Returns:
            pd.DataFrame: Merged DataFrame from all matching files
    """

    ## Construct file pattern based on strategy (e.g., 'live_data_traffic_analysis*')
    pattern = f"live_data_{strategy}*"
    full_path = os.path.join(data_dir, pattern)

    ## Search for all files matching the pattern
    files = glob.glob(full_path)

    ## Loop through each matched file and load it depending on its extension
    dfs = []    
    for file in files:
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(file, low_memory=False))
        elif file.endswith(".json"):
            dfs.append(pd.read_json(file))

    ## If no files were loaded, raise an error
    if not dfs:
        raise FileNotFoundError(f"No files found for pattern: {full_path}")

    ## Concatenate all loaded DataFrames into one
    return pd.concat(dfs, ignore_index=True)

def clean_columns_and_rows(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
        Clean dataset by dropping columns and rows with too many missing values

        Args:
            df (pd.DataFrame): The input DataFrame to clean
            threshold (float): Maximum allowed percentage of missing values (default: 0.9 = 90%)

        Returns:
            pd.DataFrame: A cleaned DataFrame with low-quality columns and rows removed
    """
    
    ## Drop columns with more than 'threshold' proportion of missing values
    col_thresh = int((1 - threshold) * len(df))
    df = df.dropna(axis=1, thresh=col_thresh)

    ## Drop rows with more than 'threshold' proportion of missing values
    row_thresh = int((1 - threshold) * df.shape[1])
    df = df.dropna(axis=0, thresh=row_thresh)

    return df
 
@log_execution_time_and_path
def create_target(df: pd.DataFrame, target: str) -> pd.Series:
    """
        Extracts the target column from the dataset. Raises error if it doesn't exist

        Args:
            df (pd.DataFrame): Input data with target column
            target (str): Target column name to extract

        Returns:
            pd.Series: The target variable
    """
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")
    
    return df[target]

def safe_eval(val):
    """
        Safely parse stringified lists from JSON fields such as delays or magnitudes

        Args:
            val (str or list): Input to convert into a list

        Returns:
            list: Parsed list or empty list on failure
    """
    
    try:
        if isinstance(val, str):
            return json.loads(val.replace("'", '"'))
        elif isinstance(val, list):
            return val
    except Exception:
        return []
    
    return []

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
        logger.error(f"Max daily API calls reached ({calls_today}/{MAX_CALLS_PER_DAY}). Exiting.")
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
        logger.error(f"HTTP error: {errh}. Exiting.")
        sys.exit(1)

    except requests.exceptions.RequestException as err:
        logger.error(f"Request failed: {err}. Exiting.")
        sys.exit(1)