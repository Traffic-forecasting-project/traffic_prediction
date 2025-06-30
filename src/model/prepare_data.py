''' 
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.6.3"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data preparation module: feature engineering and target construction for traffic prediction."
'''

import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Literal
from sklearn.impute import SimpleImputer
from logging_utils import log_execution_time_and_path

## Mapping of all supported targets with allowed strategies
TARGET_METADATA = {
    "congestion_label": {"strategies": ["traffic_analysis", "incident_analysis"]},
    "jam_factor": {"strategies": ["traffic_analysis", "incident_analysis"]},
    "incident_duration_min": {"strategies": ["incident_analysis"]}
}

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
            dfs.append(pd.read_csv(file))
        elif file.endswith(".json"):
            dfs.append(pd.read_json(file))

    ## If no files were loaded, raise an error
    if not dfs:
        raise FileNotFoundError(f"No files found for pattern: {full_path}")

    ## Concatenate all loaded DataFrames into one
    return pd.concat(dfs, ignore_index=True)

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

@log_execution_time_and_path
def create_features(df: pd.DataFrame, target: str, strategy: str) -> pd.DataFrame:
    """
        Generate target-specific and general features for the model

        Args:
            df (pd.DataFrame): Raw input dataframe
            target (str): Target variable name
            strategy (str): Analysis strategy

        Returns:
            pd.DataFrame: Feature-enhanced dataframe
    """
    
    df = df.copy()

    ## Convert timestamp columns to datetime format safely
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["start_time"] = pd.to_datetime(df.get("start_time"), errors="coerce")
    df["end_time"] = pd.to_datetime(df.get("end_time"), errors="coerce")

    ## Create basic time-based features for temporal modeling
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5

    ## Estimate a pseudo-arrondissement by binning latitude (approximate spatial location)
    if "latitude" in df.columns:
        df["arrondissement"] = pd.cut(df["latitude"], bins=20, labels=False)

    ## ---------- Target-specific logic ---------- #
    if target == "jam_factor":
        
        ## If jam_factor is missing, we generate synthetic uniform values as fallback
        if "jam_factor" not in df.columns or df["jam_factor"].isna().all():
            df["jam_factor"] = np.random.uniform(0, 10, len(df))
        else:
            ## Otherwise, impute missing values using mean
            df["jam_factor"] = df["jam_factor"].fillna(df["jam_factor"].mean())

    elif target == "congestion_label":
        
        ## If jam_factor is missing, generate synthetic values
        if "jam_factor" not in df.columns:
            df["jam_factor"] = np.random.uniform(0, 10, len(df))
        
        ## Binary classification: jam > 5 considered congested
        df["congestion_label"] = (df["jam_factor"] > 5).astype(int)

    elif target == "incident_duration_min":
        
        ## Ensure that datetime fields are properly parsed
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

        ## Only keep rows with valid datetime objects
        valid_time = df["start_time"].apply(lambda x: pd.notnull(x) and isinstance(x, pd.Timestamp)) & \
                     df["end_time"].apply(lambda x: pd.notnull(x) and isinstance(x, pd.Timestamp))
        df = df[valid_time].copy()

        ## Compute duration in minutes
        df["incident_duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

        ## Remove non-positive or excessively long durations (> 3 hours)
        df = df[df["incident_duration_min"].notna() & (df["incident_duration_min"] > 0)]
        df = df[df["incident_duration_min"] < 180]

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