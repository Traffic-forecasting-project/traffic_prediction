''' 
__author__ = -
__copyright__ = None
__version__ = "1.6.3"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data preparation module: feature engineering and target construction for traffic prediction."
'''

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from Services.FastAPI.src.logging_utils import get_logger, log_execution_time_and_path

from Services.FastAPI.src.constants import (
    FILTER_STANDARD_INCIDENTS,
    TARGET_METADATA,
    ENABLE_EXTRA_FEATURES
)

from Services.utils.utils import (
    safe_eval,
    clean_columns_and_rows,
    load_and_merge_files
)

logger = get_logger("prepare_data")

def detect_leakage(df: pd.DataFrame, target_column: str, threshold: float = 0.9, drop: bool = False) -> pd.DataFrame:
    """
        Detect and optionally remove features strongly correlated with the target variable (potential data leakage)

        Args:
            df (pd.DataFrame): The dataset including the target column
            target_column (str): The name of the target variable
            threshold (float): Threshold for flagging strong correlation (default = 0.9)
            drop (bool): If True, remove the leaked features from the DataFrame

        Returns:
            pd.DataFrame: The cleaned DataFrame if drop=True, otherwise the original DataFrame
    """
    
    ## Check if target column exists in the dataset
    if target_column not in df.columns:
        logger.info(f"[leakage detection] Target column '{target_column}' not found in DataFrame.")
        return df

    logger.info("Checking for data leakage...")

    ## Keep only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include='number').copy()
    numeric_df = numeric_df.dropna()  ## Remove rows with missing values

    ## Ensure the target column is in the numeric subset
    if target_column in numeric_df.columns:
        
        ## Compute correlation with target
        target_corr = numeric_df.corr()[target_column].drop(target_column)

        ## Identify features with absolute correlation above the threshold
        leaked_features = target_corr[abs(target_corr) >= threshold]

        ## If any feature shows strong correlation, report them
        if not leaked_features.empty:
            logger.info(f"\t Potential data leakage detected! Features strongly correlated with '{target_column}':")
            logger.info(f"\t {leaked_features.sort_values(ascending=False)}")

            ## Optionally drop leaked features
            if drop:
                logger.info(f"\t Dropping {len(leaked_features)} leaked features: {list(leaked_features.index)}")
                df = df.drop(columns=leaked_features.index)
        else:
            logger.info("\t No strong leakage detected.")
    else:
        ## Target column is missing from numeric columns or too many NaNs
        logger.info(f"\t [leakage detection] Target column '{target_column}' is not numeric or has too many NaNs.")

    return df

def advanced_create_features(df: pd.DataFrame, selected_features: list = None) -> pd.DataFrame:
    """
        Generate advanced features to improve prediction
        Only used when enable_extra_features is True

        Args:
            df : pd.DataFrame
                DataFrame containing raw or partially processed input features
            selected_features : list
                List of feature names to generate. Valid options include:
                ['delay_before_start', 'time_until_end', 'minutes_since_last_report',
                 'weekday_flags', 'hour_x_jam', 'time_period']

        Returns:

            pd.DataFrame
                DataFrame with new engineered features
    """

    if selected_features is None:
        selected_features = [
            'delay_before_start', 'time_until_end', 'minutes_since_last_report', 'weekday_flags',
            'hour_x_jam', 'time_period', 'minutes_since_midnight', 'is_weekend', 'jam_x_magnitude',
            'incident_progress_ratio', 'weekend_flag', 'hour_bin', 'jam_magnitude_ratio',
            'incident_duration_bucket', 'length_x_jam', 'weather_combo', 'hour_x_weekend',
            'slowdown_ratio', 'log_length', 'log_delay_before_start', 'hour_period', 'is_slow',
            'relative_slowdown', 'temp_x_slowdown', 'hour_x_delay', 'weekday_x_length',
            'is_rush_hour', 'is_heavy_delay', 'delay_per_km', 'log_slowdown_ratio'
        ]

    # == Convert datetime columns ==
    ## Ensure datetime columns are timezone-naive and converted properly
    for col in ["timestamp", "start_time", "end_time", "last_report"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.tz_localize(None)

    # === BASIC FEATURES ===
    
    ## Delay before start
    if "delay_before_start" in selected_features:
        if "timestamp" in df.columns and "start_time" in df.columns:
            df["delay_before_start"] = (
                (df["start_time"] - df["timestamp"]).dt.total_seconds() / 60
            )

    ## Time until end
    if "time_until_end" in selected_features:
        if "timestamp" in df.columns and "end_time" in df.columns:
            df["time_until_end"] = (
                (df["end_time"] - df["timestamp"]).dt.total_seconds() / 60
            )

    ## Minutes since last report
    if "minutes_since_last_report" in selected_features:
        if "timestamp" in df.columns and "last_report" in df.columns:
            df["minutes_since_last_report"] = (
                (df["timestamp"] - df["last_report"]).dt.total_seconds() / 60
            )

    ## Weekday flags
    if "weekday_flags" in selected_features:
        if "weekday" in df.columns:
            df["is_monday"] = df["weekday"] == 0
            df["is_friday"] = df["weekday"] == 4

    ## Hour x Jam interaction
    if "hour_x_jam" in selected_features:
        if "hour" in df.columns and "jam_factor" in df.columns:
            df["hour_x_jam"] = df["hour"] * df["jam_factor"]

    ## Time period categorization
    if "time_period" in selected_features:
        def time_period(hour):
            if pd.isna(hour): return "unknown"
            elif 7 <= hour <= 9 or 17 <= hour <= 19: return "rush"
            elif 0 <= hour <= 6: return "night"
            else: return "normal"

        if "hour" in df.columns:
            df["time_period"] = df["hour"].apply(time_period)
            if "time_period" in df.columns:
                df = pd.get_dummies(df, columns=["time_period"], drop_first=True)

    # === INTERMEDIATE FEATURES ===
    
    ## Minutes since midnight
    if "minutes_since_midnight" in selected_features:
        if "hour" in df.columns and "minute" in df.columns:
            df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]

    ## Weekend binary flag
    if "is_weekend" in selected_features:
        if "weekday" in df.columns:
            df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    ## Jam x magnitude
    if "jam_x_magnitude" in selected_features:
        if "jam_factor" in df.columns and "mean_magnitude" in df.columns:
            df["jam_x_magnitude"] = df["jam_factor"] * df["mean_magnitude"]

    ## Drop leakage-related columns
    leakage_cols = []
    if "time_until_end" in selected_features: leakage_cols += ["end_time", "time_until_end"]
    if "minutes_since_last_report" in selected_features: leakage_cols.append("last_report")
    df.drop(columns=[col for col in leakage_cols if col in df.columns], inplace=True)

    # === ADVANCED FEATURES ===
    
    ## Incident progress ratio
    if "incident_progress_ratio" in selected_features:
        if all(col in df.columns for col in ["timestamp", "start_time", "end_time"]):
            duration = (df["end_time"] - df["start_time"]).dt.total_seconds()
            elapsed = (df["timestamp"] - df["start_time"]).dt.total_seconds()
            df["incident_progress_ratio"] = elapsed / duration.replace(0, 1)
            df["incident_progress_ratio"] = df["incident_progress_ratio"].clip(0, 1)

    ## Weekend flag again (boolean)
    if "weekend_flag" in selected_features:
        if "weekday" in df.columns:
            df["is_weekend"] = df["weekday"].isin([5, 6])

    ## Hour binning
    if "hour_bin" in selected_features:
        if "hour" in df.columns:
            bins = [0, 6, 10, 16, 20, 24]
            labels = ["night", "morning", "afternoon", "evening", "late"]
            df["hour_bin"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False)
            df = pd.get_dummies(df, columns=["hour_bin"], drop_first=True)

    ## Jam to magnitude ratio
    if "jam_magnitude_ratio" in selected_features:
        if all(col in df.columns for col in ["jam_factor", "mean_magnitude"]):
            df["jam_magnitude_ratio"] = df["jam_factor"] / (df["mean_magnitude"] + 1e-6)

    ## Incident duration buckets
    if "incident_duration_bucket" in selected_features:
        if "incident_duration_min" in df.columns:
            bins = [0, 10, 30, 60, 120, 999]
            labels = ["short", "medium", "long", "very_long", "extreme"]
            df["incident_duration_bucket"] = pd.cut(df["incident_duration_min"], bins=bins, labels=labels)
            df = pd.get_dummies(df, columns=["incident_duration_bucket"], drop_first=True)

    ## Hour period encoding
    if "hour" in df.columns and "hour_period" in selected_features:
        df["hour_period"] = pd.cut(
            df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True)
        df = pd.get_dummies(df, columns=["hour_period"], drop_first=True)

    ## Slow traffic indicator
    if all(col in df.columns for col in ["avg_speed", "free_flow_speed"]) and "slowdown_ratio" in selected_features:
        df["slowdown_ratio"] = df["avg_speed"] / (df["free_flow_speed"] + 1e-5)  ## Avoid division by zero
        # df["slowdown_ratio"] = df["avg_speed"] / df["free_flow_speed"]
        df["is_slow"] = (df["slowdown_ratio"] < 0.6).astype(int)

    ## Hour x Weekend interaction
    if "hour" in df.columns and "is_weekend" in df.columns and "hour_x_weekend" in selected_features:
        df["hour_x_weekend"] = df["hour"] * df["is_weekend"]

    ## Length x Jam interaction
    if "length" in df.columns and "jam_factor" in df.columns and "length_x_jam" in selected_features:
        df["length_x_jam"] = df["length"] * df["jam_factor"]

    ## Log transform of length
    if "length" in df.columns and "log_length" in selected_features:
        df["log_length"] = np.log1p(df["length"].clip(lower=0))
        # df["log_length"] = np.log1p(df["length"]
        
    ## Log transform of delay before start
    if "delay_before_start" in df.columns and "log_delay_before_start" in selected_features:
        df["log_delay_before_start"] = np.log1p(df["delay_before_start"].clip(lower=0))
        # df["log_delay_before_start"] = np.log1p(df["delay_before_start"])
        
    ## Weather combo feature
    if all(x in df.columns for x in ["rain", "temp", "wind"]) and "weather_combo" in selected_features:
        df["weather_combo"] = df["rain"].astype(str) + "_" + pd.cut(df["temp"], bins=4, labels=False).astype(str) + "_" + pd.cut(df["wind"], bins=4, labels=False).astype(str)
        # df["weather_combo"] = df["wind"] * df["rain"]
        
    ## Relative slowdown
    if all(col in df.columns for col in ["avg_speed", "free_flow_speed"]) and "relative_slowdown" in selected_features:
        df["relative_slowdown"] = (df["free_flow_speed"] - df["avg_speed"]) / df["free_flow_speed"]
        df["relative_slowdown"] = df["relative_slowdown"].clip(lower=0)

    ## Temp x slowdown interaction
    if "temp" in df.columns and "relative_slowdown" in df.columns and "temp_x_slowdown" in selected_features:
        df["temp_x_slowdown"] = df["temp"] * df["relative_slowdown"]

    ## Hour x delay interaction
    if "delay_before_start" in df.columns and "hour" in df.columns and "hour_x_delay" in selected_features:
        df["hour_x_delay"] = df["hour"] * df["delay_before_start"]

    ## Weekday x length interaction
    if "weekday" in df.columns and "length" in df.columns and "weekday_x_length" in selected_features:
        df["weekday_x_length"] = df["weekday"] * df["length"]

    ## Rush hour flag
    if "hour" in df.columns and "is_rush_hour" in selected_features:
        df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if 7 <= h <= 9 or 16 <= h <= 19 else 0)

    ## Heavy delay flag
    if "delay_before_start" in df.columns and "is_heavy_delay" in selected_features:
        df["is_heavy_delay"] = (df["delay_before_start"] > 30).astype(int)

    ## Delay per km
    if "length" in df.columns and "delay_before_start" in df.columns and "delay_per_km" in selected_features:
        df["delay_per_km"] = df["delay_before_start"] / (df["length"] + 1e-3)

    ## Log transform of slowdown ratio
    if "slowdown_ratio" in df.columns and "log_slowdown_ratio" in selected_features:
        df["log_slowdown_ratio"] = np.log1p(df["slowdown_ratio"])

    return df
    
#@log_execution_time_and_path
def create_features(df: pd.DataFrame, data_dir_output: str, strategy: str, target_column: str = "incident_duration_min") -> pd.DataFrame:
    """
        Create additional features from the input DataFrame based on the selected strategy

        Args:
            df (pd.DataFrame): Input data.
            strategy (str): Feature engineering strategy ('incident_analysis', etc.)
            target_column (str, optional): Column name to use as the target. Defaults to "incident_duration_min"

        Returns:
            pd.DataFrame: DataFrame with new features added
    """

    logger.info("Create new features, including target variable.")
  
    df = df.copy()

    ## Convert timestamp columns to datetime format safely
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["start_time"] = pd.to_datetime(df.get("start_time"), errors="coerce")
    df["end_time"] = pd.to_datetime(df.get("end_time"), errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday

    ## Create basic time-based features for temporal modeling
    df["is_weekend"] = df["weekday"] >= 5

    ## Estimate a pseudo-arrondissement by binning latitude (approximate spatial location)
    if "latitude" in df.columns:
        df["arrondissement"] = pd.cut(df["latitude"], bins=20, labels=False)

    ## ---------- Target-specific logic ---------- #
    if target_column  == "jam_factor":
        
        ## If jam_factor is missing, we generate synthetic uniform values as fallback
        if "jam_factor" not in df.columns or df["jam_factor"].isna().all():
            df["jam_factor"] = np.random.uniform(0, 10, len(df))
        else:
            ## Otherwise, impute missing values using mean
            df["jam_factor"] = df["jam_factor"].fillna(df["jam_factor"].mean())

    elif target_column  == "incident_duration_min":
        
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

    elif target_column  == "mean_magnitude":
        
        ## Parse 'incident_magnitudes' and compute mean
        df["mean_magnitude"] = df["incident_magnitudes"].apply(lambda x: np.nanmean(safe_eval(x)) if safe_eval(x) else np.nan)

        ## Impute missing values (no specific cap here)
        imputer = SimpleImputer(strategy="mean")
        df[["mean_magnitude"]] = imputer.fit_transform(df[["mean_magnitude"]])

    ## OPTIONAL : Extra engineered features to improve model performance
    if ENABLE_EXTRA_FEATURES:
        ## GOOD RESULTS !!!
        # df = advanced_create_features(df, selected_features=['minutes_since_last_report', 'hour_x_jam'])    
        # df = advanced_create_features(df, selected_features=['hour_x_jam', 'time_period'])
        # df = advanced_create_features(df, selected_features=['delay_before_start', 'time_until_end', 'minutes_since_last_report', 'weekday_flags', 'hour_x_jam', 'time_period'])
        #df = advanced_create_features(df, selected_features=['delay_before_start', 'time_until_end', 'minutes_since_last_report', 'weekday_flags', 'hour_x_jam', 'time_period', 'minutes_since_midnight', 'is_weekend', 'jam_x_magnitude'])
        df = advanced_create_features(df, selected_features= ['delay_before_start','time_until_end','minutes_since_last_report','weekday_flags','hour_x_jam','time_period','minutes_since_midnight','is_weekend','jam_x_magnitude','incident_progress_ratio', 'weekend_flag', 'hour_bin', 'jam_magnitude_ratio', 'incident_duration_bucket', 'length_x_jam', 'weather_combo', 'hour_x_weekend', 'slowdown_ratio', 'log_length', 'log_delay_before_start', 'hour_period', 'is_slow', 'relative_slowdown', 'temp_x_slowdown', 'hour_x_delay', 'weekday_x_length', 'is_rush_hour', 'is_heavy_delay', 'delay_per_km', 'log_slowdown_ratio' ])

    ## Clean columns and rows with excessive missing values
    df = clean_columns_and_rows(df)
    
    ## Detect potential data leakage by identifying features too correlated with the target (potential data leakage/overfitting)
    ## If drop=True, those features are automatically removed from the dataset    
    df = detect_leakage(df, target_column=target_column , drop=True)    
    
    if target_column not in df.columns:
        logger.error(f"Target '{target_column}' missing from DataFrame. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target '{target_column}' not found in DataFrame.")

    logger.info(f"Feature set finalized with {df.shape[1]} columns after leakage control.")

    ## Optional outlier filtering (for regression targets only) ===
    if FILTER_STANDARD_INCIDENTS and target_column in ["incident_duration_min"]:
        threshold = df[target_column].quantile(0.95)
        original_len = len(df)
        df = df[df[target_column] < threshold]
        logger.info(f"\t Filtered extreme values for {target_column} < 95th percentile ({threshold:.2f}). Rows: {original_len} -> {len(df)}")

    feature_path = os.path.join(os.getcwd(), f"Services/{data_dir_output}/df_features_{strategy}_{target_column}.csv")

    return (df, feature_path)

def run_prepare_data_pipeline(strategy: str | None = None,
                              targets_input: list | str | None = None,
                              data_dir_input: str | None = None,
                              data_dir_output: str | None = None) -> None:
    """
        Run the prepare data (feature engineering) loop using the specified strategy.
        If arguments are None / empty, fall back to environment variables:

            DATAPREP_DEFAULT_STRATEGY
            DATAPREP_DEFAULT_INPUT_DIR
            DATAPREP_DEFAULT_OUTPUT_DIR
            DATAPREP_DEFAULT_TARGETS (comma-separated)
            DATAPREP_ENABLE_EXTRA_FEATURES (0/1 or true/false)

        Args:
            strategy (str|None)
            targets_input (list|str|None)
            data_dir_input (str|None)
            data_dir_output (str|None)
    """
    # Resolve env-based defaults
    strategy = strategy or os.getenv("DATAPREP_DEFAULT_STRATEGY", "incident_analysis")
    data_dir_input = data_dir_input or os.getenv("DATAPREP_DEFAULT_INPUT_DIR", "Services/DataCollection/data/raw")
    data_dir_output = data_dir_output or os.getenv("DATAPREP_DEFAULT_OUTPUT_DIR", "Services/DataPreparation/data/processed")

    os.makedirs(data_dir_output, exist_ok=True)

    # Targets: if list empty / blank string -> env
    if (not targets_input) or (isinstance(targets_input, str) and targets_input.strip() == ""):
        env_targets = os.getenv("DATAPREP_DEFAULT_TARGETS", "")
        targets_input = [t.strip() for t in env_targets.split(",") if t.strip()] or None

    # Normalize relative paths (keep absolute as-is)
    if not os.path.isabs(data_dir_input):
        data_dir_input = os.path.join(os.getcwd(), data_dir_input)
    if not os.path.isabs(data_dir_output):
        data_dir_output = os.path.join(os.getcwd(), data_dir_output)

    data_dir_output = data_dir_output.replace("/workspace/DataPreparation","/DataPreparation")
    os.makedirs(data_dir_output, exist_ok=True)

    logger.info(f"[DataPrep Config] strategy={strategy} input={data_dir_input} output={data_dir_output} targets={targets_input}")

    # Optional override for extra features
    extra_flag = os.getenv("DATAPREP_ENABLE_EXTRA_FEATURES")
    if extra_flag:
        enabled = str(extra_flag).lower() in ("1", "true", "yes", "on")
        if enabled != ENABLE_EXTRA_FEATURES:
            logger.info(f"Overriding ENABLE_EXTRA_FEATURES -> {enabled}")

    try:
        df = load_and_merge_files(strategy, data_dir_input)

        default_targets = [t for t, meta in TARGET_METADATA.items() if strategy in meta["strategies"]]
        selected_targets = targets_input if targets_input else default_targets
        targets = [t for t in selected_targets if strategy in TARGET_METADATA.get(t, {}).get("strategies", [])]

        if not targets:
            logger.warning(f"No valid targets for strategy '{strategy}'. Skipping.")
            return

        for target_name in targets:
            logger.info(f"\t TARGET : {target_name}")
            try:
                df_clean, feature_path = create_features(df, data_dir_output, strategy, target_name)
                df_clean.to_csv(feature_path, index=False)
                logger.info(f"\t Saved feature-engineered DataFrame to {feature_path}")
            except Exception as e:
                logger.warning(f"Error during prepare data for target '{target_name}': {e}")

    except Exception as e:
        logger.warning(f"Error while feature engineering strategy '{strategy}': {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run prepare_data pipeline")
    parser.add_argument("--strategy", "-s",  default="incident_analysis", help="Strategy")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional list of targets")
    parser.add_argument("--input", default="Services/DataCollection/data/live", help="Input data dir")
    parser.add_argument("--output", default="Services/DataPreparation/data/processed", help="Output data dir")
    args = parser.parse_args()

    args = parser.parse_args()
    
    ## Set absolute to relative path
    data_dir_output = args.output
    if not os.path.isabs(data_dir_output):
        data_dir_output = os.path.join("/workspace", data_dir_output)

    data_dir_output = data_dir_output.replace("/workspace/Services/","")
    os.makedirs(data_dir_output, exist_ok= True)
    logger.info("=== Running prepare data pipeline (standalone mode) ===")
    logger.info(f"=== data_dir_input ==={args.input}")
    logger.info(f"=== data_dir_output ==={data_dir_output}")
    
    run_prepare_data_pipeline(
        strategy=args.strategy,
        targets_input=args.targets,
        data_dir_input=args.input,
        data_dir_output=data_dir_output
    )
