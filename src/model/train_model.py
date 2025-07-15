''' 
__author__ = -
__copyright__ = None
__version__ = "1.6.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Training script for predictive modeling with optional outlier filtering and top feature selection."
'''

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import glob
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    classification_report,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE
from prepare_data import create_features, create_target, TARGET_METADATA
from logging_utils import get_logger, log_execution_time_and_path

## ========== Global Configurations ==========
FILTER_STANDARD_INCIDENTS = True   ## Filter out extreme outliers for regression targets
USE_TOP_FEATURES_ONLY = True      ## Restrict to top features if previous importances exist
EDA_ENABLED = True
TOP_FEATURES_FILE = "exports/feature_importances.json"
TOP_N_FEATURES = 15

logger = get_logger("train_model")

@log_execution_time_and_path
def load_and_merge_files(data_dir: str, strategy: str) -> pd.DataFrame:
    """
        Load all CSV files matching a given strategy pattern from a folder

        Args:
            data_dir (str): Path to data directory
            strategy (str): Name of the strategy (e.g., "incident_analysis")

        Returns:
            pd.DataFrame: Concatenated and deduplicated DataFrame
    """
    
    ## Build the glob pattern to match all CSV files related to the strategy
    pattern = os.path.join(data_dir, f"live_data_{strategy}.*.csv")
    
    ## Find all files matching the pattern and rase an error if no files found
    file_list = glob.glob(pattern)
    if not file_list:
        raise FileNotFoundError(f"No files found for strategy '{strategy}' in {data_dir}")
    
    ## Read all CSV files into separate DataFrames, concatenate and drop duplicate rows
    dfs = [pd.read_csv(f, low_memory=False) for f in file_list]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    
    logger.info(f"Loaded {len(df)} rows from {len(file_list)} file(s) for strategy '{strategy}'")
    
    return df

@log_execution_time_and_path
def train_model(df: pd.DataFrame, strategy: str, target_name: str) -> dict:
    """
        Train a model (classification or regression) for the given target variable.
        
        Args:
            df (pd.DataFrame): Input dataset.
            strategy (str): Strategy name (e.g., "incident_analysis")
            target_name (str): Name of the target column
        
        Returns:
            dict: Dictionary of metrics and paths related to the trained model
    """
    
    ## === Step 1: Feature engineering ===
    df = create_features(df,  strategy, target_name)

    ## === Step 2: Optional outlier filtering (for regression targets only) ===
    if FILTER_STANDARD_INCIDENTS and target_name in ["incident_duration_min"]:
        threshold = df[target_name].quantile(0.95)
        original_len = len(df)
        df = df[df[target_name] < threshold]
        logger.info(f"\t Filtered extreme values for {target_name} < 95th percentile ({threshold:.2f}). Rows: {original_len} -> {len(df)}")

    ## === Step 3: Save processed DataFrame ===
    os.makedirs("exports", exist_ok=True)
    feature_path = f"exports/df_features_{strategy}_{target_name}.csv"
    df.to_csv(feature_path, index=False)
    logger.info(f"\t Saved feature-engineered DataFrame to {feature_path}")

    ## === Step 4: Create input features (X) and target (y) ===
    y = create_target(df, target_name)
    X = df.drop(columns=[target_name], errors="ignore").select_dtypes(include=[np.number])

    ## === Step 5: Remove constant or NaN-only columns ===
    logger.info(f"\t X shape before cleaning: {X.shape}")
    logger.info(f"\t Columns before cleaning: {X.columns.tolist()}")
    X = X.loc[:, X.columns[X.notna().any()]]
    feature_names_before_imputation = X.columns.tolist()

    ## === Step 6: Imputation + optional pruning for regression targets ===
    imputer = SimpleImputer(strategy="mean")
    X_imputed_array = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_array, columns=feature_names_before_imputation)
    logger.info(f"\t X_imputed shape: {X_imputed.shape}")

    ## === Step 7: Optional reduction to known top features if enabled ===
    if USE_TOP_FEATURES_ONLY and os.path.exists(TOP_FEATURES_FILE):
        with open(TOP_FEATURES_FILE, "r") as f:
            top_features_dict = json.load(f)
        top_feats = top_features_dict.get(f"{strategy}_{target_name}", [])
        if top_feats:
            logger.info(f"\t Using top {len(top_feats)} features from importances for {target_name}")
            X_imputed = X_imputed[top_feats]

    ## === Step 8: Split into train/test ===
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    ## === Step 9: Model definition ===
    if target_name == "congestion_label":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        if target_name in ["incident_duration_min"]:
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    ## === Step 10: Training and prediction ===
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ## === Step 11: Reverse log transform if needed ===
    if target_name in ["incident_duration_min"]:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
        

    ## === Step 12: Save top features for reuse (always executed) ===
    importances = model.feature_importances_
    feature_names = list(X_imputed.columns)
    top_indices = np.argsort(importances)[-TOP_N_FEATURES:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    logger.info(f"\t Top {TOP_N_FEATURES} features for {target_name}: {top_features}")

    existing = {}
    if os.path.exists(TOP_FEATURES_FILE):
        with open(TOP_FEATURES_FILE, "r") as f:
            existing = json.load(f)
    existing[f"{strategy}_{target_name}"] = top_features
    with open(TOP_FEATURES_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    ## === Step 13: Evaluation and metrics ===
    result = {
        "strategy": strategy,
        "target": target_name,
        "model_type": "classifier" if target_name == "congestion_label" else "regressor",
    }

    if target_name == "congestion_label":
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        logger.info(f"\t\t[{strategy}] Accuracy: {acc:.3f}")
        logger.info(f"\t\t[{strategy}] Classification report:\n{report}")
        logger.info(f"\t\t[{strategy}] Confusion matrix:\n{matrix}")
        result.update({"metric": "accuracy", "value": acc})
    else:
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"\t\t[{strategy}] RMSE: {rmse:.3f}")
        logger.info(f"\t\t[{strategy}] MAE: {mae:.3f}")
        logger.info(f"\t\t[{strategy}] Median AE: {medae:.3f}")
        logger.info(f"\t\t[{strategy}] R²: {r2:.3f}")
        result.update({"metric": "rmse", "value": rmse, "mae": mae, "r2": r2, "medae": medae})

    ## === Step 14: Save trained model ===
    os.makedirs("models", exist_ok=True)
    model_path = f"models/model_{strategy}_{target_name}.joblib"
    joblib.dump(model, model_path)
    result["model_path"] = model_path
    logger.info(f"Model saved to {model_path}")

    ## === Step 15: Log MLflow tracking ===
    with mlflow.start_run(run_name=f"{strategy}_{target_name}"):

        ## Log main parameters
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("target", target_name)
        mlflow.log_param("model_type", result["model_type"])
        mlflow.log_param("top_features", USE_TOP_FEATURES_ONLY)
        mlflow.log_param("filtered_outliers", FILTER_STANDARD_INCIDENTS)

        ## Log metrics
        for key in ["value", "mae", "r2", "medae"]:
            if key in result:
                mlflow.log_metric(key, result[key])

        ## Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        ## Log trained feature file and importance JSON as artifacts
        mlflow.log_artifact(feature_path)
        if os.path.exists(TOP_FEATURES_FILE):
            mlflow.log_artifact(TOP_FEATURES_FILE)

    return result

@log_execution_time_and_path
def parse_args():
    """
        Parse CLI arguments for strategy, target, input folder, and output path

        Returns:
            argparse.Namespace: Parsed arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="incident_analysis",
                        choices=["incident_analysis", "traffic_analysis", "all"],
                        help="Training strategy to use")
    parser.add_argument("--targets", nargs="*", default=[],
                        help="Optional list of target variables")
    parser.add_argument("--data_dir", type=str, default="src/data/raw",
                        help="Directory containing CSV input data")
    parser.add_argument("--output_stats", type=str, default="metrics/model_stats.csv",
                        help="Path to output stats file")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    strategies = [args.strategy] if args.strategy != "all" else ["incident_analysis", "traffic_analysis"]
    results = []

    ## Loop through each selected strategy
    for strategy in strategies:

        logger.info(f"STRATEGY : {strategy} \n")

        try:
            ## Load and merge the CSV files for the given strategy
            df = load_and_merge_files(args.data_dir, strategy)

            ## Determine the default targets allowed for this strategy, either default or user provided
            default_targets = [t for t, meta in TARGET_METADATA.items() if strategy in meta["strategies"]]
            selected_targets = args.targets if args.targets else default_targets

            ## Filter valid targets for the current strategy, and return error if no valid targets
            targets = [t for t in selected_targets if strategy in TARGET_METADATA.get(t, {}).get("strategies", [])]
            if not targets:
                logger.warning(f"No valid targets for strategy '{strategy}'. Skipping.")
                continue

            ## Loop through each target and train the model
            for target in targets:

                logger.info(f"\t TARGET : {target} \n")
                try:
                    result = train_model(df, strategy, target)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error during training for target '{target}': {e}")


        except Exception as e:
            logger.warning(f"Error while processing strategy '{strategy}': {e}")

    ## If training was successful, save stats to CSV
    if results:
        pd.DataFrame(results).to_csv(args.output_stats, index=False)
        logger.info(f"Model stats saved to {args.output_stats}")
    else:
        logger.warning("No models were successfully trained.")

    if results:
        pd.DataFrame(results).to_csv("models/metrics.csv", index=False)
        
    ## Attempt to run EDA after training
    if EDA_ENABLED == True:
        try:
            import eda_all
        except Exception as e:
            logger.warning(f"EDA script execution failed: {e}")