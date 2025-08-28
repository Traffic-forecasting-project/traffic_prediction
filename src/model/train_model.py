''' 
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.6.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Training script for predictive modeling with optional outlier filtering and top feature selection."
'''

import pandas as pd
import numpy as np
import joblib
import os
import json
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    #root_mean_squared_error,
    mean_squared_error,    
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

from src.data.prepare_data import create_features, TARGET_METADATA
from src.core.logging_utils import get_logger, log_execution_time_and_path

from src.utils.utils import (
    load_and_merge_files,
    create_target
)

from src.core.constants import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FILTER_STANDARD_INCIDENTS,
    USE_TOP_FEATURES_ONLY,
    TOP_FEATURES_FILE,
    TOP_N_FEATURES,
    MLFLOW_ENABLE_REMOTE,
    MLFLOW_LOCAL_URI,
    MLFLOW_REMOTE_URL,

)

logger = get_logger("train_model")

@log_execution_time_and_path
def train_model(df: pd.DataFrame, strategy: str, target_name: str, regenerate_features = False, data_dir_output: str = "data/processed") -> dict:
    """
        Train a model (classification or regression) for the given target variable.
        
        Args:
            df (pd.DataFrame): Input dataset.
            strategy (str): Strategy name (e.g., "incident_analysis")
            target_name (str): Name of the target column
        
        Returns:
            dict: Dictionary of metrics and paths related to the trained model
    """
    if MLFLOW_ENABLE_REMOTE:
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, 
                     repo_name=DAGSHUB_REPO_NAME,
                        mlflow=True)
    else:
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
    ## === Step 1: Feature engineering regeneration (optional) ===
    if regenerate_features :
        result = create_features(df, data_dir_output, strategy, target_name)
        (df, feature_path) = (result[0], result[1])
        
        df.to_csv(feature_path, index=False)
        logger.info(f"\t Saved feature-engineered DataFrame to {feature_path}")
    
    ## === Step 2: Optional outlier filtering (for regression targets only) ===
    if FILTER_STANDARD_INCIDENTS and target_name in ["incident_duration_min"]:
        threshold = df[target_name].quantile(0.95)
        original_len = len(df)
        df = df[df[target_name] < threshold]
        logger.info(f"\t Filtered extreme values for {target_name} < 95th percentile ({threshold:.2f}). Rows: {original_len} -> {len(df)}")


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
        #rmse = root_mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"\t\t[{strategy}] RMSE: {rmse:.3f}")
        logger.info(f"\t\t[{strategy}] MAE: {mae:.3f}")
        logger.info(f"\t\t[{strategy}] Median AE: {medae:.3f}")
        logger.info(f"\t\t[{strategy}] R²: {r2:.3f}")
        result.update({"metric": "rmse", "value": rmse, "mae": mae, "r2": r2, "medae": medae})

    ## === Step 14: Save trained model ===
    model_path = f"{MODELS_DIR}/model_{strategy}_{target_name}.joblib"
    joblib.dump(model, model_path)
    result["model_path"] = model_path
    logger.info(f"Model saved to {model_path}")

    ## === Step 15: Log MLflow tracking ===
    experiment_name = f"{strategy}_{target_name}"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=result["model_type"]):

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
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        ## Log trained feature file and importance JSON as artifacts
        ## TODO : move this part elsewhere
        if regenerate_features:
            mlflow.log_artifact(feature_path)
        if os.path.exists(TOP_FEATURES_FILE):
            mlflow.log_artifact(TOP_FEATURES_FILE)

    return result

def run_train_model_pipeline(strategy: str, 
                             regenerate_features = False,
                             data_dir_input: str = "data",
                               data_dir_stats: str = "metrics") -> None:
    """
        Run the training pipeline according to the strategy:

        Args:
            strategy (str): 'traffic_analysis' or 'incident_analysis'
            data_dir_input (str): path to live data collected, by default "data/live"
            data_dir_stats (str): path to produced stats, by default "metrics"            
    """
    
    results = []
    logger.info(f"STRATEGY : {strategy} \n")

    try:
        if regenerate_features:
            data_dir_input = os.path.join(data_dir_input,"raw")
            ## Load and merge the CSV files for the given strategy
            df = load_and_merge_files(strategy, data_dir_input)
        else:
            data_dir_input = os.path.join(data_dir_input,"processed")

        ## Determine the default targets allowed for this strategy, either default or user provided
        selected_targets = [t for t, meta in TARGET_METADATA.items() if strategy in meta["strategies"]]

        ## Filter valid targets for the current strategy, and return error if no valid targets
        targets = [t for t in selected_targets if strategy in TARGET_METADATA.get(t, {}).get("strategies", [])]
        if not targets:
            logger.warning(f"No valid targets for strategy '{strategy}'. Skipping.")

        ## Loop through each target and train the model
        for target in targets:
            features_target_path = os.path.join(data_dir_input,f"df_features_{strategy}_{target}.csv")
            df = pd.read_csv(features_target_path)
            logger.info(f"\t TARGET : {target} \n")
            try:
                result = train_model(df, strategy, target,regenerate_features = regenerate_features)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error during training for target '{target}': {e}")

    except Exception as e:
        logger.warning(f"Error while processing strategy '{strategy}': {e}")

    ## If training was successful, save stats to CSV
    if results:
        pd.DataFrame(results).to_csv(f"{data_dir_stats}/metrics.csv", index=False)
        logger.info(f"Model stats saved to {data_dir_stats}")
    else:
        logger.warning("No models were successfully trained.")        