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

## Import constants and logger based on project structure
try:
    from Services.DataPreparation.src.prepare_data import create_features, TARGET_METADATA
    from Services.FastAPI.src.logging_utils import get_logger, log_execution_time_and_path
    from Services.utils.utils import (
        load_and_merge_files,
        create_target
    )
    from Services.FastAPI.src.constants import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FILTER_STANDARD_INCIDENTS,
        USE_TOP_FEATURES_ONLY,
        TOP_FEATURES_FILE,
        TOP_N_FEATURES,
        MLFLOW_ENABLE_REMOTE,
        MLFLOW_LOCAL_URI,
        DAGSHUB_REPO_NAME,
        DAGSHUB_REPO_OWNER,
    )
except:
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
        DAGSHUB_REPO_NAME,
        DAGSHUB_REPO_OWNER,
    )

logger = get_logger("train_model")

@log_execution_time_and_path
def train_model(
    df: pd.DataFrame,
    strategy: str,
    target_name: str,
    regenerate_features: bool = False,
    data_dir_output: str = os.path.join("data", "processed"),
    models_dir: str = os.path.join("model"),
    top_features_file: str = os.path.join("model", "feature_importances.json"),
) -> dict:
    """
        Train a model (classification or regression) for the given target variable
        
        Args:
            df (pd.DataFrame): Input dataset.
            strategy (str): Strategy name (e.g., "incident_analysis")
            target_name (str): Name of the target column
            regenerate_features (bool): Whether to regenerate features before training
            data_dir_output (str): Directory to save processed data
            models_dir (str): Directory to save trained models
            top_features_file (str): Path to save top features JSON
        
        Returns:
            dict: Dictionary of metrics and paths related to the trained model
    """

    ## === Step 0: MLflow / DagsHub init ===
    if MLFLOW_ENABLE_REMOTE:
        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
        except Exception as e:
            logger.warning(f"[Dagshub] Repo already exists or cannot be created: {e}. Continuing with existing repo.")

    ## === Step 1: Optional feature regeneration ===
    if regenerate_features:
        result = create_features(df, data_dir_output, strategy, target_name)
        (df, feature_path) = (result[0], result[1])
        df.to_csv(feature_path, index=False)
        logger.info(f"\t Saved feature-engineered DataFrame to {feature_path}")

    ## === Step 2: Optional outlier filtering (for regression targets only) ===
    if FILTER_STANDARD_INCIDENTS and target_name in ["incident_duration_min"]:
        threshold = df[target_name].quantile(0.95)
        original_len = len(df)
        df = df[df[target_name] < threshold]
        logger.info(
            f"\t Filtered extreme values for {target_name} < 95th percentile ({threshold:.2f}). "
            f"Rows: {original_len} -> {len(df)}"
        )

    ## === Step 4: Create input features (X) and target (y) ===
    y = create_target(df, target_name)
    X = df.drop(columns=[target_name], errors="ignore").select_dtypes(include=[np.number])

    ## === Step 5: Remove constant or NaN-only columns ===
    logger.info(f"\t X shape before cleaning: {X.shape}")
    logger.info(f"\t Columns before cleaning: {X.columns.tolist()}")
    X = X.loc[:, X.columns[X.notna().any()]]
    feature_names_before_imputation = X.columns.tolist()

    ## === Step 6: Imputation ===
    imputer = SimpleImputer(strategy="mean")
    X_imputed_array = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_array, columns=feature_names_before_imputation)
    logger.info(f"\t X_imputed shape: {X_imputed.shape}")

    ## === Step 7: Optional feature pruning ===
    if USE_TOP_FEATURES_ONLY and os.path.exists(top_features_file):
        with open(top_features_file, "r") as f:
            top_features_dict = json.load(f)
        top_feats = top_features_dict.get(f"{strategy}_{target_name}", [])
        if top_feats:
            logger.info(f"\t Using top {len(top_feats)} features from importances for {target_name}")
            X_imputed = X_imputed[top_feats]

    ## === Step 8: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    ## === Step 9: Model definition ===
    if target_name == "congestion_label":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        if target_name in ["incident_duration_min"]:
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    ## === Step 10: Training and prediction ===
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ## === Step 11: Reverse log-transform if needed ===
    if target_name in ["incident_duration_min"]:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

    ## === Step 12: Save top features ===
    importances = model.feature_importances_
    feature_names = list(X_imputed.columns)
    top_indices = np.argsort(importances)[-TOP_N_FEATURES:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    logger.info(f"\t Top {TOP_N_FEATURES} features for {target_name}: {top_features}")

    existing = {}
    if os.path.exists(top_features_file):
        with open(top_features_file, "r") as f:
            existing = json.load(f)
    existing[f"{strategy}_{target_name}"] = top_features
    os.makedirs(os.path.dirname(top_features_file), exist_ok=True)
    with open(top_features_file, "w") as f:
        json.dump(existing, f, indent=2)

    ## === Step 13: Evaluation ===
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
        ## rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5 ## bug even if we use 1.7.2 of scikit-learn
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"\t\t[{strategy}] RMSE: {rmse:.3f}")
        logger.info(f"\t\t[{strategy}] MAE: {mae:.3f}")
        logger.info(f"\t\t[{strategy}] Median AE: {medae:.3f}")
        logger.info(f"\t\t[{strategy}] R²: {r2:.3f}")
        result.update({"metric": "rmse", "value": rmse, "mae": mae, "r2": r2, "medae": medae})

    ## === Step 14: Save model ===
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{strategy}_{target_name}.joblib")
    joblib.dump(model, model_path)
    result["model_path"] = model_path
    logger.info(f"Model saved to {model_path}")

    ## === Step 15: MLflow logging ===
    experiment_name = f"{strategy}_{target_name}"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=result["model_type"]):
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("target", target_name)
        mlflow.log_param("model_type", result["model_type"])
        mlflow.log_param("top_features", USE_TOP_FEATURES_ONLY)
        mlflow.log_param("filtered_outliers", FILTER_STANDARD_INCIDENTS)

        for key in ["value", "mae", "r2", "medae"]:
            if key in result:
                mlflow.log_metric(key, result[key])

        mlflow.sklearn.log_model(model, artifact_path="model")

        if regenerate_features:
            mlflow.log_artifact(feature_path)
        if os.path.exists(top_features_file):
            mlflow.log_artifact(top_features_file)

    return result

def run_train_model_pipeline(
    strategy: str,
    regenerate_features: bool = False,
    data_dir_input: str = "data",
    data_dir_stats: str = "metrics",
    models_dir: str = os.path.join("model"),
    top_features_file: str = os.path.join("model", "feature_importances.json"),
) -> None:
    """
        Run the training pipeline according to the strategy

        Args:
            strategy (str): 'traffic_analysis' or 'incident_analysis'
            regenerate_features (bool): Whether to regenerate features
            data_dir_input (str): Base directory for input data
            data_dir_stats (str): Directory for saving training metrics
            models_dir (str): Directory to save trained models
            top_features_file (str): Path to save top features JSON
    """

    results = []
    logger.info(f"STRATEGY : {strategy} \n")

    try:
        ## Ensure input subfolder exists
        if regenerate_features:
            data_dir_input = os.path.join(data_dir_input, "raw")
        else:
            data_dir_input = os.path.join(data_dir_input, "processed")

        if not os.path.exists(data_dir_input):
            logger.warning(f"Input directory not found: {data_dir_input}")
            return

        ## Determine valid targets
        selected_targets = [t for t, meta in TARGET_METADATA.items() if strategy in meta["strategies"]]
        targets = [t for t in selected_targets if strategy in TARGET_METADATA.get(t, {}).get("strategies", [])]
        if not targets:
            logger.warning(f"No valid targets for strategy '{strategy}'. Skipping.")
            return

        ## Train models for each target
        for target in targets:
            features_target_path = os.path.join(data_dir_input, f"df_features_{strategy}_{target}.csv")
            if not os.path.exists(features_target_path):
                logger.warning(f"Missing file: {features_target_path}")
                continue

            df = pd.read_csv(features_target_path)
            logger.info(f"\t TARGET : {target} \n")

            try:
                result = train_model(
                    df=df,
                    strategy=strategy,
                    target_name=target,
                    regenerate_features=regenerate_features,
                    data_dir_output=data_dir_input,
                    models_dir=models_dir,
                    top_features_file=top_features_file,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error during training for target '{target}': {e}")

    except Exception as e:
        logger.warning(f"Error while processing strategy '{strategy}': {e}")

    ## Save results
    if results:
        os.makedirs(data_dir_stats, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(data_dir_stats, "metrics.csv"), index=False)
        logger.info(f"Model stats saved to {data_dir_stats}")
    else:
        logger.warning("No models were successfully trained.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model pipeline")
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        required=True,
        help="Strategy to run: 'traffic_analysis' or 'incident_analysis'"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate features before training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data",
        help="Path to input data directory"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="metrics",
        help="Path to stats output directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join("model"),
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--top-features",
        type=str,
        default=os.path.join("model", "feature_importances.json"),
        help="Path to save top features JSON"
    )

    args = parser.parse_args()

    run_train_model_pipeline(
        strategy=args.strategy,
        regenerate_features=args.regenerate,
        data_dir_input=args.input,
        data_dir_stats=args.stats,
        models_dir=args.models_dir,
        top_features_file=args.top_features,
    )