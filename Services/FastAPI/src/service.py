'''
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service exposing prediction endpoints for incident_duration_min with JWT authentication."
'''

import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from pydantic import BaseModel
import csv
import numpy as np
import os
import json
import sys
from pathlib import Path

## Imports for "microservice" et "legacy" structures respectively
try:
 
    ## Authentication and authorization
    from Services.FastAPI.src.auth.jwt_auth import create_access_token
    from Services.FastAPI.src.auth.dependencies import RoleChecker
    from Services.FastAPI.src.auth.roles import ROLE_ADMIN, ROLE_USER

    ## Import dynamic schema
    from Services.FastAPI.src.dynamic_schema import DynamicFeatures, load_feature_order

    ## Logging utilities
    from Services.FastAPI.src.logging_utils import get_logger, log_execution_time_and_path

    ## Project constants
    from Services.FastAPI.src.constants import (
        MODEL_PATH,
        ACCESS_TOKEN_EXPIRE_MINUTES,
        FAKE_USERS_DB
    )

except:

    ## Authentication and authorization
    from src.auth.jwt_auth import create_access_token
    from src.auth.dependencies import RoleChecker
    from src.auth.roles import ROLE_ADMIN, ROLE_USER

    ## Import dynamic schema
    from src.dynamic_schema import DynamicFeatures, load_feature_order

    ## Logging utilities
    from src.logging_utils import get_logger, log_execution_time_and_path

    ## Project constants
    from src.constants import (
        MODEL_PATH,
        ACCESS_TOKEN_EXPIRE_MINUTES,
        FAKE_USERS_DB
    )

## ============================
## Setup logger
## ============================
logger = get_logger(__name__)

## ============================
## Global variables
## ============================
model = None         # Holds the trained ML model
FEATURE_ORDER = []   # Stores expected feature order
FEATURE_FILE = Path("resources/feature_importances.json")  # Fallback file for features

## MODEL_PATH is now safe and works everywhere
MODEL_PATH = MODEL_PATH.replace("Services\\", "")
MODEL_PATH = MODEL_PATH.replace("Services/", "")

## ============================
## FastAPI app initialization
## ============================
app = FastAPI(
    title="Incident Duration Prediction API",
    description="API to predict incident_duration_min with JWT protection.",
    version="1.0.0"
)

## ============================
## Authentication routes
## ============================
@app.post("/login", summary="Authenticate and return access token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
        Authenticate a user and return a JWT token

        Args:
            form_data (OAuth2PasswordRequestForm): Submitted username and password

        Returns:
            dict: JWT access token and token type

        Raises:
            HTTPException(401): Invalid username or password
    """
    
    ## Validate against in-memory users db (for demo/tests)
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        logger.warning("Invalid login credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    ## Create JWT with role embedded
    token = create_access_token(
        data={"sub": form_data.username, "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/test-token", summary="Validate token access")
async def test_token(current_user=Depends(RoleChecker([ROLE_ADMIN, ROLE_USER]))):
    """
        Validate JWT token and role authorization

        Args:
            current_user: Authenticated user injected by RoleChecker

        Returns:
            dict: Confirmation message with username and role

        Raises:
            HTTPException(401|403): Invalid token or unauthorized role
    """
    
    ## If RoleChecker passes, the token/role is valid
    return {
        "msg": "Token is valid",
        "username": current_user.username,
        "role": current_user.role
    }

@app.get("/healthcheck", summary="Healthcheck endpoint")
async def healthcheck():
    """
        Public endpoint to confirm API availability

        Returns:
            dict: Service status and current UTC timestamp
    """
    
    ## Keep it simple and dependency-free for liveness checks
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

## ============================
## Feature engineering (conditional)
## ============================
def maybe_add_engineered_columns(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
        Add engineered features **only if** they are required by the model (present in feature_order)

        This keeps compatibility with models trained either:
          - on raw/base features only, or
          - on raw + engineered features

        Args:
            df (pd.DataFrame): DataFrame built from request payload (raw features)
            feature_order (list[str]): Final ordered set of features expected by model

        Returns:
            pd.DataFrame: DataFrame with engineered features added if requested
    """
    
    ## Safe access helpers
    def has(cols: list[str]) -> bool:
        return all(c in df.columns for c in cols)

    ## Compute engineered features only when requested by the model
    if "log_length" in feature_order and "length" in df.columns:
        df["log_length"] = np.log1p(df["length"])

    if "length_x_jam" in feature_order and has(["length", "jam_factor"]):
        df["length_x_jam"] = df["length"] * df["jam_factor"]

    if "temp_x_slowdown" in feature_order and has(["temp", "avg_speed", "free_flow_speed"]):
        ## Avoid division by zero
        denom = (df["free_flow_speed"].replace(0, np.nan))
        df["temp_x_slowdown"] = df["temp"] * (df["avg_speed"] / denom)
        df["temp_x_slowdown"] = df["temp_x_slowdown"].replace([np.inf, -np.inf], np.nan)

    if "hour_x_jam" in feature_order and has(["hour", "jam_factor"]):
        df["hour_x_jam"] = df["hour"] * df["jam_factor"]

    if "hour_x_delay" in feature_order and has(["hour", "delay_before_start"]):
        df["hour_x_delay"] = df["hour"] * df["delay_before_start"]

    if "weekday_x_length" in feature_order and has(["weekday", "length"]):
        df["weekday_x_length"] = df["weekday"] * df["length"]

    if "delay_per_km" in feature_order and "delay_before_start" in df.columns and "length" in df.columns:
        df["delay_per_km"] = df["delay_before_start"] / (df["length"] + 1e-2)

    ## Clean numerical issues
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df

def build_full_feature_vector(input_data: BaseModel, feature_order: list[str]) -> pd.DataFrame:
    """
        Construct a feature vector that strictly matches the model's expected schema

            - Start with payload fields (dynamic model ensures presence)
            - Add engineered columns conditionally if requested by the model
            - Reorder columns exactly as `feature_order`

        Args:
            input_data (BaseModel): Dynamic Pydantic model instance mirroring model features
            feature_order (list[str]): Ordered features expected by the model

        Returns:
            pd.DataFrame: Single-row feature matrix ready for prediction

        Raises:
            HTTPException(500): If required columns are missing and cannot be created
    """
    
    ## Pydantic v2: prefer model_dump; fallback to dict()
    payload = input_data.model_dump() if hasattr(input_data, "model_dump") else input_data.dict()
    df = pd.DataFrame([payload])

    ## Add engineered columns only if the model expects them
    df = maybe_add_engineered_columns(df, feature_order)

    ## Validate presence of all required columns
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        ## Give a clear actionable error for debugging
        logger.error(f"Missing required features for prediction: {missing}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing required features for prediction: {missing}"
        )

    ## Enforce exact column order
    df = df[feature_order]
    
    return df

## ============================
## Prediction endpoint
## ============================
@app.post("/predict", summary="Predict incident_duration_min")
def predict(
    input_data: DynamicFeatures,
    current_user=Depends(RoleChecker([ROLE_ADMIN, ROLE_USER]))
):
    """
        Predict the incident duration in minutes

        This endpoint:
            - Loads (or reuses) the trained model from disk
            - Loads the expected feature order from the model or a JSON fallback
            - Builds a feature vector from the request payload, adding engineered features
            only if the model expects them
            - Returns the prediction

        Args:
            input_data (DynamicFeatures): Input payload matching the model's current feature set
            current_user: Authenticated user with role (admin/user)

        Returns:
            dict: {"predicted_incident_duration_min": <float>}

        Raises:
            HTTPException(500): If preprocessing or prediction fails
            HTTPException(503): If the model is not available
    """
    
    logger.info(f"Received prediction request from user: {current_user.username}")

    try:
        ## Ensure model is available
        if not os.path.exists(MODEL_PATH):
            ## No model on disk -> cannot predict (tests may skip this case)
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise HTTPException(
                status_code=503,
                detail="Model not available. Train the model before prediction."
            )

        ## Load model (idempotent for tests; could be cached in prod)
        logger.info(f"Loading model from: {MODEL_PATH}")

        try:
            mdl = joblib.load(MODEL_PATH)
            logger.info("Model successfully loaded from %s", os.path.abspath(MODEL_PATH))
        except (EOFError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
            logger.error("The model file may be missing, empty, or corrupted. Please retrain it.")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Unexpected error while loading model from {MODEL_PATH}: {e}")
            sys.exit(1)
        
        ## Determine feature order (from model or JSON fallback)
        feature_order = []
        if hasattr(mdl, "feature_names_in_"):
            feature_order = list(mdl.feature_names_in_)
            ## Persist to JSON for future runs without the model
            try:
                FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(FEATURE_FILE, "w", encoding="utf-8") as f:
                    json.dump(feature_order, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not persist features to JSON: {e}")
        else:
            ## Fallback to JSON if model lacks attribute (rare)
            feature_order = load_feature_order()

        if not feature_order:
            logger.error("Model feature order missing and JSON fallback unavailable.")
            raise HTTPException(status_code=500, detail="Model feature order missing")

        ## Build feature matrix strictly matching expected schema
        features_df = build_full_feature_vector(input_data, feature_order)
        logger.debug(f"Features DataFrame before prediction:\n{features_df}")

        ## Predict and format
        pred = mdl.predict(features_df)[0]
        logger.info(f"Prediction completed: {pred:.2f} minutes")

        return {"predicted_incident_duration_min": round(float(pred), 2)}

    except HTTPException:
        ## Re-raise FastAPI HTTP errors untouched
        raise
    except Exception as e:
        ## For any other runtime error, return a stable 500 for tests/clients
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/batch-predict", summary="Predict incident_duration_min for multiple samples")
def batch_predict(
    input_list: list[DynamicFeatures],
    current_user=Depends(RoleChecker([ROLE_ADMIN, ROLE_USER]))
):
    """
        Predict multiple incident durations in batch mode

        This endpoint processes a list of prediction requests, each following the DynamicFeatures schema, 
        and returns all predictions in a single response

        Args:
            input_list (list[DynamicFeatures]): List of payloads containing
                feature values for each sample
            current_user: Authenticated user (admin or user role)

        Returns:
            dict: JSON object containing a list of predictions
                Example:
                {
                    "predictions": [
                        {"sample_id": 1, "predicted_incident_duration_min": 3.72},
                        {"sample_id": 2, "predicted_incident_duration_min": 5.48}
                    ]
                }

        Raises:
            HTTPException(400): If input list is empty
            HTTPException(500): If preprocessing or prediction fails
            HTTPException(503): If the model is not available
    """

    ## Log the username for traceability
    logger.info(f"Received batch prediction request from user: {current_user.username}")

    ## Check if input is empty to avoid silent failure
    if not input_list:
        logger.warning("Empty input list for batch prediction.")
        raise HTTPException(status_code=400, detail="Input list cannot be empty.")

    ## Check that the model file exists before prediction
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model before prediction."
        )

    try:
        ## Load the trained model
        mdl = joblib.load(MODEL_PATH)

        ## Get the feature order (from model or JSON fallback)
        feature_order = list(mdl.feature_names_in_) if hasattr(mdl, "feature_names_in_") else load_feature_order()

        ## Validate that the feature order is not empty
        if not feature_order:
            logger.error("Model feature order missing and JSON fallback unavailable.")
            raise HTTPException(status_code=500, detail="Model feature order missing.")

        ## Convert each request object to a DataFrame row
        df_list = [build_full_feature_vector(item, feature_order) for item in input_list]

        ## Concatenate all rows into one batch DataFrame
        features_df = pd.concat(df_list, ignore_index=True)

        ## Debug print: display first few rows of the DataFrame
        logger.debug(f"Batch features DataFrame:\n{features_df.head()}")

        ## Perform predictions
        preds = mdl.predict(features_df)

        ## Build structured JSON results
        results = [
            {"sample_id": i + 1, "predicted_incident_duration_min": round(float(p), 2)}
            for i, p in enumerate(preds)
        ]

        ## Log the successful completion of the batch
        logger.info(f"Batch prediction completed successfully for {len(results)} samples.")

        ## Return predictions to client
        return {"predictions": results}

    except HTTPException:
        ## Let HTTPExceptions propagate (already well formatted)
        raise
    except Exception as e:
        ## Capture unexpected exceptions and log them
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

## ============================
## Metrics endpoint
## ============================
@app.get("/metrics", summary="Get latest model metrics")
async def get_metrics(current_user=Depends(RoleChecker([ROLE_ADMIN]))):
    """
        Return latest model evaluation metrics

        Args:
            current_user: Authenticated admin user

        Returns:
            dict: Last row of metrics.csv file

        Raises:
            HTTPException(500): If reading metrics fails
    """
    
    metrics_path = "metrics/metrics.csv"
    
    try:
        with open(metrics_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ValueError("metrics.csv is empty")
            return rows[-1]
    except Exception as e:
        logger.error(f"Failed to read metrics.csv: {e}")
        raise HTTPException(status_code=500, detail="Could not read model metrics.")

## ============================
## Run pipeline function
## ============================
def run_fastapi_service_pipeline(
    reload: bool = False,
    model_dir: str = os.path.join(".", "model"),
    model_filename: str = "model_incident_analysis_incident_duration_min.joblib"
) -> None:
    """
        Load model, prepare feature order, and launch FastAPI app

        Args:
            reload (bool): If True, runs uvicorn in reload mode for development
            model_dir (str): Directory containing the trained model. Defaults to "./model"
            model_filename (str): Model filename to load. Defaults to the incident_analysis model
    """

    global model, FEATURE_ORDER

    ## Build full model path dynamically
    model_path = os.path.join(model_dir, model_filename)
    logger.info(f"Loading model from {os.path.abspath(model_path)}")

    ## Load the model file
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info("Model successfully loaded from %s", os.path.abspath(model_path))
    except (EOFError, OSError, FileNotFoundError) as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.error("The model file may be missing, empty, or corrupted. Please retrain it.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error while loading model from {model_path}: {e}")
        sys.exit(1)

    ## Load features from model or JSON fallback
    try:
        if hasattr(model, "feature_names_in_"):
            FEATURE_ORDER = list(model.feature_names_in_)
            FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(FEATURE_FILE, "w", encoding="utf-8") as f:
                json.dump(FEATURE_ORDER, f, indent=2, ensure_ascii=False)
            logger.info(f"Loaded features from model: {FEATURE_ORDER}")
        else:
            FEATURE_ORDER = load_feature_order()
            logger.info(f"Loaded features from JSON fallback: {FEATURE_ORDER}")

        if not FEATURE_ORDER:
            logger.error("No features available to serve predictions.")
            raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to initialize feature order: {e}")
        raise SystemExit(1)

    ## Start the FastAPI server
    if reload:
        ## Exclude the problematic 'logs' directory to avoid OSError on Windows
        exclude_dirs = ["logs", "venv", "test_win"]
        uvicorn.run(
            "src.service:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_excludes=exclude_dirs,  ## prevent watching log folders
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

## ============================
## Local execution for debugging
## ============================
if __name__ == "__main__":
    """
        Allow direct execution with autoreload for development.
    """
    exclude_dirs = ["logs", "venv", "test_win"]

    uvicorn.run(
        "src.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=exclude_dirs,  ## ignore logs to avoid WinError
    )