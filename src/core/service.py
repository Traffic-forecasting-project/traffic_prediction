'''
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service exposing prediction endpoints for incident_duration_min with JWT authentication."
'''

## ============================
## Imports
## ============================
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from pydantic import BaseModel
import csv
import numpy as np

from src.auth.jwt_auth import create_access_token, verify_token
from src.core.logging_utils import get_logger, log_execution_time_and_path

from src.core.constants import (
    MODEL_PATH,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    FAKE_USERS_DB
)

logger = get_logger(__name__)

## Model directory and filename
model = None
FEATURE_ORDER = []

## Global FastAPI app (required for route decorators to work)
app = FastAPI(
    title="Incident Duration Prediction API",
    description="API to predict incident_duration_min with JWT protection.",
    version="1.0.0"
)
@app.post("/login", summary="Authenticate and return access token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
        Authenticate user and return JWT token

        Args:
            form_data (OAuth2PasswordRequestForm): User credentials

        Returns:
            dict: JWT access token and token type
        
        Raises:
            HTTPException: 401 if authentication fails
    """
    
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        logger.warning("Invalid login credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": token, "token_type": "bearer"}

@app.get("/test-token", summary="Validate token access")
async def test_token(username: str = Depends(verify_token)):
    """Test route to check token validity"""
    return {"msg": "Token is valid", "username": username}

@app.get("/healthcheck", summary="Healthcheck endpoint")
async def healthcheck():
    """Simple healthcheck endpoint to verify service status"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

## ============================
## Input Schema
## ============================
class IncidentFeatures(BaseModel):
    """Schema representing full input features for prediction"""
    delay_before_start: float
    length: float
    avg_speed: float
    free_flow_speed: float
    jam_factor: float
    temp: float
    wind: float
    rain: float
    hour: int
    weekday: int

## ============================
## Feature engineering (same logic as prepare_data)
## ============================
def build_full_feature_vector(input_data: IncidentFeatures) -> pd.DataFrame:
    """
        Construct a DataFrame with the exact feature order expected by the trained model.

        Args:
            input_data (IncidentFeatures): Input data from the API payload

        Returns:
            pd.DataFrame: Feature vector matching the model training schema
    """

    ## Convert input data to DataFrame
    df = pd.DataFrame([input_data.dict()])

    ## Derived features
    df["log_length"] = np.log1p(df["length"])
    df["length_x_jam"] = df["length"] * df.get("jam_factor", 1)  # fallback if jam_factor not provided
    df["temp_x_slowdown"] = df["temp"] * (df.get("avg_speed", 1) / df.get("free_flow_speed", 1))
    df["hour_x_jam"] = df["hour"] * df.get("jam_factor", 1)
    df["hour_x_delay"] = df["hour"] * df["delay_before_start"]
    df["weekday_x_length"] = df["weekday"] * df["length"]
    df["delay_per_km"] = df["delay_before_start"] / (df["length"] + 0.01)

    ## Fill NaNs or Infs if any
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    ## Enforce correct feature order
    df = df[FEATURE_ORDER]

    return df
    
@app.post("/predict", summary="Predict incident_duration_min")
def predict(input_data: IncidentFeatures, username: str = Depends(verify_token)):
    """
        Predict the incident duration using a trained model

        Args:
            input_data (IncidentFeatures): Input features from the user
            username (str): Authenticated user from JWT token

        Returns:
            dict: Dictionary containing the predicted incident duration in minutes

        Raises:
            HTTPException: If prediction fails due to preprocessing or model errors
    """

    logger.info(f"Received prediction request from user: {username}")

    try:
        ## Step 1: Build the feature vector from raw input
        features_df = build_full_feature_vector(input_data)

        ## Step 1b: Reorder columns to match training order
        ## features_df = features_df[feature_order]
        features_df = features_df[FEATURE_ORDER]

        ## Step 2: Load the trained model
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        ## Step 3: Make prediction
        prediction = model.predict(features_df)[0]
        logger.info(f"Prediction completed: {prediction:.2f} minutes")

        return {"predicted_incident_duration_min": round(float(prediction), 2)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/metrics", summary="Get latest model metrics")
async def get_metrics():
    """
        Get latest model evaluation results from CSV file

        Returns:
            dict: Last row of metrics.csv
    """
    
    metrics_path = "models/metrics.csv"
    
    try:
        with open(metrics_path, mode="r") as f:
            reader = csv.DictReader(f)
            last_row = list(reader)[-1]
            return last_row
    except Exception as e:
        logger.error(f"Failed to read metrics.csv: {e}")
        raise HTTPException(status_code=500, detail="Could not read model metrics.")


def run_fastapi_service_pipeline(reload: bool = False) -> None:
    """
        Load model, extract feature order, and start FastAPI app with uvicorn
    """
    
    global model, FEATURE_ORDER

    ## Load the trained model from file
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    ## Extract feature names used during model training
    FEATURE_ORDER = []
    try:
        FEATURE_ORDER = list(model.feature_names_in_)
        logger.info(f"Loaded features: {FEATURE_ORDER}")
    except AttributeError:
        logger.error("Model does not contain 'feature_names_in_'")
        exit(1)

    ## Start FastAPI server with live reload for local development
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    if reload:
        uvicorn.run("src.core.service:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

## ============================
## Local execution for debugging
## ============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)    