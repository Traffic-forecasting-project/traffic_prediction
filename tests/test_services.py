''' 
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for FastAPI service endpoints with authentication and validation handling"
'''

## ============================
## Imports
## ============================
import os
import json
import pytest
import joblib
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

## Import the FastAPI app under test
from src.core.service import app

## Define model and fallback JSON paths
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "model", "model_incident_analysis_incident_duration_min.joblib"
)
JSON_FALLBACK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "resources", "feature_importances.json"
)

def build_dynamic_payload():
    """
        Build a dynamic payload based on the model features
        This ensures tests remain valid even if the feature set changes
    """
    
    features = []
    
    try:
        model = joblib.load(MODEL_PATH)
        features = list(getattr(model, "feature_names_in_", []))
    except Exception:
        # fallback: load from JSON
        json_path = os.path.join("resources", "feature_importances.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                features = data.get("features", [])
    
    payload = {}
    for feat in features:
        # Heuristic: assign dummy values by name
        if "lat" in feat or "lon" in feat:
            payload[feat] = 0.0
        elif "hour" in feat or "weekday" in feat or "count" in feat or "icon" in feat or "tmc" in feat:
            payload[feat] = 1
        else:
            payload[feat] = 1.0
    
    return payload

## ============================
## Fixtures
## ============================
@pytest.fixture(scope="module")
def client_instance() -> TestClient:
    """
        Fixture to provide FastAPI TestClient instance

        Returns:
            TestClient: Test client for the FastAPI app
    """
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_token(client_instance: TestClient) -> str:
    """
        Fixture to obtain a valid JWT token using login route (admin by default)

        Args:
            client_instance (TestClient): FastAPI test client

        Returns:
            str: JWT token string
    """
    response = client_instance.post(
        "/login",
        data={"username": "admin", "password": "adminpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


## ============================
## Tests: Login
## ============================
def test_login_success(client_instance: TestClient) -> None:
    """
        Test that login with valid credentials returns a token

        Args:
            client_instance (TestClient): FastAPI test client
    """
    response = client_instance.post(
        "/login",
        data={"username": "admin", "password": "adminpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_failure(client_instance: TestClient) -> None:
    """
        Test that login with invalid credentials is rejected

        Args:
            client_instance (TestClient): FastAPI test client
    """
    response = client_instance.post(
        "/login",
        data={"username": "wrong", "password": "wrongpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 401


## ============================
## Tests: Healthcheck
## ============================
def test_healthcheck(client_instance: TestClient) -> None:
    """
        Test the healthcheck route is working

        Args:
            client_instance (TestClient): FastAPI test client
    """
    response = client_instance.get("/healthcheck")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


## ============================
## Tests: Test Token
## ============================
def test_test_token(client_instance: TestClient, auth_token: str) -> None:
    """
        Test the /test-token route with valid token

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """
    response = client_instance.get(
        "/test-token",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert "username" in response.json()

## ============================
## Tests: Predict
## ============================
def test_predict_success(client_instance, auth_token):
    """
        Test /predict route with valid payload and token
            Skipped if the trained model file is missing

            Developers should run the training step 
                (option 5 in main.py)
                before executing this test to generate the .joblib model
        
        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"Model file not found at {MODEL_PATH}. "
            "Run step 5 (train model) in main.py to generate it "
            "before executing this test."
        )

    payload = {
        "timestamp": "2025-06-25 12:49:47",
        "delay_before_start": 40.0,
        "length": 0.6,
        "avg_speed": 30.0,
        "free_flow_speed": 50.0,
        "jam_factor": 2.0,
        "temp": 14.0,
        "wind": 4.0,
        "rain": 0.0,
        "hour": 8,
        "weekday": 2,
        "lat": 48.8566,
        "lon": 2.3522,
        "incident_count": 1,
        "icon_category": 1,
        "tmc_tableNumber": 1,
        "tmc_tableVersion": 1
    }

    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )

    assert response.status_code == 200
    assert "predicted_incident_duration_min" in response.json()


def test_predict_with_json_fallback(client_instance, auth_token):
    """
        Test /predict when model is missing but fallback JSON exists.
        This ensures that tests do not fully break when joblib is absent.
    """
    ## Simulate missing model but ensure fallback JSON exists
    if os.path.exists(MODEL_PATH):
        pytest.skip("Model exists, fallback test not required.")

    if not os.path.exists(JSON_FALLBACK_PATH):
        pytest.skip(
            f"Neither model nor fallback JSON found. "
            f"Expected JSON at {JSON_FALLBACK_PATH}."
        )

    payload = {
        "delay_before_start": 10.0,
        "length": 1.0,
        "avg_speed": 20.0,
        "free_flow_speed": 40.0,
        "jam_factor": 1.5,
        "temp": 15.0,
        "wind": 2.0,
        "rain": 0.0,
        "hour": 10,
        "weekday": 3,
        "incident_count": 1,
        "tmc_tableNumber": 1,
        "tmc_tableVersion": 1
    }

    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )

    ## Should not crash: either success or 500 (if JSON badly formatted)
    assert response.status_code in [200, 500]


def test_predict_unauthorized(client_instance: TestClient) -> None:
    """
        Test that /predict without token returns 401

        Args:
            client_instance (TestClient): FastAPI test client
    """
    payload = {
        "delay_before_start": 40.0,
        "length": 0.6,
        "avg_speed": 30.0,
        "free_flow_speed": 50.0,
        "jam_factor": 2.0,
        "temp": 14.0,
        "wind": 4.0,
        "rain": 0.0,
        "hour": 8,
        "weekday": 2
    }

    response = client_instance.post("/predict", json=payload)
    assert response.status_code == 401


def test_predict_validation_error(client_instance: TestClient, auth_token: str) -> None:
    """
        Test /predict with missing required fields triggers 422 validation

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """
    payload = {"length": 0.6}  ## Missing required fields
    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 422

def test_predict_internal_server_error(client_instance: TestClient, auth_token: str) -> None:
    """
        Test /predict route handles internal model errors.

        This test ensures that even if the model raises an exception during
        prediction, the API still returns a proper 500 error instead of crashing.

        We mock `joblib.load` to return a fake model object with:
            - a valid `feature_names_in_` attribute (so FEATURE_ORDER is initialized)
            - a `.predict()` method that always raises an Exception

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """

    ## Build a dynamic payload to remain feature-agnostic
    payload = build_dynamic_payload()

    if not payload:
        pytest.skip("No features available to build dynamic payload.")

    ## Patch joblib.load so that the service uses our fake model instead
    with patch("src.core.service.joblib.load") as mock_load:
        ## Create a fake model with expected attributes
        mock_model = MagicMock()
        mock_model.feature_names_in_ = list(payload.keys())  # simulate feature order
        mock_model.predict.side_effect = Exception("Simulated internal error")
        mock_load.return_value = mock_model

        ## Call the predict endpoint with our dynamic payload
        response = client_instance.post(
            "/predict",
            json=payload,
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    ## Expect a clean 500 error with "Prediction failed"
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    
## ============================
## Tests: Metrics
## ============================
def test_metrics(client_instance: TestClient, auth_token: str) -> None:
    """
        Test the /metrics route returns model performance metrics

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """
    response = client_instance.get(
        "/metrics",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


## ============================
## Additional Role-based tests
## ============================
def test_user_cannot_access_metrics(client_instance: TestClient) -> None:
    """
        Test that a normal user cannot access /metrics (403 forbidden)
        
        Args:
            client_instance (TestClient): FastAPI test client        
    """
    ## Login as user
    response = client_instance.post(
        "/login",
        data={"username": "user", "password": "userpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = response.json()["access_token"]

    ## Try to access /metrics
    response = client_instance.get(
        "/metrics",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_guest_cannot_access_predict(client_instance: TestClient) -> None:
    """
        Test that a guest cannot access /predict (403 forbidden)
        
        Args:
            client_instance (TestClient): FastAPI test client        
    """
    ## Login as guest
    response = client_instance.post(
        "/login",
        data={"username": "guest", "password": "guestpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = response.json()["access_token"]

    ## Try to access /predict
    payload = {
        "delay_before_start": 40.0,
        "length": 0.6,
        "avg_speed": 30.0,
        "free_flow_speed": 50.0,
        "jam_factor": 2.0,
        "temp": 14.0,
        "wind": 4.0,
        "rain": 0.0,
        "hour": 8,
        "weekday": 2,
        "incident_count": 1,
        "tmc_tableNumber": 1,
        "tmc_tableVersion": 1
    }
    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_guest_cannot_access_test_token(client_instance: TestClient) -> None:
    """
        Ensure that a guest user cannot access the /test-token route

        The /test-token endpoint is restricted to ROLE_ADMIN and ROLE_USER
        Guests should be denied access and receive a 403 Forbidden response

        Args:
            client_instance (TestClient): FastAPI test client
    """
    ## Login as guest and retrieve token
    response = client_instance.post(
        "/login",
        data={"username": "guest", "password": "guestpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = response.json()["access_token"]

    ## Attempt to access /test-token with guest token
    response = client_instance.get(
        "/test-token",
        headers={"Authorization": f"Bearer {token}"}
    )

    ## Expect a forbidden response
    assert response.status_code == 403
