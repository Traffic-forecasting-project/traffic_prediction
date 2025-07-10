''' 
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for FastAPI service endpoints with authentication and validation handling"
'''

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from service import app
import json

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
        Fixture to obtain a valid JWT token using login route

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

def test_healthcheck(client_instance: TestClient) -> None:
    """
        Test the healthcheck route is working

        Args:
            client_instance (TestClient): FastAPI test client
    """
    
    response = client_instance.get("/healthcheck")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

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

def test_predict_success(client_instance: TestClient, auth_token: str) -> None:
    """
        Test /predict route with valid payload and token

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
    """
    
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
        "weekday": 2
    }
      
    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    assert "predicted_incident_duration_min" in response.json()

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
    
    payload = {
        "length": 0.6  ## Missing required fields
    }
    
    response = client_instance.post(
        "/predict",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 422
    
from unittest.mock import patch, MagicMock

def test_predict_internal_server_error(client_instance: TestClient, auth_token: str) -> None:
    """
        Test /predict route handles internal model errors

        Args:
            client_instance (TestClient): FastAPI test client
            auth_token (str): JWT token
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

    with patch("service.joblib.load") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Simulated internal error")
        mock_load.return_value = mock_model

        response = client_instance.post(
            "/predict",
            json=payload,
            headers={"Authorization": f"Bearer {auth_token}"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}

def test_metrics(client_instance: TestClient, auth_token: str) -> None:
    """
        Test the /metrics route returns model performance metrics.

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