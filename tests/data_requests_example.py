'''
__author__ = "Georges Nassopoulos"
__version__ = "1.0.6"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI client to test FastAPI JWT/RBAC endpoints with manual or auto mode."
'''

import os
import time
import json
import requests
from dotenv import load_dotenv

## ============================================================
## Load environment variables
## ============================================================
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "adminpass")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")

## ============================================================
## API CALL FUNCTIONS
## ============================================================
def login(username: str, password: str) -> str | None:
    """
        Authenticate user using credentials and obtain JWT token

        Args:
            username (str): Username for authentication
            password (str): Password for authentication

        Returns:
            str | None: JWT token if authentication succeeds, otherwise None
    """
    
    url = f"{BASE_URL}/login"
    data = {"username": username, "password": password}
    try:
        response = requests.post(url, data=data)
        print(f"\n=== /login ===")
        print(f"Status code: {response.status_code}")
        print(response.text)

        if response.status_code == 200:
            token = response.json().get("access_token")
            print("Login successful.")
            return token
        print("Login failed.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error during login: {e}")
        return None

def call_route(token: str, route: str, method: str = "get", payload: dict | None = None) -> None:
    """
        Send a request to a protected FastAPI route using JWT token

        Args:
            token (str): JWT access token for authorization
            route (str): API route (e.g., '/predict')
            method (str): HTTP method ('get' or 'post')
            payload (dict | None): Data payload for POST requests

        Returns:
            None
    """
    
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{BASE_URL}{route}"

    try:
        if method.lower() == "get":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, json=payload, headers=headers)

        print(f"\n=== {route} ===")
        print(f"Status code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error calling {route}: {e}")

def healthcheck() -> None:
    """
        Verify if the FastAPI service is running and reachable

        Args:
            None

        Returns:
            None
    """
    
    url = f"{BASE_URL}/healthcheck"
    try:
        response = requests.get(url)
        print("\n=== /healthcheck ===")
        print(f"Status code: {response.status_code}")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Healthcheck failed: {e}")

def test_token(token: str) -> None:
    """
        Verify JWT token validity and associated user role

        Args:
            token (str): JWT token

        Returns:
            None
    """
    
    call_route(token, "/test-token")

def predict_example(token: str) -> None:
    """
        Send a prediction request to the /predict endpoint

        Args:
            token (str): JWT token for authorization

        Returns:
            None
    """
    
    payload = {
        "lat": 48.8566,
        "lon": 2.3522,
        "incident_count": 2,
        "icon_category": 3,
        "length": 1.5,
        "tmc_tableNumber": 100,
        "tmc_tableVersion": 10,
        "avg_speed": 42.0,
        "free_flow_speed": 50.0,
        "jam_factor": 6.8,
        "temp": 18.5,
        "wind": 5.2,
        "rain": 0.0,
        "hour": 8,
        "weekday": 2
    }

    call_route(token, "/predict", "post", payload)

def batch_predict_example(token: str) -> None:
    """
        Send a batch prediction request to the /batch-predict endpoint

        Args:
            token (str): JWT token for authorization

        Returns:
            None
    """
    
    ## Example payload with multiple rows
    payload = [
        {
            "lat": 48.8566,
            "lon": 2.3522,
            "incident_count": 1,
            "icon_category": 3,
            "length": 1.2,
            "tmc_tableNumber": 100,
            "tmc_tableVersion": 10,
            "avg_speed": 45.0,
            "free_flow_speed": 50.0,
            "jam_factor": 5.6,
            "temp": 19.0,
            "wind": 4.0,
            "rain": 0.0,
            "hour": 9,
            "weekday": 3
        },
        {
            "lat": 48.8666,
            "lon": 2.3322,
            "incident_count": 3,
            "icon_category": 4,
            "length": 2.3,
            "tmc_tableNumber": 101,
            "tmc_tableVersion": 10,
            "avg_speed": 37.0,
            "free_flow_speed": 55.0,
            "jam_factor": 7.2,
            "temp": 17.5,
            "wind": 5.0,
            "rain": 0.0,
            "hour": 17,
            "weekday": 5
        }
    ]


    ## Call the batch prediction route
    call_route(token, "/batch-predict", "post", payload)

def get_metrics(token: str) -> None:
    """
        Retrieve internal system metrics (admin-only route)

        Args:
            token (str): JWT token for authorization

        Returns:
            None
    """
    call_route(token, "/metrics")

## ============================================================
## EXECUTION SEQUENCES
## ============================================================
def execute_all_tests() -> None:
    """
        Execute the complete API test sequence automatically
    """
    
    print("\nExecuting full automated test sequence...")
    time.sleep(1)

    ## Public route first
    healthcheck()
    token = login(ADMIN_USER, ADMIN_PASS)

    if token:
        test_token(token)
        predict_example(token)
        get_metrics(token)
        batch_predict_example(token)
    else:
        print("Login failed, skipping protected endpoints.")

def main() -> None:
    """
        Interactive CLI for manually or automatically testing API endpoints
    """
    
    token = None
    while True:
        print("\n==============================")
        print(" FastAPI RBAC Test Client")
        print("==============================")
        print("1. Login (.env credentials)")
        print("2. Healthcheck")
        print("3. Test token")
        print("4. Predict example")
        print("5. Batch predict example")
        print("6. Get metrics")
        print("7. Exit")
        print("8. Execute all automatically")
        print("==============================")

        choice = input("Select option: ").strip()

        if choice == "1":
            token = login(ADMIN_USER, ADMIN_PASS)
        elif choice == "2":
            healthcheck()
        elif choice == "3":
            if token:
                test_token(token)
            else:
                print("Please login first.")
        elif choice == "4":
            if token:
                predict_example(token)
            else:
                print("Please login first.")
        elif choice == "5":
            if token:
                batch_predict_example(token)
            else:
                print("Please login first.")
        elif choice == "6":
            if token:
                get_metrics(token)
            else:
                print("Please login first.")                
        elif choice == "7":
            print("Exiting client...")
            break
        elif choice == "8":
            execute_all_tests()
            break
        else:
            print("Invalid option.")

## ============================================================
## ENTRY POINT
## ============================================================
if __name__ == "__main__":
    main()