'''
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com", "ingmatvillaa@gmail.com", "elqounss.karim@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for live data collection"
'''

import pytest
import argparse
import logging
import sys
import os
import io
from dotenv import load_dotenv
from src.data.push_data import run_push_data


from src.data.live_data_collector import run_live_data_pipeline
from src.data.sync_live_to_raw import update_raw_from_live

from src.data.prepare_data import run_prepare_data_pipeline
from src.eda.eda_analysis import run_eda_pipeline
from src.model.train_model import run_train_model_pipeline
from src.model.register_model import run_register_model
from src.core.service import run_fastapi_service_pipeline

from src.core.constants import MODEL_PATH
from src.core.logging_utils import get_logger

## Ensure UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

## Load environment variables
load_dotenv()

## Setup logging
logger = get_logger(__name__)

## Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/live", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("eda", exist_ok=True)
os.makedirs("model", exist_ok=True)

## Valid parameters
VALID_STRATEGIES = ["traffic_analysis", "incident_analysis"]
VALID_ARRONDISSEMENTS = list(range(1, 21))

def parse_arguments() -> argparse.Namespace:
    """
        Parse optional arguments (used only for validation)
    """
    
    parser = argparse.ArgumentParser(description="Traffic Pipeline CLI")
    parser.add_argument("-a", "--arrondissement", type=int, default = 17,help="Paris arrondissement (1–20)")
    parser.add_argument("-s", "--strategy", type=str, default="incident_analysis", choices=["incident_analysis", "traffic_analysis", "all"], help="Data/train collection strategy")
    return parser.parse_args()

def main():
    """
        Main menu-based CLI logic
    """
    
    args = parse_arguments()

    if args.arrondissement is not None and args.arrondissement not in VALID_ARRONDISSEMENTS:
        logger.error(f"Invalid arrondissement '{args.arrondissement}'. Must be between 1 and 20.")
        sys.exit(1)

    if args.strategy is not None and args.strategy not in VALID_STRATEGIES:
        logger.error(f"Invalid strategy '{args.strategy}'. Choose from {VALID_STRATEGIES}.")
        sys.exit(1)

    print("=== TRAFFIC LIVE DATA MENU ===")
    print("1. Run live data collection")
    print("2. Sync live files into raw directory")
    print("3. Run feature engineering on collected data")
    print("4. Run EDA analysis on data")
    print("5. Run train model on data")
    print("6. Register model for staging")
    print("7. Launch FastAPI service with uvicorn")
    print("8. Run tests avec pytest")
    print("9. Push data to DVC remote")
    print("q. Quit")
    print("==============================")
    choice = input("Select an option: ").strip().lower()

    strategy = args.strategy if args.strategy in VALID_STRATEGIES else "incident_analysis"

    if choice == "q":
        logger.info("User chose to quit. Exiting.")
        sys.exit(0)

    elif choice == "1":
        logger.info(f"Running live collection with arrondissement={args.arrondissement}, strategy='{strategy}'")
        run_live_data_pipeline(arrondissement=args.arrondissement, strategy=strategy)

    elif choice == "2":
        delete_input = input("Delete live files after sync? (y/n): ").strip().lower()
        delete_live = delete_input == "y"
        logger.info(f"Synchronizing live files into raw directory (delete_live={delete_live})")
        update_raw_from_live(delete_live=delete_live)

    elif choice == "3":
        logger.info(f"Running feature engineering pipeline with strategy='{strategy}'")
        run_prepare_data_pipeline(strategy)

    elif choice == "4":
        logger.info(f"Running EDA analysis pipeline with strategy='{strategy}'")
        run_eda_pipeline(strategy)

    elif choice == "5":
        logger.info(f"Running train model pipeline with strategy='{strategy}'")
        run_train_model_pipeline(strategy)

    elif choice == "6":
        run_register_model()
    elif choice == "7":
        logger.info(f"Launching FastAPI uvicorn server for routes with trained model path ==> '{MODEL_PATH}'")
        run_fastapi_service_pipeline(reload=True)

    elif choice == "8":
        logger.info("Running pytest on tests/")
        retcode = pytest.main(["tests"])
        sys.exit(retcode)

    elif choice == "9":
        remote = input("DVC remote name, leave blank for default: ").strip() or None
        msg = input("Git commit message, leave blank for default: ").strip() or None
        dtype = input("Select what to push [raw/processed/train/all] (default: raw): ").strip().lower() or "raw"
        logger.info(f"Pushing data to DVC remote (type={dtype})")
        run_push_data(dtype, remote=remote, message=msg)


    else:
        logger.warning(f"Invalid menu option '{choice}'. Please choose a valid option.")
        sys.exit(1)




if __name__ == "__main__":
    main()