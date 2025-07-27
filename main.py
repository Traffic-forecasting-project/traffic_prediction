'''
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com", "ingmatvillaa@gmail.com", "elqounss.karim@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for live data collection"
'''

import argparse
import logging
import sys
import os
import io
from dotenv import load_dotenv
from src.data.live_data_collector import run_live_data_pipeline
from src.core.logging_utils import get_logger

## Ensure UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

## Load environment variables
load_dotenv()

## Setup logging
logger = get_logger(__name__)

## Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

## Ensure live directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("data/live", exist_ok=True)

## Valid parameters
VALID_STRATEGIES = ["traffic_analysis", "incident_analysis"]
VALID_ARRONDISSEMENTS = list(range(1, 21))

def parse_arguments() -> argparse.Namespace:
    """
        Parse command-line arguments for arrondissement and strategy (optional)
    """
    
    parser = argparse.ArgumentParser(description="Traffic Data Collector CLI")
    parser.add_argument("-a", "--arrondissement", type=int, help="Paris arrondissement (1-20)")
    parser.add_argument("-s", "--strategy", type=str, help="Strategy: 'traffic_analysis' or 'incident_analysis'")
    
    return parser.parse_args()

def main():
    """
        Main entrypoint to control flow depending on user selection
        Defaults to arrondissement=17 and strategy='incident_analysis' for option 1
    """
    
    args = parse_arguments()

    ## Display main menu
    print("=== TRAFFIC LIVE DATA MENU ===")
    print("1. Run live data collection (default values)")
    print("q. Quit")
    print("==============================")
    choice = input("Select an option: ").strip().lower()

    if choice == "q":
        logger.info("User chose to quit. Exiting.")
        sys.exit(0)

    elif choice == "1":
        arrondissement = args.arrondissement if args.arrondissement else 17
        strategy = args.strategy if args.strategy else "incident_analysis"

        ## Check validity of inputs
        if arrondissement not in VALID_ARRONDISSEMENTS:
            logger.error(f"Invalid arrondissement: {arrondissement}. Must be between 1 and 20.")
            sys.exit(1)
        if strategy not in VALID_STRATEGIES:
            logger.error(f"Invalid strategy: '{strategy}'. Must be one of: {VALID_STRATEGIES}")
            sys.exit(1)

        logger.info(f"Running live collection with arrondissement={arrondissement}, strategy='{strategy}'")
        run_live_data_pipeline(arrondissement=arrondissement, strategy=strategy)

    else:
        logger.warning("Invalid menu option. Please choose '1' or 'q'.")
        sys.exit(1)


if __name__ == "__main__":
    main()