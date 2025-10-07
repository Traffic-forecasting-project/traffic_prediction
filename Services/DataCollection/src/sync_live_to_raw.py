''' 
__author__ = "Mateo Villa Arias"
__contributors__ = ""
__copyright__ = None
__version__ = "1.0.2"
__email__ = "ingmatvillaa@gmail.com"
__status__ = "Dev"
__desc__ = "Data synchronization script from live to raw folder."
'''

import os
import glob
import pandas as pd

## ============================================================
##  Import constants and logger dynamically depending on context
## ============================================================
try:
    from Services.FastAPI.src.constants import TIMESTAMP_COLUMN
    from Services.FastAPI.src.logging_utils import get_logger
except ImportError:
    from src.core.constants import TIMESTAMP_COLUMN
    from src.core.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
##  Synchronize live CSV data into the raw dataset folder
## ============================================================
def update_raw_from_live(
    delete_live: bool = False,
    live_dir: str = os.path.join("data", "live"),
    raw_dir: str = os.path.join("data", "raw"),
) -> None:
    """
        Synchronize all live data files into the raw dataset folder

        - Reads CSV files from the live data directory
        - Merges them with existing raw data (if present)
        - Removes duplicate timestamps and updates the raw CSV files

        Args:
            delete_live (bool): Whether to delete live files after synchronization
            live_dir (str): Directory containing live CSV files (default: ./data/live)
            raw_dir (str): Directory to store or update raw CSV files (default: ./data/raw)

        Returns:
            None
    """
    
    ## Build pattern for live CSV files
    pattern = os.path.join(live_dir, "live_data_incident_analysis.*.csv")
    live_files = glob.glob(pattern)

    if not live_files:
        logger.info(f"No live files found matching pattern: {pattern}")
        return

    ## Process each live data file
    for live_path in live_files:
        fname = os.path.basename(live_path)
        raw_path = os.path.join(raw_dir, fname)
        logger.info(f"Processing file: {fname}")

        ## Try reading the live file
        try:
            live_df = pd.read_csv(live_path, parse_dates=[TIMESTAMP_COLUMN])
        except Exception as e:
            logger.warning(f"Failed to read live file {fname}: {e}")
            continue

        live_count = len(live_df)
        logger.info(f"Loaded {live_count} rows from live file.")

        ## Merge with existing raw data if available
        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path, parse_dates=[TIMESTAMP_COLUMN])
            except Exception as e:
                logger.warning(f"Failed to read raw file {fname}: {e}")
                continue

            if list(live_df.columns) != list(raw_df.columns):
                logger.error(f"Column mismatch in {fname}. Skipping file.")
                continue

            combined = pd.concat([raw_df, live_df], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
            after = len(combined)
            new_rows = after - len(raw_df)
            dropped = before - after

            logger.info(f"Raw file had {len(raw_df)} rows.")
            logger.info(f"Added {new_rows} new rows. Dropped {dropped} duplicates.")
            logger.info(f"Updated total: {after} rows.")

        ## Otherwise, create a new raw file
        else:
            combined = live_df.dropna(how="all")
            combined = combined.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
            logger.info(f"No raw file found. Created new one with {len(combined)} rows.")

        ## Clean up and save
        combined = combined.dropna(how="all")
        combined = combined.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
        combined.to_csv(raw_path, index=False)
        logger.info(f"Saved raw file: {raw_path}")

        ## Optionally delete the live file
        if delete_live:
            try:
                os.remove(live_path)
                logger.info(f"Deleted live file: {live_path}")
            except Exception as e:
                logger.warning(f"Failed to delete live file {live_path}: {e}")

## ============================================================
##  CLI Entry Point
## ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update raw data folder from live data files.")
    parser.add_argument(
        "--live-dir",
        type=str,
        default=os.path.join( "data", "live"),
        help="Directory containing live CSV files (default: Services/data/live)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Directory to write/update raw CSV files (default: Services/data/raw)",
    )
    parser.add_argument(
        "--delete-live",
        action="store_true",
        help="Delete live files after processing.",
    )

    args = parser.parse_args()

    update_raw_from_live(
        delete_live=args.delete_live,
        live_dir=args.live_dir,
        raw_dir=args.raw_dir,
    )