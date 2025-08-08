'''
__author__ = "Mateo Villa Arias"
__contributors__ = 
__copyright__ = None
__version__ = "1.0.0"
__email__ = ""
__status__ = "Dev"
__desc__ = "Data synchronization script from live to raw folder"
'''


import os
import glob
import pandas as pd

from src.core.constants import LIVE_DATA_DIR, RAW_DATA_DIR, TIMESTAMP_COLUMN
from src.core.logging_utils import get_logger

logger = get_logger(__name__)

def update_raw_from_live(delete_live=False):
    pattern = os.path.join(LIVE_DATA_DIR, 'live_data_incident_analysis.*.csv')
    live_files = glob.glob(pattern)

    if not live_files:
        logger.info(f"No live files found matching pattern: {pattern}")
        return

    for live_path in live_files:
        fname = os.path.basename(live_path)
        raw_path = os.path.join(RAW_DATA_DIR, fname)

        logger.info(f"Processing file: {fname}")

        try:
            live_df = pd.read_csv(live_path, parse_dates=[TIMESTAMP_COLUMN])
        except Exception as e:
            logger.warning(f"Failed to read live file {fname}: {e}")
            continue

        live_count = len(live_df)
        logger.info(f"Loaded {live_count} rows from live file.")

        if os.path.exists(raw_path):
            try:
                raw_df = pd.read_csv(raw_path, parse_dates=[TIMESTAMP_COLUMN])
            except Exception as e:
                logger.warning(f"Failed to read raw file {fname}: {e}")
                continue

            if list(live_df.columns) != list(raw_df.columns):
                logger.error(f"Column mismatch in {fname}. Skipping this file.")
                continue

            combined = pd.concat([raw_df, live_df], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep='last')
            after = len(combined)
            new_rows = after - len(raw_df)
            dropped = before - after

            logger.info(f"Raw file had {len(raw_df)} rows.")
            logger.info(f"Added {new_rows} new rows. Dropped {dropped} duplicate timestamps.")
            logger.info(f"Updated total: {after} rows.")
        else:
            combined = live_df.dropna(how='all')
            combined = combined.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
            logger.info(f"No raw file found. Creating new file with {len(combined)} rows.")
            new_rows = len(combined)

        combined = combined.dropna(how='all')
        combined = combined.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
        combined.to_csv(raw_path, index=False)
        logger.info(f"Saved raw file: {raw_path}")

        if delete_live:
            try:
                os.remove(live_path)
                logger.info(f"Deleted live file: {live_path}")
            except Exception as e:
                logger.warning(f"Failed to delete live file {live_path}: {e}")
