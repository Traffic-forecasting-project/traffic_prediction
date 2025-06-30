''' 
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Logging utility with execution time and error trace tracking."
'''

import logging
import os
import time
import traceback
from functools import wraps
from datetime import datetime

## Create log directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
        Configure and return a logger with both file and console handlers

        Parameters:
            name (str): The name of the logger, typically the module name

        Returns:
            logging.Logger: Configured logger instance
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        log_file = os.path.join(LOG_DIR, f"{name}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def log_execution_time_and_path(func):
    """
        Decorator to log the execution time and errors of a function

        Logs the time taken to execute the decorated function and
        captures any exceptions raised during its execution

        Parameters:
            func (callable): The function to decorate

        Returns:
            callable: The wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"[{func.__name__}] completed in {duration:.2f} seconds.")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"[{func.__name__}] failed after {duration:.2f} seconds.")
            logger.error(f"Exception: {e}")
            logger.error(traceback.format_exc())
            raise

    return wrapper