'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Logging configuration utilities for console and file logging"
'''

import logging
import os

def setup_logging(log_filename: str = "live_data_collector") -> None:
    """
        Set up the root logger with both console and file output

        Args:
            log_filename (str): Name of the log file (without extension). Default is 'live_data_collector'

        This function ensures:
            - All logs from `logging.info`, `logging.warning`, etc. are captured
            - Messages are shown in the console
            - Messages are also saved in logs/{log_filename}.log
    """

    ## Create logs/ directory if not present
    os.makedirs("logs", exist_ok=True)

    ## Format for log messages
    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ## Root logger (used by logging.info(...))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ## Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:

        ## File handler
        file_handler = logging.FileHandler(f"logs/{log_filename}.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        ## Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
        Create or retrieve a named logger with the same configuration as the root logger

        Args:
            name (str): Name of the logger to retrieve

        Returns:
            logging.Logger: Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ## Add handlers only if they are not already set
    if not logger.handlers:
        setup_logging()  # In case root logger not set
        for handler in logging.getLogger().handlers:
            logger.addHandler(handler)

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