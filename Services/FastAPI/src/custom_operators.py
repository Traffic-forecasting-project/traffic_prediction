'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom Airflow Operators for live data monitoring and conditional execution"
'''

import sys, os
sys.path.insert(0, os.path.abspath("/workspace"))

## ==================================================================
## Standard library imports
## ==================================================================
import time
from pathlib import Path
from typing import Optional

## ==================================================================
## Airflow imports
## ==================================================================
from airflow.exceptions import AirflowSkipException
from airflow.providers.docker.operators.docker import DockerOperator

## ==================================================================
## Project imports
## ==================================================================
from Services.FastAPI.src.logging_utils import get_logger

## ==================================================================
## Logger setup
## ==================================================================
logger = get_logger(__name__)

## ==================================================================
## Custom Operator: LiveDataDockerOperator
## ==================================================================
class LiveDataDockerOperator(DockerOperator):
    """
        ## LiveDataDockerOperator
            - Extends DockerOperator with monitoring logic for live data collection
            - Allows stopping container execution when:
                1. A maximum duration is reached (time-based criterion)
                2. A monitored file has increased by a minimum number of lines (size-based criterion)
                3. Depending on 'mode', one, both, or strictly both criteria are enforced
    """

    def __init__(
        self,
        monitor_file: Optional[str] = None,
        max_duration: Optional[int] = None,
        min_line_increase: Optional[int] = None,
        mode: str = "time",  # options: "time", "lines", "both", "strict"
        *args,
        **kwargs,
    ) -> None:
        """
            ## Initialization

            Args:
                monitor_file (Optional[str]):
                    Path to the file being monitored (CSV or log)
                    If None → only time-based monitoring is applied

                max_duration (Optional[int]):
                    Maximum allowed duration in seconds
                    If reached → container execution is stopped

                min_line_increase (Optional[int]):
                    Minimum number of new lines required in the monitored file
                    If reached → container execution is stopped

                mode (str):
                    Defines which criteria to use:
                    - "time": only max_duration is enforced
                    - "lines": only min_line_increase is enforced
                    - "both": stop when either criterion is met (logical OR)
                    - "strict": stop only when both criteria are met (logical AND)

                *args, **kwargs:
                    Forwarded to the base `DockerOperator`
        """
        
        super().__init__(*args, **kwargs)
        self.monitor_file = Path(monitor_file) if monitor_file else None
        self.max_duration = max_duration
        self.min_line_increase = min_line_increase
        self.mode = mode

    def execute(self, context):
        """
        ## Execution logic
            - Runs the Docker container
            - After execution, verifies criteria based on mode
        """
        
        logger.info("Starting LiveDataDockerOperator with mode=%s", self.mode)
        start_time = time.time()
        initial_lines = self._count_lines() if self.monitor_file else 0

        ## Run the container process
        result = super().execute(context)

        ## Measure elapsed time & file growth
        elapsed_time = time.time() - start_time
        current_lines = self._count_lines() if self.monitor_file else 0
        line_diff = current_lines - initial_lines

        logger.debug(
            "Elapsed: %.2fs | Lines diff: %d | Mode: %s",
            elapsed_time,
            line_diff,
            self.mode,
        )

        ## Criterion 1: Time exceeded
        time_ok = self.max_duration and elapsed_time >= self.max_duration

        ## Criterion 2: Enough lines collected
        lines_ok = self.min_line_increase and line_diff >= self.min_line_increase

        ## Mode handling
        if self.mode == "time" and time_ok:
            logger.info("Max duration reached (%.2fs).", elapsed_time)
        elif self.mode == "lines" and lines_ok:
            logger.info("File grew by %d lines.", line_diff)
        elif self.mode == "both" and (time_ok or lines_ok):
            logger.info("Condition met in 'both' mode: time=%s, lines=%s", time_ok, lines_ok)
        elif self.mode == "strict" and (time_ok and lines_ok):
            logger.info("Both conditions met in 'strict' mode: time=%s, lines=%s", time_ok, lines_ok)

        return result

    def _count_lines(self) -> int:
        """
            ## Count lines in monitored file
                - Returns 0 if file does not exist or cannot be read
        """
        
        if not self.monitor_file or not self.monitor_file.exists():
            return 0
        try:
            with self.monitor_file.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error("Error reading file %s: %s", self.monitor_file, str(e))
            return 0

## ==================================================================
## Custom Operator: ConditionalDockerOperator
## ==================================================================
class ConditionalDockerOperator(DockerOperator):
    """
        ## ConditionalDockerOperator
            - Runs a DockerOperator only if certain file/folder conditions are met
            - Example use cases:
                * Skip model evaluation if no `.joblib` model exists
                * Skip training if no live data file is present or files are empty
    """

    def __init__(
        self,
        required_path: Optional[str] = None,
        must_exist: bool = True,
        must_have_lines: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
            ## Initialization

            Args:
                required_path (Optional[str]):
                    Path to a file or folder that must meet conditions before execution

                must_exist (bool):
                    If True, the path must exist
                    If False, the existence check is ignored

                must_have_lines (bool):
                    If True and path is a file, it must contain at least 1 line
                    Useful to ensure data files are not empty

                *args, **kwargs:
                    Forwarded to the base `DockerOperator`
        """
        
        super().__init__(*args, **kwargs)
        self.required_path = Path(required_path) if required_path else None
        self.must_exist = must_exist
        self.must_have_lines = must_have_lines

    def execute(self, context):
        """
            ## Execution logic
                - Verifies the required conditions before running the Docker container
                - If conditions fail → raises AirflowSkipException
                - Otherwise → behaves like a normal DockerOperator
        """
        
        if self.required_path:
            logger.info("Checking condition for path: %s", self.required_path)

            ## Check existence of the path
            if self.must_exist and not self.required_path.exists():
                msg = f"Skipping task: path {self.required_path} does not exist"
                logger.warning(msg)
                raise AirflowSkipException(msg)

            ## Check if file has lines (only if it's a file)
            if self.must_have_lines and self.required_path.is_file():
                try:
                    with self.required_path.open("r", encoding="utf-8") as f:
                        if sum(1 for _ in f) == 0:
                            msg = f"Skipping task: file {self.required_path} has no lines"
                            logger.warning(msg)
                            raise AirflowSkipException(msg)
                except Exception as e:
                    msg = f"Error reading {self.required_path}: {e}"
                    logger.error(msg)
                    raise AirflowSkipException(msg)

        ## If all checks passed → run container
        return super().execute(context)
