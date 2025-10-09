''' 
__author__ = "-"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG for the traffic prediction MLOps pipeline, with selective task skipping, sensors, and refined Docker volume mounts."
'''

## ==================================================================
## Standard library imports
## ==================================================================
import os
import platform
import subprocess
from datetime import datetime
from typing import Dict, Optional
from docker.types import Mount

## ==================================================================
## Airflow imports
## ==================================================================
from airflow import DAG
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

## ==================================================================
## Project imports
## ==================================================================
from Services.FastAPI.src.logging_utils import get_logger
from Services.FastAPI.src.custom_operators import (
    LiveDataDockerOperator,
    ConditionalDockerOperator,
)

## ==================================================================
## Logger setup
## ==================================================================
logger = get_logger(__name__)

## ==================================================================
## Environment detection (Windows vs Unix)
## ==================================================================
IS_WINDOWS = platform.system().lower().startswith("win")
MOUNT_TMP_DIR = not IS_WINDOWS  # Disable mount_tmp_dir on Windows
logger.info(f"Running on {'Windows' if IS_WINDOWS else 'Unix-like'} system. mount_tmp_dir={MOUNT_TMP_DIR}")

## ==================================================================
## Configuration variables
## ==================================================================
MAX_DURATION = int(os.getenv("MAX_DURATION", 100))
MIN_LINE_INCREASE = int(os.getenv("MIN_LINE_INCREASE", 1000))
MODE = os.getenv("LIVE_MONITOR_MODE", "time")
RUN_MODE = os.getenv("RUN_MODE", "manual")
LIVE_FILE_TIMEOUT = int(os.getenv("LIVE_FILE_TIMEOUT", 3600))
MODEL_FILE_TIMEOUT = int(os.getenv("MODEL_FILE_TIMEOUT", 1800))

## ==================================================================
## Helper for skip flags
## ==================================================================
def get_skip_flag(name: str) -> bool:
    """Retrieve a skip flag for a given task."""
    return Variable.get(name, os.getenv(name, "false")).lower() in ["true", "1", "yes"]

## ==================================================================
## Skip flags per stage
## ==================================================================
SKIP_COLLECT_LIVE_DATA = get_skip_flag("SKIP_COLLECT_LIVE_DATA")
SKIP_SYNC_DATA = get_skip_flag("SKIP_SYNC_DATA")
SKIP_PREPROCESS_DATA = get_skip_flag("SKIP_PREPROCESS_DATA")
SKIP_EDA_ANALYSIS = get_skip_flag("SKIP_EDA_ANALYSIS")
SKIP_TRAIN_MODEL = get_skip_flag("SKIP_TRAIN_MODEL")

logger.info(
    "Skip options: COLLECT=%s | SYNC=%s | PREPROCESS=%s | EDA=%s | TRAIN=%s",
    SKIP_COLLECT_LIVE_DATA, SKIP_SYNC_DATA, SKIP_PREPROCESS_DATA, SKIP_EDA_ANALYSIS, SKIP_TRAIN_MODEL
)

## ==================================================================
## Default DAG arguments
## ==================================================================
default_args: Dict[str, object] = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
    "retries": 0,
}

## ==================================================================
## DAG definition
## ==================================================================
dag = DAG(
    dag_id="main_pipeline_dag",
    description="Traffic prediction MLOps pipeline with selective skipping and volume mounts.",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "traffic", "pipeline"],
)

## ==================================================================
## Mount definitions (auto-detected from inside the container)
## ==================================================================
logger.info("***************************************************************")
logger.info("********************** ABSOLUTE PATH DEBUG ********************")
logger.info("***************************************************************")

try:
    ## Discover absolute project root dynamically
    PROJECT_ROOT = os.path.abspath(os.getcwd())
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
    LIVE_DATA_FILE = os.path.join(DATA_PATH, "live", "live_data_incident_analysis.1.csv")
    MODEL_FILE = os.path.join(MODEL_PATH, "model_incident_analysis_incident_duration_min.joblib")

    ## Print detected paths
    logger.info(f"*** PROJECT_ROOT DETECTED: {PROJECT_ROOT}")
    logger.info(f"*** DATA_PATH DETECTED: {DATA_PATH}")
    logger.info(f"*** MODEL_PATH DETECTED: {MODEL_PATH}")
    logger.info(f"*** LIVE_DATA_FILE DETECTED: {LIVE_DATA_FILE}")
    logger.info(f"*** MODEL_FILE DETECTED: {MODEL_FILE}")
    
    ## List key directories for debugging
    logger.info("*** RUNNING PWD CHECK ***")
    logger.info(subprocess.check_output(["pwd"], text=True))

    logger.info("*** LISTING PROJECT ROOT ***")
    logger.info(subprocess.check_output(["ls", "-l", PROJECT_ROOT], text=True))

    logger.info("*** LISTING DATA PATH ***")
    logger.info(subprocess.check_output(["ls", "-l", DATA_PATH], text=True))

    logger.info("*** LISTING MODEL PATH ***")
    logger.info(subprocess.check_output(["ls", "-l", MODEL_PATH], text=True))
except Exception as e:
    logger.warning(f"ABSOLUTE PATH DISCOVERY FAILED: {e}")

logger.info("***************************************************************")
logger.info("******************** END ABSOLUTE PATH DEBUG ******************")
logger.info("***************************************************************")

## ==================================================================
## FIX: Ensure required subdirectories exist inside /opt/airflow/data
## ==================================================================
try:
    for subfolder in ["live", "raw", "processed"]:
        folder_path = os.path.join("/opt/airflow/data", subfolder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created missing folder: {folder_path}")
        else:
            logger.info(f"Verified existing folder: {folder_path}")
except Exception as e:
    logger.warning(f"FOLDER CREATION FAILED: {e}")

## ==================================================================
## FIXED HOST MOUNTS (STATIC PATH)
## ==================================================================
try:
    #HOST_ROOT = "/mnt/c/Users/user/Desktop/mlops_tomtom_traffic/traffic_prediction"
    HOST_ROOT = os.getenv("HOST_ROOT", os.getcwd())
    DATA_MOUNT = Mount(
        source=f"{HOST_ROOT}/data",
        target="/opt/airflow/data",
        type="bind"
    )
    MODEL_MOUNT = Mount(
        source=f"{HOST_ROOT}/model",
        target="/opt/airflow/model",
        type="bind"
    )

    logger.info("***************************************************************")
    logger.info("********************** MOUNT CONFIRMATION *********************")
    logger.info("***************************************************************")
    logger.info(f"DATA_MOUNT SOURCE: {getattr(DATA_MOUNT, 'source', None)}")
    logger.info(f"MODEL_MOUNT SOURCE: {getattr(MODEL_MOUNT, 'source', None)}")
    logger.info(f"DATA_MOUNT TARGET: {getattr(DATA_MOUNT, 'target', None)}")
    logger.info(f"MODEL_MOUNT TARGET: {getattr(MODEL_MOUNT, 'target', None)}")
    logger.info("***************************************************************")
    logger.info("******************** END MOUNT CONFIRMATION *******************")
    logger.info("***************************************************************")

except Exception as e:
    logger.warning(f"MOUNT CONFIGURATION FAILED: {e}")

## ==================================================================
## Task 1: Live Data Collection
## ==================================================================
if not SKIP_COLLECT_LIVE_DATA:
    collect_live_data = LiveDataDockerOperator(
        task_id="collect_live_data",
        image="data_collector:latest",
        command=(
            "python /opt/airflow/Services/DataCollection/src/live_data_collector.py "
            f"--arrondissement 1 --strategy incident_analysis "
            f"--max-rows {MIN_LINE_INCREASE} --max-duration {MAX_DURATION} "
            f"--arrondissements-path /opt/airflow/data/arrondissements.csv"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[DATA_MOUNT, MODEL_MOUNT],
        monitor_file="/opt/airflow/data/live/live_data_incident_analysis.1.csv",
        max_duration=MAX_DURATION,
        min_line_increase=MIN_LINE_INCREASE,
        mode=MODE,
        mount_tmp_dir=False,
        dag=dag,
    )
else:
    collect_live_data = None
    logger.info("Skipping task: collect_live_data")

## ==================================================================
## Task 1b: Wait for live data file
## ==================================================================
if RUN_MODE == "continuous":
    wait_for_live_file = FileSensor(
        task_id="wait_for_live_file",
        fs_conn_id="fs_project",
        filepath="/opt/airflow/data/live/live_data_incident_analysis.1.csv",
        poke_interval=60,
        timeout=LIVE_FILE_TIMEOUT,
        mode="poke",
        dag=dag,
    )
else:
    wait_for_live_file = None
    logger.info("Skipping FileSensor for live data (RUN_MODE=manual)")

## ==================================================================
## Task 1c: Synchronize live → raw
## ==================================================================
if not SKIP_SYNC_DATA:
    sync_data = ConditionalDockerOperator(
        task_id="sync_data",
        image="data_collector:latest",
        command="python -u /opt/airflow/Services/DataCollection/src/sync_live_to_raw.py --delete-live",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[DATA_MOUNT, MODEL_MOUNT],
        mount_tmp_dir=False,
        dag=dag,
    )
else:
    sync_data = None
    logger.info("Skipping task: sync_data")

## ==================================================================
## Task 2: Data Preparation
## ==================================================================
if not SKIP_PREPROCESS_DATA:
    preprocess_data = ConditionalDockerOperator(
        task_id="preprocess_data",
        image="data_preparation:latest",
        command="python -u /opt/airflow/Services/DataPreparation/src/prepare_data.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[DATA_MOUNT, MODEL_MOUNT],
        mount_tmp_dir=False,
        dag=dag,
    )
else:
    preprocess_data = None
    logger.info("Skipping task: preprocess_data")

## ==================================================================
## Task 3: EDA Analysis
## ==================================================================
if not SKIP_EDA_ANALYSIS:
    eda_analysis = LiveDataDockerOperator(
        task_id="eda_analysis",
        image="data_preparation:latest",
        command="python /opt/airflow/Services/DataPreparation/src/eda_analysis.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[DATA_MOUNT, MODEL_MOUNT],
        mount_tmp_dir=False,
        dag=dag,
    )
else:
    eda_analysis = None
    logger.info("Skipping task: eda_analysis")

## ==================================================================
## Task 4: Model Training
## ==================================================================
if not SKIP_TRAIN_MODEL:
    train_model = ConditionalDockerOperator(
        task_id="train_model",
        image="model_training:latest",
        command="python /opt/airflow/Services/ModelTraining/src/train_model.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[DATA_MOUNT, MODEL_MOUNT],
        required_path="/opt/airflow/data/processed",
        must_exist=True,
        must_have_lines=True,
        mount_tmp_dir=False,
        dag=dag,
    )
else:
    train_model = None
    logger.info("Skipping task: train_model")

## ==================================================================
## Task 5b: Wait for Model File
## ==================================================================
wait_for_model_file = FileSensor(
    task_id="wait_for_model_file",
    #fs_conn_id="fs_project",
    fs_conn_id="fs_default",
    filepath="/opt/airflow/model/model_incident_analysis_incident_duration_min.joblib",
    poke_interval=30,
    timeout=MODEL_FILE_TIMEOUT,
    mode="poke",
    dag=dag,
)

## ==================================================================
## Task 5: Model Registration
## ==================================================================
register_model = ConditionalDockerOperator(
    task_id="register_model",
    image="model_training:latest",
    command="python /opt/airflow/Services/ModelTraining/src/register_model.py",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    mounts=[DATA_MOUNT, MODEL_MOUNT],
    required_path="/opt/airflow/model/model_incident_analysis_incident_duration_min.joblib",
    must_exist=True,
    must_have_lines=False,
    mount_tmp_dir=False,
    dag=dag,
)

## ==================================================================
## DAG Flow Definition
## ==================================================================
previous_task: Optional[object] = None
for task in [
    collect_live_data,
    wait_for_live_file,
    sync_data,
    preprocess_data,
    eda_analysis,
    train_model,
    wait_for_model_file,
    register_model,
]:
    if task:
        if previous_task:
            previous_task >> task
        previous_task = task
