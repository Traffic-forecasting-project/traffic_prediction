''' 
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG with selective skipping of tasks, conditional evaluation, and FileSensors"
'''

## ==================================================================
## Standard library imports
## ==================================================================
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

## ==================================================================
## Airflow imports
## ==================================================================
from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

## ==================================================================
## Project imports
## ==================================================================
from Services.FastAPI.src.logging_utils import get_logger
from Services.FastAPI.src.custom_operators import LiveDataDockerOperator, ConditionalDockerOperator

## ==================================================================
## Logger setup
## ==================================================================
logger = get_logger(__name__)

## ==================================================================
## Project paths
## ==================================================================
## Fix directly the project root

PROJECT_ROOT = os.getenv("APP_INPUTDIR")
DATA_PATH = os.path.join(PROJECT_ROOT,os.getenv("DATA_PATH")) 

LIVE_DATA_PATH: os.path.join(DATA_PATH,"live")

LIVE_DATA_FILE: str = os.path.join(LIVE_DATA_PATH,"live_data_incident_analysis.17.csv")

MODEL_PATH: str = "Services/model/model_incident_analysis_incident_duration_min.joblib"


## ==================================================================
## Global monitoring options (for LiveDataDockerOperator)
## ==================================================================
MAX_DURATION: int = int(os.getenv("MAX_DURATION", 100))  ## seconds
MIN_LINE_INCREASE: int = int(os.getenv("MIN_LINE_INCREASE", 1000))
MODE: str = os.getenv("LIVE_MONITOR_MODE", "time")  ## "time", "lines", "both"

## ==================================================================
## FileSensor options
## ==================================================================
RUN_MODE: str = os.getenv("RUN_MODE", "manual")  ## "manual" or "continuous"
LIVE_FILE_TIMEOUT: int = int(os.getenv("LIVE_FILE_TIMEOUT", 3600))  ## seconds
MODEL_FILE_TIMEOUT: int = int(os.getenv("MODEL_FILE_TIMEOUT", 1800))  ## seconds


## ==================================================================
## Skip flags for optional tasks
## ==================================================================
def get_skip_flag(name: str) -> bool:
    """
    ## Helper function to read skip flag
    - Priority: Airflow Variable > Environment variable > default False
    """
    return Variable.get(
        name,
        os.getenv(name, "false")
    ).lower() in ["true", "1", "yes"]

#SKIP_COLLECT_LIVE_DATA = True
SKIP_COLLECT_LIVE_DATA: bool = get_skip_flag("SKIP_COLLECT_LIVE_DATA")
SKIP_PREPROCESS_DATA: bool = get_skip_flag("SKIP_PREPROCESS_DATA")
SKIP_EDA_ANALYSIS: bool = get_skip_flag("SKIP_EDA_ANALYSIS")
SKIP_TRAIN_MODEL: bool = get_skip_flag("SKIP_TRAIN_MODEL")

logger.info(
    "Task skipping options: COLLECT=%s, PREPROCESS=%s, EDA=%s, TRAIN=%s",
    SKIP_COLLECT_LIVE_DATA,
    SKIP_PREPROCESS_DATA,
    SKIP_EDA_ANALYSIS,
    SKIP_TRAIN_MODEL,
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
    description="MLOps pipeline for traffic prediction with selective skips and sensors",
    default_args=default_args,
    schedule_interval=None,  ## Manual trigger only
    catchup=False,
    tags=["mlops", "traffic", "pipeline"],
)

## ==================================================================
## Task 1: Data Collection (optional)
## ==================================================================
if not SKIP_COLLECT_LIVE_DATA:
    collect_live_data = LiveDataDockerOperator(
        task_id="collect_live_data",
        image="data_collector:latest",
	command = (
    		"python /workspace/Services/DataCollection/src/live_data_collector.py "
    		f"--arrondissement 17 --strategy incident_analysis "
    		f"--max-rows {MIN_LINE_INCREASE} "
    		f"--max-duration {MAX_DURATION} "
    		f"--arrondissements-path  {DATA_PATH}/data/arrondissements.csv"
	),
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
        monitor_file=str(LIVE_DATA_FILE),
        max_duration=MAX_DURATION,
        min_line_increase=MIN_LINE_INCREASE,
        mode=MODE,
        dag=dag,
    )
else:
    collect_live_data = None
    logger.info("Skipping task: collect_live_data")
   
## ==================================================================
## Task 1b: FileSensor for live data (only in continuous mode)
## ==================================================================
if RUN_MODE == "continuous":
    wait_for_live_file = FileSensor(
        task_id="wait_for_live_file",
        fs_conn_id="fs_project",
        filepath=LIVE_DATA_FILE,
        poke_interval=60,
        timeout=LIVE_FILE_TIMEOUT,
        mode="poke",
        dag=dag,
    )

else:
    wait_for_live_file = None
    logger.info("Skipping FileSensor for live data (RUN_MODE=manual)")

## ==================================================================
## Task 2: Data Preparation (optional)
## ==================================================================
if not SKIP_PREPROCESS_DATA:
    preprocess_data = ConditionalDockerOperator(
        task_id="preprocess_data",
        image="data_preparation:latest",
        command="python -u Services/DataPreparation/src/prepare_data.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
        dag=dag,
    )

else:
    preprocess_data = None
    logger.info("Skipping task: preprocess_data")

## ==================================================================
## Task 3: EDA Analysis (optional)
## ==================================================================
if not SKIP_EDA_ANALYSIS:
    eda_analysis = LiveDataDockerOperator(
        task_id="eda_analysis",
        image="data_preparation:latest",
        command="python Services/DataPreparation/src/eda_analysis.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
        dag=dag,
    )
else:
    eda_analysis = None
    logger.info("Skipping task: eda_analysis")

## ==================================================================
## Task 4: Model Training (optional, conditional)
## ==================================================================
if not SKIP_TRAIN_MODEL:
    train_model = ConditionalDockerOperator(
        task_id="train_model",
        image="model_training:latest",
        command="python Services/ModelTraining/src/train_model.py -s incident_analysis",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
        #required_path=str(LIVE_DATA_PATH),
        required_path=f"/workspace/Services/DataCollection/data/live",
        must_exist=True,
        must_have_lines=True,
        dag=dag,
    )
else:
    train_model = None
    logger.info("Skipping task: train_model")

## ==================================================================
## Task 5b: FileSensor for model presence
## ==================================================================
wait_for_model_file = FileSensor(
    task_id="wait_for_model_file",
    fs_conn_id="fs_project",
    filepath="/workspace/Services/model/model_incident_analysis_incident_duration_min.joblib",
    poke_interval=30,
    timeout=MODEL_FILE_TIMEOUT,
    mode="poke",
    dag=dag,
)

## ==================================================================
## Task 5: Model Registration (conditional)
## ==================================================================
register_model = ConditionalDockerOperator(
    task_id="register_model",
    image="model_training:latest",
    command="python Services/ModelTraining/src/register_model.py",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")],
    required_path=str("/workspace/" + MODEL_PATH),
    must_exist=True,
    must_have_lines=False,
    mount_tmp_dir=False,
    dag=dag,
)

## ==================================================================
## DAG Flow (dynamic chaining)
## ==================================================================
previous_task: Optional[object] = None
for task in [
    collect_live_data,
    wait_for_live_file,   ## only active in continuous mode
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
