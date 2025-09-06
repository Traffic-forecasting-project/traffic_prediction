'''
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG for the main MLOps pipeline: data collection, preparation, EDA, training and evaluation"
'''

import os
import subprocess
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from Services.DataCollection.src.live_data_collector import run_live_data_pipeline
# from Services.DataPreparation.src.prepare_data import run_prepare_data_pipeline
# from src.eda.eda_analysis import run_eda_pipeline
# from src.model.train_model import run_train_model_pipeline, run_model_evaluation

from Services.FastAPI.src.logging_utils import get_logger

logger = get_logger(__name__)

def check_if_live_empty() -> str:
    """
        Branch task: Check if live data folder is empty
    """
    # Print current dir
    print(f"Current dir {os.getcwd()}")
    live_folder = os.path.join(os.getcwd(), "Services","DataCollection","data", "live")

    #Create if not existing
    os.makedirs(live_folder, exist_ok=True)
    try:
        is_empty = not any(os.scandir(live_folder))
    except FileNotFoundError:
        is_empty = True
    logger.info(f"'data/live/' folder is {'empty' if is_empty else 'not empty'}")
    # return "task_live_data" if is_empty else "skip_live_data"
    return "task_live_data"

def launch_airflow_ui():
    """
        Utility to launch Airflow UI (webserver + scheduler)
    """
    
    try:
        logger.info("Launching Airflow webserver and scheduler...")
        subprocess.Popen(["airflow", "webserver", "--port", "8080"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["airflow", "scheduler"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Airflow UI available at http://localhost:8080")
    except Exception as e:
        logger.error(f"Failed to launch Airflow UI: {e}")
        raise

### Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

dag = DAG(
    dag_id="main_pipeline_dag",
    description="MLOps pipeline for traffic prediction",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "traffic", "pipeline"]
)

"""
    This DAG performs the following steps:
        1. Branch: check if live data folder is empty
        2. If empty → collect live data, else skip
        3. Prepare data
        4. EDA analysis
        5. Train model
        6. Evaluate model
        7. (Optional) Monitor drift
"""

### Branching task to decide whether to run live data collection
branch_task = BranchPythonOperator(
    task_id="check_if_live_empty",
    python_callable=check_if_live_empty,
    dag=dag
)

### Task to collect live data
task_live_data = PythonOperator(
    task_id="task_live_data",
    python_callable=run_live_data_pipeline,
    op_kwargs={"arrondissement": 17, "strategy": "incident_analysis"},
    dag=dag
)

### Dummy task if data is already available
skip_live_data = EmptyOperator(
    task_id="skip_live_data",
    dag=dag
)

### Data preparation task
# task_prepare_data = PythonOperator(
#     task_id="task_prepare_data",
#     python_callable=run_prepare_data_pipeline,
#     op_kwargs={"strategy": "incident_analysis"},
#     dag=dag
# )

### EDA analysis task
# task_eda = PythonOperator(
#     task_id="task_eda",
#     python_callable=run_eda_pipeline,
#     op_kwargs={"strategy": "incident_analysis"},
#     dag=dag
# )

### Model training task
# task_train_model = PythonOperator(
#     task_id="task_train_model",
#     python_callable=run_train_model_pipeline,
#     op_kwargs={"strategy": "incident_analysis"},
#     dag=dag
# )

### Model evaluation task
# task_evaluate = PythonOperator(
#     task_id="task_evaluate",
#     python_callable=run_model_evaluation,
#     op_kwargs={"strategy": "incident_analysis"},
#     dag=dag
# )

### Optional: model drift monitoring task
## task_monitor_drift = PythonOperator(
##     task_id="task_monitor_drift",
##     python_callable=run_monitor_drift,
##     op_kwargs={"strategy": "incident_analysis"},
##     dag=dag
## )

### DAG flow
branch_task >> [task_live_data, skip_live_data]


# [task_live_data, skip_live_data] >> task_prepare_data
# task_prepare_data >> task_eda >> task_train_model >> task_evaluate
## task_evaluate >> task_monitor_drift  ### Uncomment when used
