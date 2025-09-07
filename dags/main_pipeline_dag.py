from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from docker.types import Mount
import os


default_args = {"owner": "airflow", "start_date": datetime(2023, 1, 1)}

PROJECT_ROOT="/home/karimelq/Documents/Progress/Projects/TomTom/traffic_prediction"


with DAG("mlops_pipeline", default_args=default_args, schedule_interval=None, catchup=False) as dag:


    collect = DockerOperator(
        task_id="collect_data",
        image="data_collector:latest",
        command="bash -c 'echo PROJECT_ROOT=$PROJECT_ROOT && python /workspace/Services/DataCollection/src/live_data_collector.py -a 17 -s incident_analysis'",
        docker_url="unix://var/run/docker.sock",
        auto_remove=True,
        mount_tmp_dir=False,
        mounts=[Mount(source=PROJECT_ROOT, target="/workspace", type="bind")]
    )

    prep = DockerOperator(
        task_id="prepare_data",
        image="data_preparation:latest",
        command="python /workspace/Services/DataPreparation/src/prepare_data.py --strategy incident_analysis",
        docker_url="unix://var/run/docker.sock",
        auto_remove=True
    )

    train = DockerOperator(
        task_id="train_model",
        image="model_training:latest",
        command="python /workspace/Services/ModelTraining/src/train_model.py --strategy incident_analysis",
        docker_url="unix://var/run/docker.sock",
        auto_remove=True
    )

    collect >> prep >> train
