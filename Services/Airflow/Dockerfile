## Airflow MLOPS image (Python 3.10)
FROM apache/airflow:2.10.2-python3.10

WORKDIR /opt/airflow

## Copier requirements
COPY requirements.txt /opt/airflow/requirements.txt

## Installer les dépendances Python en tant qu'utilisateur airflow
USER airflow
RUN pip install --no-cache-dir --progress-bar=off -r /opt/airflow/requirements.txt

COPY --chown=airflow:airflow src /opt/airflow/src
COPY --chown=airflow:airflow dags /opt/airflow/dags

ENV PYTHONPATH="/opt/airflow:/opt/airflow/src:/opt/airflow/Services:${PYTHONPATH}"

EXPOSE 8080
CMD ["standalone"]
