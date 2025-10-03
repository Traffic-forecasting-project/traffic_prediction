FROM apache/airflow:2.9.0-python3.10

WORKDIR /opt/airflow

COPY requirements.txt /opt/airflow/requirements.txt

USER airflow
RUN pip install --no-cache-dir --progress-bar=off -r /opt/airflow/requirements.txt

#COPY --chown=airflow:airflow src /opt/airflow/src
# COPY --chown=airflow:airflow dags /opt/airflow/dags

ENV PYTHONPATH="/opt/airflow:/opt/airflow/src:/opt/airflow/Services:${PYTHONPATH}"

EXPOSE 8080
CMD ["bash"]
