# ===========================================================
# AIRFLOW SERVICE DOCKERFILE
# ===========================================================
FROM apache/airflow:2.8.3-python3.10

# Switch to root to prepare folders
USER root

# Create required directories
RUN mkdir -p /opt/airflow/dags \
    && mkdir -p /opt/airflow/logs \
    && mkdir -p /opt/airflow/data \
    && mkdir -p /opt/airflow/model \
    && mkdir -p /opt/airflow/Services \
    && chmod -R 777 /opt/airflow

# Copy requirements
COPY requirements.txt /requirements.txt

# Switch to airflow user for pip install (Airflow restriction)
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

# Back to root to copy project code
USER root
COPY . /opt/airflow/

# Environment setup
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow

# Switch back to airflow user
USER airflow

# Default command
CMD ["bash", "-c", "airflow db init && airflow webserver & airflow scheduler"]
