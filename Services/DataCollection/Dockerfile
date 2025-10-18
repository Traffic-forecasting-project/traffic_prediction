# ==========================
# Dockerfile for DataCollector Service
# Compatible with WSL2 and Linux (Airflow integration)
# ==========================

# Use a lightweight official Python base image
FROM python:3.11-slim

# Set environment variables for consistent Python behavior
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Define the working directory inside the container
WORKDIR /opt/airflow

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client ca-certificates bash tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create required directories (to prevent "directory not found" errors)
RUN mkdir -p /opt/airflow/data/live \
    && mkdir -p /opt/airflow/data/raw \
    && mkdir -p /opt/airflow/data/processed \
    && chmod -R 777 /opt/airflow/data

# Copy everything from the current project into the container
# This ensures all Services, data, utils, DAGs, configs, etc. are available
COPY . /opt/airflow/

# Copy the service-specific requirements file (redundant but explicit)
COPY Services/DataCollection/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Default command to run the DataCollector script
CMD ["python", "/opt/airflow/Services/DataCollection/src/live_data_collector.py"]
