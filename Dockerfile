FROM python:3.10-slim

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

WORKDIR /home/appuser/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy required source files
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Change ownership
RUN chown -R appuser:appuser .

# Use non-root user
USER appuser

# Run the data collector script
CMD ["python", "src/data/live_data_collector_both_strategies.py"]