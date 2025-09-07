FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace:/workspace/Services/DataCollection/src:/workspace/Services/utils:/workspace/Services/FastAPI:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client ca-certificates bash tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "/workspace/Services/DataCollection/src/live_data_collector.py"]