FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client ca-certificates bash tzdata \
  && rm -rf /var/lib/apt/lists/*

# Python deps for collector + DVC (uses your requirements.txt)
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /tmp/requirements.txt

# Avoid git "dubious ownership" when mounting the repo
RUN git config --global --add safe.directory /workspace

ENTRYPOINT ["bash", "-lc"]
CMD ["bash", "-lc", "tail -f /dev/null"]
