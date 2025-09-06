FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dépendances système minimales (pour wheels numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances spécifiques à l’API
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Copier le code FastAPI isolé
COPY src/ /app/src/

# Ports (indicatif)
EXPOSE 8000

# Lancer Uvicorn (port configurable via APP_PORT dans docker-compose)
CMD ["sh", "-c", "uvicorn src.service:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]