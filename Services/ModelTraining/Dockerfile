FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/workspace:/workspace/Services/ModelTraining/src:/workspace/Services/utils:/workspace/Services"

WORKDIR /workspace

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Copier le code complet (y compris ModelTraining et utils)
COPY . /workspace/

# Point d’entrée : entraîner un modèle
CMD ["python", "Services/ModelTraining/src/train_model.py", "-s", "incident_analysis"]
