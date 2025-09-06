#!/usr/bin/env bash
set -e

# Paramètres
APP_PORT="${APP_PORT:-8000}"
RELOAD="${RELOAD:-false}"
MODEL_PATH="${MODEL_PATH:-/app/models/model.joblib}"

# Avertir si le modèle n'est pas présent (l’API démarre quand même)
if [ ! -f "$MODEL_PATH" ]; then
  echo "WARNING: Model not found at $MODEL_PATH. /predict risque d'échouer."
fi

UVICORN_ARGS=""
if [ "$RELOAD" = "true" ]; then
  UVICORN_ARGS="--reload"
fi

# L’app FastAPI doit être exposée comme src.service:app
exec uvicorn src.service:app --host 0.0.0.0 --port "$APP_PORT" $UVICORN_ARGS