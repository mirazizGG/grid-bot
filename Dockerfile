FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY training /app/training
COPY data_pipeline /app/data_pipeline
COPY scripts /app/scripts
COPY compose.yaml Dockerfile /app/
RUN python -c "from pathlib import Path; forbidden=['backend/.env','backend/runtime','backend/analyses.db','backend/uploads','backend/server_log.txt','backend/model/chart_signal_model.onnx']; assert not any(Path(p).exists() for p in forbidden), 'Private runtime files entered the image'"
RUN mkdir -p /opt/grid-bot-model /app/data /app/backups /app/bridge \
    && cp /app/backend/model/numeric_model.joblib /opt/grid-bot-model/ \
    && cp /app/backend/model/numeric_model_meta.json /opt/grid-bot-model/

WORKDIR /app/backend
CMD ["python", "docker_entrypoint.py"]
