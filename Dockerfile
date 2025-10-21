# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

EXPOSE 8000

ENV ORCH_CONFIG_DIR=/app/config \
    ORCH_USE_DUMMY=1

CMD ["uvicorn", "src.orch.server:app", "--host", "0.0.0.0", "--port", "8000"]
