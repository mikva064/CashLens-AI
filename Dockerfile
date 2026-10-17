# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CASHLENS_ANALYZER=model \
    CASHLENS_MODEL_DIR=/models/qwen3-vl-2b \
    CASHLENS_RULES=/app/config/rules.json \
    CASHLENS_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY analyzer ./analyzer
COPY app ./app
COPY config/rules.json ./config/rules.json
COPY web ./web

RUN python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install ".[service,model]" \
    && python -m pip check \
    && useradd --create-home --uid 10001 cashlens \
    && mkdir -p /data \
    && chown cashlens:cashlens /data

ENV HOME=/home/cashlens
USER cashlens
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
