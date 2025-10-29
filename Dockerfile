FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System basics only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install deps first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Env toggles
ENV ENABLE_MT5=0 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=10000 \
    UVICORN_WORKERS=1

EXPOSE 10000
# Respect Render's PORT env if provided, default to 10000
CMD ["sh", "-c", "python -m uvicorn global_agents.api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]