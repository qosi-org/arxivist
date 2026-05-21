FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    tor \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Volumes: SQLite DB + config
VOLUME ["/app/data", "/app/configs"]

# Default: DRY_RUN trading loop
CMD ["python", "run.py", "--config", "configs/config.yaml", "--mode", "DRY_RUN"]
