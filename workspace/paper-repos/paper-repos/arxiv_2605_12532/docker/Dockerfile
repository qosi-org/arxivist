FROM python:3.11-slim

# System deps: tor for private channel, curl for healthchecks
RUN apt-get update && apt-get install -y \
    tor \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Data volume for SQLite persistence
VOLUME ["/app/data"]

# Environment variables (override at runtime)
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"
ENV EXECUTION_MODE="DRY_RUN"
ENV PYTHONUNBUFFERED=1

# Default: run dry-run with 14 default assets
CMD ["python", "scripts/run_dryrun.py", "--config", "configs/default.yaml", "--log-level", "INFO"]
