# Dockerfile — forecast_risk
# Paper: "Quantifying the Risk-Return Tradeoff in Forecasting"
# arXiv: 2605.09712

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p data results/losses results/metrics results/figures

# Default: run evaluation with default config
CMD ["python", "run_evaluation.py", "--config", "configs/default_config.yaml"]
