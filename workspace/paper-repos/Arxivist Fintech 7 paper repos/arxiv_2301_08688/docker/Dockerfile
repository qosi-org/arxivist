# Dockerfile — APEX LOB Trader (arXiv:2301.08688)
# Base image matches PyTorch version used with RLlib
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install project in editable mode
COPY . .
RUN pip install -e .

# Default: show training help
CMD ["python", "train.py", "--help"]
