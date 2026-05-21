# QSVM Fraud Detection — Docker image
# Base: python:3.10-slim (no CUDA — all quantum sim is CPU-based)
FROM python:3.10-slim

LABEL description="Quantum SVM for Fraud Detection (Ren & Zhang, IEEE CCPQT 2025)"
LABEL arxivist_paper_id="paper_qsvm_fraud_detection"

# System dependencies for Qiskit/numpy build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Persistent directories
VOLUME ["/app/data", "/app/checkpoints", "/app/results"]

# Kaggle credentials (required for data download script)
ENV KAGGLE_USERNAME=""
ENV KAGGLE_KEY=""

# Default: run training with primary config
CMD ["python", "train.py", "--config", "configs/config.yaml"]
