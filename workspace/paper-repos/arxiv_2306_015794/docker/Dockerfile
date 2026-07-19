# Base image matches the architecture plan (PyTorch 2.1 + CUDA 11.8)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project
WORKDIR /workspace
COPY . .
RUN pip install -e .

# Default: show help
CMD ["python", "train.py", "--help"]
