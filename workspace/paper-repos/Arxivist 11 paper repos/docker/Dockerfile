# Base image matches the framework version chosen in the architecture plan.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project
WORKDIR /workspace
COPY . .
RUN pip install -e .

# Default: show help
CMD ["python", "train.py", "--help"]
