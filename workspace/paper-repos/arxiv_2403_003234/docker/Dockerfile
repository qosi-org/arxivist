# Base image matches the architecture plan (PyTorch 2.1 + CUDA 11.8)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget curl build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Mamba kernels build against the installed torch/CUDA (needs a CUDA toolchain).
RUN pip install --no-cache-dir mamba-ssm causal-conv1d || \
    echo "WARN: mamba-ssm/causal-conv1d not built; CPU reference path only."

WORKDIR /workspace
COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
