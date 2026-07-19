# ArXivist-generated Dockerfile for arxiv_2607_04280
# This is a CPU-bound discrete-event simulation (no GPU/CUDA required for
# reproduction), but we retain the standard PyTorch base image for template
# consistency and in case a user wants GPU-vectorized batches later.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
