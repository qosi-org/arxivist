# Base image matches the architecture plan's framework selection (PyTorch 2.2.2).
# The paper's reported runs were CPU-only (device=cpu); this CUDA-capable image
# also works fine for CPU execution.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
