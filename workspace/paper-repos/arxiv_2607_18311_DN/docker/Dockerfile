# Reproduction environment for arXiv:2607.18311
# "Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks"
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libxml2-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e . || true

CMD ["python", "train.py", "--help"]
