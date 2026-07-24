# Reproduction environment for arXiv:2607.20168
# "Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage"
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e . || true

CMD ["python", "run_study.py", "--config", "configs/config.yaml", "--study", "main", "--debug"]
