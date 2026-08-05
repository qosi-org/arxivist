# ArXivist-generated Dockerfile for:
# "Quantitative Universal Approximation for Noisy Quantum Neural Networks"
# CPU-only image: circuits are small (5-8 qubits) and classical optimisation of
# the few circuit angles theta does not require a GPU (architecture_plan.json).

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "train.py", "--help"]
