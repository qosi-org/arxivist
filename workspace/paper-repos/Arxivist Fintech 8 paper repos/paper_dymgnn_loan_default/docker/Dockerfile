# Dockerfile — DYMGNN Loan Default Predictor
# Zandi et al. (EJOR 2025) — DOI: 10.1016/j.ejor.2024.09.025
# Compute: AMD Milan 7413 + NVidia A100 40GB (Table C.2)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
