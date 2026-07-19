FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ ./src/
COPY configs/ ./configs/
COPY detect.py analyze.py causal.py run_all.py setup.py ./

# Install as editable package
RUN pip install -e .

# Create data/results dirs
RUN mkdir -p data results

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default: run full pipeline (mount data volume first)
CMD ["python", "run_all.py", "--buyers", "data/pumpfun_buyers.jsonl", "--launches", "data/pumpfun_launches.jsonl"]
