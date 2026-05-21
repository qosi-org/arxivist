FROM python:3.10-slim

RUN apt-get update && apt-get install -y git wget build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "run_detection.py", "--help"]
