FROM python:3.10-slim

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

VOLUME ["/app/data", "/app/results", "/app/models"]
CMD ["python", "train.py", "--help"]
