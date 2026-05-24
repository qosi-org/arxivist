FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

VOLUME ["/app/data", "/app/results", "/app/models"]
CMD ["python", "train_all.py", "--config", "configs/config.yaml", "--debug"]
