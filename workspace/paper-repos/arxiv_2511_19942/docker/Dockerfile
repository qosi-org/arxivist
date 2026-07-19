FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
