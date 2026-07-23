FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "gymnasium[atari]" ale-py
RUN python -m ale_py.roms --import-all || true

COPY . .
RUN pip install -e .

CMD ["python", "train.py", "--help"]
