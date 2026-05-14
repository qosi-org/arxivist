FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install project in editable mode
COPY . .
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p results data

# Default: show help
CMD ["python", "train.py", "--help"]
