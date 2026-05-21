FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install project
COPY . .
RUN pip install -e .

# Default: show help
CMD ["python", "run_simulation.py", "--help"]
