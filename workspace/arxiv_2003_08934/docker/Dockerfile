# Dockerfile — NeRF reproduction (arXiv:2003.08934)
# Base image matches architecture_plan.json framework.primary = tensorflow.
FROM tensorflow/tensorflow:2.15.0-gpu

# System deps: colmap (real-scene pose estimation), ffmpeg (fly-through video export),
# libsm6/libxext6 (OpenCV runtime deps), git.
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nerf

# Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source.
COPY . .
RUN pip install -e .

# Default: show CLI help so `docker run` without arguments is informative.
CMD ["python", "train.py", "--config", "configs/lego.yaml", "--datadir", "/data/lego", "--expname", "lego_full"]
