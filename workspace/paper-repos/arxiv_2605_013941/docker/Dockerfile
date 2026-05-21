FROM python:3.11-slim

# System dependencies (Appendix E: CPU-only, SQLite required)
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt tokenizer (required for metrics)
RUN python -m nltk.downloader punkt punkt_tab

# Project code
COPY . .
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p outputs logs checkpoints

CMD ["python", "answer.py", "--help"]
