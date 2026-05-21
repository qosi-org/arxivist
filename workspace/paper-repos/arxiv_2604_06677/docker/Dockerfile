FROM python:3.11-slim

WORKDIR /repo

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install package in editable mode
COPY src/ src/
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Copy rest of repo
COPY . .

ENV PYTHONPATH=/repo/src

ENTRYPOINT ["python", "scripts/reproduce_tables.py"]
CMD ["--table", "all", "--no-mc"]
