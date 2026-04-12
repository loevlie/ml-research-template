# Reproducible training environment (Rule 20).
# Build: docker build -t my_project .
# Run:   docker run --gpus all -v $(pwd):/workspace my_project python src/my_project/train.py

FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Install Python dependencies (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy project
COPY . .

RUN pip install --no-cache-dir -e .
