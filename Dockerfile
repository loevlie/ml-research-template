# Reproducible training environment (Rule 23).
# Build: docker build -t my_project .
# Run:   docker run --gpus all -v $(pwd):/workspace my_project uv run python src/my_project/train.py

FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Install Python + dependencies (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project

# Copy project and install
COPY . .
RUN uv sync --locked
