# Multi-stage build: Build stage with compilation tools
# Platform flag removed for faster local development
# For production builds, add: FROM --platform=linux/amd64 python:3.11-slim AS builder
FROM python:3.11-slim AS builder

# Build arguments for user ID mapping
ARG USER_ID=1000
ARG GROUP_ID=1000

# Install build dependencies
# Note: psutil may have pre-built wheels, but we install build tools just in case
# Use DEBIAN_FRONTEND=noninteractive to avoid prompts
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv (fast Python package installer, written in Rust)
# Using standalone installer - much faster than pip
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null || \
    (mkdir -p /usr/local/bin && cp /root/.local/bin/uv /usr/local/bin/uv) && \
    chmod +x /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages using uv
# uv is 10-100x faster than pip and handles dependencies better
# Copy requirements.txt first for better Docker layer caching
# (This layer will be cached unless requirements.txt changes)
# Create a virtual environment with pip and setuptools seeded, then install packages
COPY requirements.txt .
RUN uv venv --seed /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements.txt

# Runtime stage: Minimal image with only runtime dependencies
# Platform flag removed for faster local development
# For production builds, add: FROM --platform=linux/amd64 python:3.11-slim
FROM python:3.11-slim

# Build arguments for user ID mapping
ARG USER_ID=1000
ARG GROUP_ID=1000

# Set environment variables
ENV APP_HOME=/app
ENV PYTHONUNBUFFERED=True
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH

WORKDIR $APP_HOME

# Install only runtime dependencies (no build tools)
# Note: OpenCV-python includes its own dependencies, minimal runtime deps needed
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder stage
# Try to copy virtual environment first, fallback to .local if using --system
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY . .

# Setup user with matching UID/GID from host
# This ensures files created in container have correct permissions on host
RUN groupadd -g ${GROUP_ID} appuser || true && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash appuser || true && \
    mkdir -p /home/appuser && \
    chown -R ${USER_ID}:${GROUP_ID} /home/appuser /app && \
    # Copy Python packages to user's local directory
    cp -r /root/.local /home/appuser/.local 2>/dev/null || true && \
    chown -R ${USER_ID}:${GROUP_ID} /home/appuser/.local 2>/dev/null || true && \
    # Also make venv accessible to appuser
    chown -R ${USER_ID}:${GROUP_ID} /opt/venv 2>/dev/null || true

# Create cache directory with correct ownership
RUN mkdir -p /app/.npCacheDir && \
    chown -R ${USER_ID}:${GROUP_ID} /app/.npCacheDir

EXPOSE 8000

# Run with Uvicorn as appuser
USER appuser
ENV PATH=/opt/venv/bin:/home/appuser/.local/bin:$PATH
# Use python -m uvicorn to ensure it's found
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
