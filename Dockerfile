# syntax=docker/dockerfile:1
#
# ---------------------------------------------------------------------------
# Dockerfile — LLM Reliability & Experimentation Platform
# ---------------------------------------------------------------------------
# Two-stage build:
#   1. "builder" stage installs build tooling + Python deps into a venv.
#      This keeps compilers and header files (needed to build wheels for
#      packages like torch/sentence-transformers if present in
#      requirements.txt) out of the final image.
#   2. "runtime" stage copies only the venv + application source, on top
#      of a clean python:3.12-slim base, so the shipped image stays small.
#
# NOTE: sentence-transformers / torch pull in native extensions. If your
# requirements.txt does NOT include them, the extra build-essential/
# cmake packages below are harmless but unused — remove them to shave
# more size off the builder stage if you want a stricter build.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files / buffering stdout, speeds up
# and cleans up the build layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System build dependencies required to compile native extensions used by
# torch / sentence-transformers / tokenizers (rust-based) and by common
# scientific-python packages (numpy/scipy) when no manylinux wheel matches.
# - build-essential: gcc/g++/make for compiling C/C++ extensions
# - cmake, git:      required by some sentence-transformers/torch builds
# - libgomp1:        OpenMP runtime needed by torch at *build* time here;
#                     also duplicated in the runtime stage since torch
#                     needs it at *run* time too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so we can copy just this
# directory into the runtime stage (instead of the whole system site-packages).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (separate layer) so that code-only
# changes don't invalidate the (often slow) dependency-install cache layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Streamlit-specific: disable telemetry and the "email prompt" that
    # blocks first-run in non-interactive/container environments.
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Runtime-only system dependencies:
# - libgomp1:    OpenMP runtime required by torch/sentence-transformers
#                at inference time (no compiler needed here).
# - curl:        used by the HEALTHCHECK below to hit Streamlit's
#                built-in health endpoint.
# Kept deliberately minimal — no compilers ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    # Run as a non-root user for defense-in-depth.
    && useradd --create-home --uid 1000 --shell /bin/bash appuser

# Bring in the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application source code.
# (.dockerignore keeps local junk such as .env, data/, mlruns/, __pycache__
# out of the build context so this stays a clean copy of just the code.)
COPY --chown=appuser:appuser . .

# Ensure mount points for persisted state exist and are writable by
# appuser even before docker-compose bind-mounts ./data and ./mlruns
# over them (docker-compose volumes inherit host directory ownership,
# so this mainly matters for `docker run` without compose).
RUN mkdir -p /app/data /app/mlruns /app/logs && \
    chown -R appuser:appuser /app/data /app/mlruns /app/logs

USER appuser

# Streamlit's default port.
EXPOSE 8501

# Basic container-level health check against Streamlit's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]