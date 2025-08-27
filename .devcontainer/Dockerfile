# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install native deps once at build-time (no Features, no postCreate apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git-lfs \
 && rm -rf /var/lib/apt/lists/*

# Make sure we drop back to the 'vscode' user (default in base image)
USER vscode
