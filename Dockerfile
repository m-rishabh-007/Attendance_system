# Dockerfile
# Multi-stage Dockerfile for Face Attendance System
# Optimized for Raspberry Pi (ARM architecture)
# 
# Build: docker build -t face-attendance:latest .
# Run:   docker run --rm --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix face-attendance:latest

# Use Python 3.9 slim image (compatible with ARM)
FROM python:3.9-slim-bullseye

# Set metadata
LABEL maintainer="your-email@example.com"
LABEL description="Face Attendance System - V3 HYBRID Architecture with YOLOv8n TFLite and BoT-SORT"
LABEL version="3.0"
LABEL architecture="V3 HYBRID - Pipeline Orchestration + Design Patterns"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Video device access
    v4l-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR $APP_HOME

# Copy requirements first (for layer caching)
COPY requirements-docker.txt .

# Install Python dependencies
# Use pip with no cache to reduce image size
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code (V3 HYBRID structure)
COPY models/ ./models/
COPY pipeline/ ./pipeline/
COPY detectors/ ./detectors/
COPY tracking/ ./tracking/
COPY common/ ./common/
COPY aligners/ ./aligners/
COPY recognizers/ ./recognizers/
COPY database/ ./database/
COPY server/ ./server/
COPY tests/ ./tests/
COPY attendance_system.py .
COPY config.yaml .
COPY ARCHITECTURE.md .
COPY README.md .
COPY docs/ ./docs/

# Create logs directory
RUN mkdir -p logs

# Create a non-root user for security
RUN useradd -m -u 1000 attendance && \
    chown -R attendance:attendance $APP_HOME

# Switch to non-root user
USER attendance

# Expose ports for future API server
# EXPOSE 8000

# Set working directory to app root
WORKDIR $APP_HOME

# Health check (checks if main script exists)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('attendance_system.py') else 1)"

# Default command: run the V3 HYBRID attendance system
# Override with: docker run ... python -c "your custom command"
CMD ["python", "attendance_system.py"]
