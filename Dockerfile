# Standalone Dockerfile for ShopOps OpenEnv environment.
# Uses python:3.11-slim so docker build works without access to internal base images.
#
# Build:
#   docker build -t shopops-env:latest .
#
# Run:
#   docker run -p 8000:8000 shopops-env:latest

FROM python:3.11-slim

WORKDIR /app

# Install curl (needed for HEALTHCHECK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the full project so server.app:app and models.py are importable
COPY . /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
