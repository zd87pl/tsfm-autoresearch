# Forecast Service — GCP Cloud Run Deployment
# M10: Production-scale multi-tenant forecasting service

FROM python:3.12-slim

WORKDIR /app

# System deps for FAISS/Qdrant
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv and deps
RUN pip install uv && uv sync --frozen --no-dev

# Copy source
COPY src/ src/
COPY experiments/ experiments/
COPY autoresearch_agent/ autoresearch_agent/

# TimesFM model weights are downloaded at startup from HuggingFace Hub
# Cached to /app/model_cache to survive container restarts
ENV HF_HOME=/app/model_cache
ENV TIMESFM_MODEL_ID=google/timesfm-2.5-200m-pytorch

# FastAPI service
EXPOSE 8080
CMD ["uv", "run", "uvicorn", "deploy.gcp.service:app", "--host", "0.0.0.0", "--port", "8080"]
