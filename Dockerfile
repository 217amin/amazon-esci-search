FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0

# System deps for faiss + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements-serve.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements-serve.txt

# Pre-download NLTK data needed by BM25 stemming, so first request is fast
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab', quiet=True)"

# Copy project source. Artifacts are NOT copied in — they mount from a Volume.
COPY src/ ./src/
COPY configs/ ./configs/

# Defaults you can override at run time.
ENV CE_DEVICE=cpu \
    ARTIFACTS_DIR=/data/artifacts/systemA \
    PROCESSED_DIR=/data/processed \
    MATRYOSHKA_DIR=/data/artifacts/matryoshka_models \
    MATRYOSHKA_SUBDIR=us

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]