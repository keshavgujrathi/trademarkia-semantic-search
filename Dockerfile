FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim
WORKDIR /app

ENV PYTHONPATH=/app

# Copy dependencies and application files
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY faiss_index/ ./faiss_index/

# Set up the non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Pre-download the SentenceTransformer model into the appuser's cache layer.
# This ensures zero download latency on container startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]