# ---------------------------------------------------------------------------
# Rare Neurological Diseases MRI Classification - inference API
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    MODEL_BACKEND=convnext \
    DEVICE=cpu

WORKDIR /app

# libgomp1 is required by scikit-learn; the slim base does not ship it
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only wheels: the full CUDA build would add several GB for no benefit here
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py gradcam.py ./

# Model artifacts are mounted at runtime via docker-compose rather than baked in,
# so the image does not need rebuilding every time a model is retrained.
# To ship a self-contained image instead, uncomment the next line:
# COPY models/ /app/models/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
