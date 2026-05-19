FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy shared module
COPY shared/ /app/shared/

# Copy service code
COPY services/recommender/ /app/services/recommender/

# Install dependencies
COPY services/recommender/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV SERVICE_NAME=recommender
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "services.recommender.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
