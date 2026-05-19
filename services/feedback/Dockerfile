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
COPY services/feedback/ /app/services/feedback/

# Install dependencies
COPY services/feedback/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV SERVICE_NAME=feedback
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "services.feedback.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "2"]
