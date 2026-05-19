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
COPY services/generator/ /app/services/generator/

# Install dependencies
COPY services/generator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV SERVICE_NAME=generator
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "services.generator.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
