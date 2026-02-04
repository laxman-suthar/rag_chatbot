FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.prod.txt

# Download spaCy model (optional)
# RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/media /app/staticfiles /app/data/vector_store

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f ${HEALTHCHECK_URL:-http://localhost:8000/api/health/} || exit 1

# Run gunicorn
CMD ["gunicorn", "rag_chatbot.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
