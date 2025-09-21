FROM python:3.9-slim

WORKDIR /code

# Copy requirements files
COPY requirements.txt packages.txt ./

# Install system dependencies and cleanup in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 \
    && cat packages.txt | xargs apt-get install -y --no-install-recommends \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure langchain-huggingface is installed
RUN pip install --no-cache-dir langchain-huggingface

# Create cache directories with correct permissions
RUN mkdir -p /tmp/huggingface_cache && \
    mkdir -p /tmp/sentence_transformers && \
    mkdir -p /tmp/nltk_data && \
    chmod -R 777 /tmp/huggingface_cache && \
    chmod -R 777 /tmp/sentence_transformers && \
    chmod -R 777 /tmp/nltk_data

# Set environment variables (updated for new HF_HOME)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    PORT=7860 \
    HF_HOME=/tmp/huggingface_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers \
    NLTK_DATA=/tmp/nltk_data

# Pre-download the model to avoid runtime issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 7860

# Health check (adjust endpoint if needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Command to run the application
CMD ["python", "App.py"]
