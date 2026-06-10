# ── Base image ────────────────────────────────────────────────────────────────
# Use slim Python image; GPU users can swap to nvidia/cuda base
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="Deep News Summarizer"
LABEL description="End-to-end NLP pipeline: scrape → BART → keywords → sentiment → Streamlit dashboard"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller image, no CUDA needed for basic use)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
COPY app.py .
COPY scraper.py .
COPY summarizer.py .
COPY keywords.py .
COPY sentiment.py .
COPY .streamlit/ .streamlit/

# ── Streamlit config ──────────────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Pre-download models at build time so container starts instantly
# Comment out if you want a smaller image (models download on first run instead)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
print('Downloading DistilBART...'); \
AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6'); \
AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6'); \
print('Models cached.')"

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
