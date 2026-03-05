FROM python:3.10-slim

# System dependencies for audio/image processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    Flask==2.2.5 \
    Werkzeug==2.2.3 \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    Pillow \
    opencv-python-headless \
    tensorflow-cpu \
    praat-parselmouth \
    librosa \
    gunicorn

# Copy project files
COPY . .

# Create folders needed at runtime
RUN mkdir -p static/img upload

EXPOSE 8080

# Use PORT env var (set by Railway/Render/Fly.io automatically)
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120 --preload main:app
