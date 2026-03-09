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
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy project files
COPY . .

# Create folders needed at runtime
RUN mkdir -p static/img upload

EXPOSE 10000

# Use PORT env var (set by Render automatically)
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --timeout 120 main:app
