FROM python:3.10-slim

WORKDIR /app

# Install only essential system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results

# Set environment variable to avoid pyplot issues
ENV DISPLAY=:0

EXPOSE 5000

# Use simpler command for debugging
# Change the last line in Dockerfile to:
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "--log-level", "debug", "--access-logfile", "-"]

