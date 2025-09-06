FROM python:3.10-slim

WORKDIR /app

# First install system dependencies alone to avoid conflicts
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
