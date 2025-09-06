FROM python:3.10-alpine

WORKDIR /app

# Alpine uses apk instead of apt-get
RUN apk update && \
    apk add --no-cache \
    libstdc++ \
    libgcc \
    musl-dev \
    linux-headers \
    g++ \
    make

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads results

EXPOSE 5000

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
