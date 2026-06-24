FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Flask app via Gunicorn (PORT is injected by Render/Railway)
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000}"]
