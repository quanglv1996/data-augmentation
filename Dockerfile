FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set working directory to webapp folder
WORKDIR /app/webapp

# Create necessary directories
RUN mkdir -p uploads outputs

# Expose port
EXPOSE 2222

# Set environment variable
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
