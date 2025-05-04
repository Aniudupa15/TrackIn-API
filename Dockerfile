FROM python:3.9-slim

# Install only necessary system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy opencv-python-headless Pillow python-multipart

# Create directory for uploaded files
RUN mkdir -p /app/uploaded_images && chmod 777 /app/uploaded_images

# Expose FastAPI port
EXPOSE 7860

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]