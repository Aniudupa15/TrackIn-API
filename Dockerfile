FROM python:3.10-slim

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --use-pep517 --no-cache-dir -r requirements.txt


# Copy app source code
COPY . /app
WORKDIR /app

# Expose port and run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
