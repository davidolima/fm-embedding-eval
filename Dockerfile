#Start from a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Authenticate to HF
RUN if [ ! -z "$HUGGINGFACE_TOKEN" ]; then \
    pip install -U "huggingface_hub[cli]" && \
    huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential; \
    fi

# Copy all files from current directory to working directory
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Download models
RUN python3 download_models.py

# Default command - you can override this when running the container
CMD ["python3", "test.py"]
