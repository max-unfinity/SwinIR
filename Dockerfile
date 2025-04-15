FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This installed redundant nvidia-cu12 packages, maybe use torch==2.0.1 + torchvision==0.15.2?
# Install Python packages directly
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    timm==0.4.12 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    opencv-python \
    Pillow \
    tqdm \
    fastapi \
    uvicorn \
    requests \
    pydantic \
    python-json-logger \
    python-multipart

# COPY . .
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
