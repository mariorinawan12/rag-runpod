# RunPod Serverless - LLM + Reranker + Embedding
# Using standard NVIDIA CUDA base image

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch and dependencies
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    vllm>=0.4.0 \
    sentence-transformers>=3.0.0 \
    runpod \
    huggingface_hub \
    numpy \
    transformers \
    accelerate

# Set HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Pre-download models for Fast Boot
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["python", "-u", "/app/handler.py"]
