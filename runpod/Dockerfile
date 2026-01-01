# RunPod Serverless - LLM + Reranker + Embedding
# Optimized for Fast Boot with pre-downloaded models

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    vllm>=0.4.0 \
    sentence-transformers>=3.0.0 \
    runpod \
    huggingface_hub \
    numpy

# Set HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Pre-download models for Fast Boot
# LLM - Qwen 3B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"

# Reranker
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Embedding
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV NVIDIA_VISIBLE_DEVICES=all

# Run handler
CMD ["python", "-u", "/app/handler.py"]
