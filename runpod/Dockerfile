# RunPod Serverless - LLM + Reranker + Embedding
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch first (CUDA 12.1)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    vllm==0.4.2 \
    sentence-transformers==3.0.1 \
    runpod \
    huggingface_hub \
    transformers \
    accelerate

# HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface

# Pre-download models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY handler.py /app/handler.py

ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["python", "-u", "/app/handler.py"]
