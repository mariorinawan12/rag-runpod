# RunPod Serverless - LLM + Reranker
# Optimized for Fast Boot with pre-downloaded models

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    vllm==0.4.0 \
    sentence-transformers>=2.2.2 \
    runpod \
    huggingface_hub \
    torch>=2.1.0

# Set HuggingFace cache to persistent volume
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Pre-download LLM model for Fast Boot
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3-8B-Instruct', local_dir='/app/models/llama3')"

# Pre-download Reranker model for Fast Boot
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

# Run handler
CMD ["python", "-u", "/app/handler.py"]
