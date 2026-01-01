# RunPod Serverless - LLM + Reranker + Embedding
# Models downloaded at first run (cached by RunPod Flash Boot)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
RUN pip install --no-cache-dir \
    vllm==0.4.2 \
    sentence-transformers==3.0.1 \
    runpod \
    huggingface_hub \
    transformers \
    accelerate

# NOTE: Models downloaded at runtime, cached by RunPod Flash Boot
# First cold start will be slower (~2-3 min), subsequent starts fast

COPY handler.py /app/handler.py

ENV NVIDIA_VISIBLE_DEVICES=all
ENV HF_HOME=/runpod-volume/huggingface

CMD ["python", "-u", "/app/handler.py"]
