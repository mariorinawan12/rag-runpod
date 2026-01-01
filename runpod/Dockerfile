# RunPod vLLM Worker - Pre-built with vLLM
FROM runpod/worker-vllm:stable-cuda12.1.0

WORKDIR /app

# Add sentence-transformers for embed + rerank
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.1 \
    runpod

# Copy handler
COPY handler.py /app/handler.py

ENV HF_HOME=/runpod-volume/huggingface
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["python", "-u", "/app/handler.py"]
