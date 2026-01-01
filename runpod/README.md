# RunPod Serverless Deployment Guide

## Quick Start

### 1. Prerequisites
- RunPod account with API key
- Docker Hub account
- GitHub account (for Actions)

### 2. Environment Variables
Add to your `.env`:
```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
INFERENCE_BACKEND=runpod
LLM_BACKEND=runpod
```

### 3. Build & Push Docker Image

**Option A: GitHub Actions (recommended)**
1. Push this folder to GitHub
2. Set secrets: `DOCKER_USERNAME`, `DOCKER_TOKEN`
3. Action auto-builds on push

**Option B: Local Build**
```bash
cd runpod
docker build -t yourusername/runpod-rag:latest .
docker push yourusername/runpod-rag:latest
```

### 4. Deploy to RunPod
1. Go to RunPod Serverless
2. Create new endpoint
3. Select your Docker image
4. Set GPU type: RTX 4090 or A100
5. Enable "Fast Boot"
6. Copy endpoint ID to `.env`

### 5. Test
```python
from runpod_client import get_client

client = get_client()

# Test LLM
result = await client.generate_simple(
    system_instruction="You are helpful",
    user_content="Hello!"
)
print(result["text"])

# Test Reranker
scores = await client.rerank(
    query="What is AI?",
    documents=["AI is...", "Machine learning..."]
)
print(scores)
```

## Files
- `handler.py` - RunPod serverless handler
- `Dockerfile` - Docker build config
- `requirements.txt` - Python dependencies
- `runpod_client.py` - Local client (copy to `app/clients/`)
