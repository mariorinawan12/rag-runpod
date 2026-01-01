"""
RunPod Serverless Handler - LLM + Reranker + Embedding
=======================================================

Multi-action handler for RAG inference:
- action: "generate" → vLLM text generation
- action: "rerank" → CrossEncoder reranking
- action: "embed" → Embedding (simple or with tokens for late chunking)

Models are pre-loaded at startup for Fast Boot.
"""

import runpod
import torch
import numpy as np
from typing import List, Dict, Any

# ============================================================================
# GLOBAL MODELS (Loaded once at startup)
# ============================================================================

print("🚀 Loading models...")

# LLM via vLLM
from vllm import LLM, SamplingParams

LLM_MODEL = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",  # Fast cold start + reliable citations
    gpu_memory_utilization=0.40,       # Small footprint, room for embed+rerank
    max_model_len=8192,
    dtype="float16"
)
print("✅ LLM loaded")

# Reranker via CrossEncoder
from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",  # Your config
    device="cuda"
)
RERANKER.model.half()
print("✅ Reranker loaded")

# Embedding Model
from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer(
    "BAAI/bge-m3",  # Your config - 1024 dim
    device="cuda"
)
EMBEDDER.half()
TOKENIZER = EMBEDDER.tokenizer
print("✅ Embedder loaded")

print("🎉 All models ready!")


# ============================================================================
# HANDLER FUNCTIONS
# ============================================================================

def generate_text(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using vLLM.
    
    Input:
        messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_tokens: int (default 1000)
        temperature: float (default 0.2)
    
    Output:
        text: str
        usage: {prompt_tokens, completion_tokens, total_tokens}
    """
    messages = input_data.get("messages", [])
    max_tokens = input_data.get("max_tokens", 1000)
    temperature = input_data.get("temperature", 0.2)
    
    if not messages:
        return {"error": "No messages provided"}
    
    # Build prompt from messages
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.01,
        top_p=0.95
    )
    
    outputs = LLM_MODEL.generate([prompt], params)
    generated_text = outputs[0].outputs[0].text
    
    prompt_tokens = len(prompt.split()) * 1.3
    completion_tokens = len(generated_text.split()) * 1.3
    
    return {
        "text": generated_text,
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens)
        }
    }


def rerank_documents(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rerank documents using CrossEncoder.
    
    Input:
        query: str
        documents: List[str]
    
    Output:
        scores: List[float]
    """
    query = input_data.get("query", "")
    documents = input_data.get("documents", [])
    
    if not query or not documents:
        return {"error": "Query and documents required", "scores": []}
    
    pairs = [[query, doc] for doc in documents]
    
    with torch.inference_mode():
        scores = RERANKER.predict(pairs, batch_size=32, convert_to_tensor=True)
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy().tolist()
        else:
            scores = list(scores)
    
    return {"scores": scores}


def embed_text(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embed text - supports both simple and token-level embedding.
    
    Input:
        text: str (single text) OR texts: List[str] (batch)
        return_tokens: bool (default False)
            - False: return single embedding vector
            - True: return token embeddings + offset mapping (for late chunking)
    
    Output (return_tokens=False):
        embedding: List[float]  # Single vector
        OR
        embeddings: List[List[float]]  # Batch
    
    Output (return_tokens=True):
        token_embeddings: List[List[float]]  # Per-token vectors
        offset_mapping: List[List[int]]  # (char_start, char_end) per token
    """
    text = input_data.get("text", "")
    texts = input_data.get("texts", [])
    return_tokens = input_data.get("return_tokens", False)
    
    # Handle batch or single
    if texts:
        input_texts = texts
    elif text:
        input_texts = [text]
    else:
        return {"error": "No text provided"}
    
    with torch.inference_mode():
        if return_tokens:
            # TOKEN-LEVEL EMBEDDING (for late chunking)
            # Only works with single text
            if len(input_texts) > 1:
                return {"error": "return_tokens only supports single text"}
            
            single_text = input_texts[0]
            
            # Tokenize with offset mapping
            tokenized = TOKENIZER(
                single_text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            offset_mapping = tokenized["offset_mapping"][0].tolist()
            
            # Get token embeddings
            token_embeddings = EMBEDDER.encode(
                single_text,
                output_value="token_embeddings",
                convert_to_numpy=False,
                show_progress_bar=False
            )
            
            if torch.is_tensor(token_embeddings):
                token_embeddings = token_embeddings.cpu().numpy()
            
            if len(token_embeddings.shape) == 3:
                token_embeddings = token_embeddings[0]
            
            return {
                "token_embeddings": token_embeddings.tolist(),
                "offset_mapping": offset_mapping
            }
        
        else:
            # SIMPLE EMBEDDING (pooled vector)
            embeddings = EMBEDDER.encode(
                input_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            if len(input_texts) == 1:
                return {"embedding": embeddings[0].tolist()}
            else:
                return {"embeddings": embeddings.tolist()}


# ============================================================================
# MAIN HANDLER
# ============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler.
    
    Routes based on 'action' field:
    - "generate" → LLM text generation
    - "rerank" → Document reranking
    - "embed" → Text embedding (simple or with tokens)
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "generate")
        
        if action == "generate":
            return generate_text(input_data)
        elif action == "rerank":
            return rerank_documents(input_data)
        elif action == "embed":
            return embed_text(input_data)
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless
runpod.serverless.start({"handler": handler})

