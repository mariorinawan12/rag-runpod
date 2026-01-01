"""
LLM Service - Hybrid Backend (Local / RunPod / Grok API)
Switch via LLM_BACKEND env var: "local", "runpod", or "grok"
"""
import logging
import time
import asyncio
from typing import Dict, Any, Generator, AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# LOCAL MODEL (Lazy loaded)
# ============================================================================
_local_model = None
_local_tokenizer = None

def _get_local_model():
    """Get local LLM model (lazy load)."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = settings.RUNPOD_LLM_MODEL
        logger.info(f"🧠 Loading local LLM: {model_name}...")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if settings.DEVICE == "cuda" else torch.float32,
            device_map="auto" if settings.DEVICE == "cuda" else None
        )
        if settings.DEVICE == "cpu":
            _local_model = _local_model.to("cpu")
        logger.info(f"✅ Local LLM loaded: {model_name}")
    return _local_model, _local_tokenizer

def _generate_local(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """Generate using local GPU/CPU."""
    model, tokenizer = _get_local_model()
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    start = time.time()
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    latency_ms = int((time.time() - start) * 1000)
    
    logger.info(f"✅ LLM (Local) | {latency_ms}ms")
    
    return {
        "text": response,
        "model_used": settings.RUNPOD_LLM_MODEL,
        "backend": "local",
        "latency_ms": latency_ms,
        "token_usage": {"total_tokens": len(outputs[0])}
    }

# ============================================================================
# GROK CLIENT
# ============================================================================
_grok_client = None

def _get_grok_client():
    """Get Grok API client."""
    global _grok_client
    if _grok_client is None:
        from openai import OpenAI
        _grok_client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
        logger.info(f"✅ Grok client connected: {settings.LLM_MODEL}")
    return _grok_client

def _generate_grok(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """Generate using Grok API."""
    client = _get_grok_client()
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    start = time.time()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    latency_ms = int((time.time() - start) * 1000)
    usage = response.usage
    
    logger.info(f"✅ LLM (Grok) | {latency_ms}ms | {usage.total_tokens} tokens")
    
    return {
        "text": response.choices[0].message.content,
        "model_used": settings.LLM_MODEL,
        "backend": "grok",
        "latency_ms": latency_ms,
        "token_usage": {
            "input_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
    }

def _generate_grok_stream(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Generator[str, None, None]:
    """Streaming from Grok API."""
    client = _get_grok_client()
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    stream = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ============================================================================
# RUNPOD
# ============================================================================
async def _generate_runpod(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """Generate using RunPod."""
    from app.clients.runpod_client import runpod_client
    
    start = time.time()
    result = await runpod_client.generate_simple(
        system_instruction=system_instruction,
        user_content=user_content,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    latency_ms = int((time.time() - start) * 1000)
    logger.info(f"✅ LLM (RunPod) | {latency_ms}ms")
    
    return {
        "text": result.get("text", ""),
        "model_used": settings.RUNPOD_LLM_MODEL,
        "backend": "runpod",
        "latency_ms": latency_ms,
        "token_usage": result.get("usage", {})
    }

# ============================================================================
# PUBLIC API
# ============================================================================
async def generate(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    backend: str = None
) -> Dict[str, Any]:
    """
    Generate text - routes based on backend arg or LLM_BACKEND env var.
    
    Options: "local", "runpod", "grok"
    """
    if not backend:
        backend = settings.LLM_BACKEND.lower()
    else:
        backend = backend.lower()
    
    if backend == "local":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _generate_local, system_instruction, user_content, temperature, max_tokens
        )
    elif backend == "grok":
        # Make Grok async to prevent blocking loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _generate_grok, system_instruction, user_content, temperature, max_tokens
        )
    else:
        return await _generate_runpod(system_instruction, user_content, temperature, max_tokens)

async def generate_stream(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    backend: str = None
):
    """Streaming generate (ASYNC) - routes based on backend or LLM_BACKEND."""
    if not backend:
        backend = settings.LLM_BACKEND.lower()
    else:
        backend = backend.lower()
    
    if backend == "grok":
        # Grok stream is sync, so we iterate
         for token in _generate_grok_stream(system_instruction, user_content, temperature, max_tokens):
            yield token
            await asyncio.sleep(0) # Yield control
            
    elif backend == "local":
        # Local doesn't support true streaming, simulate
        result = await generate(system_instruction, user_content, temperature, max_tokens)
        text = result["text"]
        for i in range(0, len(text), 10):
            yield text[i:i+10]
            await asyncio.sleep(0.01)
    else:
        # RunPod doesn't support true streaming, simulate
        result = await _generate_runpod(system_instruction, user_content, temperature, max_tokens)
        text = result["text"]
        for i in range(0, len(text), 10):
            yield text[i:i+10]
            await asyncio.sleep(0.01)

def generate_stream_sync(
    system_instruction: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    backend: str = None
) -> Generator[str, None, None]:
    """
    Streaming generate (SYNC) - for backward compatibility.
    Use this if your caller uses regular 'for' loop.
    """
    if not backend:
        backend = settings.LLM_BACKEND.lower()
    else:
        backend = backend.lower()
    
    if backend == "grok":
        # Grok stream is already sync
        for token in _generate_grok_stream(system_instruction, user_content, temperature, max_tokens):
            yield token
    elif backend == "local":
        # Generate full response then yield chunks
        # Handle async loop invocation
        try:
             # If called from sync context, this works
             result = _generate_local(system_instruction, user_content, temperature, max_tokens)
        except:
             result = {"text": "Error local generation"}

        text = result.get("text", "")
        for i in range(0, len(text), 10):
            yield text[i:i+10]
    else:
        # RunPod - run async in sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_generate_runpod(system_instruction, user_content, temperature, max_tokens))
        finally:
            loop.close()
        
        text = result.get("text", "")
        for i in range(0, len(text), 10):
            yield text[i:i+10]

def get_current_backend() -> str:
    """Get current LLM backend."""
    return settings.LLM_BACKEND.lower()

def get_current_model() -> str:
    """Get current model name."""
    backend = settings.LLM_BACKEND.lower()
    if backend == "grok":
        return settings.LLM_MODEL
    return settings.RUNPOD_LLM_MODEL

async def check_connection() -> bool:
    """Check LLM backend health."""
    backend = settings.LLM_BACKEND.lower()
    try:
        if backend == "grok":
            # Just do a dry run or user info check, but generate is safest
            # Use small tokens to be cheap
            await generate("Test", "Hi", max_tokens=1)
            return True
        elif backend == "local":
            _get_local_model()
            return True
        else:
            from app.clients.runpod_client import runpod_client
            return await runpod_client.check_health()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
