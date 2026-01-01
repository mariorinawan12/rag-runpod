"""
RunPod Serverless Client
========================

Client to call RunPod serverless endpoints for LLM, Reranker, and Embedding.
Uses async polling for serverless requests.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RunPodClient:
    """Client for RunPod Serverless API."""
    
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def _run_job(self, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Run a job on RunPod serverless with polling."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/run",
                json={"input": payload},
                headers=self.headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"RunPod submit failed: {error_text}")
                
                result = await response.json()
                job_id = result.get("id")
                
                if not job_id:
                    raise Exception("No job ID returned")
            
            poll_url = f"{self.base_url}/status/{job_id}"
            elapsed = 0
            poll_interval = 0.5
            
            while elapsed < timeout:
                async with session.get(poll_url, headers=self.headers) as response:
                    result = await response.json()
                    status = result.get("status")
                    
                    if status == "COMPLETED":
                        return result.get("output", {})
                    elif status == "FAILED":
                        raise Exception(f"Job failed: {result.get('error', 'Unknown error')}")
                    elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        poll_interval = min(poll_interval * 1.2, 2.0)
                    else:
                        logger.warning(f"Unknown status: {status}")
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
            
            raise TimeoutError(f"Job timed out after {timeout}s")
    
    # =========================================================================
    # LLM
    # =========================================================================
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Generate text using LLM."""
        payload = {
            "action": "generate",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        return await self._run_job(payload)
    
    async def generate_simple(
        self,
        system_instruction: str,
        user_content: str,
        max_tokens: int = 1000,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Simplified generate interface."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        return await self.generate(messages, max_tokens, temperature)
    
    # =========================================================================
    # RERANK
    # =========================================================================
    
    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents. Returns list of scores."""
        payload = {
            "action": "rerank",
            "query": query,
            "documents": documents
        }
        result = await self._run_job(payload)
        return result.get("scores", [])
    
    # =========================================================================
    # EMBEDDING
    # =========================================================================
    
    async def embed(self, text: str) -> List[float]:
        """Embed single text. Returns pooled vector."""
        payload = {
            "action": "embed",
            "text": text,
            "return_tokens": False
        }
        result = await self._run_job(payload)
        return result.get("embedding", [])
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Returns list of vectors."""
        payload = {
            "action": "embed",
            "texts": texts,
            "return_tokens": False
        }
        result = await self._run_job(payload)
        # Handle both single and batch
        if "embeddings" in result:
            return result["embeddings"]
        elif "embedding" in result:
            return [result["embedding"]]
        return []
    
    async def embed_with_tokens(self, text: str) -> Dict[str, Any]:
        """
        Embed text with token-level output for late chunking.
        
        Returns:
            {
                "token_embeddings": [[...], [...], ...],  # Per-token vectors
                "offset_mapping": [[0, 3], [3, 7], ...]   # Char spans per token
            }
        """
        payload = {
            "action": "embed",
            "text": text,
            "return_tokens": True
        }
        result = await self._run_job(payload)
        return {
            "token_embeddings": result.get("token_embeddings", []),
            "offset_mapping": result.get("offset_mapping", [])
        }
    
    async def check_health(self) -> bool:
        """Check if endpoint is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception:
            return False


# ============================================================================
# SINGLETON CLIENT
# ============================================================================

_client: Optional[RunPodClient] = None


def get_client() -> RunPodClient:
    """Get singleton RunPod client."""
    global _client
    if _client is None:
        from app.config import settings
        _client = RunPodClient(
            api_key=settings.RUNPOD_API_KEY,
            endpoint_id=settings.RUNPOD_ENDPOINT_ID
        )
        logger.info("✅ RunPod client initialized")
    return _client


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

async def generate_simple(
    system_instruction: str,
    user_content: str,
    max_tokens: int = 1000,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """Convenience wrapper for generate."""
    return await get_client().generate_simple(system_instruction, user_content, max_tokens, temperature)


async def rerank(query: str, documents: List[str]) -> List[float]:
    """Convenience wrapper for rerank."""
    return await get_client().rerank(query, documents)


async def embed(text: str) -> List[float]:
    """Convenience wrapper for embed single text."""
    return await get_client().embed(text)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convenience wrapper for embed batch."""
    return await get_client().embed_texts(texts)


async def embed_with_tokens(text: str) -> Dict[str, Any]:
    """Convenience wrapper for late embedding."""
    return await get_client().embed_with_tokens(text)


async def check_health() -> bool:
    """Convenience wrapper for health check."""
    return await get_client().check_health()

