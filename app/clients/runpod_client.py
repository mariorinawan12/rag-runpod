"""
RunPod Serverless Client
========================

Client to call RunPod serverless endpoints for LLM, Reranker, and Embedding.
Uses async polling (or sync endpoint) for serverless requests.
"""
import logging
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class RunPodClient:
    """
    Client for RunPod Serverless Endpoint.
    Handles LLM Generation, Embeddings, and Reranking.
    """
    
    def __init__(self):
        self.api_key = settings.RUNPOD_API_KEY
        self.endpoint_id = settings.RUNPOD_ENDPOINT_ID
        self.timeout = 120  # seconds
        
        if not self.api_key or not self.endpoint_id:
            logger.warning("⚠️ RunPod API Key or Endpoint ID not set!")
            
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _post_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send sync request to RunPod (wait for result)."""
        url = f"{self.base_url}/runsync"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Add input wrapper if not present, RunPod standard
                if "input" not in payload:
                    payload = {"input": payload}
                    
                async with session.post(url, json=payload, headers=self.headers, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"❌ RunPod API Error {resp.status}: {text}")
                        raise Exception(f"RunPod Error: {text}")
                    
                    data = await resp.json()
                    
                    if "error" in data:
                         logger.error(f"❌ RunPod Job Error: {data['error']}")
                         raise Exception(f"RunPod Job Error: {data['error']}")
                         
                    return data
            except asyncio.TimeoutError:
                logger.error("❌ RunPod request timed out")
                raise Exception("RunPod request timed out")
            except Exception as e:
                logger.error(f"❌ RunPod Connection Error: {e}")
                raise

    async def check_health(self) -> bool:
        """Ping endpoint to check health."""
        try:
             # run health action
             payload = {"action": "health_check"}
             await self._post_sync(payload)
             return True
        except:
             # Fallback
             try:
                 await self.embed("test")
                 return True
             except:
                 return False

    async def generate_simple(
        self, 
        system_instruction: str, 
        user_content: str,
        max_tokens: int = 1000,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Generate text using RunPod LLM."""
        payload = {
            "action": "generate",
            "system_prompt": system_instruction,
            "prompt": user_content, 
            "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        data = await self._post_sync(payload)
        output = data.get("output", {})
        
        return {
            "text": output.get("text", ""),
            "usage": output.get("usage", {})
        }

    async def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        payload = {
            "action": "embed",
            "text": text
        }
        data = await self._post_sync(payload)
        return data.get("output", {}).get("embedding", [])
        
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        payload = {
            "action": "embed_batch",
            "texts": texts
        }
        data = await self._post_sync(payload)
        return data.get("output", {}).get("embeddings", [])

    async def embed_with_tokens(self, text: str) -> Dict[str, Any]:
        """
        Embed text with token-level output for late chunking.
        Returns: {
            "token_embeddings": [[...], ...],
            "offset_mapping": [[0, 3], ...]
        }
        """
        payload = {
            "action": "embed",
            "text": text,
            "return_tokens": True  # Flag to tell worker to return detailed tokens
        }
        data = await self._post_sync(payload)
        output = data.get("output", {})
        return {
            "token_embeddings": output.get("token_embeddings", []),
            "offset_mapping": output.get("offset_mapping", [])
        }

    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents given a query."""
        payload = {
            "action": "rerank",
            "query": query,
            "documents": documents
        }
        data = await self._post_sync(payload)
        return data.get("output", {}).get("scores", [])

# Global instance
runpod_client = RunPodClient()
