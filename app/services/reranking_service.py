import logging
import asyncio
import torch
import numpy as np
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from app.config import settings

logger = logging.getLogger(__name__)

# Optimal workers untuk GPU task biasanya rendah (1-4) agar tidak terjadi CUDA context switching
_executor = ThreadPoolExecutor(max_workers=4) 

# ============================================================================
# LOCAL RERANKER (SOTA CUDA FP16)
# ============================================================================
_local_reranker = None

def _get_local_reranker():
    """Get local CrossEncoder reranker dengan optimasi FP16."""
    global _local_reranker
    if _local_reranker is None:
        from sentence_transformers import CrossEncoder
        
        # Load model langsung ke device target
        _local_reranker = CrossEncoder(
            settings.RERANKER_MODEL_NAME,
            device=settings.DEVICE
        )
        
        # AKTIVASI FP16: Paksa model ke Half-Precision untuk RTX/CUDA Cores
        if settings.DEVICE == "cuda":
            _local_reranker.model.half()
            logger.info(f"🚀 CUDA FP16 Activated for {settings.RERANKER_MODEL_NAME}")
        
        logger.info(f"✅ Local reranker ready on {settings.DEVICE}")
    return _local_reranker

def _rerank_local_core(query: Optional[str], documents: Optional[List[str]], pairs: Optional[List[List[str]]] = None) -> List[float]:
    """Core logic reranking dengan inference_mode untuk speed boosing."""
    if not pairs and (not query or not documents):
        return []
    
    # Build pairs jika belum ada
    input_pairs = pairs if pairs else [[query, doc] for doc in documents]
    
    reranker = _get_local_reranker()
    
    # 🔥 SOTA OPTIMIZATION: Gunakan inference_mode agar tidak hitung gradient (Hemat VRAM & Speedup)
    with torch.inference_mode():
        # Batch size otomatis diatur oleh CrossEncoder, tapi kita pastikan convert ke list
        scores = reranker.predict(
            input_pairs, 
            batch_size=32,       # Optimal batch untuk FP16
            convert_to_tensor=True # Tetap di GPU selama mungkin
        )
        
        # Pindahkan ke CPU hanya saat final return
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
            
    return scores.tolist()

# ============================================================================
# PUBLIC API - ASYNC (THE FASTEST PATH)
# ============================================================================

async def rerank_async(query: str, documents: List[str]) -> List[float]:
    """Rerank async dengan auto-backend selection."""
    if not query or not documents: return []
    
    backend = settings.INFERENCE_BACKEND.lower()
    if backend == "local":
        loop = asyncio.get_event_loop()
        # Jalankan di executor agar tidak memblokir event loop utama
        return await loop.run_in_executor(_executor, _rerank_local_core, query, documents)
    else:
        from app.clients import runpod_client
        return await runpod_client.rerank(query, documents)

async def rerank_pairs_async(pairs: List[List[str]]) -> List[float]:
    """Rerank via pairs (Async)."""
    if not pairs: return []
    
    backend = settings.INFERENCE_BACKEND.lower()
    if backend == "local":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _rerank_local_core, None, None, pairs)
    else:
        from app.clients import runpod_client
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        return await runpod_client.rerank(query, documents)

# ============================================================================
# COMPATIBILITY WRAPPER (DROP-IN FOR OLD CODE)
# ============================================================================

def get_reranker():
    """Wrapper agar tetap kompatibel dengan reranker.predict(pairs)."""
    class HybridReranker:
        def predict(self, pairs: List[List[str]]) -> np.ndarray:
            # Kita jalankan secara sinkron untuk compatibility
            backend = settings.INFERENCE_BACKEND.lower()
            if backend == "local":
                scores = _rerank_local_core(None, None, pairs)
            else:
                # Fallback blocking untuk RunPod (Tidak disarankan tapi jalan)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply() # Izin panggil loop dalam loop
                from app.clients import runpod_client
                query = pairs[0][0]
                documents = [p[1] for p in pairs]
                scores = loop.run_until_complete(runpod_client.rerank(query, documents))
            
            return np.array(scores)
    
    return HybridReranker()
