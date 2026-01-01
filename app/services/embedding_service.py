"""
Late Embedding Service - SOTA Span-Based Pooling with Hybrid Backend
=====================================================================

Features:
1. Late Chunking (Ingestion): 
   - Uses LOCAL SOTA Span-Based Pooling logic.
   - RunPod implementation fetches token embeddings remotely but performs pooling LOCALLY.
   - Ensures precise character-to-token alignment regardless of backend.
   
2. Query Embedding (Retrieval):
   - Offloads to RunPod if configured.

Backend Switch via `settings.INFERENCE_BACKEND` ("local" vs "runpod").
"""
import logging
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union

from app.config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Get local embedding model (lazy load) - used for Local mode."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"🧠 Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.DEVICE
        )
        if settings.DEVICE == "cuda":
            _model.half()
        logger.info(f"✅ Local model loaded on {settings.DEVICE}")
    return _model


def _get_tokenizer():
    """Get the tokenizer from the local model."""
    model = _get_model()
    return model.tokenizer

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip."""
    import re
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================================
# SPAN FINDING & POOLING LOGIC (SHARED)
# ============================================================================

def _find_sentence_char_spans_with_indices(
    paragraph_text: str, 
    sentence_texts: List[str]
) -> List[Tuple[int, Tuple[int, int]]]:
    """Find character positions of each sentence in paragraph."""
    results = []
    seen_cursors: Dict[str, int] = {}
    para_normalized = _normalize_whitespace(paragraph_text)
    
    for orig_idx, sent_text in enumerate(sentence_texts):
        if not sent_text.strip():
            results.append((orig_idx, (0, 0)))
            continue
        
        sent_key = sent_text.strip()
        sent_normalized = _normalize_whitespace(sent_text)
        search_start = seen_cursors.get(sent_normalized, 0)
        
        idx = paragraph_text.find(sent_text, search_start)
        found_len = len(sent_text)
        
        if idx == -1:
            idx = paragraph_text.find(sent_key, search_start)
            found_len = len(sent_key)
            
        if idx == -1:
            idx = paragraph_text.find(sent_key, 0)
            found_len = len(sent_key)
        
        if idx == -1:
            norm_idx = para_normalized.find(sent_normalized, 0)
            if norm_idx != -1:
                ratio = norm_idx / max(len(para_normalized), 1)
                idx = int(ratio * len(paragraph_text))
                found_len = len(sent_normalized)
            else:
                logger.warning(f"⚠️ Sentence not found: '{sent_text[:40]}...'")
                results.append((orig_idx, (0, len(sent_text))))
                continue
        
        char_start = idx
        char_end = min(idx + found_len, len(paragraph_text))
        results.append((orig_idx, (char_start, char_end)))
        seen_cursors[sent_normalized] = char_end
    
    return results


def _find_sentence_char_spans(paragraph_text: str, sentence_texts: List[str]) -> List[Tuple[int, int]]:
    indexed_spans = _find_sentence_char_spans_with_indices(paragraph_text, sentence_texts)
    indexed_spans.sort(key=lambda x: x[1][0])
    return [span for _, span in indexed_spans]


def _get_sentence_order_mapping(paragraph_text: str, sentence_texts: List[str]) -> List[int]:
    indexed_spans = _find_sentence_char_spans_with_indices(paragraph_text, sentence_texts)
    indexed_spans.sort(key=lambda x: x[1][0])
    return [orig_idx for orig_idx, _ in indexed_spans]


def _map_char_spans_to_token_spans(
    char_spans: List[Tuple[int, int]],
    token_offsets: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    token_spans = []
    for char_start, char_end in char_spans:
        if char_start == char_end:
            token_spans.append((0, 0))
            continue
        token_start = None
        token_end = None
        for tok_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
            if tok_char_start == 0 and tok_char_end == 0 and tok_idx > 0: continue
            if tok_char_end > char_start and tok_char_start < char_end:
                if token_start is None: token_start = tok_idx
                token_end = tok_idx + 1
        
        if token_start is None: token_spans.append((0, 0))
        else: token_spans.append((token_start, token_end))
    return token_spans


def _pool_tokens_to_vectors(
    token_embeddings: np.ndarray,
    token_spans: List[Tuple[int, int]],
    fallback_vec: Optional[List[float]] = None
) -> List[List[float]]:
    """Pool token embeddings for each sentence span."""
    vectors = []
    num_tokens = len(token_embeddings)
    
    for token_start, token_end in token_spans:
        if token_start >= token_end or token_start >= num_tokens:
            if fallback_vec: vectors.append(fallback_vec)
            elif vectors: vectors.append(vectors[-1])
            else: vectors.append([0.0] * token_embeddings.shape[1])
            continue
        
        token_end = min(token_end, num_tokens)
        chunk_tokens = token_embeddings[token_start:token_end]
        chunk_vec = np.mean(chunk_tokens, axis=0)
        
        norm = np.linalg.norm(chunk_vec)
        if norm > 0: chunk_vec = chunk_vec / norm
        vectors.append(chunk_vec.tolist())
    
    return vectors


# ============================================================================
# LOCAL IMPLEMENTATION
# ============================================================================

def _late_chunk_paragraph_local(
    paragraph_text: str,
    sentence_texts: List[str]
) -> List[List[float]]:
    if not paragraph_text or not sentence_texts: return []
    
    model = _get_model()
    try:
        import torch
        tokenized = model.tokenizer(
            paragraph_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        token_offsets = tokenized["offset_mapping"][0].tolist()
        
        result = model.encode(
            paragraph_text,
            output_value="token_embeddings",
            convert_to_numpy=False,
            show_progress_bar=False
        )
        
        # Unpack result
        if isinstance(result, dict):
            encoded = result.get("token_embeddings", result.get("last_hidden_state"))
        elif isinstance(result, torch.Tensor):
            encoded = result
        else:
            encoded = result
            
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        if len(encoded.shape) == 3:
            encoded = encoded[0]
            
    except Exception as e:
        logger.warning(f"⚠️ Local Late chunking failed: {e}")
        simple_vec = model.encode(paragraph_text, convert_to_numpy=True, normalize_embeddings=True)
        return [simple_vec.tolist()] * len(sentence_texts)
    
    char_spans = _find_sentence_char_spans(paragraph_text, sentence_texts)
    token_spans = _map_char_spans_to_token_spans(char_spans, token_offsets)
    
    return _pool_tokens_to_vectors(encoded, token_spans)


# ============================================================================
# RUNPOD IMPLEMENTATION
# ============================================================================

async def _late_chunk_paragraph_runpod(
    paragraph_text: str,
    sentence_texts: List[str]
) -> List[List[float]]:
    """Late chunk using RunPod for embeddings, Local for pooling."""
    if not paragraph_text or not sentence_texts: return []
    
    try:
        from app.clients.runpod_client import runpod_client
        
        # Get tokens + offsets from RunPod
        result = await runpod_client.embed_with_tokens(paragraph_text)
        token_embeddings = np.array(result["token_embeddings"])
        # offsets from JSON are list of lists, convert to tuples if needed?
        # Python list of lists [ [0,3], ... ] works fine with our logic
        token_offsets = result["offset_mapping"]
        
        if len(token_embeddings) == 0:
            raise ValueError("Empty embeddings from RunPod")
            
    except Exception as e:
        logger.warning(f"⚠️ RunPod Late chunking failed: {e}")
        # Fallback to simple? No, late chunking logic breaks without tokens.
        # Fallback to local logic if model is loaded? Or return empty.
        # Let's return local fallback if possible or just simple vector
        try:
             # Just use simple embedding as fallback for all sents
             vec = await runpod_client.embed(paragraph_text)
             return [vec] * len(sentence_texts)
        except:
             return []

    # Local pooling
    char_spans = _find_sentence_char_spans(paragraph_text, sentence_texts)
    token_spans = _map_char_spans_to_token_spans(char_spans, token_offsets)
    
    return _pool_tokens_to_vectors(token_embeddings, token_spans)


# ============================================================================
# PUBLIC API
# ============================================================================

def late_chunk_paragraph(
    paragraph_text: str,
    sentence_texts: List[str]
) -> List[List[float]]:
    """
    Late chunk: embed paragraph → split into sentence vectors.
    """
    backend = settings.INFERENCE_BACKEND.lower()
    
    if backend == "local":
        return _late_chunk_paragraph_local(paragraph_text, sentence_texts)
    else:
        # Run sync wrapper for async RunPod
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                _late_chunk_paragraph_runpod(paragraph_text, sentence_texts)
            )
        except Exception as e:
            logger.error(f"RunPod wrapper failed: {e}")
            return _late_chunk_paragraph_local(paragraph_text, sentence_texts)
        finally:
            loop.close()


def assign_late_vectors_to_sentences(
    content_chunks: List[Dict],
    search_chunks: List[Dict]
) -> List[Dict]:
    """Late chunk all paragraphs and assign vectors."""
    paragraphs = [c for c in content_chunks if c.get("chunk_type") == "paragraph"]
    if not paragraphs: return search_chunks
    
    logger.info(f"🔮 Late chunking {len(paragraphs)} paragraphs...")
    
    para_sentences = {}
    for sent in search_chunks:
        para_id = sent.get("content_chunk_id")
        if para_id:
            if para_id not in para_sentences: para_sentences[para_id] = []
            para_sentences[para_id].append(sent)
            
    enriched_count = 0
    for para in paragraphs:
        para_id = para["chunk_id"]
        sentences = para_sentences.get(para_id, [])
        if not sentences: continue
        
        para_text = para.get("raw_text", "")
        sentence_texts = [s.get("raw_text", "") for s in sentences]
        
        # Get Vectors (Local or RunPod)
        vectors = late_chunk_paragraph(para_text, sentence_texts)
        order_mapping = _get_sentence_order_mapping(para_text, sentence_texts)
        
        if len(vectors) == len(sentences) and len(order_mapping) == len(sentences):
            for sorted_idx, orig_idx in enumerate(order_mapping):
                if sorted_idx < len(vectors) and orig_idx < len(sentences):
                    sentences[orig_idx]["late_vector"] = vectors[sorted_idx]
                    enriched_count += 1
        else:
            for sent, vec in zip(sentences, vectors):
                sent["late_vector"] = vec
                enriched_count += 1
                
    return search_chunks


def embed_query(text: str) -> List[float]:
    """Embed query via Local or RunPod."""
    text = _normalize_whitespace(text)
    backend = settings.INFERENCE_BACKEND.lower()
    
    if backend == "runpod":
        loop = asyncio.new_event_loop()
        try:
            from app.clients.runpod_client import runpod_client
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(runpod_client.embed(text))
        except Exception:
            pass # Fallthrough
        finally:
            loop.close()
            
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


async def embed_texts_async(texts: List[str], is_query: bool = True) -> List[List[float]]:
    """Async embedding via Local or RunPod."""
    backend = settings.INFERENCE_BACKEND.lower()
    normalized_texts = [_normalize_whitespace(t) for t in texts]
    
    if backend == "runpod":
        from app.clients.runpod_client import runpod_client
        return await runpod_client.embed_batch(normalized_texts)
        
    model = _get_model()
    loop = asyncio.get_event_loop()
    def _embed_batch():
        return model.encode(normalized_texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
    return await loop.run_in_executor(None, _embed_batch)
