"""
Late Embedding Service - Hybrid (Local / RunPod)
=================================================

Late Chunking: Embed paragraph → split into sentence vectors using span-based pooling.
Switch via INFERENCE_BACKEND env var: "local" or "runpod"

RunPod MODE:
- Calls RunPod for token_embeddings + offset_mapping
- Span pooling happens LOCALLY (no change to your logic!)

Local MODE:
- Everything runs on local GPU (as before)
"""
import logging
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)

# ============================================================================
# LOCAL MODEL (Lazy loaded) - Only used when INFERENCE_BACKEND=local
# ============================================================================

_model = None


def _get_model():
    """Get embedding model (lazy load)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"🧠 Loading model for late chunking: {settings.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.DEVICE
        )
        if settings.DEVICE == "cuda":
            _model.half()
        logger.info(f"✅ Model loaded on {settings.DEVICE}")
    return _model


def _get_tokenizer():
    """Get the tokenizer from the model."""
    model = _get_model()
    return model.tokenizer


# ============================================================================
# SPAN FINDING LOGIC (Always runs locally - no model needed)
# ============================================================================

def _normalize_whitespace(text: str) -> str:
    import re
    return re.sub(r'\s+', ' ', text).strip()


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
    """Map character spans to token spans using offset mapping."""
    token_spans = []
    
    for char_start, char_end in char_spans:
        if char_start == char_end:
            token_spans.append((0, 0))
            continue
        
        token_start = None
        token_end = None
        
        for tok_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
            if tok_char_start == 0 and tok_char_end == 0 and tok_idx > 0:
                continue
            
            if tok_char_end > char_start and tok_char_start < char_end:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx + 1
        
        if token_start is None:
            token_spans.append((0, 0))
        else:
            token_spans.append((token_start, token_end))
    
    return token_spans


# ============================================================================
# CORE LATE CHUNK LOGIC
# ============================================================================

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
            if fallback_vec:
                vectors.append(fallback_vec)
            elif vectors:
                vectors.append(vectors[-1])
            else:
                vectors.append([0.0] * token_embeddings.shape[1])
            continue
        
        token_end = min(token_end, num_tokens)
        chunk_tokens = token_embeddings[token_start:token_end]
        chunk_vec = np.mean(chunk_tokens, axis=0)
        
        norm = np.linalg.norm(chunk_vec)
        if norm > 0:
            chunk_vec = chunk_vec / norm
        
        vectors.append(chunk_vec.tolist())
    
    return vectors


# ============================================================================
# LOCAL IMPLEMENTATION
# ============================================================================

def _late_chunk_paragraph_local(
    paragraph_text: str,
    sentence_texts: List[str]
) -> List[List[float]]:
    """Late chunk using LOCAL model."""
    if not paragraph_text or not sentence_texts:
        return []
    
    model = _get_model()
    tokenizer = _get_tokenizer()
    
    try:
        import torch
        
        tokenized = tokenizer(
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
        logger.warning(f"⚠️ Token embedding failed: {e}, using fallback")
        single_vec = model.encode(paragraph_text, convert_to_numpy=True, normalize_embeddings=True)
        return [single_vec.tolist()] * len(sentence_texts)
    
    if len(encoded) == 0:
        single_vec = model.encode(paragraph_text, convert_to_numpy=True, normalize_embeddings=True)
        return [single_vec.tolist()] * len(sentence_texts)
    
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
    """Late chunk using RUNPOD for embedding, LOCAL for span pooling."""
    if not paragraph_text or not sentence_texts:
        return []
    
    try:
        from app.clients import runpod_client
        
        # Call RunPod for token embeddings + offset mapping
        result = await runpod_client.embed_with_tokens(paragraph_text)
        token_embeddings = np.array(result["token_embeddings"])
        token_offsets = result["offset_mapping"]
        
        if len(token_embeddings) == 0:
            # Fallback to simple embedding
            simple_vec = await runpod_client.embed(paragraph_text)
            return [simple_vec] * len(sentence_texts)
        
    except Exception as e:
        logger.warning(f"⚠️ RunPod embed failed: {e}")
        # Could fallback to local or return empty
        return []
    
    # Span mapping and pooling happens LOCALLY (your logic unchanged!)
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
    Routes based on INFERENCE_BACKEND.
    """
    backend = settings.INFERENCE_BACKEND.lower()
    
    if backend == "local":
        return _late_chunk_paragraph_local(paragraph_text, sentence_texts)
    else:
        # RunPod is async, run in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _late_chunk_paragraph_runpod(paragraph_text, sentence_texts)
            )
        finally:
            loop.close()


async def late_chunk_paragraph_async(
    paragraph_text: str,
    sentence_texts: List[str]
) -> List[List[float]]:
    """Async version of late_chunk_paragraph."""
    backend = settings.INFERENCE_BACKEND.lower()
    
    if backend == "local":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            _late_chunk_paragraph_local,
            paragraph_text,
            sentence_texts
        )
    else:
        return await _late_chunk_paragraph_runpod(paragraph_text, sentence_texts)


def assign_late_vectors_to_sentences(
    content_chunks: List[Dict],
    search_chunks: List[Dict]
) -> List[Dict]:
    """
    Late chunk all paragraphs and assign vectors to their sentences.
    Works with both local and RunPod backends.
    """
    paragraphs = [c for c in content_chunks if c.get("chunk_type") == "paragraph"]
    
    if not paragraphs:
        logger.warning("⚠️ No paragraphs for late chunking")
        return search_chunks
    
    backend = settings.INFERENCE_BACKEND.lower()
    logger.info(f"🔮 Late chunking {len(paragraphs)} paragraphs (backend={backend})...")
    
    para_sentences: Dict[str, List[Dict]] = {}
    for sent in search_chunks:
        para_id = sent.get("content_chunk_id")
        if para_id:
            if para_id not in para_sentences:
                para_sentences[para_id] = []
            para_sentences[para_id].append(sent)
    
    enriched_count = 0
    for para in paragraphs:
        para_id = para["chunk_id"]
        para_text = para.get("raw_text", "")
        sentences = para_sentences.get(para_id, [])
        
        if not sentences:
            continue
        
        sentence_texts = [s.get("raw_text", "") for s in sentences]
        order_mapping = _get_sentence_order_mapping(para_text, sentence_texts)
        vectors = late_chunk_paragraph(para_text, sentence_texts)
        
        if len(vectors) == len(sentences) and len(order_mapping) == len(sentences):
            for sorted_idx, orig_idx in enumerate(order_mapping):
                if sorted_idx < len(vectors) and orig_idx < len(sentences):
                    sentences[orig_idx]["late_vector"] = vectors[sorted_idx]
                    enriched_count += 1
        else:
            logger.warning(f"⚠️ Order mapping mismatch for {para_id}, using fallback")
            for sent, vec in zip(sentences, vectors):
                sent["late_vector"] = vec
                enriched_count += 1
    
    logger.info(f"✅ Late chunking complete: {enriched_count}/{len(search_chunks)} sentences")
    
    return search_chunks
