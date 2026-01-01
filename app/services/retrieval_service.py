"""
Retrieval Service V11 - ASYNC PARALLEL HYBRID (Vector + FTS + Smart Threshold)
==============================================================================
UPGRADE: Full Asynchronous Execution for Ultra-Low Latency

PHILOSOPHY: "Concurrency is Speed"
- Parallel Embedding for all sub-queries in one batch.
- Parallel Vector + FTS search using asyncio.gather.
- Parallel fetching of paragraph text and highlights.
- CUDA FP16 Optimized Reranking.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

# Menggunakan service yang sudah mendukung async
from app.services import embedding_service
# NOTE: postgres_repo must be implemented in app/repositories/postgres_repo.py
from app.repositories import postgres_repo
from app.services.reranking_service import rerank_async

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

K_VECTOR_PER_QUERY = 70   # Vector results per sub-query
K_FTS_PER_QUERY = 30      # FTS results per sub-query
MAX_PARAGRAPHS = 12       # Maximum paragraphs to return

# Smart Threshold Config
MIN_KEEP = 3              # Always keep at least 3 paragraphs (safety)
MAX_DROP_PERCENT = 50     # Cut if drop > 50% from previous
MIN_RATIO_TO_TOP = 0.15   # Must be at least 15% of top score (safety floor)

# ============================================================================
# SMART THRESHOLD FILTER
# ============================================================================

def smart_threshold_filter(paragraphs: List[Dict]) -> List[Dict]:
    """Smart filtering based on score drops (Identical to V10 logic)."""
    if len(paragraphs) <= MIN_KEEP:
        return paragraphs
    
    scores = [p.get("rerank_score", 0) for p in paragraphs]
    top_score = scores[0]
    
    if top_score < 0.1:
        logger.info(f"⚠️ Top score low ({top_score:.3f}), keeping up to {MAX_PARAGRAPHS}")
        return paragraphs[:MAX_PARAGRAPHS]
    
    cutoff_idx = len(paragraphs)
    for i in range(MIN_KEEP, len(paragraphs)):
        prev_score = scores[i - 1]
        curr_score = scores[i]
        
        if prev_score > 0:
            drop_percent = ((prev_score - curr_score) / prev_score) * 100
            if drop_percent > MAX_DROP_PERCENT:
                logger.info(f"📉 Drop detected at [{i+1}]: {drop_percent:.1f}% drop")
                cutoff_idx = i
                break
        
        ratio_to_top = curr_score / top_score if top_score > 0 else 0
        if ratio_to_top < MIN_RATIO_TO_TOP:
            logger.info(f"📉 Low ratio at [{i+1}]: {ratio_to_top:.1%} of top")
            cutoff_idx = i
            break
            
    final_count = min(cutoff_idx, MAX_PARAGRAPHS)
    return paragraphs[:final_count]

# ============================================================================
# MAIN RETRIEVAL - ASYNC PARALLEL HYBRID
# ============================================================================

async def retrieve_and_rerank(
    query: str,
    doc_ids: List[str],
    sub_queries: Optional[List[str]] = None,
    k_vector: int = K_VECTOR_PER_QUERY,
    k_fts: int = K_FTS_PER_QUERY
) -> Dict[str, Any]:
    """
    Async parallel hybrid retrieval. 
    Reduces latency by executing all I/O bound tasks concurrently.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 ASYNC HYBRID START: '{query}'")
    logger.info(f"{'='*60}")

    # 1. Prepare unique queries (Main + Decomposed)
    all_queries = list(set(sub_queries or [query]))
    
    # ---------- STEP 1: BATCH EMBEDDING (Async) ----------
    try:
        # Menghindari loop, kirim list sekaligus ke GPU
        # Check if embedding_service supports list batching - YES IT DOES via RunPod
        q_embeddings = await embedding_service.embed_texts_async(all_queries, is_query=True)
        query_map = dict(zip(all_queries, q_embeddings))
    except Exception as e:
        logger.error(f"❌ Batch embedding failed: {e}")
        return _empty_result()

    # ---------- STEP 2: PARALLEL DATABASE SEARCH ----------
    search_tasks = []
    for q in all_queries:
        # Jalankan Vector & FTS Search secara paralel untuk setiap query
        search_tasks.append(postgres_repo.search_chunks_vector_async(
            query_map[q], doc_ids, k_vector
        ))
        search_tasks.append(postgres_repo.search_chunks_fts_async(
            q, doc_ids, k_fts
        ))

    logger.info(f"📡 Executing {len(search_tasks)} parallel DB tasks...")
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # ---------- STEP 3: ASYNC DEDUPE & MAPPING ----------
    all_sentences = {}
    for res in search_results:
        if isinstance(res, Exception):
            logger.warning(f"⚠️ Search task failed: {res}")
            continue
        for r in res:
            chunk_id = r["chunk_id"]
            if chunk_id not in all_sentences:
                all_sentences[chunk_id] = r
    
    if not all_sentences:
        logger.warning("⚠️ No results found")
        return _empty_result()

    # Group by paragraph parent (content_chunk_id)
    para_sentences = {}
    for sent in all_sentences.values():
        para_id = sent.get("content_chunk_id")
        if para_id:
            if para_id not in para_sentences:
                para_sentences[para_id] = []
            para_sentences[para_id].append(sent)

    # Fetch paragraph texts in parallel
    para_ids = list(para_sentences.keys())
    para_chunks = await postgres_repo.get_chunks_by_ids_async(para_ids)

    paragraphs = []
    for p in para_chunks:
        para_id = p["chunk_id"]
        sentences = para_sentences.get(para_id, [])
        paragraphs.append({
            "chunk_id": para_id,
            "doc_id": sentences[0]["doc_id"] if sentences else None,
            "raw_text": p.get("raw_text", ""),
            "section_context": p.get("section_context", ""),
            "sentence_count": len(sentences)
        })

    if not paragraphs:
        return _empty_result()

    # ---------- STEP 4: ASYNC RERANK (CUDA FP16) ----------
    logger.info(f"🎯 Reranking {len(paragraphs)} paragraphs (Async)...")
    
    # Build text for reranker (late chunking context)
    rerank_texts = [
        f"{p.get('section_context', '')}: {p.get('raw_text', '')}" 
        if p.get('section_context') else p.get('raw_text', '')
        for p in paragraphs
    ]

    try:
        # Memanggil fungsi async reranker yang dioptimasi CUDA FP16
        scores = await rerank_async(query, rerank_texts)
        for i, p in enumerate(paragraphs):
            p["rerank_score"] = float(scores[i])
    except Exception as e:
        logger.error(f"❌ Rerank failed: {e}")
        for p in paragraphs: p["rerank_score"] = 0.0

    # Sort and Apply Smart Threshold
    paragraphs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    final_paragraphs = smart_threshold_filter(paragraphs)

    # ---------- STEP 5: PARALLEL FORMATTING & HIGHLIGHTS ----------
    # Menarik bboxes secara asinkron untuk mempercepat output formatting
    return await _format_output_async(final_paragraphs, query)

# ============================================================================
# ASYNC FORMATTING HELPERS
# ============================================================================

async def _format_output_async(paragraphs: List[Dict], query: str) -> Dict[str, Any]:
    """Format and fetch highlights concurrently with RELEVANCE SCORE inclusion."""
    if not paragraphs:
        return _empty_result()

    # Fetch highlights for all final paragraphs in parallel
    para_ids = [p["chunk_id"] for p in paragraphs]
    hl_results = await postgres_repo.get_highlights_by_chunk_ids_async(para_ids)
    hl_map = {str(hl["chunk_id"]): hl.get("highlights", []) for hl in hl_results}

    output_parts = []
    sources = []

    for idx, para in enumerate(paragraphs):
        chunk_num = idx + 1
        score = para.get("rerank_score", 0) # Ini angka akurasi dari Cross-Encoder
        section = para.get("section_context", "")
        content = para.get("raw_text", "").strip()
        
        # Kita selipin skor akurasi di header context biar LLM juga tau seberapa valid datanya
        header = f"[S-{chunk_num}] (Akurasi: {score:.2f}) (Section: {section})" if section else f"[S-{chunk_num}] (Akurasi: {score:.2f})"
        output_parts.append(f"{header}\n{content}")
        
        chunk_id = para["chunk_id"]
        para_hl = hl_map.get(str(chunk_id), [])
        bboxes = []
        page = None
        for h in para_hl:
            if h.get("bbox"):
                bboxes.append({"page": h.get("page_no"), "bbox": h.get("bbox")})
                if page is None: page = h.get("page_no")

        # Ini yang bakal dikirim ke frontend lu
        sources.append({
            "citation_number": chunk_num,
            "chunk_id": chunk_id,
            "doc_id": para.get("doc_id"),
            "page": page,
            "bboxes": bboxes,
            "relevance_score": round(score, 4), # Akurasi tiap chunk (0.0 - 1.0)
            "accuracy_percent": f"{score * 100:.1f}%", # Versi persentase buat UI
            "section": section,
            "text_preview": content[:150] + "..."
        })

    top_score = paragraphs[0].get("rerank_score", 0)
    return {
        "context": "\n\n".join(output_parts),
        "sources": sources,
        "confidence": min(0.95, max(0.3, top_score)),
        "stats": {
            "paragraphs_returned": len(paragraphs),
            "top_score": round(top_score, 4),
            "all_scores": [round(p.get("rerank_score", 0), 4) for p in paragraphs] # List akurasi 12 chunk
        }
    }

def _empty_result() -> Dict[str, Any]:
    return {
        "context": "Tidak ada data relevan ditemukan dalam knowledge base.",
        "sources": [],
        "confidence": 0,
        "stats": {"paragraphs_returned": 0, "top_score": 0}
    }

# ============================================================================
# ALIAS FOR BACKWARDS COMPATIBILITY
# ============================================================================

async def complex_pipeline_search(
    original_query: str,
    sub_queries: Optional[List[str]],
    doc_ids: List[str]
) -> Dict[str, Any]:
    """Async entry point for existing pipelines."""
    return await retrieve_and_rerank(
        query=original_query,
        doc_ids=doc_ids,
        sub_queries=sub_queries
    )
