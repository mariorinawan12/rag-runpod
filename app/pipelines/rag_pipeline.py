# app/pipelines/rag_pipeline.py - UPDATED VERSION
#
# Replace existing rag_pipeline.py dengan versi ini
#

import logging
import time
import json
from typing import Dict, Any, Generator
import asyncio

from app.services import retrieval_service
from app.services.llm_service import generate, generate_stream_sync
from app.utils import (
    decompose_query,
    RAG_INSTRUCTION,
)
from app.utils.builders import build_rag_prompt, format_conversation_history

logger = logging.getLogger(__name__)


def run_sync(coro):
    """Helper untuk menjalankan coroutine di dalam fungsi sinkron tanpa bentrok loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import threading
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    else:
        return loop.run_until_complete(coro)


# ============================================================================
# NON-STREAMING PIPELINE - UPDATED
# ============================================================================

async def run_rag_pipeline(request) -> Dict[str, Any]:
    """RAG Pipeline: decompose → retrieve → rerank → generate"""
    
    # DUAL QUERY: Separate queries for retrieval vs generation
    retrieval_query = request.get_retrieval_query()
    original_query = request.get_original_query()
    
    doc_ids = request.doc_ids or []
    
    # Format conversation history
    if request.conversation_history:
        history_dicts = []
        for msg in request.conversation_history:
            if hasattr(msg, 'dict'):
                history_dicts.append(msg.dict())
            elif hasattr(msg, 'model_dump'):
                history_dicts.append(msg.model_dump())
            else:
                history_dicts.append(msg)
        conversation_history = format_conversation_history(history_dicts)
    else:
        conversation_history = request.conversation_context or ""
    
    if not doc_ids:
        raise ValueError("RAG pipeline requires doc_ids")
    
    try:
        # DECOMPOSE - use retrieval query
        sub_queries = await decompose_query(retrieval_query)
        all_queries = [retrieval_query] + [q for q in sub_queries if q.lower() != retrieval_query.lower()]
        
        # RETRIEVE - use retrieval query
        retrieval_result = retrieval_service.complex_pipeline_search(
            original_query=retrieval_query,
            sub_queries=all_queries,
            doc_ids=doc_ids
        )
        
        context = retrieval_result.get("context", "")
        sources = retrieval_result.get("sources", [])
        confidence = retrieval_result.get("confidence", 0)

        logger.info(f"📥 Retrieval query: '{retrieval_query[:50]}...'")
        logger.info(f"📥 Original query: '{original_query[:50]}...'")
        
        if context and confidence > 0:
            # GENERATE - use original query + history
            user_content = build_rag_prompt(
                original_query,      # Original untuk LLM
                context,
                conversation_history  # Injected history
            )

            result = await generate(
                system_instruction=RAG_INSTRUCTION,
                user_content=user_content
            )
            answer = result["text"]
            token_usage = result["token_usage"]
        else:
            answer = "Maaf, tidak ditemukan informasi yang relevan dalam dokumen."
            token_usage = {}
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "token_usage": token_usage,
            "pipeline": "rag",
            "stats": retrieval_result.get("stats", {})
        }
        
    except Exception as e:
        logger.error(f"❌ RAG FAILED: {str(e)}", exc_info=True)
        raise


# ============================================================================
# STREAMING PIPELINE - UPDATED
# ============================================================================

def run_rag_pipeline_stream(request) -> Generator[str, None, None]:
    """
    RAG Pipeline with streaming output - UPDATED dengan answer_source routing.
    
    Yields NDJSON lines:
    - {"type": "status", "data": "..."}
    - {"type": "token", "data": "Hello"}
    - {"type": "sources", "data": [...]}
    - {"type": "done"}
    - {"type": "error", "data": "..."}
    """
    import re
    
    logger.info("=" * 70)
    logger.info("🚀 RAG PIPELINE STREAM - START")
    logger.info("=" * 70)
    
    # Get queries and answer_source
    retrieval_query = request.get_retrieval_query()
    original_query = request.get_original_query()
    answer_source = request.get_answer_source()  # NEW: "document" | "history" | "both"
    
    doc_ids = request.doc_ids or []
    
    # Format conversation history
    if request.conversation_history:
        history_dicts = []
        for msg in request.conversation_history:
            if hasattr(msg, 'dict'):
                history_dicts.append(msg.dict())
            elif hasattr(msg, 'model_dump'):
                history_dicts.append(msg.model_dump())
            else:
                history_dicts.append(msg)
        conversation_history = format_conversation_history(history_dicts)
    else:
        conversation_history = request.conversation_context or ""
    
    logger.info(f"📥 INPUT:")
    logger.info(f"   retrieval_query='{retrieval_query[:50]}...'")
    logger.info(f"   original_query='{original_query[:50]}...'")
    logger.info(f"   answer_source='{answer_source}'")  # NEW
    logger.info(f"   docs={len(doc_ids)}")
    logger.info(f"   has_history={bool(conversation_history)}")
    
    try:
        # ========== ROUTE BASED ON ANSWER_SOURCE ==========
        
        if answer_source == "history":
            # ========== PATH 1: HISTORY ONLY ==========
            # Skip retrieval, answer from conversation history
            logger.info("📍 PATH: HISTORY ONLY - Skip retrieval")
            
            yield json.dumps({"type": "status", "data": "Preparing answer from conversation..."}) + "\n"
            
            # Build prompt WITHOUT document context
            user_content = build_rag_prompt(
                original_query,
                "",  # No document context
                conversation_history,
                answer_source="history"
            )
            
            try:
                generated_answer = []
                for token in generate_stream_sync(
                    system_instruction=RAG_INSTRUCTION,
                    user_content=user_content
                ):
                    if token is not None and isinstance(token, str):
                        yield json.dumps({"type": "token", "data": token}) + "\n"
                        generated_answer.append(token)
                
                # No sources for history-only
                yield json.dumps({
                    "type": "sources",
                    "data": [],
                    "confidence": 1.0,
                    "answer_source": "history"
                }) + "\n"
                
            except Exception as e:
                logger.error(f"❌ LLM stream failed: {e}")
                yield json.dumps({"type": "error", "data": str(e)}) + "\n"
            
            yield json.dumps({"type": "done", "pipeline": "rag", "answer_source": "history"}) + "\n"
            logger.info("✅ RAG Stream SUCCESS (history only)")
            logger.info("=" * 70)
            return
        
        # ========== PATH 2 & 3: DOCUMENT or BOTH ==========
        # Full retrieval pipeline
        logger.info(f"📍 PATH: {answer_source.upper()} - Full retrieval")
        
        # Validate input
        if not retrieval_query or not retrieval_query.strip():
            yield json.dumps({"type": "error", "data": "Query cannot be empty"}) + "\n"
            return
            
        if not doc_ids:
            yield json.dumps({"type": "error", "data": "RAG pipeline requires doc_ids"}) + "\n"
            return
        
        yield json.dumps({"type": "status", "data": "Retrieving documents..."}) + "\n"
        
        # ========== DECOMPOSE QUERY ==========
        try:
            sub_queries = run_sync(decompose_query(retrieval_query))
            all_queries = [retrieval_query] + [q for q in sub_queries if q.lower() != retrieval_query.lower()]
            logger.info(f"📝 Decomposed: {len(all_queries)} queries")
        except Exception as e:
            logger.warning(f"⚠️ Decompose failed: {e}")
            all_queries = [retrieval_query]
        
        # ========== RETRIEVE ==========
        try:
            retrieval_result = retrieval_service.complex_pipeline_search(
                original_query=retrieval_query,
                sub_queries=all_queries,
                doc_ids=doc_ids
            )
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            yield json.dumps({"type": "error", "data": f"Retrieval failed: {str(e)}"}) + "\n"
            return
        
        # ========== EXTRACT RESULTS ==========
        if not isinstance(retrieval_result, dict):
            retrieval_result = {}
        
        context = retrieval_result.get("context") or ""
        sources = retrieval_result.get("sources") or []
        confidence = retrieval_result.get("confidence") or 0
        stats = retrieval_result.get("stats") or {}

        logger.info(f"📄 Context length: {len(context)} chars")
        
        if not isinstance(sources, list):
            sources = []
        
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        
        # ========== GENERATE ANSWER ==========
        filtered_sources = []
        
        if context and confidence > 0:
            yield json.dumps({"type": "status", "data": "Generating answer..."}) + "\n"
            
            # Build prompt with answer_source
            user_content = build_rag_prompt(
                original_query,
                context,
                conversation_history,
                answer_source=answer_source  # Pass answer_source for dynamic rules
            )

            logger.info(f"📝 Generation using original_query: '{original_query[:50]}...'")
            
            try:
                token_count = 0
                generated_answer = []
                
                for token in generate_stream_sync(
                    system_instruction=RAG_INSTRUCTION,
                    user_content=user_content
                ):
                    if token is not None and isinstance(token, str):
                        yield json.dumps({"type": "token", "data": token}) + "\n"
                        generated_answer.append(token)
                        token_count += 1
                
                # ========== AFTER LLM DONE: FILTER SOURCES ==========
                full_answer = "".join(generated_answer)
                
                if token_count > 0:
                    cited_indices = re.findall(r'\[S-(\d+)\]', full_answer)
                    cited_indices_int = list(set(int(x) for x in cited_indices))
                    
                    logger.info(f"🔗 LLM cited: {cited_indices_int} out of {len(sources)} sources")
                    
                    for src in sources:
                        citation_num = src.get("citation_number")
                        if citation_num in cited_indices_int:
                            filtered_sources.append(src)
                    
                    yield json.dumps({
                        "type": "sources",
                        "data": filtered_sources,
                        "confidence": round(confidence, 3),
                        "stats": stats,
                        "doc_count": len(doc_ids),
                        "total_retrieved": len(sources),
                        "total_cited": len(filtered_sources),
                        "answer_source": answer_source
                    }) + "\n"
                else:
                    yield json.dumps({
                        "type": "token", 
                        "data": "Maaf, tidak dapat menghasilkan jawaban."
                    }) + "\n"
                    yield json.dumps({
                        "type": "sources",
                        "data": [],
                        "confidence": 0,
                        "stats": stats,
                        "doc_count": len(doc_ids)
                    }) + "\n"
                    
            except Exception as e:
                logger.error(f"❌ LLM stream failed: {e}")
                yield json.dumps({
                    "type": "token", 
                    "data": f"\n\n[Error generating answer: {str(e)}]"
                }) + "\n"
                yield json.dumps({
                    "type": "sources",
                    "data": [],
                    "confidence": 0,
                    "error": True
                }) + "\n"
        else:
            yield json.dumps({
                "type": "token", 
                "data": "Maaf, tidak ditemukan informasi yang relevan dalam dokumen."
            }) + "\n"
            yield json.dumps({
                "type": "sources",
                "data": [],
                "confidence": 0,
                "stats": stats,
                "doc_count": len(doc_ids)
            }) + "\n"
        
        # ========== DONE ==========
        yield json.dumps({
            "type": "done", 
            "pipeline": "rag",
            "sources": filtered_sources,
            "answer_source": answer_source
        }) + "\n"
        
        logger.info(f"✅ RAG Stream SUCCESS")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ RAG Stream FAILED: {str(e)}", exc_info=True)
        
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        yield json.dumps({
            "type": "sources",
            "data": [],
            "confidence": 0,
            "error": True
        }) + "\n"
        yield json.dumps({
            "type": "done", "pipeline": "rag", "success": False
        }) + "\n"
