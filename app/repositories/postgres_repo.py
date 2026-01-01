"""
PostgreSQL Repository V9 - ASYNC HYBRID (Vector + FTS)
Schema: rag.chunks
======================================================
UPGRADE: Added Async Wrappers for Parallel Retrieval
"""
import logging
import json
import asyncio
from typing import List, Dict, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Pool executor khusus untuk Database I/O agar tidak rebutan dengan CPU/GPU task
# max_workers=10 cukup untuk handle parallel hybrid search (Vector + FTS)
_db_executor = ThreadPoolExecutor(max_workers=10)
_conn = None


def get_connection():
    global _conn
    
    if _conn is not None:
        try:
            with _conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            logger.warning("⚠️ Connection lost, reconnecting...")
            _conn = None
    
    if _conn is None:
        from app.config import settings
        logger.info("🔌 Connecting to PostgreSQL (rag schema)...")
        _conn = psycopg2.connect(
            settings.DATABASE_URL,
            options="-c search_path=rag",
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        _conn.autocommit = True
        logger.info("✅ Connected to PostgreSQL")
    
    return _conn


def get_cursor():
    conn = get_connection()
    return conn.cursor(cursor_factory=RealDictCursor)


# ============================================================================
# ASYNC WRAPPERS (FOR PARALLEL EXECUTION)
# ============================================================================

async def search_chunks_vector_async(query_embedding: List[float], doc_ids: List[str], top_k: int = 70, chunk_types: Optional[List[str]] = None) -> List[Dict]:
    """Async wrapper untuk vector search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_db_executor, search_chunks_vector, query_embedding, doc_ids, top_k, chunk_types)

async def search_chunks_fts_async(query: str, doc_ids: List[str], top_k: int = 30, chunk_types: Optional[List[str]] = None) -> List[Dict]:
    """Async wrapper untuk full-text search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_db_executor, search_chunks_fts, query, doc_ids, top_k, chunk_types)

async def get_chunks_by_ids_async(chunk_ids: List[str]) -> List[Dict]:
    """Async wrapper untuk mengambil chunks berdasarkan ID."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_db_executor, get_chunks_by_ids, chunk_ids)

async def get_highlights_by_chunk_ids_async(chunk_ids: List[str]) -> List[Dict]:
    """Async wrapper untuk mengambil highlights."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_db_executor, get_highlights_by_chunk_ids, chunk_ids)


# ==================== UPSERT (WITH INTENT_SUMMARY) ====================

def upsert_chunks(chunks: list, doc_id: str):
    """
    Upsert chunks with DUAL VECTORS + INTENT_SUMMARY.
    NOTE: tsv column is auto-generated (GENERATED ALWAYS AS), no need to insert!
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    sql = """
        INSERT INTO rag.chunks(
            chunk_id, doc_id, parent_block_id, content_chunk_id,
            raw_text, section_context, intent_summary,
            raw_vector, section_vector,
            chunk_type, chunk_strategy, metadata, highlights,
            is_embedded
        ) VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            raw_text = EXCLUDED.raw_text,
            section_context = EXCLUDED.section_context,
            intent_summary = EXCLUDED.intent_summary,
            raw_vector = EXCLUDED.raw_vector,
            section_vector = EXCLUDED.section_vector,
            is_embedded = EXCLUDED.is_embedded,
            metadata = EXCLUDED.metadata,
            highlights = EXCLUDED.highlights
    """
    
    batch_size = 100
    total = 0
    embedded_count = 0
    intent_count = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        values = []
        for chunk in batch:
            highlights = chunk.get("highlights", [])
            highlights_json = json.dumps(highlights) if highlights else '[]'
            
            raw_emb = chunk.get("raw_embedding")
            section_emb = chunk.get("section_embedding")
            is_embedded = True if (raw_emb and len(raw_emb) > 0) else False
            
            if raw_emb and len(raw_emb) > 0:
                raw_vector_str = f'[{",".join(map(str, raw_emb))}]'
                embedded_count += 1
            else:
                raw_vector_str = None
            
            if section_emb and len(section_emb) > 0:
                section_vector_str = f'[{",".join(map(str, section_emb))}]'
            else:
                section_vector_str = None
            
            intent_summary = chunk.get("intent_summary")
            if intent_summary:
                intent_count += 1
            
            values.append((
                chunk["chunk_id"],
                doc_id,
                chunk.get("parent_block_id"),
                chunk.get("content_chunk_id"),
                chunk.get("raw_text", ""),
                chunk.get("section_context", ""),
                intent_summary,
                raw_vector_str,
                section_vector_str,
                chunk.get("chunk_type", ""),
                chunk.get("chunk_strategy", "hierarchical_v7"),
                json.dumps(chunk.get("metadata", {})),
                highlights_json,
                is_embedded
            ))
        
        execute_values(
            cursor, 
            sql, 
            values,
            template="(%s, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s::vector, %s::vector, %s, %s, %s, %s::jsonb, %s)"
        )
        
        conn.commit()
        total += len(batch)
        logger.info(f"📤 Upserted batch {i//batch_size + 1}: {len(batch)} chunks")
    
    logger.info(f"✅ Total upserted: {total} chunks for doc_id={doc_id}")
    logger.info(f"   🔢 With vectors: {embedded_count} | With intent: {intent_count}")
    cursor.close()
    return total


# ==================== VECTOR SEARCH ====================

def search_chunks_vector(
    query_embedding: List[float],
    doc_ids: List[str],
    top_k: int = 70,
    chunk_types: Optional[List[str]] = None
) -> List[Dict]:
    """Vector search using raw_vector (cosine similarity)."""
    cursor = get_cursor()
    vector_str = f'[{",".join(map(str, query_embedding))}]'
    
    sql = """
        SELECT 
            chunk_id, doc_id, parent_block_id, content_chunk_id,
            raw_text, section_context, intent_summary,
            chunk_type, metadata
        FROM rag.chunks
        WHERE doc_id = ANY(%s::uuid[])
        AND content_chunk_id IS NOT NULL
        AND raw_vector IS NOT NULL
    """
    
    params = [doc_ids]
    
    if chunk_types:
        sql += " AND chunk_type = ANY(%s)"
        params.append(chunk_types)
    
    sql += " ORDER BY raw_vector <=> %s::vector LIMIT %s"
    params.extend([vector_str, top_k])
    
    try:
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        matches = []
        for row in results:
            matches.append({
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "parent_block_id": str(row["parent_block_id"]) if row["parent_block_id"] else None,
                "content_chunk_id": str(row["content_chunk_id"]) if row["content_chunk_id"] else None,
                "raw_text": row["raw_text"] or "",
                "section_context": row["section_context"] or "",
                "intent_summary": row["intent_summary"] or "",
                "chunk_type": row["chunk_type"],
                "metadata": row["metadata"] or {}
            })
        
        logger.info(f"🔍 Vector search: {len(matches)} results")
        cursor.close()
        return matches
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        cursor.close()
        return []


# ==================== FTS SEARCH (TRILINGUAL + WEBSEARCH OR) ====================

def search_chunks_fts(
    query: str,
    doc_ids: List[str],
    top_k: int = 30,
    chunk_types: Optional[List[str]] = None
) -> List[Dict]:
    """
    Full-Text Search using trilingual tsv column.
    """
    cursor = get_cursor()
    
    words = [w.strip() for w in query.split() if len(w.strip()) > 1]
    if not words:
        logger.warning(f"⚠️ FTS: No valid words in query '{query}'")
        return []
    
    or_query = ' or '.join(words)
    logger.debug(f"🔍 FTS websearch query: '{or_query}'")
    
    sql = """
        SELECT 
            chunk_id, doc_id, parent_block_id, content_chunk_id,
            raw_text, section_context, intent_summary,
            chunk_type, metadata,
            (
                COALESCE(ts_rank(tsv, websearch_to_tsquery('english', %s)), 0) +
                COALESCE(ts_rank(tsv, websearch_to_tsquery('indonesian', %s)), 0) +
                COALESCE(ts_rank(tsv, websearch_to_tsquery('simple', %s)), 0)
            ) as rank
        FROM rag.chunks
        WHERE doc_id = ANY(%s::uuid[])
        AND content_chunk_id IS NOT NULL
        AND (
            tsv @@ websearch_to_tsquery('english', %s)
            OR tsv @@ websearch_to_tsquery('indonesian', %s)
            OR tsv @@ websearch_to_tsquery('simple', %s)
        )
    """
    
    params = [or_query, or_query, or_query, doc_ids, or_query, or_query, or_query]
    
    if chunk_types:
        sql += " AND chunk_type = ANY(%s)"
        params.append(chunk_types)
    
    sql += """
        ORDER BY rank DESC
        LIMIT %s
    """
    params.append(top_k)
    
    try:
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        matches = []
        for row in results:
            matches.append({
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "parent_block_id": str(row["parent_block_id"]) if row["parent_block_id"] else None,
                "content_chunk_id": str(row["content_chunk_id"]) if row["content_chunk_id"] else None,
                "raw_text": row["raw_text"] or "",
                "section_context": row["section_context"] or "",
                "intent_summary": row["intent_summary"] or "",
                "chunk_type": row["chunk_type"],
                "metadata": row["metadata"] or {}
            })
        
        logger.info(f"📝 FTS search: {len(matches)} results for '{query}'")
        cursor.close()
        return matches
    except Exception as e:
        logger.error(f"❌ FTS search failed: {e}")
        cursor.close()
        return []


# ==================== LEGACY ALIAS ====================

def search_chunks(
    query_embedding: List[float],
    doc_ids: List[str],
    top_k: int = 150,
    chunk_types: Optional[List[str]] = None
) -> List[Dict]:
    """Legacy alias for vector search (backwards compatibility)."""
    return search_chunks_vector(query_embedding, doc_ids, top_k, chunk_types)


# ==================== DEFERRED EMBEDDING ====================

def get_unembedded_chunks_for_bulk(doc_ids: List[str]) -> List[Dict]:
    """Get sentence chunks that need embedding."""
    cursor = get_cursor()
    sql = """
        SELECT chunk_id, raw_text, section_context, intent_summary, content_chunk_id
        FROM rag.chunks 
        WHERE doc_id = ANY(%s::uuid[]) 
        AND chunk_type = 'sentence'
        AND is_embedded = false
    """
    cursor.execute(sql, [doc_ids])
    results = cursor.fetchall()
    cursor.close()
    return [{
        **dict(r),
        "chunk_id": str(r["chunk_id"]),
        "content_chunk_id": str(r["content_chunk_id"]) if r["content_chunk_id"] else None
    } for r in results]


def get_chunks_by_ids(chunk_ids: List[str]) -> List[Dict]:
    """Get chunks by their IDs."""
    if not chunk_ids:
        return []
    
    cursor = get_cursor()
    sql = """
        SELECT chunk_id, raw_text, section_context, chunk_type
        FROM rag.chunks 
        WHERE chunk_id = ANY(%s::uuid[])
    """
    cursor.execute(sql, [chunk_ids])
    results = cursor.fetchall()
    cursor.close()
    return [{
        "chunk_id": str(r["chunk_id"]),
        "raw_text": r["raw_text"] or "",
        "section_context": r["section_context"] or "",
        "chunk_type": r["chunk_type"] or ""
    } for r in results]


def update_chunk_vectors(updates: List[Dict]):
    """Batch update vectors."""
    conn = get_connection()
    cursor = conn.cursor()
    
    sql = """
        UPDATE rag.chunks AS c SET
            raw_vector = v.raw_v::vector,
            is_embedded = true
        FROM (VALUES %s) AS v(id, raw_v)
        WHERE c.chunk_id = v.id::uuid
    """
    values = []
    for u in updates:
        raw_v = f'[{",".join(map(str, u["raw_vector"]))}]'
        values.append((u["chunk_id"], raw_v))
    
    execute_values(cursor, sql, values)
    conn.commit()
    cursor.close()
    logger.info(f"✅ Updated vectors for {len(updates)} chunks")


# ==================== HIGHLIGHTS ====================

def get_highlights_by_chunk_ids(chunk_ids: List[str]) -> List[Dict]:
    if not chunk_ids:
        return []
    
    cursor = get_cursor()
    sql = """
        SELECT chunk_id, chunk_type, highlights
        FROM rag.chunks
        WHERE chunk_id = ANY(%s::uuid[])
    """
    
    try:
        cursor.execute(sql, [chunk_ids])
        results = cursor.fetchall()
        cursor.close()
        
        return [{
            "chunk_id": str(row["chunk_id"]),
            "chunk_type": row["chunk_type"] or "",
            "highlights": row["highlights"] or []
        } for row in results]
        
    except Exception as e:
        logger.error(f"❌ Get highlights failed: {e}")
        cursor.close()
        return []


def get_highlights_by_chunk_ids_with_fallback(chunk_ids: List[str]) -> List[Dict]:
    """Get highlights with parent fallback."""
    if not chunk_ids:
        return []
    
    cursor = get_cursor()
    
    sql = """
        SELECT chunk_id, parent_block_id, chunk_type, highlights
        FROM rag.chunks
        WHERE chunk_id = ANY(%s::uuid[])
    """
    
    try:
        cursor.execute(sql, [chunk_ids])
        chunk_results = cursor.fetchall()
        
        output = []
        parent_ids_needed = set()
        chunks_needing_fallback = []
        
        for row in chunk_results:
            highlights = row["highlights"] or []
            
            if highlights:
                output.append({
                    "chunk_id": str(row["chunk_id"]),
                    "chunk_type": row["chunk_type"] or "",
                    "highlights": highlights,
                    "source": "direct"
                })
            else:
                chunks_needing_fallback.append({
                    "chunk_id": str(row["chunk_id"]),
                    "parent_block_id": str(row["parent_block_id"]) if row["parent_block_id"] else None,
                    "chunk_type": row["chunk_type"] or ""
                })
                if row["parent_block_id"]:
                    parent_ids_needed.add(str(row["parent_block_id"]))
        
        parent_map = {}
        if parent_ids_needed:
            sql_parents = """
                SELECT parent_block_id, highlights
                FROM rag.chunks
                WHERE parent_block_id = ANY(%s::uuid[])
                AND chunk_type = 'section'
            """
            cursor.execute(sql_parents, [list(parent_ids_needed)])
            for row in cursor.fetchall():
                parent_map[str(row["parent_block_id"])] = row["highlights"] or []
        
        for chunk in chunks_needing_fallback:
            pid = chunk["parent_block_id"]
            fallback = parent_map.get(pid, []) if pid else []
            output.append({
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk["chunk_type"],
                "highlights": fallback,
                "source": "parent_fallback" if fallback else "none"
            })
        
        cursor.close()
        return output
        
    except Exception as e:
        logger.error(f"❌ Get highlights with fallback failed: {e}")
        cursor.close()
        return []


# ==================== DELETE ====================

def delete_by_doc_id(doc_id: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM rag.chunks WHERE doc_id = %s::uuid", [doc_id])
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️ Deleted {deleted} chunks for doc_id={doc_id}")
        cursor.close()
        return deleted
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
        cursor.close()
        return 0


# ==================== HEALTH CHECK ====================

def check_connection() -> bool:
    try:
        cursor = get_cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(raw_vector) as with_vectors,
                COUNT(intent_summary) as with_intent
            FROM rag.chunks
        """)
        result = cursor.fetchone()
        logger.info(f"📊 Chunks: {result['total']} total | {result['with_vectors']} vectors | {result['with_intent']} intent")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False
