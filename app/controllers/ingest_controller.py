"""
Ingest Controller - Document Ingestion Pipeline
================================================
"""
import os
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

TEMP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "temp_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Pool executor
_process_executor = ThreadPoolExecutor(max_workers=2)

async def process_and_store_only(
    file_content: bytes, 
    filename: str, 
    doc_id: str, 
    target_oss_path: str,
    with_intent: bool = True
) -> Dict[str, Any]:
    from app.services.pdf_repair_service import repair_service
    from app.services.oss_service import oss_service
    from app.services.document_converter_service import converter_service
    from app.services.pdf_processor import process_pdf_to_healed_with_bbox
    
    # [RESTORED ORIGINAL IMPORT]
    from app.services.chunker_service import HierarchicalChunker
    
    from app.repositories import postgres_repo
    
    unique_name = f"{uuid.uuid4()}_{filename}"
    temp_raw_path = os.path.join(TEMP_UPLOAD_DIR, unique_name)
    
    temp_converted_path = None
    temp_clean_path = None
    is_file_uploaded_to_oss = False

    try:
        # A. Save raw file
        with open(temp_raw_path, "wb") as f:
            f.write(file_content)
        
        # B. Convert to PDF
        temp_converted_path = await converter_service.convert_to_pdf(temp_raw_path)

        # C. Repair/Standardize PDF
        temp_clean_path = repair_service.repair_pdf(temp_converted_path)

        if temp_clean_path == temp_converted_path:
            logger.warning("⚠️ Ghostscript Repair unchanged, using converted file")

        # D. Upload to OSS
        logger.info("✨ Uploading to OSS...")
        oss_service.upload_file(temp_clean_path, target_oss_path)
        is_file_uploaded_to_oss = True

        # E. Docling Process - NON BLOCKING WAY
        logger.info("📄 Running Docling (Async)...")
        
        loop = asyncio.get_event_loop()
        healed_json, elements_bbox = await loop.run_in_executor(
            _process_executor,
            process_pdf_to_healed_with_bbox,
            temp_clean_path
        )
        
        healed_json["doc_id"] = doc_id
        
        # F. Chunking
        logger.info("🔪 Chunking document...")
        chunker = HierarchicalChunker()
        result = chunker.chunk_document(healed_json)
        
        search_chunks = result["search_chunks"]
        content_chunks = result["content_chunks"]
        parent_chunks = result["parent_chunks"]
        doc_overview = result["document_overview"]
        stats = result["stats"]
        
        # G. Intent Enrichment
        if with_intent and len(content_chunks) > 0:
            logger.info("🔑 Enriching intent...")
            try:
                search_chunks = chunker.enrich_with_intent_summary(
                    search_chunks,
                    content_chunks
                )
                
                enriched_count = sum(1 for c in search_chunks if c.get("intent_summary"))
                stats["sentences_with_intent"] = enriched_count
                logger.info(f"✅ Intent enrichment complete: {enriched_count}/{len(search_chunks)}")
                
            except Exception as e:
                logger.error(f"⚠️ Intent enrichment failed (continuing without): {e}")
                stats["sentences_with_intent"] = 0
        else:
            stats["sentences_with_intent"] = 0
        
        # H. Combine all chunks
        all_chunks = search_chunks + content_chunks + parent_chunks + doc_overview
        
        for chunk in all_chunks:
            chunk["doc_id"] = doc_id
            chunk["raw_embedding"] = None
            chunk["section_embedding"] = None
        
        # I. Upsert to DB
        logger.info(f"💾 Upserting {len(all_chunks)} chunks to DB...")
        postgres_repo.upsert_chunks(all_chunks, doc_id)
        
        return {
            "status": "success", 
            "doc_id": doc_id, 
            "stats": stats
        }

    except Exception as e:
        if is_file_uploaded_to_oss:
            logger.warning(f"⚠️ Pipeline failed, rolling back OSS: {target_oss_path}")
            oss_service.delete_file(target_oss_path)
        raise e

    finally:
        # Cleanup temp files
        if os.path.exists(temp_raw_path): 
            os.remove(temp_raw_path)
        if temp_converted_path and os.path.exists(temp_converted_path) and temp_converted_path != temp_raw_path:
            os.remove(temp_converted_path)
        if temp_clean_path and os.path.exists(temp_clean_path) and temp_clean_path != temp_converted_path:
            os.remove(temp_clean_path)

async def run_bulk_deferred_ingest(doc_ids: List[str]) -> Dict[str, Any]:
    from app.repositories import postgres_repo
    
    # [RESTORED ORIGINAL IMPORTS]
    from app.services.late_embedding_service import late_chunk_paragraph, embed_query
    
    try:
        chunks_to_embed = postgres_repo.get_unembedded_chunks_for_bulk(doc_ids)
        
        if not chunks_to_embed:
            return {"status": "info", "message": "All documents already embedded"}

        logger.info(f"🔮 Late chunking {len(chunks_to_embed)} chunks...")
        
        para_sentences: Dict[str, List[Dict]] = {}
        for c in chunks_to_embed:
            para_id = c.get("content_chunk_id")
            if para_id:
                if para_id not in para_sentences:
                    para_sentences[para_id] = []
                para_sentences[para_id].append(c)
        
        para_chunks = postgres_repo.get_chunks_by_ids(list(para_sentences.keys()))
        para_texts = {p["chunk_id"]: p.get("raw_text", "") for p in para_chunks}
        
        updates = []
        for para_id, sentences in para_sentences.items():
            para_text = para_texts.get(para_id, "")
            if not para_text or not sentences:
                continue
            
            sentence_texts = [s.get("raw_text", "") for s in sentences]
            
            # Use restored function
            vectors = late_chunk_paragraph(para_text, sentence_texts)
            
            for sent, vec in zip(sentences, vectors):
                updates.append({
                    "chunk_id": sent["chunk_id"],
                    "raw_vector": vec
                })
        
        orphan_count = 0
        for c in chunks_to_embed:
            if not c.get("content_chunk_id"):
                raw_text = c.get("raw_text") or c.get("intent_summary") or ""
                
                # Use restored function
                updates.append({
                    "chunk_id": c["chunk_id"],
                    "raw_vector": embed_query(raw_text),
                    "section_vector": embed_query(c.get("section_context") or "")
                })
                orphan_count += 1
        
        if orphan_count:
            logger.info(f"   ℹ️ {orphan_count} orphan chunks embedded normally")
        
        postgres_repo.update_chunk_vectors(updates)
        
        return {
            "status": "success",
            "processed_docs": len(doc_ids),
            "total_embedded_chunks": len(updates),
            "message": f"Late chunked {len(updates)} chunks from {len(doc_ids)} documents"
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk ingest failed: {e}")
        raise

async def ingest_document(
    file_content: bytes, 
    filename: str, 
    doc_id: str, 
    target_oss_path: str,
    with_ingest: bool = False,
    with_intent: bool = True
) -> Dict[str, Any]:
    res = await process_and_store_only(
        file_content, filename, doc_id, target_oss_path,
        with_intent=with_intent
    )
    
    if with_ingest:
        logger.info(f"🚀 Immediate ingest for {filename}")
        bulk_res = await run_bulk_deferred_ingest([doc_id])
        embedded_count = bulk_res.get('total_embedded_chunks', 0)
        res["status"] = "success"
        res["message"] = f"Upload+Ingest complete ({embedded_count} chunks embedded)"
        res["ingest_details"] = bulk_res
    else:
        res["message"] = "Upload complete (text extracted, ready for embedding)"
        res["status"] = "parsed"
        
    return res

async def delete_document(doc_id: str) -> Dict[str, Any]:
    from app.repositories import postgres_repo
    try:
        logger.info(f"🗑️ Deleting doc_id: {doc_id}")
        deleted_count = postgres_repo.delete_by_doc_id(doc_id)
        return {
            "status": "success",
            "doc_id": doc_id,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
        raise
