"""
Chunker Service - Hierarchical Document Chunking
=================================================

ARCHITECTURE:
- SECTION chunks (parent_block_id = null)
    └── PARAGRAPH/TABLE chunks (parent_block_id = section_id)
            └── SENTENCE chunks (content_chunk_id = paragraph_id OR table_id)

Tables are SPLIT into MAX 3 search chunks to balance recall & storage.
"""
import logging
import uuid
import json
import re
import math
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Hierarchical chunker for document processing.
    
    Creates 3 levels:
    1. SECTION - from blocks
    2. PARAGRAPH/TABLE - from content items
    3. SENTENCE - from sentences within paragraphs OR table splits
    """
    
    def __init__(self, chunk_strategy: str = "hierarchical"):
        self.chunk_strategy = chunk_strategy
        logger.info(f"✅ Hierarchical Chunker initialized")
    
    def chunk_document(self, healed_json: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main chunking method.
        
        Returns:
            {
                "search_chunks": [...],      # SENTENCE chunks (for vector search)
                "content_chunks": [...],     # PARAGRAPH/TABLE chunks
                "parent_chunks": [...],      # SECTION chunks
                "document_overview": [...],  # Doc-level
                "stats": {...}
            }
        """
        doc_id = healed_json.get("doc_id", str(uuid.uuid4()))
        doc_metadata = healed_json.get("metadata", {})
        blocks = healed_json.get("blocks", [])
        
        logger.info(f"📄 START CHUNKING: {doc_metadata.get('source_filename', 'Unknown')}")
        
        search_chunks = []
        content_chunks = []
        parent_chunks = []
        
        stats = {
            "sections": 0,
            "paragraphs": 0,
            "tables": 0,
            "sentences": 0,
            "sentences_with_bbox": 0
        }
        
        for block in blocks:
            # 1. Create SECTION chunk -> parent_chunks
            section_chunk = self._create_section_chunk(block, doc_id)
            parent_chunks.append(section_chunk)
            stats["sections"] += 1
            
            section_id = section_chunk["chunk_id"]
            
            # Process content items
            for item in block.get("content", []):
                item_type = item.get("type", "unknown")
                
                # ============================================================
                # PARAGRAPH LOGIC
                # ============================================================
                if item_type == "paragraph":
                    para_chunk = self._create_paragraph_chunk(
                        item, doc_id, section_id, block
                    )
                    if para_chunk:
                        content_chunks.append(para_chunk)
                        stats["paragraphs"] += 1
                        
                        para_id = para_chunk["chunk_id"]
                        
                        sentences = item.get("sentences", [])
                        for sent in sentences:
                            sent_chunk = self._create_sentence_chunk(
                                sent, doc_id, section_id, para_id, block
                            )
                            if sent_chunk:
                                search_chunks.append(sent_chunk)
                                stats["sentences"] += 1
                                if sent.get("bbox_info"):
                                    stats["sentences_with_bbox"] += 1
                
                # ============================================================
                # TABLE LOGIC (SPLIT MAX 3 CHUNKS)
                # ============================================================
                elif item_type == "table":
                    table_chunk = self._create_table_chunk(
                        item, doc_id, section_id, block
                    )
                    
                    if table_chunk:
                        content_chunks.append(table_chunk)
                        stats["tables"] += 1
                        
                        table_id = table_chunk["chunk_id"]
                        table_text = item.get("text", "")
                        
                        if table_text:
                            rows = [r for r in table_text.split('\n') if r.strip()]
                            
                            if rows:
                                target_chunks = 3
                                batch_size = math.ceil(len(rows) / target_chunks)
                                batch_size = max(1, int(batch_size))
                                
                                row_batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
                                
                                logger.debug(f" 🔪 Table split: {len(rows)} rows -> {len(row_batches)} chunks")

                                for batch in row_batches:
                                    batch_text = "\n".join(batch)
                                    
                                    pseudo_sent = {
                                        "text": batch_text, 
                                        "bbox_info": item.get("bbox_info")
                                    }
                                    
                                    table_search_chunk = self._create_sentence_chunk(
                                        sent=pseudo_sent,
                                        doc_id=doc_id,
                                        section_id=section_id,
                                        paragraph_id=table_id,
                                        block=block
                                    )
                                    
                                    if table_search_chunk:
                                        table_search_chunk["metadata"]["is_table_representation"] = True
                                        table_search_chunk["metadata"]["is_partial_table"] = True
                                        table_search_chunk["metadata"]["rows_covered"] = len(batch)
                                        
                                        search_chunks.append(table_search_chunk)
                                        stats["sentences"] += 1 
                                        if pseudo_sent.get("bbox_info"):
                                            stats["sentences_with_bbox"] += 1
        
        doc_overview = self._create_document_overview(healed_json, doc_id)
        
        logger.info(f"📊 CHUNKING COMPLETE:")
        logger.info(f"   SECTION (parent): {stats['sections']}")
        logger.info(f"   PARAGRAPH/TABLE (content): {stats['paragraphs'] + stats['tables']}")
        logger.info(f"   SENTENCE (search): {stats['sentences']} ({stats['sentences_with_bbox']} with bbox)")
        
        return {
            "search_chunks": search_chunks,
            "content_chunks": content_chunks,
            "parent_chunks": parent_chunks,
            "document_overview": [doc_overview] if doc_overview else [],
            "stats": stats
        }
    
    # ==================== INTENT SUMMARY ====================
    
    def enrich_with_intent_summary(
        self,
        search_chunks: List[Dict],
        content_chunks: List[Dict]
    ) -> List[Dict]:
        """For late chunking: intent_summary = NULL."""
        for sent_chunk in search_chunks:
            sent_chunk["intent_summary"] = None
        logger.info(f"📝 Set intent_summary = NULL for {len(search_chunks)} sentences")
        return search_chunks
    
    def _parse_llm_json_response(self, response_text: str) -> Dict[str, str]:
        """Parse LLM JSON response with fallback."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
            if json_match:
                try: return json.loads(json_match.group())
                except: pass
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try: return json.loads(json_match.group())
                except: pass
            logger.warning(f"⚠️ Failed to parse LLM response")
            return {}
    
    # ==================== SECTION CHUNKS ====================
    
    def _create_section_chunk(self, block: Dict, doc_id: str) -> Dict[str, Any]:
        """Create SECTION chunk from block."""
        block_id = block.get("id", str(uuid.uuid4()))
        title = block.get("title", "Untitled Section")
        breadcrumb = block.get("breadcrumb", [])
        
        all_texts = []
        all_highlights = []
        
        for item in block.get("content", []):
            text = item.get("text", "")
            if text:
                all_texts.append(text)
            
            bbox_info = item.get("bbox_info")
            if bbox_info:
                all_highlights.append({
                    "page_no": bbox_info.get("page_no"),
                    "bbox": bbox_info.get("bbox"),
                    "text_snippet": text[:80] if text else ""
                })
            
            for sent in item.get("sentences", []):
                sent_bbox = sent.get("bbox_info")
                if sent_bbox:
                    all_highlights.append({
                        "page_no": sent_bbox.get("page_no"),
                        "bbox": sent_bbox.get("bbox"),
                        "text_snippet": sent.get("text", "")[:80]
                    })
        
        raw_text = "\n\n".join(all_texts)
        section_context = self._build_section_context(title, breadcrumb)
        
        return {
            "chunk_id": block_id,
            "doc_id": doc_id,
            "parent_block_id": None,
            "content_chunk_id": None,
            "raw_text": raw_text,
            "section_context": section_context,
            "chunk_type": "section",
            "chunk_strategy": self.chunk_strategy,
            "highlights": all_highlights,
            "intent_summary": None,
            "metadata": {
                "title": title,
                "level": block.get("level", 1),
                "breadcrumb": breadcrumb,
                "content_count": len(block.get("content", [])),
                "word_count": len(raw_text.split()),
                "char_count": len(raw_text)
            }
        }
    
    # ==================== PARAGRAPH CHUNKS ====================
    
    def _create_paragraph_chunk(self, item: Dict, doc_id: str, section_id: str, block: Dict) -> Optional[Dict[str, Any]]:
        """Create PARAGRAPH chunk from content item."""
        text = item.get("text", "")
        if not text or not text.strip(): return None
        if len(text.strip()) < 10: return None
        
        chunk_id = str(uuid.uuid4())
        title = block.get("title", "")
        breadcrumb = block.get("breadcrumb", [])
        section_context = self._build_section_context(title, breadcrumb)
        
        highlights = []
        bbox_info = item.get("bbox_info")
        if bbox_info:
            highlights.append({
                "page_no": bbox_info.get("page_no"),
                "bbox": bbox_info.get("bbox"),
                "text_snippet": text[:80]
            })
        
        sentences = item.get("sentences", [])
        sentences_with_bbox = sum(1 for s in sentences if s.get("bbox_info"))
        
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "parent_block_id": section_id,
            "content_chunk_id": None,
            "raw_text": text,
            "section_context": section_context,
            "chunk_type": "paragraph",
            "chunk_strategy": self.chunk_strategy,
            "highlights": highlights,
            "intent_summary": None,
            "metadata": {
                "title": title,
                "word_count": len(text.split()),
                "char_count": len(text),
                "sentence_count": len(sentences),
                "sentences_with_bbox": sentences_with_bbox,
                "page_no": bbox_info.get("page_no") if bbox_info else None
            }
        }
    
    # ==================== SENTENCE CHUNKS ====================
    
    def _create_sentence_chunk(self, sent: Dict, doc_id: str, section_id: str, paragraph_id: str, block: Dict) -> Optional[Dict[str, Any]]:
        """Create SENTENCE chunk from sentence data."""
        text = sent.get("text", "")
        if not text or not text.strip(): return None
        
        chunk_id = str(uuid.uuid4())
        title = block.get("title", "")
        breadcrumb = block.get("breadcrumb", [])
        section_context = self._build_section_context(title, breadcrumb)
        
        highlights = []
        bbox_info = sent.get("bbox_info")
        if bbox_info:
            highlights.append({
                "page_no": bbox_info.get("page_no"),
                "bbox": bbox_info.get("bbox"),
                "text_snippet": text[:80]
            })
        
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "parent_block_id": section_id,
            "content_chunk_id": paragraph_id,
            "raw_text": text,
            "section_context": section_context,
            "chunk_type": "sentence",
            "chunk_strategy": self.chunk_strategy,
            "highlights": highlights,
            "intent_summary": None,
            "metadata": {
                "title": title,
                "word_count": len(text.split()),
                "char_count": len(text),
                "match_confidence": sent.get("match_confidence"),
                "has_bbox": bbox_info is not None,
                "page_no": bbox_info.get("page_no") if bbox_info else None
            }
        }
    
    # ==================== TABLE CHUNKS ====================
    
    def _create_table_chunk(self, item: Dict, doc_id: str, section_id: str, block: Dict) -> Optional[Dict[str, Any]]:
        """Create TABLE chunk from table content item."""
        text = item.get("text", "")
        if not text or not text.strip(): return None
        
        chunk_id = str(uuid.uuid4())
        title = block.get("title", "")
        breadcrumb = block.get("breadcrumb", [])
        section_context = self._build_section_context(title, breadcrumb)
        
        highlights = []
        bbox_info = item.get("bbox_info")
        if bbox_info:
            highlights.append({
                "page_no": bbox_info.get("page_no"),
                "bbox": bbox_info.get("bbox"),
                "text_snippet": f"[TABLE] {text[:60]}..."
            })
        
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "parent_block_id": section_id,
            "content_chunk_id": None,
            "raw_text": text,
            "section_context": section_context,
            "chunk_type": "table",
            "chunk_strategy": self.chunk_strategy,
            "highlights": highlights,
            "intent_summary": None,
            "metadata": {
                "title": title,
                "word_count": len(text.split()),
                "char_count": len(text),
                "page_no": bbox_info.get("page_no") if bbox_info else None,
                "is_table": True
            }
        }
    
    # ==================== DOCUMENT OVERVIEW ====================
    
    def _create_document_overview(self, healed_json: Dict, doc_id: str) -> Optional[Dict[str, Any]]:
        """Create document overview chunk."""
        metadata = healed_json.get("metadata", {})
        filename = metadata.get("source_filename", "Unknown")
        
        return {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "parent_block_id": None,
            "content_chunk_id": None,
            "raw_text": f"DOCUMENT: {filename}",
            "section_context": f"Document Overview | {filename}",
            "chunk_type": "document_overview",
            "chunk_strategy": self.chunk_strategy,
            "highlights": [],
            "intent_summary": None,
            "metadata": {
                "filename": filename,
                "is_overview": True,
                "total_pages": metadata.get("page_count", 0)
            }
        }
    
    # ==================== HELPERS ====================
    
    def _build_section_context(self, title: str, breadcrumb: List[str]) -> str:
        """Build section context string from breadcrumb."""
        if breadcrumb:
            path = " > ".join(breadcrumb[-3:])
            if title and title not in path:
                return f"{path} > {title}"
            return path
        elif title:
            return title
        else:
            return "Document Content"


# ==================== BACKWARDS COMPATIBLE ALIAS ====================
ThreeLayerChunker = HierarchicalChunker


# ==================== CONVENIENCE FUNCTION ====================

def chunk_healed_json(healed_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to chunk a healed JSON document."""
    chunker = HierarchicalChunker()
    return chunker.chunk_document(healed_json)
