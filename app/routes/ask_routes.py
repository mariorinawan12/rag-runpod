# app/routes/ask_routes.py - UPDATED VERSION
#
# Replace existing ask_routes.py dengan versi ini
#

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Literal

from app.controllers import ask_controller
from app.pipelines.rag_pipeline import run_rag_pipeline_stream
from app.pipelines.chat_pipeline import run_chat_pipeline_stream

router = APIRouter(prefix="/ask", tags=["Ask"])


# ============================================================================
# MODELS
# ============================================================================

class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: Literal["user", "assistant"]
    content: str


class QueryMetadata(BaseModel):
    """Metadata dari /query/process endpoint."""
    is_ambiguous: bool = False
    is_transform: bool = False
    answer_source: Literal["document", "history", "both"] = "document"


class AskRequest(BaseModel):
    """Request untuk /ask endpoint - UPDATED dengan dual query support."""
    pipeline: Literal["rag", "chat"]
    
    # NEW: Dual query support (dari /query/process)
    original_query: Optional[str] = None      # Untuk LLM generation
    retrieval_query: Optional[str] = None     # Untuk search/retrieval
    
    # LEGACY: Backward compatibility
    query: Optional[str] = None
    
    doc_ids: Optional[List[str]] = None
    
    # NEW: Structured conversation history
    conversation_history: Optional[List[ConversationMessage]] = None
    
    # LEGACY: Backward compatibility
    conversation_context: Optional[str] = None
    
    # NEW: Query metadata (optional, for logging/analytics)
    query_metadata: Optional[QueryMetadata] = None
    
    def get_retrieval_query(self) -> str:
        """Get query untuk retrieval. Prioritas: retrieval_query > query > original_query"""
        return self.retrieval_query or self.query or self.original_query or ""
    
    def get_original_query(self) -> str:
        """Get query untuk LLM generation. Prioritas: original_query > query"""
        return self.original_query or self.query or ""
    
    def get_answer_source(self) -> str:
        """Get answer_source dari query_metadata. Default: document"""
        if self.query_metadata:
            return self.query_metadata.answer_source
        return "document"


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("")
async def ask(request: AskRequest):
    """Non-streaming ask endpoint"""
    try:
        result = await ask_controller.process_request(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def ask_stream(request: AskRequest):
    """Streaming ask endpoint - supports dual queries."""
    if request.pipeline == "rag":
        return StreamingResponse(
            run_rag_pipeline_stream(request),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    elif request.pipeline == "chat":
        return StreamingResponse(
            run_chat_pipeline_stream(request),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {request.pipeline}")
