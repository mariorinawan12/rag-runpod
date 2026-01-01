# app/routes/query_routes.py
#
# NEW: Query processing endpoint for classify + rewrite
#

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Literal

from app.controllers import query_controller

router = APIRouter(prefix="/query", tags=["Query Processing"])


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryProcessRequest(BaseModel):
    """Request untuk /query/process endpoint."""
    current_query: str
    conversation_history: Optional[List[ConversationMessage]] = None


class QueryProcessResponse(BaseModel):
    """Response dari /query/process endpoint."""
    is_ambiguous: bool
    is_transform: bool
    answer_source: Literal["document", "history", "both"] = "document"
    retrieval_query: str
    original_query: str


@router.post("/process", response_model=QueryProcessResponse)
async def process_query(request: QueryProcessRequest):
    """
    Classify query dan generate optimized retrieval query.
    
    - is_ambiguous: Apakah query butuh context untuk dipahami
    - is_transform: Apakah user mau manipulasi/reformat data
    - retrieval_query: Query yang sudah di-rewrite untuk search
    - original_query: Query asli user
    """
    return await query_controller.process_query(request)
