"""
Ingest Routes - API endpoints untuk document ingestion
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.controllers import ingest_controller

router = APIRouter(prefix="/ingest", tags=["Ingestion"])




@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete all chunks for a document.
    
    - **doc_id**: Document ID to delete
    """
    try:
        result = await ingest_controller.delete_document(doc_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunks/texts")
async def get_chunk_texts(request: dict):
    """Get text_for_ui for array of chunk_ids."""
    chunk_ids = request.get("chunk_ids", [])
    result = await ingest_controller.get_chunk_texts(chunk_ids)
    return result


# ============== NEW: HIGHLIGHT FETCHER ==============
@router.post("/chunks/highlights")
async def get_chunk_highlights(request: dict):
    """
    Get highlights (bbox) for array of chunk_ids.
    
    Request body:
    {
        "chunk_ids": ["uuid-1", "uuid-2", ...]
    }
    
    Response:
    {
        "success": true,
        "data": [
            {
                "chunk_id": "uuid-1",
                "chunk_type": "paragraph_content",
                "highlights": [
                    {"page_no": 1, "bbox": {...}, "text_snippet": "..."}
                ]
            },
            ...
        ],
        "stats": {
            "total_chunks": 5,
            "total_highlights": 12
        }
    }
    """
    chunk_ids = request.get("chunk_ids", [])
    result = await ingest_controller.get_chunk_highlights(chunk_ids)
    return result


@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    target_oss_path: str = Form(...),
    with_ingest: bool = Form(False)
    
):
    content = await file.read()
    return await ingest_controller.ingest_document(
        file_content=content,
        filename=file.filename,
        doc_id=doc_id,
        with_ingest=with_ingest,
        target_oss_path= target_oss_path
    )

@router.post("/run-ingest")
async def trigger_bulk_deferred_ingest(request: dict):
    """
    Ingest dokumen yang sudah ter-upload berdasarkan list ID.
    Request body: { "document_ids": ["uuid-1", "uuid-2"] }
    """
    doc_ids = request.get("document_ids", [])
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    try:
        # Panggil controller untuk proses list ID
        result = await ingest_controller.run_bulk_deferred_ingest(doc_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
