"""
Python RAG Service - Hybrid Backend (Local GPU / RunPod Serverless)
"""
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import ingest_router, ask_router
from app.routes.query_routes import router as query_router  # NEW
from app.repositories import postgres_repo

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-ENGINE")


def preload_models():
    """Pre-load all ML models into GPU memory."""
    logger.info("📦 Pre-loading models into GPU...")
    
    # 1. Embedding model
    logger.info("   Loading embedding model...")
    from app.services.embedding_service import _get_local_model
    _get_local_model()
    
    # 2. Reranker model
    logger.info("   Loading reranker model...")
    from app.services.reranking_service import _get_local_reranker
    _get_local_reranker()
    
    # 3. Docling PDF Processor (Warm Up Singleton)
    logger.info("   Warming up PDF Processor (Docling)...")
    from app.services.pdf_processor import get_global_processor
    get_global_processor() # Ini akan load heavy models saat startup
    
    logger.info("✅ All models loaded and ready!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("🚀 Starting RAG Engine...")
    logger.info(f"   Inference Backend: {settings.INFERENCE_BACKEND}")
    logger.info(f"   LLM Backend: {settings.LLM_BACKEND}")
    logger.info(f"   Device: {settings.DEVICE}")
    
    if settings.INFERENCE_BACKEND == "local":
        # PRE-LOAD ALL MODELS ON STARTUP
        preload_models()
    else:
        logger.info(f"   RunPod Endpoint: {settings.RUNPOD_ENDPOINT_ID}")
        # Tetep warm up docling karena itu jalan di lokal (Python App) bukan di RunPod
        logger.info("   Warming up PDF Processor (Docling)...")
        from app.services.pdf_processor import get_global_processor
        get_global_processor()
    
    # Test DB connection
    logger.info("🔌 Testing database connection...")
    if postgres_repo.check_connection():
        logger.info("✅ Database connected!")
    else:
        logger.warning("⚠️ Database connection failed!")
    
    logger.info("✅ RAG Engine ready!")
    
    yield
    
    # SHUTDOWN
    logger.info("👋 Shutting down RAG Engine...")


app = FastAPI(
    title="Python RAG Service",
    description="Hybrid RAG Engine - Local GPU or RunPod Serverless",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(query_router)  # NEW


@app.get("/health")
async def health_check():
    """Health check with backend info."""
    db_ok = postgres_repo.check_connection()
    
    # Check inference backend
    if settings.INFERENCE_BACKEND == "runpod":
        from app.clients import runpod_client
        inference_ok = await runpod_client.check_health()
    else:
        inference_ok = True  # Local always "available"
    
    return {
        "status": "healthy" if (db_ok and inference_ok) else "degraded",
        "inference_backend": settings.INFERENCE_BACKEND,
        "llm_backend": settings.LLM_BACKEND,
        "device": settings.DEVICE,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "database_connected": db_ok,
        "inference_healthy": inference_ok
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
