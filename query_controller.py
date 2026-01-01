# app/controllers/query_controller.py
#
# NEW: Query classification + rewrite logic
#

import json
import logging
from typing import List, Optional

import asyncio
from app.services.llm_service import generate
from app.clients.runpod_client import runpod_client
from app.utils.instructions import QUERY_PROCESS_INSTRUCTION
from app.utils.builders import build_query_process_prompt

logger = logging.getLogger(__name__)


async def process_query(request) -> dict:
    """
    Classify query dan generate optimized retrieval query.
    
    Returns:
        {
            is_ambiguous: bool,
            is_transform: bool,
            retrieval_query: str,
            original_query: str
        }
    """
    current_query = request.current_query
    history = request.conversation_history or []
    
    # Convert Pydantic models to dicts if needed
    history_dicts = []
    for msg in history:
        if hasattr(msg, 'dict'):
            history_dicts.append(msg.dict())
        elif hasattr(msg, 'model_dump'):
            history_dicts.append(msg.model_dump())
        else:
            history_dicts.append(msg)
    
    # Build prompt
    user_content = build_query_process_prompt(
        current_query=current_query,
        conversation_history=history_dicts
    )
    
    logger.info(f"🔍 Processing query: '{current_query[:50]}...'")
    
    # [WARMUP] Fire-and-forget health check to RunPod
    # This ensures RunPod is spinning up while we wait for Grok
    asyncio.create_task(runpod_client.check_health())
    logger.info("🔥 Triggered RunPod warmup...")
    
    try:
        # [ROUTING] Use GROK for Intent Detection
        result = await generate(
            system_instruction=QUERY_PROCESS_INSTRUCTION,
            user_content=user_content,
            temperature=0,
            max_tokens=300,
            backend="grok"
        )
        
        # Parse JSON from response
        text = result["text"].strip()
        
        # Clean markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        parsed = json.loads(text)
        
        response = {
            "is_ambiguous": bool(parsed.get("is_ambiguous", False)),
            "is_transform": bool(parsed.get("is_transform", False)),
            "answer_source": parsed.get("answer_source", "document"),
            "retrieval_query": parsed.get("retrieval_query", current_query),
            "original_query": current_query
        }
        
        logger.info(f"✅ Query processed: ambiguous={response['is_ambiguous']}, transform={response['is_transform']}, source={response['answer_source']}")
        logger.info(f"   Retrieval query: '{response['retrieval_query'][:50]}...'")
        
        return response
        
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON parse failed: {e}, using defaults")
        return {
            "is_ambiguous": False,
            "is_transform": False,
            "answer_source": "document",
            "retrieval_query": current_query,
            "original_query": current_query
        }
    except Exception as e:
        logger.error(f"❌ Query process failed: {e}")
        return {
            "is_ambiguous": False,
            "is_transform": False,
            "answer_source": "document",
            "retrieval_query": current_query,
            "original_query": current_query
        }
