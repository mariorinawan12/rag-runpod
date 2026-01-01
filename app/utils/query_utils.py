# app/utils/query_utils.py
#
# Query utilities - decompose, rewrite, dll
#

import json
import logging
from typing import List

from app.services.llm_service import generate
from app.utils.instructions import DECOMPOSE_INSTRUCTION
from app.utils.builders import build_decompose_prompt

logger = logging.getLogger(__name__)

# Config
MIN_WORDS_FOR_DECOMPOSITION = 3


async def decompose_query(query: str) -> List[str]:
    """
    Pecah query kompleks jadi sub-queries.
    
    Args:
        query: User query
    
    Returns:
        List of sub-queries (minimal 1, original query)
    """
    
    # Skip kalau query pendek
    word_count = len(query.strip().split())
    if word_count < MIN_WORDS_FOR_DECOMPOSITION:
        logger.info(f"⚡ Skip decompose: {word_count} words")
        return [query]
    
    logger.info(f'🧠 Decomposing: "{query[:50]}..."')
    
    try:
        result = await generate(
            system_instruction=DECOMPOSE_INSTRUCTION,
            user_content=build_decompose_prompt(query),
            temperature=0,
            max_tokens=300
        )
        
        # Parse JSON dari response
        text = result["text"].strip()
        text = text.replace("```json", "").replace("```", "")
        
        # Coba parse
        sub_queries = json.loads(text)
        
        # Validate
        if not isinstance(sub_queries, list) or len(sub_queries) == 0:
            raise ValueError("Invalid array")
        
        # Filter valid strings, max 4
        valid = [q for q in sub_queries if isinstance(q, str) and q.strip()][:4]
        
        if not valid:
            raise ValueError("No valid queries")
        
        logger.info(f"✅ Decomposed: {len(valid)} sub-queries")
        return valid
        
    except Exception as e:
        logger.warning(f"⚠️ Decompose failed: {e}, using original")
        return [query]
