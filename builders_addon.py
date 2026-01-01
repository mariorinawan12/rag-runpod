# app/utils/builders.py
#
# Kumpulan PROMPT BUILDERS
# Ini ngatur "APA YANG DIKIRIM" ke LLM
# Fungsi-fungsi yang build user_content
#

import json
from typing import List, Dict, Optional

# ============================================================================
# CONFIG
# ============================================================================

MIN_WORDS_FOR_DECOMPOSITION = 5


# ============================================================================
# RAG PROMPT - UPDATED dengan conversation history injection
# ============================================================================

def build_rag_prompt(
    user_query: str, 
    context_string: str, 
    conversation_history: str = "",
    answer_source: str = "document"
) -> str:
    """
    Build RAG prompt dengan dynamic source rules based on answer_source.
    
    Args:
        user_query: Original user query (untuk LLM generation)
        context_string: Retrieved context dari dokumen
        conversation_history: Formatted history string
        answer_source: "document" | "history" | "both"
    
    Returns:
        Complete prompt for RAG generation
    """
    
    parts = []
    
    # 1. Retrieved context (jika ada)
    if context_string and context_string.strip():
        parts.append(f"""DATA KONTEKS (dari dokumen):
============================================================
{context_string}
============================================================""")
    
    # 2. Conversation history (jika ada)
    if conversation_history and conversation_history.strip():
        parts.append(f"""KONTEKS PERCAKAPAN:
{conversation_history}""")
    
    # 3. Dynamic rules based on answer_source
    if answer_source == "document":
        source_rules = """ATURAN SUMBER:
- HANYA gunakan DATA KONTEKS sebagai sumber fakta
- Konteks percakapan untuk gaya bahasa saja, BUKAN sumber fakta
- Jika tidak ada di DATA KONTEKS, katakan: "Informasi tidak tersedia dalam dokumen."
- Setiap klaim WAJIB memiliki sitasi [S-X]"""
        
    elif answer_source == "history":
        source_rules = """ATURAN SUMBER:
- Ini adalah pertanyaan KLARIFIKASI tentang jawaban Anda sebelumnya
- Gunakan KONTEKS PERCAKAPAN untuk menjelaskan istilah/kategorisasi yang ANDA buat
- Jelaskan dengan jelas apa maksud dari istilah tersebut berdasarkan jawaban Anda sebelumnya
- TIDAK perlu sitasi [S-X] karena ini menjelaskan interpretasi Anda sendiri"""
        
    elif answer_source == "both":
        source_rules = """ATURAN SUMBER:
- UTAMAKAN DATA KONTEKS sebagai sumber utama
- BOLEH gunakan KONTEKS PERCAKAPAN untuk klarifikasi istilah yang Anda buat
- Jika ada konflik antara keduanya, DATA KONTEKS menang
- Setiap fakta dari dokumen WAJIB memiliki sitasi [S-X]"""
    else:
        source_rules = """ATURAN SUMBER:
- Gunakan DATA KONTEKS sebagai sumber utama"""
    
    parts.append(source_rules)
    
    # 4. User question + formatting instructions
    parts.append(f"""PERTANYAAN USER:
"{user_query}"

INSTRUKSI FORMATTING:
- Gunakan Markdown standar
- Gunakan **bold** untuk menyorot istilah penting
- Jika ada data komparatif, gunakan tabel dengan heading #### Judul""")
    
    return "\n\n".join(parts)


# ============================================================================
# QUERY PROCESS PROMPT - NEW! untuk /query/process endpoint
# ============================================================================

def build_query_process_prompt(
    current_query: str,
    conversation_history: Optional[List[dict]] = None
) -> str:
    """
    Build user content for query processing LLM call.
    
    Args:
        current_query: The user's current message
        conversation_history: List of {role, content} from previous turns
    
    Returns:
        Formatted prompt string untuk QUERY_PROCESS_INSTRUCTION
    """
    
    # Format conversation history
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        # Take last 4 messages (2 turns) max
        for msg in conversation_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_str = "\n\n".join(history_lines)
    else:
        history_str = "(no previous conversation)"
    
    # Build final prompt
    return f"""=== CONVERSATION HISTORY ===
{history_str}

=== CURRENT QUERY ===
{current_query}

Analyze and output JSON:"""


# ============================================================================
# CONVERSATION HISTORY FORMATTER - untuk RAG prompt injection
# ============================================================================

def format_conversation_history(
    history: Optional[List[dict]], 
    max_turns: int = 2
) -> str:
    """
    Format conversation history untuk injection ke RAG prompt.
    
    Args:
        history: List of {role, content} messages
        max_turns: Max turns to include (1 turn = 1 user + 1 assistant)
    
    Returns:
        Formatted string untuk prompt injection, empty string jika tidak ada
    """
    if not history:
        return ""
    
    # Take last N turns (each turn = 2 messages)
    recent = history[-(max_turns * 2):]
    
    if not recent:
        return ""
    
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    
    return "\n\n".join(lines)


# ============================================================================
# DECOMPOSE PROMPT
# ============================================================================

def build_decompose_prompt(query):
    """Build user content untuk query decomposition."""
    return f'Query: "{query}"\nJSON:'


# ============================================================================
# CLASSIFY PROMPT
# ============================================================================

def build_classify_prompt(user_message):
    """Build user content untuk intent classification."""
    return f'Klasifikasikan: "{user_message}"'


# ============================================================================
# INTENT SUMMARY PROMPT - OPTIMIZED FOR LOW TOKENS + HIGH ACCURACY
# ============================================================================

INTENT_SYSTEM_PROMPT = """Ekstrak 3-5 KEYWORD dari setiap paragraf akademik.

ATURAN:
- Keyword singkat (1-3 kata)
- Bahasa sama dengan teks
- Fokus: TOPIK, METODE, SUBJEK
- Output HANYA JSON

CONTOH OUTPUT:
{"abc12345": "stres mahasiswa, prevalensi, faktor risiko"}"""


def get_intent_system_prompt() -> str:
    """Get system prompt for intent extraction."""
    return INTENT_SYSTEM_PROMPT


def build_intent_extraction_prompt(paragraphs: List[Dict]) -> str:
    """
    OPTIMIZED: ~80-100 tokens per paragraph (vs 290 before)
    
    Format: Compact lines with short ID
    """
    lines = []
    id_mapping = {}  # Map short_id -> full chunk_id
    
    for i, p in enumerate(paragraphs):
        full_id = p["chunk_id"]
        short_id = f"P{i+1}"  # P1, P2, P3... (super short)
        id_mapping[short_id] = full_id
        
        # Take first 150 chars only (enough for context)
        text = p.get("text", "")[:150].replace("\n", " ").strip()
        lines.append(f"[{short_id}] {text}")
    
    prompt = f"""{chr(10).join(lines)}

JSON: {{"P1": "kw1, kw2", "P2": "kw3, kw4"}}"""

    # Store mapping for later use
    build_intent_extraction_prompt._id_mapping = id_mapping
    
    return prompt


def get_intent_id_mapping() -> Dict[str, str]:
    """Get the short_id -> full_id mapping from last prompt build."""
    return getattr(build_intent_extraction_prompt, '_id_mapping', {})
