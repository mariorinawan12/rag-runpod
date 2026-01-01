# RAG Dual-Query Architecture Refactor

## Files di folder ini

| File | Action | Description |
|------|--------|-------------|
| `query_routes.py` | **NEW** | Copy ke `app/routes/query_routes.py` |
| `query_controller.py` | **NEW** | Copy ke `app/controllers/query_controller.py` |
| `instructions_addon.py` | **MERGE** | Tambahkan ke `app/utils/instructions.py` |
| `builders_addon.py` | **MERGE** | Tambahkan ke `app/utils/builders.py` |
| `ask_routes_updated.py` | **REPLACE** | Ganti `app/routes/ask_routes.py` |
| `rag_pipeline_updated.py` | **REPLACE** | Ganti `app/pipelines/rag_pipeline.py` |
| `main_addon.py` | **MERGE** | Tambahkan ke `app/main.py` |

## Quick Start

1. Copy file NEW ke lokasi yang sesuai
2. Merge file ADDON ke file existing
3. Replace file yang perlu diganti
4. Restart server

## Test Endpoints

```bash
# Test /query/process
curl -X POST http://localhost:8000/query/process \
  -H "Content-Type: application/json" \
  -d '{
    "current_query": "Bagian A itu apa?",
    "conversation_history": [
      {"role": "user", "content": "Apa isi dokumen ini?"},
      {"role": "assistant", "content": "Dokumen berisi: Bagian A: Pendahuluan ML, Bagian B: Metodologi..."}
    ]
  }'

# Test /ask/stream dengan dual query
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "rag",
    "original_query": "Bagian A itu apa?",
    "retrieval_query": "Jelaskan detail Bagian A: Pendahuluan tentang Machine Learning",
    "doc_ids": ["doc-123"],
    "conversation_history": [
      {"role": "user", "content": "Apa isi dokumen?"},
      {"role": "assistant", "content": "Dokumen berisi..."}
    ]
  }'
```

## Flow Diagram

```
Node.js Orchestrator:

1. User kirim message
   │
   ▼
2. POST /query/process
   │  Request: { current_query, conversation_history }
   │  Response: { is_ambiguous, is_transform, retrieval_query, original_query }
   │
   ▼
3. POST /ask/stream
   │  Request: { original_query, retrieval_query, doc_ids, conversation_history }
   │  Response: Streaming tokens + sources
   │
   ▼
4. Save to DB & return to user
```
