# Writing Assistant Backend

FastAPI backend for a multi-agent writing pipeline (plan -> draft -> review -> rewrite) with optional RAG and version history.

## Architecture Overview
- Agents: WritingAgent, ReviewerAgent, EditorAgent (hello-agents)
- Services: PlanningService, DraftingService, ReviewingService, RewritingService, WritingPipeline
- RAG: RAGService + optional Qdrant vector store (HashEmbedding fallback)
- Storage: SQLite for documents, draft versions, citations
- API: FastAPI routers for writing, pipeline, rag, citations, versions

## Project Layout
```
backend/
  app/
    agents/        # LLM agents and prompts
    api/           # FastAPI app + routers
    models/        # Pydantic schemas + entities
    services/      # Pipeline, RAG, storage, embeddings
    config.py      # Env configuration
  data/            # SQLite db file (default)
  main.py          # Uvicorn entry
  requirements.txt
  README.md
```

## Requirements
- Python 3.10+

## Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in `backend/` (or copy from `.env.example` if you have one) and set what you need.

### LLM
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_TIMEOUT` (seconds)
- `LLM_MAX_TOKENS` (max output tokens)

### LLM reliability (optional)
- `LLM_RETRY_MAX`
- `LLM_RETRY_BACKOFF` (seconds)
- `LLM_COOLDOWN_SECONDS` (seconds)

### LLM context control (recommended)
- `LLM_MAX_INPUT_CHARS` (hard cap for prompt input)
- `LLM_MAX_CONTEXT_TOKENS` (model context window)
- `LLM_INPUT_SAFETY_MARGIN` (reserved tokens to avoid overflow)
- `LLM_CHARS_PER_TOKEN` (heuristic, e.g. 0.6)
- `LLM_HISTORY_MAX_CHARS` (cap in-memory history)

### LLM context compression (optional)
- `LLM_CONTEXT_COMPRESS_ENABLED=true|false`
- `LLM_CONTEXT_COMPRESS_THRESHOLD` (trigger ratio, e.g. 0.85)
- `LLM_CONTEXT_COMPRESS_TARGET` (target ratio after compression, e.g. 0.5)
- `LLM_CONTEXT_COMPRESS_KEEP_LAST` (keep last N messages verbatim)
- `LLM_CONTEXT_COMPRESS_MAX_TOKENS` (summary max tokens)
- `LLM_CONTEXT_COMPRESS_INPUT_CHARS` (summary input cap)
- `LLM_CONTEXT_COMPRESS_MERGE_THRESHOLD` (merge summaries ratio)
- `LLM_CONTEXT_COMPRESS_MERGE_TARGET` (merged summary target ratio)

### Storage
- `STORAGE_PATH` (default `data/app.db`)

### Memory / Vector Store
- `MEMORY_MODE=short_term` (SQLite only)
- `MEMORY_MODE=long_term` (SQLite + Qdrant if configured)

Qdrant (enabled when `MEMORY_MODE=long_term` and `QDRANT_URL` is set):
- `QDRANT_URL`
- `QDRANT_API_KEY` (optional)
- `QDRANT_COLLECTION` (default `writing_memory`)
- `QDRANT_EMBED_DIM` (default 256)
- `QDRANT_DISTANCE` (cosine | dot | euclid)
- `QDRANT_TIMEOUT` (seconds, default 30)

Embeddings (used by Qdrant when `EMBEDDING_PROVIDER` is not `hash`):
- `EMBEDDING_PROVIDER` (default `openai_compatible`)
- `EMBEDDING_MODEL`
- `EMBEDDING_API_BASE` (falls back to `LLM_API_BASE` if empty)
- `EMBEDDING_API_KEY`
- `EMBEDDING_TIMEOUT` (seconds, default 20)
- `EMBEDDING_PROBE=true|false` (probe embedding dim at startup)
- `AUTO_UPDATE_ENV=true|false` (update `.env` QDRANT_EMBED_DIM when probe differs)

### Upload limits
- `UPLOAD_MAX_MB` (default 10)

### Pipeline throttling (optional)
- `PIPELINE_STAGE_SLEEP` (seconds between stages)

### Example `.env`
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_API_KEY=your_key
LLM_API_BASE=https://api.openai.com/v1
LLM_TIMEOUT=300
LLM_MAX_TOKENS=1200
LLM_MAX_INPUT_CHARS=12000
LLM_MAX_CONTEXT_TOKENS=32768
LLM_INPUT_SAFETY_MARGIN=8000
LLM_CHARS_PER_TOKEN=0.6

MEMORY_MODE=long_term
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=writing_memory
QDRANT_EMBED_DIM=256

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_API_KEY=your_key
PIPELINE_STAGE_SLEEP=3
```

## Run
```bash
python main.py
```

## Health Check
- `GET /healthz`
- `GET /healthz/detail`
- `GET /healthz/llm`

## API
All routes are prefixed with `/api`.

### Writing
- `POST /api/plan`
- `POST /api/draft`
- `POST /api/draft/stream` (SSE)
- `POST /api/review`
- `POST /api/review/stream` (SSE)
- `POST /api/rewrite`
- `POST /api/rewrite/stream` (SSE)

### Pipeline (plan -> research -> draft -> review -> rewrite -> citations)
- `POST /api/pipeline`
- `POST /api/pipeline/stream` (SSE)

### RAG
- `POST /api/rag/upload` (JSON documents)
- `POST /api/rag/upload-file` (files: .txt / .pdf / .docx)
- `POST /api/rag/search`
- `GET /api/rag/documents`
- `DELETE /api/rag/documents/{doc_id}`

### Citations
- `POST /api/citations`

### Versions
- `GET /api/versions`
- `GET /api/versions/{version_id}`
- `GET /api/versions/{version_id}/diff`
- `DELETE /api/versions/{version_id}`

## Streaming (SSE)
Streaming endpoints return `text/event-stream` with JSON events like:
- `status` (step updates)
- `ping` (keepalive)
- `result` (final payload)
- `error` (error detail)

## Notes
- File uploads enforce `UPLOAD_MAX_MB` and reject larger payloads.
- PDF and DOCX parsing requires `pypdf` and `python-docx` (already in `requirements.txt`).
- Qdrant collections must match `QDRANT_EMBED_DIM` and `QDRANT_DISTANCE`.

## Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
