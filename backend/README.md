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
- `LLM_TIMEOUT` (seconds, default 60)
- `LLM_MAX_TOKENS` (optional)

### LLM reliability (optional)
- `LLM_RETRY_MAX` (default 2)
- `LLM_RETRY_BACKOFF` (seconds, default 2.0)
- `LLM_COOLDOWN_SECONDS` (default 0)

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

### Example `.env`
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_API_KEY=your_key
LLM_API_BASE=https://api.openai.com/v1

MEMORY_MODE=long_term
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=writing_memory
QDRANT_EMBED_DIM=256

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_API_KEY=your_key
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
