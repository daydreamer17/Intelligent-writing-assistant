# Writing Assistant Backend

基于 FastAPI 的多 Agent 写作后端，提供分步写作、一键 Pipeline、RAG 检索、引用约束、会话记忆和离线检索评测能力。

## 1. 核心能力

- 写作流程：`plan -> draft -> review -> rewrite -> citations`
- 流式接口：分步与 Pipeline 均支持 SSE
- RAG：文档上传、检索、动态 `top_k`、HyDE（可选）、Rerank（可选）
- 引用机制：可选强制引用、两段式证据生成、覆盖率评估与拒答保护
- 记忆机制：会话隔离（`session_id`）、上下文压缩、冷存写入与冷存召回
- 评测能力：离线检索评测（Recall/Precision/HitRate/MRR/nDCG）与历史持久化
- MCP：可选 GitHub MCP 显式工具调用

## 2. 目录结构（简版）

```text
backend/
  app/
    agents/        # Writing / Reviewer / Editor
    api/routes/    # writing / pipeline / rag / versions / settings / mcp_github
    services/      # pipeline, rag, citation, retrieval_eval, storage...
    models/
    config.py
  data/
  main.py
  requirements.txt
  .env.example
```

## 3. 快速启动

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py
```

服务地址：`http://localhost:8000`

## 4. 最小配置

完整配置见 `backend/.env.example`。

### LLM

```env
LLM_PROVIDER=openai
LLM_MODEL=YOUR_MODEL
LLM_API_KEY=YOUR_API_KEY
LLM_API_BASE=YOUR_BASE_URL
LLM_TIMEOUT=600
LLM_MAX_TOKENS=8000
```

### 检索与记忆

```env
RETRIEVAL_MODE=sqlite_only          # 或 hybrid
CONVERSATION_MEMORY_MODE=session    # 或 global
STORAGE_PATH=data/app.db
```

### Qdrant（可选）

```env
QDRANT_URL=YOUR_QDRANT_URL
QDRANT_API_KEY=YOUR_QDRANT_KEY
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_EMBED_DIM=1024
QDRANT_DISTANCE=cosine
```

### RAG（建议）

```env
RAG_DYNAMIC_TOPK_ENABLED=true
RAG_RERANK_ENABLED=true
RAG_HYDE_ENABLED=false
RAG_CITATION_ENFORCE=false
RAG_REFUSAL_ENABLED=true
```

### GitHub MCP（可选）

```env
MCP_GITHUB_ENABLED=false
GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_GITHUB_TOKEN
MCP_GITHUB_TOOL_SCOPE=search
MCP_GITHUB_MAX_TOOLS=5
```

## 5. API 概览

所有接口前缀：`/api`

### 写作分步
- `POST /api/plan`
- `POST /api/draft` / `POST /api/draft/stream`
- `POST /api/review` / `POST /api/review/stream`
- `POST /api/rewrite` / `POST /api/rewrite/stream`

### 一键流程
- `POST /api/pipeline`
- `POST /api/pipeline/stream`

### RAG
- `POST /api/rag/upload`
- `POST /api/rag/upload-file`（`.txt / .pdf / .docx / .md / .markdown`）
- `POST /api/rag/search`
- `GET /api/rag/documents`
- `DELETE /api/rag/documents/{doc_id}`

### 离线评测
- `POST /api/rag/evaluate`
- `GET /api/rag/evaluations`
- `GET /api/rag/evaluations/{run_id}`
- `DELETE /api/rag/evaluations/{run_id}`

### 引用与设置
- `POST /api/citations`
- `GET /api/settings/citation`
- `POST /api/settings/citation`
- `POST /api/settings/session-memory/clear`

### 版本管理
- `GET /api/versions`
- `GET /api/versions/{version_id}`
- `GET /api/versions/{version_id}/diff`
- `DELETE /api/versions/{version_id}`

### MCP（GitHub）
- `GET /api/mcp/github/tools`
- `POST /api/mcp/github/call`

## 6. 健康检查与文档

- `GET /healthz`
- `GET /healthz/detail`
- `GET /healthz/llm`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 7. 运维注意项

- `session` 记忆模式下，请从前端持续传同一个 `session_id`，避免会话历史丢失。
- `hybrid` 模式下，Qdrant 不可用会自动降级到 SQLite 检索。
- 流式链路依赖 SSE，代理层需允许 `text/event-stream`。
- 离线评测结果持久化在 SQLite 表 `retrieval_eval_runs`。
