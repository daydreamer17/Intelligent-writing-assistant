# Writing Assistant Backend

基于 FastAPI 的多 Agent 写作后端，提供分步写作、一键 Pipeline、RAG 检索、引用约束、会话记忆和离线检索评测能力。

## 1. 核心能力

- 写作流程：`plan -> draft -> review -> rewrite -> citations`
- 流式接口：分步与 Pipeline 均支持 SSE
- RAG：文档上传、检索、动态 `top_k`、HyDE（可选）、Rerank（可选）
- 三态调用模式：`rag_only / hybrid / creative`
- 引用机制：
  - `rag_only`：严格证据约束、可拒答
  - `hybrid`：有证据段落加 `[n]`，无证据段落加 `[推断]`
  - `creative`：不强制引用，支持运行时开关 MCP
- 拒答判定增强：拒答查询默认精简（不直接拼接超长大纲/草稿），并按 original/bilingual/HyDE 变体最优分数判定
- 覆盖率明细增强：区分语义段落覆盖率与词面段落覆盖率
- 记忆机制：会话隔离（`session_id`）、上下文压缩、冷存写入与冷存召回；支持任务前自动重置
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
RAG_GENERATION_MODE=rag_only        # rag_only / hybrid / creative
RAG_CREATIVE_MCP_ENABLED=true       # creative 模式下是否启用 MCP
RAG_CREATIVE_MEMORY_ENABLED=false   # creative 模式是否启用会话记忆
RAG_HYBRID_INFERENCE_TAG=[推断]
RAG_HYBRID_MIN_PARAGRAPH_CHARS=12
RAG_REFUSAL_ENABLED=true
RAG_REFUSAL_QUERY_MAX_CHARS=480
RAG_REFUSAL_INCLUDE_OUTLINE=false
RAG_REFUSAL_INCLUDE_DRAFT=false
```

### GitHub MCP（可选）

```env
MCP_GITHUB_ENABLED=false
GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_GITHUB_TOKEN
MCP_GITHUB_TOOL_SCOPE=search
MCP_GITHUB_MAX_TOOLS=5
```

### 分阶段工具路由（可选，建议配合 MCP）

```env
LLM_AGENT_TOOL_CALLING_ENABLED=false   # 关闭则仅显式 MCP API 可用
LLM_STAGE_BASED_TOOLS_ENABLED=true
LLM_TOOLS_PLAN_STAGE=github_search_repositories,github_search_code
LLM_TOOLS_DRAFT_STAGE=github_search_repositories,github_search_code,github_get_file_contents
LLM_TOOLS_REVIEW_STAGE=github_search_repositories,github_search_code
LLM_TOOLS_REWRITE_STAGE=
LLM_TOOL_POLICY_MODE=rules
LLM_TOOL_POLICY_SEARCH_KEYWORDS=github,repo,repository,issue,pr,commit,code,readme
LLM_TOOL_POLICY_READ_KEYWORDS=owner/repo,path,README,.md,.py
LLM_TOOL_POLICY_DISABLE_WHEN_RAG_STRONG=true
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
- `GET /api/settings/generation-mode`
- `POST /api/settings/generation-mode`
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
- `POST /api/settings/generation-mode` 支持 `creative_mcp_enabled`，用于运行时切换 creative 模式 MCP。
- 会话重置会同时清理内存历史与 SQLite 冷存（按会话作用域，包含 `session_id::tool_profile` 变体）。
- `hybrid` 模式下，Qdrant 不可用会自动降级到 SQLite 检索。
- `hybrid` 模式会优先注入可匹配的 `[n]`，仅对无证据段落补 `[推断]`。
- 覆盖率展示建议优先看语义段落覆盖率；词面段落覆盖率在中英混合语料下可能偏低。
- 拒答日志会显示 `base:<variant>` / `fallback@k:<variant>`，用于定位本轮命中的查询变体。
- 流式链路依赖 SSE，代理层需允许 `text/event-stream`。
- 离线评测结果持久化在 SQLite 表 `retrieval_eval_runs`。
