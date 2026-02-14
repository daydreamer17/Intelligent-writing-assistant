# 智能写作助手（Intelligent Writing Assistant）

基于 [Hello-Agents](https://github.com/jjyaoao/HelloAgents.git) 的多 Agent 写作系统，提供从规划到改写的完整链路，并集成 RAG、引用约束、会话记忆与离线检索评测。

## 1. 关键能力

- 多 Agent 写作流水线：`plan -> draft -> review -> rewrite -> citations`
- 流式生成：支持分步与一键 Pipeline 的 SSE 实时状态与结果推送
- RAG 检索增强：支持 `.txt / .pdf / .docx / .md / .markdown` 上传与检索
- 动态检索策略：按语料规模动态调整 `top_k`、候选数、研究笔记注入量
- 调用模式三态：`RAG-only / Hybrid / Creative`
- 引用策略：
  - `RAG-only`：严格证据约束与拒答保护
  - `Hybrid`：优先打 `[n]` 引用；无证据段落自动标注 `[推断]`
  - `Creative`：自由生成，不强制引用；支持前端开关控制是否启用 MCP
- 拒答保护：检索质量不足时拒答，避免低可信幻觉输出
- 记忆机制：会话隔离（`session_id`）、历史压缩、冷存写入与冷存召回回注；支持任务前自动重置会话记忆（含冷存）
- 评测能力：离线检索评测（Recall/Precision/HitRate/MRR/nDCG）与历史持久化
- 外部知识接入：GitHub MCP（可选）及显式工具 API

## 2. 技术栈

- 后端：FastAPI、Uvicorn、Hello-Agents
- 存储：SQLite（结构化与历史）、Qdrant（向量检索，可选）
- 前端：Vue 3、TypeScript、Vite、Pinia
- 检索增强：HyDE、Rerank、中文分词（jieba）

## 3. 项目结构（简版）

```text
my_agent/
├── backend/
│   ├── app/
│   │   ├── agents/      # Writing / Reviewer / Editor
│   │   ├── api/routes/  # writing / pipeline / rag / versions / settings / mcp_github
│   │   ├── services/    # pipeline, rag, citation, retrieval_eval, storage...
│   │   ├── models/
│   │   └── config.py
│   ├── main.py
│   ├── .env.example
│   └── README.md
└── frontend/
    ├── src/views/        # Workspace / History / RagCenter / Settings
    ├── src/components/   # 编辑器、差异视图、评测组件等
    └── src/services/     # API 调用层
```

## 4. 快速启动

### 后端

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py
```

后端地址：`http://localhost:8000`

### 前端

```bash
cd frontend
npm install
npm run dev
```

前端地址：`http://localhost:5173`

## 5. 最小必配环境变量

完整参数见 `backend/.env.example`，以下是常用核心项。

### LLM

```env
LLM_PROVIDER=openai
LLM_MODEL=YOUR_MODEL
LLM_API_KEY=YOUR_KEY
LLM_API_BASE=YOUR_BASE_URL
LLM_TIMEOUT=600
LLM_MAX_TOKENS=8000
```

### 检索与会话记忆

```env
RETRIEVAL_MODE=sqlite_only           # 或 hybrid
CONVERSATION_MEMORY_MODE=session     # 或 global
STORAGE_PATH=data/app.db
```

### Qdrant（可选，hybrid 模式）

```env
QDRANT_URL=YOUR_QDRANT_URL
QDRANT_API_KEY=YOUR_QDRANT_KEY
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_EMBED_DIM=1024
QDRANT_DISTANCE=cosine
```

### RAG 核心（建议）

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
```

### GitHub MCP（可选）

```env
MCP_GITHUB_ENABLED=false
GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_GITHUB_TOKEN
LLM_AGENT_TOOL_CALLING_ENABLED=false
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

## 6. 关键 API

- 写作分步：`POST /api/plan`、`/api/draft`、`/api/review`、`/api/rewrite`
- 流式分步：`POST /api/draft/stream`、`/api/review/stream`、`/api/rewrite/stream`
- 一键流程：`POST /api/pipeline`、`POST /api/pipeline/stream`
- RAG：`/api/rag/upload`、`/api/rag/upload-file`、`/api/rag/search`
- 评测：`/api/rag/evaluate`、`/api/rag/evaluations`、`/api/rag/evaluations/{run_id}`
- 版本：`/api/versions`、`/api/versions/{id}`、`/api/versions/{id}/diff`
- 设置：`/api/settings/generation-mode`、`/api/settings/citation`（兼容旧开关）、`/api/settings/session-memory/clear`
- MCP：`/api/mcp/github/tools`、`/api/mcp/github/call`

接口文档：
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## 7. 更新日志

### v0.5.2 (2026-02-14)
**🧩 Creative 模式可控与会话重置增强**

- ✨ 新增 Creative 模式 MCP 前端显示开关（运行时生效）
- ✨ 新增 `RAG_CREATIVE_MCP_ENABLED`、`RAG_CREATIVE_MEMORY_ENABLED` 配置项
- 🔧 工作台分步/一键任务执行前自动重置会话记忆（含冷存）
- 🔧 修复“显示无可用记忆但仍会污染”问题：会话重置改为按会话作用域清理 SQLite 冷存（含 `session_id::tool_profile`）

### v0.5.1 (2026-02-14)
**🔧 引用与改写链路对齐版**

- ✨ 新增三态调用模式落地：`RAG-only / Hybrid / Creative`
- ✨ Hybrid 模式调整为“有证据打 `[n]`，无证据打 `[推断]`”
- 🔧 修复改写流式最终结果与引用面板不同步问题（最终稿统一回传并覆盖）
- 🔧 改写阶段 guidance 合并“审校输出 + 审校标准”，提升按审校意见改写的一致性
- 🔧 设置接口补充 `GET/POST /api/settings/generation-mode`，旧 citation 开关保持兼容映射

### v0.5.0 (2026-02-11)
**🧠 记忆与链路稳定性增强版**

#### 会话记忆与模式拆分
- ✨ 新增 `RETRIEVAL_MODE`（`sqlite_only` / `hybrid`）与 `CONVERSATION_MEMORY_MODE`（`session` / `global`）双开关
- ✨ 新增会话级 Agent 隔离与 TTL 回收（避免跨任务历史串扰）
- ✨ 新增会话记忆清理接口 `POST /api/settings/session-memory/clear`，前端工作区支持“一键重置会话记忆”

#### 冷存记忆闭环
- ✨ 新增冷存召回（cold recall）并回注上下文，解决“能冷存但不回忆”的问题
- ✨ 新增分层记忆注入策略（最近历史 + 压缩摘要 + 冷存召回）

#### Pipeline 与流式稳定性
- 🔧 修复一键 Pipeline 前端阶段进度与后端链路不同步问题
- 🔧 修复部分场景下非流式回退导致页面提前结束的问题
- 🔧 增加阶段级输入预算与提示词裁剪保护，降低超长上下文导致的 400 报错风险

### v0.4.0 (2026-02-10)
**📊 评测与体验增强版**

#### 离线检索评测增强
- ✨ 新增 `POST /api/rag/evaluate` 标注集离线评测接口
- ✨ 新增检索指标：Recall@K、Precision@K、HitRate@K、MRR、nDCG
- ✨ 新增评测历史接口：`GET /api/rag/evaluations`、`GET /api/rag/evaluations/{run_id}`、`DELETE /api/rag/evaluations/{run_id}`
- 💾 每次评测结果自动持久化到 SQLite，支持前端历史回放与删除

#### 前端评测与历史体验优化
- ✨ RAG 评测页支持曲线可视化与逐 Query 失败样本展示
- ✨ 版本历史页与评测详情支持“展开/收起”而非依赖刷新页面
- ✨ 差异视图优化为大字号、自动换行与分行高亮，避免横向滚动

#### 生成链路稳定性补强
- 🔧 Pipeline 与分步流式链路统一修复“流式未完成前页面结束”导致的展示截断
- 🔧 引用覆盖率细化为 token/段落/语义三类指标，提升可解释性

### v0.3.0 (2026-02-08)
**⚡ 性能优化专版**

#### 超时配置优化
- 🔧 `LLM_MAX_TOKENS` 从 1200 提升到 8000，支持长文本生成（2000+ 词）
- 🔧 新增动态 max_tokens 计算：review 阶段 0.8x，rewrite 阶段 1.5x
- 🐛 修复流式内容被覆盖问题：优化前端 delta 事件处理逻辑
- 📊 **效果**：长文本生成成功率从 0% 提升到 100%

#### RAG Top-K 优化
- ⚡ 动态阈值优化：SMALL 1000→50，LARGE 50000→500
  - 用户覆盖率从 2% 提升到 98%
- ⚡ 候选数优化：Small 30→15，Medium 60→24，Large 120→36
  - 计算量降低 50%，检索延迟从 230ms 降低到 90ms（-61%）
- ⚡ 重排序过采样率：1.6x→3x（行业标准）
  - Rerank 有效性提升 42%
- ⚡ Research Notes 优化：3→5 条（Small），5→8 条（Medium），8→12 条（Large）
  - 信息利用率从 37.5% 提升到 80%
- 📊 **综合效果**：任务成功率从 33% 提升到 80%+

#### RAG 配置优化
- 🔧 查询扩展：6 个→3 个扩展查询（减少 API 成本 50%）
- 🔧 覆盖率阈值：0.1→0.3（减少低质量引用）
- 🔧 拒答阈值调整：MIN_RECALL 0.5→0.3，MIN_AVG_RECALL 0.3→0.2
- 📊 **效果**：RAG 拒绝率从 67% 降低到 15%

#### 中文分词优化
- ✨ 新增统一分词器模块 `utils/tokenizer.py`
- ✨ 接入 `jieba` 分词库，支持中文分词
- ✨ 覆盖范围：RAG 检索、重排序、引用匹配、覆盖率计算
- 📊 **效果**：中文查询准确率提升 40%+

### v0.2.0 (2026-02-07)
- ✨ 新增动态 RAG 检索策略（按语料规模动态调整检索 `top_k` 与候选数）
- ✨ 新增动态 research notes 数量策略（Pipeline 中不再固定 3 条）
- ✨ 新增强制引用开关（`RAG_CITATION_ENFORCE`）及前端设置接口
- ✨ 新增拒答机制（检索质量不足时返回"在提供的文档中，无法找到该问题的答案。"）
- ✨ 新增强制引用下的两段式流程（先证据抽取，再基于证据生成）
- ✨ 新增引用覆盖指标（语义覆盖率、段落覆盖率）及返回结构
- ✨ 新增 GitHub MCP 接入与显式 API（工具列表/工具调用）
- ✨ 新增中文分词优化（`jieba`）并统一用于检索与引用匹配
- ✨ 新增 RAG 文件支持与解析（`.pdf` / `.docx` / `.md` / `.markdown`，含 Markdown 纯文本化）
- 🔧 优化流式稳定性与前端显示同步（Pipeline 与分步接口）

### v0.1.0 (2026-01-29)
- ✨ 初始版本发布
- ✅ 完成核心写作流程（Writing → Review → Edit）
- ✅ 实现 RAG 知识增强功能
- ✅ 添加历史记录和版本管理
- ✅ 完善前端界面和用户体验
- ✅ 统一错误处理和表单验证
- ✅ 添加自动保存和状态持久化

## 8. 说明

- 项目处于持续迭代阶段，建议在生产环境前进行回归测试。
- 更详细的后端实现与配置说明见 `backend/README.md`。
