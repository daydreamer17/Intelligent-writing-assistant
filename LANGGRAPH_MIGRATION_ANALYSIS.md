# LangGraph 迁移分析报告

> 项目：`my_agent` 文档写作助手
> 目标：将现有 pipeline 重构为基于 LangGraph 的工作流，最大化复用已有业务能力
> 日期：2026-03-18

---

## 1. 业务能力层（Business Capability Layer）— 适合保留复用

这些模块做"实际工作"，与流程调度无关。

### Agents（核心推理能力）

| 文件 | 职责 |
|------|------|
| `app/agents/base.py` | 1912行，LLM调用基础设施、流式、工具调用、token截断 |
| `app/agents/writing_agent.py` | 起草能力（draft / draft_stream / draft_long） |
| `app/agents/reviewer_agent.py` | 审阅能力（review / review_stream） |
| `app/agents/editor_agent.py` | 改写能力（rewrite / rewrite_stream） |

### RAG & 检索

- `rag_service.py` — 文档管理
- `vector_store.py` — Qdrant 集成
- `embedding_service.py` — 向量生成
- `rag_query_expander.py` — HyDE 查询扩展
- `rag_reranker.py` — 文档重排
- `research_service.py` — TF-IDF 排名

### 引用 & 证据

- `citation_service.py` — 引用生成
- `citation_enforcer.py` — 引用强制执行与覆盖率统计
- `evidence_service.py` — 证据提取

### 基础设施

- `storage_service.py` — SQLite 持久化
- `generation_mode.py` — 模式切换（rag_only / hybrid / creative）
- `inference_marker.py` — 推断段落标记
- `tool_policy.py` — 工具路由决策
- `github_context.py` — GitHub MCP 上下文
- `models/schemas.py`, `models/entities.py` — 数据模型
- `config.py`, `utils/tokenizer.py` — 配置与工具

---

## 2. 工作流调度层（Workflow Scheduling Layer）— 适合迁移为 LangGraph

这些模块的核心职责是"把谁串联给谁"。

| 文件 | 当前职责 | 问题 |
|------|---------|------|
| `services/pipeline_service.py` | 顺序串联5个阶段 | 硬编码顺序，无状态共享，无条件分支 |
| `services/drafting_service.py` | 串联 draft→review→rewrite | 三个阶段耦合在一个类里，无法单独重试 |
| `services/planner_service.py` | 包装 WritingAgent 做规划 | 薄包装，逻辑简单 |
| `services/reviewing_service.py` | 包装 ReviewerAgent | 薄包装 |
| `services/rewriting_service.py` | 包装 EditorAgent（含分块） | 含分块逻辑，但本质是执行层 |
| `api/routes/pipeline.py` | 触发 pipeline，处理 SSE | 需要对接新图 |
| `api/deps.py` | 服务注入/初始化 | 需要注入 LangGraph 图实例 |

---

## 3. 迁移边界建议

### 先不动（Freeze）

所有业务能力层文件，一行不改：

```
app/agents/                    # 全部
app/services/rag_*.py
app/services/vector_store.py
app/services/embedding_service.py
app/services/citation_*.py
app/services/evidence_service.py
app/services/storage_service.py
app/services/generation_mode.py
app/services/inference_marker.py
app/services/tool_policy.py
app/services/github_context.py
app/models/
app/config.py
app/utils/
```

### 加适配层（Adapter）

保留原有方法，只在外部加 LangGraph node 函数包装：

| 文件 | 保留方法 | 说明 |
|------|---------|------|
| `planner_service.py` | `plan_outline()` | node 直接调用，不改内部逻辑 |
| `reviewing_service.py` | `review()` | node 直接调用 |
| `rewriting_service.py` | `rewrite()` | node 直接调用 |
| `research_service.py` | `collect_notes()` | node 直接调用 |

### 直接重写（Rewrite）

| 文件 | 原因 |
|------|------|
| `services/pipeline_service.py` | `WritingPipeline.run()` 替换为图调用；保留类壳作为 API 适配器 |
| `services/drafting_service.py` | draft/review/rewrite 三个方法拆分为独立 node，类本身可废弃或保留为兼容层 |
| `api/routes/pipeline.py` | 对接新图的 invoke / stream |
| `api/deps.py` | 注入编译好的 LangGraph 图 |

---

## 4. 最小可行迁移方案（MVP）

### 目标

用 LangGraph 替换 `WritingPipeline.run()` 的顺序调度，不改变任何业务能力，不改变 API 接口。

### MVP 刻意排除

- 不做条件分支（如质量不达标自动重试）
- 不做并行节点
- 不做 LangGraph checkpointing
- 不改变流式实现方式（保留现有 SSE generator）

### MVP 图结构

```
START
  → plan_node          (调用 PlanningService.plan_outline)
  → research_node      (调用 ResearchService.collect_notes)
  → draft_node         (调用 WritingAgent.draft)
  → review_node        (调用 ReviewingService.review)
  → rewrite_node       (调用 RewritingService.rewrite)
  → post_process_node  (CitationEnforcer + InferenceMarker)
END
```

### 共享状态 `WritingState` 字段

| 字段 | 类型 | 来源阶段 |
|------|------|---------|
| `topic` | str | 输入 |
| `sources` | list | 输入 |
| `session_id` | str | 输入 |
| `config` | dict | 输入 |
| `outline` | str | plan_node |
| `research_notes` | list | research_node |
| `draft` | str | draft_node |
| `review` | str | review_node |
| `revised` | str | rewrite_node |
| `citations` | list | post_process_node |
| `coverage` | dict | post_process_node |
| `stage` | str | 各节点更新 |
| `errors` | list | 各节点写入 |

### API 层变化

`pipeline.py` 路由从调用 `WritingPipeline.run()` 改为调用 `graph.invoke(state)`，接口签名不变。

---

## 5. 建议新增/修改的文件列表

### 新增文件

```
app/graph/
├── __init__.py
├── state.py          # WritingState TypedDict 定义
├── nodes.py          # 6个 node 函数（包装现有 services）
└── graph.py          # 图定义、边、编译（返回 CompiledGraph）
```

### 修改文件

| 文件 | 修改范围 |
|------|---------|
| `app/services/pipeline_service.py` | `WritingPipeline.run()` 改为调用 `graph.invoke()`；保留类作为 API 适配器 |
| `app/api/deps.py` | `AppServices` 增加 `graph: CompiledGraph` 字段；初始化时构建图 |
| `app/api/routes/pipeline.py` | 替换 `pipeline_service.run()` 调用；流式端点对接 `graph.stream()` |
| `requirements.txt` | 增加 `langgraph>=0.2` |

### 不动文件（确认）

```
app/agents/           # 全部不动
app/services/         # 除 pipeline_service.py 外全部不动
app/models/           # 全部不动
app/config.py         # 不动
app/utils/            # 不动
app/api/routes/       # 除 pipeline.py 外全部不动
```

---

## 6. 迁移风险评估

| 风险点 | 等级 | 说明 |
|--------|------|------|
| 流式兼容性 | 中 | 现有 SSE generator 需要与 LangGraph stream 对接 |
| 状态序列化 | 低 | WritingState 全为基础类型，无序列化问题 |
| 依赖注入 | 低 | AppServices 加一个字段，影响范围小 |
| 业务逻辑回归 | 低 | node 函数只是薄包装，业务逻辑零改动 |
| 分块逻辑 | 中 | rewriting_service 的长文分块逻辑需确认在 node 内正常工作 |

**迁移风险最低点**：`nodes.py` 里每个 node 函数只是 `service.method(state) → state` 的薄包装，业务逻辑零改动。整个迁移的核心变化只有 `graph.py`（图结构）和 `state.py`（共享状态定义）两个新文件。
