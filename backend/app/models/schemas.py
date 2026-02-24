from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceDocumentInput(BaseModel):
    doc_id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档内容")
    url: str = Field("", description="来源链接")


class SourceDocumentResponse(BaseModel):
    doc_id: str
    title: str
    content: str
    url: str = ""


class ResearchNoteResponse(BaseModel):
    doc_id: str
    title: str
    summary: str
    url: str = ""


class PlanRequest(BaseModel):
    topic: str = Field(..., description="写作主题")
    audience: str = Field("", description="目标读者")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")
    constraints: str = Field("", description="限制条件")
    key_points: str = Field("", description="关键要点")
    session_id: str = Field("", description="会话ID（用于隔离记忆）")


class PlanResponse(BaseModel):
    outline: str
    assumptions: str = ""
    open_questions: str = ""


class DraftRequest(BaseModel):
    topic: str = Field(..., description="写作主题")
    outline: str = Field(..., description="写作大纲")
    research_notes: str = Field("", description="研究资料或笔记")
    constraints: str = Field("", description="限制条件")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")
    session_id: str = Field("", description="会话ID（用于隔离记忆）")


class DraftResponse(BaseModel):
    draft: str


class ReviewRequest(BaseModel):
    draft: str = Field(..., description="待审校的草稿")
    criteria: str = Field("", description="审校标准")
    sources: str = Field("", description="可用来源")
    audience: str = Field("", description="目标读者")
    session_id: str = Field("", description="会话ID（用于隔离记忆）")


class ReviewResponse(BaseModel):
    review: str


class RewriteRequest(BaseModel):
    draft: str = Field(..., description="待改写草稿")
    guidance: str = Field("", description="改写指导")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")
    session_id: str = Field("", description="会话ID（用于隔离记忆）")


class RewriteResponse(BaseModel):
    revised: str
    citations: list["CitationItemResponse"] = Field(default_factory=list)
    bibliography: str = ""
    coverage: float | None = None
    coverage_detail: CoverageDetail | None = None
    citation_enforced: bool = False


class PipelineRequest(BaseModel):
    topic: str = Field(..., description="写作主题")
    audience: str = Field("", description="目标读者")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")
    constraints: str = Field("", description="限制条件")
    key_points: str = Field("", description="关键要点")
    review_criteria: str = Field("", description="审校标准")
    sources: list[SourceDocumentInput] = Field(default_factory=list, description="可用资料源")
    session_id: str = Field("", description="会话ID（用于隔离记忆）")


class CoverageDetail(BaseModel):
    token_coverage: float = Field(0.0, description="按 token 统计的覆盖率")
    paragraph_coverage: float = Field(0.0, description="按段落统计的覆盖率")
    semantic_coverage: float = Field(0.0, description="按语义相似度统计的覆盖率")
    covered_tokens: int = 0
    total_tokens: int = 0
    covered_paragraphs: int = 0
    total_paragraphs: int = 0
    semantic_covered_paragraphs: int = 0
    semantic_total_paragraphs: int = 0


class PipelineResponse(BaseModel):
    outline: str
    assumptions: str = ""
    open_questions: str = ""
    research_notes: list[ResearchNoteResponse] = Field(default_factory=list)
    draft: str
    review: str
    revised: str
    citations: list[CitationItemResponse] = Field(default_factory=list)
    bibliography: str = ""
    version_id: int | None = None
    coverage: float | None = None
    coverage_detail: CoverageDetail | None = None
    citation_enforced: bool = False


class UploadDocumentsRequest(BaseModel):
    documents: list[SourceDocumentInput] = Field(default_factory=list)


class UploadDocumentsResponse(BaseModel):
    documents: list[SourceDocumentResponse] = Field(default_factory=list)


class SearchDocumentsRequest(BaseModel):
    query: str = Field(..., description="检索关键词")
    top_k: int = Field(5, description="返回条数")


class SearchDocumentsResponse(BaseModel):
    documents: list[SourceDocumentResponse] = Field(default_factory=list)


class RetrievalEvalCaseInput(BaseModel):
    query: str = Field(..., description="Retrieval query")
    relevant_doc_ids: list[str] = Field(default_factory=list, description="Relevant document IDs")
    query_id: str = Field("", description="Optional query sample ID")


class RagEvalConfigOverride(BaseModel):
    rerank_enabled: bool | None = Field(None, description="Temporarily enable/disable rerank for this eval run")
    hyde_enabled: bool | None = Field(None, description="Temporarily enable/disable HyDE query expansion for this eval run")
    bilingual_rewrite_enabled: bool | None = Field(
        None,
        description="Temporarily enable/disable bilingual query rewrite for this eval run",
    )


class RetrievalMetricAtK(BaseModel):
    k: int
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float


class RetrievalEvalCaseResult(BaseModel):
    query: str
    query_id: str = ""
    relevant_count: int
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    metrics: list[RetrievalMetricAtK] = Field(default_factory=list)


class RetrievalEvalRequest(BaseModel):
    cases: list[RetrievalEvalCaseInput] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5], description="K values to evaluate")
    rag_config_override: RagEvalConfigOverride | None = Field(
        default=None,
        description="Per-request retrieval config override (eval only)",
    )


class RetrievalEvalResponse(BaseModel):
    eval_run_id: int | None = None
    created_at: str = ""
    total_queries: int
    queries_with_relevance: int
    k_values: list[int] = Field(default_factory=list)
    macro_metrics: list[RetrievalMetricAtK] = Field(default_factory=list)
    per_query: list[RetrievalEvalCaseResult] = Field(default_factory=list)


class RetrievalEvalRunSummaryResponse(BaseModel):
    run_id: int
    created_at: str = ""
    total_queries: int
    queries_with_relevance: int
    k_values: list[int] = Field(default_factory=list)
    macro_metrics: list[RetrievalMetricAtK] = Field(default_factory=list)


class RetrievalEvalRunsResponse(BaseModel):
    runs: list[RetrievalEvalRunSummaryResponse] = Field(default_factory=list)


class DeleteRetrievalEvalRunResponse(BaseModel):
    deleted: bool

class DeleteDocumentResponse(BaseModel):
    deleted: bool


class CitationNoteInput(BaseModel):
    doc_id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    summary: str = Field("", description="摘要")
    url: str = Field("", description="来源链接")


class CitationRequest(BaseModel):
    notes: list[CitationNoteInput] = Field(default_factory=list)


class CitationItemResponse(BaseModel):
    label: str
    title: str
    url: str = ""


class CitationResponse(BaseModel):
    citations: list[CitationItemResponse] = Field(default_factory=list)
    bibliography: str = ""


class DraftVersionResponse(BaseModel):
    version_id: int
    topic: str
    outline: str
    research_notes: str
    draft: str
    review: str
    revised: str
    created_at: str = ""


class VersionsResponse(BaseModel):
    versions: list[DraftVersionResponse] = Field(default_factory=list)


class VersionDetailResponse(BaseModel):
    version: DraftVersionResponse


class DeleteVersionResponse(BaseModel):
    deleted: bool


class VersionDiffResponse(BaseModel):
    from_version_id: int
    to_version_id: int
    field: str
    diff: str


class CitationSettingRequest(BaseModel):
    enabled: bool = Field(..., description="是否启用强制引用")


class CitationSettingResponse(BaseModel):
    enabled: bool


class GenerationModeSettingRequest(BaseModel):
    mode: str = Field(..., description="生成模式：rag_only | hybrid | creative")
    creative_mcp_enabled: bool | None = Field(
        default=None, description="仅 creative 模式使用：是否启用 MCP"
    )


class GenerationModeSettingResponse(BaseModel):
    mode: str
    citation_enforce: bool
    mcp_allowed: bool
    inference_mark_required: bool
    creative_mcp_enabled: bool


class SessionMemoryClearRequest(BaseModel):
    session_id: str = Field("", description="会话ID；为空时清理默认会话")
    drop_agent: bool = Field(True, description="是否同时移除会话实例")
    clear_cold: bool = Field(False, description="是否清理该会话的冷存记忆")


class SessionMemoryClearResponse(BaseModel):
    session_id: str
    cleared: bool
    cleared_agents: list[str] = Field(default_factory=list)


class MCPToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="MCP 工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class MCPToolCallResponse(BaseModel):
    result: str


class MCPToolsResponse(BaseModel):
    tools: list[dict[str, str]] = Field(default_factory=list)
    raw: str = ""

