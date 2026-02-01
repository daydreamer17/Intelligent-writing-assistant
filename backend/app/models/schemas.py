from __future__ import annotations

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


class DraftResponse(BaseModel):
    draft: str


class ReviewRequest(BaseModel):
    draft: str = Field(..., description="待审校的草稿")
    criteria: str = Field("", description="审校标准")
    sources: str = Field("", description="可用来源")
    audience: str = Field("", description="目标读者")


class ReviewResponse(BaseModel):
    review: str


class RewriteRequest(BaseModel):
    draft: str = Field(..., description="待改写草稿")
    guidance: str = Field("", description="改写指导")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")


class RewriteResponse(BaseModel):
    revised: str


class PipelineRequest(BaseModel):
    topic: str = Field(..., description="写作主题")
    audience: str = Field("", description="目标读者")
    style: str = Field("", description="写作风格")
    target_length: str = Field("", description="目标长度")
    constraints: str = Field("", description="限制条件")
    key_points: str = Field("", description="关键要点")
    review_criteria: str = Field("", description="审校标准")
    sources: list[SourceDocumentInput] = Field(default_factory=list, description="可用资料源")


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


class UploadDocumentsRequest(BaseModel):
    documents: list[SourceDocumentInput] = Field(default_factory=list)


class UploadDocumentsResponse(BaseModel):
    documents: list[SourceDocumentResponse] = Field(default_factory=list)


class SearchDocumentsRequest(BaseModel):
    query: str = Field(..., description="检索关键词")
    top_k: int = Field(5, description="返回条数")


class SearchDocumentsResponse(BaseModel):
    documents: list[SourceDocumentResponse] = Field(default_factory=list)


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
