import type { SourceDocumentInput } from "./rag";

export interface ResearchNoteResponse {
  doc_id: string;
  title: string;
  summary: string;
  url?: string;
}

export interface CoverageDetail {
  token_coverage: number;
  paragraph_coverage: number;
  semantic_coverage: number;
  covered_tokens: number;
  total_tokens: number;
  covered_paragraphs: number;
  total_paragraphs: number;
  semantic_covered_paragraphs?: number;
  semantic_total_paragraphs?: number;
}

export interface CitationItemResponse {
  label: string;
  title: string;
  url?: string;
}

export interface PipelineRequest {
  topic: string;
  audience?: string;
  style?: string;
  target_length?: string;
  constraints?: string;
  key_points?: string;
  review_criteria?: string;
  sources?: SourceDocumentInput[];
  session_id?: string;
}

export interface PipelineResponse {
  outline: string;
  assumptions: string;
  open_questions: string;
  research_notes: ResearchNoteResponse[];
  draft: string;
  review: string;
  revised: string;
  citations: CitationItemResponse[];
  bibliography: string;
  version_id?: number;
  coverage?: number;
  coverage_detail?: CoverageDetail;
  citation_enforced?: boolean;
}

export interface PipelineV2Request extends PipelineRequest {
  thread_id?: string;
}

export interface PipelineV2ResumeRequest {
  thread_id: string;
  outline_override?: string;
  draft_override?: string;
}

export interface PipelineV2InterruptPayload {
  thread_id?: string;
  interrupt_stage?: string;
  outline?: string;
  draft?: string;
  review_text?: string;
  needs_rewrite?: boolean;
  reason?: string;
  score?: number | null;
  assumptions?: string;
  open_questions?: string;
  [key: string]: unknown;
}

export interface PipelineV2Interrupt {
  kind: string;
  payload: PipelineV2InterruptPayload;
}

export interface PipelineV2Response {
  status: "interrupted" | "completed";
  thread_id: string;
  interrupt?: PipelineV2Interrupt | null;
  result?: PipelineResponse | null;
}

export interface PipelineV2CheckpointSummary {
  thread_id: string;
  status: string;
  current_stage: string;
  updated_at: string;
}

export interface PipelineV2CheckpointListResponse {
  checkpoints: PipelineV2CheckpointSummary[];
}

export interface PipelineV2CheckpointDetailResponse {
  thread_id: string;
  session_id: string;
  mode: string;
  status: string;
  current_stage: string;
  interrupt_stage: string;
  created_at: string;
  updated_at: string;
  can_resume: boolean;
  outline: string;
  draft: string;
  review_text: string;
  needs_rewrite: boolean | null;
  reason: string;
  score: number | null;
  assumptions: string;
  open_questions: string;
  last_error: string;
}

export interface PipelineV2CheckpointCleanupRequest {
  older_than_hours?: number;
  status?: string;
  dry_run?: boolean;
  limit?: number;
}

export interface PipelineV2CheckpointCleanupResponse {
  dry_run: boolean;
  matched: number;
  deleted: number;
  thread_ids: string[];
  older_than_hours: number;
  status: string;
}

export interface DeletePipelineV2CheckpointResponse {
  deleted: boolean;
}
