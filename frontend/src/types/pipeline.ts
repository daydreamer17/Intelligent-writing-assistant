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
