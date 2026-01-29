import type { SourceDocumentInput } from "./rag";

export interface ResearchNoteResponse {
  doc_id: string;
  title: string;
  summary: string;
  url?: string;
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
}
