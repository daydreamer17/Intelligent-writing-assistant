export interface PlanRequest {
  topic: string;
  audience?: string;
  style?: string;
  target_length?: string;
  constraints?: string;
  key_points?: string;
  session_id?: string;
}

export interface PlanResponse {
  outline: string;
  assumptions: string;
  open_questions: string;
}

export interface DraftRequest {
  topic: string;
  outline: string;
  research_notes?: string;
  constraints?: string;
  style?: string;
  target_length?: string;
  session_id?: string;
}

export interface DraftResponse {
  draft: string;
}

export interface ReviewRequest {
  draft: string;
  criteria?: string;
  sources?: string;
  audience?: string;
  session_id?: string;
}

export interface ReviewResponse {
  review: string;
}

export interface RewriteRequest {
  draft: string;
  guidance?: string;
  style?: string;
  target_length?: string;
  session_id?: string;
}

export interface RewriteResponse {
  revised: string;
  citations?: { label: string; title: string; url?: string }[];
  bibliography?: string;
  coverage?: number;
  coverage_detail?: {
    token_coverage: number;
    paragraph_coverage: number;
    semantic_coverage: number;
    covered_tokens: number;
    total_tokens: number;
    covered_paragraphs: number;
    total_paragraphs: number;
    semantic_covered_paragraphs?: number;
    semantic_total_paragraphs?: number;
  };
  citation_enforced?: boolean;
}
