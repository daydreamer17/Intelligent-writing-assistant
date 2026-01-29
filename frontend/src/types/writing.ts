export interface PlanRequest {
  topic: string;
  audience?: string;
  style?: string;
  target_length?: string;
  constraints?: string;
  key_points?: string;
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
}

export interface DraftResponse {
  draft: string;
}

export interface ReviewRequest {
  draft: string;
  criteria?: string;
  sources?: string;
  audience?: string;
}

export interface ReviewResponse {
  review: string;
}

export interface RewriteRequest {
  draft: string;
  guidance?: string;
  style?: string;
  target_length?: string;
}

export interface RewriteResponse {
  revised: string;
}
