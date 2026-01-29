export interface CitationNoteInput {
  doc_id: string;
  title: string;
  summary?: string;
  url?: string;
}

export interface CitationRequest {
  notes: CitationNoteInput[];
}

export interface CitationItemResponse {
  label: string;
  title: string;
  url?: string;
}

export interface CitationResponse {
  citations: CitationItemResponse[];
  bibliography: string;
}
