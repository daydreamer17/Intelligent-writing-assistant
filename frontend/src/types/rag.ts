export interface SourceDocumentInput {
  doc_id: string;
  title: string;
  content: string;
  url?: string;
}

export interface SourceDocumentResponse {
  doc_id: string;
  title: string;
  content: string;
  url: string;
}

export interface UploadDocumentsRequest {
  documents: SourceDocumentInput[];
}

export interface UploadDocumentsResponse {
  documents: SourceDocumentResponse[];
}

export interface SearchDocumentsRequest {
  query: string;
  top_k?: number;
}

export interface SearchDocumentsResponse {
  documents: SourceDocumentResponse[];
}

export interface DeleteDocumentResponse {
  deleted: boolean;
}

export interface RetrievalEvalCaseInput {
  query: string;
  relevant_doc_ids: string[];
  query_id?: string;
}

export interface RetrievalMetricAtK {
  k: number;
  recall: number;
  precision: number;
  hit_rate: number;
  mrr: number;
  ndcg: number;
}

export interface RetrievalEvalCaseResult {
  query: string;
  query_id: string;
  relevant_count: number;
  retrieved_doc_ids: string[];
  metrics: RetrievalMetricAtK[];
}

export interface RetrievalEvalRequest {
  cases: RetrievalEvalCaseInput[];
  k_values: number[];
}

export interface RetrievalEvalResponse {
  eval_run_id?: number;
  created_at?: string;
  total_queries: number;
  queries_with_relevance: number;
  k_values: number[];
  macro_metrics: RetrievalMetricAtK[];
  per_query: RetrievalEvalCaseResult[];
}

export interface RetrievalEvalRunSummaryResponse {
  run_id: number;
  created_at: string;
  total_queries: number;
  queries_with_relevance: number;
  k_values: number[];
  macro_metrics: RetrievalMetricAtK[];
}

export interface RetrievalEvalRunsResponse {
  runs: RetrievalEvalRunSummaryResponse[];
}

export interface DeleteRetrievalEvalRunResponse {
  deleted: boolean;
}
