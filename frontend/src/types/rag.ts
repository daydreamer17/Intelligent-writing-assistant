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
