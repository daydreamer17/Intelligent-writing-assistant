import { getApi } from "./api";
import type {
  SearchDocumentsRequest,
  SearchDocumentsResponse,
  UploadDocumentsRequest,
  UploadDocumentsResponse,
  DeleteDocumentResponse,
  DeleteRetrievalEvalRunResponse,
  RetrievalEvalRequest,
  RetrievalEvalResponse,
  RetrievalEvalRunsResponse,
} from "../types";

export const uploadDocuments = async (payload: UploadDocumentsRequest) => {
  const { data } = await getApi().post<UploadDocumentsResponse>("/api/rag/upload", payload);
  return data;
};

export const searchDocuments = async (payload: SearchDocumentsRequest) => {
  const { data } = await getApi().post<SearchDocumentsResponse>("/api/rag/search", payload);
  return data;
};

export const listDocuments = async (limit: number = 100) => {
  const { data } = await getApi().get<SearchDocumentsResponse>("/api/rag/documents", {
    params: { limit },
  });
  return data;
};

export const deleteDocument = async (docId: string) => {
  const { data } = await getApi().delete<DeleteDocumentResponse>(`/api/rag/documents/${docId}`);
  return data;
};

export const uploadFiles = async (files: File[]) => {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  const { data } = await getApi().post<UploadDocumentsResponse>("/api/rag/upload-file", form);
  return data;
};

export const evaluateRetrieval = async (payload: RetrievalEvalRequest) => {
  const { data } = await getApi().post<RetrievalEvalResponse>("/api/rag/evaluate", payload);
  return data;
};

export const listRetrievalEvaluations = async (limit: number = 20) => {
  const { data } = await getApi().get<RetrievalEvalRunsResponse>("/api/rag/evaluations", {
    params: { limit },
  });
  return data;
};

export const getRetrievalEvaluation = async (runId: number) => {
  const { data } = await getApi().get<RetrievalEvalResponse>(`/api/rag/evaluations/${runId}`);
  return data;
};

export const deleteRetrievalEvaluation = async (runId: number) => {
  const { data } = await getApi().delete<DeleteRetrievalEvalRunResponse>(`/api/rag/evaluations/${runId}`);
  return data;
};
