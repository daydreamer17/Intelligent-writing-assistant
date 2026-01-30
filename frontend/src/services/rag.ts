import { getApi } from "./api";
import type {
  SearchDocumentsRequest,
  SearchDocumentsResponse,
  UploadDocumentsRequest,
  UploadDocumentsResponse,
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

export const uploadFiles = async (files: File[]) => {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  const { data } = await getApi().post<UploadDocumentsResponse>("/api/rag/upload-file", form);
  return data;
};
