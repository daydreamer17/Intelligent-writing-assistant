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

export const uploadFiles = async (files: File[]) => {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const { data } = await getApi().post<UploadDocumentsResponse>("/api/rag/upload-file", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};
