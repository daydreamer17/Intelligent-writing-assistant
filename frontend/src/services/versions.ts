import { getApi } from "./api";
import type {
  DeleteVersionResponse,
  VersionDetailResponse,
  VersionDiffResponse,
  VersionsResponse,
} from "../types";

export const listVersions = async (limit?: number) => {
  const { data } = await getApi().get<VersionsResponse>("/api/versions", {
    params: { limit },
  });
  return data;
};

export const getVersion = async (versionId: number) => {
  const { data } = await getApi().get<VersionDetailResponse>(`/api/versions/${versionId}`);
  return data;
};

export const deleteVersion = async (versionId: number) => {
  const { data } = await getApi().delete<DeleteVersionResponse>(`/api/versions/${versionId}`);
  return data;
};

export const diffVersion = async (versionId: number, field = "revised", compareTo?: number) => {
  const { data } = await getApi().get<VersionDiffResponse>(`/api/versions/${versionId}/diff`, {
    params: { field, compare_to: compareTo },
  });
  return data;
};
