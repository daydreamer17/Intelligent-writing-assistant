import { getApi } from "./api";
import type { CitationRequest, CitationResponse } from "../types";

export const buildCitations = async (payload: CitationRequest) => {
  const { data } = await getApi().post<CitationResponse>("/api/citations", payload);
  return data;
};
