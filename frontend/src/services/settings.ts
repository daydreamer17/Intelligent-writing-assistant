import { getApi } from "./api";

export interface CitationSettingResponse {
  enabled: boolean;
}

export interface SessionMemoryClearRequest {
  session_id?: string;
  drop_agent?: boolean;
  clear_cold?: boolean;
}

export interface SessionMemoryClearResponse {
  session_id: string;
  cleared: boolean;
  cleared_agents: string[];
}

export const getCitationSetting = async () => {
  const { data } = await getApi().get<CitationSettingResponse>("/api/settings/citation");
  return data;
};

export const setCitationSetting = async (enabled: boolean) => {
  const { data } = await getApi().post<CitationSettingResponse>("/api/settings/citation", {
    enabled,
  });
  return data;
};

export const clearSessionMemory = async (payload: SessionMemoryClearRequest) => {
  const { data } = await getApi().post<SessionMemoryClearResponse>(
    "/api/settings/session-memory/clear",
    payload
  );
  return data;
};
