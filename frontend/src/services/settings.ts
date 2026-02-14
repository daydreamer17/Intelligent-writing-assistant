import { getApi } from "./api";

export interface CitationSettingResponse {
  enabled: boolean;
}

export type GenerationMode = "rag_only" | "hybrid" | "creative";

export interface GenerationModeSettingResponse {
  mode: GenerationMode;
  citation_enforce: boolean;
  mcp_allowed: boolean;
  inference_mark_required: boolean;
  creative_mcp_enabled: boolean;
}

export interface GenerationModeSettingRequest {
  mode: GenerationMode;
  creative_mcp_enabled?: boolean;
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

export const getGenerationMode = async () => {
  const { data } = await getApi().get<GenerationModeSettingResponse>(
    "/api/settings/generation-mode"
  );
  return data;
};

export const setGenerationMode = async (
  mode: GenerationMode,
  creativeMcpEnabled?: boolean
) => {
  const payload: GenerationModeSettingRequest = { mode };
  if (typeof creativeMcpEnabled === "boolean") {
    payload.creative_mcp_enabled = creativeMcpEnabled;
  }
  const { data } = await getApi().post<GenerationModeSettingResponse>(
    "/api/settings/generation-mode",
    payload
  );
  return data;
};

export const clearSessionMemory = async (payload: SessionMemoryClearRequest) => {
  const { data } = await getApi().post<SessionMemoryClearResponse>(
    "/api/settings/session-memory/clear",
    payload
  );
  return data;
};
