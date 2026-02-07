import { getApi } from "./api";

export interface CitationSettingResponse {
  enabled: boolean;
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
