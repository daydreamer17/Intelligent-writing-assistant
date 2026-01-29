import { getApi } from "./api";
import type { HealthDetailResponse } from "../types";

export const getHealthDetail = async () => {
  const { data } = await getApi().get<HealthDetailResponse>("/healthz/detail");
  return data;
};
