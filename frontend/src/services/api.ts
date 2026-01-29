import type { AxiosInstance } from "axios";
import { apiClient } from "../utils/request";
import { useAppStore } from "../store";

let client: AxiosInstance | null = null;
let currentBaseURL: string | null = null;

export const getApi = () => {
  const store = useAppStore();

  // 如果baseURL变化，重置客户端
  if (!client || currentBaseURL !== store.apiBase) {
    client = apiClient();
    currentBaseURL = store.apiBase;
  }

  return client;
};
