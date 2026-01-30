import axios from "axios";
import { useAppStore } from "../store";

export const apiClient = () => {
  const store = useAppStore();
  return axios.create({
    baseURL: store.apiBase,
    timeout: 600000, // 10分钟超时，适应LLM重试和长时间生成
    headers: {},
  });
};
