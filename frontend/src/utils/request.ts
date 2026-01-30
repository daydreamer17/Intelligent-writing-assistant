import axios from "axios";
import { useAppStore } from "../store";

export const apiClient = () => {
  const store = useAppStore();
  return axios.create({
    baseURL: store.apiBase,
    timeout: 120000,
    headers: {},
  });
};
