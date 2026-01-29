import { defineStore } from "pinia";

export const useAppStore = defineStore("app", {
  state: () => ({
    apiBase: "http://localhost:8000",
    ragSnippets: [] as string[],
  }),
  actions: {
    setApiBase(value: string) {
      this.apiBase = value;
    },
    addSnippet(text: string) {
      if (text && !this.ragSnippets.includes(text)) {
        this.ragSnippets.push(text);
      }
    },
    clearSnippets() {
      this.ragSnippets = [];
    },
  },
  persist: {
    key: 'my-agent-store',
    storage: localStorage,
    paths: ['apiBase', 'ragSnippets'],
  },
});
