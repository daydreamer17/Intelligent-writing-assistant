<template>
  <section class="card">
    <h3>检索</h3>
    <div class="field">
      <label>查询</label>
      <input v-model="query" />
    </div>
    <button class="btn ghost" @click="handleSearch" :disabled="loading">搜索</button>
    <div class="grid-2 results" v-if="pagedResults.length">
      <div class="card result-card" v-for="item in pagedResults" :key="item.doc_id">
        <h4>{{ item.title }}</h4>
        <p class="muted url">{{ item.url }}</p>
        <pre class="snippet" v-html="highlight(item.content)"></pre>
        <button class="btn ghost" @click="useSnippet(item.content)">加入工作台</button>
      </div>
    </div>
    <EmptyState v-else text="暂无检索结果" />
    <div style="display:flex; gap:8px; align-items:center;">
      <button class="btn ghost" @click="prevPage" :disabled="page === 1">上一页</button>
      <div class="muted">第 {{ page }} / {{ totalPages }} 页</div>
      <button class="btn ghost" @click="nextPage" :disabled="page >= totalPages">下一页</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { useAppStore } from "../store";
import { searchDocuments } from "../services/rag";
import { handleApiError } from "../utils/errorHandler";
import type { SourceDocumentResponse } from "../types";
import { EmptyState } from "./index";

const emit = defineEmits<{
  (e: "error", message: string): void;
}>();

const query = ref("");
const results = ref<SourceDocumentResponse[]>([]);
const loading = ref(false);
const store = useAppStore();
const page = ref(1);
const pageSize = 4;

const totalPages = computed(() => Math.max(1, Math.ceil(results.value.length / pageSize)));
const pagedResults = computed(() => {
  const start = (page.value - 1) * pageSize;
  return results.value.slice(start, start + pageSize);
});

const handleSearch = async () => {
  try {
    loading.value = true;
    const res = await searchDocuments({ query: query.value, top_k: 5 });
    results.value = res.documents;
    page.value = 1;
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.value = false;
  }
};

const useSnippet = (text: string) => {
  store.addSnippet(text);
};

const nextPage = () => {
  page.value = Math.min(totalPages.value, page.value + 1);
};
const prevPage = () => {
  page.value = Math.max(1, page.value - 1);
};

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const highlight = (text: string) => {
  if (!query.value) return escapeHtml(text);
  const safe = escapeHtml(text);
  const pattern = new RegExp(query.value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  return safe.replace(pattern, (match) => `<mark>${match}</mark>`);
};
</script>

<style scoped>
.results {
  align-items: start;
}
.result-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.result-card h4 {
  margin: 0;
  font-size: 15px;
}
.url {
  margin: 0;
  font-size: 12px;
  word-break: break-all;
}
.snippet {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.6;
  font-size: 13px;
  max-height: 220px;
  overflow: auto;
  padding-right: 4px;
}
mark {
  background: rgba(180, 83, 9, 0.2);
  color: inherit;
}
</style>
