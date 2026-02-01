<template>
  <section class="card">
    <div class="header">
      <h3>已上传素材</h3>
      <button class="btn ghost" @click="refresh" :disabled="loading">刷新</button>
    </div>
    <div class="grid-2" v-if="uniqueDocuments.length">
      <div class="card item" v-for="item in uniqueDocuments" :key="item.doc_id">
        <h4>{{ item.title }}</h4>
        <p class="muted url">{{ item.url }}</p>
        <pre class="snippet">{{ item.content }}</pre>
        <div class="actions">
          <button class="btn ghost" @click="useSnippet(item.content)">加入工作台</button>
          <button class="btn ghost danger" @click="handleDelete(item.doc_id)">移除</button>
        </div>
      </div>
    </div>
    <EmptyState v-else text="暂无上传内容" />
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { deleteDocument, listDocuments } from "../services/rag";
import { useAppStore } from "../store";
import { handleApiError } from "../utils/errorHandler";
import type { SourceDocumentResponse } from "../types";
import { EmptyState } from "./index";

const emit = defineEmits<{
  (e: "error", message: string): void;
}>();

const documents = ref<SourceDocumentResponse[]>([]);
const loading = ref(false);
const store = useAppStore();

const refresh = async () => {
  try {
    loading.value = true;
    const res = await listDocuments(100);
    documents.value = res.documents || [];
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.value = false;
  }
};

const uniqueDocuments = computed(() => {
  const seen = new Set<string>();
  const deduped: SourceDocumentResponse[] = [];
  for (const doc of documents.value) {
    const key = `${doc.title}::${doc.content}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(doc);
  }
  return deduped;
});

const useSnippet = (text: string) => {
  store.addSnippet(text);
};

const handleDelete = async (docId: string) => {
  try {
    await deleteDocument(docId);
    documents.value = documents.value.filter((doc) => doc.doc_id !== docId);
  } catch (err) {
    emit("error", handleApiError(err));
  }
};

onMounted(() => {
  refresh();
});

defineExpose({ refresh });
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.item h4 {
  margin: 0;
  font-size: 15px;
  line-height: 1.4;
  word-break: break-word;
  overflow-wrap: anywhere;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
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
  max-height: 200px;
  overflow: auto;
  padding-right: 4px;
}
.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.danger {
  border-color: #fca5a5;
  color: #b91c1c;
}
</style>
