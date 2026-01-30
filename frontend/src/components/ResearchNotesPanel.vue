<template>
  <section class="card">
    <h3>RAG 命中文档</h3>
    <div class="grid-2" v-if="notes.length">
      <div class="card item" v-for="note in notes" :key="note.doc_id">
        <h4>{{ note.title || note.doc_id }}</h4>
        <p class="muted id">ID: {{ note.doc_id }}</p>
        <pre class="snippet">{{ note.summary }}</pre>
        <button class="btn ghost" @click="useSnippet(note.summary)">加入工作台</button>
      </div>
    </div>
    <EmptyState v-else text="暂无命中文档" />
  </section>
</template>

<script setup lang="ts">
import type { ResearchNoteResponse } from "../types";
import { useAppStore } from "../store";
import { EmptyState } from "./index";

const props = defineProps<{
  notes: ResearchNoteResponse[];
}>();

const store = useAppStore();
const useSnippet = (text: string) => {
  store.addSnippet(text);
};
</script>

<style scoped>
.item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.item h4 {
  margin: 0;
  font-size: 15px;
}
.id {
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
  max-height: 180px;
  overflow: auto;
  padding-right: 4px;
}
</style>
