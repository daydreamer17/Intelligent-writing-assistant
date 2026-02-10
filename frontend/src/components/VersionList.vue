<template>
  <section class="card">
    <div class="list-header">
      <h3>版本列表</h3>
      <button class="btn ghost" @click="$emit('refresh')">刷新列表</button>
    </div>
    <div class="pager-row">
      <button class="btn ghost" @click="$emit('prev')" :disabled="page <= 1">上一页</button>
      <div class="muted">第 {{ page }} / {{ totalPages }} 页</div>
      <button class="btn ghost" @click="$emit('next')" :disabled="page >= totalPages">下一页</button>
    </div>
    <div class="version-grid">
      <div class="card version-item" v-for="item in versions" :key="item.version_id">
        <h4>版本 {{ item.version_id }}</h4>
        <p class="muted">{{ item.created_at }}</p>
        <p class="version-topic">{{ item.topic }}</p>
        <div class="version-actions">
          <button class="btn ghost" @click="$emit('select', item.version_id)">详情</button>
          <button class="btn ghost" @click="$emit('diff', item.version_id)">差异</button>
          <button class="btn" @click="$emit('delete', item.version_id)">删除</button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { DraftVersionResponse } from "../types";

defineProps<{
  versions: DraftVersionResponse[];
  page: number;
  totalPages: number;
}>();

defineEmits<{
  (e: "refresh"): void;
  (e: "prev"): void;
  (e: "next"): void;
  (e: "select", id: number): void;
  (e: "diff", id: number): void;
  (e: "delete", id: number): void;
}>();
</script>

<style scoped>
.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.list-header h3 {
  margin: 0;
}

.pager-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

.version-grid {
  display: grid;
  gap: 14px;
  margin-top: 12px;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}

.version-item {
  padding: 14px;
  font-size: 14px;
}

.version-item h4 {
  margin: 0 0 6px 0;
  font-size: 16px;
}

.version-topic {
  margin: 10px 0 8px 0;
  font-size: 14px;
  line-height: 1.55;
}

.version-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

@media (max-width: 900px) {
  .version-grid {
    grid-template-columns: 1fr;
  }
}
</style>
