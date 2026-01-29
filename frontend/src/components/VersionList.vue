<template>
  <section class="card">
    <h3>版本列表</h3>
    <button class="btn ghost" @click="$emit('refresh')">刷新列表</button>
    <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
      <button class="btn ghost" @click="$emit('prev')" :disabled="page <= 1">上一页</button>
      <div class="muted">第 {{ page }} / {{ totalPages }} 页</div>
      <button class="btn ghost" @click="$emit('next')" :disabled="page >= totalPages">下一页</button>
    </div>
    <div class="grid-2">
      <div class="card" v-for="item in versions" :key="item.version_id">
        <h4>版本 {{ item.version_id }}</h4>
        <p class="muted">{{ item.created_at }}</p>
        <p>{{ item.topic }}</p>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
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
