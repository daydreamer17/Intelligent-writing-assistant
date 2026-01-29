<template>
  <section class="card">
    <h3>大纲</h3>
    <div class="markdown" v-text="outline" />
    <div class="meta">
      <span class="label">假设：</span>
      <span class="text" v-text="assumptions" />
    </div>
    <div class="meta">
      <span class="label">待确认问题：</span>
      <span class="text" v-text="openQuestions" />
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  outline: string;
  assumptions: string;
  openQuestions: string;
}>();
</script>

<style scoped>
.markdown {
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.7;
  font-size: 14px;
  max-height: 560px;
  overflow: auto;
  padding-right: 4px;
}
.meta {
  margin-top: 12px;
  color: var(--muted);
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.6;
}
.label {
  font-weight: 600;
}
</style>
