<template>
  <div class="empty">
    <div class="icon">✦</div>
    <div>{{ text }}</div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  text: string;
}>();
</script>

<style scoped>
.empty {
  padding: 24px;
  text-align: center;
  color: #64748b;
  border: 1px dashed var(--border);
  border-radius: 12px;
}
.icon {
  font-size: 22px;
  margin-bottom: 8px;
}
</style>
