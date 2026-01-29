<template>
  <div class="overlay" v-if="active">
    <div class="spinner"></div>
    <div>{{ text }}</div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  active: boolean;
  text?: string;
}>();
</script>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.35);
  display: grid;
  place-items: center;
  color: #fff;
  z-index: 998;
  font-size: 16px;
}
.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 12px;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
