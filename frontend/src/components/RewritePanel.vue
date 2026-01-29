<template>
  <section class="card">
    <h3>改写</h3>
    <div class="markdown" v-text="revised" />
  </section>
</template>

<script setup lang="ts">
defineProps<{
  revised: string;
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
</style>
