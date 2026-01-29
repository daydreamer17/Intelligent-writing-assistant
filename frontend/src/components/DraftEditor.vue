<template>
  <section class="card">
    <div class="row">
      <h3>草稿</h3>
      <button class="btn ghost" @click="emit('update', localDraft)">保存编辑</button>
    </div>
    <textarea v-model="localDraft" rows="14"></textarea>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{
  draft: string;
}>();

const emit = defineEmits<{
  (e: "update", value: string): void;
}>();

const localDraft = ref(props.draft);

watch(
  () => props.draft,
  (value) => {
    localDraft.value = value;
  }
);
</script>

<style scoped>
.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
textarea {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  font: inherit;
}
</style>
