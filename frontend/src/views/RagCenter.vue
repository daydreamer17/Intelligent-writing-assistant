<template>
  <section class="page">
    <div class="card">
      <h2>RAG 素材中心</h2>
      <p class="muted">支持 TXT / PDF / DOCX / MD 上传与检索。</p>
    </div>

    <div class="grid-2">
      <RagUploader @uploaded="handleUploaded" @error="setError" />
      <RagSearch @error="setError" />
    </div>
    <RagLibrary ref="libraryRef" @error="setError" />

    <div v-if="error" class="muted">错误：{{ error }}</div>
    <Toast :message="toast" />
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import RagUploader from "../components/RagUploader.vue";
import RagSearch from "../components/RagSearch.vue";
import RagLibrary from "../components/RagLibrary.vue";
import Toast from "../components/Toast.vue";
const error = ref("");
const toast = ref("");
const libraryRef = ref<InstanceType<typeof RagLibrary> | null>(null);
const handleUploaded = () => {
  error.value = "";
  toast.value = "上传完成";
  setTimeout(() => (toast.value = ""), 2000);
  libraryRef.value?.refresh();
};
const setError = (message: string) => {
  error.value = message;
};
</script>
