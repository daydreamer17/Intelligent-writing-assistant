<template>
  <section class="card">
    <h3>文本上传</h3>
    <div class="field">
      <label>文档标题</label>
      <input v-model="doc.title" />
    </div>
    <div class="field">
      <label>内容</label>
      <textarea v-model="doc.content" rows="8"></textarea>
    </div>
    <button class="btn" @click="handleTextUpload" :disabled="loading.uploadText">上传文本</button>

    <h3 style="margin-top:16px;">文件上传</h3>
    <input type="file" multiple @change="handleFileSelect" />
    <div class="muted">{{ fileNames }}</div>
    <button class="btn secondary" @click="handleFileUpload" :disabled="loading.uploadFiles">上传文件</button>
  </section>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { uploadDocuments, uploadFiles } from "../services/rag";
import { handleApiError } from "../utils/errorHandler";
import { validateFiles } from "../utils/validation";

const emit = defineEmits<{
  (e: "uploaded"): void;
  (e: "error", message: string): void;
}>();

const doc = reactive({
  title: "",
  content: "",
});

const files = ref<File[]>([]);
const loading = reactive({
  uploadText: false,
  uploadFiles: false,
});

const fileNames = computed(() => files.value.map((f) => f.name).join(", "));

const createDocId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = Math.random() * 16;
    const v = c === "x" ? r : (r % 4) + 8;
    return Math.floor(v).toString(16);
  });
};

const handleTextUpload = async () => {
  try {
    loading.uploadText = true;
    await uploadDocuments({
      documents: [
        {
          doc_id: createDocId(),
          title: doc.title || "untitled",
          content: doc.content,
          url: "",
        },
      ],
    });
    doc.title = "";
    doc.content = "";
    emit("uploaded");
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.uploadText = false;
  }
};

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  files.value = target.files ? Array.from(target.files) : [];
};

const handleFileUpload = async () => {
  try {
    // 文件验证
    const validation = validateFiles(files.value);
    if (!validation.valid) {
      emit("error", validation.errors.join('; '));
      return;
    }

    loading.uploadFiles = true;
    await uploadFiles(files.value);
    files.value = [];
    emit("uploaded");
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.uploadFiles = false;
  }
};
</script>
