<template>
  <section class="page">
    <div class="card">
      <h2>版本历史</h2>
      <p class="muted">查看、对比与删除历史稿件。</p>
    </div>

    <div class="card">
      <div class="grid-2">
        <div class="field">
          <label>筛选关键词</label>
          <input v-model="keyword" placeholder="主题/内容关键词" />
        </div>
        <div class="field">
          <label>Diff 字段</label>
          <select v-model="diffField">
            <option value="revised">revised</option>
            <option value="draft">draft</option>
            <option value="review">review</option>
            <option value="outline">outline</option>
          </select>
        </div>
        <div class="field">
          <label>对比版本 ID（可选）</label>
          <input v-model="compareTo" placeholder="例如：12" />
        </div>
      </div>
    </div>

    <VersionList
      :versions="paged"
      :page="page"
      :total-pages="totalPages"
      @refresh="loadVersions"
      @prev="prevPage"
      @next="nextPage"
      @select="selectVersion"
      @diff="showDiff"
      @delete="removeVersion"
    />

    <div class="card" v-if="detail">
      <h3>版本详情</h3>
      <pre>{{ detail.draft }}</pre>
    </div>

    <DiffViewer v-if="diffText" :diff="diffText" />
    <EmptyState v-else text="暂无差异内容" />
    <Toast :message="toast" />
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { deleteVersion, diffVersion, getVersion, listVersions } from "../services/versions";
import { handleApiError } from "../utils/errorHandler";
import type { DraftVersionResponse } from "../types";
import VersionList from "../components/VersionList.vue";
import DiffViewer from "../components/DiffViewer.vue";
import { EmptyState } from "../components";
import Toast from "../components/Toast.vue";

const versions = ref<DraftVersionResponse[]>([]);
const detail = ref<DraftVersionResponse | null>(null);
const diffText = ref("");
const loading = ref(false);
const keyword = ref("");
const diffField = ref("revised");
const toast = ref("");
const error = ref("");
const page = ref(1);
const pageSize = 5;
const compareTo = ref("");

const filtered = computed(() => {
  if (!keyword.value) return versions.value;
  const key = keyword.value.toLowerCase();
  return versions.value.filter((item) =>
    [item.topic, item.outline, item.draft, item.review, item.revised].some((v) =>
      (v || "").toLowerCase().includes(key)
    )
  );
});

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / pageSize)));
const paged = computed(() => {
  const start = (page.value - 1) * pageSize;
  return filtered.value.slice(start, start + pageSize);
});

const loadVersions = async () => {
  try {
    loading.value = true;
    error.value = "";
    const res = await listVersions(200);
    versions.value = res.versions;
    page.value = 1;
  } catch (err) {
    error.value = handleApiError(err);
    toast.value = error.value;
    setTimeout(() => (toast.value = ""), 3000);
  } finally {
    loading.value = false;
  }
};

const selectVersion = async (versionId: number) => {
  try {
    error.value = "";
    const res = await getVersion(versionId);
    detail.value = res.version;
  } catch (err) {
    error.value = handleApiError(err);
    toast.value = error.value;
    setTimeout(() => (toast.value = ""), 3000);
  }
};

const showDiff = async (versionId: number) => {
  try {
    error.value = "";
    const compareId = compareTo.value ? Number(compareTo.value) : undefined;
    const res = await diffVersion(versionId, diffField.value, compareId);
    diffText.value = res.diff;
  } catch (err) {
    error.value = handleApiError(err);
    toast.value = error.value;
    setTimeout(() => (toast.value = ""), 3000);
  }
};

const removeVersion = async (versionId: number) => {
  try {
    error.value = "";
    await deleteVersion(versionId);
    await loadVersions();
    toast.value = "已删除版本";
    setTimeout(() => (toast.value = ""), 2000);
  } catch (err) {
    error.value = handleApiError(err);
    toast.value = error.value;
    setTimeout(() => (toast.value = ""), 3000);
  }
};

const nextPage = () => {
  page.value = Math.min(totalPages.value, page.value + 1);
};
const prevPage = () => {
  page.value = Math.max(1, page.value - 1);
};

onMounted(loadVersions);
</script>
