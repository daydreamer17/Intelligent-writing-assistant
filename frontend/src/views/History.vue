<template>
  <section class="page">
    <div class="card">
      <h2>版本历史</h2>
      <p class="muted">查看、对比与删除历史稿件。</p>
    </div>

    <div class="card">
      <div class="grid-2 filter-grid">
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
      <div class="section-header">
        <h3>版本详情</h3>
        <button class="btn ghost" @click="detailCollapsed = !detailCollapsed">
          {{ detailCollapsed ? "展开详情" : "收起详情" }}
        </button>
      </div>
      <pre v-if="!detailCollapsed" class="detail-content">{{ detail.draft }}</pre>
      <p v-else class="muted">已收起版本详情。</p>
    </div>

    <div class="card" v-if="diffText">
      <div class="section-header">
        <h3>差异对比</h3>
        <button class="btn ghost" @click="diffCollapsed = !diffCollapsed">
          {{ diffCollapsed ? "展开差异" : "收起差异" }}
        </button>
      </div>
      <DiffViewer v-if="!diffCollapsed" :diff="diffText" :show-header="false" />
      <p v-else class="muted">已收起差异详情。</p>
    </div>
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
const detailCollapsed = ref(false);
const diffCollapsed = ref(false);

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
    detailCollapsed.value = false;
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
    diffCollapsed.value = false;
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

<style scoped>
.page {
  width: 100%;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.section-header h3 {
  margin: 0;
}

.filter-grid {
  grid-template-columns: 1.4fr 1fr 0.9fr;
}

.detail-content {
  margin-top: 12px;
  font-size: 14px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #f8fafc;
}

@media (max-width: 1100px) {
  .filter-grid {
    grid-template-columns: 1fr;
  }
}
</style>
