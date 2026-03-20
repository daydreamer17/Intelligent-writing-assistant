<template>
  <section class="page">
    <ApiBaseSwitcher />

    <StatusBar :data="health" />

    <div class="card settings-actions">
      <button class="btn ghost" @click="refreshAll" :disabled="loading.health || loading.checkpoints || cleanupRunning">
        刷新
      </button>
    </div>

    <div class="card">
      <div class="checkpoint-card__header">
        <div>
          <h3>LangGraph v2 Checkpoints</h3>
          <p class="muted">最近 10 条 checkpoint，用于查看、复制 thread_id、删除和清理过期 completed 记录。</p>
        </div>
        <button class="btn secondary" @click="cleanupCompleted" :disabled="cleanupRunning || loading.checkpoints">
          {{ cleanupRunning ? "清理中..." : "清理过期 completed" }}
        </button>
      </div>

      <div v-if="checkpointMessage" class="muted">{{ checkpointMessage }}</div>
      <div v-if="checkpointError" class="muted">错误：{{ checkpointError }}</div>
      <div v-if="loading.checkpoints" class="muted">正在加载 checkpoints...</div>

      <div v-else-if="checkpoints.length" class="checkpoint-list">
        <div v-for="item in checkpoints" :key="item.thread_id" class="checkpoint-item">
          <div class="checkpoint-item__main">
            <div><strong>thread_id</strong>：{{ item.thread_id }}</div>
            <div class="muted">状态：{{ item.status }} | 阶段：{{ item.current_stage || '-' }}</div>
            <div class="muted">更新时间：{{ item.updated_at || '-' }}</div>
          </div>
          <div class="checkpoint-item__actions">
            <button class="btn ghost" @click="copyThreadId(item.thread_id)">复制 thread_id</button>
            <button class="btn ghost" @click="deleteCheckpointItem(item.thread_id)">删除</button>
          </div>
        </div>
      </div>
      <div v-else class="muted">当前没有可展示的 LangGraph v2 checkpoint。</div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { getHealthDetail } from "../services/health";
import {
  cleanupPipelineV2Checkpoints,
  deletePipelineV2Checkpoint,
  listPipelineV2Checkpoints,
} from "../services/pipeline";
import type { PipelineV2CheckpointSummary } from "../types";
import { handleApiError } from "../utils/errorHandler";
import StatusBar from "../components/StatusBar.vue";
import ApiBaseSwitcher from "../components/ApiBaseSwitcher.vue";

const health = ref(null as any);
const checkpoints = ref<PipelineV2CheckpointSummary[]>([]);
const checkpointMessage = ref("");
const checkpointError = ref("");
const cleanupRunning = ref(false);
const loading = ref({
  health: false,
  checkpoints: false,
});

const setMessage = (value: string) => {
  checkpointMessage.value = value;
  if (value) {
    setTimeout(() => {
      if (checkpointMessage.value === value) {
        checkpointMessage.value = "";
      }
    }, 2500);
  }
};

const loadHealth = async () => {
  loading.value.health = true;
  try {
    health.value = await getHealthDetail();
  } catch {
    health.value = null;
  } finally {
    loading.value.health = false;
  }
};

const loadCheckpoints = async () => {
  loading.value.checkpoints = true;
  checkpointError.value = "";
  try {
    const res = await listPipelineV2Checkpoints({ limit: 10 });
    checkpoints.value = res.checkpoints || [];
  } catch (err) {
    checkpointError.value = handleApiError(err);
    checkpoints.value = [];
  } finally {
    loading.value.checkpoints = false;
  }
};

const refreshAll = async () => {
  await Promise.all([loadHealth(), loadCheckpoints()]);
};

const copyThreadId = async (threadId: string) => {
  try {
    await navigator.clipboard.writeText(threadId);
    setMessage("thread_id 已复制");
  } catch {
    checkpointError.value = "复制 thread_id 失败，请手动复制。";
  }
};

const deleteCheckpointItem = async (threadId: string) => {
  checkpointError.value = "";
  try {
    const res = await deletePipelineV2Checkpoint(threadId);
    if (res.deleted) {
      checkpoints.value = checkpoints.value.filter((item) => item.thread_id !== threadId);
      setMessage("checkpoint 已删除");
      return;
    }
    checkpointError.value = "未删除任何 checkpoint。";
  } catch (err) {
    checkpointError.value = handleApiError(err);
  }
};

const cleanupCompleted = async () => {
  cleanupRunning.value = true;
  checkpointError.value = "";
  try {
    const res = await cleanupPipelineV2Checkpoints({});
    setMessage(`cleanup 完成：匹配 ${res.matched} 条，删除 ${res.deleted} 条`);
    await loadCheckpoints();
  } catch (err) {
    checkpointError.value = handleApiError(err);
  } finally {
    cleanupRunning.value = false;
  }
};

onMounted(() => {
  void refreshAll();
});
</script>

<style scoped>
.settings-actions {
  display: flex;
  justify-content: flex-end;
}

.checkpoint-card__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.checkpoint-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.checkpoint-item {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 12px;
  padding: 12px 14px;
  flex-wrap: wrap;
}

.checkpoint-item__main {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.checkpoint-item__actions {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  flex-wrap: wrap;
}
</style>
