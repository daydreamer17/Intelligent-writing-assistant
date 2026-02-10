<template>
  <section class="card">
    <div class="header">
      <h3>离线检索评测</h3>
      <button class="btn ghost" @click="resetAll" :disabled="loading">清空</button>
    </div>
    <p class="muted tip">
      支持上传 JSON / JSONL 标注集，字段示例：query, relevant_doc_ids, query_id。
    </p>

    <div class="field">
      <label>上传标注集</label>
      <input type="file" accept=".json,.jsonl,.txt" @change="handleFileSelect" />
      <div class="muted">{{ fileName || "未选择文件" }}</div>
    </div>

    <div class="field">
      <label>或粘贴标注集内容</label>
      <textarea
        v-model="rawDataset"
        rows="6"
        placeholder='{"cases":[{"query":"agent","relevant_doc_ids":["d1","d2"]}],"k_values":[1,3,5]}'
      />
    </div>

    <div class="grid-2 control-grid">
      <div class="field">
        <label>K 值（逗号分隔）</label>
        <input v-model="kInput" placeholder="1,3,5,10" />
      </div>
      <div class="field">
        <label>失败样本判定 K</label>
        <select v-model.number="failureK">
          <option v-for="k in availableKs" :key="k" :value="k">K={{ k }}</option>
        </select>
      </div>
    </div>

    <button class="btn" @click="runEvaluation" :disabled="loading">运行评测</button>
    <div v-if="report?.eval_run_id" class="muted saved-note">
      已保存评测结果：#{{ report.eval_run_id }} {{ report.created_at ? `(${report.created_at})` : "" }}
    </div>

    <div class="history-card">
      <div class="history-header">
        <h4>评测历史</h4>
        <button class="btn ghost" @click="refreshHistory" :disabled="historyLoading">刷新历史</button>
      </div>
      <div v-if="history.length" class="history-list">
        <div class="history-item" v-for="row in history" :key="row.run_id">
          <div>
            <div class="history-title">#{{ row.run_id }} {{ row.created_at || "-" }}</div>
            <div class="muted">
              queries={{ row.total_queries }} / labeled={{ row.queries_with_relevance }} / K={{ row.k_values.join(",") }}
            </div>
          </div>
          <div class="history-actions">
            <button class="btn ghost" @click="loadRun(row.run_id)">加载</button>
            <button class="btn ghost danger" @click="removeRun(row.run_id)">删除</button>
          </div>
        </div>
      </div>
      <div v-else class="muted">暂无历史评测结果。</div>
    </div>

    <div v-if="report" class="result-wrap">
      <div class="result-header">
        <h4>评测结果详情</h4>
        <button class="btn ghost" @click="reportCollapsed = !reportCollapsed">
          {{ reportCollapsed ? "展开详情" : "收起详情" }}
        </button>
      </div>
      <p v-if="reportCollapsed" class="muted">已收起评测详情。</p>
      <template v-else>
        <div class="metric-grid">
          <div class="metric-box">
            <div class="label">总 Query</div>
            <div class="value">{{ report.total_queries }}</div>
          </div>
          <div class="metric-box">
            <div class="label">有标注 Query</div>
            <div class="value">{{ report.queries_with_relevance }}</div>
          </div>
          <div class="metric-box">
            <div class="label">失败样本</div>
            <div class="value">{{ failedQueries.length }}</div>
          </div>
        </div>

        <div class="chart-card">
          <div class="legend">
            <span v-for="item in legend" :key="item.key" class="legend-item">
              <i :style="{ backgroundColor: item.color }"></i>{{ item.name }}
            </span>
          </div>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" class="chart">
            <line :x1="padding" :y1="padding" :x2="padding" :y2="chartHeight - padding" class="axis" />
            <line
              :x1="padding"
              :y1="chartHeight - padding"
              :x2="chartWidth - padding"
              :y2="chartHeight - padding"
              class="axis"
            />
            <line
              v-for="tick in [0, 0.25, 0.5, 0.75, 1]"
              :key="`y-${tick}`"
              :x1="padding"
              :y1="yScale(tick)"
              :x2="chartWidth - padding"
              :y2="yScale(tick)"
              class="grid-line"
            />
            <polyline
              v-for="item in legend"
              :key="item.key"
              :points="pointsFor(item.key)"
              fill="none"
              :stroke="item.color"
              stroke-width="2"
            />
            <text
              v-for="k in availableKs"
              :key="`x-${k}`"
              :x="xScale(k)"
              :y="chartHeight - 8"
              class="axis-text"
              text-anchor="middle"
            >
              {{ k }}
            </text>
          </svg>
        </div>

        <table class="metric-table">
          <thead>
            <tr>
              <th>K</th>
              <th>Recall</th>
              <th>Precision</th>
              <th>HitRate</th>
              <th>MRR</th>
              <th>nDCG</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in report.macro_metrics" :key="row.k">
              <td>{{ row.k }}</td>
              <td>{{ toPct(row.recall) }}</td>
              <td>{{ toPct(row.precision) }}</td>
              <td>{{ toPct(row.hit_rate) }}</td>
              <td>{{ row.mrr.toFixed(3) }}</td>
              <td>{{ row.ndcg.toFixed(3) }}</td>
            </tr>
          </tbody>
        </table>

        <div class="fail-header">
          <h4>失败样本（K={{ failureK }}）</h4>
          <span class="muted">判定规则：hit_rate&lt;1 或 recall&lt;1</span>
        </div>
        <div v-if="failedQueries.length" class="fail-list">
          <div class="fail-item" v-for="row in failedQueries" :key="`${row.query_id}-${row.query}`">
            <div class="fail-title">
              {{ row.query_id || "-" }} | {{ row.query }}
            </div>
            <div class="muted">
              relevant={{ row.relevant_count }}; retrieved={{ row.retrieved_doc_ids.join(", ") || "-" }}
            </div>
            <div class="muted">
              recall={{ selectedMetric(row)?.recall.toFixed(3) || "-" }},
              hit_rate={{ selectedMetric(row)?.hit_rate.toFixed(3) || "-" }}
            </div>
          </div>
        </div>
        <div v-else class="muted">当前 K 下无失败样本。</div>
      </template>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  deleteRetrievalEvaluation,
  evaluateRetrieval,
  getRetrievalEvaluation,
  listRetrievalEvaluations,
} from "../services/rag";
import { handleApiError } from "../utils/errorHandler";
import type {
  RetrievalEvalCaseInput,
  RetrievalEvalCaseResult,
  RetrievalEvalRunSummaryResponse,
  RetrievalEvalResponse,
} from "../types";

const emit = defineEmits<{
  (e: "error", message: string): void;
}>();

type MetricKey = "recall" | "precision" | "hit_rate" | "mrr" | "ndcg";

const chartWidth = 760;
const chartHeight = 280;
const padding = 36;

const loading = ref(false);
const rawDataset = ref("");
const fileName = ref("");
const kInput = ref("1,3,5");
const report = ref<RetrievalEvalResponse | null>(null);
const reportCollapsed = ref(false);
const failureK = ref(1);
const history = ref<RetrievalEvalRunSummaryResponse[]>([]);
const historyLoading = ref(false);

const legend: { key: MetricKey; name: string; color: string }[] = [
  { key: "recall", name: "Recall", color: "#b45309" },
  { key: "precision", name: "Precision", color: "#0f766e" },
  { key: "hit_rate", name: "HitRate", color: "#0284c7" },
  { key: "mrr", name: "MRR", color: "#7c3aed" },
  { key: "ndcg", name: "nDCG", color: "#dc2626" },
];

const availableKs = computed(() => report.value?.k_values ?? parseKValues(kInput.value));

const xScale = (k: number): number => {
  const ks = availableKs.value;
  const idx = ks.indexOf(k);
  if (idx < 0 || ks.length <= 1) {
    return chartWidth / 2;
  }
  return padding + (idx / (ks.length - 1)) * (chartWidth - padding * 2);
};

const yScale = (v: number): number => {
  const clamped = Math.max(0, Math.min(1, v));
  return chartHeight - padding - clamped * (chartHeight - padding * 2);
};

const pointsFor = (metric: MetricKey): string => {
  if (!report.value) return "";
  return report.value.macro_metrics
    .map((row) => `${xScale(row.k)},${yScale(row[metric])}`)
    .join(" ");
};

const selectedMetric = (row: RetrievalEvalCaseResult) =>
  row.metrics.find((m) => m.k === failureK.value) || row.metrics[row.metrics.length - 1];

const failedQueries = computed(() => {
  if (!report.value) return [];
  return report.value.per_query.filter((row) => {
    if (row.relevant_count <= 0) return false;
    const m = selectedMetric(row);
    if (!m) return true;
    return m.hit_rate < 1 || m.recall < 1;
  });
});

const toPct = (v: number) => `${(v * 100).toFixed(1)}%`;

const resetAll = () => {
  rawDataset.value = "";
  fileName.value = "";
  kInput.value = "1,3,5";
  report.value = null;
  reportCollapsed.value = false;
  failureK.value = 1;
};

const handleFileSelect = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;
  fileName.value = file.name;
  rawDataset.value = await file.text();
};

const runEvaluation = async () => {
  try {
    loading.value = true;
    const cases = parseCases(rawDataset.value);
    if (!cases.length) {
      emit("error", "标注集为空或格式不正确");
      return;
    }
    const kValues = parseKValues(kInput.value);
    const res = await evaluateRetrieval({ cases, k_values: kValues });
    report.value = res;
    reportCollapsed.value = false;
    failureK.value = res.k_values[res.k_values.length - 1] ?? 1;
    await refreshHistory();
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.value = false;
  }
};

const refreshHistory = async () => {
  try {
    historyLoading.value = true;
    const res = await listRetrievalEvaluations(30);
    history.value = res.runs ?? [];
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    historyLoading.value = false;
  }
};

const loadRun = async (runId: number) => {
  try {
    loading.value = true;
    const res = await getRetrievalEvaluation(runId);
    report.value = res;
    reportCollapsed.value = false;
    failureK.value = res.k_values[res.k_values.length - 1] ?? 1;
  } catch (err) {
    emit("error", handleApiError(err));
  } finally {
    loading.value = false;
  }
};

const removeRun = async (runId: number) => {
  try {
    await deleteRetrievalEvaluation(runId);
    if (report.value?.eval_run_id === runId) {
      report.value = null;
    }
    await refreshHistory();
  } catch (err) {
    emit("error", handleApiError(err));
  }
};

const parseKValues = (value: string): number[] => {
  const parsed = value
    .split(",")
    .map((n) => Number.parseInt(n.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0);
  const deduped = Array.from(new Set(parsed)).sort((a, b) => a - b).slice(0, 12);
  return deduped.length ? deduped : [1, 3, 5];
};

const parseCases = (value: string): RetrievalEvalCaseInput[] => {
  const raw = value.trim();
  if (!raw) return [];

  const normalizeCase = (item: unknown, idx: number): RetrievalEvalCaseInput | null => {
    if (!item || typeof item !== "object") return null;
    const row = item as Record<string, unknown>;
    const query = String(row.query ?? "").trim();
    if (!query) return null;
    const rawIds = row.relevant_doc_ids ?? row.relevant_ids ?? row.labels ?? [];
    const relevant = Array.isArray(rawIds)
      ? rawIds.map((id) => String(id).trim()).filter(Boolean)
      : String(rawIds)
          .split(",")
          .map((id) => id.trim())
          .filter(Boolean);
    return {
      query,
      relevant_doc_ids: relevant,
      query_id: String(row.query_id ?? `q-${idx + 1}`),
    };
  };

  try {
    const parsed = JSON.parse(raw) as unknown;
    const cases = Array.isArray(parsed)
      ? parsed
      : parsed && typeof parsed === "object" && Array.isArray((parsed as { cases?: unknown[] }).cases)
      ? (parsed as { cases: unknown[] }).cases
      : [];
    return cases
      .map((item, idx) => normalizeCase(item, idx))
      .filter((item): item is RetrievalEvalCaseInput => item !== null);
  } catch {
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, idx) => {
        try {
          return normalizeCase(JSON.parse(line), idx);
        } catch {
          return null;
        }
      })
      .filter((item): item is RetrievalEvalCaseInput => item !== null);
  }
};

onMounted(() => {
  refreshHistory();
});
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.tip {
  margin-top: 6px;
}
.control-grid {
  margin: 10px 0 12px;
}
.saved-note {
  margin-top: 8px;
}
.history-card {
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: #fff;
}
.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.history-header h4 {
  margin: 0;
}
.history-list {
  display: grid;
  gap: 8px;
}
.history-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
}
.history-title {
  font-weight: 700;
}
.history-actions {
  display: flex;
  gap: 8px;
}
.result-wrap {
  margin-top: 14px;
  display: grid;
  gap: 14px;
}
.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.result-header h4 {
  margin: 0;
}
.metric-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
}
.metric-box {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
  background: #fff;
}
.metric-box .label {
  font-size: 12px;
  color: var(--muted);
}
.metric-box .value {
  font-size: 20px;
  font-weight: 700;
}
.chart-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: #fff;
}
.chart {
  width: 100%;
  height: 280px;
  display: block;
}
.axis {
  stroke: #94a3b8;
  stroke-width: 1;
}
.grid-line {
  stroke: #e2e8f0;
  stroke-width: 1;
}
.axis-text {
  fill: #64748b;
  font-size: 11px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 8px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.legend-item i {
  display: inline-block;
  width: 14px;
  height: 3px;
  border-radius: 2px;
}
.metric-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.metric-table th,
.metric-table td {
  border-bottom: 1px solid var(--border);
  padding: 8px 10px;
  text-align: left;
  font-size: 13px;
}
.metric-table th {
  background: #f8fafc;
}
.fail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.fail-header h4 {
  margin: 0;
}
.fail-list {
  display: grid;
  gap: 8px;
}
.fail-item {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
}
.fail-title {
  font-weight: 700;
  margin-bottom: 4px;
}
.danger {
  border-color: #fca5a5;
  color: #b91c1c;
}
</style>
