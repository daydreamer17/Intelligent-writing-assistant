<template>
  <section class="page">
    <div class="card">
      <h2>写作工作台</h2>
      <p class="muted">填写主题与要求，然后选择一键流程或分步执行。</p>
    </div>

    <StatusBar :data="health" />

    <div class="card">
      <h3>输入区</h3>
      <div class="grid-2">
        <div class="field">
          <label>主题</label>
          <input v-model="form.topic" placeholder="例如：AI 对教育的影响" />
        </div>
        <div class="field">
          <label>目标读者</label>
          <input v-model="form.audience" placeholder="例如：高校教师" />
        </div>
        <div class="field">
          <label>风格</label>
          <input v-model="form.style" placeholder="例如：学术、简洁、客观" />
        </div>
        <div class="field">
          <label>目标长度</label>
          <input v-model="form.target_length" placeholder="例如：1200 字" />
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          <label>约束条件</label>
          <textarea v-model="form.constraints" rows="4"></textarea>
        </div>
        <div class="field">
          <label>关键要点</label>
          <textarea v-model="form.key_points" rows="4"></textarea>
        </div>
      </div>
      <div class="field">
        <label>审校标准</label>
        <input v-model="form.review_criteria" placeholder="例如：逻辑清晰、论据充分、避免夸大" />
      </div>
      <div class="field">
        <label>研究素材（可选，来自 RAG 或手动粘贴）</label>
        <textarea v-model="form.research_notes" rows="4"></textarea>
        <div v-if="ragSnippets.length" class="muted">
          已收集素材：{{ ragSnippets.length }}
          <button class="btn ghost" @click="appendSnippets">加入到素材</button>
          <button class="btn ghost" @click="clearSnippets">清空</button>
        </div>
      </div>
      <div class="field">
        <label>大纲（用于分步草稿）</label>
        <textarea v-model="form.outline" rows="4"></textarea>
      </div>
      <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <button class="btn" @click="handlePipeline" :disabled="loading.pipeline || loading.pipelineV2">一键 Pipeline</button>
        <button class="btn secondary" @click="handlePlan" :disabled="loading.plan || loading.pipelineV2">仅生成大纲</button>
        <button class="btn ghost" @click="handleDraft" :disabled="loading.draft || loading.pipelineV2">生成草稿</button>
        <button class="btn ghost" @click="handleReview" :disabled="loading.review || loading.pipelineV2">审校</button>
        <button class="btn ghost" @click="handleRewrite" :disabled="loading.rewrite || loading.pipelineV2">改写</button>
        <button class="btn secondary" @click="handlePipelineV2" :disabled="isPrimaryActionDisabled">
          {{ loading.pipelineV2 ? "LangGraph v2 中..." : "LangGraph v2 Demo" }}
        </button>
        <button class="btn ghost" @click="openPipelineV2Panel" :disabled="loading.pipeline || loading.pipelineV2">
          Open V2 Panel
        </button>
        <button
          class="btn ghost"
          @click="handleResetSessionMemory"
          :disabled="loading.pipeline || loading.pipelineV2 || sessionMemoryResetting"
        >
          {{ sessionMemoryResetting ? "重置中..." : "重置会话记忆" }}
        </button>
        <button class="btn ghost" @click="clearGenerated" :disabled="loading.pipeline || loading.pipelineV2 || loading.draft || loading.review || loading.rewrite">清空生成内容</button>
        <button class="btn ghost" @click="exportText">导出 TXT</button>
        <button class="btn ghost" @click="exportMarkdown">导出 MD</button>
        <button class="btn ghost" @click="exportHtml">导出 HTML</button>
      </div>
      <div style="display:flex; gap:12px; align-items:center; margin-top:12px;">
        <label style="display:flex; align-items:center; gap:8px;">
          调用模式
          <select
            :value="generationMode || 'rag_only'"
            :disabled="modeUpdating"
            @change="handleGenerationModeChange"
          >
            <option value="rag_only">RAG-only（默认）</option>
            <option value="hybrid">Hybrid</option>
            <option value="creative">Creative</option>
          </select>
        </label>
        <span v-if="generationMode === null" class="muted">设置加载失败</span>
        <span v-else class="muted">{{ generationModeDescription }}</span>
      </div>
      <div v-if="generationMode === 'creative'" style="display:flex; gap:12px; align-items:center; margin-top:8px;">
        <label style="display:flex; align-items:center; gap:8px;" class="muted">
          <input
            type="checkbox"
            :checked="creativeMcpEnabled"
            :disabled="modeUpdating"
            @change="handleCreativeMcpToggle"
          />
          Creative 模式启用 MCP
        </label>
      </div>
      <div v-if="showCoverageMetrics" class="muted">
        <div>
          引用覆盖率（优先看语义）：{{ Math.round(((output.coverage_detail?.semantic_coverage ?? output.coverage) || 0) * 100) }}%
        </div>
        <div v-if="output.coverage_detail">
          语义段落覆盖率：{{ Math.round((output.coverage_detail.semantic_coverage || 0) * 100) }}%
          （{{ output.coverage_detail.semantic_covered_paragraphs ?? 0 }}/{{ output.coverage_detail.semantic_total_paragraphs ?? output.coverage_detail.total_paragraphs }}）
        </div>
        <div v-if="output.coverage_detail">
          词面复用覆盖率（严格）：{{ Math.round((output.coverage_detail.paragraph_coverage || 0) * 100) }}%
          （{{ output.coverage_detail.covered_paragraphs }}/{{ output.coverage_detail.total_paragraphs }}）
        </div>
      </div>
      <div v-if="error" class="muted">错误：{{ error }}</div>
    </div>
    <section v-if="pipelineV2.panelOpen" class="card v2-panel">
      <div class="v2-panel__header">
        <div>
          <h3>LangGraph v2 Demo</h3>
          <p class="muted">Status: {{ pipelineV2StatusText }}</p>
        </div>
        <div class="v2-panel__actions">
          <button
            v-if="pipelineV2.status === 'interrupted'"
            class="btn"
            @click="handlePipelineV2Resume"
            :disabled="loading.pipelineV2 || !pipelineV2.threadId"
          >
            {{ loading.pipelineV2 ? "Resuming..." : "Resume" }}
          </button>
          <button
            v-if="pipelineV2.threadId"
            class="btn ghost"
            @click="copyPipelineV2ThreadId"
            :disabled="loading.pipelineV2"
          >
            Copy thread_id
          </button>
          <button class="btn ghost" @click="resetPipelineV2State" :disabled="loading.pipelineV2">
            Exit Demo
          </button>
        </div>
      </div>

      <ProgressIndicator
        :visible="showPipelineV2Progress"
        title="LangGraph v2 Progress"
        :steps="pipelineV2Steps"
        :current-step="currentPipelineV2Step"
        :inline="true"
      />

      <div class="v2-panel__meta">
        <div><span class="v2-panel__label">thread_id: </span>{{ pipelineV2.threadId || "-" }}</div>
        <div v-if="pipelineV2.status === 'completed'" class="muted">This result was completed from a resumed thread.</div>
      </div>

      <div class="field">
        <label>Load existing thread_id</label>
        <div class="v2-panel__lookup">
          <input v-model="pipelineV2.lookupThreadId" placeholder="Paste an existing thread_id" />
          <button
            class="btn ghost"
            @click="handlePipelineV2LoadCheckpoint"
            :disabled="loading.pipelineV2 || !hasText(pipelineV2.lookupThreadId)"
          >
            Load checkpoint
          </button>
        </div>
        <div class="muted">Use this to inspect or resume an existing LangGraph v2 checkpoint.</div>
      </div>

      <div v-if="pipelineV2.note" class="muted">{{ pipelineV2.note }}</div>

      <div v-if="pipelineV2.assumptions" class="field">
        <label>Assumptions</label>
        <div class="v2-panel__text">{{ pipelineV2.assumptions }}</div>
      </div>

      <div v-if="pipelineV2.openQuestions" class="field">
        <label>Open Questions</label>
        <div class="v2-panel__text">{{ pipelineV2.openQuestions }}</div>
      </div>

      <div class="field">
        <label>
          {{
            pipelineV2.interruptStage !== 'outline_review'
              ? 'Outline Context'
              : pipelineV2.status === 'completed'
                ? 'Resolved Outline'
                : 'Outline Review'
          }}
        </label>
        <textarea
          v-model="pipelineV2.outlineDraft"
          rows="8"
          :readonly="pipelineV2.status !== 'interrupted' || pipelineV2.interruptStage !== 'outline_review'"
          :placeholder="pipelineV2.status === 'interrupted' && pipelineV2.interruptStage === 'outline_review' ? 'Edit the outline here, then click Resume.' : ''"
        />
        <div v-if="pipelineV2.status === 'interrupted' && pipelineV2.interruptStage === 'outline_review'" class="muted">
          If left empty, Resume will reuse the saved outline instead of deleting it.
        </div>
      </div>

      <div v-if="pipelineV2.interruptStage === 'draft_review'" class="field">
        <label>{{ pipelineV2.status === 'completed' ? 'Resolved Draft' : 'Draft Review' }}</label>
        <textarea
          v-model="pipelineV2.draftDraft"
          rows="10"
          :readonly="pipelineV2.status !== 'interrupted'"
          :placeholder="pipelineV2.status === 'interrupted' ? 'Edit the draft here, then click Resume.' : ''"
        />
        <div v-if="pipelineV2.status === 'interrupted'" class="muted">
          If left empty, Resume will reuse the saved draft instead of deleting it.
        </div>
      </div>

      <div v-if="pipelineV2.interruptStage === 'review_confirmation' || hasText(pipelineV2.reviewText)" class="field">
        <label>{{ pipelineV2.status === 'completed' ? 'Resolved Review Decision' : 'Review Confirmation' }}</label>
        <div class="v2-panel__text">{{ pipelineV2.reviewText || "-" }}</div>
        <div class="muted" style="margin-top:8px;">
          needs_rewrite: {{ pipelineV2.needsRewrite === null ? "-" : (pipelineV2.needsRewrite ? "true" : "false") }}
          <span v-if="hasText(pipelineV2.reviewReason)"> | reason: {{ pipelineV2.reviewReason }}</span>
          <span v-if="pipelineV2.reviewScore !== null && pipelineV2.reviewScore !== undefined">
            | score: {{ pipelineV2.reviewScore.toFixed(2) }}
          </span>
        </div>
        <div v-if="pipelineV2.status === 'interrupted' && pipelineV2.interruptStage === 'review_confirmation'" class="muted">
          Review feedback is read-only in this interrupt. Click Resume to continue.
        </div>
      </div>

      <div v-if="pipelineV2.error" class="muted">V2 error: {{ pipelineV2.error }}</div>
    </section>

    <div class="grid-2">
      <OutlinePanel
        :outline="output.outline"
        :assumptions="output.assumptions"
        :open-questions="output.open_questions"
      />
      <DraftEditor :draft="output.draft" @update="updateDraft" />
      <ReviewPanel :review="output.review" />
      <RewritePanel :revised="output.revised" />
    </div>

    <ResearchNotesPanel :notes="output.research_notes" />

    <CitationPanel
      :bibliography="output.bibliography"
      :version-id="output.version_id"
      :citations="output.citations"
      @insert="insertCitation"
    />
    <ProgressIndicator
      :visible="loading.pipeline"
      title="Pipeline 执行中"
      :steps="pipelineSteps"
      :current-step="currentPipelineStep"
    />
    <LoadingOverlay :active="loading.draft || loading.review || loading.rewrite" text="处理中..." />
    <Toast :message="toast" />
  </section>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useAppStore } from "../store";
import { draftStream, plan, reviewStream, rewriteStream } from "../services/writing";
import {
  getPipelineV2Checkpoint,
  runPipelineStream,
  runPipelineV2Stream,
  resumePipelineV2Stream,
} from "../services/pipeline";
import { getHealthDetail } from "../services/health";
import {
  clearSessionMemory,
  getGenerationMode,
  setGenerationMode,
  type GenerationMode,
} from "../services/settings";
import { handleApiError } from "../utils/errorHandler";
import { validatePipelineRequest, validateDraftRequest, validateReviewRequest, validateRewriteRequest } from "../utils/validation";
import { debounce } from "../utils/debounce";
import type {
  CoverageDetail,
  HealthDetailResponse,
  PipelineRequest,
  PipelineResponse,
  PipelineV2CheckpointDetailResponse,
  PipelineV2Request,
} from "../types";
import OutlinePanel from "../components/OutlinePanel.vue";
import DraftEditor from "../components/DraftEditor.vue";
import ReviewPanel from "../components/ReviewPanel.vue";
import RewritePanel from "../components/RewritePanel.vue";
import CitationPanel from "../components/CitationPanel.vue";
import ResearchNotesPanel from "../components/ResearchNotesPanel.vue";
import StatusBar from "../components/StatusBar.vue";
import LoadingOverlay from "../components/LoadingOverlay.vue";
import ProgressIndicator from "../components/ProgressIndicator.vue";
import Toast from "../components/Toast.vue";

const WORKSPACE_SESSION_KEY = "workspace-session-id";

const createSessionId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
};

const getOrCreateSessionId = () => {
  try {
    const current = localStorage.getItem(WORKSPACE_SESSION_KEY);
    if (current && current.trim()) {
      return current.trim();
    }
    const created = createSessionId();
    localStorage.setItem(WORKSPACE_SESSION_KEY, created);
    return created;
  } catch {
    return createSessionId();
  }
};

const form = reactive({
  topic: "",
  audience: "",
  style: "",
  target_length: "",
  constraints: "",
  key_points: "",
  review_criteria: "",
  research_notes: "",
  outline: "",
});

const output = reactive({
  outline: "",
  assumptions: "",
  open_questions: "",
  research_notes: [] as { doc_id: string; title: string; summary: string; url?: string }[],
  draft: "",
  review: "",
  revised: "",
  bibliography: "",
  citations: [] as { label: string; title: string; url?: string }[],
  version_id: undefined as number | undefined,
  coverage: undefined as number | undefined,
  coverage_detail: undefined as CoverageDetail | undefined,
});

const loading = reactive({
  plan: false,
  draft: false,
  review: false,
  rewrite: false,
  pipeline: false,
  pipelineV2: false,
});

type PipelineV2UiStatus = "idle" | "planning" | "interrupted" | "resuming" | "completed" | "error";

const pipelineV2 = reactive({
  panelOpen: false,
  status: "idle" as PipelineV2UiStatus,
  threadId: "",
  lookupThreadId: "",
  interruptStage: "" as "" | "outline_review" | "draft_review" | "review_confirmation",
  outlineDraft: "",
  draftDraft: "",
  reviewText: "",
  needsRewrite: null as boolean | null,
  reviewReason: "",
  reviewScore: null as number | null,
  assumptions: "",
  openQuestions: "",
  note: "",
  error: "",
});

const error = ref("");
const toast = ref("");
const health = ref<HealthDetailResponse | null>(null);
const generationMode = ref<GenerationMode | null>(null);
const creativeMcpEnabled = ref(true);
const modeUpdating = ref(false);
const sessionMemoryResetting = ref(false);
const sessionId = ref(getOrCreateSessionId());
const store = useAppStore();
const { ragSnippets } = storeToRefs(store);

// 进度指示器状态
const pipelineSteps = ['生成大纲', '收集研究笔记', '创作初稿', '审阅反馈', '修改润色', '生成引用'];
const currentPipelineStep = ref(0);
const pipelineStageIndex: Record<string, number> = {
  plan: 0,
  research: 1,
  draft: 2,
  review: 3,
  rewrite: 4,
  citations: 5,
};

const pipelineV2Steps = ['生成大纲', '等待人工修订', '收集研究笔记', '创作初稿', '审阅反馈', '修改润色', '生成引用'];
const currentPipelineV2Step = ref(0);
const pipelineV2StageIndex: Record<string, number> = {
  plan: 0,
  interrupted: 1,
  research: 2,
  draft: 3,
  review: 4,
  rewrite: 5,
  citations: 6,
};

const generationModeDescriptionMap: Record<GenerationMode, string> = {
  rag_only: "RAG-only：仅证据输出、禁外部资料、缺证据会停",
  hybrid: "Hybrid：允许补全，非证据段落自动标注 [推断]",
  creative: "Creative：自由生成，不强制引用与拒答（默认启用 MCP）",
};

const generationModeDescription = computed(() => {
  if (!generationMode.value) return "";
  return generationModeDescriptionMap[generationMode.value];
});

const showCoverageMetrics = computed(() => {
  if (!generationMode.value || generationMode.value === "creative") return false;
  return Boolean(output.coverage_detail || output.coverage !== undefined);
});

const isPrimaryActionDisabled = computed(
  () =>
    loading.pipeline ||
    loading.pipelineV2 ||
    loading.plan ||
    loading.draft ||
    loading.review ||
    loading.rewrite ||
    sessionMemoryResetting.value
);

const pipelineV2StatusTextMap: Record<PipelineV2UiStatus, string> = {
  idle: "Idle",
  planning: "Planning",
  interrupted: "Interrupted",
  resuming: "Resuming",
  completed: "Completed",
  error: "Error",
};

const pipelineV2StatusText = computed(() => {
  if (pipelineV2.status === "interrupted") {
    if (pipelineV2.interruptStage === "review_confirmation") {
      return "Interrupted - waiting for review confirmation";
    }
    return pipelineV2.interruptStage === "draft_review"
      ? "Interrupted - waiting for draft review"
      : "Interrupted - waiting for outline review";
  }
  return pipelineV2StatusTextMap[pipelineV2.status];
});

const showPipelineV2Progress = computed(() => pipelineV2.status !== "idle" && pipelineV2.status !== "error");

const hasText = (value: string | null | undefined) => Boolean((value || "").trim());

const buildRewriteGuidance = () => {
  const parts = [output.review, form.review_criteria]
    .map((item) => (item || "").trim())
    .filter((item, idx, arr) => Boolean(item) && arr.indexOf(item) === idx);
  return parts.join("\n\n");
};

const resetError = () => {
  error.value = "";
};

const resetPipelineProgress = () => {
  loading.pipeline = false;
  currentPipelineStep.value = 0;
};

const resetPipelineV2State = () => {
  pipelineV2.panelOpen = false;
  pipelineV2.status = "idle";
  pipelineV2.threadId = "";
  pipelineV2.lookupThreadId = "";
  pipelineV2.interruptStage = "";
  pipelineV2.outlineDraft = "";
  pipelineV2.draftDraft = "";
  pipelineV2.reviewText = "";
  pipelineV2.needsRewrite = null;
  pipelineV2.reviewReason = "";
  pipelineV2.reviewScore = null;
  pipelineV2.assumptions = "";
  pipelineV2.openQuestions = "";
  pipelineV2.note = "";
  pipelineV2.error = "";
  currentPipelineV2Step.value = 0;
  loading.pipelineV2 = false;
};

const pipelineV2InterruptStep = (interruptStage: string) => {
  if (interruptStage === "draft_review") return pipelineV2StageIndex.draft;
  if (interruptStage === "review_confirmation") return pipelineV2StageIndex.review;
  return pipelineV2StageIndex.interrupted;
};

const pipelineV2InterruptMessage = (interruptStage: string) => {
  if (interruptStage === "draft_review") {
    return "LangGraph v2 已在草稿完成后中断，等待人工修订草稿";
  }
  if (interruptStage === "review_confirmation") {
    return "LangGraph v2 已在审阅完成后中断，等待人工确认";
  }
  return "LangGraph v2 已中断，等待人工修订大纲";
};

const openPipelineV2Panel = () => {
  pipelineV2.panelOpen = true;
  pipelineV2.note = "";
  pipelineV2.error = "";
};

const copyPipelineV2ThreadId = async () => {
  if (!pipelineV2.threadId) return;
  try {
    await navigator.clipboard.writeText(pipelineV2.threadId);
    showToast("thread_id copied");
  } catch {
    error.value = "Failed to copy thread_id. Please copy it manually.";
  }
};

const applyPipelineV2CheckpointDetail = (detail: PipelineV2CheckpointDetailResponse) => {
  pipelineV2.panelOpen = true;
  pipelineV2.threadId = detail.thread_id;
  pipelineV2.lookupThreadId = detail.thread_id;
  pipelineV2.interruptStage =
    detail.interrupt_stage === "draft_review"
      ? "draft_review"
      : detail.interrupt_stage === "review_confirmation"
        ? "review_confirmation"
        : "outline_review";
  pipelineV2.outlineDraft = detail.outline || "";
  pipelineV2.draftDraft = detail.draft || "";
  pipelineV2.reviewText = detail.review_text || "";
  pipelineV2.needsRewrite = detail.needs_rewrite ?? null;
  pipelineV2.reviewReason = detail.reason || "";
  pipelineV2.reviewScore = detail.score ?? null;
  pipelineV2.assumptions = detail.assumptions || "";
  pipelineV2.openQuestions = detail.open_questions || "";
  pipelineV2.error = "";
  pipelineV2.note = "";
  form.outline = detail.outline || form.outline;
  output.outline = detail.outline || output.outline;
  output.draft = detail.draft || output.draft;
  output.review = detail.review_text || output.review;
  output.assumptions = detail.assumptions || output.assumptions;
  output.open_questions = detail.open_questions || output.open_questions;

  if (detail.status === "interrupted") {
    pipelineV2.status = "interrupted";
    currentPipelineV2Step.value = pipelineV2InterruptStep(pipelineV2.interruptStage);
    pipelineV2.note =
      detail.interrupt_stage === "draft_review"
        ? "Loaded a draft review checkpoint."
        : detail.interrupt_stage === "review_confirmation"
          ? "Loaded a review confirmation checkpoint."
        : "Loaded an interrupted checkpoint.";
    return;
  }

  if (detail.status === "completed") {
    pipelineV2.status = "completed";
    currentPipelineV2Step.value = pipelineV2Steps.length;
    pipelineV2.note = "This thread is already completed. Use Settings to inspect or clean it.";
    return;
  }

  pipelineV2.status = "idle";
  currentPipelineV2Step.value = 0;
  pipelineV2.note =
    detail.status === "failed"
      ? `This thread is failed${detail.last_error ? `: ${detail.last_error}` : ""}`
      : `This thread is ${detail.status}. Frontend resume is not available for this status.`;
};

const handlePipelineV2LoadCheckpoint = async () => {
  const threadId = pipelineV2.lookupThreadId.trim();
  if (!threadId) {
    pipelineV2.error = "Please enter a valid thread_id.";
    return;
  }
  try {
    resetError();
    pipelineV2.panelOpen = true;
    pipelineV2.error = "";
    pipelineV2.note = "";
    loading.pipelineV2 = true;
    const detail = await getPipelineV2Checkpoint(threadId);
    applyPipelineV2CheckpointDetail(detail);
    showToast(`Loaded thread_id: ${threadId}`);
  } catch (err) {
    const message = handleApiError(err);
    error.value = message;
    pipelineV2.status = "error";
    pipelineV2.error = message;
    pipelineV2.panelOpen = true;
  } finally {
    loading.pipelineV2 = false;
  }
};

const applyPipelineResponse = (res: PipelineResponse) => {
  output.outline = res.outline;
  output.assumptions = res.assumptions;
  output.open_questions = res.open_questions;
  output.research_notes = res.research_notes || [];
  output.draft = res.draft || output.draft;
  output.review = res.review || output.review;
  output.revised = res.revised || output.revised;
  output.bibliography = res.bibliography;
  output.version_id = res.version_id;
  output.citations = res.citations || [];
  if (res.citation_enforced) {
    output.revised = res.revised;
  }
  if (typeof res.coverage === "number") {
    output.coverage = res.coverage;
  }
  if (res.coverage_detail) {
    output.coverage_detail = res.coverage_detail;
  }
};

const applyPipelineV2InterruptPayload = (payload: Record<string, unknown> | undefined, kind = "outline_review") => {
  const interruptStage =
    String(payload?.interrupt_stage || kind || "outline_review") === "draft_review"
      ? "draft_review"
      : String(payload?.interrupt_stage || kind || "outline_review") === "review_confirmation"
        ? "review_confirmation"
        : "outline_review";
  pipelineV2.interruptStage = interruptStage;
  pipelineV2.outlineDraft = payload?.outline ? String(payload.outline) : "";
  pipelineV2.draftDraft = payload?.draft ? String(payload.draft) : "";
  pipelineV2.reviewText = payload?.review_text ? String(payload.review_text) : "";
  pipelineV2.needsRewrite =
    payload?.needs_rewrite === undefined || payload?.needs_rewrite === null
      ? null
      : Boolean(payload.needs_rewrite);
  pipelineV2.reviewReason = payload?.reason ? String(payload.reason) : "";
  pipelineV2.reviewScore =
    payload?.score === undefined || payload?.score === null ? null : Number(payload.score);
  pipelineV2.assumptions = payload?.assumptions ? String(payload.assumptions) : "";
  pipelineV2.openQuestions = payload?.open_questions ? String(payload.open_questions) : "";
  output.outline = pipelineV2.outlineDraft;
  if (interruptStage === "draft_review") {
    output.draft = pipelineV2.draftDraft;
  }
  if (interruptStage === "review_confirmation") {
    output.review = pipelineV2.reviewText;
  }
  output.assumptions = pipelineV2.assumptions;
  output.open_questions = pipelineV2.openQuestions;
  form.outline = pipelineV2.outlineDraft;
};

const buildSourceDocuments = () =>
  ragSnippets.value.map((content, idx) => ({
    doc_id: `snippet-${Date.now()}-${idx}`,
    title: `RAG片段 ${idx + 1}`,
    content,
    url: "",
  }));

const buildPipelineRequestPayload = (): PipelineRequest => ({
  topic: form.topic,
  audience: form.audience,
  style: form.style,
  target_length: form.target_length,
  constraints: form.constraints,
  key_points: form.key_points,
  review_criteria: form.review_criteria,
  sources: buildSourceDocuments(),
  session_id: sessionId.value,
});

const resetSessionMemoryForNewTask = async () => {
  await clearSessionMemory({
    session_id: sessionId.value,
    drop_agent: true,
    clear_cold: true,
  });
};

const clearGenerated = () => {
  output.outline = "";
  output.assumptions = "";
  output.open_questions = "";
  output.research_notes = [];
  output.draft = "";
  output.review = "";
  output.revised = "";
  output.bibliography = "";
  output.citations = [];
  output.version_id = undefined;
  output.coverage = undefined;
  output.coverage_detail = undefined;
  showToast("已清空生成内容");
};

const handlePipelineV2 = async () => {
  try {
    resetError();
    pipelineV2.error = "";
    resetPipelineProgress();

    const validation = validatePipelineRequest({
      topic: form.topic,
      audience: form.audience,
      style: form.style,
      target_length: form.target_length,
    });
    if (!validation.valid) {
      const message = validation.errors.join("; ");
      error.value = message;
      pipelineV2.status = "error";
      pipelineV2.error = message;
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.pipelineV2 = true;
    pipelineV2.panelOpen = true;
    pipelineV2.status = "planning";
    currentPipelineV2Step.value = pipelineV2StageIndex.plan;
    pipelineV2.threadId = "";
    pipelineV2.lookupThreadId = "";
    pipelineV2.interruptStage = "";
    pipelineV2.outlineDraft = "";
    pipelineV2.draftDraft = "";
    pipelineV2.assumptions = "";
    pipelineV2.openQuestions = "";
    pipelineV2.note = "";
    output.outline = "";
    output.assumptions = "";
    output.open_questions = "";

    const payload: PipelineV2Request = {
      ...buildPipelineRequestPayload(),
    };
    const streamSummary = await runPipelineV2Stream(payload, (evt) => {
      if (evt.type === "status" && evt.step === "plan") {
        currentPipelineV2Step.value = pipelineV2StageIndex.plan;
      }
      if (evt.type === "delta" && evt.stage === "plan") {
        pipelineV2.outlineDraft += evt.content || "";
        output.outline = pipelineV2.outlineDraft;
      }
      if (evt.type === "outline") {
        applyPipelineV2InterruptPayload(evt.payload || {}, "outline_review");
      }
      if (evt.type === "interrupt") {
        pipelineV2.status = "interrupted";
        const interruptKind = String(evt.kind || evt.payload?.interrupt_stage || "outline_review");
        currentPipelineV2Step.value = pipelineV2InterruptStep(interruptKind);
        pipelineV2.threadId = String(evt.payload?.thread_id || "");
        applyPipelineV2InterruptPayload(evt.payload || {}, interruptKind);
        showToast(pipelineV2InterruptMessage(interruptKind));
      }
      if (evt.type === "error") {
        const message = evt.detail || "LangGraph v2 执行失败";
        error.value = message;
        pipelineV2.status = "error";
        pipelineV2.error = message;
      }
    });

    if (pipelineV2.status === "planning") {
      if (streamSummary?.interruptPayload) {
        pipelineV2.status = "interrupted";
        const interruptKind = String(streamSummary.interruptPayload?.interrupt_stage || "outline_review");
        currentPipelineV2Step.value = pipelineV2InterruptStep(interruptKind);
        pipelineV2.threadId = String(streamSummary.interruptPayload?.thread_id || "");
        applyPipelineV2InterruptPayload(streamSummary.interruptPayload || {}, interruptKind);
        showToast(pipelineV2InterruptMessage(interruptKind));
        return;
      }
      throw new Error("LangGraph v2 流式执行结束，但未收到 interrupt 事件");
    }
  } catch (err) {
    const message = handleApiError(err);
    error.value = message;
    pipelineV2.panelOpen = true;
    pipelineV2.status = "error";
    pipelineV2.error = message;
  } finally {
    loading.pipelineV2 = false;
  }
};

const handlePipelineV2Resume = async () => {
  if (!pipelineV2.threadId) {
    const message = "缺少 thread_id，无法继续执行";
    error.value = message;
    pipelineV2.status = "error";
    pipelineV2.error = message;
    return;
  }

  try {
    resetError();
    pipelineV2.error = "";
    pipelineV2.note = "";
    resetPipelineProgress();
    loading.pipelineV2 = true;
    pipelineV2.status = "resuming";
    currentPipelineV2Step.value =
      pipelineV2.interruptStage === "draft_review"
        ? pipelineV2StageIndex.review
        : pipelineV2.interruptStage === "review_confirmation"
          ? pipelineV2StageIndex.rewrite
          : pipelineV2StageIndex.research;
    if (pipelineV2.interruptStage !== "review_confirmation") {
      output.research_notes = [];
      output.draft = "";
      output.review = "";
    }
    output.revised = "";
    output.coverage = undefined;
    output.coverage_detail = undefined;

    const streamSummary = await resumePipelineV2Stream(
      {
        thread_id: pipelineV2.threadId,
        outline_override: pipelineV2.outlineDraft,
        draft_override: pipelineV2.draftDraft,
      },
      (evt) => {
        if (evt.type === "status" && evt.step && evt.step in pipelineV2StageIndex) {
          currentPipelineV2Step.value =
            pipelineV2StageIndex[evt.step as keyof typeof pipelineV2StageIndex];
        }
        if (evt.type === "outline") {
          applyPipelineV2InterruptPayload(evt.payload || {}, "outline_review");
        }
        if (evt.type === "research") {
          output.research_notes = evt.payload?.notes || [];
        }
        if (evt.type === "delta") {
          if (evt.stage === "draft") {
            output.draft += evt.content || "";
          }
          if (evt.stage === "review") {
            output.review += evt.content || "";
          }
          if (evt.stage === "rewrite") {
            output.revised += evt.content || "";
          }
        }
        if (evt.type === "draft") {
          output.draft = evt.payload?.draft || output.draft;
          pipelineV2.draftDraft = evt.payload?.draft || pipelineV2.draftDraft;
        }
        if (evt.type === "review") {
          output.review = evt.payload?.review || output.review;
        }
        if (evt.type === "review_decision") {
          pipelineV2.reviewText = evt.payload?.review_text || pipelineV2.reviewText;
          pipelineV2.needsRewrite =
            evt.payload?.needs_rewrite === undefined || evt.payload?.needs_rewrite === null
              ? pipelineV2.needsRewrite
              : Boolean(evt.payload.needs_rewrite);
          pipelineV2.reviewReason = evt.payload?.reason || pipelineV2.reviewReason;
          pipelineV2.reviewScore =
            evt.payload?.score === undefined || evt.payload?.score === null
              ? pipelineV2.reviewScore
              : Number(evt.payload.score);
        }
        if (evt.type === "rewrite") {
          output.revised = evt.payload?.revised || output.revised;
          if (typeof evt.payload?.coverage === "number") {
            output.coverage = evt.payload.coverage;
          }
          if (evt.payload?.coverage_detail) {
            output.coverage_detail = evt.payload.coverage_detail;
          }
        }
        if (evt.type === "interrupt") {
          pipelineV2.status = "interrupted";
          const interruptKind = String(evt.kind || evt.payload?.interrupt_stage || "outline_review");
          currentPipelineV2Step.value = pipelineV2InterruptStep(interruptKind);
          pipelineV2.threadId = String(evt.payload?.thread_id || pipelineV2.threadId);
          applyPipelineV2InterruptPayload(evt.payload || {}, interruptKind);
          showToast(pipelineV2InterruptMessage(interruptKind));
        }
        if (evt.type === "result" && evt.payload) {
          applyPipelineResponse(evt.payload);
          pipelineV2.status = "completed";
          currentPipelineV2Step.value = pipelineV2Steps.length;
          pipelineV2.threadId = pipelineV2.threadId || String(evt.payload?.thread_id || "");
          pipelineV2.outlineDraft = evt.payload.outline || pipelineV2.outlineDraft;
          pipelineV2.draftDraft = evt.payload.draft || pipelineV2.draftDraft;
          pipelineV2.assumptions = evt.payload.assumptions || pipelineV2.assumptions;
          pipelineV2.openQuestions = evt.payload.open_questions || pipelineV2.openQuestions;
          form.outline = evt.payload.outline || form.outline;
          showToast("LangGraph v2 已恢复执行并完成");
        }
        if (evt.type === "error") {
          const message = evt.detail || "LangGraph v2 Resume 执行失败";
          error.value = message;
          pipelineV2.status = "error";
          pipelineV2.error = message;
        }
      }
    );

    if (pipelineV2.status === "resuming") {
      if (streamSummary?.interruptPayload) {
        const interruptKind = String(streamSummary.interruptPayload?.interrupt_stage || "outline_review");
        pipelineV2.status = "interrupted";
        currentPipelineV2Step.value = pipelineV2InterruptStep(interruptKind);
        pipelineV2.threadId = pipelineV2.threadId || String(streamSummary.interruptPayload?.thread_id || "");
        applyPipelineV2InterruptPayload(streamSummary.interruptPayload || {}, interruptKind);
        showToast(pipelineV2InterruptMessage(interruptKind));
        return;
      }
      if (streamSummary?.resultPayload) {
        applyPipelineResponse(streamSummary.resultPayload);
        pipelineV2.status = "completed";
        currentPipelineV2Step.value = pipelineV2Steps.length;
        pipelineV2.threadId = pipelineV2.threadId || String(streamSummary.resultPayload?.thread_id || "");
        pipelineV2.outlineDraft = streamSummary.resultPayload.outline || pipelineV2.outlineDraft;
        pipelineV2.draftDraft = streamSummary.resultPayload.draft || pipelineV2.draftDraft;
        pipelineV2.assumptions = streamSummary.resultPayload.assumptions || pipelineV2.assumptions;
        pipelineV2.openQuestions = streamSummary.resultPayload.open_questions || pipelineV2.openQuestions;
        form.outline = streamSummary.resultPayload.outline || form.outline;
        showToast("LangGraph v2 已恢复执行并完成");
        return;
      }
      throw new Error("LangGraph v2 Resume 流式执行结束，但未收到完成结果");
    }
  } catch (err) {
    const message = handleApiError(err);
    error.value = message;
    pipelineV2.panelOpen = true;
    pipelineV2.status = "error";
    pipelineV2.error = message;
  } finally {
    loading.pipelineV2 = false;
  }
};

const handlePlan = async () => {
  try {
    resetError();
    resetPipelineV2State();

    // 表单验证
    if (!form.topic || form.topic.trim() === '') {
      error.value = '请输入主题';
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.plan = true;
    const res = await plan({
      topic: form.topic,
      audience: form.audience,
      style: form.style,
      target_length: form.target_length,
      constraints: form.constraints,
      key_points: form.key_points,
      session_id: sessionId.value,
    });
    output.outline = res.outline;
    output.assumptions = res.assumptions;
    output.open_questions = res.open_questions;
    form.outline = res.outline;
    showToast("大纲已生成");
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    loading.plan = false;
  }
};

const handleDraft = async () => {
  try {
    resetError();
    resetPipelineV2State();

    // 表单验证
    const validation = validateDraftRequest({ topic: form.topic, outline: form.outline });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.draft = true;
    output.draft = "";
    await draftStream(
      {
        topic: form.topic,
        outline: form.outline,
        research_notes: form.research_notes,
        constraints: form.constraints,
        style: form.style,
        target_length: form.target_length,
        session_id: sessionId.value,
      },
      (evt) => {
        if (evt.type === "delta") {
          output.draft += evt.content || "";
        }
        if (evt.type === "result") {
          // 只在流式内容为空时才使用 result（兜底）
          if (!hasText(output.draft)) {
            output.draft = evt.payload?.draft || "";
          }
          showToast("草稿已生成");
        }
        if (evt.type === "error") {
          error.value = evt.detail || "Draft failed";
        }
      }
    );
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    loading.draft = false;
  }
};

const handleReview = async () => {
  try {
    resetError();
    resetPipelineV2State();

    // 表单验证
    const validation = validateReviewRequest({ draft: output.draft || form.research_notes });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.review = true;
    output.review = "";
    await reviewStream(
      {
        draft: output.draft || form.research_notes,
        criteria: form.review_criteria,
        sources: form.research_notes,
        audience: form.audience,
        session_id: sessionId.value,
      },
      (evt) => {
        if (evt.type === "delta") {
          output.review += evt.content || "";
        }
        if (evt.type === "result") {
          // 只在流式内容为空时才使用 result（兜底）
          if (!hasText(output.review)) {
            output.review = evt.payload?.review || "";
          }
          showToast("审校完成");
        }
        if (evt.type === "error") {
          error.value = evt.detail || "Review failed";
        }
      }
    );
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    loading.review = false;
  }
};

const handleRewrite = async () => {
  try {
    resetError();
    resetPipelineV2State();
    const rewriteGuidance = buildRewriteGuidance();

    // 表单验证
    const validation = validateRewriteRequest({ draft: output.draft, guidance: rewriteGuidance });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.rewrite = true;
    output.revised = "";
    output.bibliography = "";
    output.citations = [];
    output.coverage = undefined;
    output.coverage_detail = undefined;
    await rewriteStream(
      {
        draft: output.draft,
        guidance: rewriteGuidance,
        style: form.style,
        target_length: form.target_length,
        session_id: sessionId.value,
      },
      (evt) => {
        if (evt.type === "delta") {
          output.revised += evt.content || "";
        }
        if (evt.type === "result") {
          // 结果事件是后端后处理后的最终稿（含引用补全），必须覆盖流式中间文本
          output.revised = evt.payload?.revised || output.revised;
          if (Array.isArray(evt.payload?.citations)) {
            output.citations = evt.payload.citations;
          }
          if (typeof evt.payload?.bibliography === "string") {
            output.bibliography = evt.payload.bibliography;
          }
          if (typeof evt.payload?.coverage === "number") {
            output.coverage = evt.payload.coverage;
          }
          if (evt.payload?.coverage_detail) {
            output.coverage_detail = evt.payload.coverage_detail;
          }
          showToast("改写完成");
        }
        if (evt.type === "error") {
          error.value = evt.detail || "Rewrite failed";
        }
      }
    );
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    loading.rewrite = false;
  }
};

const handlePipeline = async () => {
  try {
    resetError();
    resetPipelineV2State();

    // 表单验证
    const validation = validatePipelineRequest({
      topic: form.topic,
      audience: form.audience,
      style: form.style,
      target_length: form.target_length,
    });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    await resetSessionMemoryForNewTask();
    loading.pipeline = true;
    currentPipelineStep.value = 0;
    output.outline = "";
    output.assumptions = "";
    output.open_questions = "";
    output.research_notes = [];
    output.draft = "";
    output.review = "";
    output.revised = "";
    output.bibliography = "";
    output.citations = [];
    output.version_id = undefined;
    output.coverage = undefined;
    output.coverage_detail = undefined;

    const payload: PipelineRequest = buildPipelineRequestPayload();

    await runPipelineStream(payload, (evt) => {
      if (evt.type === "status") {
        const idx = pipelineStageIndex[String(evt.step || "").trim()] ?? 0;
        currentPipelineStep.value = idx;
      }
      if (evt.type === "outline") {
        currentPipelineStep.value = 1;
        output.outline = evt.payload?.outline || "";
        output.assumptions = evt.payload?.assumptions || "";
        output.open_questions = evt.payload?.open_questions || "";
        form.outline = output.outline;
      }
      if (evt.type === "research") {
        currentPipelineStep.value = 2;
        const notesText = evt.payload?.notes_text || "";
        if (notesText) {
          if (!form.research_notes) {
            form.research_notes = notesText;
          } else if (!form.research_notes.includes("GitHub MCP Context")) {
            form.research_notes = [form.research_notes, notesText].filter(Boolean).join("\n\n");
          }
        }
        if (evt.payload?.notes) {
          output.research_notes = evt.payload.notes;
        }
      }
      if (evt.type === "draft") {
        currentPipelineStep.value = 3;
        // 阶段结果事件视为权威结果，覆盖当前草稿（避免流式丢尾）
        output.draft = evt.payload?.draft || output.draft;
      }
      if (evt.type === "delta" && evt.stage === "draft") {
        currentPipelineStep.value = pipelineStageIndex.draft;
        output.draft += evt.content || "";
      }
      if (evt.type === "review") {
        currentPipelineStep.value = 4;
        // 阶段结果事件视为权威结果，覆盖当前审校（避免流式丢尾）
        output.review = evt.payload?.review || output.review;
      }
      if (evt.type === "delta" && evt.stage === "review") {
        currentPipelineStep.value = pipelineStageIndex.review;
        output.review += evt.content || "";
      }
      if (evt.type === "rewrite") {
        currentPipelineStep.value = 5;
        if (evt.payload?.final) {
          output.revised = evt.payload?.revised || "";
          if (typeof evt.payload?.coverage === "number") {
            output.coverage = evt.payload.coverage;
          }
          if (evt.payload?.coverage_detail) {
            output.coverage_detail = evt.payload.coverage_detail;
          }
        } else if (!hasText(output.revised)) {
          // 非最终事件兜底：仅当当前没有有效文本时覆盖
          output.revised = evt.payload?.revised || "";
        }
      }
      if (evt.type === "delta" && evt.stage === "rewrite") {
        currentPipelineStep.value = pipelineStageIndex.rewrite;
        output.revised += evt.content || "";
      }
      if (evt.type === "error") {
        error.value = evt.detail || "Pipeline failed";
      }
      if (evt.type === "result") {
        const res = evt.payload;
        currentPipelineStep.value = pipelineSteps.length;
        applyPipelineResponse(res);
        showToast("Pipeline 完成");
      }
      if (evt.type === "done") {
        loading.pipeline = false;
        currentPipelineStep.value = 0;
      }
    });
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    loading.pipeline = false;
    currentPipelineStep.value = 0;
  }
};

const updateDraft = (value: string) => {
  output.draft = value;
};

const appendSnippets = () => {
  if (ragSnippets.value.length) {
    form.research_notes = [form.research_notes, ...ragSnippets.value].filter(Boolean).join("\n\n");
  }
};

const clearSnippets = () => {
  store.clearSnippets();
  showToast("已清空 RAG 素材");
};

const handleResetSessionMemory = async () => {
  resetError();
  sessionMemoryResetting.value = true;
  try {
    const res = await clearSessionMemory({
      session_id: sessionId.value,
      drop_agent: true,
      clear_cold: true,
    });
    if (res.cleared) {
      showToast("当前会话记忆已重置");
      return;
    }
    showToast("当前会话没有可清理记忆");
  } catch (err) {
    error.value = handleApiError(err);
  } finally {
    sessionMemoryResetting.value = false;
  }
};

const exportText = () => {
  const content = output.revised || output.draft || "";
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "writing.txt";
  a.click();
  URL.revokeObjectURL(url);
  showToast("已导出 TXT");
};

const exportMarkdown = () => {
  const content = output.revised || output.draft || "";
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "writing.md";
  a.click();
  URL.revokeObjectURL(url);
  showToast("已导出 MD");
};

const exportHtml = () => {
  const content = output.revised || output.draft || "";
  const html = `<!doctype html><html><head><meta charset="utf-8"></head><body><pre>${escapeHtml(
    content
  )}</pre></body></html>`;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "writing.html";
  a.click();
  URL.revokeObjectURL(url);
  showToast("已导出 HTML");
};

const insertCitation = (item: { label: string; title: string; url?: string }) => {
  const text = `${item.label} ${item.title}${item.url ? ` (${item.url})` : ""}`;
  output.draft = [output.draft, text].filter(Boolean).join("\n");
  showToast("已插入引用");
};

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const loadHealth = async () => {
  try {
    health.value = await getHealthDetail();
  } catch {
    health.value = null;
  }
};

loadHealth();

const loadGenerationModeSetting = async () => {
  try {
    const res = await getGenerationMode();
    generationMode.value = res.mode;
    creativeMcpEnabled.value = res.creative_mcp_enabled;
  } catch {
    generationMode.value = null;
    creativeMcpEnabled.value = true;
  }
};

loadGenerationModeSetting();

const handleGenerationModeChange = async (event: Event) => {
  const target = event.target as HTMLSelectElement;
  const nextValue = target.value as GenerationMode;
  modeUpdating.value = true;
  try {
    const res = await setGenerationMode(nextValue);
    generationMode.value = res.mode;
    creativeMcpEnabled.value = res.creative_mcp_enabled;
    showToast(`调用模式已切换为 ${res.mode}`);
  } catch (err) {
    error.value = handleApiError(err);
    target.value = generationMode.value || "rag_only";
  } finally {
    modeUpdating.value = false;
  }
};

const handleCreativeMcpToggle = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  const checked = target.checked;
  const currentMode = generationMode.value || "creative";
  modeUpdating.value = true;
  try {
    const res = await setGenerationMode(currentMode, checked);
    generationMode.value = res.mode;
    creativeMcpEnabled.value = res.creative_mcp_enabled;
    showToast(`Creative MCP 已${res.creative_mcp_enabled ? "开启" : "关闭"}`);
  } catch (err) {
    error.value = handleApiError(err);
    target.checked = creativeMcpEnabled.value;
  } finally {
    modeUpdating.value = false;
  }
};

const showToast = (message: string) => {
  toast.value = message;
  setTimeout(() => {
    toast.value = "";
  }, 2000);
};

// 自动保存到 localStorage
const autoSave = debounce(() => {
  const hasFormInput = Object.values(form).some((value) => String(value || "").trim().length > 0);
  const hasOutput = Boolean(output.revised || output.draft || output.outline || output.review);
  if (!hasFormInput && !hasOutput) return;

  try {
    const autoSaveData = {
      form: { ...form },
      output: { ...output },
      timestamp: new Date().toISOString(),
    };
    localStorage.setItem('workspace-autosave', JSON.stringify(autoSaveData));
  } catch (err) {
    console.error('自动保存失败', err);
  }
}, 1000);

// 监听内容变化，触发自动保存
watch(() => output.revised, () => {
  autoSave();
});

watch(() => output.draft, () => {
  autoSave();
});

watch(
  () => form,
  () => {
    autoSave();
  },
  { deep: true }
);

// 页面加载时恢复自动保存的内容
const restoreAutoSave = () => {
  try {
    const saved = localStorage.getItem('workspace-autosave');
    if (saved) {
      const data = JSON.parse(saved);
      const savedTime = new Date(data.timestamp);
      const now = new Date();
      const hoursDiff = (now.getTime() - savedTime.getTime()) / (1000 * 60 * 60);

      // 只恢复24小时内的自动保存
      if (hoursDiff < 24) {
        Object.assign(form, data.form);
        Object.assign(output, data.output);
        showToast('已恢复上次编辑内容');
      }
    }
  } catch (err) {
    console.error('恢复自动保存失败', err);
  }
};

restoreAutoSave();
</script>

<style scoped>
.v2-panel {
  border: 1px solid rgba(41, 121, 255, 0.18);
}

.v2-panel__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
}

.v2-panel__actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.v2-panel__meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 12px 0;
}

.v2-panel__label {
  font-weight: 600;
}

.v2-panel__lookup {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.v2-panel__lookup input {
  flex: 1 1 280px;
}

.v2-panel__text {
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.7;
  font-size: 14px;
}
</style>
