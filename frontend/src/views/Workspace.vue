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
        <div v-if="snippets.length" class="muted">
          已收集素材：{{ snippets.length }}
          <button class="btn ghost" @click="appendSnippets">加入到素材</button>
          <button class="btn ghost" @click="clearSnippets">清空</button>
        </div>
      </div>
      <div class="field">
        <label>大纲（用于分步草稿）</label>
        <textarea v-model="form.outline" rows="4"></textarea>
      </div>
      <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <button class="btn" @click="handlePipeline" :disabled="loading.pipeline">一键 Pipeline</button>
        <button class="btn secondary" @click="handlePlan" :disabled="loading.plan">仅生成大纲</button>
        <button class="btn ghost" @click="handleDraft" :disabled="loading.draft">生成草稿</button>
        <button class="btn ghost" @click="handleReview" :disabled="loading.review">审校</button>
        <button class="btn ghost" @click="handleRewrite" :disabled="loading.rewrite">改写</button>
        <button class="btn ghost" @click="exportText">导出 TXT</button>
        <button class="btn ghost" @click="exportMarkdown">导出 MD</button>
        <button class="btn ghost" @click="exportHtml">导出 HTML</button>
      </div>
      <div v-if="error" class="muted">错误：{{ error }}</div>
    </div>

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
import { reactive, ref, watch } from "vue";
import { useAppStore } from "../store";
import { draftStream, plan, reviewStream, rewriteStream } from "../services/writing";
import { runPipelineStream } from "../services/pipeline";
import { getHealthDetail } from "../services/health";
import { handleApiError } from "../utils/errorHandler";
import { validatePipelineRequest, validateDraftRequest, validateReviewRequest, validateRewriteRequest } from "../utils/validation";
import { debounce } from "../utils/debounce";
import type { PipelineRequest, HealthDetailResponse } from "../types";
import OutlinePanel from "../components/OutlinePanel.vue";
import DraftEditor from "../components/DraftEditor.vue";
import ReviewPanel from "../components/ReviewPanel.vue";
import RewritePanel from "../components/RewritePanel.vue";
import CitationPanel from "../components/CitationPanel.vue";
import StatusBar from "../components/StatusBar.vue";
import LoadingOverlay from "../components/LoadingOverlay.vue";
import ProgressIndicator from "../components/ProgressIndicator.vue";
import Toast from "../components/Toast.vue";

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
  draft: "",
  review: "",
  revised: "",
  bibliography: "",
  citations: [] as { label: string; title: string; url?: string }[],
  version_id: undefined as number | undefined,
});

const loading = reactive({
  plan: false,
  draft: false,
  review: false,
  rewrite: false,
  pipeline: false,
});

const error = ref("");
const toast = ref("");
const health = ref<HealthDetailResponse | null>(null);
const store = useAppStore();
const snippets = store.ragSnippets;

// 进度指示器状态
const pipelineSteps = ['生成大纲', '收集研究笔记', '创作初稿', '审阅反馈', '修改润色', '生成引用'];
const currentPipelineStep = ref(0);

const resetError = () => {
  error.value = "";
};

const handlePlan = async () => {
  try {
    resetError();

    // 表单验证
    if (!form.topic || form.topic.trim() === '') {
      error.value = '请输入主题';
      return;
    }

    loading.plan = true;
    const res = await plan({
      topic: form.topic,
      audience: form.audience,
      style: form.style,
      target_length: form.target_length,
      constraints: form.constraints,
      key_points: form.key_points,
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

    // 表单验证
    const validation = validateDraftRequest({ topic: form.topic, outline: form.outline });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

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
      },
      (evt) => {
        if (evt.type === "result") {
          output.draft = evt.payload?.draft || "";
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

    // 表单验证
    const validation = validateReviewRequest({ draft: output.draft || form.research_notes });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    loading.review = true;
    output.review = "";
    await reviewStream(
      {
        draft: output.draft || form.research_notes,
        criteria: form.review_criteria,
        sources: form.research_notes,
        audience: form.audience,
      },
      (evt) => {
        if (evt.type === "result") {
          output.review = evt.payload?.review || "";
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

    // 表单验证
    const validation = validateRewriteRequest({ draft: output.draft, guidance: output.review || form.review_criteria });
    if (!validation.valid) {
      error.value = validation.errors.join('; ');
      return;
    }

    loading.rewrite = true;
    output.revised = "";
    await rewriteStream(
      {
        draft: output.draft,
        guidance: output.review || form.review_criteria,
        style: form.style,
        target_length: form.target_length,
      },
      (evt) => {
        if (evt.type === "result") {
          output.revised = evt.payload?.revised || "";
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

    loading.pipeline = true;
    currentPipelineStep.value = 0;

    // 将RAG snippets转换为sources格式
    const sources = snippets.map((content, idx) => ({
      doc_id: `snippet-${Date.now()}-${idx}`,
      title: `RAG片段 ${idx + 1}`,
      content: content,
      url: ""
    }));

    const payload: PipelineRequest = {
      topic: form.topic,
      audience: form.audience,
      style: form.style,
      target_length: form.target_length,
      constraints: form.constraints,
      key_points: form.key_points,
      review_criteria: form.review_criteria,
      sources: sources,
    };

    await runPipelineStream(payload, (evt) => {
      if (evt.type === "outline") {
        currentPipelineStep.value = 1;
        output.outline = evt.payload?.outline || "";
        output.assumptions = evt.payload?.assumptions || "";
        output.open_questions = evt.payload?.open_questions || "";
        form.outline = output.outline;
      }
      if (evt.type === "research") {
        currentPipelineStep.value = 2;
        if (!form.research_notes) {
          form.research_notes = evt.payload?.notes_text || "";
        }
      }
      if (evt.type === "draft") {
        currentPipelineStep.value = 3;
        output.draft = evt.payload?.draft || "";
      }
      if (evt.type === "review") {
        currentPipelineStep.value = 4;
        output.review = evt.payload?.review || "";
      }
      if (evt.type === "rewrite") {
        currentPipelineStep.value = 5;
        output.revised = evt.payload?.revised || "";
      }
      if (evt.type === "error") {
        error.value = evt.detail || "Pipeline failed";
      }
      if (evt.type === "result") {
        const res = evt.payload;
        currentPipelineStep.value = pipelineSteps.length;
        output.outline = res.outline;
        output.assumptions = res.assumptions;
        output.open_questions = res.open_questions;
        output.draft = res.draft;
        output.review = res.review;
        output.revised = res.revised;
        output.bibliography = res.bibliography;
        output.version_id = res.version_id;
        output.citations = res.citations || [];
        showToast("Pipeline 完成");
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
  if (snippets.length) {
    form.research_notes = [form.research_notes, ...snippets].filter(Boolean).join("\n\n");
  }
};

const clearSnippets = () => {
  store.clearSnippets();
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
