<template>
  <section class="card">
    <h3 v-if="showHeader">差异对比</h3>
    <div class="diff">
      <div v-for="(line, idx) in lines" :key="idx" :class="line.className">
        <span class="prefix">{{ line.prefix }}</span>
        <span class="content">{{ line.content }}</span>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    diff: string;
    showHeader?: boolean;
  }>(),
  {
    showHeader: true,
  }
);

type DiffLine = {
  prefix: string;
  content: string;
  className: string;
};

const lines = computed<DiffLine[]>(() =>
  props.diff.split("\n").map((text) => {
    const prefix = text.length > 0 ? text[0] : " ";
    const content = text.slice(1);
    if (text.startsWith("+++") || text.startsWith("---") || text.startsWith("@@")) {
      return { prefix, content, className: "line meta" };
    }
    if (text.startsWith("+")) {
      return { prefix, content, className: "line add" };
    }
    if (text.startsWith("-")) {
      return { prefix, content, className: "line del" };
    }
    return { prefix: " ", content: text, className: "line" };
  })
);
</script>

<style scoped>
.diff {
  background: #0b1324;
  color: #dbe6f5;
  padding: 14px;
  border-radius: 12px;
  max-height: 68vh;
  overflow-y: auto;
  overflow-x: hidden;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 14px;
  line-height: 1.65;
  border: 1px solid #1f2a44;
}
.line {
  display: grid;
  grid-template-columns: 1.2em minmax(0, 1fr);
  gap: 8px;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  padding: 3px 8px;
  border-radius: 8px;
}
.prefix {
  opacity: 0.92;
  font-weight: 700;
}
.content {
  min-width: 0;
}
.add {
  color: #9ae6b4;
  background: rgba(34, 197, 94, 0.14);
}
.del {
  color: #fda4af;
  background: rgba(244, 63, 94, 0.14);
}
.meta {
  color: #fcd34d;
  background: rgba(251, 191, 36, 0.12);
}
</style>
