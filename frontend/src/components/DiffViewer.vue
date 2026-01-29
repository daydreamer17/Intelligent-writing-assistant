<template>
  <section class="card">
    <h3>差异对比</h3>
    <div class="diff">
      <div v-for="(line, idx) in lines" :key="idx" :class="line.className">
        {{ line.text }}
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  diff: string;
}>();

const lines = computed(() =>
  props.diff.split("\n").map((text) => ({
    text,
    className: text.startsWith("+")
      ? "line add"
      : text.startsWith("-")
      ? "line del"
      : text.startsWith("@@")
      ? "line meta"
      : "line",
  }))
);
</script>

<style scoped>
.diff {
  background: #0f172a;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 10px;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.line {
  white-space: pre;
}
.add {
  color: #86efac;
}
.del {
  color: #fca5a5;
}
.meta {
  color: #fbbf24;
}
</style>

<!-- function defineProps<T>() {
  throw new Error("Function not implemented.");
} -->
