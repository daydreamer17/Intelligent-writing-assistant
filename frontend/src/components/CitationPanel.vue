<template>
  <section class="card">
    <h3>引用</h3>
    <ul v-if="citations.length">
      <li v-for="item in citations" :key="item.label">
        <button class="btn ghost" @click="emit('insert', item)">
          插入
        </button>
        {{ item.label }} {{ item.title }}
        <a v-if="item.url" class="muted" :href="item.url" target="_blank" rel="noreferrer">
          {{ item.url }}
        </a>
      </li>
    </ul>
    <pre>{{ bibliography }}</pre>
    <div v-if="versionId" class="muted">版本号：{{ versionId }}</div>
  </section>
</template>

<script setup lang="ts">
const emit = defineEmits<{
  (e: "insert", item: { label: string; title: string; url?: string }): void;
}>();

defineProps<{
  bibliography: string;
  versionId?: number;
  citations: { label: string; title: string; url?: string }[];
}>();
</script>
