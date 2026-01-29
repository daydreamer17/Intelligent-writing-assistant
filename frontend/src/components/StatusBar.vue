<template>
  <section class="card">
    <h3>系统状态</h3>
    <div class="grid-3">
      <div>
        <div class="muted">Memory</div>
        <div>{{ data?.memory_mode }}</div>
      </div>
      <div>
        <div class="muted">Qdrant</div>
        <div>{{ data?.qdrant?.enabled ? "enabled" : "disabled" }}</div>
      </div>
      <div>
        <div class="muted">Embedding</div>
        <div>{{ data?.embedding?.mode }}</div>
      </div>
    </div>
    <pre class="muted">{{ data }}</pre>
  </section>
</template>

<script setup lang="ts">
import type { HealthDetailResponse } from "../types";

defineProps<{
  data: HealthDetailResponse | null;
}>();
</script>
