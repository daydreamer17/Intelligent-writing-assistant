<template>
  <section class="page">
    <ApiBaseSwitcher />

    <StatusBar :data="health" />
    <div class="card">
      <button class="btn ghost" @click="loadHealth" :disabled="loading">刷新</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { getHealthDetail } from "../services/health";
import StatusBar from "../components/StatusBar.vue";
import ApiBaseSwitcher from "../components/ApiBaseSwitcher.vue";

const health = ref(null as any);
const loading = ref(false);

const loadHealth = async () => {
  loading.value = true;
  health.value = await getHealthDetail();
  loading.value = false;
};
</script>
