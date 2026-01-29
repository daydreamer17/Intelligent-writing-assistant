<template>
  <section class="card">
    <h3>API Base</h3>
    <div class="field">
      <label>Base URL</label>
      <input v-model="localValue" />
    </div>
    <button class="btn" @click="apply">应用</button>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import { useAppStore } from "../store";

const store = useAppStore();
const localValue = ref(store.apiBase);

watch(
  () => store.apiBase,
  (value) => {
    localValue.value = value;
  }
);

const apply = () => {
  store.setApiBase(localValue.value);
};
</script>
