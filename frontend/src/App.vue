<template>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <h1>Writing Studio</h1>
        <div class="muted">Plan · Draft · Review · Rewrite</div>
      </div>
      <nav>
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/workspace">Workspace</RouterLink>
        <RouterLink to="/rag">RAG</RouterLink>
        <RouterLink to="/history">History</RouterLink>
        <RouterLink to="/settings">Settings</RouterLink>
      </nav>
    </header>
    <main class="app-main">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts"></script>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: linear-gradient(135deg, #111827, #1f2937);
  color: #f9fafb;
}
.app-header nav {
  display: flex;
  gap: 12px;
}
.app-header a {
  color: #e5e7eb;
  text-decoration: none;
}
.app-main {
  flex: 1;
  padding: 24px;
  background: #f3f4f6;
}
</style>
