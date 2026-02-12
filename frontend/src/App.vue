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
  background-color: var(--bg-app);
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  height: 64px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(12px); /* 毛玻璃效果 */
  border-bottom: 1px solid var(--border);
  position: sticky; /* 顶部吸附 */
  top: 0;
  z-index: 100;
}

.app-header h1 {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: -0.5px;
}

/* 导航链接美化 */
.app-header nav {
  display: flex;
  gap: 8px;
  background: var(--bg-input);
  padding: 4px;
  border-radius: 99px;
}

.app-header a {
  padding: 6px 16px;
  border-radius: 99px;
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.app-header a:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.5);
}

/* 激活状态 (需要 vue-router 的 link-active-class) */
.app-header a.router-link-active {
  background: #fff;
  color: var(--primary);
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.app-main {
  flex: 1;
  padding: 32px;
  width: 100%;
  max-width: 1400px; /* 限制最大宽度，防止大屏过宽 */
  margin: 0 auto;
}
</style>
