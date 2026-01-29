import { createRouter, createWebHistory } from "vue-router";
import Home from "../views/Home.vue";
import Workspace from "../views/Workspace.vue";
import RagCenter from "../views/RagCenter.vue";
import History from "../views/History.vue";
import Settings from "../views/Settings.vue";

const routes = [
  { path: "/", component: Home },
  { path: "/workspace", component: Workspace },
  { path: "/rag", component: RagCenter },
  { path: "/history", component: History },
  { path: "/settings", component: Settings },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
