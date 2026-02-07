import { getApi } from "./api";
import { useAppStore } from "../store";
import type {
  DraftRequest,
  DraftResponse,
  PlanRequest,
  PlanResponse,
  ReviewRequest,
  ReviewResponse,
  RewriteRequest,
  RewriteResponse,
} from "../types";

export const plan = async (payload: PlanRequest) => {
  const { data } = await getApi().post<PlanResponse>("/api/plan", payload, {
    timeout: 180000,
  });
  return data;
};

export const draft = async (payload: DraftRequest) => {
  const { data } = await getApi().post<DraftResponse>("/api/draft", payload, {
    timeout: 600000,
  });
  return data;
};

export const draftStream = async (
  payload: DraftRequest,
  onEvent: (event: any) => void
) => {
  const store = useAppStore();
  const response = await fetch(`${store.apiBase}/api/draft/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const chunk of parts) {
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const jsonStr = line.replace("data:", "").trim();
        if (!jsonStr) continue;
        try {
          const evt = JSON.parse(jsonStr);
          onEvent(evt);
        } catch {
          // ignore parse errors
        }
      }
    }
  } catch (err: any) {
    const message = String(err?.message || "");
    if (!message.includes("BodyStreamBuffer was aborted")) {
      throw err;
    }
  } finally {
    if (buffer.trim()) {
      const line = buffer.split("\n").find((l) => l.startsWith("data:"));
      if (line) {
        const jsonStr = line.replace("data:", "").trim();
        if (jsonStr) {
          try {
            const evt = JSON.parse(jsonStr);
            onEvent(evt);
          } catch {
            // ignore parse errors
          }
        }
      }
    }
  }
};

export const review = async (payload: ReviewRequest) => {
  const { data } = await getApi().post<ReviewResponse>("/api/review", payload, {
    timeout: 600000,
  });
  return data;
};

export const reviewStream = async (
  payload: ReviewRequest,
  onEvent: (event: any) => void
) => {
  const store = useAppStore();
  const response = await fetch(`${store.apiBase}/api/review/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const chunk of parts) {
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const jsonStr = line.replace("data:", "").trim();
        if (!jsonStr) continue;
        try {
          const evt = JSON.parse(jsonStr);
          onEvent(evt);
        } catch {
          // ignore parse errors
        }
      }
    }
  } catch (err: any) {
    const message = String(err?.message || "");
    if (!message.includes("BodyStreamBuffer was aborted")) {
      throw err;
    }
  } finally {
    if (buffer.trim()) {
      const line = buffer.split("\n").find((l) => l.startsWith("data:"));
      if (line) {
        const jsonStr = line.replace("data:", "").trim();
        if (jsonStr) {
          try {
            const evt = JSON.parse(jsonStr);
            onEvent(evt);
          } catch {
            // ignore parse errors
          }
        }
      }
    }
  }
};

export const rewrite = async (payload: RewriteRequest) => {
  const { data } = await getApi().post<RewriteResponse>("/api/rewrite", payload, {
    timeout: 600000,
  });
  return data;
};

export const rewriteStream = async (
  payload: RewriteRequest,
  onEvent: (event: any) => void
) => {
  const store = useAppStore();
  const response = await fetch(`${store.apiBase}/api/rewrite/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const chunk of parts) {
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const jsonStr = line.replace("data:", "").trim();
        if (!jsonStr) continue;
        try {
          const evt = JSON.parse(jsonStr);
          onEvent(evt);
        } catch {
          // ignore parse errors
        }
      }
    }
  } catch (err: any) {
    const message = String(err?.message || "");
    if (!message.includes("BodyStreamBuffer was aborted")) {
      throw err;
    }
  } finally {
    if (buffer.trim()) {
      const line = buffer.split("\n").find((l) => l.startsWith("data:"));
      if (line) {
        const jsonStr = line.replace("data:", "").trim();
        if (jsonStr) {
          try {
            const evt = JSON.parse(jsonStr);
            onEvent(evt);
          } catch {
            // ignore parse errors
          }
        }
      }
    }
  }
};
