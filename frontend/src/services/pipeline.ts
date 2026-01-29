import { getApi } from "./api";
import type { PipelineRequest, PipelineResponse } from "../types";
import { useAppStore } from "../store";

export const runPipeline = async (payload: PipelineRequest) => {
  const { data } = await getApi().post<PipelineResponse>("/api/pipeline", payload, {
    timeout: 180000,
  });
  return data;
};

export const runPipelineStream = async (
  payload: PipelineRequest,
  onEvent: (event: any) => void
) => {
  const store = useAppStore();
  const response = await fetch(`${store.apiBase}/api/pipeline/stream`, {
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
};
