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

  // 创建 AbortController 用于超时控制
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 900000); // 15分钟超时

  try {
    const response = await fetch(`${store.apiBase}/api/pipeline/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Connection": "keep-alive", // 保持连接
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
      // 禁用浏览器默认的超时行为
      keepalive: true,
    });

    if (!response.ok || !response.body) {
      throw new Error(`stream failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let lastActivityTime = Date.now();
    const activityTimeout = 600000; // 10分钟无数据才认为连接断开
    let receivedResult = false;

    while (true) {
      // 检查是否长时间无数据
      if (Date.now() - lastActivityTime > activityTimeout) {
        throw new Error("Stream timeout: no data received for 10 minutes");
      }

      const { value, done } = await reader.read();
      if (done) break;

      lastActivityTime = Date.now(); // 更新活动时间
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
          if (evt?.type === "result") {
            receivedResult = true;
          }
          onEvent(evt);
        } catch {
          // ignore parse errors
        }
      }
    }

    if (!receivedResult) {
      throw new Error("Stream ended before final result");
    }
  } finally {
    clearTimeout(timeoutId); // 清除超时定时器
  }
};
