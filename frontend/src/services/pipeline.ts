import { getApi } from "./api";
import type {
  PipelineRequest,
  PipelineResponse,
  PipelineV2Request,
  PipelineV2Response,
  PipelineV2ResumeRequest,
} from "../types";
import { useAppStore } from "../store";

export const runPipeline = async (payload: PipelineRequest) => {
  const { data } = await getApi().post<PipelineResponse>("/api/pipeline", payload, {
    timeout: 180000,
  });
  return data;
};

export const runPipelineV2 = async (payload: PipelineV2Request) => {
  const { data } = await getApi().post<PipelineV2Response>("/api/pipeline/v2", payload, {
    timeout: 1800000,
  });
  return data;
};

export const resumePipelineV2 = async (payload: PipelineV2ResumeRequest) => {
  const { data } = await getApi().post<PipelineV2Response>("/api/pipeline/v2/resume", payload, {
    timeout: 1800000,
  });
  return data;
};

const runEventStream = async (
  path: string,
  payload: unknown,
  onEvent: (event: any) => void,
  options?: { allowInterrupt?: boolean }
) => {
  const store = useAppStore();
  const allowInterrupt = Boolean(options?.allowInterrupt);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 1800000);

  try {
    const response = await fetch(`${store.apiBase}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`stream failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let lastActivityTime = Date.now();
    const activityTimeout = 600000;
    let receivedResult = false;
    let receivedAny = false;
    let receivedError = false;
    let receivedInterrupt = false;
    let errorDetail = "";
    let shouldStop = false;

    try {
      while (!shouldStop) {
        if (Date.now() - lastActivityTime > activityTimeout) {
          throw new Error("Stream timeout: no data received for 10 minutes");
        }

        const { value, done } = await reader.read();
        if (done) break;

        lastActivityTime = Date.now(); // 更新活动时间
        buffer += decoder.decode(value, { stream: true });
        // 统一行结束符，避免 \r\n 跨 chunk 时分隔失败
        buffer = buffer.replace(/\r/g, "");
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const chunk of parts) {
          const dataLines = chunk
            .split("\n")
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.replace("data:", "").trim())
            .filter(Boolean);
          if (!dataLines.length) continue;
          const jsonStr = dataLines.join("\n");
          if (!jsonStr) continue;
          try {
            const evt = JSON.parse(jsonStr);
            receivedAny = true;
            const isTerminal =
              evt?.type === "result" || evt?.type === "error" || evt?.type === "done";
            if (evt?.type === "result") {
              receivedResult = true;
            }
            if (evt?.type === "interrupt") {
              receivedInterrupt = true;
            }
            if (evt?.type === "error") {
              receivedError = true;
              errorDetail = String(evt?.detail || "Pipeline stream failed");
            }
            if (isTerminal) {
              shouldStop = true;
            }
            try {
              onEvent(evt);
            } catch {
              // 回调异常不应阻断流收尾
            }
            if (isTerminal) {
              break;
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (err: any) {
      const message = String(err?.message || "");
      if (receivedResult || message.includes("BodyStreamBuffer was aborted") || message.includes("aborted")) {
        return;
      }
      throw err;
    } finally {
      if (shouldStop) {
        try {
          await reader.cancel();
        } catch {
        }
      }
      if (!shouldStop && buffer.trim()) {
        const dataLines = buffer
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.replace("data:", "").trim())
          .filter(Boolean);
        if (dataLines.length) {
          const jsonStr = dataLines.join("\n");
          if (jsonStr) {
            try {
              const evt = JSON.parse(jsonStr);
              receivedAny = true;
              const isTerminal =
                evt?.type === "result" || evt?.type === "error" || evt?.type === "done";
              if (evt?.type === "result") {
                receivedResult = true;
              }
              if (evt?.type === "interrupt") {
                receivedInterrupt = true;
              }
              if (evt?.type === "error") {
                receivedError = true;
                errorDetail = String(evt?.detail || "Pipeline stream failed");
              }
              if (isTerminal) {
                shouldStop = true;
              }
              try {
                onEvent(evt);
              } catch {
                // ignore callback errors
              }
            } catch {
            }
          }
        }
      }
    }

    if (receivedError) {
      throw new Error(errorDetail || "Pipeline stream failed");
    }
    if (allowInterrupt && receivedInterrupt) {
      return;
    }
    if (!receivedResult) {
      if (receivedAny) {
        throw new Error("Pipeline stream ended before final result");
      }
      throw new Error("Stream ended before receiving any pipeline event");
    }
  } finally {
    clearTimeout(timeoutId);
  }
};

export const runPipelineStream = async (
  payload: PipelineRequest,
  onEvent: (event: any) => void
) => {
  return runEventStream("/api/pipeline/stream", payload, onEvent);
};

export const runPipelineV2Stream = async (
  payload: PipelineV2Request,
  onEvent: (event: any) => void
) => {
  return runEventStream("/api/pipeline/v2/stream", payload, onEvent, { allowInterrupt: true });
};

export const resumePipelineV2Stream = async (
  payload: PipelineV2ResumeRequest,
  onEvent: (event: any) => void
) => {
  return runEventStream("/api/pipeline/v2/resume/stream", payload, onEvent);
};
