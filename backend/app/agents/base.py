from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import logging
import time
import threading

from hello_agents import HelloAgentsLLM, SimpleAgent, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry

logger = logging.getLogger("app.llm")

@dataclass(frozen=True)
class AgentRuntimeConfig:
    name: str
    system_prompt: str
    temperature: float = 0.2
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    request_timeout: Optional[float] = None
    max_tokens: Optional[int] = None


def build_llm(config: AgentRuntimeConfig) -> HelloAgentsLLM:
    provider = config.provider or os.getenv("LLM_PROVIDER") or ""
    model = config.model or os.getenv("LLM_MODEL")
    api_key = config.api_key or os.getenv("LLM_API_KEY")
    base_url = config.base_url or os.getenv("LLM_API_BASE")

    llm_kwargs: dict[str, Any] = {"temperature": config.temperature}
    if provider:
        llm_kwargs["provider"] = provider
    if model:
        llm_kwargs["model"] = model
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    timeout = config.request_timeout
    if timeout is None:
        env_timeout = os.getenv("LLM_TIMEOUT")
        if env_timeout:
            try:
                timeout = float(env_timeout)
            except ValueError:
                timeout = None
    if timeout is not None:
        llm_kwargs["timeout"] = timeout
    max_tokens = config.max_tokens
    if max_tokens is None:
        env_max_tokens = os.getenv("LLM_MAX_TOKENS")
        if env_max_tokens:
            try:
                max_tokens = int(env_max_tokens)
            except ValueError:
                max_tokens = None
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    has_key = bool(api_key)
    has_base = bool(base_url)
    if has_key and has_base:
        logger.info("LLM config OK (api_key=set, base_url=%s)", base_url)
    else:
        logger.warning(
            "LLM config incomplete (api_key=%s, base_url=%s).",
            "set" if has_key else "missing",
            base_url or "missing",
        )

    try:
        return HelloAgentsLLM(**llm_kwargs)
    except TypeError:
        if "timeout" in llm_kwargs:
            llm_kwargs.pop("timeout", None)
            return HelloAgentsLLM(**llm_kwargs)
        raise


class BaseWritingAgent:
    def __init__(self, config: AgentRuntimeConfig, tool_registry: ToolRegistry | None = None) -> None:
        self.config = config
        self.llm = build_llm(config)
        self._lock = threading.Lock()

        # Keep a persistent agent instance to preserve conversation memory across calls.
        # We prune history to a bounded window to avoid hitting provider max_seq_len.
        if tool_registry is not None:
            self.agent = ToolAwareSimpleAgent(
                name=config.name,
                llm=self.llm,
                system_prompt=config.system_prompt,
                enable_tool_calling=True,
                tool_registry=tool_registry,
            )
        else:
            self.agent = SimpleAgent(
                name=config.name,
                llm=self.llm,
                system_prompt=config.system_prompt,
            )

    def run(self, input_text: str, **kwargs: Any) -> str:
        input_text = _truncate_input(input_text)
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        attempt = 0
        while True:
            with self._lock:
                _prune_agent_history(self.agent, upcoming_user_chars=len(input_text))
            try:
                with self._lock:
                    response = self.agent.run(input_text, **kwargs)
                logger.info("LLM call success. Output:\n%s", response)
                _cooldown_after_call()
                return response
            except Exception as exc:
                # 上下文超限：缩短输入并重试（避免 max_total_tokens > max_seq_len）
                if _is_context_overflow_error(exc) and attempt < max_retries:
                    input_text = _shrink_text(input_text, factor=0.8)
                    logger.warning(
                        "LLM context overflow. Shrinking input and retrying (attempt %s/%s).",
                        attempt + 1,
                        max_retries,
                    )
                    attempt += 1
                    continue
                # 速率限制错误
                if _is_rate_limit_error(exc) and attempt < max_retries:
                    delay = backoff_base * (2 ** attempt)
                    logger.warning(
                        "LLM rate limited. Retrying in %.1fs (attempt %s/%s).",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                # 连接中断错误
                if _is_connection_error(exc) and attempt < max_retries:
                    delay = min(5.0, backoff_base * 0.5) * (2 ** attempt)
                    logger.warning(
                        "LLM connection error. Retrying in %.1fs (attempt %s/%s).",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                logger.error("LLM call failed: %s", exc, exc_info=True)
                raise

    def stream(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        input_text = _truncate_input(input_text)
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        attempt = 0
        while True:
            emitted = False
            with self._lock:
                _prune_agent_history(self.agent, upcoming_user_chars=len(input_text))
            try:
                # Hold the lock during the whole stream so concurrent requests don't
                # interleave and corrupt the shared conversation history.
                with self._lock:
                    for chunk in self.agent.stream_run(input_text, **kwargs):
                        if not chunk:
                            continue
                        emitted = True
                        yield chunk
                _cooldown_after_call()
                return
            except Exception as exc:
                # 上下文超限：只在未发送数据时重试，避免前端拼接重复/断裂
                if _is_context_overflow_error(exc) and attempt < max_retries and not emitted:
                    input_text = _shrink_text(input_text, factor=0.8)
                    logger.warning(
                        "LLM stream context overflow. Shrinking input and retrying (attempt %s/%s).",
                        attempt + 1,
                        max_retries,
                    )
                    attempt += 1
                    continue
                # 速率限制错误：只在未发送数据时重试
                if _is_rate_limit_error(exc) and attempt < max_retries and not emitted:
                    delay = backoff_base * (2 ** attempt)
                    logger.warning(
                        "LLM stream rate limited. Retrying in %.1fs (attempt %s/%s).",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                # 连接中断错误：即使已发送部分数据也重试（但会丢失已发送的部分）
                if _is_connection_error(exc) and attempt < max_retries:
                    delay = min(5.0, backoff_base * 0.5) * (2 ** attempt)  # 更短的重试间隔
                    logger.warning(
                        "LLM stream connection error. Retrying in %.1fs (attempt %s/%s). %s",
                        delay,
                        attempt + 1,
                        max_retries,
                        "Partial output may be lost." if emitted else "",
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                logger.error("LLM stream failed: %s", exc, exc_info=True)
                raise


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate limiting" in message
        or "tpm limit" in message
        or "too many requests" in message
        or "429" in message
    )


def _is_connection_error(exc: Exception) -> bool:
    """检测连接中断错误，这些错误可以重试"""
    message = str(exc).lower()
    return (
        "peer closed connection" in message
        or "incomplete chunked read" in message
        or "connection reset" in message
        or "connection aborted" in message
        or "broken pipe" in message
        or "remotedisconnected" in message
        or "connection error" in message
        or "timeout" in message
    )


def _is_context_overflow_error(exc: Exception) -> bool:
    message = str(exc).lower()
    # Common providers return an error like:
    # "max_total_tokens (...) must be less than or equal to max_seq_len (...)"
    return (
        ("max_total_tokens" in message and "max_seq_len" in message)
        or "context length" in message
        or "maximum context" in message
        or "prompt is too long" in message
    )


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _truncate_input(text: str) -> str:
    max_chars = _parse_int_env("LLM_MAX_INPUT_CHARS", default=0)
    max_context = _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    max_output = _parse_int_env("LLM_MAX_TOKENS", default=0)
    safety_margin = _parse_int_env("LLM_INPUT_SAFETY_MARGIN", default=4000)
    # Rough heuristic: with CJK text 1 token is often ~1 char, but prompts can contain
    # JSON/markdown/URLs; default to a conservative ratio to avoid context overflow.
    chars_per_token = _parse_float_env("LLM_CHARS_PER_TOKEN", default=0.5)
    if max_context > 0:
        budget = max_context - max_output - safety_margin
        if budget <= 0:
            fallback = 4000
            max_chars = min(max_chars, fallback) if max_chars > 0 else fallback
        else:
            budget_chars = int(budget * chars_per_token)
            if budget_chars > 0:
                max_chars = budget_chars if max_chars <= 0 else min(max_chars, budget_chars)
        logger.info(
            "LLM context budget: max_context=%s max_output=%s safety_margin=%s chars_per_token=%.3f => max_input_chars=%s",
            max_context,
            max_output,
            safety_margin,
            chars_per_token,
            max_chars if max_chars > 0 else "unset",
        )
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    truncated = text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()
    logger.warning(
        "Input truncated from %s to %s chars to respect LLM_MAX_INPUT_CHARS.",
        len(text),
        len(truncated),
    )
    return truncated


def _shrink_text(text: str, factor: float = 0.8) -> str:
    # Emergency shrink for provider-side context overflow errors. Keep head+tail to
    # preserve instruction + recent content.
    if not text:
        return text
    if factor <= 0 or factor >= 1:
        factor = 0.8
    max_chars = max(2000, int(len(text) * factor))
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    truncated = text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()
    logger.warning("Emergency shrink input from %s to %s chars.", len(text), len(truncated))
    return truncated


def _prune_agent_history(agent: Any, *, upcoming_user_chars: int) -> None:
    """
    Keep conversation memory, but bound it so the provider context window isn't exceeded.

    hello_agents SimpleAgent stores history in-memory; since our services are singletons,
    this history can grow across requests and eventually hit max_seq_len.
    """
    max_context = _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    max_output = _parse_int_env("LLM_MAX_TOKENS", default=0)
    safety_margin = _parse_int_env("LLM_INPUT_SAFETY_MARGIN", default=4000)
    chars_per_token = _parse_float_env("LLM_CHARS_PER_TOKEN", default=0.5)

    # Optional hard cap for history size (chars). If unset, derive from context budget.
    history_cap = _parse_int_env("LLM_HISTORY_MAX_CHARS", default=0)

    if max_context <= 0:
        return
    budget_tokens = max_context - max_output - safety_margin
    if budget_tokens <= 0:
        # Too risky: drop history entirely.
        if hasattr(agent, "clear_history"):
            agent.clear_history()
        return

    budget_chars = int(budget_tokens * chars_per_token)
    allowed_history_chars = max(0, budget_chars - upcoming_user_chars)
    if history_cap > 0:
        allowed_history_chars = min(allowed_history_chars, history_cap)

    if allowed_history_chars <= 0:
        if hasattr(agent, "clear_history"):
            agent.clear_history()
        logger.warning("Cleared agent history due to tight context budget.")
        return

    if not hasattr(agent, "get_history") or not hasattr(agent, "clear_history") or not hasattr(agent, "add_message"):
        return

    history = agent.get_history()
    if not history:
        return

    kept: list[Any] = []
    total = 0
    for msg in reversed(history):
        content = getattr(msg, "content", "") or ""
        msg_len = len(content)
        if kept and total + msg_len > allowed_history_chars:
            break
        if not kept and msg_len > allowed_history_chars:
            # Keep at least the last message, but truncate it to fit the cap.
            truncated = _truncate_text_tail(content, allowed_history_chars)
            kept = [_clone_message(msg, truncated)]
            total = len(truncated)
            break
        kept.append(msg)
        total += msg_len
    kept.reverse()

    if len(kept) == len(history):
        return

    agent.clear_history()
    for msg in kept:
        agent.add_message(msg)
    logger.info(
        "Pruned agent history: kept %s/%s messages (%s/%s chars allowed=%s).",
        len(kept),
        len(history),
        total,
        sum(len(getattr(m, "content", "") or "") for m in history),
        allowed_history_chars,
    )


def _truncate_text_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars < 32:
        return text[-max_chars:]
    return "...\n" + text[-(max_chars - 4):]


def _clone_message(msg: Any, new_content: str) -> Any:
    role = getattr(msg, "role", "assistant")
    metadata = getattr(msg, "metadata", None)
    timestamp = getattr(msg, "timestamp", None)
    try:
        # hello_agents Message signature is (content, role, **kwargs)
        kwargs: dict[str, Any] = {}
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        if metadata is not None:
            kwargs["metadata"] = metadata
        return msg.__class__(new_content, role, **kwargs)
    except Exception:
        return msg


def _cooldown_after_call() -> None:
    cooldown = _parse_float_env("LLM_COOLDOWN_SECONDS", default=0.0)
    if cooldown > 0:
        time.sleep(cooldown)
