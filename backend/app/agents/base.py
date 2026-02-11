from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import logging
import time
import threading
from datetime import datetime

from hello_agents import HelloAgentsLLM, SimpleAgent, ToolAwareSimpleAgent
from hello_agents.core.message import Message
from hello_agents.tools import ToolRegistry
from ..utils.tokenizer import tokenize

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
        self._tool_registry = tool_registry
        self._session_agents: dict[str, Any] = {}
        self._session_last_seen: dict[str, float] = {}

        # Keep a default session for compatibility; real traffic should pass session_id.
        default = self._create_agent()
        self._session_agents["__default__"] = default
        self._session_last_seen["__default__"] = time.time()
        self.agent = default

    def _create_agent(self) -> Any:
        if self._tool_registry is not None:
            return ToolAwareSimpleAgent(
                name=self.config.name,
                llm=self.llm,
                system_prompt=self.config.system_prompt,
                enable_tool_calling=True,
                tool_registry=self._tool_registry,
            )
        return SimpleAgent(
            name=self.config.name,
            llm=self.llm,
            system_prompt=self.config.system_prompt,
        )

    def _session_key(self, session_id: Optional[str]) -> str:
        mode = (os.getenv("CONVERSATION_MEMORY_MODE", "session") or "session").strip().lower()
        if mode == "global":
            return "__default__"
        raw = (session_id or "").strip()
        if not raw:
            return "__default__"
        return raw[:64]

    def _get_session_agent(self, session_id: Optional[str]) -> tuple[str, Any]:
        key = self._session_key(session_id)
        agent = self._session_agents.get(key)
        if agent is None:
            agent = self._create_agent()
            self._session_agents[key] = agent
            logger.info("Created isolated LLM session: agent=%s session_id=%s", self.config.name, key)
        self._session_last_seen[key] = time.time()
        self._evict_old_sessions()
        return key, agent

    def _evict_old_sessions(self) -> None:
        max_agents = _parse_int_env("LLM_SESSION_MAX_AGENTS", default=32)
        ttl_seconds = _parse_int_env("LLM_SESSION_TTL_SECONDS", default=3600)
        now = time.time()
        keys = list(self._session_agents.keys())
        for key in keys:
            if key == "__default__":
                continue
            last_seen = self._session_last_seen.get(key, now)
            if ttl_seconds > 0 and now - last_seen > ttl_seconds:
                self._session_agents.pop(key, None)
                self._session_last_seen.pop(key, None)
                logger.info("Evicted idle LLM session: agent=%s session_id=%s", self.config.name, key)

        if max_agents > 0 and len(self._session_agents) > max_agents:
            active = sorted(
                ((k, self._session_last_seen.get(k, 0.0)) for k in self._session_agents.keys() if k != "__default__"),
                key=lambda item: item[1],
            )
            while len(self._session_agents) > max_agents and active:
                key, _ = active.pop(0)
                self._session_agents.pop(key, None)
                self._session_last_seen.pop(key, None)
                logger.info("Evicted overflow LLM session: agent=%s session_id=%s", self.config.name, key)

    def clear_session_memory(
        self,
        *,
        session_id: str = "",
        drop_agent: bool = False,
        clear_cold: bool = False,
    ) -> bool:
        key = self._session_key(session_id)
        with self._lock:
            agent = self._session_agents.get(key)
            if agent is None:
                return False

            cleared = False
            if hasattr(agent, "clear_history"):
                try:
                    agent.clear_history()
                    cleared = True
                except Exception:
                    pass

            if clear_cold and hasattr(agent, "_cold_memory"):
                try:
                    setattr(agent, "_cold_memory", [])
                    _delete_cold_memory_rows(agent_name=getattr(agent, "name", "unknown"), session_id=key)
                    cleared = True
                except Exception:
                    pass

            if drop_agent and key != "__default__":
                self._session_agents.pop(key, None)
                self._session_last_seen.pop(key, None)
                cleared = True
                logger.info("Dropped LLM session: agent=%s session_id=%s", self.config.name, key)

            return cleared

    def run(self, input_text: str, **kwargs: Any) -> str:
        session_id = kwargs.pop("session_id", None)
        max_input_chars = kwargs.pop("max_input_chars", None)
        max_context_tokens = kwargs.pop("max_context_tokens", None)
        input_safety_margin = kwargs.pop("input_safety_margin", None)
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        clear_on_overflow = os.getenv("LLM_CLEAR_SESSION_ON_OVERFLOW", "true").lower() in ("1", "true", "yes")
        cleared_session_once = False
        attempt = 0
        while True:
            tool_calling_temporarily_disabled = False
            original_tool_calling = None
            with self._lock:
                session_key, agent = self._get_session_agent(session_id)
                setattr(agent, "_session_id", session_key)
                prepared_input = _augment_with_cold_recall(agent, input_text)
                prepared_input = _truncate_input(
                    prepared_input,
                    max_chars=max_input_chars,
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                _maybe_compress_history(agent, self.llm)
                _prune_agent_history(
                    agent,
                    upcoming_user_chars=len(prepared_input),
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                _hard_cap_agent_history(agent)
                if hasattr(agent, "enable_tool_calling"):
                    original_tool_calling = getattr(agent, "enable_tool_calling", None)
                tool_calling_temporarily_disabled = _maybe_disable_tool_prompt(
                    agent,
                    upcoming_user_chars=len(prepared_input),
                )
            try:
                try:
                    with self._lock:
                        response = agent.run(prepared_input, **kwargs)
                finally:
                    if (
                        tool_calling_temporarily_disabled
                        and original_tool_calling is not None
                        and hasattr(agent, "enable_tool_calling")
                    ):
                        setattr(agent, "enable_tool_calling", original_tool_calling)
                logger.info("LLM call success. Output:\n%s", response)
                _cooldown_after_call()
                return response
            except Exception as exc:
                # 上下文超限：缩短输入并重试（避免 max_total_tokens > max_seq_len）
                if _is_context_overflow_error(exc) and attempt < max_retries:
                    if clear_on_overflow and not cleared_session_once:
                        cleared = self.clear_session_memory(session_id=session_key, drop_agent=True)
                        if cleared:
                            logger.warning(
                                "LLM context overflow. Cleared session memory and retrying: agent=%s session_id=%s.",
                                self.config.name,
                                session_key,
                            )
                        cleared_session_once = True
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
        session_id = kwargs.pop("session_id", None)
        max_input_chars = kwargs.pop("max_input_chars", None)
        max_context_tokens = kwargs.pop("max_context_tokens", None)
        input_safety_margin = kwargs.pop("input_safety_margin", None)
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        clear_on_overflow = os.getenv("LLM_CLEAR_SESSION_ON_OVERFLOW", "true").lower() in ("1", "true", "yes")
        cleared_session_once = False
        attempt = 0
        while True:
            emitted = False
            tool_calling_temporarily_disabled = False
            original_tool_calling = None
            with self._lock:
                session_key, agent = self._get_session_agent(session_id)
                setattr(agent, "_session_id", session_key)
                prepared_input = _augment_with_cold_recall(agent, input_text)
                prepared_input = _truncate_input(
                    prepared_input,
                    max_chars=max_input_chars,
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                _maybe_compress_history(agent, self.llm)
                _prune_agent_history(
                    agent,
                    upcoming_user_chars=len(prepared_input),
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                _hard_cap_agent_history(agent)
                if hasattr(agent, "enable_tool_calling"):
                    original_tool_calling = getattr(agent, "enable_tool_calling", None)
                tool_calling_temporarily_disabled = _maybe_disable_tool_prompt(
                    agent,
                    upcoming_user_chars=len(prepared_input),
                )
            try:
                try:
                    # Hold the lock during the whole stream so concurrent requests don't
                    # interleave and corrupt the shared conversation history.
                    with self._lock:
                        for chunk in agent.stream_run(prepared_input, **kwargs):
                            if not chunk:
                                continue
                            emitted = True
                            yield chunk
                finally:
                    if (
                        tool_calling_temporarily_disabled
                        and original_tool_calling is not None
                        and hasattr(agent, "enable_tool_calling")
                    ):
                        setattr(agent, "enable_tool_calling", original_tool_calling)
                _cooldown_after_call()
                return
            except Exception as exc:
                # 上下文超限：只在未发送数据时重试，避免前端拼接重复/断裂
                if _is_context_overflow_error(exc) and attempt < max_retries and not emitted:
                    if clear_on_overflow and not cleared_session_once:
                        cleared = self.clear_session_memory(session_id=session_key, drop_agent=True)
                        if cleared:
                            logger.warning(
                                "LLM stream context overflow. Cleared session memory and retrying: agent=%s session_id=%s.",
                                self.config.name,
                                session_key,
                            )
                        cleared_session_once = True
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


def _truncate_input(
    text: str,
    *,
    max_chars: int | None = None,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    safety_margin: int | None = None,
) -> str:
    max_chars = max_chars if isinstance(max_chars, int) and max_chars >= 0 else _parse_int_env("LLM_MAX_INPUT_CHARS", default=0)
    max_context = (
        max_context_tokens
        if isinstance(max_context_tokens, int) and max_context_tokens >= 0
        else _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    )
    max_output = (
        max_output_tokens
        if isinstance(max_output_tokens, int) and max_output_tokens >= 0
        else _parse_int_env("LLM_MAX_TOKENS", default=0)
    )
    safety_margin = (
        safety_margin if isinstance(safety_margin, int) and safety_margin >= 0 else _parse_int_env("LLM_INPUT_SAFETY_MARGIN", default=4000)
    )
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


def _history_chars(agent: Any) -> int:
    if not hasattr(agent, "get_history"):
        return 0
    try:
        history = agent.get_history() or []
    except Exception:
        return 0
    total = 0
    for msg in history:
        total += len(getattr(msg, "content", "") or "")
    return total


def _maybe_disable_tool_prompt(agent: Any, *, upcoming_user_chars: int) -> bool:
    """
    Prevent tool schema bloat from exploding prompt tokens.

    ToolAwareSimpleAgent injects all tool descriptions into the system prompt.
    When MCP expands many tools, this can exceed provider max_prompt_tokens even
    if user input/history were already truncated.
    """
    if not hasattr(agent, "enable_tool_calling") or not getattr(agent, "enable_tool_calling", False):
        return False
    if not hasattr(agent, "_get_enhanced_system_prompt"):
        return False

    try:
        enhanced_prompt = agent._get_enhanced_system_prompt()
    except Exception:
        return False

    chars_per_token = _parse_float_env("LLM_CHARS_PER_TOKEN", default=0.5)
    system_tokens = _estimate_tokens(len(enhanced_prompt), chars_per_token)
    history_tokens = _estimate_tokens(_history_chars(agent), chars_per_token)
    input_tokens = _estimate_tokens(upcoming_user_chars, chars_per_token)
    projected_total = system_tokens + history_tokens + input_tokens

    max_tool_prompt_tokens = _parse_int_env("LLM_TOOL_PROMPT_MAX_TOKENS", default=4000)
    max_context_tokens = _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    context_ratio = _parse_float_env("LLM_TOOL_PROMPT_CONTEXT_RATIO", default=0.8)

    oversize_system = system_tokens > max_tool_prompt_tokens
    near_context = max_context_tokens > 0 and projected_total > int(max_context_tokens * context_ratio)
    if not (oversize_system or near_context):
        return False

    setattr(agent, "enable_tool_calling", False)
    logger.warning(
        "Temporarily disabled tool calling due to oversized tool prompt: "
        "system_tokens=%s history_tokens=%s input_tokens=%s projected=%s max_context=%s.",
        system_tokens,
        history_tokens,
        input_tokens,
        projected_total,
        max_context_tokens if max_context_tokens > 0 else "unset",
    )
    return True


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


def _prune_agent_history(
    agent: Any,
    *,
    upcoming_user_chars: int,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    safety_margin: int | None = None,
) -> None:
    """
    Keep conversation memory, but bound it so the provider context window isn't exceeded.

    hello_agents SimpleAgent stores history in-memory; since our services are singletons,
    this history can grow across requests and eventually hit max_seq_len.
    """
    max_context = (
        max_context_tokens
        if isinstance(max_context_tokens, int) and max_context_tokens >= 0
        else _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    )
    max_output = (
        max_output_tokens
        if isinstance(max_output_tokens, int) and max_output_tokens >= 0
        else _parse_int_env("LLM_MAX_TOKENS", default=0)
    )
    safety_margin = (
        safety_margin if isinstance(safety_margin, int) and safety_margin >= 0 else _parse_int_env("LLM_INPUT_SAFETY_MARGIN", default=4000)
    )
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

    sanitized, deferred, forgotten = _apply_memory_policy(history)
    _log_memory_policy_stats(sanitized, deferred, forgotten, context="prune")
    if deferred:
        _push_cold_memory(agent, deferred)
    if forgotten:
        logger.info("Forgot %s memory items by policy.", len(forgotten))

    pinned = [msg for msg, action in sanitized if action == "pin"]
    normal = [msg for msg, action in sanitized if action in ("normal", "compress")]

    # Bound pinned ratio first, otherwise pin-only history can starve normal context.
    max_pin_ratio = _parse_float_env("LLM_MEMORY_MAX_PIN_RATIO", default=0.35)
    max_pinned_count = max(1, int(len(sanitized) * _clamp(max_pin_ratio, 0.05, 0.95)))
    if len(pinned) > max_pinned_count:
        pinned = pinned[-max_pinned_count:]

    pinned_chars = sum(len(getattr(m, "content", "") or "") for m in pinned)
    if pinned and pinned_chars > allowed_history_chars:
        # Keep newest pinned messages and trim the last one as needed so total <= allowed.
        kept_pinned: list[Any] = []
        used = 0
        for msg in reversed(pinned):
            if used >= allowed_history_chars:
                break
            content = getattr(msg, "content", "") or ""
            room = allowed_history_chars - used
            if room <= 0:
                break
            if len(content) > room:
                content = _truncate_text_tail(content, room)
                msg = _clone_message(msg, content)
            kept_pinned.append(msg)
            used += len(getattr(msg, "content", "") or "")
        kept_pinned.reverse()
        pinned = kept_pinned
        pinned_chars = used

    available = max(0, allowed_history_chars - pinned_chars)
    kept_norm: list[Any] = []
    total = 0
    for msg in reversed(normal):
        content = getattr(msg, "content", "") or ""
        msg_len = len(content)
        if kept_norm and total + msg_len > available:
            break
        if not kept_norm and msg_len > available:
            if available > 0:
                truncated = _truncate_text_tail(content, available)
                kept_norm = [_clone_message(msg, truncated)]
                total = len(truncated)
            break
        kept_norm.append(msg)
        total += msg_len
    kept_norm.reverse()

    keep_set = {id(msg) for msg in (pinned + kept_norm)}
    kept: list[Any] = [msg for msg, _ in sanitized if id(msg) in keep_set]

    if len(kept) == len(history):
        return

    agent.clear_history()
    for msg in kept:
        agent.add_message(msg)
    logger.info(
        "Pruned agent history: kept %s/%s messages (%s/%s chars allowed=%s).",
        len(kept),
        len(history),
        sum(len(getattr(m, "content", "") or "") for m in kept),
        sum(len(getattr(m, "content", "") or "") for m in history),
        allowed_history_chars,
    )


def _hard_cap_agent_history(agent: Any) -> None:
    if not hasattr(agent, "get_history") or not hasattr(agent, "clear_history") or not hasattr(agent, "add_message"):
        return
    max_messages = max(0, _parse_int_env("LLM_SESSION_MAX_HISTORY_MESSAGES", default=80))
    max_chars = max(0, _parse_int_env("LLM_SESSION_MAX_HISTORY_CHARS", default=12000))
    if max_messages <= 0 and max_chars <= 0:
        return

    history = agent.get_history() or []
    if not history:
        return
    total_chars = sum(len(getattr(m, "content", "") or "") for m in history)
    if (max_messages <= 0 or len(history) <= max_messages) and (max_chars <= 0 or total_chars <= max_chars):
        return

    summary_prefix = ("[LONG_TERM_SUMMARY]", "[RECENT_SUMMARY]")
    pinned_summaries = [m for m in history if (getattr(m, "content", "") or "").startswith(summary_prefix)]
    normal = [m for m in history if m not in pinned_summaries]

    kept_rev: list[Any] = []
    used_chars = sum(len(getattr(m, "content", "") or "") for m in pinned_summaries)
    max_normal_messages = max_messages - len(pinned_summaries) if max_messages > 0 else 0
    for msg in reversed(normal):
        if max_messages > 0 and len(kept_rev) >= max(0, max_normal_messages):
            break
        content = getattr(msg, "content", "") or ""
        if max_chars > 0 and used_chars + len(content) > max_chars:
            if not kept_rev:
                room = max(0, max_chars - used_chars)
                if room > 0:
                    trimmed = _truncate_text_tail(content, room)
                    kept_rev.append(_clone_message(msg, trimmed))
                    used_chars += len(trimmed)
            break
        kept_rev.append(msg)
        used_chars += len(content)
    kept_rev.reverse()
    rebuilt = [*pinned_summaries, *kept_rev]

    agent.clear_history()
    for msg in rebuilt:
        agent.add_message(msg)
    logger.info(
        "Hard-capped session history: kept_msgs=%s/%s kept_chars=%s/%s max_msgs=%s max_chars=%s",
        len(rebuilt),
        len(history),
        sum(len(getattr(m, "content", "") or "") for m in rebuilt),
        total_chars,
        max_messages if max_messages > 0 else "unset",
        max_chars if max_chars > 0 else "unset",
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


def _maybe_compress_history(agent: Any, llm: HelloAgentsLLM) -> None:
    """
    When history is close to context limit, summarize older messages into a compact
    memory block to keep continuity without exceeding max_seq_len.
    """
    enabled = os.getenv("LLM_CONTEXT_COMPRESS_ENABLED", "false").lower() in ("1", "true", "yes")
    if not enabled:
        return
    if not hasattr(agent, "get_history") or not hasattr(agent, "clear_history") or not hasattr(agent, "add_message"):
        return

    max_context = _parse_int_env("LLM_MAX_CONTEXT_TOKENS", default=0)
    max_output = _parse_int_env("LLM_MAX_TOKENS", default=0)
    safety_margin = _parse_int_env("LLM_INPUT_SAFETY_MARGIN", default=4000)
    chars_per_token = _parse_float_env("LLM_CHARS_PER_TOKEN", default=0.5)
    threshold_ratio = _parse_float_env("LLM_CONTEXT_COMPRESS_THRESHOLD", default=0.85)
    target_ratio = _parse_float_env("LLM_CONTEXT_COMPRESS_TARGET", default=0.5)
    keep_last = _parse_int_env("LLM_CONTEXT_COMPRESS_KEEP_LAST", default=4)
    max_summary_tokens = _parse_int_env("LLM_CONTEXT_COMPRESS_MAX_TOKENS", default=600)
    max_summary_chars = _parse_int_env("LLM_CONTEXT_COMPRESS_INPUT_CHARS", default=12000)

    if max_context <= 0:
        return

    budget_tokens = max_context - max_output - safety_margin
    if budget_tokens <= 0:
        return
    budget_chars = int(budget_tokens * chars_per_token)

    history = agent.get_history()
    if not history:
        return

    sanitized, deferred, forgotten = _apply_memory_policy(history)
    _log_memory_policy_stats(sanitized, deferred, forgotten, context="compress")
    if deferred:
        _push_cold_memory(agent, deferred)
    if forgotten:
        logger.info("Forgot %s memory items by policy.", len(forgotten))

    # Only compress normal + compress messages; pinned remain verbatim.
    pinned = [msg for msg, action in sanitized if action == "pin"]
    normal_msgs = [msg for msg, action in sanitized if action in ("normal", "compress")]

    total_chars = sum(len(getattr(m, "content", "") or "") for m in normal_msgs) + sum(
        len(getattr(m, "content", "") or "") for m in pinned
    )
    history_tokens_est = _estimate_tokens(total_chars, chars_per_token)
    budget_tokens_est = _estimate_tokens(budget_chars, chars_per_token)
    trigger_tokens_est = int(budget_tokens_est * threshold_ratio)
    logger.debug(
        "History size: msgs=%s chars=%s est_tokens=%s budget_chars=%s est_budget_tokens=%s trigger_tokens=%s",
        len(history),
        total_chars,
        history_tokens_est,
        budget_chars,
        budget_tokens_est,
        trigger_tokens_est,
    )
    if history_tokens_est <= trigger_tokens_est and not any(
        action == "compress" for _, action in sanitized
    ):
        return

    # Layered memory: keep (1) long-term summary, (2) recent summary, (3) last N messages.
    keep_last = max(0, keep_last)
    summary_tags = ("[LONG_TERM_SUMMARY]", "[RECENT_SUMMARY]")
    long_summary, recent_summary, filtered_msgs = _extract_layered_memory(normal_msgs, summary_tags)
    # Keep pinned messages separate; they always go into context as-is.

    # Compose transcript from normal messages only (exclude existing summaries).
    head = filtered_msgs[:-keep_last] if keep_last else filtered_msgs
    tail = filtered_msgs[-keep_last:] if keep_last else []
    if not head:
        return

    transcript = _build_transcript(head, max_summary_chars)

    summary_prompt = (
        "You are compressing a conversation for a writing assistant.\n"
        "Summarize the key facts, constraints, decisions, and user preferences.\n"
        "Keep it concise, bullet points preferred. Do not add new facts.\n\n"
        f"Conversation:\n{transcript}\n\nSummary:"
    )

    compressor = SimpleAgent(
        name="Context Compressor",
        llm=llm,
        system_prompt="You summarize conversations without adding new information.",
    )
    try:
        recent_summary_text = compressor.run(summary_prompt, max_tokens=max_summary_tokens)
    except Exception as exc:
        logger.warning("Context compression failed: %s", exc)
        return

    target_chars = int(budget_chars * target_ratio)
    if len(recent_summary_text) > target_chars:
        recent_summary_text = _truncate_text_tail(recent_summary_text, target_chars)

    # Merge into long-term summary if we already have one and recent summary grows too big.
    merge_threshold = _parse_float_env("LLM_CONTEXT_COMPRESS_MERGE_THRESHOLD", default=0.6)
    merge_target = _parse_float_env("LLM_CONTEXT_COMPRESS_MERGE_TARGET", default=0.35)
    layered_long_ratio = _parse_float_env("LLM_LAYERED_LONG_RATIO", default=0.35)
    layered_recent_ratio = _parse_float_env("LLM_LAYERED_RECENT_RATIO", default=0.25)
    layered_tail_ratio = _parse_float_env("LLM_LAYERED_TAIL_RATIO", default=0.4)
    long_summary_text = long_summary or ""
    if long_summary_text and recent_summary_text:
        combined_len = len(long_summary_text) + len(recent_summary_text)
        if combined_len > int(budget_chars * merge_threshold):
            merge_prompt = (
                "You are consolidating two memory summaries for a writing assistant.\n"
                "Merge them into one compact summary. Keep key constraints and decisions.\n"
                "Do not add new facts.\n\n"
                f"Summary A:\n{long_summary_text}\n\n"
                f"Summary B:\n{recent_summary_text}\n\nMerged Summary:"
            )
            try:
                long_summary_text = compressor.run(merge_prompt, max_tokens=max_summary_tokens)
            except Exception as exc:
                logger.warning("Summary merge failed: %s", exc)
            if len(long_summary_text) > int(budget_chars * merge_target):
                long_summary_text = _truncate_text_tail(long_summary_text, int(budget_chars * merge_target))
            recent_summary_text = ""

    long_cap = int(budget_chars * max(0.05, min(layered_long_ratio, 0.9)))
    recent_cap = int(budget_chars * max(0.05, min(layered_recent_ratio, 0.9)))
    tail_cap = int(budget_chars * max(0.05, min(layered_tail_ratio, 0.9)))
    if long_summary_text and len(long_summary_text) > long_cap:
        long_summary_text = _truncate_text_tail(long_summary_text, long_cap)
    if recent_summary_text and len(recent_summary_text) > recent_cap:
        recent_summary_text = _truncate_text_tail(recent_summary_text, recent_cap)
    tail = _truncate_tail_messages(tail, tail_cap)

    # Rebuild history with layered summaries + tail messages.
    agent.clear_history()
    for msg in pinned:
        agent.add_message(msg)
    if long_summary_text:
        agent.add_message(Message(f"[LONG_TERM_SUMMARY]\n{long_summary_text}", "assistant"))
    if recent_summary_text:
        agent.add_message(Message(f"[RECENT_SUMMARY]\n{recent_summary_text}", "assistant"))
    for msg in tail:
        agent.add_message(msg)

    logger.info(
        "Layered compression: total=%s chars (est_tokens=%s). long=%s/%s recent=%s/%s kept_tail=%s msgs (tail_cap=%s).",
        total_chars,
        history_tokens_est,
        len(long_summary_text),
        long_cap,
        len(recent_summary_text),
        recent_cap,
        len(tail),
        tail_cap,
    )


def _cooldown_after_call() -> None:
    cooldown = _parse_float_env("LLM_COOLDOWN_SECONDS", default=0.0)
    if cooldown > 0:
        time.sleep(cooldown)


def _extract_layered_memory(history: list[Any], summary_tags: tuple[str, str]) -> tuple[str | None, str | None, list[Any]]:
    long_summary = None
    recent_summary = None
    normal_msgs: list[Any] = []
    for idx, msg in enumerate(history):
        content = getattr(msg, "content", "") or ""
        if content.startswith(summary_tags[0]):
            long_summary = content[len(summary_tags[0]):].lstrip()
            continue
        if content.startswith(summary_tags[1]):
            recent_summary = content[len(summary_tags[1]):].lstrip()
            continue
        normal_msgs.append(msg)
    return long_summary, recent_summary, normal_msgs


def _build_transcript(messages: list[Any], max_chars: int) -> str:
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", "assistant")
        content = getattr(msg, "content", "") or ""
        lines.append(f"[{role}] {content}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = _truncate_text_tail(transcript, max_chars)
    return transcript


def _estimate_tokens(chars: int, chars_per_token: float) -> int:
    if chars <= 0:
        return 0
    if chars_per_token <= 0:
        chars_per_token = 0.5
    return int(chars / chars_per_token)


def _apply_memory_policy(history: list[Any]) -> tuple[list[tuple[Any, str]], list[Any], list[Any]]:
    """
    Classify history items into memory buckets and return:
    - sanitized list of (message, action)
    - deferred messages (cold store)
    - forgotten messages (drop)
    """
    tag_pin = os.getenv("LLM_MEMORY_TAG_PIN", "[MEM:PIN]")
    tag_compress = os.getenv("LLM_MEMORY_TAG_COMPRESS", "[MEM:COMPRESS]")
    tag_defer = os.getenv("LLM_MEMORY_TAG_DEFER", "[MEM:DEFER]")
    tag_forget = os.getenv("LLM_MEMORY_TAG_FORGET", "[MEM:FORGET]")
    prefer_meta = os.getenv("LLM_MEMORY_META_PRIORITY", "true").lower() in ("1", "true", "yes")

    sanitized: list[tuple[Any, str]] = []
    deferred: list[Any] = []
    forgotten: list[Any] = []

    for idx, msg in enumerate(history):
        content = getattr(msg, "content", "") or ""
        meta = getattr(msg, "metadata", None) or {}
        action = meta.get("memory") if prefer_meta else None
        action = (action or "").lower()

        if not action:
            if content.startswith(tag_pin):
                action = "pin"
                content = content[len(tag_pin):].lstrip()
            elif content.startswith(tag_compress):
                action = "compress"
                content = content[len(tag_compress):].lstrip()
            elif content.startswith(tag_defer):
                action = "defer"
                content = content[len(tag_defer):].lstrip()
            elif content.startswith(tag_forget):
                action = "forget"
                content = content[len(tag_forget):].lstrip()
            else:
                action = "normal"

        if action == "normal":
            action = _auto_memory_action(msg, idx=idx, total=max(1, len(history)))

        if action == "forget":
            forgotten.append(msg)
            continue
        if action == "defer":
            deferred.append(msg)
            continue

        if content != getattr(msg, "content", ""):
            msg = _clone_message(msg, content)

        if action not in ("pin", "compress", "normal"):
            action = "normal"

        sanitized.append((msg, action))

    return sanitized, deferred, forgotten


def _truncate_tail_messages(messages: list[Any], max_chars: int) -> list[Any]:
    if max_chars <= 0 or not messages:
        return []
    kept: list[Any] = []
    total = 0
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        size = len(content)
        if kept and total + size > max_chars:
            break
        if not kept and size > max_chars:
            trimmed = _truncate_text_tail(content, max_chars)
            kept.append(_clone_message(msg, trimmed))
            total = len(trimmed)
            break
        kept.append(msg)
        total += size
    kept.reverse()
    return kept


def _log_memory_policy_stats(
    sanitized: list[tuple[Any, str]],
    deferred: list[Any],
    forgotten: list[Any],
    *,
    context: str,
) -> None:
    verbose = os.getenv("LLM_MEMORY_LOG_VERBOSE", "true").lower() in ("1", "true", "yes")
    if not verbose:
        return
    counts = {"pin": 0, "compress": 0, "normal": 0}
    for _, action in sanitized:
        counts[action] = counts.get(action, 0) + 1
    total = len(sanitized) + len(deferred) + len(forgotten)
    logger.info(
        "Memory policy (%s): total=%s pin=%s compress=%s normal=%s defer=%s forget=%s",
        context,
        total,
        counts.get("pin", 0),
        counts.get("compress", 0),
        counts.get("normal", 0),
        len(deferred),
        len(forgotten),
    )


def _auto_memory_action(msg: Any, *, idx: int, total: int) -> str:
    enabled = os.getenv("LLM_MEMORY_AUTO_POLICY_ENABLED", "true").lower() in ("1", "true", "yes")
    if not enabled:
        return "normal"

    content = (getattr(msg, "content", "") or "").strip()
    if not content:
        return "forget"

    min_chars = max(0, _parse_int_env("LLM_MEMORY_MIN_CHARS", default=16))
    if len(content) < min_chars:
        return "forget"

    score = _memory_score(msg, idx=idx, total=total)
    pin_threshold = _parse_float_env("LLM_MEMORY_PIN_THRESHOLD", default=0.82)
    compress_threshold = _parse_float_env("LLM_MEMORY_COMPRESS_THRESHOLD", default=0.56)
    defer_threshold = _parse_float_env("LLM_MEMORY_DEFER_THRESHOLD", default=0.3)

    if score >= pin_threshold:
        return "pin"
    if score >= compress_threshold:
        return "compress"
    if score >= defer_threshold:
        return "defer"
    return "forget"


def _memory_score(msg: Any, *, idx: int, total: int) -> float:
    role = (getattr(msg, "role", "") or "").lower()
    content = (getattr(msg, "content", "") or "")
    lc = content.lower()

    role_score_map = {
        "system": 1.0,
        "user": 0.95,
        "tool": 0.8,
        "assistant": 0.65,
    }
    role_score = role_score_map.get(role, 0.5)

    recency = (idx + 1) / max(1, total)
    length_score = min(1.0, len(content) / max(1.0, _parse_float_env("LLM_MEMORY_LENGTH_BASE", default=1200.0)))

    keywords_raw = os.getenv(
        "LLM_MEMORY_IMPORTANT_KEYWORDS",
        "constraint,requirements,must,style,format,citation,source,topic,goal,不要,必须,约束,风格,引用,目标",
    )
    keywords = [item.strip().lower() for item in keywords_raw.split(",") if item.strip()]
    hits = sum(1 for kw in keywords if kw in lc)
    keyword_score = min(1.0, hits / max(1.0, _parse_float_env("LLM_MEMORY_KEYWORD_HITS_BASE", default=3.0)))

    recency_w = _parse_float_env("LLM_MEMORY_RECENCY_WEIGHT", default=0.4)
    role_w = _parse_float_env("LLM_MEMORY_ROLE_WEIGHT", default=0.2)
    length_w = _parse_float_env("LLM_MEMORY_LENGTH_WEIGHT", default=0.1)
    keyword_w = _parse_float_env("LLM_MEMORY_KEYWORD_WEIGHT", default=0.3)
    weighted = (recency * recency_w) + (role_score * role_w) + (length_score * length_w) + (keyword_score * keyword_w)
    return _clamp(weighted, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _push_cold_memory(agent: Any, messages: list[Any]) -> None:
    """Store deferred memories on the agent instance (not included in context)."""
    if not messages:
        return
    cold = getattr(agent, "_cold_memory", None)
    if cold is None:
        cold = []
        setattr(agent, "_cold_memory", cold)
    cold.extend(messages)
    _persist_cold_memory(agent, messages)


def _persist_cold_memory(agent: Any, messages: list[Any]) -> None:
    enabled = os.getenv("LLM_COLD_STORE_ENABLED", "true").lower() in ("1", "true", "yes")
    if not enabled:
        return
    db_path = os.getenv("LLM_COLD_STORE_PATH") or os.getenv("STORAGE_PATH") or ""
    if not db_path:
        return
    max_chars = _parse_int_env("LLM_COLD_STORE_MAX_CHARS", default=5000)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cold_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cols = conn.execute("PRAGMA table_info(cold_memory)").fetchall()
        col_names = {str(row[1]).lower() for row in cols}
        if "session_id" not in col_names:
            conn.execute("ALTER TABLE cold_memory ADD COLUMN session_id TEXT")
        now = datetime.utcnow().isoformat()
        cur = conn.cursor()
        agent_name = getattr(agent, "name", "unknown")
        session_id = getattr(agent, "_session_id", "__default__")
        for msg in messages:
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "") or ""
            if max_chars > 0 and len(content) > max_chars:
                content = _truncate_text_tail(content, max_chars)
            metadata = getattr(msg, "metadata", None)
            if metadata is not None:
                try:
                    metadata = json.dumps(metadata, ensure_ascii=False, default=str)
                except Exception:
                    metadata = None
            cur.execute(
                """
                INSERT INTO cold_memory (agent_name, session_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (agent_name, session_id, role, content, metadata, now),
            )
        conn.commit()
        conn.close()
        logger.info("Persisted %s cold memory items to SQLite.", len(messages))
    except Exception as exc:
        logger.warning("Failed to persist cold memory: %s", exc)


def _delete_cold_memory_rows(*, agent_name: str, session_id: str) -> None:
    db_path = os.getenv("LLM_COLD_STORE_PATH") or os.getenv("STORAGE_PATH") or ""
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            "DELETE FROM cold_memory WHERE agent_name = ? AND session_id = ?",
            (agent_name, session_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to clear cold memory rows: %s", exc)


def _augment_with_cold_recall(agent: Any, input_text: str) -> str:
    if not input_text:
        return input_text
    enabled = os.getenv("LLM_COLD_RECALL_ENABLED", "true").lower() in ("1", "true", "yes")
    if not enabled:
        return input_text
    recalled = _recall_cold_memory(agent, input_text)
    if not recalled:
        return input_text
    block = "\n\n[Recovered memory]\n" + "\n".join(f"- {item}" for item in recalled)
    merged = input_text + block
    logger.info("Injected cold memory into prompt: %s snippets", len(recalled))
    return merged


def _recall_cold_memory(agent: Any, query: str) -> list[str]:
    db_path = os.getenv("LLM_COLD_STORE_PATH") or os.getenv("STORAGE_PATH") or ""
    if not db_path:
        return []
    top_k = max(1, _parse_int_env("LLM_COLD_RECALL_TOP_K", default=3))
    max_chars = max(200, _parse_int_env("LLM_COLD_RECALL_MAX_CHARS", default=300))
    lookback = max(20, _parse_int_env("LLM_COLD_RECALL_LOOKBACK", default=200))
    agent_name = getattr(agent, "name", "unknown")
    session_id = getattr(agent, "_session_id", "__default__")
    query_terms = tokenize(query, lowercase=True)
    if not query_terms:
        return []

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT content FROM cold_memory
            WHERE agent_name = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (agent_name, session_id, lookback),
        ).fetchall()
        conn.close()
    except Exception:
        return []

    scored: list[tuple[float, str]] = []
    for row in rows:
        content = (row["content"] or "").strip()
        if not content:
            continue
        terms = tokenize(content, lowercase=True)
        if not terms:
            continue
        overlap = len(query_terms & terms)
        if overlap <= 0:
            continue
        score = overlap / max(1, len(query_terms))
        scored.append((score, content))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [text[:max_chars] for _, text in scored[:top_k]]
    return selected
