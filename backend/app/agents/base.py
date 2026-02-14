from __future__ import annotations

import os
import json
import sqlite3
import re
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
_TOOL_REGISTRY_UNSET = object()
_TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:[^\]]+\]")
_TOOL_CALL_PREFIX = "[TOOL_CALL:"

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
        llm = HelloAgentsLLM(**llm_kwargs)
    except TypeError:
        if "timeout" in llm_kwargs:
            llm_kwargs.pop("timeout", None)
            llm = HelloAgentsLLM(**llm_kwargs)
        else:
            raise
    _patch_llm_think(llm)
    return llm


def _patch_llm_think(llm: HelloAgentsLLM) -> None:
    """Replace framework ``think()`` so raw ``[TOOL_CALL:]`` markers are never
    printed to stdout.

    The upstream ``HelloAgentsLLM.think()`` does ``print(content, end="",
    flush=True)`` for every streamed chunk.  When a small model emits hundreds
    of ``[TOOL_CALL:]`` markers the console becomes unreadable and it looks as
    if the pipeline is broken (even though our agent-level guards DO filter the
    markers before they reach pipeline output).

    This patch:
    * Redirects the informational prints to the logger.
    * Strips ``[TOOL_CALL:…]`` markers from chunks before printing so the
      console only shows real content.
    * Still **yields** every raw chunk unchanged so that the downstream
      ``ToolAwareSimpleAgent.stream_run()`` can parse tool calls normally.
    """
    import types
    from hello_agents.core.exceptions import HelloAgentsException

    original_client_attr = "_client"  # OpenAI client attribute

    def _patched_think(
        self: Any,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        logger.info("Calling %s …", self.model)
        visible_buffer = ""

        def _drain_visible(*, final_pass: bool = False) -> str:
            """Drop TOOL_CALL markers from a streaming buffer, including split chunks."""
            nonlocal visible_buffer
            parts: list[str] = []

            while True:
                marker_index = visible_buffer.find(_TOOL_CALL_PREFIX)
                if marker_index == -1:
                    safe_len = len(visible_buffer)
                    if not final_pass:
                        safe_len = max(0, safe_len - (len(_TOOL_CALL_PREFIX) - 1))
                    if safe_len <= 0:
                        break
                    parts.append(visible_buffer[:safe_len])
                    visible_buffer = visible_buffer[safe_len:]
                    break

                if marker_index > 0:
                    parts.append(visible_buffer[:marker_index])
                    visible_buffer = visible_buffer[marker_index:]
                    continue

                marker_end = visible_buffer.find("]")
                if marker_end == -1:
                    break
                visible_buffer = visible_buffer[marker_end + 1 :]

            return "".join(parts)

        try:
            response = getattr(self, original_client_attr).chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            logger.info("LLM stream opened (max_tokens=%s).", max_tokens)
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    visible_buffer += content
                    # Print only non-tool-call text so the console stays clean.
                    visible = _drain_visible(final_pass=False)
                    if visible.strip():
                        print(visible, end="", flush=True)
                    # Always yield the raw content so the agent layer can parse
                    # tool calls from it.
                    yield content
            tail_visible = _drain_visible(final_pass=True)
            if tail_visible.strip():
                print(tail_visible, end="", flush=True)
            print()  # newline after stream ends
        except Exception as e:
            logger.error("LLM API error: %s", e)
            raise HelloAgentsException(f"LLM调用失败: {e}")

    def _patched_stream_invoke(
        self: Any,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        # Upstream stream_invoke only forwards temperature; we preserve all kwargs
        # so stage-level max_tokens/max_input policies actually take effect.
        yield from self.think(messages, **kwargs)

    llm.think = types.MethodType(_patched_think, llm)


def _resolve_conversation_memory_enabled(explicit: Any | None = None) -> tuple[bool, str]:
    if explicit is not None:
        return bool(explicit), "explicit"
    try:
        from ..services.generation_mode import (
            conversation_memory_enabled_for_mode,
            get_generation_mode,
        )

        mode = get_generation_mode()
        return conversation_memory_enabled_for_mode(mode), f"mode:{mode}"
    except Exception:
        return True, "fallback"
    llm.stream_invoke = types.MethodType(_patched_stream_invoke, llm)
    logger.debug("Patched LLM.think() to suppress [TOOL_CALL:] console noise.")


def _patch_tool_call_limit(agent: Any) -> None:
    """Monkey-patch ToolAwareSimpleAgent to limit tool calls per LLM response.

    Qwen3-8B (and similar small models) may generate hundreds of identical
    [TOOL_CALL:...] markers in a single response instead of producing one or two
    calls and waiting for results.  Without a guard, the framework will execute
    every single call, accumulate massive tool-result text, and blow up the
    context window on the next LLM round-trip.

    This patch wraps ``_parse_tool_calls`` so that:
    1. Duplicate calls (same tool_name + parameters) are collapsed.
    2. At most ``LLM_MAX_TOOL_CALLS_PER_RESPONSE`` unique calls are kept.
    """
    max_calls = max(1, _parse_int_env("LLM_MAX_TOOL_CALLS_PER_RESPONSE", default=2))
    max_calls_per_request = max(max_calls, _parse_int_env("LLM_MAX_TOOL_CALLS_PER_REQUEST", default=4))
    max_repeated_calls = max(1, _parse_int_env("LLM_MAX_REPEATED_TOOL_CALLS", default=2))

    if not hasattr(agent, "_parse_tool_calls"):
        return

    # Capture the *bound* method before we overwrite the instance attribute,
    # so the closure calls the original implementation correctly.
    original_parse = agent._parse_tool_calls

    def _call_key(tool_name: str, parameters: str) -> str:
        normalized_name = (tool_name or "").strip().lower()
        normalized_parameters = re.sub(r"\s+", " ", (parameters or "").strip()).lower()
        return f"{normalized_name}::{normalized_parameters}"

    def _limited_parse(text: str) -> list:
        calls = original_parse(text)
        if not calls:
            return calls
        # Deduplicate by (tool_name, parameters)
        seen: set[tuple[str, str]] = set()
        unique: list = []
        for call in calls:
            key = (call.get("tool_name", ""), call.get("parameters", ""))
            if key not in seen:
                seen.add(key)
                unique.append(call)
        if len(unique) > max_calls:
            logger.warning(
                "Truncated tool calls from %s unique (%s raw) to %s per response "
                "(LLM_MAX_TOOL_CALLS_PER_RESPONSE=%s).",
                len(unique),
                len(calls),
                max_calls,
                max_calls,
            )
            unique = unique[:max_calls]
        elif len(calls) > len(unique):
            logger.info(
                "De-duplicated tool calls: %s raw -> %s unique.",
                len(calls),
                len(unique),
            )
        unique = unique[:max_calls]

        state = getattr(agent, "_tool_guard_state", None)
        if not isinstance(state, dict):
            return unique

        repeat_counts = state.setdefault("seen_counts", {})
        planned_total = int(state.get("planned_total", 0))
        filtered: list[dict[str, Any]] = []
        suppressed_repeat = 0
        suppressed_budget = 0

        for call in unique:
            key = _call_key(call.get("tool_name", ""), call.get("parameters", ""))
            next_repeat = int(repeat_counts.get(key, 0)) + 1
            repeat_counts[key] = next_repeat

            if next_repeat > max_repeated_calls:
                suppressed_repeat += 1
                continue
            if planned_total >= max_calls_per_request:
                suppressed_budget += 1
                state["loop_detected"] = True
                continue

            filtered.append(call)
            planned_total += 1

        state["planned_total"] = planned_total

        if suppressed_repeat > 0:
            state["loop_detected"] = True
            logger.warning(
                "Suppressed %s repeated tool call(s) in one request "
                "(LLM_MAX_REPEATED_TOOL_CALLS=%s).",
                suppressed_repeat,
                max_repeated_calls,
            )
        if suppressed_budget > 0:
            logger.warning(
                "Suppressed %s tool call(s) due to request-level budget "
                "(LLM_MAX_TOOL_CALLS_PER_REQUEST=%s).",
                suppressed_budget,
                max_calls_per_request,
            )
        return filtered

    max_iterations = max(1, _parse_int_env("LLM_MAX_TOOL_ITERATIONS", default=2))

    # Bind as instance method so ``self._parse_tool_calls(text)`` hits our wrapper
    import types
    agent._parse_tool_calls = types.MethodType(lambda self, text: _limited_parse(text), agent)

    # ── Truncate tool results to prevent token overflow ──────────────────
    # GitHub search results can be enormous (hundreds of KB).  When they are
    # fed back into the messages for the next LLM iteration the context
    # explodes (483K tokens observed).  Truncating each individual tool
    # result is the most targeted way to prevent this.
    max_result_chars = max(500, _parse_int_env("LLM_MAX_TOOL_RESULT_CHARS", default=3000))

    if hasattr(agent, "_execute_tool_call"):
        original_execute = agent._execute_tool_call

        def _limited_execute(tool_name: str, parameters: str) -> str:
            state = getattr(agent, "_tool_guard_state", None)
            if isinstance(state, dict):
                key = _call_key(tool_name, parameters)
                executed_total = int(state.get("executed_total", 0))
                if executed_total >= max_calls_per_request:
                    state["loop_detected"] = True
                    logger.warning(
                        "Skip tool %s: request-level tool budget reached (%s).",
                        tool_name,
                        max_calls_per_request,
                    )
                    return "⚠️ 本轮工具调用次数已达上限，请直接输出最终答案，不要继续调用工具。"

                executed_counts = state.setdefault("executed_counts", {})
                repeat = int(executed_counts.get(key, 0)) + 1
                if repeat > max_repeated_calls:
                    state["loop_detected"] = True
                    logger.warning(
                        "Skip repeated tool %s call: repeat=%s limit=%s.",
                        tool_name,
                        repeat,
                        max_repeated_calls,
                    )
                    return f"⚠️ 工具 {tool_name} 出现重复调用，请直接输出最终答案，不要继续调用工具。"

                executed_counts[key] = repeat
                state["executed_total"] = executed_total + 1

            result = original_execute(tool_name, parameters)
            if result and len(result) > max_result_chars:
                logger.warning(
                    "Truncated tool result for %s from %s to %s chars "
                    "(LLM_MAX_TOOL_RESULT_CHARS=%s).",
                    tool_name, len(result), max_result_chars, max_result_chars,
                )
                result = result[:max_result_chars] + "\n…[结果已截断]"
            return result

        agent._execute_tool_call = types.MethodType(
            lambda self, tool_name, parameters: _limited_execute(tool_name, parameters),
            agent,
        )

    # ── Guard run / stream_run from endless tool-call loops ──────────────
    if hasattr(agent, "run"):
        original_run = agent.run

        def _guarded_run(input_text: str, *args: Any, **kwargs: Any) -> str:
            setattr(
                agent,
                "_tool_guard_state",
                {
                    "seen_counts": {},
                    "planned_total": 0,
                    "executed_counts": {},
                    "executed_total": 0,
                    "loop_detected": False,
                },
            )
            kwargs.setdefault("max_tool_iterations", max_iterations)
            response = original_run(input_text, *args, **kwargs)
            cleaned = _strip_tool_markers(response, trim=True)
            state = getattr(agent, "_tool_guard_state", None)
            loop_detected = bool(isinstance(state, dict) and state.get("loop_detected"))
            if cleaned.strip():
                if loop_detected and len(cleaned) < 80:
                    logger.warning(
                        "Likely tool loop detected with short final response (run). Forcing no-tool fallback."
                    )
                    fallback = _force_answer_without_tools(agent, input_text, kwargs)
                    if fallback:
                        return fallback
                if cleaned != response:
                    logger.info("Stripped tool-call markers from final response (run).")
                return cleaned
            if "[TOOL_CALL:" in (response or ""):
                logger.warning(
                    "Tool-call-only response detected (run). Forcing final answer without tools."
                )
                return _force_answer_without_tools(agent, input_text, kwargs)
            return response

        agent.run = types.MethodType(lambda self, input_text, *args, **kwargs: _guarded_run(input_text, *args, **kwargs), agent)

    if hasattr(agent, "stream_run"):
        original_stream_run = agent.stream_run

        def _guarded_stream_run(input_text: str, *args: Any, **kwargs: Any) -> Iterator[str]:
            setattr(
                agent,
                "_tool_guard_state",
                {
                    "seen_counts": {},
                    "planned_total": 0,
                    "executed_counts": {},
                    "executed_total": 0,
                    "loop_detected": False,
                },
            )
            kwargs.setdefault("max_tool_iterations", max_iterations)
            emitted_any = False
            for chunk in original_stream_run(input_text, *args, **kwargs):
                cleaned = _strip_tool_markers(chunk, trim=False)
                if not cleaned or not cleaned.strip():
                    continue
                emitted_any = True
                yield cleaned
            if not emitted_any:
                logger.warning(
                    "Stream run emitted no content after tool calls. Forcing final answer without tools."
                )
                fallback = _force_answer_without_tools(agent, input_text, kwargs)
                if fallback:
                    yield fallback

        agent.stream_run = types.MethodType(
            lambda self, input_text, *args, **kwargs: _guarded_stream_run(input_text, *args, **kwargs),
            agent,
        )


def _strip_tool_markers(text: str, *, trim: bool = False) -> str:
    if not text:
        return text
    # Remove [TOOL_CALL:...] directives that the model may leave in final output.
    cleaned = _TOOL_CALL_PATTERN.sub("", text)
    return cleaned.strip() if trim else cleaned


def _force_answer_without_tools(agent: Any, input_text: str, kwargs: dict[str, Any]) -> str:
    llm = getattr(agent, "llm", None)
    if llm is None:
        return ""

    system_prompt = getattr(agent, "system_prompt", "") or ""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n当前回合禁止任何工具调用。请直接给出最终答案，不要输出任何 [TOOL_CALL:...] 标记。"
            ),
        }
    ]

    history = getattr(agent, "_history", None) or []
    # Keep only a small recent window to avoid prompt explosion in fallback.
    for msg in history[-6:]:
        role = getattr(msg, "role", "assistant")
        content = getattr(msg, "content", "") or ""
        if content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                "请基于已获得的信息直接回答，不要再调用工具。\n\n"
                f"原始请求：\n{input_text}"
            ),
        }
    )

    invoke_kwargs: dict[str, Any] = {}
    for key in ("max_tokens", "temperature"):
        value = kwargs.get(key)
        if value is not None:
            invoke_kwargs[key] = value

    try:
        response = llm.invoke(messages, **invoke_kwargs)
    except Exception as exc:
        logger.warning("Forced no-tool fallback failed: %s", exc)
        return ""
    return _strip_tool_markers(response, trim=True)


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

    def _create_agent(self, tool_registry_override: Any = _TOOL_REGISTRY_UNSET) -> Any:
        if tool_registry_override is _TOOL_REGISTRY_UNSET:
            registry = self._tool_registry
        else:
            registry = tool_registry_override
        if registry is not None:
            agent = ToolAwareSimpleAgent(
                name=self.config.name,
                llm=self.llm,
                system_prompt=self.config.system_prompt,
                enable_tool_calling=True,
                tool_registry=registry,
            )
            _patch_tool_call_limit(agent)
            return agent
        return SimpleAgent(
            name=self.config.name,
            llm=self.llm,
            system_prompt=self.config.system_prompt,
        )

    def _session_key(self, session_id: Optional[str], tool_profile_id: Optional[str] = None) -> str:
        mode = (os.getenv("CONVERSATION_MEMORY_MODE", "session") or "session").strip().lower()
        if mode == "global":
            base = "__default__"
        else:
            raw = (session_id or "").strip()
            if not raw:
                base = "__default__"
            else:
                base = raw[:64]
        profile = (tool_profile_id or "").strip()
        if profile:
            return f"{base}::{profile[:64]}"
        return base

    def _matches_session_scope(self, key: str, session_id: Optional[str]) -> bool:
        base = self._session_key(session_id)
        return key == base or key.startswith(base + "::")

    def _get_session_agent(
        self,
        session_id: Optional[str],
        *,
        tool_profile_id: Optional[str] = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> tuple[str, Any]:
        key = self._session_key(session_id, tool_profile_id)
        agent = self._session_agents.get(key)
        if agent is None:
            if tool_profile_id is not None:
                selected_registry = tool_registry_override
            else:
                selected_registry = _TOOL_REGISTRY_UNSET
            agent = self._create_agent(tool_registry_override=selected_registry)
            self._session_agents[key] = agent
            logger.info(
                "Created isolated LLM session: agent=%s session_id=%s tool_profile=%s",
                self.config.name,
                key,
                tool_profile_id or "default",
            )
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
        with self._lock:
            base_session_key = self._session_key(session_id)
            target_keys = [
                key
                for key in self._session_agents.keys()
                if self._matches_session_scope(key, session_id)
            ]

            cleared = False
            cold_deleted_rows = 0
            for key in target_keys:
                agent = self._session_agents.get(key)
                if agent is None:
                    continue
                if hasattr(agent, "clear_history"):
                    try:
                        agent.clear_history()
                        cleared = True
                    except Exception:
                        pass

                if clear_cold:
                    if hasattr(agent, "_cold_memory"):
                        try:
                            setattr(agent, "_cold_memory", [])
                            cleared = True
                        except Exception:
                            pass
                    # Always clear SQLite rows for this concrete key.
                    deleted_before = cold_deleted_rows
                    cold_deleted_rows += _delete_cold_memory_scope(
                        agent_name=getattr(agent, "name", self.config.name),
                        session_base=key,
                    )
                    if cold_deleted_rows > deleted_before:
                        cleared = True

                if drop_agent and key != "__default__":
                    self._session_agents.pop(key, None)
                    self._session_last_seen.pop(key, None)
                    cleared = True
                    logger.info("Dropped LLM session: agent=%s session_id=%s", self.config.name, key)

            # Even if no in-memory session exists, still clear cold-memory rows for this session scope.
            if clear_cold:
                deleted_before = cold_deleted_rows
                cold_deleted_rows += _delete_cold_memory_scope(
                    agent_name=self.config.name,
                    session_base=base_session_key,
                )
                if cold_deleted_rows > deleted_before:
                    cleared = True

            if not target_keys and not cleared:
                return False

            if clear_cold:
                logger.info(
                    "Cleared session memory: agent=%s base_session=%s keys=%s cold_rows=%s drop_agent=%s",
                    self.config.name,
                    base_session_key,
                    len(target_keys),
                    cold_deleted_rows,
                    drop_agent,
                )
            return cleared

    def run(self, input_text: str, **kwargs: Any) -> str:
        session_id = kwargs.pop("session_id", None)
        tool_profile_id = kwargs.pop("tool_profile_id", None)
        tool_registry_override = kwargs.pop("tool_registry_override", None)
        conversation_memory_enabled_arg = kwargs.pop("conversation_memory_enabled", None)
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
                session_key, agent = self._get_session_agent(
                    session_id,
                    tool_profile_id=tool_profile_id,
                    tool_registry_override=tool_registry_override,
                )
                setattr(agent, "_session_id", session_key)
                memory_enabled, memory_source = _resolve_conversation_memory_enabled(
                    conversation_memory_enabled_arg
                )
                if not memory_enabled and hasattr(agent, "clear_history"):
                    try:
                        agent.clear_history()
                    except Exception:
                        pass
                prepared_input = input_text
                if memory_enabled:
                    prepared_input = _augment_with_cold_recall(agent, prepared_input)
                prepared_input = _truncate_input(
                    prepared_input,
                    max_chars=max_input_chars,
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                if memory_enabled:
                    _maybe_compress_history(agent, self.llm)
                    _prune_agent_history(
                        agent,
                        upcoming_user_chars=len(prepared_input),
                        max_context_tokens=max_context_tokens,
                        max_output_tokens=kwargs.get("max_tokens"),
                        safety_margin=input_safety_margin,
                    )
                    _hard_cap_agent_history(agent)
                else:
                    logger.info(
                        "Conversation memory disabled for this call: agent=%s session_id=%s source=%s",
                        self.config.name,
                        session_key,
                        memory_source,
                    )
                if hasattr(agent, "enable_tool_calling"):
                    original_tool_calling = getattr(agent, "enable_tool_calling", None)
                tool_calling_temporarily_disabled = _maybe_disable_tool_prompt(
                    agent,
                    upcoming_user_chars=len(prepared_input),
                )
            try:
                try:
                    with self._lock:
                        run_kwargs = dict(kwargs)
                        if hasattr(agent, "enable_tool_calling"):
                            run_kwargs.setdefault(
                                "max_tool_iterations",
                                max(1, _parse_int_env("LLM_MAX_TOOL_ITERATIONS", default=2)),
                            )
                        response = agent.run(prepared_input, **run_kwargs)
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
        tool_profile_id = kwargs.pop("tool_profile_id", None)
        tool_registry_override = kwargs.pop("tool_registry_override", None)
        conversation_memory_enabled_arg = kwargs.pop("conversation_memory_enabled", None)
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
                session_key, agent = self._get_session_agent(
                    session_id,
                    tool_profile_id=tool_profile_id,
                    tool_registry_override=tool_registry_override,
                )
                setattr(agent, "_session_id", session_key)
                memory_enabled, memory_source = _resolve_conversation_memory_enabled(
                    conversation_memory_enabled_arg
                )
                if not memory_enabled and hasattr(agent, "clear_history"):
                    try:
                        agent.clear_history()
                    except Exception:
                        pass
                prepared_input = input_text
                if memory_enabled:
                    prepared_input = _augment_with_cold_recall(agent, prepared_input)
                prepared_input = _truncate_input(
                    prepared_input,
                    max_chars=max_input_chars,
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=kwargs.get("max_tokens"),
                    safety_margin=input_safety_margin,
                )
                if memory_enabled:
                    _maybe_compress_history(agent, self.llm)
                    _prune_agent_history(
                        agent,
                        upcoming_user_chars=len(prepared_input),
                        max_context_tokens=max_context_tokens,
                        max_output_tokens=kwargs.get("max_tokens"),
                        safety_margin=input_safety_margin,
                    )
                    _hard_cap_agent_history(agent)
                else:
                    logger.info(
                        "Conversation memory disabled for this stream: agent=%s session_id=%s source=%s",
                        self.config.name,
                        session_key,
                        memory_source,
                    )
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
                        stream_kwargs = dict(kwargs)
                        if hasattr(agent, "enable_tool_calling"):
                            stream_kwargs.setdefault(
                                "max_tool_iterations",
                                max(1, _parse_int_env("LLM_MAX_TOOL_ITERATIONS", default=2)),
                            )
                        for chunk in agent.stream_run(prepared_input, **stream_kwargs):
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
    # SiliconFlow: "number of input tokens (N) have exceeded max_prompt_tokens (M) limit."
    return (
        ("max_total_tokens" in message and "max_seq_len" in message)
        or ("input tokens" in message and "max_prompt_tokens" in message)
        or "context length" in message
        or "maximum context" in message
        or "prompt is too long" in message
        or ("exceeded" in message and "token" in message and "limit" in message)
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


def _delete_cold_memory_scope(*, agent_name: str, session_base: str) -> int:
    """Delete cold-memory rows for a base session and all its tool-profile variants."""
    db_path = os.getenv("LLM_COLD_STORE_PATH") or os.getenv("STORAGE_PATH") or ""
    if not db_path:
        return 0
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM cold_memory
            WHERE agent_name = ?
              AND (session_id = ? OR session_id LIKE ?)
            """,
            (agent_name, session_base, f"{session_base}::%"),
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()
        conn.close()
        return deleted
    except Exception as exc:
        logger.warning("Failed to clear cold memory scope: %s", exc)
        return 0


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
