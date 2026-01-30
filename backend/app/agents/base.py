from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import logging
import time

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
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        attempt = 0
        while True:
            try:
                response = self.agent.run(input_text, **kwargs)
                logger.info("LLM call success. Output:\n%s", response)
                _cooldown_after_call()
                return response
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_retries:
                    # 指数退避：10s, 20s, 40s, 80s, 160s
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
                logger.error("LLM call failed: %s", exc, exc_info=True)
                raise

    def stream(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        max_retries = _parse_int_env("LLM_RETRY_MAX", default=5)
        backoff_base = _parse_float_env("LLM_RETRY_BACKOFF", default=10.0)
        attempt = 0
        while True:
            emitted = False
            try:
                for chunk in self.agent.stream_run(input_text, **kwargs):
                    if not chunk:
                        continue
                    emitted = True
                    yield chunk
                _cooldown_after_call()
                return
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_retries and not emitted:
                    # 指数退避：10s, 20s, 40s, 80s, 160s (最长2.6分钟)
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


def _cooldown_after_call() -> None:
    cooldown = _parse_float_env("LLM_COOLDOWN_SECONDS", default=0.0)
    if cooldown > 0:
        time.sleep(cooldown)
