from __future__ import annotations

from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a skilled editor. "
    "Rewrite and polish text while preserving meaning. "
    "Improve clarity, coherence, and tone. "
    "Avoid adding unsupported facts."
)


class EditorAgent(BaseWritingAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        name: str = "Editor Agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        config = AgentRuntimeConfig(
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )
        super().__init__(config, tool_registry=tool_registry)

    def rewrite(
        self,
        *,
        draft: str,
        guidance: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        parts = [f"Draft:\n{draft}"]
        if guidance:
            parts.append(f"Editing guidance:\n{guidance}")
        if style:
            parts.append(f"Style:\n{style}")
        if target_length:
            parts.append(f"Target length:\n{target_length}")

        prompt = (
            "Rewrite the draft based on the guidance below. "
            "Keep structure when possible.\n\n"
            + "\n\n".join(parts)
        )
        return self.run(prompt, max_tokens=max_tokens)
