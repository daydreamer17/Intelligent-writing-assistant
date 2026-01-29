from __future__ import annotations

from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful reviewer. "
    "Identify logical issues, unclear claims, and missing evidence. "
    "Return actionable, prioritized feedback. "
    "Do not rewrite the draft unless asked."
)


class ReviewerAgent(BaseWritingAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        name: str = "Reviewer Agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
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

    def review(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
    ) -> str:
        parts = [f"Draft:\n{draft}"]
        if criteria:
            parts.append(f"Review criteria:\n{criteria}")
        if sources:
            parts.append(f"Available sources:\n{sources}")
        if audience:
            parts.append(f"Audience:\n{audience}")

        prompt = (
            "Review the draft and provide feedback grouped by severity. "
            "Highlight factual claims that need citations.\n\n"
            + "\n\n".join(parts)
        )
        return self.run(prompt)
