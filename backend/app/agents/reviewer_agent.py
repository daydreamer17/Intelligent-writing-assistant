from __future__ import annotations

from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful reviewer. "
    "Identify logical issues, unclear claims, and missing evidence. "
    "Return actionable, prioritized feedback. "
    "Do not rewrite the draft unless asked. "
    "When claims need verification and tools are available, call tools and "
    "use the results to ground your feedback."
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
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> str:
        # 截断过长的内容以避免超过模型限制
        max_draft_chars = 15000  # 约 6000 tokens
        max_sources_chars = 8000  # 约 3200 tokens

        if len(draft) > max_draft_chars:
            draft = draft[:max_draft_chars] + "\n...(内容过长已截断)"

        if len(sources) > max_sources_chars:
            sources = sources[:max_sources_chars] + "\n...(来源过长已截断)"

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
        return self.run(
            prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )

    def review_stream(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ):
        # 截断过长的内容以避免超过模型限制
        max_draft_chars = 15000  # 约 6000 tokens
        max_sources_chars = 8000  # 约 3200 tokens

        if len(draft) > max_draft_chars:
            draft = draft[:max_draft_chars] + "\n...(内容过长已截断)"

        if len(sources) > max_sources_chars:
            sources = sources[:max_sources_chars] + "\n...(来源过长已截断)"

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
        yield from self.stream(
            prompt,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )
