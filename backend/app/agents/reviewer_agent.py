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
        prompt = self._build_review_prompt(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
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
        prompt = self._build_review_prompt(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
        )
        yield from self.stream(
            prompt,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )

    def review_decision(
        self,
        *,
        review: str,
        criteria: str = "",
        audience: str = "",
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> str:
        prompt = self._build_review_decision_prompt(
            review=review,
            criteria=criteria,
            audience=audience,
        )
        return self.run(
            prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )

    def _build_review_prompt(
        self,
        *,
        draft: str,
        criteria: str,
        sources: str,
        audience: str,
    ) -> str:
        parts = self._build_review_parts(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
        )
        return (
            "Review the draft and provide feedback grouped by severity. "
            "Highlight factual claims that need citations.\n\n"
            + "\n\n".join(parts)
        )

    def _build_review_decision_prompt(
        self,
        *,
        review: str,
        criteria: str,
        audience: str,
    ) -> str:
        max_review_chars = 12000
        if len(review) > max_review_chars:
            review = review[:max_review_chars] + "\n...(review truncated)"
        parts = [f"Review feedback:\n{review}"]
        if criteria:
            parts.append(f"Review criteria:\n{criteria}")
        if audience:
            parts.append(f"Audience:\n{audience}")
        schema = (
            '{"review_text":"text","needs_rewrite":true,'
            '"reason":"why","score":0.82}'
        )
        return (
            "Convert the existing review feedback into a structured decision object. "
            "Return exactly one JSON object. "
            "Do not add markdown fences, prose, or extra keys.\n"
            "Use this schema exactly: "
            f"{schema}\n"
            "Rules:\n"
            "- `review_text` must be a concise plain-text restatement of the review feedback.\n"
            "- `needs_rewrite` must be true when the feedback requires content changes before publication.\n"
            "- Set `needs_rewrite` to false only when the feedback says the draft is publishable as-is, aside from optional polish.\n"
            "- Missing citations or unsupported claims count as needing rewrite.\n\n"
            "- `reason` must briefly explain the decision in one sentence.\n"
            "- `score` must be a confidence score between 0.0 and 1.0.\n\n"
            + "\n\n".join(parts)
        )

    def _build_review_parts(
        self,
        *,
        draft: str,
        criteria: str,
        sources: str,
        audience: str,
    ) -> list[str]:
        max_draft_chars = 15000
        max_sources_chars = 8000

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
        return parts
