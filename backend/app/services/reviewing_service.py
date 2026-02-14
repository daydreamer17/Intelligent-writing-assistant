from __future__ import annotations

from dataclasses import dataclass

from hello_agents.tools import ToolRegistry

from ..agents.reviewer_agent import ReviewerAgent


@dataclass(frozen=True)
class ReviewResult:
    review: str


class ReviewingService:
    def __init__(self, agent: ReviewerAgent) -> None:
        self.agent = agent

    def review(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> ReviewResult:
        review_text = self.agent.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )
        return ReviewResult(review=review_text)
