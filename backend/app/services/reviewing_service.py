from __future__ import annotations

from dataclasses import dataclass

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
    ) -> ReviewResult:
        review_text = self.agent.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
        )
        return ReviewResult(review=review_text)
