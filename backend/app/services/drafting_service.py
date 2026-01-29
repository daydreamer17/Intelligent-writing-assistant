from __future__ import annotations

from dataclasses import dataclass
import re

from ..agents.writing_agent import WritingAgent
from .reviewing_service import ReviewingService
from .rewriting_service import RewritingService


@dataclass(frozen=True)
class DraftResult:
    outline: str
    research_notes: str
    draft: str
    review: str
    revised: str


class DraftingService:
    def __init__(
        self,
        *,
        writing_agent: WritingAgent,
        reviewing_service: ReviewingService | None = None,
        rewriting_service: RewritingService | None = None,
    ) -> None:
        self.writing_agent = writing_agent
        self.reviewing_service = reviewing_service
        self.rewriting_service = rewriting_service

    def create_draft(
        self,
        *,
        topic: str,
        outline: str,
        research_notes: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
    ) -> str:
        target_len = _parse_target_length(target_length)
        notes = research_notes or "No external sources provided."
        prompt_constraints = constraints
        if research_notes:
            prompt_constraints = (constraints + "\n\nUse the research notes below.\n" + notes).strip()
        else:
            prompt_constraints = (constraints + "\n\nDo not invent facts.").strip()

        if target_len is not None and target_len >= 1800:
            return self.writing_agent.draft_long(
                topic=topic,
                outline=outline,
                constraints=prompt_constraints,
                style=style,
                target_length=target_length,
            )

        return self.writing_agent.draft(
            topic=topic,
            outline=outline,
            constraints=prompt_constraints,
            style=style,
            target_length=target_length,
        )

    def review_draft(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
    ) -> str:
        if not self.reviewing_service:
            return ""
        return self.reviewing_service.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
        ).review

    def revise_draft(
        self,
        *,
        draft: str,
        guidance: str,
        style: str = "",
        target_length: str = "",
    ) -> str:
        if not self.rewriting_service:
            return draft
        return self.rewriting_service.rewrite(
            draft=draft,
            guidance=guidance,
            style=style,
            target_length=target_length,
        ).revised

    def run_full(
        self,
        *,
        topic: str,
        outline: str,
        research_notes: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
        review_criteria: str = "",
        audience: str = "",
    ) -> DraftResult:
        draft = self.create_draft(
            topic=topic,
            outline=outline,
            research_notes=research_notes,
            constraints=constraints,
            style=style,
            target_length=target_length,
        )
        review = self.review_draft(
            draft=draft,
            criteria=review_criteria,
            sources=research_notes,
            audience=audience,
        )
        revised = self.revise_draft(
            draft=draft,
            guidance=review,
            style=style,
            target_length=target_length,
        )
        return DraftResult(
            outline=outline,
            research_notes=research_notes,
            draft=draft,
            review=review,
            revised=revised,
        )


def _parse_target_length(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None
