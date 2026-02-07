from __future__ import annotations

from dataclasses import dataclass
import os
import re

from ..agents.writing_agent import WritingAgent
from .evidence_service import EvidenceExtractor
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
        evidence_extractor: EvidenceExtractor | None = None,
    ) -> None:
        self.writing_agent = writing_agent
        self.reviewing_service = reviewing_service
        self.rewriting_service = rewriting_service
        self.evidence_extractor = evidence_extractor

    def create_draft(
        self,
        *,
        topic: str,
        outline: str,
        research_notes: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
        evidence_text: str = "",
    ) -> str:
        target_len = _parse_target_length(target_length)
        prompt_constraints = self.build_constraints(
            research_notes=research_notes,
            constraints=constraints,
            evidence_text=evidence_text,
        )

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
        evidence_text: str = "",
    ) -> str:
        if not self.rewriting_service:
            return draft
        final_guidance = guidance
        if evidence_text:
            final_guidance = (guidance + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
        return self.rewriting_service.rewrite(
            draft=draft,
            guidance=final_guidance,
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
        evidence_text = self.extract_evidence(research_notes)
        draft = self.create_draft(
            topic=topic,
            outline=outline,
            research_notes=research_notes,
            constraints=constraints,
            style=style,
            target_length=target_length,
            evidence_text=evidence_text,
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
            evidence_text=evidence_text,
        )
        return DraftResult(
            outline=outline,
            research_notes=research_notes,
            draft=draft,
            review=review,
            revised=revised,
        )

    def extract_evidence(self, research_notes: str) -> str:
        if not _citations_enabled():
            return ""
        if not self.evidence_extractor or not research_notes.strip():
            return ""
        return self.evidence_extractor.extract(research_notes)

    def build_constraints(
        self,
        *,
        research_notes: str,
        constraints: str,
        evidence_text: str = "",
    ) -> str:
        notes = research_notes or "No external sources provided."
        if _citations_enabled():
            evidence = evidence_text or self.extract_evidence(research_notes)
            if evidence:
                return (constraints + "\n\nUse ONLY the evidence below. Do not add new facts.\n" + evidence).strip()
        if research_notes:
            return (constraints + "\n\nUse the research notes below.\n" + notes).strip()
        return (constraints + "\n\nDo not invent facts.").strip()


def _parse_target_length(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _citations_enabled() -> bool:
    return os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")
