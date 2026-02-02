from __future__ import annotations

from dataclasses import dataclass
import time
import os
from typing import Iterable, List

from .drafting_service import DraftResult, DraftingService
from .planner_service import OutlinePlan, PlanningService
from .research_service import ResearchNote, ResearchService, SourceDocument


@dataclass(frozen=True)
class PipelineResult:
    outline: OutlinePlan
    research_notes: List[ResearchNote]
    draft_result: DraftResult


class WritingPipeline:
    def __init__(
        self,
        *,
        planner: PlanningService,
        researcher: ResearchService,
        drafter: DraftingService,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.drafter = drafter

    def run(
        self,
        *,
        topic: str,
        audience: str = "",
        style: str = "",
        target_length: str = "",
        constraints: str = "",
        key_points: str = "",
        sources: Iterable[SourceDocument] | None = None,
        review_criteria: str = "",
    ) -> PipelineResult:
        outline = self.planner.plan_outline(
            topic=topic,
            audience=audience,
            style=style,
            target_length=target_length,
            constraints=constraints,
            key_points=key_points,
        )

        _pipeline_throttle()

        notes = self._collect_research_notes(topic, outline, sources)
        notes_text = self.researcher.format_notes(notes)

        _pipeline_throttle()

        draft_result = self.drafter.run_full(
            topic=topic,
            outline=outline.outline,
            research_notes=notes_text,
            constraints=constraints,
            style=style,
            target_length=target_length,
            review_criteria=review_criteria,
            audience=audience,
        )

        return PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)

    def _collect_research_notes(
        self,
        topic: str,
        outline: OutlinePlan,
        sources: Iterable[SourceDocument] | None,
    ) -> List[ResearchNote]:
        if not sources:
            return []
        query = f"{topic}\n{outline.outline}"
        return self.researcher.collect_notes(query=query, sources=sources)


def _pipeline_throttle() -> None:
    """Sleep between pipeline stages to avoid hitting TPM limits."""
    raw = os.getenv("PIPELINE_STAGE_SLEEP", "0")
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if seconds > 0:
        time.sleep(seconds)
