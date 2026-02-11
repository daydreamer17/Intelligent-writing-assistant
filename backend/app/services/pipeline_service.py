from __future__ import annotations

from dataclasses import dataclass
import time
import os
import logging
from typing import Iterable, List

from .drafting_service import DraftResult, DraftingService
from .planner_service import OutlinePlan, PlanningService
from .research_service import ResearchNote, ResearchService, SourceDocument

logger = logging.getLogger("app.pipeline")


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
        plan_max_tokens: int | None = None,
        draft_max_tokens: int | None = None,
        review_max_tokens: int | None = None,
        rewrite_max_tokens: int | None = None,
        plan_max_input_chars: int | None = None,
        draft_max_input_chars: int | None = None,
        review_max_input_chars: int | None = None,
        rewrite_max_input_chars: int | None = None,
        session_id: str = "",
    ) -> PipelineResult:
        outline = self.planner.plan_outline(
            topic=topic,
            audience=audience,
            style=style,
            target_length=target_length,
            constraints=constraints,
            key_points=key_points,
            max_tokens=plan_max_tokens,
            max_input_chars=plan_max_input_chars,
            session_id=session_id,
        )

        _pipeline_throttle()

        notes = self.collect_research_notes(topic, outline, sources)
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
            plan_max_tokens=plan_max_tokens,
            draft_max_tokens=draft_max_tokens,
            review_max_tokens=review_max_tokens,
            rewrite_max_tokens=rewrite_max_tokens,
            plan_max_input_chars=plan_max_input_chars,
            draft_max_input_chars=draft_max_input_chars,
            review_max_input_chars=review_max_input_chars,
            rewrite_max_input_chars=rewrite_max_input_chars,
            session_id=session_id,
        )

        return PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)

    def collect_research_notes(
        self,
        topic: str,
        outline: OutlinePlan,
        sources: Iterable[SourceDocument] | None,
    ) -> List[ResearchNote]:
        source_list = list(sources) if sources else []
        if not source_list:
            return []
        query = f"{topic}\n{outline.outline}"
        top_k = _resolve_notes_top_k(len(source_list))
        logger.info(
            "Pipeline research notes plan: sources=%s top_k=%s",
            len(source_list),
            top_k,
        )
        return self.researcher.collect_notes(query=query, sources=source_list, top_k=top_k)


def _resolve_notes_top_k(source_count: int) -> int:
    source_count = max(0, source_count)
    if source_count == 0:
        return 0

    dynamic_enabled = os.getenv("RAG_NOTES_DYNAMIC_ENABLED", "true").lower() in ("1", "true", "yes")
    fixed_top_k = max(1, _parse_int_env("RAG_NOTES_TOP_K", 3))
    if not dynamic_enabled:
        return min(source_count, fixed_top_k)

    small_threshold = max(1, _parse_int_env("RAG_NOTES_SMALL_THRESHOLD", 10))
    large_threshold = max(small_threshold + 1, _parse_int_env("RAG_NOTES_LARGE_THRESHOLD", 100))
    top_k_small = max(1, _parse_int_env("RAG_NOTES_TOP_K_SMALL", 3))
    top_k_medium = max(1, _parse_int_env("RAG_NOTES_TOP_K_MEDIUM", 5))
    top_k_large = max(1, _parse_int_env("RAG_NOTES_TOP_K_LARGE", 8))

    if source_count <= small_threshold:
        target = top_k_small
    elif source_count >= large_threshold:
        target = top_k_large
    else:
        target = top_k_medium

    return min(source_count, target)


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _pipeline_throttle() -> None:
    """Sleep between pipeline stages to avoid hitting TPM limits."""
    raw = os.getenv("PIPELINE_STAGE_SLEEP", "0")
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if seconds > 0:
        time.sleep(seconds)
