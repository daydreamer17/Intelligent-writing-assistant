from __future__ import annotations

from dataclasses import dataclass
import os
import re

from ..agents.writing_agent import WritingAgent
from .evidence_service import EvidenceExtractor
from .reviewing_service import ReviewingService
from .rewriting_service import RewritingService
from ..utils.tokenizer import tokenize


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
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
    ) -> str:
        target_len = _parse_target_length(target_length)
        prompt_constraints = self.build_constraints(
            research_notes=research_notes,
            constraints=constraints,
            evidence_text=evidence_text,
            topic=topic,
            outline=outline,
        )

        if target_len is not None and target_len >= 1800:
            return self.writing_agent.draft_long(
                topic=topic,
                outline=outline,
                constraints=prompt_constraints,
                style=style,
                target_length=target_length,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
            )

        return self.writing_agent.draft(
            topic=topic,
            outline=outline,
            constraints=prompt_constraints,
            style=style,
            target_length=target_length,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
        )

    def review_draft(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
    ) -> str:
        if not self.reviewing_service:
            return ""
        return self.reviewing_service.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
        ).review

    def revise_draft(
        self,
        *,
        draft: str,
        guidance: str,
        style: str = "",
        target_length: str = "",
        evidence_text: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
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
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
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
        plan_max_tokens: int | None = None,
        draft_max_tokens: int | None = None,
        review_max_tokens: int | None = None,
        rewrite_max_tokens: int | None = None,
        plan_max_input_chars: int | None = None,
        draft_max_input_chars: int | None = None,
        review_max_input_chars: int | None = None,
        rewrite_max_input_chars: int | None = None,
        session_id: str = "",
    ) -> DraftResult:
        _ = plan_max_tokens
        _ = plan_max_input_chars
        evidence_text = self.extract_evidence(research_notes)
        draft = self.create_draft(
            topic=topic,
            outline=outline,
            research_notes=research_notes,
            constraints=constraints,
            style=style,
            target_length=target_length,
            evidence_text=evidence_text,
            max_tokens=draft_max_tokens,
            max_input_chars=draft_max_input_chars,
            session_id=session_id,
        )
        review = self.review_draft(
            draft=draft,
            criteria=review_criteria,
            sources=research_notes,
            audience=audience,
            max_tokens=review_max_tokens,
            max_input_chars=review_max_input_chars,
            session_id=session_id,
        )
        revised = self.revise_draft(
            draft=draft,
            guidance=review,
            style=style,
            target_length=target_length,
            evidence_text=evidence_text,
            max_tokens=rewrite_max_tokens,
            max_input_chars=rewrite_max_input_chars,
            session_id=session_id,
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
        topic: str = "",
        outline: str = "",
    ) -> str:
        query_text = (topic + "\n" + outline).strip()
        notes = _sanitize_research_notes_for_prompt(research_notes, query=query_text)
        notes = notes or "No external sources provided."
        if _citations_enabled():
            evidence = evidence_text or self.extract_evidence(research_notes)
            if evidence:
                return (constraints + "\n\nUse ONLY the evidence below. Do not add new facts.\n" + evidence).strip()
        if notes and notes != "No external sources provided.":
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


def _sanitize_research_notes_for_prompt(notes: str, *, query: str = "") -> str:
    text = (notes or "").strip()
    if not text:
        return ""

    max_chars = _parse_int_env("RAG_NOTES_INJECTION_MAX_CHARS", 4000)
    max_items = _parse_int_env("RAG_NOTES_INJECTION_MAX_ITEMS", 8)
    min_chars = _parse_int_env("RAG_NOTES_INJECTION_MIN_CHARS", 120)

    blocks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("- ") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    if not blocks:
        blocks = [text]

    deduped: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        norm = " ".join(block.split()).lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(block)
        if len(deduped) >= max_items:
            break

    filtered = deduped
    min_score = _parse_float_env("RAG_NOTES_INJECTION_MIN_SCORE", 0.03)
    if query and min_score > 0:
        q_tokens = tokenize(query, lowercase=True)
        if q_tokens:
            scored = [(_note_relevance_score(item, q_tokens), item) for item in deduped if item]
            kept = [item for score, item in scored if score >= min_score]
            if not kept and scored:
                kept = [max(scored, key=lambda item: item[0])[1]]
            filtered = kept

    merged = "\n".join(item for item in filtered if item).strip()
    if max_chars > 0 and len(merged) > max_chars:
        merged = merged[:max_chars].rstrip() + "\n...(research notes truncated)"
    if len(merged) < max(0, min_chars):
        return ""
    return merged


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _note_relevance_score(note_text: str, query_tokens: set[str]) -> float:
    tokens = tokenize(note_text, lowercase=True)
    if not tokens or not query_tokens:
        return 0.0
    overlap = len(tokens & query_tokens)
    return overlap / max(1, len(query_tokens))
