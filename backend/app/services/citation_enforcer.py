from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ..utils.tokenizer import tokenize_for_citation
from .research_service import ResearchNote
from .vector_store import Embedder

logger = logging.getLogger("app.citations")


@dataclass(frozen=True)
class CoverageReport:
    coverage: float
    token_coverage: float
    total_tokens: int
    covered_tokens: int
    total_paragraphs: int
    covered_paragraphs: int
    semantic_coverage: float | None


class CitationEnforcer:
    def __init__(self) -> None:
        self._threshold = _parse_float_env("RAG_COVERAGE_THRESHOLD", default=0.3)

    def enforce(
        self,
        text: str,
        notes: Iterable[ResearchNote],
        *,
        apply_labels: bool = True,
        embedder: Embedder | None = None,
    ) -> Tuple[str, CoverageReport]:
        notes_list = list(notes)
        if not text.strip():
            return text, CoverageReport(
                coverage=0.0,
                token_coverage=0.0,
                total_tokens=0,
                covered_tokens=0,
                total_paragraphs=0,
                covered_paragraphs=0,
                semantic_coverage=None,
            )
        if not notes_list:
            return text, CoverageReport(
                coverage=0.0,
                token_coverage=0.0,
                total_tokens=0,
                covered_tokens=0,
                total_paragraphs=0,
                covered_paragraphs=0,
                semantic_coverage=None,
            )

        labels = [f"[{idx}]" for idx in range(1, len(notes_list) + 1)]
        note_tokens = [self._tokenize(note.title + " " + note.summary) for note in notes_list]

        paragraphs = _split_paragraphs(text)
        total_tokens = 0
        covered_tokens = 0
        covered_paragraphs = 0
        enforced_paragraphs: List[str] = []
        paragraph_tokens: List[List[str]] = []

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            tokens = self._tokenize(stripped)
            paragraph_tokens.append(tokens)
            total_tokens += len(tokens)
            best_idx, best_ratio = self._best_match(tokens, note_tokens)
            if best_ratio >= self._threshold:
                covered_tokens += len(tokens)
                covered_paragraphs += 1
            if apply_labels and not _has_citation(stripped, labels) and labels:
                label = labels[best_idx] if best_idx is not None else labels[0]
                stripped = f"{stripped} {label}"
            enforced_paragraphs.append(stripped)

        if apply_labels and _parse_bool_env("RAG_CITATION_REQUIRE_ALL_LABELS", True):
            enforced_paragraphs = self._ensure_all_labels(
                paragraphs=enforced_paragraphs,
                paragraph_tokens=paragraph_tokens,
                labels=labels,
                note_tokens=note_tokens,
            )

        total_paragraphs = len(enforced_paragraphs)
        token_coverage = covered_tokens / total_tokens if total_tokens > 0 else 0.0
        semantic_coverage = self._semantic_coverage(
            paragraphs=enforced_paragraphs,
            notes=notes_list,
            embedder=embedder,
        )
        coverage = max(token_coverage, semantic_coverage or 0.0)
        report = CoverageReport(
            coverage=coverage,
            token_coverage=token_coverage,
            total_tokens=total_tokens,
            covered_tokens=covered_tokens,
            total_paragraphs=total_paragraphs,
            covered_paragraphs=covered_paragraphs,
            semantic_coverage=semantic_coverage,
        )
        logger.info(
            "Citation coverage: %.2f (token=%.2f semantic=%s covered_tokens=%s total_tokens=%s covered_paragraphs=%s total_paragraphs=%s)",
            coverage,
            token_coverage,
            f"{semantic_coverage:.2f}" if semantic_coverage is not None else "n/a",
            covered_tokens,
            total_tokens,
            covered_paragraphs,
            total_paragraphs,
        )
        output_text = "\n\n".join(enforced_paragraphs) if apply_labels else text
        return output_text, report

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """使用统一的分词工具（用于引用匹配）"""
        return tokenize_for_citation(text)

    def _best_match(self, tokens: List[str], note_tokens: List[List[str]]) -> Tuple[int | None, float]:
        if not tokens:
            return None, 0.0
        token_set = set(tokens)
        best_idx = None
        best_ratio = 0.0
        for idx, note in enumerate(note_tokens):
            if not note:
                continue
            overlap = token_set.intersection(note)
            ratio = len(overlap) / max(len(token_set), 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        return best_idx, best_ratio

    def _ensure_all_labels(
        self,
        *,
        paragraphs: List[str],
        paragraph_tokens: List[List[str]],
        labels: List[str],
        note_tokens: List[List[str]],
    ) -> List[str]:
        if not paragraphs or not labels:
            return paragraphs

        used_labels = _labels_in_text("\n".join(paragraphs), labels)
        missing = [idx for idx, label in enumerate(labels) if label not in used_labels]
        if not missing:
            return paragraphs

        paragraphs_out = list(paragraphs)
        para_used_for_backfill: set[int] = set()

        for note_idx in missing:
            label = labels[note_idx]
            target_idx = self._best_paragraph_for_note(
                note_tokens[note_idx] if note_idx < len(note_tokens) else [],
                paragraph_tokens,
                prefer_unused=True,
                used_paragraphs=para_used_for_backfill,
            )
            if target_idx is None:
                target_idx = self._best_paragraph_for_note(
                    note_tokens[note_idx] if note_idx < len(note_tokens) else [],
                    paragraph_tokens,
                    prefer_unused=False,
                    used_paragraphs=para_used_for_backfill,
                )
            if target_idx is None:
                continue
            if label not in paragraphs_out[target_idx]:
                paragraphs_out[target_idx] = f"{paragraphs_out[target_idx]} {label}"
            para_used_for_backfill.add(target_idx)

        final_used = _labels_in_text("\n".join(paragraphs_out), labels)
        logger.info(
            "Citation label coverage: used=%s/%s missing_before=%s",
            len(final_used),
            len(labels),
            len(missing),
        )
        return paragraphs_out

    def _best_paragraph_for_note(
        self,
        note_tokens: List[str],
        paragraph_tokens: List[List[str]],
        *,
        prefer_unused: bool,
        used_paragraphs: set[int],
    ) -> int | None:
        if not paragraph_tokens:
            return None
        note_set = set(note_tokens)
        best_idx: int | None = None
        best_score = -1.0
        for idx, para_tokens in enumerate(paragraph_tokens):
            if prefer_unused and idx in used_paragraphs:
                continue
            token_set = set(para_tokens)
            if note_set and token_set:
                overlap = len(note_set.intersection(token_set))
                score = overlap / max(len(note_set), 1)
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _semantic_coverage(
        self,
        *,
        paragraphs: List[str],
        notes: List[ResearchNote],
        embedder: Embedder | None,
    ) -> float | None:
        if not embedder or not _parse_bool_env("RAG_COVERAGE_SEMANTIC_ENABLED", default=False):
            return None
        if not paragraphs or not notes:
            return 0.0
        max_paragraphs = max(1, _parse_int_env("RAG_COVERAGE_SEMANTIC_MAX_PARAGRAPHS", 20))
        max_notes = max(1, _parse_int_env("RAG_COVERAGE_SEMANTIC_MAX_NOTES", 12))
        threshold = _parse_float_env("RAG_COVERAGE_SEMANTIC_THRESHOLD", 0.25)
        para_slice = paragraphs[:max_paragraphs]
        note_texts = [(note.title + " " + note.summary).strip() for note in notes[:max_notes]]
        try:
            embeddings = embedder.embed(para_slice + note_texts)
        except Exception as exc:
            logger.warning("Semantic coverage failed: %s", exc)
            return None
        para_vecs = embeddings[: len(para_slice)]
        note_vecs = embeddings[len(para_slice) :]
        if not note_vecs:
            return 0.0
        covered = 0
        for para_vec in para_vecs:
            best = max((_cosine(para_vec, note_vec) for note_vec in note_vecs), default=0.0)
            if best >= threshold:
                covered += 1
        return covered / len(para_slice) if para_slice else 0.0


def _split_paragraphs(text: str) -> List[str]:
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return parts or [text.strip()]


def _has_citation(text: str, labels: List[str]) -> bool:
    for label in labels:
        if label in text:
            return True
    return False


def _labels_in_text(text: str, labels: List[str]) -> set[str]:
    used: set[str] = set()
    for label in labels:
        if label in text:
            used.add(label)
    return used


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes")


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for i in range(min(len(left), len(right))):
        lv = left[i]
        rv = right[i]
        dot += lv * rv
        left_norm += lv * lv
        right_norm += rv * rv
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))
