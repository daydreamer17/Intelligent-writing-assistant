from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..utils.tokenizer import tokenize


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    content: str
    url: str = ""


@dataclass(frozen=True)
class ResearchNote:
    doc_id: str
    title: str
    summary: str
    url: str = ""


@dataclass(frozen=True)
class RelevanceReport:
    query_terms: int
    docs: int
    best_recall: float
    avg_recall: float


class ResearchService:
    def __init__(self, *, max_snippet_chars: int = 600) -> None:
        self.max_snippet_chars = max_snippet_chars

    def collect_notes(
        self,
        *,
        query: str,
        sources: Iterable[SourceDocument],
        top_k: int = 3,
    ) -> List[ResearchNote]:
        ranked = sorted(
            sources,
            key=lambda doc: self._score(query, doc.content, doc.title),
            reverse=True,
        )
        notes: List[ResearchNote] = []
        for doc in ranked[:top_k]:
            snippet = doc.content.strip().replace("\n", " ")
            summary = snippet[: self.max_snippet_chars].strip()
            notes.append(
                ResearchNote(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    summary=summary,
                    url=doc.url,
                )
            )
        return notes

    def format_notes(self, notes: Iterable[ResearchNote]) -> str:
        blocks = []
        for note in notes:
            header = f"- {note.title} ({note.doc_id})"
            if note.url:
                header += f" [{note.url}]"
            blocks.append(header + "\n  " + note.summary)
        return "\n".join(blocks)

    def relevance_report(
        self,
        query: str,
        sources: Iterable[SourceDocument],
    ) -> RelevanceReport:
        terms = self._tokenize(query)
        total_terms = len(terms)
        recalls: List[float] = []
        if total_terms == 0:
            return RelevanceReport(query_terms=0, docs=0, best_recall=0.0, avg_recall=0.0)
        for doc in sources:
            doc_terms = self._tokenize(doc.title + " " + doc.content)
            if not doc_terms:
                continue
            matched = len(terms & doc_terms)
            recalls.append(matched / total_terms)
        if not recalls:
            return RelevanceReport(query_terms=total_terms, docs=0, best_recall=0.0, avg_recall=0.0)
        best = max(recalls)
        avg = sum(recalls) / len(recalls)
        return RelevanceReport(query_terms=total_terms, docs=len(recalls), best_recall=best, avg_recall=avg)

    @staticmethod
    def _score(query: str, content: str, title: str) -> int:
        query_terms = ResearchService._tokenize(query)
        if not query_terms:
            return 0
        content_terms = ResearchService._tokenize(content)
        title_terms = ResearchService._tokenize(title)
        return len(query_terms & content_terms) + 2 * len(query_terms & title_terms)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """使用统一的分词工具（支持中文）"""
        return tokenize(text, lowercase=True)
