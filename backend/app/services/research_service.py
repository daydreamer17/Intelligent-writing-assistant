from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


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
        return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token}
