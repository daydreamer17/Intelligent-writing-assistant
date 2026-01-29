from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .research_service import ResearchNote


@dataclass(frozen=True)
class Citation:
    label: str
    title: str
    url: str = ""


class CitationService:
    def build_citations(self, notes: Iterable[ResearchNote]) -> List[Citation]:
        citations: List[Citation] = []
        for idx, note in enumerate(notes, start=1):
            label = f"[{idx}]"
            citations.append(Citation(label=label, title=note.title, url=note.url))
        return citations

    def format_bibliography(self, citations: Iterable[Citation]) -> str:
        lines = []
        for citation in citations:
            line = f"{citation.label} {citation.title}"
            if citation.url:
                line += f" - {citation.url}"
            lines.append(line)
        return "\n".join(lines)
