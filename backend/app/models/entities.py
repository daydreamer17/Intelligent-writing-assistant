from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentEntity:
    doc_id: str
    title: str
    content: str
    url: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class DraftVersion:
    version_id: int
    topic: str
    outline: str
    research_notes: str
    draft: str
    review: str
    revised: str
    created_at: str = ""


@dataclass(frozen=True)
class CitationEntity:
    citation_id: int
    label: str
    title: str
    url: str = ""
    created_at: str = ""
