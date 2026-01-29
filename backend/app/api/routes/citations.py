from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...models.schemas import (
    CitationItemResponse,
    CitationRequest,
    CitationResponse,
)
from ...models.entities import CitationEntity
from ...services.research_service import ResearchNote
from ..deps import AppServices, get_services

router = APIRouter(tags=["citations"])


@router.post("/citations", response_model=CitationResponse)
def build_citations(
    payload: CitationRequest,
    services: AppServices = Depends(get_services),
) -> CitationResponse:
    try:
        notes = [
            ResearchNote(
                doc_id=item.doc_id,
                title=item.title,
                summary=item.summary,
                url=item.url,
            )
            for item in payload.notes
        ]
        citations = services.citations.build_citations(notes)
        bibliography = services.citations.format_bibliography(citations)
        entities = [
            CitationEntity(
                citation_id=0,
                label=item.label,
                title=item.title,
                url=item.url,
            )
            for item in citations
        ]
        services.storage.save_citations(entities)
        response_items = [
            CitationItemResponse(label=item.label, title=item.title, url=item.url)
            for item in citations
        ]
        return CitationResponse(citations=response_items, bibliography=bibliography)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Citation build failed") from exc
