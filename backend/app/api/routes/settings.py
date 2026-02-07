from __future__ import annotations

import os

from fastapi import APIRouter

from ...models.schemas import CitationSettingRequest, CitationSettingResponse

router = APIRouter(tags=["settings"])


def _get_flag() -> bool:
    return os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")


@router.get("/settings/citation", response_model=CitationSettingResponse)
def get_citation_setting() -> CitationSettingResponse:
    return CitationSettingResponse(enabled=_get_flag())


@router.post("/settings/citation", response_model=CitationSettingResponse)
def set_citation_setting(payload: CitationSettingRequest) -> CitationSettingResponse:
    os.environ["RAG_CITATION_ENFORCE"] = "true" if payload.enabled else "false"
    return CitationSettingResponse(enabled=_get_flag())
