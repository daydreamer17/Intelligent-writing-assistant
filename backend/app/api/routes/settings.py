from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi import Depends

from ...models.schemas import (
    CitationSettingRequest,
    CitationSettingResponse,
    SessionMemoryClearRequest,
    SessionMemoryClearResponse,
)
from ..deps import AppServices, get_services

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


@router.post("/settings/session-memory/clear", response_model=SessionMemoryClearResponse)
def clear_session_memory(
    payload: SessionMemoryClearRequest,
    services: AppServices = Depends(get_services),
) -> SessionMemoryClearResponse:
    targets = [
        ("writing", services.planner.agent),
        ("reviewer", services.reviewer.agent),
        ("editor", services.rewriter.agent),
    ]
    cleared_agents: list[str] = []
    seen: set[int] = set()
    for name, agent in targets:
        identity = id(agent)
        if identity in seen:
            continue
        seen.add(identity)
        clear_fn = getattr(agent, "clear_session_memory", None)
        if not callable(clear_fn):
            continue
        if clear_fn(
            session_id=payload.session_id,
            drop_agent=payload.drop_agent,
            clear_cold=payload.clear_cold,
        ):
            cleared_agents.append(name)
    return SessionMemoryClearResponse(
        session_id=payload.session_id or "__default__",
        cleared=bool(cleared_agents),
        cleared_agents=cleared_agents,
    )
