from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends

from ...models.schemas import (
    CitationSettingRequest,
    CitationSettingResponse,
    GenerationModeSettingRequest,
    GenerationModeSettingResponse,
    SessionMemoryClearRequest,
    SessionMemoryClearResponse,
)
from ...services.generation_mode import (
    get_creative_mcp_enabled,
    get_generation_mode,
    inference_mark_required,
    mcp_allowed_for_mode,
    set_creative_mcp_enabled,
    set_generation_mode,
)
from ..deps import AppServices, get_services

router = APIRouter(tags=["settings"])


def _generation_mode_response() -> GenerationModeSettingResponse:
    mode = get_generation_mode()
    return GenerationModeSettingResponse(
        mode=mode,
        citation_enforce=(mode == "rag_only"),
        mcp_allowed=mcp_allowed_for_mode(mode),
        inference_mark_required=inference_mark_required(mode),
        creative_mcp_enabled=get_creative_mcp_enabled(),
    )


@router.get("/settings/citation", response_model=CitationSettingResponse)
def get_citation_setting() -> CitationSettingResponse:
    return CitationSettingResponse(enabled=(get_generation_mode() == "rag_only"))


@router.post("/settings/citation", response_model=CitationSettingResponse)
def set_citation_setting(payload: CitationSettingRequest) -> CitationSettingResponse:
    # Backward-compatible mapping for old UI:
    # enabled=true  -> rag_only
    # enabled=false -> hybrid
    mode = "rag_only" if payload.enabled else "hybrid"
    set_generation_mode(mode)
    return CitationSettingResponse(enabled=(get_generation_mode() == "rag_only"))


@router.get("/settings/generation-mode", response_model=GenerationModeSettingResponse)
def get_generation_mode_setting() -> GenerationModeSettingResponse:
    return _generation_mode_response()


@router.post("/settings/generation-mode", response_model=GenerationModeSettingResponse)
def set_generation_mode_setting(
    payload: GenerationModeSettingRequest,
) -> GenerationModeSettingResponse:
    set_generation_mode(payload.mode)
    if payload.creative_mcp_enabled is not None:
        set_creative_mcp_enabled(payload.creative_mcp_enabled)
    return _generation_mode_response()


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
