
from __future__ import annotations

import json
import logging
import time
from queue import Empty, Queue
from threading import Thread
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...models.schemas import (
    DeletePipelineV2CheckpointResponse,
    PipelineRequest,
    PipelineResponse,
    PipelineV2CheckpointCleanupRequest,
    PipelineV2CheckpointCleanupResponse,
    PipelineV2CheckpointDetailResponse,
    PipelineV2CheckpointListResponse,
    PipelineV2CheckpointSummary,
    PipelineV2Interrupt,
    PipelineV2Request,
    PipelineV2Response,
    PipelineV2ResumeRequest,
)
from ...services.github_context import maybe_fetch_github_context
from ...services.pipeline_langgraph_v2 import (
    clear_pipeline_v2_graph_thread,
    cleanup_pipeline_v2_checkpoints,
    delete_pipeline_v2_checkpoint,
    get_pipeline_v2_checkpoint_detail,
    has_pipeline_v2_interrupt_checkpoint,
    list_pipeline_v2_checkpoints,
    load_pipeline_v2_resume_state,
    mark_pipeline_v2_checkpoint_failed,
    resume_pipeline_v2_full_workflow,
    run_pipeline_v2_full_stream,
    run_pipeline_v2_full_sync,
    resume_pipeline_v2_workflow,
    start_pipeline_v2_workflow,
    upsert_pipeline_v2_checkpoint,
)
from ...services.planner_service import OutlinePlan
from ...services.research_service import SourceDocument
from ..deps import AppServices, get_services
from .pipeline import (
    _append_block,
    _citations_enabled,
    _log_task_success_rate,
    _mcp_enabled,
    _pipeline_max_input_chars,
    _pipeline_max_tokens,
    _resolve_stage_tools,
    _runtime_tool_profile_id,
    _runtime_tool_registry,
)

router = APIRouter(tags=["pipeline"])
logger = logging.getLogger("app.pipeline.v2")


def _log_v2_stage(
    stage: str,
    *,
    thread_id: str,
    resolved_session_id: str = "",
    started_at: float | None = None,
) -> None:
    elapsed_ms = ((time.perf_counter() - started_at) * 1000) if started_at is not None else 0.0
    logger.info(
        "pipeline v2 route: stage=%s thread_id=%s session_id=%s elapsed_ms=%.2f",
        stage,
        thread_id or "__unknown__",
        resolved_session_id or "__default__",
        elapsed_ms,
    )


def _event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_workflow_input(
    *,
    payload: PipelineV2Request,
    thread_id: str,
    resolved_session_id: str,
    github_context: str,
    effective_constraints: str,
    plan_tool_allowed_tools: list[str],
    plan_tool_profile_id: str | None,
    prefetched_plan_result: dict[str, Any] | None = None,
) -> dict[str, object]:
    workflow_input: dict[str, object] = {
        "thread_id": thread_id,
        "resolved_session_id": resolved_session_id,
        "request": payload.model_dump(exclude={"thread_id"}),
        "github_context": github_context,
        "effective_constraints": effective_constraints,
        "plan_max_tokens": _pipeline_max_tokens("plan"),
        "plan_max_input_chars": _pipeline_max_input_chars("plan"),
        "plan_tool_profile_id": plan_tool_profile_id,
        "plan_tool_allowed_tools": plan_tool_allowed_tools,
    }
    if prefetched_plan_result:
        workflow_input["prefetched_plan_result"] = prefetched_plan_result
    return workflow_input


def _completed_v2_response(thread_id: str, response_payload: dict[str, Any]) -> PipelineV2Response:
    return PipelineV2Response(
        status="completed",
        thread_id=thread_id,
        result=PipelineResponse.model_validate(response_payload),
    )


def _interrupted_v2_response(thread_id: str, *, kind: str, payload: dict[str, Any]) -> PipelineV2Response:
    return PipelineV2Response(
        status="interrupted",
        thread_id=thread_id,
        interrupt=PipelineV2Interrupt(kind=kind, payload=payload),
    )


def _load_resume_checkpoint(thread_id: str) -> dict[str, Any] | None:
    detail = load_pipeline_v2_resume_state(thread_id)
    if not detail:
        return None
    return {
        "status": str(detail.get("status") or ""),
        "current_stage": str(detail.get("current_stage") or ""),
        "payload": dict(detail.get("payload") or {}),
    }


def _completed_response_from_checkpoint(thread_id: str, resume_checkpoint: dict[str, Any]) -> PipelineV2Response:
    payload = dict(resume_checkpoint.get("payload") or {})
    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        raise HTTPException(status_code=500, detail="Pipeline v2 completed checkpoint is corrupted")
    return _completed_v2_response(thread_id, response_payload)


def _build_effective_constraints(payload: PipelineRequest, github_context: str) -> str:
    return _append_block(payload.constraints, github_context, "GitHub参考")


def _build_sources(payload: PipelineRequest, github_context: str) -> list[SourceDocument]:
    sources = [
        SourceDocument(
            doc_id=doc.doc_id,
            title=doc.title,
            content=doc.content,
            url=doc.url,
        )
        for doc in payload.sources
    ]
    if github_context:
        sources.append(
            SourceDocument(
                doc_id="github:mcp",
                title="GitHub MCP Context",
                content=github_context,
                url="",
            )
        )
    return sources


def _request_from_dict(request_data: dict[str, Any]) -> PipelineRequest:
    return PipelineRequest.model_validate(request_data)


def _outline_from_data(data: dict[str, Any]) -> OutlinePlan:
    return OutlinePlan(
        outline=str(data.get("outline") or ""),
        assumptions=str(data.get("assumptions") or ""),
        open_questions=str(data.get("open_questions") or ""),
    )


def _extract_resume_context(thread_id: str) -> tuple[dict[str, Any] | None, str]:
    resume_checkpoint = _load_resume_checkpoint(thread_id)
    if not resume_checkpoint:
        return None, ""
    payload = dict(resume_checkpoint.get("payload") or {})
    resolved_session_id = str(payload.get("resolved_session_id") or payload.get("session_id") or "")
    return resume_checkpoint, resolved_session_id


def _resolve_resume_stage(resume_checkpoint: dict[str, Any]) -> str:
    stage = str(resume_checkpoint.get("current_stage") or "").strip() or "outline_accepted"
    if stage == "draft_review":
        return "draft_done"
    if stage == "review_confirmation":
        return "review_done"
    if stage not in {
        "outline_accepted",
        "research_done",
        "draft_done",
        "review_done",
        "rewrite_done",
        "completed",
    }:
        return "outline_accepted"
    return stage


def _build_full_input(
    *,
    thread_id: str,
    mode: str,
    request_model: PipelineRequest,
    resolved_session_id: str,
    effective_constraints: str,
    github_context: str,
    outline: OutlinePlan,
    research_notes: list[Any],
    notes_text: str,
    draft: str,
    review: str = "",
    needs_rewrite: bool | None = None,
    review_reason: str = "",
    review_score: float | None = None,
    revised: str = "",
    start_stage: str = "outline_accepted",
    source_count: int = 0,
    services: AppServices,
) -> dict[str, Any]:
    review_tool_decision, _ = _resolve_stage_tools(
        services=services,
        stage="review",
        topic=request_model.topic,
        outline=outline.outline,
        draft=draft,
        guidance=request_model.review_criteria,
        research_notes=notes_text,
        rag_enforced=_citations_enabled(),
        source_count=source_count,
    )
    rewrite_tool_decision, _ = _resolve_stage_tools(
        services=services,
        stage="rewrite",
        topic=request_model.topic,
        outline=outline.outline,
        draft=draft,
        guidance=request_model.review_criteria,
        research_notes=notes_text,
        rag_enforced=_citations_enabled(),
        source_count=source_count,
    )
    return {
        "thread_id": thread_id,
        "mode": mode,
        "request": request_model.model_dump(),
        "resolved_session_id": resolved_session_id,
        "effective_constraints": effective_constraints,
        "github_context": github_context,
        "outline": outline.outline,
        "assumptions": outline.assumptions,
        "open_questions": outline.open_questions,
        "research_notes": list(research_notes),
        "notes_text": notes_text,
        "draft": draft,
        "review": review,
        "review_text": review,
        "needs_rewrite": needs_rewrite,
        "reason": review_reason,
        "score": review_score,
        "revised": revised,
        "revised_candidate": revised,
        "review_criteria": request_model.review_criteria,
        "audience": request_model.audience,
        "style": request_model.style,
        "target_length": request_model.target_length,
        "review_tool_profile_id": _runtime_tool_profile_id(services, review_tool_decision),
        "review_tool_allowed_tools": review_tool_decision.allowed_tools if review_tool_decision.enabled else [],
        "rewrite_tool_profile_id": _runtime_tool_profile_id(services, rewrite_tool_decision),
        "rewrite_tool_allowed_tools": rewrite_tool_decision.allowed_tools if rewrite_tool_decision.enabled else [],
        "start_stage": start_stage,
        "source_count": source_count,
        "task_status": {
            "plan": True,
            "research": None,
            "draft": None,
            "review": None,
            "rewrite": None,
            "citations": None,
        },
    }


def _apply_resume_stage_to_full_input(full_input: dict[str, Any], resume_stage: str) -> dict[str, Any]:
    task_status = dict(full_input.get("task_status") or {})
    if resume_stage in {"research_done", "draft_done", "review_done", "rewrite_done"}:
        task_status["research"] = None if not int(full_input.get("source_count") or 0) else bool(
            full_input.get("research_notes")
        )
    if resume_stage in {"draft_done", "review_done", "rewrite_done"}:
        task_status["draft"] = bool(str(full_input.get("draft") or "").strip())
    if resume_stage in {"review_done", "rewrite_done"}:
        task_status["review"] = bool(str(full_input.get("review") or "").strip())
    if resume_stage == "rewrite_done":
        task_status["rewrite"] = bool(
            str(full_input.get("revised_candidate") or full_input.get("revised") or "").strip()
        )
    return {
        **full_input,
        "start_stage": resume_stage,
        "task_status": task_status,
    }


def _extract_graph_interrupt(raw_result: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    interrupts = raw_result.get("__interrupt__") if isinstance(raw_result, dict) else None
    if not interrupts:
        return None
    payload = dict(getattr(interrupts[0], "value", {}) or {})
    kind = str(payload.pop("kind", payload.get("interrupt_stage") or "draft_review"))
    return kind, payload


def _task_status_from_full_state(full_state: dict[str, Any]) -> dict[str, bool | None]:
    task_status = dict(full_state.get("task_status") or {})
    if not task_status:
        task_status = {
            "plan": True,
            "research": None,
            "draft": None,
            "review": None,
            "rewrite": None,
            "citations": None,
        }
    return task_status


@router.post("/pipeline/v2", response_model=PipelineV2Response)
def run_pipeline_v2(
    payload: PipelineV2Request,
    services: AppServices = Depends(get_services),
) -> PipelineV2Response:
    thread_id = (payload.thread_id or "").strip() or str(uuid4())
    resolved_session_id = (payload.session_id or "").strip() or thread_id
    t0 = time.perf_counter()
    _log_v2_stage("start", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
    try:
        github_context = (
            maybe_fetch_github_context(
                services.github_mcp_tool,
                query=payload.topic,
                context="\n".join(
                    [
                        payload.constraints,
                        payload.key_points,
                        payload.style,
                        payload.audience,
                        payload.review_criteria,
                    ]
                ),
            )
            if _mcp_enabled()
            else ""
        )
        effective_constraints = _build_effective_constraints(payload, github_context)
        source_count_for_policy = len(payload.sources or []) + (1 if github_context else 0)
        plan_tool_decision, _ = _resolve_stage_tools(
            services=services,
            stage="plan",
            topic=payload.topic,
            constraints=effective_constraints,
            research_notes=payload.key_points,
            rag_enforced=_citations_enabled(),
            source_count=source_count_for_policy,
        )
        upsert_pipeline_v2_checkpoint(
            thread_id=thread_id,
            session_id=resolved_session_id,
            mode="sync",
            status="running",
            current_stage="plan",
        )
        workflow_input = _build_workflow_input(
            payload=payload,
            thread_id=thread_id,
            resolved_session_id=resolved_session_id,
            github_context=github_context,
            effective_constraints=effective_constraints,
            plan_tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
            plan_tool_allowed_tools=plan_tool_decision.allowed_tools if plan_tool_decision.enabled else [],
        )
        raw_result = start_pipeline_v2_workflow(workflow_input, thread_id=thread_id)
        interrupts = raw_result.get("__interrupt__") if isinstance(raw_result, dict) else None
        if not interrupts:
            mark_pipeline_v2_checkpoint_failed(
                thread_id=thread_id,
                current_stage="plan",
                last_error="Pipeline v2 did not interrupt as expected",
            )
            raise HTTPException(status_code=500, detail="Pipeline v2 did not interrupt as expected")

        interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
        kind = str(interrupt_payload.pop("kind", "outline_review"))
        upsert_pipeline_v2_checkpoint(
            thread_id=thread_id,
            session_id=resolved_session_id,
            mode="sync",
            status="interrupted",
            current_stage="outline_review",
            outline_preview=str(interrupt_payload.get("outline") or ""),
        )
        _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
        return PipelineV2Response(
            status="interrupted",
            thread_id=thread_id,
            interrupt=PipelineV2Interrupt(kind=kind, payload=interrupt_payload),
        )
    except HTTPException:
        raise
    except Exception as exc:
        mark_pipeline_v2_checkpoint_failed(
            thread_id=thread_id,
            current_stage="plan",
            last_error=str(exc),
        )
        logger.exception(
            "pipeline v2 route: stage=failed thread_id=%s session_id=%s elapsed_ms=%.2f",
            thread_id,
            resolved_session_id,
            (time.perf_counter() - t0) * 1000,
        )
        raise HTTPException(status_code=500, detail="Pipeline v2 execution failed") from exc

@router.post("/pipeline/v2/resume", response_model=PipelineV2Response)
def resume_pipeline_v2(
    payload: PipelineV2ResumeRequest,
    services: AppServices = Depends(get_services),
) -> PipelineV2Response:
    thread_id = payload.thread_id.strip()
    resume_checkpoint, restored_session_id = _extract_resume_context(thread_id)
    checkpoint_detail = get_pipeline_v2_checkpoint_detail(thread_id)
    t0 = time.perf_counter()
    _log_v2_stage(
        "resume_started",
        thread_id=thread_id,
        resolved_session_id=restored_session_id,
        started_at=t0,
    )
    try:
        if has_pipeline_v2_interrupt_checkpoint(thread_id):
            raw_result = resume_pipeline_v2_workflow(
                thread_id=thread_id,
                outline_override=payload.outline_override,
            )
            request_model = _request_from_dict(dict(raw_result.get("request") or {}))
            resolved_session_id = str(raw_result.get("resolved_session_id") or "").strip() or thread_id
            outline = _outline_from_data(raw_result)
            github_context = str(raw_result.get("github_context") or "")
            effective_constraints = str(raw_result.get("effective_constraints") or request_model.constraints or "")
            full_state = run_pipeline_v2_full_sync(
                _build_full_input(
                    thread_id=thread_id,
                    mode="sync",
                    request_model=request_model,
                    resolved_session_id=resolved_session_id,
                    effective_constraints=effective_constraints,
                    github_context=github_context,
                    outline=outline,
                    research_notes=[],
                    notes_text="",
                    draft="",
                    start_stage="outline_accepted",
                    source_count=len(_build_sources(request_model, github_context)),
                    services=services,
                )
            )
            graph_interrupt = _extract_graph_interrupt(full_state)
            if graph_interrupt is not None:
                kind, interrupt_payload = graph_interrupt
                upsert_pipeline_v2_checkpoint(
                    thread_id=thread_id,
                    session_id=resolved_session_id,
                    mode="sync",
                    status="interrupted",
                    current_stage=(
                        "review_confirmation" if kind == "review_confirmation" else "draft_review"
                    ),
                    outline_preview=str(interrupt_payload.get("outline") or ""),
                )
                _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                return _interrupted_v2_response(thread_id, kind=kind, payload=interrupt_payload)
            response = PipelineResponse.model_validate(full_state.get("response") or {})
            task_status = _task_status_from_full_state(full_state)
            _log_task_success_rate(
                task_status,
                terminate_ok=True,
                effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
            )
            _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            return _completed_v2_response(thread_id, response.model_dump())

        if not resume_checkpoint:
            _log_v2_stage("unknown_thread_id", thread_id=thread_id, started_at=t0)
            raise HTTPException(status_code=404, detail="Unknown thread_id")

        if str(resume_checkpoint.get("status") or "") == "completed":
            _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=restored_session_id, started_at=t0)
            return _completed_response_from_checkpoint(thread_id, resume_checkpoint)

        if checkpoint_detail and checkpoint_detail.get("status") == "interrupted" and checkpoint_detail.get("current_stage") in {"draft_review", "review_confirmation"}:
            checkpoint_payload = dict((resume_checkpoint or {}).get("payload") or {})
            full_state = resume_pipeline_v2_full_workflow(
                thread_id=thread_id,
                draft_override=payload.draft_override if checkpoint_detail.get("current_stage") == "draft_review" else "",
            )
            graph_interrupt = _extract_graph_interrupt(full_state)
            if graph_interrupt is not None:
                kind, interrupt_payload = graph_interrupt
                _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=restored_session_id, started_at=t0)
                return _interrupted_v2_response(thread_id, kind=kind, payload=interrupt_payload)
            response = PipelineResponse.model_validate(full_state.get("response") or {})
            task_status = _task_status_from_full_state(full_state)
            _log_task_success_rate(
                task_status,
                terminate_ok=True,
                effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
            )
            resolved_session_id = str(
                checkpoint_payload.get("resolved_session_id") or restored_session_id or thread_id
            )
            _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            return _completed_v2_response(thread_id, response.model_dump())

        checkpoint_payload = dict(resume_checkpoint.get("payload") or {})
        request_model = _request_from_dict(dict(checkpoint_payload.get("request") or {}))
        resolved_session_id = str(checkpoint_payload.get("resolved_session_id") or restored_session_id or thread_id)
        outline = _outline_from_data(checkpoint_payload)
        github_context = str(checkpoint_payload.get("github_context") or "")
        effective_constraints = str(checkpoint_payload.get("effective_constraints") or request_model.constraints or "")
        sources = _build_sources(request_model, github_context)
        resume_stage = _resolve_resume_stage(resume_checkpoint)
        full_input = _apply_resume_stage_to_full_input(
            _build_full_input(
                thread_id=thread_id,
                mode="sync",
                request_model=request_model,
                resolved_session_id=resolved_session_id,
                effective_constraints=effective_constraints,
                github_context=github_context,
                outline=outline,
                research_notes=list(checkpoint_payload.get("research_notes") or []),
                notes_text=str(checkpoint_payload.get("notes_text") or ""),
                draft=str((payload.draft_override or "").strip() or checkpoint_payload.get("draft") or ""),
                review=str(checkpoint_payload.get("review") or ""),
                needs_rewrite=bool(checkpoint_payload.get("needs_rewrite")),
                review_reason=str(checkpoint_payload.get("reason") or ""),
                review_score=checkpoint_payload.get("score"),
                revised=str(checkpoint_payload.get("revised") or ""),
                start_stage=resume_stage,
                source_count=len(sources),
                services=services,
            ),
            resume_stage,
        )
        clear_pipeline_v2_graph_thread(thread_id)
        full_state = run_pipeline_v2_full_sync(full_input)
        graph_interrupt = _extract_graph_interrupt(full_state)
        if graph_interrupt is not None:
            kind, interrupt_payload = graph_interrupt
            _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            return _interrupted_v2_response(thread_id, kind=kind, payload=interrupt_payload)
        response = PipelineResponse.model_validate(full_state.get("response") or {})
        task_status = _task_status_from_full_state(full_state)
        _log_task_success_rate(
            task_status,
            terminate_ok=True,
            effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
        )
        _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
        return _completed_v2_response(thread_id, response.model_dump())
    except KeyError as exc:
        _log_v2_stage("unknown_thread_id", thread_id=thread_id, resolved_session_id=restored_session_id, started_at=t0)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        mark_pipeline_v2_checkpoint_failed(
            thread_id=thread_id,
            current_stage=str((resume_checkpoint or {}).get("current_stage") or "resume"),
            last_error=str(exc),
        )
        logger.exception(
            "pipeline v2 route: stage=resume_failed thread_id=%s session_id=%s elapsed_ms=%.2f",
            thread_id,
            restored_session_id or "__default__",
            (time.perf_counter() - t0) * 1000,
        )
        raise HTTPException(status_code=500, detail="Pipeline v2 resume failed") from exc


@router.post("/pipeline/v2/stream")
def run_pipeline_v2_stream(
    payload: PipelineV2Request,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    thread_id = (payload.thread_id or "").strip() or str(uuid4())
    resolved_session_id = (payload.session_id or "").strip() or thread_id
    q: Queue[dict[str, Any] | None] = Queue()

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            _log_v2_stage("start", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            github_context = (
                maybe_fetch_github_context(
                    services.github_mcp_tool,
                    query=payload.topic,
                    context="\n".join(
                        [
                            payload.constraints,
                            payload.key_points,
                            payload.style,
                            payload.audience,
                            payload.review_criteria,
                        ]
                    ),
                )
                if _mcp_enabled()
                else ""
            )
            effective_constraints = _build_effective_constraints(payload, github_context)
            source_count_for_policy = len(payload.sources or []) + (1 if github_context else 0)
            plan_tool_decision, plan_tool_registry = _resolve_stage_tools(
                services=services,
                stage="plan",
                topic=payload.topic,
                constraints=effective_constraints,
                research_notes=payload.key_points,
                rag_enforced=_citations_enabled(),
                source_count=source_count_for_policy,
            )
            upsert_pipeline_v2_checkpoint(
                thread_id=thread_id,
                session_id=resolved_session_id,
                mode="stream",
                status="running",
                current_stage="plan",
            )
            q.put({"type": "status", "step": "plan"})
            outline = services.pipeline.planner.plan_outline_stream(
                topic=payload.topic,
                audience=payload.audience,
                style=payload.style,
                target_length=payload.target_length,
                constraints=effective_constraints,
                key_points=payload.key_points,
                max_tokens=_pipeline_max_tokens("plan"),
                max_input_chars=_pipeline_max_input_chars("plan"),
                session_id=resolved_session_id,
                tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, plan_tool_registry),
                on_chunk=lambda chunk: q.put({"type": "delta", "stage": "plan", "content": chunk}),
            )
            q.put(
                {
                    "type": "outline",
                    "payload": {
                        "outline": outline.outline,
                        "assumptions": outline.assumptions,
                        "open_questions": outline.open_questions,
                    },
                }
            )
            workflow_input = _build_workflow_input(
                payload=payload,
                thread_id=thread_id,
                resolved_session_id=resolved_session_id,
                github_context=github_context,
                effective_constraints=effective_constraints,
                plan_tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
                plan_tool_allowed_tools=plan_tool_decision.allowed_tools if plan_tool_decision.enabled else [],
                prefetched_plan_result={
                    "outline": outline.outline,
                    "assumptions": outline.assumptions,
                    "open_questions": outline.open_questions,
                },
            )
            raw_result = start_pipeline_v2_workflow(workflow_input, thread_id=thread_id)
            interrupts = raw_result.get("__interrupt__") if isinstance(raw_result, dict) else None
            if not interrupts:
                mark_pipeline_v2_checkpoint_failed(
                    thread_id=thread_id,
                    current_stage="plan",
                    last_error="Pipeline v2 stream did not interrupt as expected",
                )
                raise RuntimeError("Pipeline v2 stream did not interrupt as expected")
            interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
            interrupt_payload.setdefault("thread_id", thread_id)
            upsert_pipeline_v2_checkpoint(
                thread_id=thread_id,
                session_id=resolved_session_id,
                mode="stream",
                status="interrupted",
                current_stage="outline_review",
                outline_preview=str(interrupt_payload.get("outline") or ""),
            )
            q.put({"type": "interrupt", "kind": "outline_review", "payload": interrupt_payload})
            _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            q.put({"type": "done", "ok": True})
        except Exception as exc:
            mark_pipeline_v2_checkpoint_failed(
                thread_id=thread_id,
                current_stage="plan",
                last_error=str(exc),
            )
            logger.exception(
                "pipeline v2 route: stage=failed thread_id=%s session_id=%s elapsed_ms=%.2f",
                thread_id,
                resolved_session_id,
                (time.perf_counter() - t0) * 1000,
            )
            q.put({"type": "error", "detail": str(exc), "stage": "plan"})
            q.put({"type": "done", "ok": False})
        finally:
            q.put(None)

    def generator():
        yield _event({"type": "status", "message": "started"})
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                item = q.get(timeout=2)
            except Empty:
                yield _event({"type": "ping", "ts": int(time.time())})
                continue
            if item is None:
                break
            yield _event(item)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/pipeline/v2/resume/stream")
def resume_pipeline_v2_stream(
    payload: PipelineV2ResumeRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    thread_id = payload.thread_id.strip()
    q: Queue[dict[str, Any] | None] = Queue()

    def worker() -> None:
        resume_checkpoint, restored_session_id = _extract_resume_context(thread_id)
        checkpoint_detail = get_pipeline_v2_checkpoint_detail(thread_id)
        t0 = time.perf_counter()
        try:
            _log_v2_stage(
                "resume_started",
                thread_id=thread_id,
                resolved_session_id=restored_session_id,
                started_at=t0,
            )
            if has_pipeline_v2_interrupt_checkpoint(thread_id):
                raw_result = resume_pipeline_v2_workflow(
                    thread_id=thread_id,
                    outline_override=payload.outline_override,
                )
                request_model = _request_from_dict(dict(raw_result.get("request") or {}))
                resolved_session_id = str(raw_result.get("resolved_session_id") or "").strip() or thread_id
                outline = _outline_from_data(raw_result)
                github_context = str(raw_result.get("github_context") or "")
                effective_constraints = str(raw_result.get("effective_constraints") or request_model.constraints or "")
                sources = _build_sources(request_model, github_context)
                q.put(
                    {
                        "type": "outline",
                        "payload": {
                            "outline": outline.outline,
                            "assumptions": outline.assumptions,
                            "open_questions": outline.open_questions,
                            "thread_id": thread_id,
                        },
                    }
                )
                full_state = run_pipeline_v2_full_stream(
                    q=q,
                    full_input=_build_full_input(
                        thread_id=thread_id,
                        mode="stream",
                        request_model=request_model,
                        resolved_session_id=resolved_session_id,
                        effective_constraints=effective_constraints,
                        github_context=github_context,
                        outline=outline,
                        research_notes=[],
                        notes_text="",
                        draft="",
                        start_stage="outline_accepted",
                        source_count=len(sources),
                        services=services,
                    ),
                )
                if full_state.get("interrupted"):
                    _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                    q.put({"type": "done", "ok": True})
                    return
                task_status = _task_status_from_full_state(full_state)
                _log_task_success_rate(
                    task_status,
                    terminate_ok=True,
                    effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
                )
                _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                q.put({"type": "done", "ok": True})
                return

            if not resume_checkpoint:
                _log_v2_stage("unknown_thread_id", thread_id=thread_id, started_at=t0)
                q.put({"type": "error", "detail": "Unknown thread_id", "stage": "resume"})
                q.put({"type": "done", "ok": False})
                return

            if str(resume_checkpoint.get("status") or "") == "completed":
                response = _completed_response_from_checkpoint(thread_id, resume_checkpoint)
                q.put({"type": "result", "payload": response.result.model_dump() if response.result else {}})
                _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=restored_session_id, started_at=t0)
                q.put({"type": "done", "ok": True})
                return

            if checkpoint_detail and checkpoint_detail.get("status") == "interrupted" and checkpoint_detail.get("current_stage") in {"draft_review", "review_confirmation"}:
                checkpoint_payload = dict(resume_checkpoint.get("payload") or {})
                request_model = _request_from_dict(dict(checkpoint_payload.get("request") or {}))
                resolved_session_id = str(checkpoint_payload.get("resolved_session_id") or restored_session_id or thread_id)
                outline = _outline_from_data(checkpoint_payload)
                github_context = str(checkpoint_payload.get("github_context") or "")
                effective_constraints = str(checkpoint_payload.get("effective_constraints") or request_model.constraints or "")
                sources = _build_sources(request_model, github_context)
                q.put(
                    {
                        "type": "outline",
                        "payload": {
                            "outline": outline.outline,
                            "assumptions": outline.assumptions,
                            "open_questions": outline.open_questions,
                            "thread_id": thread_id,
                        },
                    }
                )
                q.put(
                    {
                        "type": "draft",
                        "payload": {
                            "draft": str((payload.draft_override or "").strip() or checkpoint_payload.get("draft") or ""),
                            "thread_id": thread_id,
                        },
                    }
                )
                full_state = run_pipeline_v2_full_stream(
                    q=q,
                    full_input=_apply_resume_stage_to_full_input(
                        _build_full_input(
                            thread_id=thread_id,
                            mode="stream",
                            request_model=request_model,
                            resolved_session_id=resolved_session_id,
                            effective_constraints=effective_constraints,
                            github_context=github_context,
                            outline=outline,
                            research_notes=list(checkpoint_payload.get("research_notes") or []),
                            notes_text=str(checkpoint_payload.get("notes_text") or ""),
                            draft=str(
                                (
                                    (payload.draft_override or "").strip()
                                    or checkpoint_payload.get("draft")
                                    or ""
                                )
                                if checkpoint_detail.get("current_stage") == "draft_review"
                                else checkpoint_payload.get("draft")
                                or ""
                            ),
                            review=str(checkpoint_payload.get("review") or ""),
                            needs_rewrite=bool(checkpoint_payload.get("needs_rewrite")),
                            review_reason=str(checkpoint_payload.get("reason") or ""),
                            review_score=checkpoint_payload.get("score"),
                            revised=str(checkpoint_payload.get("revised") or ""),
                            start_stage="draft_done"
                            if checkpoint_detail.get("current_stage") == "draft_review"
                            else "review_done",
                            source_count=len(sources),
                            services=services,
                        ),
                        "draft_done" if checkpoint_detail.get("current_stage") == "draft_review" else "review_done",
                    ),
                )
                if full_state.get("interrupted"):
                    _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                    q.put({"type": "done", "ok": True})
                    return
                task_status = _task_status_from_full_state(full_state)
                _log_task_success_rate(
                    task_status,
                    terminate_ok=True,
                    effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
                )
                _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                q.put({"type": "done", "ok": True})
                return

            checkpoint_payload = dict(resume_checkpoint.get("payload") or {})
            request_model = _request_from_dict(dict(checkpoint_payload.get("request") or {}))
            resolved_session_id = str(checkpoint_payload.get("resolved_session_id") or restored_session_id or thread_id)
            outline = _outline_from_data(checkpoint_payload)
            github_context = str(checkpoint_payload.get("github_context") or "")
            effective_constraints = str(checkpoint_payload.get("effective_constraints") or request_model.constraints or "")
            sources = _build_sources(request_model, github_context)
            resume_stage = _resolve_resume_stage(resume_checkpoint)
            q.put(
                {
                    "type": "outline",
                    "payload": {
                        "outline": outline.outline,
                        "assumptions": outline.assumptions,
                        "open_questions": outline.open_questions,
                        "thread_id": thread_id,
                    },
                }
            )
            clear_pipeline_v2_graph_thread(thread_id)
            full_state = run_pipeline_v2_full_stream(
                q=q,
                full_input=_apply_resume_stage_to_full_input(
                    _build_full_input(
                        thread_id=thread_id,
                        mode="stream",
                        request_model=request_model,
                        resolved_session_id=resolved_session_id,
                        effective_constraints=effective_constraints,
                        github_context=github_context,
                        outline=outline,
                        research_notes=list(checkpoint_payload.get("research_notes") or []),
                        notes_text=str(checkpoint_payload.get("notes_text") or ""),
                        draft=str((payload.draft_override or "").strip() or checkpoint_payload.get("draft") or ""),
                        review=str(checkpoint_payload.get("review") or ""),
                        needs_rewrite=bool(checkpoint_payload.get("needs_rewrite")),
                        review_reason=str(checkpoint_payload.get("reason") or ""),
                        review_score=checkpoint_payload.get("score"),
                        revised=str(checkpoint_payload.get("revised") or ""),
                        start_stage=resume_stage,
                        source_count=len(sources),
                        services=services,
                    ),
                    resume_stage,
                ),
            )
            if full_state.get("interrupted"):
                _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
                q.put({"type": "done", "ok": True})
                return
            task_status = _task_status_from_full_state(full_state)
            _log_task_success_rate(
                task_status,
                terminate_ok=True,
                effective_output=bool(task_status["rewrite"]) or bool(task_status["draft"]),
            )
            _log_v2_stage("completed", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            q.put({"type": "done", "ok": True})
        except Exception as exc:
            mark_pipeline_v2_checkpoint_failed(
                thread_id=thread_id,
                current_stage=str((resume_checkpoint or {}).get("current_stage") or "resume"),
                last_error=str(exc),
            )
            logger.exception(
                "pipeline v2 route: stage=resume_failed thread_id=%s session_id=%s elapsed_ms=%.2f",
                thread_id,
                restored_session_id or "__default__",
                (time.perf_counter() - t0) * 1000,
            )
            q.put({"type": "error", "detail": str(exc), "stage": "resume"})
            q.put({"type": "done", "ok": False})
        finally:
            q.put(None)

    def generator():
        yield _event({"type": "status", "message": "started"})
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                item = q.get(timeout=2)
            except Empty:
                yield _event({"type": "ping", "ts": int(time.time())})
                continue
            if item is None:
                break
            yield _event(item)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/pipeline/v2/checkpoints", response_model=PipelineV2CheckpointListResponse)
def list_pipeline_v2_checkpoint_route(
    limit: int = Query(20, ge=1, le=200),
    status: str = Query("all"),
    thread_id: str = Query(""),
) -> PipelineV2CheckpointListResponse:
    rows = list_pipeline_v2_checkpoints(limit=limit, status=status, thread_id=thread_id)
    return PipelineV2CheckpointListResponse(
        checkpoints=[
            PipelineV2CheckpointSummary(
                thread_id=str(row.get("thread_id") or ""),
                status=str(row.get("status") or ""),
                current_stage=str(row.get("current_stage") or ""),
                updated_at=str(row.get("updated_at") or ""),
            )
            for row in rows
        ]
    )


@router.get("/pipeline/v2/checkpoints/{thread_id}", response_model=PipelineV2CheckpointDetailResponse)
def get_pipeline_v2_checkpoint_route(thread_id: str) -> PipelineV2CheckpointDetailResponse:
    detail = get_pipeline_v2_checkpoint_detail(thread_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return PipelineV2CheckpointDetailResponse.model_validate(detail)


@router.delete("/pipeline/v2/checkpoints/{thread_id}", response_model=DeletePipelineV2CheckpointResponse)
def delete_pipeline_v2_checkpoint_route(thread_id: str) -> DeletePipelineV2CheckpointResponse:
    deleted = delete_pipeline_v2_checkpoint(thread_id)
    return DeletePipelineV2CheckpointResponse(deleted=deleted)


@router.post("/pipeline/v2/checkpoints/cleanup", response_model=PipelineV2CheckpointCleanupResponse)
def cleanup_pipeline_v2_checkpoint_route(
    payload: PipelineV2CheckpointCleanupRequest,
) -> PipelineV2CheckpointCleanupResponse:
    result = cleanup_pipeline_v2_checkpoints(
        older_than_hours=payload.older_than_hours,
        status=payload.status,
        dry_run=payload.dry_run,
        limit=payload.limit,
    )
    return PipelineV2CheckpointCleanupResponse.model_validate(result)
