from __future__ import annotations

import json
import logging
import time
from queue import Empty, Queue
from threading import Thread
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...models.schemas import (
    CoverageDetail,
    PipelineRequest,
    ResearchNoteResponse,
    PipelineV2Interrupt,
    PipelineV2Request,
    PipelineV2Response,
    PipelineV2ResumeRequest,
)
from ...services.drafting_service import DraftResult
from ...services.github_context import maybe_fetch_github_context
from ...services.pipeline_langgraph_v2 import (
    resume_pipeline_v2_workflow,
    start_pipeline_v2_workflow,
)
from ...services.pipeline_service import PipelineResult
from ...services.research_service import SourceDocument
from ..deps import AppServices, get_services
from .pipeline import (
    _append_block,
    _build_refusal_query,
    _citation_enforcer,
    _citations_enabled,
    _coverage_detail,
    _github_note,
    _is_effective,
    _log_task_success_rate,
    _mcp_enabled,
    _pipeline_max_input_chars,
    _pipeline_max_tokens,
    _postprocess_final_text,
    _refusal_check_with_fallback,
    _refusal_enabled,
    _refusal_message,
    _resolve_stage_tools,
    _runtime_tool_profile_id,
    _runtime_tool_registry,
    _same_source_ids,
    _serialize_pipeline_result,
    _strict_citation_labels,
    _strict_citation_postcheck_failed,
    _zero_coverage_report,
)
from ...services.planner_service import OutlinePlan

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


def _event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit_text_deltas(q: Queue[dict | None], stage: str, text: str) -> None:
    if not text:
        return
    chunk_size = 600
    for index in range(0, len(text), chunk_size):
        q.put({"type": "delta", "stage": stage, "content": text[index : index + chunk_size]})


def _build_workflow_input(
    *,
    payload: PipelineV2Request,
    services: AppServices,
    thread_id: str,
    resolved_session_id: str,
    github_context: str,
    effective_constraints: str,
    plan_tool_allowed_tools: list[str],
    plan_tool_profile_id: str | None,
    prefetched_plan_result: dict | None = None,
) -> dict[str, object]:
    workflow_input: dict[str, object] = {
        "thread_id": thread_id,
        "resolved_session_id": resolved_session_id,
        "request": PipelineRequest.model_validate(payload.model_dump()).model_dump(),
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
        effective_constraints = _append_block(payload.constraints, github_context, "GitHub参考")
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
        workflow_input = _build_workflow_input(
            payload=payload,
            services=services,
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
            logger.error(
                "pipeline v2 route: stage=failed thread_id=%s session_id=%s reason=no_interrupt elapsed_ms=%.2f",
                thread_id,
                resolved_session_id,
                (time.perf_counter() - t0) * 1000,
            )
            raise HTTPException(status_code=500, detail="Pipeline v2 did not interrupt as expected")

        interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
        kind = str(interrupt_payload.pop("kind", "outline_review"))
        _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
        return PipelineV2Response(
            status="interrupted",
            thread_id=thread_id,
            interrupt=PipelineV2Interrupt(kind=kind, payload=interrupt_payload),
        )
    except HTTPException:
        raise
    except Exception as exc:
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
    task_status: dict[str, bool | None] = {
        "plan": None,
        "research": None,
        "draft": None,
        "review": None,
        "rewrite": None,
        "citations": None,
    }
    succeeded = False
    t0 = time.perf_counter()
    try:
        _log_v2_stage("resume_started", thread_id=payload.thread_id, started_at=t0)
        try:
            workflow_result = resume_pipeline_v2_workflow(
                thread_id=payload.thread_id,
                outline_override=payload.outline_override,
            )
        except KeyError as exc:
            logger.warning(
                "pipeline v2 route: stage=unknown_thread_id thread_id=%s elapsed_ms=%.2f",
                payload.thread_id,
                (time.perf_counter() - t0) * 1000,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        interrupts = workflow_result.get("__interrupt__") if isinstance(workflow_result, dict) else None
        if interrupts:
            interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
            kind = str(interrupt_payload.pop("kind", "outline_review"))
            _log_v2_stage("interrupted", thread_id=payload.thread_id, started_at=t0)
            return PipelineV2Response(
                status="interrupted",
                thread_id=payload.thread_id,
                interrupt=PipelineV2Interrupt(kind=kind, payload=interrupt_payload),
            )

        request_model = PipelineRequest.model_validate(workflow_result.get("request") or {})
        resolved_session_id = str(workflow_result.get("resolved_session_id") or request_model.session_id or payload.thread_id)
        github_context = str(workflow_result.get("github_context") or "")
        effective_constraints = str(workflow_result.get("effective_constraints") or request_model.constraints or "")
        outline = OutlinePlan(
            outline=str(workflow_result.get("outline") or ""),
            assumptions=str(workflow_result.get("assumptions") or ""),
            open_questions=str(workflow_result.get("open_questions") or ""),
        )
        task_status["plan"] = bool(outline.outline.strip())

        sources = [
            SourceDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                url=doc.url,
            )
            for doc in request_model.sources
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

        refusal_query = _build_refusal_query(
            topic=request_model.topic,
            key_points=request_model.key_points,
            constraints=request_model.constraints,
            audience=request_model.audience,
            style=request_model.style,
            outline=outline.outline,
        )
        effective_sources = list(sources)
        if _refusal_enabled() and effective_sources:
            refused, effective_sources = _refusal_check_with_fallback(
                query=refusal_query,
                sources=effective_sources,
                services=services,
            )
        else:
            refused = False

        notes = (
            services.pipeline.collect_research_notes(request_model.topic, outline, effective_sources)
            if effective_sources
            else []
        )
        if github_context and not any(note.doc_id == "github:mcp" for note in notes):
            notes.append(_github_note(github_context))
        task_status["research"] = None if not effective_sources else bool(notes)
        notes_text = services.pipeline.researcher.format_notes(notes)
        apply_labels = _citations_enabled()

        draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
            services=services,
            stage="draft",
            topic=request_model.topic,
            outline=outline.outline,
            constraints=effective_constraints,
            research_notes=notes_text,
            rag_enforced=apply_labels,
            source_count=len(effective_sources),
        )
        review_tool_decision, review_tool_registry = _resolve_stage_tools(
            services=services,
            stage="review",
            topic=request_model.topic,
            draft=notes_text,
            guidance=request_model.review_criteria,
            constraints=effective_constraints,
            rag_enforced=apply_labels,
            source_count=len(effective_sources),
        )
        rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
            services=services,
            stage="rewrite",
            topic=request_model.topic,
            draft=notes_text,
            guidance=request_model.review_criteria,
            constraints=effective_constraints,
            rag_enforced=apply_labels,
            source_count=len(effective_sources),
        )

        if apply_labels and refused:
            refusal_text = _refusal_message()
            draft_result = DraftResult(
                outline=outline.outline,
                research_notes=notes_text,
                draft=refusal_text,
                review=refusal_text,
                revised=refusal_text,
            )
            result = PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)
            task_status["draft"] = False
            task_status["review"] = False
            task_status["rewrite"] = False
            task_status["citations"] = False
            succeeded = True
            return PipelineV2Response(
                status="completed",
                thread_id=payload.thread_id,
                result=_serialize_pipeline_result(
                    request_model,
                    result,
                    services,
                    coverage=0.0,
                    coverage_detail=CoverageDetail(),
                    citation_enforced=False,
                ),
            )

        draft_result = services.drafter.run_full(
            topic=request_model.topic,
            outline=outline.outline,
            research_notes=notes_text,
            constraints=effective_constraints,
            style=request_model.style,
            target_length=request_model.target_length,
            review_criteria=request_model.review_criteria,
            audience=request_model.audience,
            draft_max_tokens=_pipeline_max_tokens("draft"),
            review_max_tokens=_pipeline_max_tokens("review"),
            rewrite_max_tokens=_pipeline_max_tokens("rewrite"),
            draft_max_input_chars=_pipeline_max_input_chars("draft"),
            review_max_input_chars=_pipeline_max_input_chars("review"),
            rewrite_max_input_chars=_pipeline_max_input_chars("rewrite"),
            session_id=resolved_session_id,
            draft_tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
            draft_tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
            review_tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
            review_tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
            rewrite_tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
            rewrite_tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
        )
        result = PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)

        if apply_labels and not _same_source_ids(sources, effective_sources):
            refreshed_notes = services.pipeline.collect_research_notes(
                request_model.topic,
                outline,
                effective_sources,
            )
            if github_context and not any(note.doc_id == "github:mcp" for note in refreshed_notes):
                refreshed_notes.append(_github_note(github_context))
            refreshed_notes_text = services.pipeline.researcher.format_notes(refreshed_notes)
            refreshed_draft = services.drafter.run_full(
                topic=request_model.topic,
                outline=outline.outline,
                research_notes=refreshed_notes_text,
                constraints=effective_constraints,
                style=request_model.style,
                target_length=request_model.target_length,
                review_criteria=request_model.review_criteria,
                audience=request_model.audience,
                draft_max_tokens=_pipeline_max_tokens("draft"),
                review_max_tokens=_pipeline_max_tokens("review"),
                rewrite_max_tokens=_pipeline_max_tokens("rewrite"),
                draft_max_input_chars=_pipeline_max_input_chars("draft"),
                review_max_input_chars=_pipeline_max_input_chars("review"),
                rewrite_max_input_chars=_pipeline_max_input_chars("rewrite"),
                session_id=resolved_session_id,
                draft_tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                draft_tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
                review_tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                review_tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
                rewrite_tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                rewrite_tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
            )
            result = PipelineResult(
                outline=outline,
                research_notes=refreshed_notes,
                draft_result=refreshed_draft,
            )
        elif github_context and not any(note.doc_id == "github:mcp" for note in result.research_notes):
            result.research_notes.append(_github_note(github_context))

        enforced_text, report = _citation_enforcer.enforce(
            result.draft_result.revised or result.draft_result.draft,
            result.research_notes,
            apply_labels=apply_labels,
            strict_labels=_strict_citation_labels(),
            embedder=services.rag.get_embedder(),
        )
        if apply_labels and _strict_citation_postcheck_failed(report):
            enforced_text = _refusal_message()
            report = _zero_coverage_report()
        enforced_text = _postprocess_final_text(enforced_text)
        citation_enforced = apply_labels and enforced_text != result.draft_result.revised
        final_result = PipelineResult(
            outline=result.outline,
            research_notes=result.research_notes,
            draft_result=DraftResult(
                outline=result.draft_result.outline,
                research_notes=result.draft_result.research_notes,
                draft=result.draft_result.draft,
                review=result.draft_result.review,
                revised=enforced_text,
            ),
        )
        task_status["draft"] = _is_effective(final_result.draft_result.draft)
        task_status["review"] = bool(final_result.draft_result.review.strip())
        task_status["rewrite"] = _is_effective(final_result.draft_result.revised)
        task_status["citations"] = True
        succeeded = True

        response = _serialize_pipeline_result(
            request_model,
            final_result,
            services,
            coverage=report.coverage,
            coverage_detail=_coverage_detail(report),
            citation_enforced=citation_enforced,
        )
        _log_v2_stage(
            "completed",
            thread_id=payload.thread_id,
            resolved_session_id=resolved_session_id,
            started_at=t0,
        )
        return PipelineV2Response(
            status="completed",
            thread_id=payload.thread_id,
            result=response,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "pipeline v2 route: stage=resume_failed thread_id=%s elapsed_ms=%.2f",
            payload.thread_id,
            (time.perf_counter() - t0) * 1000,
        )
        raise HTTPException(status_code=500, detail="Pipeline v2 execution failed") from exc
    finally:
        if any(v is not None for v in task_status.values()):
            effective = (
                bool(task_status["rewrite"]) or bool(task_status["draft"])
                if task_status["draft"] is not None or task_status["rewrite"] is not None
                else None
            )
            _log_task_success_rate(
                task_status,
                terminate_ok=succeeded,
                effective_output=effective if succeeded else None,
            )


@router.post("/pipeline/v2/stream")
def run_pipeline_v2_stream(
    payload: PipelineV2Request,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    q: Queue[dict | None] = Queue()

    def worker() -> None:
        thread_id = (payload.thread_id or "").strip() or str(uuid4())
        resolved_session_id = (payload.session_id or "").strip() or thread_id
        t0 = time.perf_counter()
        _log_v2_stage("start_stream", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
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
            effective_constraints = _append_block(payload.constraints, github_context, "GitHub参考")
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

            q.put({"type": "status", "step": "plan"})
            outline = services.planner.plan_outline_stream(
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
                services=services,
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
                raise RuntimeError("Pipeline v2 stream did not interrupt as expected")

            interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
            kind = str(interrupt_payload.pop("kind", "outline_review"))
            _log_v2_stage("interrupted", thread_id=thread_id, resolved_session_id=resolved_session_id, started_at=t0)
            q.put(
                {
                    "type": "interrupt",
                    "payload": {
                        "kind": kind,
                        "thread_id": thread_id,
                        **interrupt_payload,
                    },
                }
            )
            q.put({"type": "done", "ok": True, "interrupted": True})
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "pipeline v2 route: stage=stream_failed thread_id=%s session_id=%s elapsed_ms=%.2f",
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
    q: Queue[dict | None] = Queue()

    def worker() -> None:
        task_status: dict[str, bool | None] = {
            "plan": None,
            "research": None,
            "draft": None,
            "review": None,
            "rewrite": None,
            "citations": None,
        }
        succeeded = False
        effective_output: bool | None = None
        resolved_session_id = ""
        t0 = time.perf_counter()

        try:
            _log_v2_stage("resume_stream_started", thread_id=payload.thread_id, started_at=t0)
            try:
                workflow_result = resume_pipeline_v2_workflow(
                    thread_id=payload.thread_id,
                    outline_override=payload.outline_override,
                )
            except KeyError as exc:
                logger.warning(
                    "pipeline v2 route: stage=unknown_thread_id thread_id=%s elapsed_ms=%.2f",
                    payload.thread_id,
                    (time.perf_counter() - t0) * 1000,
                )
                q.put({"type": "error", "detail": str(exc), "stage": "resume"})
                q.put({"type": "done", "ok": False})
                return

            interrupts = workflow_result.get("__interrupt__") if isinstance(workflow_result, dict) else None
            if interrupts:
                interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
                kind = str(interrupt_payload.pop("kind", "outline_review"))
                _log_v2_stage("interrupted", thread_id=payload.thread_id, started_at=t0)
                q.put(
                    {
                        "type": "interrupt",
                        "payload": {
                            "kind": kind,
                            "thread_id": payload.thread_id,
                            **interrupt_payload,
                        },
                    }
                )
                q.put({"type": "done", "ok": True, "interrupted": True})
                return

            request_model = PipelineRequest.model_validate(workflow_result.get("request") or {})
            resolved_session_id = str(
                workflow_result.get("resolved_session_id") or request_model.session_id or payload.thread_id
            )
            github_context = str(workflow_result.get("github_context") or "")
            effective_constraints = str(
                workflow_result.get("effective_constraints") or request_model.constraints or ""
            )
            outline = OutlinePlan(
                outline=str(workflow_result.get("outline") or ""),
                assumptions=str(workflow_result.get("assumptions") or ""),
                open_questions=str(workflow_result.get("open_questions") or ""),
            )
            task_status["plan"] = bool(outline.outline.strip())
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

            sources = [
                SourceDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                )
                for doc in request_model.sources
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

            refusal_query = _build_refusal_query(
                topic=request_model.topic,
                key_points=request_model.key_points,
                constraints=request_model.constraints,
                audience=request_model.audience,
                style=request_model.style,
                outline=outline.outline,
            )
            effective_sources = list(sources)
            if _refusal_enabled() and effective_sources:
                refused, effective_sources = _refusal_check_with_fallback(
                    query=refusal_query,
                    sources=effective_sources,
                    services=services,
                )
            else:
                refused = False

            q.put({"type": "status", "step": "research"})
            notes = (
                services.pipeline.collect_research_notes(request_model.topic, outline, effective_sources)
                if effective_sources
                else []
            )
            if github_context and not any(note.doc_id == "github:mcp" for note in notes):
                notes.append(_github_note(github_context))
            task_status["research"] = None if not effective_sources else bool(notes)
            notes_text = services.pipeline.researcher.format_notes(notes)
            q.put(
                {
                    "type": "research",
                    "payload": {
                        "notes": [
                            ResearchNoteResponse(
                                doc_id=note.doc_id,
                                title=note.title,
                                summary=note.summary,
                                url=note.url,
                            ).model_dump()
                            for note in notes
                        ],
                        "notes_text": notes_text,
                    },
                }
            )

            apply_labels = _citations_enabled()
            draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
                services=services,
                stage="draft",
                topic=request_model.topic,
                outline=outline.outline,
                constraints=effective_constraints,
                research_notes=notes_text,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )
            review_tool_decision, review_tool_registry = _resolve_stage_tools(
                services=services,
                stage="review",
                topic=request_model.topic,
                draft=notes_text,
                guidance=request_model.review_criteria,
                constraints=effective_constraints,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )
            rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
                services=services,
                stage="rewrite",
                topic=request_model.topic,
                draft=notes_text,
                guidance=request_model.review_criteria,
                constraints=effective_constraints,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )

            if apply_labels and refused:
                refusal_text = _refusal_message()
                task_status["draft"] = False
                task_status["review"] = False
                task_status["rewrite"] = False
                task_status["citations"] = False
                q.put({"type": "draft", "payload": {"draft": refusal_text}})
                q.put({"type": "review", "payload": {"review": refusal_text}})
                q.put(
                    {
                        "type": "rewrite",
                        "payload": {
                            "revised": refusal_text,
                            "final": True,
                            "coverage": 0.0,
                            "coverage_detail": CoverageDetail().model_dump(),
                        },
                    }
                )
                draft_result = DraftResult(
                    outline=outline.outline,
                    research_notes=notes_text,
                    draft=refusal_text,
                    review=refusal_text,
                    revised=refusal_text,
                )
                result = PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)
                response = _serialize_pipeline_result(
                    request_model,
                    result,
                    services,
                    coverage=0.0,
                    coverage_detail=CoverageDetail(),
                    citation_enforced=False,
                )
                succeeded = True
                effective_output = False
                q.put({"type": "result", "payload": response.model_dump()})
                q.put({"type": "done", "ok": True})
                return

            q.put({"type": "status", "step": "draft"})
            draft_chunks: list[str] = []
            evidence_text = services.drafter.extract_evidence(notes_text)
            draft_constraints = services.drafter.build_constraints(
                research_notes=notes_text,
                constraints=effective_constraints,
                evidence_text=evidence_text,
            )
            for chunk in services.drafter.writing_agent.draft_stream(
                topic=request_model.topic,
                outline=outline.outline,
                constraints=draft_constraints,
                style=request_model.style,
                target_length=request_model.target_length,
                max_tokens=_pipeline_max_tokens("draft"),
                max_input_chars=_pipeline_max_input_chars("draft"),
                session_id=resolved_session_id,
                tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
            ):
                if not chunk:
                    continue
                draft_chunks.append(chunk)
                q.put({"type": "delta", "stage": "draft", "content": chunk})
            draft = "".join(draft_chunks)
            if not draft.strip():
                draft = services.drafter.create_draft(
                    topic=request_model.topic,
                    outline=outline.outline,
                    research_notes=notes_text,
                    constraints=effective_constraints,
                    style=request_model.style,
                    target_length=request_model.target_length,
                    max_tokens=_pipeline_max_tokens("draft"),
                    max_input_chars=_pipeline_max_input_chars("draft"),
                    session_id=resolved_session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
                )
                _emit_text_deltas(q, "draft", draft)
            task_status["draft"] = _is_effective(draft)
            q.put({"type": "draft", "payload": {"draft": draft}})

            q.put({"type": "status", "step": "review"})
            review_chunks: list[str] = []
            review_max_tokens = min(8000, max(2000, int(len(draft) * 0.8)))
            for chunk in services.reviewer.agent.review_stream(
                draft=draft,
                criteria=request_model.review_criteria,
                sources=notes_text,
                audience=request_model.audience,
                max_tokens=review_max_tokens,
                max_input_chars=_pipeline_max_input_chars("review"),
                session_id=resolved_session_id,
                tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
            ):
                if not chunk:
                    continue
                review_chunks.append(chunk)
                q.put({"type": "delta", "stage": "review", "content": chunk})
            review = "".join(review_chunks)
            if not review.strip():
                review = services.reviewer.review(
                    draft=draft,
                    criteria=request_model.review_criteria,
                    sources=notes_text,
                    audience=request_model.audience,
                    max_tokens=review_max_tokens,
                    max_input_chars=_pipeline_max_input_chars("review"),
                    session_id=resolved_session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
                ).review
                _emit_text_deltas(q, "review", review)
            task_status["review"] = bool(review.strip())
            q.put({"type": "review", "payload": {"review": review}})

            q.put({"type": "status", "step": "rewrite"})
            rewrite_chunks: list[str] = []
            rewrite_max_tokens = min(8000, max(2000, int(len(draft) * 1.5)))
            rewrite_guidance_parts = [
                part.strip()
                for part in [review, request_model.review_criteria]
                if part and part.strip()
            ]
            rewrite_guidance = "\n\n".join(dict.fromkeys(rewrite_guidance_parts))
            if evidence_text:
                rewrite_guidance = (
                    rewrite_guidance + "\n\nOnly use the evidence below:\n" + evidence_text
                ).strip()
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=draft,
                guidance=rewrite_guidance,
                style=request_model.style,
                target_length=request_model.target_length,
                max_tokens=rewrite_max_tokens,
                max_input_chars=_pipeline_max_input_chars("rewrite"),
                session_id=resolved_session_id,
                tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
            ):
                if not chunk:
                    continue
                rewrite_chunks.append(chunk)
                q.put({"type": "delta", "stage": "rewrite", "content": chunk})
            revised = "".join(rewrite_chunks)
            if not revised.strip():
                revised = services.rewriter.rewrite(
                    draft=draft,
                    guidance=rewrite_guidance,
                    style=request_model.style,
                    target_length=request_model.target_length,
                    max_tokens=rewrite_max_tokens,
                    max_input_chars=_pipeline_max_input_chars("rewrite"),
                    session_id=resolved_session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
                ).revised
                _emit_text_deltas(q, "rewrite", revised)

            enforced_text, report = _citation_enforcer.enforce(
                revised or draft,
                notes,
                apply_labels=apply_labels,
                strict_labels=_strict_citation_labels(),
                embedder=services.rag.get_embedder(),
            )
            if apply_labels and _strict_citation_postcheck_failed(report):
                enforced_text = _refusal_message()
                report = _zero_coverage_report()
            enforced_text = _postprocess_final_text(enforced_text)
            citation_enforced = apply_labels and enforced_text != revised
            revised = enforced_text
            task_status["rewrite"] = _is_effective(revised)
            q.put(
                {
                    "type": "rewrite",
                    "payload": {
                        "revised": revised,
                        "final": True,
                        "coverage": report.coverage,
                        "coverage_detail": _coverage_detail(report).model_dump(),
                    },
                }
            )

            q.put({"type": "status", "step": "citations"})
            draft_result = DraftResult(
                outline=outline.outline,
                research_notes=notes_text,
                draft=draft,
                review=review,
                revised=revised,
            )
            result = PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)
            response = _serialize_pipeline_result(
                request_model,
                result,
                services,
                coverage=report.coverage,
                coverage_detail=_coverage_detail(report),
                citation_enforced=citation_enforced,
            )
            task_status["citations"] = True
            succeeded = True
            effective_output = bool(task_status["rewrite"]) or bool(task_status["draft"])
            _log_v2_stage(
                "completed_stream",
                thread_id=payload.thread_id,
                resolved_session_id=resolved_session_id,
                started_at=t0,
            )
            q.put({"type": "result", "payload": response.model_dump()})
            q.put({"type": "done", "ok": True})
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "pipeline v2 route: stage=resume_stream_failed thread_id=%s session_id=%s elapsed_ms=%.2f",
                payload.thread_id,
                resolved_session_id or "__default__",
                (time.perf_counter() - t0) * 1000,
            )
            for key, value in list(task_status.items()):
                if value is None:
                    task_status[key] = False
            q.put({"type": "error", "detail": str(exc), "stage": "resume"})
            q.put({"type": "done", "ok": False})
        finally:
            if any(v is not None for v in task_status.values()):
                _log_task_success_rate(
                    task_status,
                    terminate_ok=succeeded,
                    effective_output=effective_output if succeeded else None,
                )
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
