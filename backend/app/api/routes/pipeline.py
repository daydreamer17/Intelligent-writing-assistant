from __future__ import annotations

import json
import time
import logging
import os
from queue import Empty, Queue
from threading import Thread

from hello_agents.tools import ToolRegistry
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...models.schemas import (
    CitationItemResponse,
    CoverageDetail,
    PipelineRequest,
    PipelineResponse,
    ResearchNoteResponse,
)
from ...services.citation_enforcer import CitationEnforcer, CoverageReport
from ...services.drafting_service import DraftResult
from ...services.generation_mode import (
    citation_labels_enabled,
    get_generation_mode,
    inference_mark_required,
    mcp_allowed_for_mode,
    refusal_enabled_for_mode,
)
from ...services.github_context import maybe_fetch_github_context
from ...services.inference_marker import mark_inference_paragraphs
from ...services.pipeline_service import PipelineResult
from ...services.research_service import ResearchNote, SourceDocument
from ...services.tool_policy import ToolDecision, build_stage_tool_registry, decide_tools
from ..deps import AppServices, get_services

router = APIRouter(tags=["pipeline"])
logger = logging.getLogger("app.pipeline")
_citation_enforcer = CitationEnforcer()


def _min_effective_chars() -> int:
    raw = os.getenv("PIPELINE_EFFECTIVE_OUTPUT_MIN_CHARS", "200")
    try:
        return max(1, int(raw))
    except ValueError:
        return 200


def _is_effective(text: str) -> bool:
    return len((text or "").strip()) >= _min_effective_chars()


def _append_block(text: str, block: str, label: str) -> str:
    if not block:
        return text
    if text:
        return f"{text}\n\n[{label}]\n{block}"
    return f"[{label}]\n{block}"


def _coverage_detail(report: CoverageReport | None) -> CoverageDetail | None:
    if report is None:
        return None
    paragraph_coverage = (
        report.covered_paragraphs / report.total_paragraphs if report.total_paragraphs > 0 else 0.0
    )
    return CoverageDetail(
        token_coverage=report.token_coverage,
        paragraph_coverage=paragraph_coverage,
        covered_tokens=report.covered_tokens,
        total_tokens=report.total_tokens,
        covered_paragraphs=report.covered_paragraphs,
        total_paragraphs=report.total_paragraphs,
        semantic_coverage=report.semantic_coverage or 0.0,
    )


def _refusal_message() -> str:
    return "在提供的文档中，无法找到该问题的答案。"


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _pipeline_max_tokens(stage: str) -> int | None:
    stage = stage.upper()
    return _parse_optional_int(f"PIPELINE_{stage}_MAX_TOKENS") or _parse_optional_int(
        f"STAGE_{stage}_MAX_TOKENS"
    )


def _pipeline_max_input_chars(stage: str) -> int | None:
    stage = stage.upper()
    return _parse_optional_int(f"PIPELINE_{stage}_MAX_INPUT_CHARS") or _parse_optional_int(
        f"STAGE_{stage}_MAX_INPUT_CHARS"
    )


def _refusal_enabled() -> bool:
    if not refusal_enabled_for_mode(get_generation_mode()):
        return False
    return os.getenv("RAG_REFUSAL_ENABLED", "true").lower() in ("1", "true", "yes")


def _citations_enabled() -> bool:
    return citation_labels_enabled(get_generation_mode())


def _strict_citation_labels() -> bool:
    return get_generation_mode() == "rag_only"


def _mcp_enabled() -> bool:
    return mcp_allowed_for_mode(get_generation_mode())


def _postprocess_final_text(text: str) -> str:
    mode = get_generation_mode()
    if inference_mark_required(mode):
        return mark_inference_paragraphs(text)
    return text


def _refusal_mode() -> str:
    raw = os.getenv("RAG_REFUSAL_MODE", "fallback").strip().lower()
    if raw in {"strict", "fallback"}:
        return raw
    return "fallback"


def _base_should_refuse(report) -> bool:
    if not _refusal_enabled():
        return False
    min_terms = max(1, _parse_int("RAG_REFUSAL_MIN_QUERY_TERMS", 2))
    min_docs = max(1, _parse_int("RAG_REFUSAL_MIN_DOCS", 1))
    min_best = _parse_float("RAG_REFUSAL_MIN_RECALL", 0.1)
    min_avg = _parse_float("RAG_REFUSAL_MIN_AVG_RECALL", 0.05)
    if report.query_terms < min_terms:
        return False
    return report.docs < min_docs or report.best_recall < min_best or report.avg_recall < min_avg


def _fallback_should_refuse(report) -> bool:
    min_terms = max(1, _parse_int("RAG_REFUSAL_MIN_QUERY_TERMS", 2))
    min_docs = max(1, _parse_int("RAG_REFUSAL_FALLBACK_MIN_DOCS", 2))
    min_best = _parse_float("RAG_REFUSAL_FALLBACK_MIN_RECALL", 0.12)
    min_avg = _parse_float("RAG_REFUSAL_FALLBACK_MIN_AVG_RECALL", 0.06)
    if report.query_terms < min_terms:
        return False
    return report.docs < min_docs or report.best_recall < min_best or report.avg_recall < min_avg


def _log_refusal(report, *, refused: bool, context: str) -> None:
    logger.info(
        "RAG refusal check (%s): docs=%s terms=%s mixed_best=%.3f mixed_avg=%.3f lex_best=%.3f tfidf_best=%.3f -> %s",
        context,
        report.docs,
        report.query_terms,
        report.best_recall,
        report.avg_recall,
        report.lexical_best,
        report.tfidf_best,
        "refuse" if refused else "pass",
    )


def _same_source_ids(left: list[SourceDocument], right: list[SourceDocument]) -> bool:
    if len(left) != len(right):
        return False
    return [item.doc_id for item in left] == [item.doc_id for item in right]


def _merge_sources(primary: list[SourceDocument], extra: list[SourceDocument]) -> list[SourceDocument]:
    merged = list(primary)
    seen = {item.doc_id for item in primary}
    for doc in extra:
        if doc.doc_id in seen:
            continue
        merged.append(doc)
        seen.add(doc.doc_id)
    return merged


def _refusal_check_with_fallback(
    *,
    query: str,
    sources: list[SourceDocument],
    services: AppServices,
) -> tuple[bool, list[SourceDocument]]:
    report = services.pipeline.researcher.relevance_report(query, sources)
    refused = _base_should_refuse(report)
    _log_refusal(report, refused=refused, context="base")
    if not refused or _refusal_mode() == "strict":
        return refused, sources

    fallback_top_k = max(5, _parse_int("RAG_REFUSAL_FALLBACK_TOP_K", 12))
    fallback_docs = services.rag.search(query, top_k=fallback_top_k)
    merged = _merge_sources(sources, fallback_docs)
    fallback_report = services.pipeline.researcher.relevance_report(query, merged)
    fallback_refused = _fallback_should_refuse(fallback_report)
    _log_refusal(
        fallback_report,
        refused=fallback_refused,
        context=f"fallback@{fallback_top_k}",
    )
    if not fallback_refused:
        return False, merged
    return True, sources


def _github_note(context: str) -> ResearchNote:
    summary = (context or "").strip().replace("\n", " ")
    return ResearchNote(
        doc_id="github:mcp",
        title="GitHub MCP Context",
        summary=summary[:600],
        url="",
    )


def _empty_tool_decision(stage: str, reason: str) -> ToolDecision:
    return ToolDecision(enabled=False, allowed_tools=[], reason=reason, profile_id=f"{stage}:none")


def _resolve_stage_tools(
    *,
    services: AppServices,
    stage: str,
    topic: str = "",
    outline: str = "",
    draft: str = "",
    guidance: str = "",
    constraints: str = "",
    research_notes: str = "",
    rag_enforced: bool = False,
    source_count: int = 0,
) -> tuple[ToolDecision, ToolRegistry | None]:
    if not _mcp_enabled():
        decision = _empty_tool_decision(stage, "generation_mode_disables_mcp")
        logger.info(
            "stage_tool_decision: stage=%s enabled=%s tools=%s reason=%s profile_id=%s",
            stage,
            decision.enabled,
            decision.allowed_tools,
            decision.reason,
            decision.profile_id,
        )
        return decision, None

    if not services.agent_tool_calling_enabled or not services.agent_tools_catalog:
        decision = _empty_tool_decision(stage, "agent_tool_calling_disabled")
        logger.info(
            "stage_tool_decision: stage=%s enabled=%s tools=%s reason=%s profile_id=%s",
            stage,
            decision.enabled,
            decision.allowed_tools,
            decision.reason,
            decision.profile_id,
        )
        return decision, None

    decision = decide_tools(
        stage=stage,  # type: ignore[arg-type]
        topic=topic,
        outline=outline,
        draft=draft,
        guidance=guidance,
        constraints=constraints,
        research_notes=research_notes,
        rag_enforced=rag_enforced,
        source_count=source_count,
    )
    registry = build_stage_tool_registry(
        tool_catalog=services.agent_tools_catalog,
        allowed_tool_names=decision.allowed_tools,
    )
    if decision.enabled and registry is None:
        decision = _empty_tool_decision(stage, decision.reason + "|tools_not_available")

    logger.info(
        "stage_tool_decision: stage=%s enabled=%s tools=%s reason=%s profile_id=%s",
        stage,
        decision.enabled,
        decision.allowed_tools,
        decision.reason,
        decision.profile_id,
    )
    return decision, registry


def _runtime_tool_profile_id(
    services: AppServices,
    decision: ToolDecision,
) -> str | None:
    if not services.agent_tool_calling_enabled:
        return None
    return decision.profile_id


def _runtime_tool_registry(
    services: AppServices,
    registry: ToolRegistry | None,
) -> ToolRegistry | None:
    if not services.agent_tool_calling_enabled:
        return None
    return registry


def _log_task_success_rate(
    task_status: dict[str, bool | None],
    *,
    terminate_ok: bool,
    effective_output: bool | None,
) -> None:
    statuses = [v for v in task_status.values() if v is not None]
    total = len(statuses)
    success = sum(1 for v in statuses if v)
    rate = (success / total) * 100 if total else 0.0
    logger.info(
        "Task Success Rate: %s/%s = %.1f%% | terminate=%s | effective_output=%s",
        success,
        total,
        rate,
        terminate_ok,
        effective_output,
    )


def _serialize_pipeline_result(
    payload: PipelineRequest,
    result,
    services: AppServices,
    *,
    coverage: float | None = None,
    coverage_detail: CoverageDetail | None = None,
    citation_enforced: bool = False,
) -> PipelineResponse:
    notes = [
        ResearchNoteResponse(
            doc_id=note.doc_id,
            title=note.title,
            summary=note.summary,
            url=note.url,
        )
        for note in result.research_notes
    ]
    citations = services.citations.build_citations(result.research_notes)
    bibliography = services.citations.format_bibliography(citations)
    version_id = services.storage.save_draft_version(
        topic=payload.topic,
        outline=result.outline.outline,
        research_notes=result.draft_result.research_notes,
        draft=result.draft_result.draft,
        review=result.draft_result.review,
        revised=result.draft_result.revised,
    )
    citation_items = [
        CitationItemResponse(label=item.label, title=item.title, url=item.url)
        for item in citations
    ]
    return PipelineResponse(
        outline=result.outline.outline,
        assumptions=result.outline.assumptions,
        open_questions=result.outline.open_questions,
        research_notes=notes,
        draft=result.draft_result.draft,
        review=result.draft_result.review,
        revised=result.draft_result.revised,
        citations=citation_items,
        bibliography=bibliography,
        version_id=version_id,
        coverage=coverage,
        coverage_detail=coverage_detail,
        citation_enforced=citation_enforced,
    )


@router.post("/pipeline", response_model=PipelineResponse)
def run_pipeline(
    payload: PipelineRequest,
    services: AppServices = Depends(get_services),
) -> PipelineResponse:
    task_status: dict[str, bool | None] = {
        "plan": None,
        "research": None,
        "draft": None,
        "review": None,
        "rewrite": None,
        "citations": None,
    }
    succeeded = False
    try:
        t0 = time.perf_counter()
        logger.info(
            "pipeline start: session_id=%s topic_len=%s source_count=%s mode=%s citation_enforce=%s mcp_allowed=%s",
            payload.session_id or "__default__",
            len(payload.topic or ""),
            len(payload.sources or []),
            get_generation_mode(),
            _citations_enabled(),
            _mcp_enabled(),
        )
        logger.info(
            "pipeline budget: plan(t=%s,c=%s) draft(t=%s,c=%s) review(t=%s,c=%s) rewrite(t=%s,c=%s)",
            _pipeline_max_tokens("plan"),
            _pipeline_max_input_chars("plan"),
            _pipeline_max_tokens("draft"),
            _pipeline_max_input_chars("draft"),
            _pipeline_max_tokens("review"),
            _pipeline_max_input_chars("review"),
            _pipeline_max_tokens("rewrite"),
            _pipeline_max_input_chars("rewrite"),
        )
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
        constraints = _append_block(payload.constraints, github_context, "GitHub参考")
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
        apply_labels = _citations_enabled()
        source_count_for_policy = len(sources)
        plan_tool_decision, plan_tool_registry = _resolve_stage_tools(
            services=services,
            stage="plan",
            topic=payload.topic,
            constraints=constraints,
            research_notes=payload.key_points,
            rag_enforced=apply_labels,
            source_count=source_count_for_policy,
        )
        draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
            services=services,
            stage="draft",
            topic=payload.topic,
            constraints=constraints,
            research_notes=payload.key_points,
            rag_enforced=apply_labels,
            source_count=source_count_for_policy,
        )
        review_tool_decision, review_tool_registry = _resolve_stage_tools(
            services=services,
            stage="review",
            topic=payload.topic,
            guidance=payload.review_criteria,
            constraints=constraints,
            rag_enforced=apply_labels,
            source_count=source_count_for_policy,
        )
        rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
            services=services,
            stage="rewrite",
            topic=payload.topic,
            guidance=payload.review_criteria,
            constraints=constraints,
            rag_enforced=apply_labels,
            source_count=source_count_for_policy,
        )
        result = services.pipeline.run(
            topic=payload.topic,
            audience=payload.audience,
            style=payload.style,
            target_length=payload.target_length,
            constraints=constraints,
            key_points=payload.key_points,
            sources=sources,
            review_criteria=payload.review_criteria,
            plan_max_tokens=_pipeline_max_tokens("plan"),
            draft_max_tokens=_pipeline_max_tokens("draft"),
            review_max_tokens=_pipeline_max_tokens("review"),
            rewrite_max_tokens=_pipeline_max_tokens("rewrite"),
            plan_max_input_chars=_pipeline_max_input_chars("plan"),
            draft_max_input_chars=_pipeline_max_input_chars("draft"),
            review_max_input_chars=_pipeline_max_input_chars("review"),
            rewrite_max_input_chars=_pipeline_max_input_chars("rewrite"),
            session_id=payload.session_id,
            plan_tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
            plan_tool_registry_override=_runtime_tool_registry(services, plan_tool_registry),
            draft_tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
            draft_tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
            review_tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
            review_tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
            rewrite_tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
            rewrite_tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
        )
        refusal_query = f"{payload.topic}\n{result.outline.outline}"
        effective_sources = list(sources)
        if _refusal_enabled():
            refused, effective_sources = _refusal_check_with_fallback(
                query=refusal_query,
                sources=effective_sources,
                services=services,
            )
        else:
            refused = False
        if refused:
            refusal_text = _refusal_message()
            draft_result = DraftResult(
                outline=result.draft_result.outline,
                research_notes=result.draft_result.research_notes,
                draft=refusal_text,
                review=refusal_text,
                revised=refusal_text,
            )
            result = PipelineResult(
                outline=result.outline,
                research_notes=result.research_notes,
                draft_result=draft_result,
            )
            task_status["plan"] = bool(result.outline.outline.strip())
            task_status["research"] = None if not sources else bool(result.research_notes)
            task_status["draft"] = False
            task_status["review"] = False
            task_status["rewrite"] = False
            task_status["citations"] = False
            succeeded = True
            response = _serialize_pipeline_result(
                payload,
                result,
                services,
                coverage=0.0,
                coverage_detail=CoverageDetail(),
                citation_enforced=False,
            )
            logger.info(
                "pipeline refused: session_id=%s elapsed_ms=%.2f",
                payload.session_id or "__default__",
                (time.perf_counter() - t0) * 1000,
            )
            return response
        if apply_labels and not _same_source_ids(sources, effective_sources):
            refreshed_notes = services.pipeline.collect_research_notes(
                payload.topic,
                result.outline,
                effective_sources,
            )
            if github_context and not any(note.doc_id == "github:mcp" for note in refreshed_notes):
                refreshed_notes.append(_github_note(github_context))
            refreshed_notes_text = services.pipeline.researcher.format_notes(refreshed_notes)
            refreshed_draft = services.drafter.run_full(
                topic=payload.topic,
                outline=result.outline.outline,
                research_notes=refreshed_notes_text,
                constraints=constraints,
                style=payload.style,
                target_length=payload.target_length,
                review_criteria=payload.review_criteria,
                audience=payload.audience,
                draft_max_tokens=_pipeline_max_tokens("draft"),
                review_max_tokens=_pipeline_max_tokens("review"),
                rewrite_max_tokens=_pipeline_max_tokens("rewrite"),
                draft_max_input_chars=_pipeline_max_input_chars("draft"),
                review_max_input_chars=_pipeline_max_input_chars("review"),
                rewrite_max_input_chars=_pipeline_max_input_chars("rewrite"),
                session_id=payload.session_id,
                draft_tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                draft_tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
                review_tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                review_tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
                rewrite_tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                rewrite_tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
            )
            result = PipelineResult(
                outline=result.outline,
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
        enforced_text = _postprocess_final_text(enforced_text)
        citation_enforced = apply_labels and enforced_text != result.draft_result.revised
        draft_result = DraftResult(
            outline=result.draft_result.outline,
            research_notes=result.draft_result.research_notes,
            draft=result.draft_result.draft,
            review=result.draft_result.review,
            revised=enforced_text,
        )
        result = PipelineResult(
            outline=result.outline,
            research_notes=result.research_notes,
            draft_result=draft_result,
        )
        task_status["plan"] = bool(result.outline.outline.strip())
        task_status["research"] = None if not sources else bool(result.research_notes)
        task_status["draft"] = _is_effective(result.draft_result.draft)
        task_status["review"] = bool(result.draft_result.review.strip())
        task_status["rewrite"] = _is_effective(result.draft_result.revised)
        task_status["citations"] = True
        succeeded = True
        response = _serialize_pipeline_result(
            payload,
            result,
            services,
            coverage=report.coverage,
            coverage_detail=_coverage_detail(report),
            citation_enforced=citation_enforced,
        )
        logger.info(
            "pipeline done: session_id=%s draft_len=%s revised_len=%s notes=%s elapsed_ms=%.2f",
            payload.session_id or "__default__",
            len(result.draft_result.draft or ""),
            len(result.draft_result.revised or ""),
            len(result.research_notes or []),
            (time.perf_counter() - t0) * 1000,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Pipeline execution failed") from exc
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


@router.post("/pipeline/stream")
def run_pipeline_stream(
    payload: PipelineRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    q: Queue[dict | None] = Queue()

    def worker() -> None:
        def _emit_text_deltas(stage: str, text: str) -> None:
            if not text:
                return
            chunk_size = max(128, _parse_int("PIPELINE_STREAM_FALLBACK_CHUNK_SIZE", 600))
            for i in range(0, len(text), chunk_size):
                q.put({"type": "delta", "stage": stage, "content": text[i : i + chunk_size]})

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
        stage_start = time.perf_counter()
        current_stage = "init"

        def _log_stage(stage_name: str) -> None:
            nonlocal stage_start
            now = time.perf_counter()
            logger.info(
                "pipeline stream stage: session_id=%s stage=%s elapsed_ms=%.2f",
                payload.session_id or "__default__",
                stage_name,
                (now - stage_start) * 1000,
            )
            stage_start = now
        try:
            logger.info(
                "pipeline stream start: session_id=%s topic_len=%s source_count=%s mode=%s citation_enforce=%s mcp_allowed=%s",
                payload.session_id or "__default__",
                len(payload.topic or ""),
                len(payload.sources or []),
                get_generation_mode(),
                _citations_enabled(),
                _mcp_enabled(),
            )
            apply_labels = _citations_enabled()
            logger.info(
                "pipeline stream budget: plan(t=%s,c=%s) draft(t=%s,c=%s) review(t=%s,c=%s) rewrite(t=%s,c=%s)",
                _pipeline_max_tokens("plan"),
                _pipeline_max_input_chars("plan"),
                _pipeline_max_tokens("draft"),
                _pipeline_max_input_chars("draft"),
                _pipeline_max_tokens("review"),
                _pipeline_max_input_chars("review"),
                _pipeline_max_tokens("rewrite"),
                _pipeline_max_input_chars("rewrite"),
            )
            current_stage = "plan"
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
            constraints = _append_block(payload.constraints, github_context, "GitHub参考")
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
            source_count_for_policy = len(sources)
            plan_tool_decision, plan_tool_registry = _resolve_stage_tools(
                services=services,
                stage="plan",
                topic=payload.topic,
                constraints=constraints,
                research_notes=payload.key_points,
                rag_enforced=apply_labels,
                source_count=source_count_for_policy,
            )
            q.put({"type": "status", "step": "plan"})
            outline = services.pipeline.planner.plan_outline_stream(
                topic=payload.topic,
                audience=payload.audience,
                style=payload.style,
                target_length=payload.target_length,
                constraints=constraints,
                key_points=payload.key_points,
                max_tokens=_pipeline_max_tokens("plan"),
                max_input_chars=_pipeline_max_input_chars("plan"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, plan_tool_registry),
                on_chunk=lambda chunk: q.put({"type": "delta", "stage": "plan", "content": chunk}),
            )
            task_status["plan"] = bool(outline.outline.strip())
            _log_stage("plan")
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

            refusal_query = f"{payload.topic}\n{outline.outline}"
            effective_sources = list(sources)
            if _refusal_enabled() and effective_sources:
                refused, effective_sources = _refusal_check_with_fallback(
                    query=refusal_query,
                    sources=effective_sources,
                    services=services,
                )
            else:
                refused = False

            current_stage = "research"
            q.put({"type": "status", "step": "research"})
            if effective_sources:
                notes = services.pipeline.collect_research_notes(payload.topic, outline, effective_sources)
            else:
                notes = []
            if github_context and not any(note.doc_id == "github:mcp" for note in notes):
                notes.append(_github_note(github_context))
            task_status["research"] = None if not effective_sources else bool(notes)
            _log_stage("research")
            notes_text = services.pipeline.researcher.format_notes(notes)
            draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
                services=services,
                stage="draft",
                topic=payload.topic,
                outline=outline.outline,
                constraints=constraints,
                research_notes=notes_text,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )
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
                    payload,
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

            current_stage = "draft"
            q.put({"type": "status", "step": "draft"})
            draft_chunks: list[str] = []
            evidence_text = services.drafter.extract_evidence(notes_text)
            draft_constraints = services.drafter.build_constraints(
                research_notes=notes_text,
                constraints=constraints,
                evidence_text=evidence_text,
            )
            for chunk in services.drafter.writing_agent.draft_stream(
                topic=payload.topic,
                outline=outline.outline,
                constraints=draft_constraints,
                style=payload.style,
                target_length=payload.target_length,
                max_tokens=_pipeline_max_tokens("draft"),
                max_input_chars=_pipeline_max_input_chars("draft"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
            ):
                if not chunk:
                    continue
                draft_chunks.append(chunk)
                q.put({"type": "delta", "stage": "draft", "content": chunk})
            draft = "".join(draft_chunks)
            if not draft.strip():
                logger.warning(
                    "pipeline stream draft produced empty output, fallback to non-stream draft generation."
                )
                draft = services.drafter.create_draft(
                    topic=payload.topic,
                    outline=outline.outline,
                    research_notes=notes_text,
                    constraints=constraints,
                    style=payload.style,
                    target_length=payload.target_length,
                    max_tokens=_pipeline_max_tokens("draft"),
                    max_input_chars=_pipeline_max_input_chars("draft"),
                    session_id=payload.session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
                )
                _emit_text_deltas("draft", draft)
            task_status["draft"] = _is_effective(draft)
            _log_stage("draft")
            q.put({"type": "draft", "payload": {"draft": draft}})

            current_stage = "review"
            q.put({"type": "status", "step": "review"})
            review_chunks: list[str] = []
            review_tool_decision, review_tool_registry = _resolve_stage_tools(
                services=services,
                stage="review",
                topic=payload.topic,
                outline=outline.outline,
                draft=draft,
                guidance=payload.review_criteria,
                research_notes=notes_text,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )
            # 根据 draft 长度动态设置 max_tokens，避免截断
            review_max_tokens = min(8000, max(2000, int(len(draft) * 0.8)))
            for chunk in services.reviewer.agent.review_stream(
                draft=draft,
                criteria=payload.review_criteria,
                sources=notes_text,
                audience=payload.audience,
                max_tokens=review_max_tokens,
                max_input_chars=_pipeline_max_input_chars("review"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
            ):
                if not chunk:
                    continue
                review_chunks.append(chunk)
                q.put({"type": "delta", "stage": "review", "content": chunk})
            review = "".join(review_chunks)
            if not review.strip():
                logger.warning(
                    "pipeline stream review produced empty output, fallback to non-stream review generation."
                )
                review = services.reviewer.review(
                    draft=draft,
                    criteria=payload.review_criteria,
                    sources=notes_text,
                    audience=payload.audience,
                    max_tokens=review_max_tokens,
                    max_input_chars=_pipeline_max_input_chars("review"),
                    session_id=payload.session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
                ).review
                _emit_text_deltas("review", review)
            task_status["review"] = bool(review.strip())
            _log_stage("review")
            q.put({"type": "review", "payload": {"review": review}})

            current_stage = "rewrite"
            q.put({"type": "status", "step": "rewrite"})
            rewrite_chunks: list[str] = []
            rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
                services=services,
                stage="rewrite",
                topic=payload.topic,
                outline=outline.outline,
                draft=draft,
                guidance=(review + "\n\n" + payload.review_criteria).strip(),
                constraints=payload.style,
                research_notes=notes_text,
                rag_enforced=apply_labels,
                source_count=len(effective_sources),
            )
            # 根据 draft 长度动态设置 max_tokens，避免截断
            rewrite_max_tokens = min(8000, max(2000, int(len(draft) * 1.5)))
            rewrite_guidance_parts = [part.strip() for part in [review, payload.review_criteria] if part and part.strip()]
            rewrite_guidance = "\n\n".join(dict.fromkeys(rewrite_guidance_parts))
            if evidence_text:
                rewrite_guidance = (rewrite_guidance + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=draft,
                guidance=rewrite_guidance,
                style=payload.style,
                target_length=payload.target_length,
                max_tokens=rewrite_max_tokens,
                max_input_chars=_pipeline_max_input_chars("rewrite"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
            ):
                if not chunk:
                    continue
                rewrite_chunks.append(chunk)
                q.put({"type": "delta", "stage": "rewrite", "content": chunk})
            revised = "".join(rewrite_chunks)
            if not revised.strip():
                logger.warning(
                    "pipeline stream rewrite produced empty output, fallback to non-stream rewrite generation."
                )
                revised = services.rewriter.rewrite(
                    draft=draft,
                    guidance=rewrite_guidance,
                    style=payload.style,
                    target_length=payload.target_length,
                    max_tokens=rewrite_max_tokens,
                    max_input_chars=_pipeline_max_input_chars("rewrite"),
                    session_id=payload.session_id,
                    tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                    tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
                ).revised
                _emit_text_deltas("rewrite", revised)
            apply_labels = _citations_enabled()
            enforced_text, report = _citation_enforcer.enforce(
                revised or draft,
                notes,
                apply_labels=apply_labels,
                strict_labels=_strict_citation_labels(),
                embedder=services.rag.get_embedder(),
            )
            enforced_text = _postprocess_final_text(enforced_text)
            citation_enforced = apply_labels and enforced_text != revised
            revised = enforced_text
            task_status["rewrite"] = _is_effective(revised)
            _log_stage("rewrite")
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
                payload,
                result,
                services,
                coverage=report.coverage,
                coverage_detail=_coverage_detail(report),
                citation_enforced=citation_enforced,
            )
            task_status["citations"] = True
            succeeded = True
            effective_output = bool(task_status["rewrite"]) or bool(task_status["draft"])
            _log_stage("citations")
            logger.info(
                "pipeline stream done: session_id=%s draft_len=%s revised_len=%s notes=%s",
                payload.session_id or "__default__",
                len(draft or ""),
                len(revised or ""),
                len(notes or []),
            )
            q.put({"type": "result", "payload": response.model_dump()})
            q.put({"type": "done", "ok": True})
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "pipeline stream failed: session_id=%s stage=%s",
                payload.session_id or "__default__",
                current_stage,
            )
            # Mark unfinished stages as failed, so success-rate isn't misleading.
            for key, value in list(task_status.items()):
                if value is None:
                    task_status[key] = False
            q.put({"type": "error", "detail": str(exc), "stage": current_stage})
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
        yield event({"type": "status", "message": "started"})
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                item = q.get(timeout=2)
            except Empty:
                yield event({"type": "ping", "ts": int(time.time())})
                continue
            if item is None:
                break
            yield event(item)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
