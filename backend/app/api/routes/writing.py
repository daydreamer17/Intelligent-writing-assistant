from __future__ import annotations

import json
import os
import time
import logging
from queue import Empty, Queue
from threading import Thread

from hello_agents.tools import ToolRegistry
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ...models.schemas import (
    CitationItemResponse,
    CoverageDetail,
    DraftRequest,
    DraftResponse,
    PlanRequest,
    PlanResponse,
    ReviewRequest,
    ReviewResponse,
    RewriteRequest,
    RewriteResponse,
)
from ...services.citation_enforcer import CitationEnforcer, CoverageReport
from ...services.generation_mode import (
    citation_labels_enabled,
    get_generation_mode,
    inference_mark_required,
    mcp_allowed_for_mode,
    refusal_enabled_for_mode,
)
from ...services.github_context import maybe_fetch_github_context
from ...services.inference_marker import mark_inference_paragraphs
from ...services.research_service import ResearchNote, ResearchService, RelevanceReport, SourceDocument
from ...services.tool_policy import ToolDecision, build_stage_tool_registry, decide_tools
from ..deps import AppServices, get_services

router = APIRouter(tags=["writing"])
logger = logging.getLogger("app.writing")
_citation_enforcer = CitationEnforcer()
_researcher = ResearchService()


def _citations_enabled() -> bool:
    return citation_labels_enabled(get_generation_mode())


def _strict_citation_labels() -> bool:
    return get_generation_mode() == "rag_only"


def _refusal_enabled() -> bool:
    if not refusal_enabled_for_mode(get_generation_mode()):
        return False
    return os.getenv("RAG_REFUSAL_ENABLED", "true").lower() in ("1", "true", "yes")


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


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _step_max_tokens(stage: str) -> int | None:
    stage = stage.upper()
    return _parse_optional_int(f"STEP_{stage}_MAX_TOKENS") or _parse_optional_int(
        f"STAGE_{stage}_MAX_TOKENS"
    )


def _step_max_input_chars(stage: str) -> int | None:
    stage = stage.upper()
    return _parse_optional_int(f"STEP_{stage}_MAX_INPUT_CHARS") or _parse_optional_int(
        f"STAGE_{stage}_MAX_INPUT_CHARS"
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _build_refusal_query(
    *,
    topic: str,
    outline: str = "",
    constraints: str = "",
    key_points: str = "",
    audience: str = "",
    style: str = "",
    draft: str = "",
) -> str:
    max_chars = max(120, _parse_int("RAG_REFUSAL_QUERY_MAX_CHARS", 480))
    include_outline = _parse_bool("RAG_REFUSAL_INCLUDE_OUTLINE", False)
    include_draft = _parse_bool("RAG_REFUSAL_INCLUDE_DRAFT", False)
    parts = [topic.strip(), key_points.strip(), constraints.strip(), audience.strip(), style.strip()]
    if include_outline and outline:
        parts.append(outline.strip())
    if include_draft and draft:
        parts.append(draft.strip())
    compact_parts: list[str] = []
    seen = set()
    for part in parts:
        if not part:
            continue
        norm = " ".join(part.split())
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        compact_parts.append(norm)
    if not compact_parts:
        fallback = topic or outline or draft
        return _truncate_text((fallback or "").strip(), max_chars)
    return _truncate_text(" | ".join(compact_parts), max_chars)


def _should_refuse(report: RelevanceReport) -> bool:
    if not _citations_enabled() or not _refusal_enabled():
        return False
    min_terms = max(1, _parse_int("RAG_REFUSAL_MIN_QUERY_TERMS", 2))
    min_docs = max(1, _parse_int("RAG_REFUSAL_MIN_DOCS", 1))
    min_best = _parse_float("RAG_REFUSAL_MIN_RECALL", 0.1)
    min_avg = _parse_float("RAG_REFUSAL_MIN_AVG_RECALL", 0.05)
    if report.query_terms < min_terms:
        return False
    if report.docs < min_docs:
        return True
    if report.best_recall < min_best:
        return True
    if report.avg_recall < min_avg:
        return True
    return False


def _log_refusal(report: RelevanceReport, *, refused: bool, context: str) -> None:
    status = "refuse" if refused else "pass"
    print(
        f"[RAG][refusal:{status}] context={context} docs={report.docs} terms={report.query_terms} "
        f"mixed_best={report.best_recall:.3f} mixed_avg={report.avg_recall:.3f} "
        f"lex_best={report.lexical_best:.3f} tfidf_best={report.tfidf_best:.3f}"
    )


def _citation_top_k() -> int:
    raw = os.getenv("RAG_CITATION_TOP_K", "5")
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


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
        semantic_covered_paragraphs=report.semantic_covered_paragraphs,
        semantic_total_paragraphs=report.semantic_total_paragraphs,
    )


def _citation_payload(
    *,
    text: str,
    notes: list[ResearchNote],
    services: AppServices,
    apply_labels: bool,
) -> tuple[str, CoverageReport, list[CitationItemResponse], str, bool]:
    enforced_text, report = _citation_enforcer.enforce(
        text,
        notes,
        apply_labels=apply_labels,
        strict_labels=_strict_citation_labels(),
        embedder=services.rag.get_embedder(),
    )
    enforced_text = _postprocess_final_text(enforced_text)
    citations = services.citations.build_citations(notes)
    bibliography = services.citations.format_bibliography(citations)
    return (
        enforced_text,
        report,
        [CitationItemResponse(label=item.label, title=item.title, url=item.url) for item in citations],
        bibliography,
        apply_labels and enforced_text != text,
    )


def _enforce_with_rag(
    text: str,
    *,
    query: str,
    services: AppServices,
    docs: list[SourceDocument] | None = None,
) -> str:
    if not _citations_enabled():
        return text
    docs = docs or services.rag.search(query, top_k=_citation_top_k())
    if not docs:
        return text
    notes = _researcher.collect_notes(query=query, sources=docs, top_k=min(len(docs), _citation_top_k()))
    enforced, _ = _citation_enforcer.enforce(
        text,
        notes,
        apply_labels=True,
        strict_labels=_strict_citation_labels(),
    )
    return enforced


def _rag_refusal_check(
    *,
    query: str,
    services: AppServices,
) -> tuple[bool, list[SourceDocument]]:
    base_top_k = _citation_top_k()
    docs = services.rag.search(query, top_k=base_top_k)
    report, report_query = _best_refusal_report(query=query, docs=docs, services=services)
    refused = _should_refuse(report)
    _log_refusal(report, refused=refused, context=f"writing:base@{base_top_k}:{report_query}")
    if not refused or _refusal_mode() == "strict":
        return refused, docs

    fallback_top_k = max(base_top_k, _parse_int("RAG_REFUSAL_FALLBACK_TOP_K", base_top_k * 2))
    fallback_docs = services.rag.search(query, top_k=fallback_top_k)
    fallback_report, fallback_query = _best_refusal_report(
        query=query, docs=fallback_docs, services=services
    )
    fallback_refused = _should_refuse_fallback(fallback_report)
    _log_refusal(
        fallback_report,
        refused=fallback_refused,
        context=f"writing:fallback@{fallback_top_k}:{fallback_query}",
    )
    if not fallback_refused:
        return False, fallback_docs
    return True, docs


def _best_refusal_report(
    *,
    query: str,
    docs: list[SourceDocument],
    services: AppServices,
) -> tuple[RelevanceReport, str]:
    candidates: list[tuple[str, str]] = [("original", query)]
    try:
        for variant in services.rag.query_variants(query):
            label = getattr(variant, "source", "variant")
            text = (getattr(variant, "text", "") or "").strip()
            if text:
                candidates.append((str(label), text))
    except Exception:
        pass

    deduped: list[tuple[str, str]] = []
    seen = set()
    for label, text in candidates:
        key = " ".join(text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((label, text))

    best_report = _researcher.relevance_report(query, docs)
    best_label = "original"
    best_key = (
        best_report.best_recall,
        best_report.avg_recall,
        best_report.lexical_best,
        best_report.tfidf_best,
    )
    for label, text in deduped:
        report = _researcher.relevance_report(text, docs)
        key = (
            report.best_recall,
            report.avg_recall,
            report.lexical_best,
            report.tfidf_best,
        )
        if key > best_key:
            best_report = report
            best_key = key
            best_label = label
    return best_report, best_label


def _should_refuse_fallback(report: RelevanceReport) -> bool:
    min_terms = max(1, _parse_int("RAG_REFUSAL_MIN_QUERY_TERMS", 2))
    min_docs = max(1, _parse_int("RAG_REFUSAL_FALLBACK_MIN_DOCS", 2))
    min_best = _parse_float("RAG_REFUSAL_FALLBACK_MIN_RECALL", 0.12)
    min_avg = _parse_float("RAG_REFUSAL_FALLBACK_MIN_AVG_RECALL", 0.06)
    if report.query_terms < min_terms:
        return False
    if report.docs < min_docs:
        return True
    if report.best_recall < min_best:
        return True
    if report.avg_recall < min_avg:
        return True
    return False


def _build_evidence_from_rag(
    *,
    query: str,
    services: AppServices,
) -> str:
    if not _citations_enabled():
        return ""
    docs = services.rag.search(query, top_k=_citation_top_k())
    if not docs:
        return ""
    notes = _researcher.collect_notes(query=query, sources=docs, top_k=min(len(docs), _citation_top_k()))
    notes_text = _researcher.format_notes(notes)
    return services.drafter.extract_evidence(notes_text)


def _append_block(text: str, block: str, label: str) -> str:
    if not block:
        return text
    if text:
        return f"{text}\n\n[{label}]\n{block}"
    return f"[{label}]\n{block}"


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


@router.post("/plan", response_model=PlanResponse)
def create_plan(
    payload: PlanRequest,
    services: AppServices = Depends(get_services),
) -> PlanResponse:
    try:
        t0 = time.perf_counter()
        logger.info(
            "plan start: session_id=%s topic_len=%s key_points_len=%s",
            payload.session_id or "__default__",
            len(payload.topic or ""),
            len(payload.key_points or ""),
        )
        logger.info(
            "plan budget: max_tokens=%s max_input_chars=%s",
            _step_max_tokens("plan"),
            _step_max_input_chars("plan"),
        )
        github_context = (
            maybe_fetch_github_context(
                services.github_mcp_tool,
                query=payload.topic,
                context="\n".join([payload.constraints, payload.key_points, payload.style, payload.audience]),
            )
            if _mcp_enabled()
            else ""
        )
        key_points = _append_block(payload.key_points, github_context, "GitHub参考")
        plan_tool_decision, plan_tool_registry = _resolve_stage_tools(
            services=services,
            stage="plan",
            topic=payload.topic,
            constraints=payload.constraints,
            research_notes=key_points,
            rag_enforced=_citations_enabled(),
            source_count=1 if key_points.strip() else 0,
        )
        plan = services.planner.plan_outline(
            topic=payload.topic,
            audience=payload.audience,
            style=payload.style,
            target_length=payload.target_length,
            constraints=payload.constraints,
            key_points=key_points,
            max_tokens=_step_max_tokens("plan"),
            max_input_chars=_step_max_input_chars("plan"),
            session_id=payload.session_id,
            tool_profile_id=_runtime_tool_profile_id(services, plan_tool_decision),
            tool_registry_override=_runtime_tool_registry(services, plan_tool_registry),
        )
        response = PlanResponse(
            outline=plan.outline,
            assumptions=plan.assumptions,
            open_questions=plan.open_questions,
        )
        logger.info(
            "plan done: session_id=%s outline_len=%s elapsed_ms=%.2f",
            payload.session_id or "__default__",
            len(response.outline or ""),
            (time.perf_counter() - t0) * 1000,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Plan generation failed") from exc


@router.post("/draft", response_model=DraftResponse)
def create_draft(
    payload: DraftRequest,
    services: AppServices = Depends(get_services),
) -> DraftResponse:
    try:
        t0 = time.perf_counter()
        logger.info(
            "draft start: session_id=%s topic_len=%s outline_len=%s notes_len=%s",
            payload.session_id or "__default__",
            len(payload.topic or ""),
            len(payload.outline or ""),
            len(payload.research_notes or ""),
        )
        logger.info(
            "draft budget: max_tokens=%s max_input_chars=%s",
            _step_max_tokens("draft"),
            _step_max_input_chars("draft"),
        )
        github_context = (
            maybe_fetch_github_context(
                services.github_mcp_tool,
                query=payload.topic,
                context="\n".join([payload.outline, payload.constraints, payload.style, payload.target_length]),
            )
            if _mcp_enabled()
            else ""
        )
        research_notes = _append_block(payload.research_notes, github_context, "GitHub参考")
        draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
            services=services,
            stage="draft",
            topic=payload.topic,
            outline=payload.outline,
            constraints=payload.constraints,
            research_notes=research_notes,
            rag_enforced=_citations_enabled(),
            source_count=1 if research_notes.strip() else 0,
        )
        refusal_query = _build_refusal_query(
            topic=payload.topic,
            outline=payload.outline,
            constraints=payload.constraints,
            style=payload.style,
        )
        refused, docs = _rag_refusal_check(query=refusal_query, services=services)
        if refused:
            return DraftResponse(draft=_refusal_message())
        draft = services.drafter.create_draft(
            topic=payload.topic,
            outline=payload.outline,
            research_notes=research_notes,
            constraints=payload.constraints,
            style=payload.style,
            target_length=payload.target_length,
            max_tokens=_step_max_tokens("draft"),
            max_input_chars=_step_max_input_chars("draft"),
            session_id=payload.session_id,
            tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
            tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
        )
        draft = _enforce_with_rag(
            draft,
            query=refusal_query,
            services=services,
            docs=docs,
        )
        response = DraftResponse(draft=draft)
        logger.info(
            "draft done: session_id=%s draft_len=%s elapsed_ms=%.2f",
            payload.session_id or "__default__",
            len(response.draft or ""),
            (time.perf_counter() - t0) * 1000,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Draft generation failed") from exc


@router.post("/draft/stream")
def create_draft_stream(
    payload: DraftRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    q: Queue[dict | None] = Queue()

    github_context = (
        maybe_fetch_github_context(
            services.github_mcp_tool,
            query=payload.topic,
            context="\n".join([payload.outline, payload.constraints, payload.style, payload.target_length]),
        )
        if _mcp_enabled()
        else ""
    )
    research_notes = _append_block(payload.research_notes, github_context, "GitHub参考")
    draft_tool_decision, draft_tool_registry = _resolve_stage_tools(
        services=services,
        stage="draft",
        topic=payload.topic,
        outline=payload.outline,
        constraints=payload.constraints,
        research_notes=research_notes,
        rag_enforced=_citations_enabled(),
        source_count=1 if research_notes.strip() else 0,
    )

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            logger.info(
                "draft stream start: session_id=%s topic_len=%s outline_len=%s",
                payload.session_id or "__default__",
                len(payload.topic or ""),
                len(payload.outline or ""),
            )
            refusal_query = _build_refusal_query(
                topic=payload.topic,
                outline=payload.outline,
                constraints=payload.constraints,
                style=payload.style,
            )
            refused, docs = _rag_refusal_check(query=refusal_query, services=services)
            if refused:
                q.put({"type": "result", "payload": {"draft": _refusal_message()}})
                return
            evidence_text = services.drafter.extract_evidence(research_notes)
            draft_constraints = services.drafter.build_constraints(
                research_notes=research_notes,
                constraints=payload.constraints,
                evidence_text=evidence_text,
            )
            chunks: list[str] = []
            for chunk in services.drafter.writing_agent.draft_stream(
                topic=payload.topic,
                outline=payload.outline,
                constraints=draft_constraints,
                style=payload.style,
                target_length=payload.target_length,
                max_tokens=_step_max_tokens("draft"),
                max_input_chars=_step_max_input_chars("draft"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, draft_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, draft_tool_registry),
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            draft = "".join(chunks)
            draft = _enforce_with_rag(
                draft,
                query=refusal_query,
                services=services,
                docs=docs,
            )
            q.put({"type": "result", "payload": {"draft": draft}})
            logger.info(
                "draft stream done: session_id=%s draft_len=%s elapsed_ms=%.2f",
                payload.session_id or "__default__",
                len(draft or ""),
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover
            q.put({"type": "error", "detail": str(exc)})
        finally:
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


@router.post("/review", response_model=ReviewResponse)
def review_draft(
    payload: ReviewRequest,
    services: AppServices = Depends(get_services),
) -> ReviewResponse:
    try:
        t0 = time.perf_counter()
        logger.info(
            "review start: session_id=%s draft_len=%s criteria_len=%s",
            payload.session_id or "__default__",
            len(payload.draft or ""),
            len(payload.criteria or ""),
        )
        logger.info(
            "review budget: max_tokens=%s max_input_chars=%s",
            _step_max_tokens("review"),
            _step_max_input_chars("review"),
        )
        github_context = (
            maybe_fetch_github_context(
                services.github_mcp_tool,
                query=(payload.criteria or payload.draft[:200]),
                context="\n".join([payload.sources, payload.audience]),
            )
            if _mcp_enabled()
            else ""
        )
        sources = _append_block(payload.sources, github_context, "GitHub参考")
        review_tool_decision, review_tool_registry = _resolve_stage_tools(
            services=services,
            stage="review",
            draft=payload.draft,
            guidance=payload.criteria,
            research_notes=sources,
            rag_enforced=_citations_enabled(),
            source_count=1 if sources.strip() else 0,
        )
        review = services.reviewer.review(
            draft=payload.draft,
            criteria=payload.criteria,
            sources=sources,
            audience=payload.audience,
            max_tokens=_step_max_tokens("review"),
            max_input_chars=_step_max_input_chars("review"),
            session_id=payload.session_id,
            tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
            tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
        )
        response = ReviewResponse(review=review.review)
        logger.info(
            "review done: session_id=%s review_len=%s elapsed_ms=%.2f",
            payload.session_id or "__default__",
            len(response.review or ""),
            (time.perf_counter() - t0) * 1000,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Review failed") from exc


@router.post("/review/stream")
def review_draft_stream(
    payload: ReviewRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    q: Queue[dict | None] = Queue()

    github_context = (
        maybe_fetch_github_context(
            services.github_mcp_tool,
            query=(payload.criteria or payload.draft[:200]),
            context="\n".join([payload.sources, payload.audience]),
        )
        if _mcp_enabled()
        else ""
    )
    sources = _append_block(payload.sources, github_context, "GitHub参考")
    review_tool_decision, review_tool_registry = _resolve_stage_tools(
        services=services,
        stage="review",
        draft=payload.draft,
        guidance=payload.criteria,
        research_notes=sources,
        rag_enforced=_citations_enabled(),
        source_count=1 if sources.strip() else 0,
    )

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            logger.info(
                "review stream start: session_id=%s draft_len=%s criteria_len=%s",
                payload.session_id or "__default__",
                len(payload.draft or ""),
                len(payload.criteria or ""),
            )
            chunks: list[str] = []
            for chunk in services.reviewer.agent.review_stream(
                draft=payload.draft,
                criteria=payload.criteria,
                sources=sources,
                audience=payload.audience,
                max_tokens=_step_max_tokens("review"),
                max_input_chars=_step_max_input_chars("review"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, review_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, review_tool_registry),
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            review_text = "".join(chunks)
            q.put({"type": "result", "payload": {"review": review_text}})
            logger.info(
                "review stream done: session_id=%s review_len=%s elapsed_ms=%.2f",
                payload.session_id or "__default__",
                len(review_text or ""),
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover
            q.put({"type": "error", "detail": str(exc)})
        finally:
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


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite_draft(
    payload: RewriteRequest,
    services: AppServices = Depends(get_services),
) -> RewriteResponse:
    try:
        t0 = time.perf_counter()
        logger.info(
            "rewrite start: session_id=%s draft_len=%s guidance_len=%s",
            payload.session_id or "__default__",
            len(payload.draft or ""),
            len(payload.guidance or ""),
        )
        logger.info(
            "rewrite budget: max_tokens=%s max_input_chars=%s",
            _step_max_tokens("rewrite"),
            _step_max_input_chars("rewrite"),
        )
        github_context = (
            maybe_fetch_github_context(
                services.github_mcp_tool,
                query=(payload.guidance or payload.draft[:200]),
                context=payload.style,
            )
            if _mcp_enabled()
            else ""
        )
        guidance = _append_block(payload.guidance, github_context, "GitHub参考")
        rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
            services=services,
            stage="rewrite",
            draft=payload.draft,
            guidance=guidance,
            constraints=payload.style,
            rag_enforced=_citations_enabled(),
            source_count=1 if guidance.strip() else 0,
        )
        refusal_query = _build_refusal_query(
            topic="",
            key_points=payload.guidance,
            style=payload.style,
            draft=payload.draft,
        )
        refused, docs = _rag_refusal_check(query=refusal_query, services=services)
        if refused:
            return RewriteResponse(
                revised=_refusal_message(),
                citations=[],
                bibliography="",
                coverage=0.0,
                coverage_detail=CoverageDetail(),
                citation_enforced=False,
            )
        evidence_text = _build_evidence_from_rag(query=refusal_query, services=services)
        if evidence_text:
            guidance = (guidance + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
        revised = services.rewriter.rewrite(
            draft=payload.draft,
            guidance=guidance,
            style=payload.style,
            target_length=payload.target_length,
            max_tokens=_step_max_tokens("rewrite"),
            max_input_chars=_step_max_input_chars("rewrite"),
            session_id=payload.session_id,
            tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
            tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
        )
        notes = _researcher.collect_notes(
            query=refusal_query,
            sources=docs,
            top_k=min(len(docs), _citation_top_k()),
        )
        apply_labels = _citations_enabled()
        text, report, citations, bibliography, citation_enforced = _citation_payload(
            text=revised.revised,
            notes=notes,
            services=services,
            apply_labels=apply_labels,
        )
        response = RewriteResponse(
            revised=text,
            citations=citations,
            bibliography=bibliography,
            coverage=report.coverage,
            coverage_detail=_coverage_detail(report),
            citation_enforced=citation_enforced,
        )
        logger.info(
            "rewrite done: session_id=%s revised_len=%s elapsed_ms=%.2f",
            payload.session_id or "__default__",
            len(response.revised or ""),
            (time.perf_counter() - t0) * 1000,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Rewrite failed") from exc


@router.post("/rewrite/stream")
def rewrite_draft_stream(
    payload: RewriteRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    q: Queue[dict | None] = Queue()

    github_context = (
        maybe_fetch_github_context(
            services.github_mcp_tool,
            query=(payload.guidance or payload.draft[:200]),
            context=payload.style,
        )
        if _mcp_enabled()
        else ""
    )
    guidance = _append_block(payload.guidance, github_context, "GitHub参考")
    rewrite_tool_decision, rewrite_tool_registry = _resolve_stage_tools(
        services=services,
        stage="rewrite",
        draft=payload.draft,
        guidance=guidance,
        constraints=payload.style,
        rag_enforced=_citations_enabled(),
        source_count=1 if guidance.strip() else 0,
    )

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            logger.info(
                "rewrite stream start: session_id=%s draft_len=%s guidance_len=%s",
                payload.session_id or "__default__",
                len(payload.draft or ""),
                len(payload.guidance or ""),
            )
            refusal_query = _build_refusal_query(
                topic="",
                key_points=payload.guidance,
                style=payload.style,
                draft=payload.draft,
            )
            refused, docs = _rag_refusal_check(query=refusal_query, services=services)
            if refused:
                q.put(
                    {
                        "type": "result",
                        "payload": {
                            "revised": _refusal_message(),
                            "citations": [],
                            "bibliography": "",
                            "coverage": 0.0,
                            "coverage_detail": CoverageDetail().model_dump(),
                            "citation_enforced": False,
                        },
                    }
                )
                return
            evidence_text = _build_evidence_from_rag(query=refusal_query, services=services)
            if evidence_text:
                guidance_with_evidence = (guidance + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
            else:
                guidance_with_evidence = guidance
            chunks: list[str] = []
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=payload.draft,
                guidance=guidance_with_evidence,
                style=payload.style,
                target_length=payload.target_length,
                max_tokens=_step_max_tokens("rewrite"),
                max_input_chars=_step_max_input_chars("rewrite"),
                session_id=payload.session_id,
                tool_profile_id=_runtime_tool_profile_id(services, rewrite_tool_decision),
                tool_registry_override=_runtime_tool_registry(services, rewrite_tool_registry),
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            revised_text = "".join(chunks)
            notes = _researcher.collect_notes(
                query=refusal_query,
                sources=docs,
                top_k=min(len(docs), _citation_top_k()),
            )
            apply_labels = _citations_enabled()
            revised_text, report, citations, bibliography, citation_enforced = _citation_payload(
                text=revised_text,
                notes=notes,
                services=services,
                apply_labels=apply_labels,
            )
            detail = _coverage_detail(report)
            q.put(
                {
                    "type": "result",
                    "payload": {
                        "revised": revised_text,
                        "citations": [item.model_dump() for item in citations],
                        "bibliography": bibliography,
                        "coverage": report.coverage,
                        "coverage_detail": detail.model_dump() if detail else None,
                        "citation_enforced": citation_enforced,
                    },
                }
            )
            logger.info(
                "rewrite stream done: session_id=%s revised_len=%s elapsed_ms=%.2f",
                payload.session_id or "__default__",
                len(revised_text or ""),
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover
            q.put({"type": "error", "detail": str(exc)})
        finally:
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
