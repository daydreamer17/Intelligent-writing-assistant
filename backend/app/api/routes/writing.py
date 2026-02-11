from __future__ import annotations

import json
import os
import time
import logging
from queue import Empty, Queue
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ...models.schemas import (
    DraftRequest,
    DraftResponse,
    PlanRequest,
    PlanResponse,
    ReviewRequest,
    ReviewResponse,
    RewriteRequest,
    RewriteResponse,
)
from ...services.citation_enforcer import CitationEnforcer
from ...services.github_context import maybe_fetch_github_context
from ...services.research_service import ResearchService, RelevanceReport, SourceDocument
from ..deps import AppServices, get_services

router = APIRouter(tags=["writing"])
logger = logging.getLogger("app.writing")
_citation_enforcer = CitationEnforcer()
_researcher = ResearchService()


def _citations_enabled() -> bool:
    return os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")


def _refusal_enabled() -> bool:
    return os.getenv("RAG_REFUSAL_ENABLED", "true").lower() in ("1", "true", "yes")


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
    enforced, _ = _citation_enforcer.enforce(text, notes, apply_labels=True)
    return enforced


def _rag_refusal_check(
    *,
    query: str,
    services: AppServices,
) -> tuple[bool, list[SourceDocument]]:
    base_top_k = _citation_top_k()
    docs = services.rag.search(query, top_k=base_top_k)
    report = _researcher.relevance_report(query, docs)
    refused = _should_refuse(report)
    _log_refusal(report, refused=refused, context=f"writing:base@{base_top_k}")
    if not refused or _refusal_mode() == "strict":
        return refused, docs

    fallback_top_k = max(base_top_k, _parse_int("RAG_REFUSAL_FALLBACK_TOP_K", base_top_k * 2))
    fallback_docs = services.rag.search(query, top_k=fallback_top_k)
    fallback_report = _researcher.relevance_report(query, fallback_docs)
    fallback_refused = _should_refuse_fallback(fallback_report)
    _log_refusal(
        fallback_report,
        refused=fallback_refused,
        context=f"writing:fallback@{fallback_top_k}",
    )
    if not fallback_refused:
        return False, fallback_docs
    return True, docs


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
        github_context = maybe_fetch_github_context(
            services.github_mcp_tool,
            query=payload.topic,
            context="\n".join([payload.constraints, payload.key_points, payload.style, payload.audience]),
        )
        key_points = _append_block(payload.key_points, github_context, "GitHub参考")
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
        github_context = maybe_fetch_github_context(
            services.github_mcp_tool,
            query=payload.topic,
            context="\n".join([payload.outline, payload.constraints, payload.style, payload.target_length]),
        )
        research_notes = _append_block(payload.research_notes, github_context, "GitHub参考")
        refusal_query = f"{payload.topic}\n{payload.outline}"
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

    github_context = maybe_fetch_github_context(
        services.github_mcp_tool,
        query=payload.topic,
        context="\n".join([payload.outline, payload.constraints, payload.style, payload.target_length]),
    )
    research_notes = _append_block(payload.research_notes, github_context, "GitHub参考")

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            logger.info(
                "draft stream start: session_id=%s topic_len=%s outline_len=%s",
                payload.session_id or "__default__",
                len(payload.topic or ""),
                len(payload.outline or ""),
            )
            refusal_query = f"{payload.topic}\n{payload.outline}"
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
        github_context = maybe_fetch_github_context(
            services.github_mcp_tool,
            query=(payload.criteria or payload.draft[:200]),
            context="\n".join([payload.sources, payload.audience]),
        )
        sources = _append_block(payload.sources, github_context, "GitHub参考")
        review = services.reviewer.review(
            draft=payload.draft,
            criteria=payload.criteria,
            sources=sources,
            audience=payload.audience,
            max_tokens=_step_max_tokens("review"),
            max_input_chars=_step_max_input_chars("review"),
            session_id=payload.session_id,
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

    github_context = maybe_fetch_github_context(
        services.github_mcp_tool,
        query=(payload.criteria or payload.draft[:200]),
        context="\n".join([payload.sources, payload.audience]),
    )
    sources = _append_block(payload.sources, github_context, "GitHub参考")

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
        github_context = maybe_fetch_github_context(
            services.github_mcp_tool,
            query=(payload.guidance or payload.draft[:200]),
            context=payload.style,
        )
        guidance = _append_block(payload.guidance, github_context, "GitHub参考")
        refusal_query = payload.draft
        refused, docs = _rag_refusal_check(query=refusal_query, services=services)
        if refused:
            return RewriteResponse(revised=_refusal_message())
        evidence_text = _build_evidence_from_rag(query=payload.draft, services=services)
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
        )
        text = _enforce_with_rag(
            revised.revised,
            query=refusal_query,
            services=services,
            docs=docs,
        )
        response = RewriteResponse(revised=text)
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

    github_context = maybe_fetch_github_context(
        services.github_mcp_tool,
        query=(payload.guidance or payload.draft[:200]),
        context=payload.style,
    )
    guidance = _append_block(payload.guidance, github_context, "GitHub参考")

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            logger.info(
                "rewrite stream start: session_id=%s draft_len=%s guidance_len=%s",
                payload.session_id or "__default__",
                len(payload.draft or ""),
                len(payload.guidance or ""),
            )
            refusal_query = payload.draft
            refused, docs = _rag_refusal_check(query=refusal_query, services=services)
            if refused:
                q.put({"type": "result", "payload": {"revised": _refusal_message()}})
                return
            evidence_text = _build_evidence_from_rag(query=payload.draft, services=services)
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
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            revised_text = "".join(chunks)
            revised_text = _enforce_with_rag(
                revised_text,
                query=refusal_query,
                services=services,
                docs=docs,
            )
            q.put({"type": "result", "payload": {"revised": revised_text}})
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
