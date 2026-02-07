from __future__ import annotations

import json
import time
import logging
import os
from queue import Empty, Queue
from threading import Thread

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
from ...services.github_context import maybe_fetch_github_context
from ...services.pipeline_service import PipelineResult
from ...services.research_service import ResearchNote, SourceDocument
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


def _refusal_enabled() -> bool:
    return os.getenv("RAG_REFUSAL_ENABLED", "true").lower() in ("1", "true", "yes")


def _should_refuse(query: str, sources: list[SourceDocument], researcher) -> bool:
    if not _refusal_enabled():
        return False
    report = researcher.relevance_report(query, sources)
    min_terms = max(1, _parse_int("RAG_REFUSAL_MIN_QUERY_TERMS", 2))
    min_docs = max(1, _parse_int("RAG_REFUSAL_MIN_DOCS", 1))
    min_best = _parse_float("RAG_REFUSAL_MIN_RECALL", 0.1)
    min_avg = _parse_float("RAG_REFUSAL_MIN_AVG_RECALL", 0.05)
    if report.query_terms < min_terms:
        logger.info(
            "RAG refusal check skipped: terms=%s < min_terms=%s",
            report.query_terms,
            min_terms,
        )
        return False
    refused = report.docs < min_docs or report.best_recall < min_best or report.avg_recall < min_avg
    logger.info(
        "RAG refusal check: docs=%s terms=%s best=%.3f avg=%.3f -> %s",
        report.docs,
        report.query_terms,
        report.best_recall,
        report.avg_recall,
        "refuse" if refused else "pass",
    )
    return refused


def _github_note(context: str) -> ResearchNote:
    summary = (context or "").strip().replace("\n", " ")
    return ResearchNote(
        doc_id="github:mcp",
        title="GitHub MCP Context",
        summary=summary[:600],
        url="",
    )


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
        github_context = maybe_fetch_github_context(
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
        result = services.pipeline.run(
            topic=payload.topic,
            audience=payload.audience,
            style=payload.style,
            target_length=payload.target_length,
            constraints=constraints,
            key_points=payload.key_points,
            sources=sources,
            review_criteria=payload.review_criteria,
        )
        if github_context and not any(note.doc_id == "github:mcp" for note in result.research_notes):
            result.research_notes.append(_github_note(github_context))
        apply_labels = os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")
        refusal_query = f"{payload.topic}\n{result.outline.outline}"
        if apply_labels and _should_refuse(refusal_query, sources, services.pipeline.researcher):
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
            return _serialize_pipeline_result(
                payload,
                result,
                services,
                coverage=0.0,
                coverage_detail=CoverageDetail(),
                citation_enforced=False,
            )
        enforced_text, report = _citation_enforcer.enforce(
            result.draft_result.revised or result.draft_result.draft,
            result.research_notes,
            apply_labels=apply_labels,
            embedder=services.rag.get_embedder(),
        )
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
        return _serialize_pipeline_result(
            payload,
            result,
            services,
            coverage=report.coverage,
            coverage_detail=_coverage_detail(report),
            citation_enforced=citation_enforced,
        )
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
        try:
            github_context = maybe_fetch_github_context(
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
            q.put({"type": "status", "step": "plan"})
            outline = services.pipeline.planner.plan_outline(
                topic=payload.topic,
                audience=payload.audience,
                style=payload.style,
                target_length=payload.target_length,
                constraints=constraints,
                key_points=payload.key_points,
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

            q.put({"type": "status", "step": "research"})
            if sources:
                notes = services.pipeline.collect_research_notes(payload.topic, outline, sources)
            else:
                notes = []
            if github_context and not any(note.doc_id == "github:mcp" for note in notes):
                notes.append(_github_note(github_context))
            task_status["research"] = None if not sources else bool(notes)
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

            apply_labels = os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")
            refusal_query = f"{payload.topic}\n{outline.outline}"
            if apply_labels and _should_refuse(refusal_query, sources, services.pipeline.researcher):
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
                return

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
            ):
                if not chunk:
                    continue
                draft_chunks.append(chunk)
                q.put({"type": "delta", "stage": "draft", "content": chunk})
            draft = "".join(draft_chunks)
            task_status["draft"] = _is_effective(draft)
            q.put({"type": "draft", "payload": {"draft": draft}})

            q.put({"type": "status", "step": "review"})
            review_chunks: list[str] = []
            # 根据 draft 长度动态设置 max_tokens，避免截断
            review_max_tokens = min(8000, max(2000, int(len(draft) * 0.8)))
            for chunk in services.reviewer.agent.review_stream(
                draft=draft,
                criteria=payload.review_criteria,
                sources=notes_text,
                audience=payload.audience,
                max_tokens=review_max_tokens,
            ):
                if not chunk:
                    continue
                review_chunks.append(chunk)
                q.put({"type": "delta", "stage": "review", "content": chunk})
            review = "".join(review_chunks)
            task_status["review"] = bool(review.strip())
            q.put({"type": "review", "payload": {"review": review}})

            q.put({"type": "status", "step": "rewrite"})
            rewrite_chunks: list[str] = []
            # 根据 draft 长度动态设置 max_tokens，避免截断
            rewrite_max_tokens = min(8000, max(2000, int(len(draft) * 1.5)))
            rewrite_guidance = review
            if evidence_text:
                rewrite_guidance = (review + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=draft,
                guidance=rewrite_guidance,
                style=payload.style,
                target_length=payload.target_length,
                max_tokens=rewrite_max_tokens,
            ):
                if not chunk:
                    continue
                rewrite_chunks.append(chunk)
                q.put({"type": "delta", "stage": "rewrite", "content": chunk})
            revised = "".join(rewrite_chunks)
            apply_labels = os.getenv("RAG_CITATION_ENFORCE", "false").lower() in ("1", "true", "yes")
            enforced_text, report = _citation_enforcer.enforce(
                revised or draft,
                notes,
                apply_labels=apply_labels,
                embedder=services.rag.get_embedder(),
            )
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
            q.put({"type": "result", "payload": response.model_dump()})
        except Exception as exc:  # pragma: no cover
            q.put({"type": "error", "detail": str(exc)})
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
