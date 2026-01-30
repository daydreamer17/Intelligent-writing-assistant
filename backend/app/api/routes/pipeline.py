from __future__ import annotations

import json
import time
from queue import Empty, Queue
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...models.schemas import (
    CitationItemResponse,
    PipelineRequest,
    PipelineResponse,
    ResearchNoteResponse,
)
from ...services.drafting_service import DraftResult
from ...services.pipeline_service import PipelineResult
from ...services.research_service import SourceDocument
from ..deps import AppServices, get_services

router = APIRouter(tags=["pipeline"])


def _serialize_pipeline_result(
    payload: PipelineRequest,
    result,
    services: AppServices,
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
    )


@router.post("/pipeline", response_model=PipelineResponse)
def run_pipeline(
    payload: PipelineRequest,
    services: AppServices = Depends(get_services),
) -> PipelineResponse:
    try:
        sources = [
            SourceDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                url=doc.url,
            )
            for doc in payload.sources
        ]
        result = services.pipeline.run(
            topic=payload.topic,
            audience=payload.audience,
            style=payload.style,
            target_length=payload.target_length,
            constraints=payload.constraints,
            key_points=payload.key_points,
            sources=sources,
            review_criteria=payload.review_criteria,
        )
        return _serialize_pipeline_result(payload, result, services)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Pipeline execution failed") from exc


@router.post("/pipeline/stream")
def run_pipeline_stream(
    payload: PipelineRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    q: Queue[dict | None] = Queue()

    def worker() -> None:
        try:
            sources = [
                SourceDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                )
                for doc in payload.sources
                ]
            q.put({"type": "status", "step": "plan"})
            outline = services.pipeline.planner.plan_outline(
                topic=payload.topic,
                audience=payload.audience,
                style=payload.style,
                target_length=payload.target_length,
                constraints=payload.constraints,
                key_points=payload.key_points,
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

            q.put({"type": "status", "step": "research"})
            if sources:
                query = f"{payload.topic}\n{outline.outline}"
                notes = services.pipeline.researcher.collect_notes(query=query, sources=sources)
            else:
                notes = []
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

            q.put({"type": "status", "step": "draft"})
            draft_chunks: list[str] = []
            for chunk in services.drafter.writing_agent.draft_stream(
                topic=payload.topic,
                outline=outline.outline,
                constraints=payload.constraints,
                style=payload.style,
                target_length=payload.target_length,
            ):
                if not chunk:
                    continue
                draft_chunks.append(chunk)
                q.put({"type": "delta", "stage": "draft", "content": chunk})
            draft = "".join(draft_chunks)
            q.put({"type": "draft", "payload": {"draft": draft}})

            q.put({"type": "status", "step": "review"})
            review_chunks: list[str] = []
            for chunk in services.reviewer.agent.review_stream(
                draft=draft,
                criteria=payload.review_criteria,
                sources=notes_text,
                audience=payload.audience,
            ):
                if not chunk:
                    continue
                review_chunks.append(chunk)
                q.put({"type": "delta", "stage": "review", "content": chunk})
            review = "".join(review_chunks)
            q.put({"type": "review", "payload": {"review": review}})

            q.put({"type": "status", "step": "rewrite"})
            rewrite_chunks: list[str] = []
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=draft,
                guidance=review,
                style=payload.style,
                target_length=payload.target_length,
            ):
                if not chunk:
                    continue
                rewrite_chunks.append(chunk)
                q.put({"type": "delta", "stage": "rewrite", "content": chunk})
            revised = "".join(rewrite_chunks)
            q.put({"type": "rewrite", "payload": {"revised": revised}})

            q.put({"type": "status", "step": "citations"})
            draft_result = DraftResult(
                outline=outline.outline,
                research_notes=notes_text,
                draft=draft,
                review=review,
                revised=revised,
            )
            result = PipelineResult(outline=outline, research_notes=notes, draft_result=draft_result)
            response = _serialize_pipeline_result(payload, result, services)
            q.put({"type": "result", "payload": response.model_dump()})
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
