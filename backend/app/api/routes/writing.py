from __future__ import annotations

import json
import time
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
from ..deps import AppServices, get_services

router = APIRouter(tags=["writing"])


@router.post("/plan", response_model=PlanResponse)
def create_plan(
    payload: PlanRequest,
    services: AppServices = Depends(get_services),
) -> PlanResponse:
    try:
        plan = services.planner.plan_outline(
            topic=payload.topic,
            audience=payload.audience,
            style=payload.style,
            target_length=payload.target_length,
            constraints=payload.constraints,
            key_points=payload.key_points,
        )
        return PlanResponse(
            outline=plan.outline,
            assumptions=plan.assumptions,
            open_questions=plan.open_questions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Plan generation failed") from exc


@router.post("/draft", response_model=DraftResponse)
def create_draft(
    payload: DraftRequest,
    services: AppServices = Depends(get_services),
) -> DraftResponse:
    try:
        draft = services.drafter.create_draft(
            topic=payload.topic,
            outline=payload.outline,
            research_notes=payload.research_notes,
            constraints=payload.constraints,
            style=payload.style,
            target_length=payload.target_length,
        )
        return DraftResponse(draft=draft)
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

    def worker() -> None:
        try:
            chunks: list[str] = []
            for chunk in services.drafter.writing_agent.draft_stream(
                topic=payload.topic,
                outline=payload.outline,
                constraints=payload.constraints,
                style=payload.style,
                target_length=payload.target_length,
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            draft = "".join(chunks)
            q.put({"type": "result", "payload": {"draft": draft}})
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
        review = services.reviewer.review(
            draft=payload.draft,
            criteria=payload.criteria,
            sources=payload.sources,
            audience=payload.audience,
        )
        return ReviewResponse(review=review.review)
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

    def worker() -> None:
        try:
            chunks: list[str] = []
            for chunk in services.reviewer.agent.review_stream(
                draft=payload.draft,
                criteria=payload.criteria,
                sources=payload.sources,
                audience=payload.audience,
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            review_text = "".join(chunks)
            q.put({"type": "result", "payload": {"review": review_text}})
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
        revised = services.rewriter.rewrite(
            draft=payload.draft,
            guidance=payload.guidance,
            style=payload.style,
            target_length=payload.target_length,
        )
        return RewriteResponse(revised=revised.revised)
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

    def worker() -> None:
        try:
            chunks: list[str] = []
            for chunk in services.rewriter.agent.rewrite_stream(
                draft=payload.draft,
                guidance=payload.guidance,
                style=payload.style,
                target_length=payload.target_length,
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                q.put({"type": "delta", "content": chunk})
            revised_text = "".join(chunks)
            q.put({"type": "result", "payload": {"revised": revised_text}})
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
