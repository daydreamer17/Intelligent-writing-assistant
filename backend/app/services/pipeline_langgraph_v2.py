from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

from ..api.deps import get_services
from .planner_service import OutlinePlan
from .research_service import ResearchNote, SourceDocument
from .tool_policy import build_stage_tool_registry

logger = logging.getLogger("app.pipeline.v2")


class SQLitePersistentSaver(InMemorySaver):
    """Minimal SQLite-backed saver for v2 interrupt/resume persistence."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(_resolve_checkpoint_db_path(db_path))
        self._lock = RLock()
        super().__init__()
        self._ensure_schema()
        self._load_from_disk()

    def _ensure_schema(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_v2_checkpoint_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload BLOB NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_v2_checkpoint_index (
                    thread_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'sync',
                    status TEXT NOT NULL DEFAULT 'interrupted',
                    current_stage TEXT NOT NULL DEFAULT 'outline_review',
                    outline_preview TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_v2_resume_state (
                    thread_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _load_from_disk(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM langgraph_v2_checkpoint_state WHERE id = 1"
            ).fetchone()
            if not row:
                return
            payload = pickle.loads(row[0])
            self.storage = _restore_storage(payload.get("storage") or {})
            self.writes = defaultdict(dict, payload.get("writes") or {})
            self.blobs = dict(payload.get("blobs") or {})

    def _persist_to_disk(self) -> None:
        payload = pickle.dumps(
            {
                "storage": _flatten_storage(self.storage),
                "writes": dict(self.writes),
                "blobs": dict(self.blobs),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO langgraph_v2_checkpoint_state (id, payload, updated_at)
                VALUES (1, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (payload,),
            )
            conn.commit()

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist_to_disk()
            return result

    def put_writes(self, config, writes, task_id, task_path=""):  # type: ignore[override]
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist_to_disk()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._persist_to_disk()
        self.delete_checkpoint(thread_id)

    def clear_graph_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._persist_to_disk()

    def upsert_checkpoint_index(
        self,
        *,
        thread_id: str,
        session_id: str = "",
        mode: str = "sync",
        status: str,
        current_stage: str,
        outline_preview: str = "",
        last_error: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO langgraph_v2_checkpoint_index (
                    thread_id, session_id, mode, status, current_stage, outline_preview, last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    mode = excluded.mode,
                    status = excluded.status,
                    current_stage = excluded.current_stage,
                    outline_preview = excluded.outline_preview,
                    last_error = excluded.last_error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    thread_id,
                    session_id or "",
                    mode,
                    status,
                    current_stage,
                    outline_preview,
                    last_error,
                ),
            )
            conn.commit()

    def mark_checkpoint_failed(self, *, thread_id: str, current_stage: str, last_error: str) -> None:
        existing = self.get_checkpoint_index(thread_id) or {}
        self.upsert_checkpoint_index(
            thread_id=thread_id,
            session_id=str(existing.get("session_id") or ""),
            mode=str(existing.get("mode") or "sync"),
            status="failed",
            current_stage=current_stage,
            outline_preview=str(existing.get("outline_preview") or ""),
            last_error=last_error,
        )

    def mark_checkpoint_completed(self, *, thread_id: str, current_stage: str = "completed") -> None:
        existing = self.get_checkpoint_index(thread_id) or {}
        self.upsert_checkpoint_index(
            thread_id=thread_id,
            session_id=str(existing.get("session_id") or ""),
            mode=str(existing.get("mode") or "sync"),
            status="completed",
            current_stage=current_stage,
            outline_preview=str(existing.get("outline_preview") or ""),
            last_error="",
        )

    def get_checkpoint_index(self, thread_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM langgraph_v2_checkpoint_index WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_checkpoints(
        self,
        *,
        limit: int = 20,
        status: str = "all",
        thread_id: str = "",
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        query = "SELECT * FROM langgraph_v2_checkpoint_index"
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def save_resume_state(
        self,
        *,
        thread_id: str,
        status: str,
        current_stage: str,
        payload: dict[str, Any],
    ) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO langgraph_v2_resume_state (
                    thread_id, status, current_stage, payload_json
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    status = excluded.status,
                    current_stage = excluded.current_stage,
                    payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, status, current_stage, payload_json),
            )
            conn.commit()

    def load_resume_state(self, thread_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM langgraph_v2_resume_state WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["payload"] = json.loads(data.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            data["payload"] = {}
        return data

    def clear_resume_state(self, thread_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM langgraph_v2_resume_state WHERE thread_id = ?",
                (thread_id,),
            )
            conn.commit()

    def delete_checkpoint(self, thread_id: str) -> bool:
        deleted = False
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM langgraph_v2_checkpoint_index WHERE thread_id = ?",
                (thread_id,),
            )
            deleted = cursor.rowcount > 0
            conn.execute(
                "DELETE FROM langgraph_v2_resume_state WHERE thread_id = ?",
                (thread_id,),
            )
            conn.commit()
        return deleted

    def cleanup_checkpoints(
        self,
        *,
        older_than_hours: int = 168,
        status: str = "completed",
        dry_run: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        threshold = datetime.now(timezone.utc) - timedelta(hours=max(1, older_than_hours))
        threshold_text = threshold.strftime("%Y-%m-%d %H:%M:%S")
        query = "SELECT thread_id, status, updated_at FROM langgraph_v2_checkpoint_index WHERE updated_at < ?"
        params: list[Any] = [threshold_text]
        if status and status != "all":
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at ASC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        thread_ids = [str(row["thread_id"]) for row in rows]
        deleted = 0
        if not dry_run:
            for thread_id in thread_ids:
                super().delete_thread(thread_id)
                self.delete_checkpoint(thread_id)
                deleted += 1
            with self._lock:
                self._persist_to_disk()
        return {
            "dry_run": dry_run,
            "matched": len(thread_ids),
            "deleted": deleted,
            "thread_ids": thread_ids,
            "older_than_hours": older_than_hours,
            "status": status,
        }


def _resolve_checkpoint_db_path(db_path: str | Path | None = None) -> Path:
    load_dotenv()
    raw_path = str(
        db_path
        or os.getenv("LANGGRAPH_V2_CHECKPOINT_DB", "data/langgraph_v2_checkpoints.sqlite")
    ).strip()
    path = Path(raw_path)
    if not path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        path = backend_root / path
    return path


def _flatten_storage(storage: Any) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for thread_id, namespaces in dict(storage).items():
        result[str(thread_id)] = {
            str(namespace): dict(checkpoints)
            for namespace, checkpoints in dict(namespaces).items()
        }
    return result


def _restore_storage(raw_storage: dict[str, dict[str, dict[str, Any]]]) -> defaultdict[str, defaultdict[str, dict[str, Any]]]:
    storage: defaultdict[str, defaultdict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for thread_id, namespaces in raw_storage.items():
        storage[thread_id] = defaultdict(
            dict,
            {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()},
        )
    return storage


def _create_default_checkpointer(db_path: str | Path | None = None) -> SQLitePersistentSaver:
    resolved_path = _resolve_checkpoint_db_path(db_path)
    saver = SQLitePersistentSaver(resolved_path)
    logger.info(
        "pipeline v2 workflow: checkpointer=%s db_path=%s",
        type(saver).__name__,
        saver.db_path,
    )
    return saver


_CHECKPOINTER = _create_default_checkpointer()


def _get_checkpointer() -> SQLitePersistentSaver:
    return _CHECKPOINTER


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _safe_outline_preview(text: str, limit: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip()


def _interrupt_payload_from_state(thread_id: str) -> dict[str, Any] | None:
    try:
        snapshot = pipeline_v2_workflow.get_state(_thread_config(thread_id))
    except Exception:
        return None
    if not snapshot or not getattr(snapshot, "interrupts", None):
        return None
    payload = dict(getattr(snapshot.interrupts[0], "value", {}) or {})
    if payload:
        payload.setdefault("kind", "outline_review")
        payload.setdefault("thread_id", thread_id)
    return payload or None


def has_pipeline_v2_interrupt_checkpoint(thread_id: str) -> bool:
    return _interrupt_payload_from_state(thread_id) is not None


def get_pipeline_v2_interrupt_payload(thread_id: str) -> dict[str, Any] | None:
    return _interrupt_payload_from_state(thread_id)


def upsert_pipeline_v2_checkpoint(
    *,
    thread_id: str,
    session_id: str = "",
    mode: str = "sync",
    status: str,
    current_stage: str,
    outline_preview: str = "",
    last_error: str = "",
) -> None:
    _get_checkpointer().upsert_checkpoint_index(
        thread_id=thread_id,
        session_id=session_id,
        mode=mode,
        status=status,
        current_stage=current_stage,
        outline_preview=_safe_outline_preview(outline_preview),
        last_error=last_error,
    )


def mark_pipeline_v2_checkpoint_failed(*, thread_id: str, current_stage: str, last_error: str) -> None:
    _get_checkpointer().mark_checkpoint_failed(
        thread_id=thread_id,
        current_stage=current_stage,
        last_error=last_error,
    )


def mark_pipeline_v2_checkpoint_completed(*, thread_id: str, current_stage: str = "completed") -> None:
    _get_checkpointer().mark_checkpoint_completed(thread_id=thread_id, current_stage=current_stage)


def list_pipeline_v2_checkpoints(*, limit: int = 20, status: str = "all", thread_id: str = "") -> list[dict[str, Any]]:
    return _get_checkpointer().list_checkpoints(limit=limit, status=status, thread_id=thread_id)


def load_pipeline_v2_resume_state(thread_id: str) -> dict[str, Any] | None:
    return _get_checkpointer().load_resume_state(thread_id)


def save_pipeline_v2_resume_state(
    *,
    thread_id: str,
    session_id: str,
    mode: str,
    status: str,
    current_stage: str,
    payload: dict[str, Any],
) -> None:
    _get_checkpointer().save_resume_state(
        thread_id=thread_id,
        status=status,
        current_stage=current_stage,
        payload=payload,
    )
    outline = str(payload.get("outline") or "")
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=session_id,
        mode=mode,
        status=status,
        current_stage=current_stage,
        outline_preview=outline,
        last_error="",
    )


def clear_pipeline_v2_resume_state(thread_id: str) -> None:
    _get_checkpointer().clear_resume_state(thread_id)


def delete_pipeline_v2_checkpoint(thread_id: str) -> bool:
    deleted = _get_checkpointer().delete_checkpoint(thread_id)
    try:
        _get_checkpointer().delete_thread(thread_id)
        deleted = True
    except Exception:
        pass
    return deleted


def clear_pipeline_v2_graph_thread(thread_id: str) -> None:
    _get_checkpointer().clear_graph_thread(thread_id)


def cleanup_pipeline_v2_checkpoints(
    *,
    older_than_hours: int = 168,
    status: str = "completed",
    dry_run: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    return _get_checkpointer().cleanup_checkpoints(
        older_than_hours=older_than_hours,
        status=status,
        dry_run=dry_run,
        limit=limit,
    )


def get_pipeline_v2_checkpoint_detail(thread_id: str) -> dict[str, Any] | None:
    index_row = _get_checkpointer().get_checkpoint_index(thread_id)
    resume_state = load_pipeline_v2_resume_state(thread_id)
    interrupt_payload = get_pipeline_v2_interrupt_payload(thread_id)
    if not index_row and not resume_state and not interrupt_payload:
        return None
    payload = dict(resume_state.get("payload") or {}) if resume_state else {}
    index_stage = str((index_row or {}).get("current_stage") or "")
    index_status = str((index_row or {}).get("status") or "")
    needs_rewrite_raw = payload.get("needs_rewrite")
    needs_rewrite = None if needs_rewrite_raw is None else bool(needs_rewrite_raw)
    score_raw = payload.get("score")
    score = None if score_raw is None else float(score_raw)
    if interrupt_payload:
        outline = str(interrupt_payload.get("outline") or "")
        assumptions = str(interrupt_payload.get("assumptions") or "")
        open_questions = str(interrupt_payload.get("open_questions") or "")
        can_resume = True
        status = index_status or "interrupted"
        current_stage = index_stage or "outline_review"
        interrupt_stage = "outline_review"
        draft = ""
        review_text = ""
        reason = ""
        score = None
        needs_rewrite = None
    elif index_stage == "draft_review":
        outline = str(payload.get("outline") or (index_row or {}).get("outline_preview") or "")
        assumptions = str(payload.get("assumptions") or "")
        open_questions = str(payload.get("open_questions") or "")
        draft = str(payload.get("draft") or "")
        can_resume = True
        status = index_status or "interrupted"
        current_stage = "draft_review"
        interrupt_stage = "draft_review"
        review_text = ""
        reason = ""
        score = None
        needs_rewrite = None
    elif index_stage == "review_confirmation":
        outline = str(payload.get("outline") or (index_row or {}).get("outline_preview") or "")
        assumptions = str(payload.get("assumptions") or "")
        open_questions = str(payload.get("open_questions") or "")
        draft = str(payload.get("draft") or "")
        can_resume = True
        status = index_status or "interrupted"
        current_stage = "review_confirmation"
        interrupt_stage = "review_confirmation"
        review_text = str(payload.get("review_text") or payload.get("review") or "")
        reason = str(payload.get("reason") or "")
    else:
        outline = str(payload.get("outline") or (index_row or {}).get("outline_preview") or "")
        assumptions = str(payload.get("assumptions") or "")
        open_questions = str(payload.get("open_questions") or "")
        draft = str(payload.get("draft") or "")
        review_text = str(payload.get("review_text") or payload.get("review") or "")
        reason = str(payload.get("reason") or "")
        status = str((resume_state or {}).get("status") or index_status or "unknown")
        current_stage = str((resume_state or {}).get("current_stage") or index_stage or "unknown")
        can_resume = status != "completed" and bool(resume_state)
        interrupt_stage = ""
    return {
        "thread_id": thread_id,
        "session_id": str((index_row or {}).get("session_id") or payload.get("resolved_session_id") or ""),
        "mode": str((index_row or {}).get("mode") or "sync"),
        "status": status,
        "current_stage": current_stage,
        "interrupt_stage": interrupt_stage,
        "created_at": str((resume_state or {}).get("created_at") or (index_row or {}).get("created_at") or ""),
        "updated_at": str((resume_state or {}).get("updated_at") or (index_row or {}).get("updated_at") or ""),
        "can_resume": can_resume,
        "outline": outline,
        "draft": draft,
        "review_text": review_text,
        "needs_rewrite": needs_rewrite,
        "reason": reason,
        "score": score,
        "assumptions": assumptions,
        "open_questions": open_questions,
        "last_error": str((index_row or {}).get("last_error") or ""),
    }


def _pipeline_route_helpers():
    from ..api.routes import pipeline as pipeline_route

    return pipeline_route


def _restore_tail_notes(raw_notes: Any) -> list[ResearchNote]:
    notes: list[ResearchNote] = []
    for raw_note in raw_notes or []:
        if not isinstance(raw_note, dict):
            continue
        notes.append(
            ResearchNote(
                doc_id=str(raw_note.get("doc_id") or ""),
                title=str(raw_note.get("title") or ""),
                summary=str(raw_note.get("summary") or ""),
                url=str(raw_note.get("url") or ""),
            )
        )
    return notes


def _base_tail_resume_payload(state: dict[str, Any]) -> dict[str, Any]:
    review_text = str(state.get("review_text") or state.get("review") or "")
    return {
        "request": dict(state.get("request") or {}),
        "resolved_session_id": str(state.get("resolved_session_id") or ""),
        "effective_constraints": str(state.get("effective_constraints") or ""),
        "github_context": str(state.get("github_context") or ""),
        "outline": str(state.get("outline") or ""),
        "assumptions": str(state.get("assumptions") or ""),
        "open_questions": str(state.get("open_questions") or ""),
        "research_notes": list(state.get("research_notes") or []),
        "notes_text": str(state.get("notes_text") or ""),
        "draft": str(state.get("draft") or ""),
        "review": review_text,
        "review_text": review_text,
        "needs_rewrite": state.get("needs_rewrite"),
        "reason": str(state.get("reason") or ""),
        "score": state.get("score"),
    }


def _save_tail_resume_boundary(
    state: dict[str, Any],
    *,
    current_stage: str,
    status: str = "running",
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload = _base_tail_resume_payload(state)
    if extra_payload:
        payload.update(extra_payload)
    save_pipeline_v2_resume_state(
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("resolved_session_id") or ""),
        mode=str(state.get("mode") or "sync"),
        status=status,
        current_stage=current_stage,
        payload=payload,
    )


def _mark_draft_review_interrupt(state: dict[str, Any]) -> None:
    payload = _base_tail_resume_payload(state)
    save_pipeline_v2_resume_state(
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("resolved_session_id") or ""),
        mode=str(state.get("mode") or "sync"),
        status="interrupted",
        current_stage="draft_done",
        payload=payload,
    )
    upsert_pipeline_v2_checkpoint(
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("resolved_session_id") or ""),
        mode=str(state.get("mode") or "sync"),
        status="interrupted",
        current_stage="draft_review",
        outline_preview=str(state.get("outline") or ""),
        last_error="",
    )


def _mark_review_confirmation_interrupt(state: dict[str, Any]) -> None:
    payload = _base_tail_resume_payload(state)
    save_pipeline_v2_resume_state(
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("resolved_session_id") or ""),
        mode=str(state.get("mode") or "sync"),
        status="interrupted",
        current_stage="review_done",
        payload=payload,
    )
    upsert_pipeline_v2_checkpoint(
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("resolved_session_id") or ""),
        mode=str(state.get("mode") or "sync"),
        status="interrupted",
        current_stage="review_confirmation",
        outline_preview=str(state.get("outline") or ""),
        last_error="",
    )


def _build_tool_runtime(
    services,
    *,
    allowed_tool_names: list[str] | None,
) -> Any:
    return build_stage_tool_registry(
        tool_catalog=services.agent_tools_catalog,
        allowed_tool_names=list(allowed_tool_names or []),
    )


def _sources_from_request_payload(request_data: dict[str, Any], github_context: str) -> list[SourceDocument]:
    sources: list[SourceDocument] = []
    for raw_doc in list(request_data.get("sources") or []):
        if not isinstance(raw_doc, dict):
            continue
        sources.append(
            SourceDocument(
                doc_id=str(raw_doc.get("doc_id") or ""),
                title=str(raw_doc.get("title") or ""),
                content=str(raw_doc.get("content") or ""),
                url=str(raw_doc.get("url") or ""),
            )
        )
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


def _graph_task_status(state: dict[str, Any]) -> dict[str, bool | None]:
    task_status = dict(state.get("task_status") or {})
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


def _research_step(state: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    request_data = dict(state.get("request") or {})
    request_model = route.PipelineRequest.model_validate(request_data)
    outline = OutlinePlan(
        outline=str(state.get("outline") or ""),
        assumptions=str(state.get("assumptions") or ""),
        open_questions=str(state.get("open_questions") or ""),
    )
    github_context = str(state.get("github_context") or "")
    sources = _sources_from_request_payload(request_data, github_context)
    task_status = _graph_task_status(state)
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "sync"),
        status="running",
        current_stage="research",
        outline_preview=str(state.get("outline") or ""),
    )
    research_state = route._run_pipeline_research_sync(
        payload=request_model,
        services=services,
        outline=outline,
        sources=sources,
        github_context=github_context,
        effective_constraints=str(state.get("effective_constraints") or request_model.constraints or ""),
        resolved_session_id=resolved_session_id,
        task_status=task_status,
    )
    next_state = {
        **state,
        "task_status": task_status,
        "research_notes": route._serialize_resume_notes(list(research_state.notes)),
        "notes_text": research_state.notes_text,
        "source_count": len(research_state.effective_sources),
    }
    if research_state.final_response is not None:
        response = research_state.final_response
        _save_tail_resume_boundary(
            next_state,
            current_stage="completed",
            status="completed",
            extra_payload={
                "response": response.model_dump(),
                "coverage": float(response.coverage or 0.0),
                "coverage_detail": (
                    response.coverage_detail.model_dump()
                    if response.coverage_detail is not None
                    else route.CoverageDetail().model_dump()
                ),
            },
        )
        mark_pipeline_v2_checkpoint_completed(thread_id=thread_id)
        return {
            **next_state,
            "response": response.model_dump(),
            "coverage": float(response.coverage or 0.0),
            "coverage_detail": (
                response.coverage_detail.model_dump()
                if response.coverage_detail is not None
                else route.CoverageDetail().model_dump()
            ),
        }
    _save_tail_resume_boundary(
        next_state,
        current_stage="research_done",
        extra_payload={
            "research_notes": list(next_state.get("research_notes") or []),
            "notes_text": str(next_state.get("notes_text") or ""),
        },
    )
    return next_state


def _draft_step(state: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    request_model = route.PipelineRequest.model_validate(dict(state.get("request") or {}))
    outline = OutlinePlan(
        outline=str(state.get("outline") or ""),
        assumptions=str(state.get("assumptions") or ""),
        open_questions=str(state.get("open_questions") or ""),
    )
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "sync"),
        status="running",
        current_stage="draft",
        outline_preview=str(state.get("outline") or ""),
    )
    draft = route._run_pipeline_draft_sync(
        payload=request_model,
        services=services,
        outline=outline,
        effective_constraints=str(state.get("effective_constraints") or request_model.constraints or ""),
        resolved_session_id=resolved_session_id,
        notes_text=str(state.get("notes_text") or ""),
        source_count=int(state.get("source_count") or 0),
    )
    task_status = _graph_task_status(state)
    task_status["draft"] = route._is_effective(draft)
    next_state = {
        **state,
        "task_status": task_status,
        "draft": draft,
    }
    _save_tail_resume_boundary(
        next_state,
        current_stage="draft_done",
        extra_payload={
            "research_notes": list(next_state.get("research_notes") or []),
            "notes_text": str(next_state.get("notes_text") or ""),
            "draft": draft,
        },
    )
    return next_state


def _review_tail_step(state: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "sync"),
        status="running",
        current_stage="review",
        outline_preview=str(state.get("outline") or ""),
    )
    review_registry = _build_tool_runtime(
        services,
        allowed_tool_names=list(state.get("review_tool_allowed_tools") or []),
    )
    review_profile_id = str(state.get("review_tool_profile_id") or "").strip() or None
    decision = services.reviewer.review_decision(
        draft=str(state.get("draft") or ""),
        criteria=str(state.get("review_criteria") or ""),
        sources=str(state.get("notes_text") or ""),
        audience=str(state.get("audience") or ""),
        max_tokens=route._dynamic_review_max_tokens(str(state.get("draft") or "")),
        max_input_chars=route._pipeline_max_input_chars("review"),
        session_id=resolved_session_id,
        tool_profile_id=review_profile_id,
        tool_registry_override=review_registry,
    )
    next_state = {
        **state,
        "task_status": {
            **_graph_task_status(state),
            "review": bool(str(decision.review_text or "").strip()),
        },
        "review": decision.review_text,
        "review_text": decision.review_text,
        "needs_rewrite": decision.needs_rewrite,
        "reason": decision.reason,
        "score": decision.score,
    }
    _save_tail_resume_boundary(
        next_state,
        current_stage="review_done",
        extra_payload={
            "review": decision.review_text,
            "review_text": decision.review_text,
            "needs_rewrite": decision.needs_rewrite,
            "reason": decision.reason,
            "score": decision.score,
        },
    )
    return next_state


def _review_tail_stream_step(*, state: dict[str, Any], q: Queue[dict[str, Any] | None]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "stream"),
        status="running",
        current_stage="review",
        outline_preview=str(state.get("outline") or ""),
    )
    review_registry = _build_tool_runtime(
        services,
        allowed_tool_names=list(state.get("review_tool_allowed_tools") or []),
    )
    review_profile_id = str(state.get("review_tool_profile_id") or "").strip() or None
    draft = str(state.get("draft") or "")
    review_chunks: list[str] = []
    for chunk in services.reviewer.agent.review_stream(
        draft=draft,
        criteria=str(state.get("review_criteria") or ""),
        sources=str(state.get("notes_text") or ""),
        audience=str(state.get("audience") or ""),
        max_tokens=route._dynamic_review_max_tokens(draft),
        max_input_chars=route._pipeline_max_input_chars("review"),
        session_id=resolved_session_id,
        tool_profile_id=review_profile_id,
        tool_registry_override=review_registry,
    ):
        if not chunk:
            continue
        review_chunks.append(chunk)
        q.put({"type": "delta", "stage": "review", "content": chunk})
    review = "".join(review_chunks)
    if not review.strip():
        review = services.reviewer.review(
            draft=draft,
            criteria=str(state.get("review_criteria") or ""),
            sources=str(state.get("notes_text") or ""),
            audience=str(state.get("audience") or ""),
            max_tokens=route._dynamic_review_max_tokens(draft),
            max_input_chars=route._pipeline_max_input_chars("review"),
            session_id=resolved_session_id,
            tool_profile_id=review_profile_id,
            tool_registry_override=review_registry,
        ).review
        route._emit_text_deltas(q, "review", review)
    decision = services.reviewer.review_decision_from_review(
        review=review,
        criteria=str(state.get("review_criteria") or ""),
        audience=str(state.get("audience") or ""),
        max_tokens=route._dynamic_review_max_tokens(review),
        max_input_chars=route._pipeline_max_input_chars("review"),
        session_id=resolved_session_id,
    )
    next_state = {
        **state,
        "task_status": {
            **_graph_task_status(state),
            "review": bool(review.strip()),
        },
        "review": decision.review_text,
        "review_text": decision.review_text,
        "needs_rewrite": decision.needs_rewrite,
        "reason": decision.reason,
        "score": decision.score,
    }
    _save_tail_resume_boundary(
        next_state,
        current_stage="review_done",
        extra_payload={
            "review": decision.review_text,
            "review_text": decision.review_text,
            "needs_rewrite": decision.needs_rewrite,
            "reason": decision.reason,
            "score": decision.score,
        },
    )
    q.put(
        {
            "type": "review_decision",
            "payload": {
                "review_text": decision.review_text,
                "needs_rewrite": decision.needs_rewrite,
                "reason": decision.reason,
                "score": decision.score,
            },
        }
    )
    q.put({"type": "review", "payload": {"review": decision.review_text}})
    return next_state


def _rewrite_tail_step(state: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "sync"),
        status="running",
        current_stage="rewrite",
        outline_preview=str(state.get("outline") or ""),
    )
    rewrite_registry = _build_tool_runtime(
        services,
        allowed_tool_names=list(state.get("rewrite_tool_allowed_tools") or []),
    )
    rewrite_profile_id = str(state.get("rewrite_tool_profile_id") or "").strip() or None
    notes_text = str(state.get("notes_text") or "")
    draft = str(state.get("draft") or "")
    review = str(state.get("review") or "")
    evidence_text = services.drafter.extract_evidence(notes_text)
    revised = services.rewriter.rewrite(
        draft=draft,
        guidance=route._rewrite_guidance(review, str(state.get("review_criteria") or ""))
        + (("\n\nOnly use the evidence below:\n" + evidence_text) if evidence_text else ""),
        style=str(state.get("style") or ""),
        target_length=str(state.get("target_length") or ""),
        max_tokens=route._dynamic_rewrite_max_tokens(draft),
        max_input_chars=route._pipeline_max_input_chars("rewrite"),
        session_id=resolved_session_id,
        tool_profile_id=rewrite_profile_id,
        tool_registry_override=rewrite_registry,
    ).revised
    task_status = _graph_task_status(state)
    task_status["rewrite"] = route._is_effective(revised)
    next_state = {
        **state,
        "task_status": task_status,
        "revised_candidate": revised,
    }
    _save_tail_resume_boundary(
        next_state,
        current_stage="rewrite_done",
        extra_payload={
            "revised": revised,
        },
    )
    return next_state


def _rewrite_tail_stream_step(*, state: dict[str, Any], q: Queue[dict[str, Any] | None]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "stream"),
        status="running",
        current_stage="rewrite",
        outline_preview=str(state.get("outline") or ""),
    )
    rewrite_registry = _build_tool_runtime(
        services,
        allowed_tool_names=list(state.get("rewrite_tool_allowed_tools") or []),
    )
    rewrite_profile_id = str(state.get("rewrite_tool_profile_id") or "").strip() or None
    notes_text = str(state.get("notes_text") or "")
    draft = str(state.get("draft") or "")
    review = str(state.get("review") or "")
    evidence_text = services.drafter.extract_evidence(notes_text)
    guidance = route._rewrite_guidance(review, str(state.get("review_criteria") or ""))
    if evidence_text:
        guidance = (guidance + "\n\nOnly use the evidence below:\n" + evidence_text).strip()
    rewrite_chunks: list[str] = []
    for chunk in services.rewriter.agent.rewrite_stream(
        draft=draft,
        guidance=guidance,
        style=str(state.get("style") or ""),
        target_length=str(state.get("target_length") or ""),
        max_tokens=route._dynamic_rewrite_max_tokens(draft),
        max_input_chars=route._pipeline_max_input_chars("rewrite"),
        session_id=resolved_session_id,
        tool_profile_id=rewrite_profile_id,
        tool_registry_override=rewrite_registry,
    ):
        if not chunk:
            continue
        rewrite_chunks.append(chunk)
        q.put({"type": "delta", "stage": "rewrite", "content": chunk})
    revised = "".join(rewrite_chunks)
    if not revised.strip():
        revised = services.rewriter.rewrite(
            draft=draft,
            guidance=guidance,
            style=str(state.get("style") or ""),
            target_length=str(state.get("target_length") or ""),
            max_tokens=route._dynamic_rewrite_max_tokens(draft),
            max_input_chars=route._pipeline_max_input_chars("rewrite"),
            session_id=resolved_session_id,
            tool_profile_id=rewrite_profile_id,
            tool_registry_override=rewrite_registry,
        ).revised
        route._emit_text_deltas(q, "rewrite", revised)
    task_status = _graph_task_status(state)
    task_status["rewrite"] = route._is_effective(revised)
    next_state = {
        **state,
        "task_status": task_status,
        "revised_candidate": revised,
    }
    _save_tail_resume_boundary(
        next_state,
        current_stage="rewrite_done",
        extra_payload={
            "revised": revised,
        },
    )
    return next_state


def _post_process_tail_step(state: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    route = _pipeline_route_helpers()
    thread_id = str(state.get("thread_id") or "")
    resolved_session_id = str(state.get("resolved_session_id") or "").strip()
    upsert_pipeline_v2_checkpoint(
        thread_id=thread_id,
        session_id=resolved_session_id,
        mode=str(state.get("mode") or "sync"),
        status="running",
        current_stage="post_process",
        outline_preview=str(state.get("outline") or ""),
    )
    outline = OutlinePlan(
        outline=str(state.get("outline") or ""),
        assumptions=str(state.get("assumptions") or ""),
        open_questions=str(state.get("open_questions") or ""),
    )
    notes = _restore_tail_notes(state.get("research_notes"))
    response, enforced_text, report, citation_enforced = route._finalize_pipeline_response_from_tail(
        payload=route.PipelineRequest.model_validate(dict(state.get("request") or {})),
        services=services,
        outline=outline,
        notes=notes,
        notes_text=str(state.get("notes_text") or ""),
        draft=str(state.get("draft") or ""),
        review=str(state.get("review") or ""),
        revised_candidate=str(state.get("revised_candidate") or state.get("draft") or ""),
    )
    current_state = {
        **state,
        "task_status": {
            **_graph_task_status(state),
            "citations": True,
            "rewrite": _graph_task_status(state).get("rewrite", False)
            if bool(state.get("needs_rewrite"))
            else route._is_effective(str(state.get("revised_candidate") or state.get("draft") or "")),
        },
    }
    _save_tail_resume_boundary(
        current_state,
        current_stage="completed",
        status="completed",
        extra_payload={
            "review": str(state.get("review") or ""),
            "review_text": str(state.get("review_text") or state.get("review") or ""),
            "needs_rewrite": bool(state.get("needs_rewrite")),
            "reason": str(state.get("reason") or ""),
            "score": state.get("score"),
            "revised": enforced_text,
            "coverage": report.coverage,
            "coverage_detail": (route._coverage_detail(report) or route.CoverageDetail()).model_dump(),
            "citation_enforced": citation_enforced,
            "response": response.model_dump(),
        },
    )
    mark_pipeline_v2_checkpoint_completed(thread_id=thread_id)
    return {
        **current_state,
        "revised": enforced_text,
        "response": response.model_dump(),
        "coverage": report.coverage,
        "coverage_detail": (route._coverage_detail(report) or route.CoverageDetail()).model_dump(),
        "citation_enforced": citation_enforced,
    }


@task
def research_node(state: dict[str, Any]) -> dict[str, Any]:
    return _research_step(state)


@task
def draft_node(state: dict[str, Any]) -> dict[str, Any]:
    return _draft_step(state)


@task
def apply_draft_resume_task(draft_state: dict[str, Any], resume_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = resume_payload or {}
    draft_override = str(payload.get("draft_override") or "").strip()
    final_draft = draft_override or str(draft_state.get("draft") or "")
    task_status = _graph_task_status(draft_state)
    task_status["draft"] = _pipeline_route_helpers()._is_effective(final_draft)
    return {
        **draft_state,
        "task_status": task_status,
        "draft": final_draft,
    }


@task
def apply_review_confirmation_resume_task(
    review_state: dict[str, Any],
    resume_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **review_state,
        "_review_resume_payload": dict(resume_payload or {}),
    }


@task
def review_node(state: dict[str, Any]) -> dict[str, Any]:
    return _review_tail_step(state)


@task
def rewrite_node(state: dict[str, Any]) -> dict[str, Any]:
    return _rewrite_tail_step(state)


@task
def post_process_node(state: dict[str, Any]) -> dict[str, Any]:
    return _post_process_tail_step(state)


@entrypoint(checkpointer=_get_checkpointer())
def pipeline_v2_full_workflow(full_input: dict[str, Any]) -> dict[str, Any]:
    start_stage = str(full_input.get("start_stage") or "outline_accepted")
    state = dict(full_input)
    if start_stage == "outline_accepted":
        state = research_node(state).result()
        if state.get("response"):
            return state
        state = draft_node(state).result()
        _mark_draft_review_interrupt(state)
        state = apply_draft_resume_task(
            state,
            interrupt(
                {
                    "kind": "draft_review",
                    "interrupt_stage": "draft_review",
                    "thread_id": str(state.get("thread_id") or ""),
                    "draft": str(state.get("draft") or ""),
                    "outline": str(state.get("outline") or ""),
                    "assumptions": str(state.get("assumptions") or ""),
                    "open_questions": str(state.get("open_questions") or ""),
                }
            ),
        ).result()
    elif start_stage == "research_done":
        state = draft_node(state).result()
        _mark_draft_review_interrupt(state)
        state = apply_draft_resume_task(
            state,
            interrupt(
                {
                    "kind": "draft_review",
                    "interrupt_stage": "draft_review",
                    "thread_id": str(state.get("thread_id") or ""),
                    "draft": str(state.get("draft") or ""),
                    "outline": str(state.get("outline") or ""),
                    "assumptions": str(state.get("assumptions") or ""),
                    "open_questions": str(state.get("open_questions") or ""),
                }
            ),
        ).result()
    if state.get("response"):
        return state
    if start_stage == "rewrite_done":
        task_status = _graph_task_status(state)
        task_status["rewrite"] = _pipeline_route_helpers()._is_effective(
            str(state.get("revised_candidate") or state.get("revised") or state.get("draft") or "")
        )
        rewritten = {
            **state,
            "task_status": task_status,
            "revised_candidate": str(
                state.get("revised_candidate") or state.get("revised") or state.get("draft") or ""
            ),
        }
        return post_process_node(rewritten).result()
    if start_stage == "review_done":
        reviewed = dict(state)
    else:
        reviewed = review_node(state).result()
        _mark_review_confirmation_interrupt(reviewed)
        reviewed = apply_review_confirmation_resume_task(
            reviewed,
            interrupt(
                {
                    "kind": "review_confirmation",
                    "interrupt_stage": "review_confirmation",
                    "thread_id": str(reviewed.get("thread_id") or ""),
                    "review_text": str(reviewed.get("review_text") or reviewed.get("review") or ""),
                    "needs_rewrite": bool(reviewed.get("needs_rewrite")),
                    "reason": str(reviewed.get("reason") or ""),
                    "score": reviewed.get("score"),
                    "draft": str(reviewed.get("draft") or ""),
                    "outline": str(reviewed.get("outline") or ""),
                    "assumptions": str(reviewed.get("assumptions") or ""),
                    "open_questions": str(reviewed.get("open_questions") or ""),
                }
            ),
        ).result()
    if bool(reviewed.get("needs_rewrite")):
        rewritten = rewrite_node(reviewed).result()
    else:
        task_status = _graph_task_status(reviewed)
        task_status["rewrite"] = _pipeline_route_helpers()._is_effective(
            str(reviewed.get("draft") or "")
        )
        rewritten = {
            **reviewed,
            "task_status": task_status,
            "revised_candidate": str(reviewed.get("draft") or ""),
            "rewrite_skipped": True,
        }
    return post_process_node(rewritten).result()


def run_pipeline_v2_full_sync(full_input: dict[str, Any]) -> dict[str, Any]:
    if str(full_input.get("start_stage") or "outline_accepted") == "outline_accepted":
        _save_tail_resume_boundary(
            dict(full_input),
            current_stage="outline_accepted",
            extra_payload={},
        )
    return pipeline_v2_full_workflow.invoke(
        full_input,
        config=_thread_config(str(full_input.get("thread_id") or "")),
    )


def resume_pipeline_v2_full_workflow(*, thread_id: str, draft_override: str = "") -> dict[str, Any]:
    return pipeline_v2_full_workflow.invoke(
        Command(resume={"draft_override": draft_override}),
        config=_thread_config(thread_id),
    )


def run_pipeline_v2_full_stream(
    *,
    q: Queue[dict[str, Any] | None],
    full_input: dict[str, Any],
) -> dict[str, Any]:
    route = _pipeline_route_helpers()
    state = dict(full_input)
    start_stage = str(state.get("start_stage") or "outline_accepted")
    if start_stage == "outline_accepted":
        _save_tail_resume_boundary(
            state,
            current_stage="outline_accepted",
            extra_payload={},
        )

    if start_stage == "outline_accepted":
        request_model = route.PipelineRequest.model_validate(dict(state.get("request") or {}))
        outline = OutlinePlan(
            outline=str(state.get("outline") or ""),
            assumptions=str(state.get("assumptions") or ""),
            open_questions=str(state.get("open_questions") or ""),
        )
        task_status = _graph_task_status(state)
        research_state = route._run_pipeline_research_stream(
            q=q,
            payload=request_model,
            services=get_services(),
            outline=outline,
            sources=_sources_from_request_payload(dict(state.get("request") or {}), str(state.get("github_context") or "")),
            github_context=str(state.get("github_context") or ""),
            effective_constraints=str(state.get("effective_constraints") or request_model.constraints or ""),
            resolved_session_id=str(state.get("resolved_session_id") or ""),
            task_status=task_status,
        )
        state = {
            **state,
            "task_status": task_status,
            "research_notes": route._serialize_resume_notes(list(research_state.notes)),
            "notes_text": research_state.notes_text,
            "source_count": len(research_state.effective_sources),
        }
        if research_state.final_response is not None:
            _save_tail_resume_boundary(
                state,
                current_stage="completed",
                status="completed",
                extra_payload={
                    "response": research_state.final_response.model_dump(),
                    "coverage": 0.0,
                    "coverage_detail": route.CoverageDetail().model_dump(),
                },
            )
            mark_pipeline_v2_checkpoint_completed(thread_id=str(state.get("thread_id") or ""))
            return {
                **state,
                "response": research_state.final_response.model_dump(),
                "coverage": 0.0,
                "coverage_detail": route.CoverageDetail().model_dump(),
            }
        _save_tail_resume_boundary(
            state,
            current_stage="research_done",
            extra_payload={
                "research_notes": list(state.get("research_notes") or []),
                "notes_text": str(state.get("notes_text") or ""),
            },
        )
        start_stage = "research_done"

    if start_stage == "research_done":
        request_model = route.PipelineRequest.model_validate(dict(state.get("request") or {}))
        outline = OutlinePlan(
            outline=str(state.get("outline") or ""),
            assumptions=str(state.get("assumptions") or ""),
            open_questions=str(state.get("open_questions") or ""),
        )
        draft = route._run_pipeline_draft_stream(
            q=q,
            payload=request_model,
            services=get_services(),
            outline=outline,
            effective_constraints=str(state.get("effective_constraints") or request_model.constraints or ""),
            resolved_session_id=str(state.get("resolved_session_id") or ""),
            notes_text=str(state.get("notes_text") or ""),
            source_count=int(state.get("source_count") or 0),
        )
        task_status = _graph_task_status(state)
        task_status["draft"] = route._is_effective(draft)
        state = {
            **state,
            "task_status": task_status,
            "draft": draft,
        }
        _save_tail_resume_boundary(
            state,
            current_stage="draft_done",
            extra_payload={
                "research_notes": list(state.get("research_notes") or []),
                "notes_text": str(state.get("notes_text") or ""),
                "draft": draft,
            },
        )
        _mark_draft_review_interrupt(state)
        q.put(
            {
                "type": "interrupt",
                "kind": "draft_review",
                "payload": {
                    "thread_id": str(state.get("thread_id") or ""),
                    "interrupt_stage": "draft_review",
                    "draft": draft,
                    "outline": str(state.get("outline") or ""),
                    "assumptions": str(state.get("assumptions") or ""),
                    "open_questions": str(state.get("open_questions") or ""),
                },
            }
        )
        return {
            **state,
            "interrupted": True,
            "interrupt_stage": "draft_review",
        }

    if start_stage not in {"review_done", "rewrite_done"}:
        q.put({"type": "status", "step": "review"})
        state = _review_tail_stream_step(state=state, q=q)
        _mark_review_confirmation_interrupt(state)
        q.put(
            {
                "type": "interrupt",
                "kind": "review_confirmation",
                "payload": {
                    "thread_id": str(state.get("thread_id") or ""),
                    "interrupt_stage": "review_confirmation",
                    "review_text": str(state.get("review_text") or state.get("review") or ""),
                    "needs_rewrite": bool(state.get("needs_rewrite")),
                    "reason": str(state.get("reason") or ""),
                    "score": state.get("score"),
                    "draft": str(state.get("draft") or ""),
                    "outline": str(state.get("outline") or ""),
                    "assumptions": str(state.get("assumptions") or ""),
                    "open_questions": str(state.get("open_questions") or ""),
                },
            }
        )
        return {
            **state,
            "interrupted": True,
            "interrupt_stage": "review_confirmation",
        }
    else:
        q.put(
            {
                "type": "review_decision",
                "payload": {
                    "review_text": str(state.get("review_text") or state.get("review") or ""),
                    "needs_rewrite": bool(state.get("needs_rewrite")),
                    "reason": str(state.get("reason") or ""),
                    "score": state.get("score"),
                    "resumed": True,
                },
            }
        )
        q.put(
            {
                "type": "review",
                "payload": {"review": str(state.get("review_text") or state.get("review") or "")},
            }
        )

    if start_stage == "rewrite_done":
        task_status = _graph_task_status(state)
        task_status["rewrite"] = route._is_effective(
            str(state.get("revised_candidate") or state.get("revised") or state.get("draft") or "")
        )
        state = {
            **state,
            "task_status": task_status,
            "revised_candidate": str(
                state.get("revised_candidate") or state.get("revised") or state.get("draft") or ""
            ),
        }
    elif bool(state.get("needs_rewrite")):
        q.put({"type": "status", "step": "rewrite"})
        state = _rewrite_tail_stream_step(state=state, q=q)
    else:
        task_status = _graph_task_status(state)
        task_status["rewrite"] = route._is_effective(str(state.get("draft") or ""))
        state = {
            **state,
            "task_status": task_status,
            "revised_candidate": str(state.get("draft") or ""),
            "rewrite_skipped": True,
        }
        q.put({"type": "rewrite_decision", "payload": {"skipped": True}})

    q.put({"type": "status", "step": "citations"})
    state = _post_process_tail_step(state)
    q.put(
        {
            "type": "rewrite",
            "payload": {
                "revised": str(state.get("revised") or state.get("draft") or ""),
                "final": True,
                "coverage": state.get("coverage"),
                "coverage_detail": dict(state.get("coverage_detail") or {}),
            },
        }
    )
    q.put({"type": "result", "payload": dict(state.get("response") or {})})
    return state


def pipeline_v2_tail_workflow(tail_input: dict[str, Any]) -> dict[str, Any]:
    return pipeline_v2_full_workflow.invoke(tail_input)


def run_pipeline_v2_tail_sync(tail_input: dict[str, Any]) -> dict[str, Any]:
    return run_pipeline_v2_full_sync(tail_input)


def run_pipeline_v2_tail_stream(*, q: Queue[dict[str, Any] | None], tail_input: dict[str, Any]) -> dict[str, Any]:
    return run_pipeline_v2_full_stream(q=q, full_input=tail_input)


@task
def plan_task(workflow_input: dict[str, Any]) -> dict[str, Any]:
    services = get_services()
    request_data = dict(workflow_input.get("request") or {})
    resolved_session_id = str(workflow_input.get("resolved_session_id") or "").strip()
    thread_id = str(workflow_input.get("thread_id") or "").strip()
    prefetched_plan_result = workflow_input.get("prefetched_plan_result") or {}
    if isinstance(prefetched_plan_result, dict) and str(prefetched_plan_result.get("outline") or "").strip():
        logger.info(
            "pipeline v2 workflow: stage=plan_prefetched thread_id=%s session_id=%s",
            thread_id or "__unknown__",
            resolved_session_id or "__default__",
        )
        return {
            "request": request_data,
            "thread_id": thread_id,
            "resolved_session_id": resolved_session_id,
            "github_context": str(workflow_input.get("github_context") or ""),
            "effective_constraints": str(workflow_input.get("effective_constraints") or ""),
            "outline": str(prefetched_plan_result.get("outline") or ""),
            "assumptions": str(prefetched_plan_result.get("assumptions") or ""),
            "open_questions": str(prefetched_plan_result.get("open_questions") or ""),
        }
    tool_profile_id = str(workflow_input.get("plan_tool_profile_id") or "").strip() or None
    allowed_tool_names = list(workflow_input.get("plan_tool_allowed_tools") or [])
    tool_registry = build_stage_tool_registry(
        tool_catalog=services.agent_tools_catalog,
        allowed_tool_names=allowed_tool_names,
    )
    if tool_registry is None:
        tool_profile_id = None
    logger.info(
        "pipeline v2 workflow: stage=plan_started thread_id=%s session_id=%s",
        thread_id or "__unknown__",
        resolved_session_id or "__default__",
    )
    outline = services.planner.plan_outline(
        topic=str(request_data.get("topic") or ""),
        audience=str(request_data.get("audience") or ""),
        style=str(request_data.get("style") or ""),
        target_length=str(request_data.get("target_length") or ""),
        constraints=str(workflow_input.get("effective_constraints") or request_data.get("constraints") or ""),
        key_points=str(request_data.get("key_points") or ""),
        max_tokens=workflow_input.get("plan_max_tokens"),
        max_input_chars=workflow_input.get("plan_max_input_chars"),
        session_id=resolved_session_id,
        tool_profile_id=tool_profile_id,
        tool_registry_override=tool_registry,
    )
    return {
        "request": request_data,
        "thread_id": thread_id,
        "resolved_session_id": resolved_session_id,
        "github_context": str(workflow_input.get("github_context") or ""),
        "effective_constraints": str(workflow_input.get("effective_constraints") or ""),
        "outline": outline.outline,
        "assumptions": outline.assumptions,
        "open_questions": outline.open_questions,
    }


@task
def apply_resume_task(plan_result: dict[str, Any], resume_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = resume_payload or {}
    outline_override = str(payload.get("outline_override") or "").strip()
    final_outline = outline_override or str(plan_result.get("outline") or "")
    return {
        **plan_result,
        "outline": final_outline,
    }


@entrypoint(checkpointer=_get_checkpointer())
def pipeline_v2_workflow(workflow_input: dict[str, Any]) -> dict[str, Any]:
    plan_result = plan_task(workflow_input).result()
    resume_payload = interrupt(
        {
            "kind": "outline_review",
            "thread_id": plan_result["thread_id"],
            "outline": plan_result["outline"],
            "assumptions": plan_result["assumptions"],
            "open_questions": plan_result["open_questions"],
        }
    )
    return apply_resume_task(plan_result, resume_payload).result()


def start_pipeline_v2_workflow(workflow_input: dict[str, Any], *, thread_id: str) -> dict[str, Any]:
    result = pipeline_v2_workflow.invoke(workflow_input, config=_thread_config(thread_id))
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        logger.info(
            "pipeline v2 workflow: stage=interrupted thread_id=%s session_id=%s outline_len=%s checkpointer=%s db_path=%s",
            thread_id,
            str(workflow_input.get("resolved_session_id") or "").strip() or "__default__",
            len(getattr(interrupts[0], "value", {}).get("outline", "") or ""),
            type(_get_checkpointer()).__name__,
            _get_checkpointer().db_path,
        )
    return result


def resume_pipeline_v2_workflow(*, thread_id: str, outline_override: str = "") -> dict[str, Any]:
    if not has_pipeline_v2_interrupt_checkpoint(thread_id):
        raise KeyError(f"thread_id '{thread_id}' has no checkpoint to resume")
    logger.info(
        "pipeline v2 workflow: stage=resume_checkpoint_found thread_id=%s checkpointer=%s db_path=%s",
        thread_id,
        type(_get_checkpointer()).__name__,
        _get_checkpointer().db_path,
    )
    result = pipeline_v2_workflow.invoke(
        Command(resume={"outline_override": outline_override}),
        config=_thread_config(thread_id),
    )
    logger.info(
        "pipeline v2 workflow: stage=resumed thread_id=%s override=%s checkpointer=%s db_path=%s",
        thread_id,
        bool((outline_override or "").strip()),
        type(_get_checkpointer()).__name__,
        _get_checkpointer().db_path,
    )
    return result
