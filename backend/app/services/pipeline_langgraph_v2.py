from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

from ..api.deps import get_services
from .tool_policy import build_stage_tool_registry

logger = logging.getLogger("app.pipeline.v2")


class SQLitePersistentSaver(InMemorySaver):
    """Minimal SQLite-backed saver for v2 interrupt/resume persistence.

    This keeps the existing InMemorySaver semantics and persists its internal
    state to a local SQLite file after each checkpoint mutation. The goal is a
    reliable local checkpoint backend with the smallest possible change surface.
    """

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
    # Minimal engineering hook: v2 persists checkpoints in a local SQLite file.
    # A later round can replace this factory with a different persistent saver
    # without changing the workflow shape or the route contract.
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
    if _get_checkpointer().get_tuple(_thread_config(thread_id)) is None:
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
