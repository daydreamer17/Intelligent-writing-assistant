from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from hello_agents.tools import ToolRegistry

Stage = Literal["plan", "draft", "review", "rewrite"]

_DEFAULT_STAGE_TOOLS = {
    "plan": "github_search_repositories,github_search_code",
    "draft": "github_search_repositories,github_search_code,github_get_file_contents",
    "review": "github_search_repositories,github_search_code",
    "rewrite": "",
}

_DEFAULT_SEARCH_KEYWORDS = "github,repo,repository,issue,pr,commit,code,readme"
_DEFAULT_READ_KEYWORDS = "owner/repo,path,readme,.md,.py,.txt,file,文件,仓库"


@dataclass(frozen=True)
class ToolDecision:
    enabled: bool
    allowed_tools: list[str]
    reason: str
    profile_id: str


def decide_tools(
    *,
    stage: Stage,
    topic: str = "",
    outline: str = "",
    draft: str = "",
    guidance: str = "",
    constraints: str = "",
    research_notes: str = "",
    rag_enforced: bool = False,
    source_count: int = 0,
) -> ToolDecision:
    stage = (stage or "").strip().lower()  # type: ignore[assignment]
    if stage not in _DEFAULT_STAGE_TOOLS:
        stage = "draft"  # type: ignore[assignment]

    stage_based = _env_flag("LLM_STAGE_BASED_TOOLS_ENABLED", True)
    mode = (os.getenv("LLM_TOOL_POLICY_MODE", "rules") or "rules").strip().lower()

    stage_allow = _stage_allowlist(stage) if stage_based else _all_stage_allowlist()
    # rewrite 阶段保守禁用工具，避免循环调用工具导致改写跑偏。
    if stage == "rewrite":
        stage_allow = []

    if mode != "rules":
        allowed = sorted(set(stage_allow))
        reason = "stage_allow_only"
        return _build_decision(stage=stage, allowed=allowed, reason=reason)

    search_keywords = _env_list("LLM_TOOL_POLICY_SEARCH_KEYWORDS", _DEFAULT_SEARCH_KEYWORDS)
    read_keywords = _env_list("LLM_TOOL_POLICY_READ_KEYWORDS", _DEFAULT_READ_KEYWORDS)
    disable_when_rag_strong = _env_flag("LLM_TOOL_POLICY_DISABLE_WHEN_RAG_STRONG", True)

    combined = "\n".join(
        item
        for item in (
            topic,
            outline,
            draft,
            guidance,
            constraints,
            research_notes,
        )
        if item
    )
    has_search_intent = _contains_any(combined, search_keywords)
    has_read_intent = _contains_any(combined, read_keywords) or _looks_like_repo_path_hint(combined)

    reasons: list[str] = [f"stage={stage}"]
    allowed = list(stage_allow)

    if not has_search_intent:
        allowed = [name for name in allowed if "search" not in name]
        reasons.append("no_search_intent")
    else:
        reasons.append("search_intent")

    if not has_read_intent:
        allowed = [name for name in allowed if "get_file_contents" not in name and "read" not in name]
        reasons.append("no_read_intent")
    else:
        reasons.append("read_intent")

    # Prevent search/search_code ping-pong when only high-level search is needed.
    if has_search_intent and not has_read_intent:
        if "github_search_repositories" in allowed and "github_search_code" in allowed:
            allowed = [name for name in allowed if name != "github_search_code"]
            reasons.append("prefer_repo_search")

    if disable_when_rag_strong and rag_enforced and source_count > 0 and not has_search_intent and not has_read_intent:
        allowed = []
        reasons.append("rag_strong_disable")

    allowed = sorted(set(allowed))
    return _build_decision(stage=stage, allowed=allowed, reason="|".join(reasons))


def build_stage_tool_registry(
    *,
    tool_catalog: Iterable[Any],
    allowed_tool_names: Iterable[str],
) -> ToolRegistry | None:
    allowed = {name for name in allowed_tool_names if name}
    if not allowed:
        return None

    registry = ToolRegistry()
    added = 0
    for tool in tool_catalog:
        name = _tool_name(tool)
        if not name or name not in allowed:
            continue
        registry.register_tool(tool)
        added += 1
    if added == 0:
        return None
    return registry


def _build_decision(*, stage: str, allowed: list[str], reason: str) -> ToolDecision:
    enabled = len(allowed) > 0
    digest = hashlib.sha1(",".join(sorted(allowed)).encode("utf-8")).hexdigest()[:10] if enabled else "none"
    profile_id = f"{stage}:{digest}"
    return ToolDecision(
        enabled=enabled,
        allowed_tools=allowed,
        reason=reason,
        profile_id=profile_id,
    )


def _stage_allowlist(stage: str) -> list[str]:
    default = _DEFAULT_STAGE_TOOLS.get(stage, "")
    return _env_list(f"LLM_TOOLS_{stage.upper()}_STAGE", default)


def _all_stage_allowlist() -> list[str]:
    names: set[str] = set()
    for stage in ("plan", "draft", "review", "rewrite"):
        names.update(_stage_allowlist(stage))
    return sorted(name for name in names if name)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for keyword in keywords:
        kw = (keyword or "").strip().lower()
        if not kw:
            continue
        if kw in lowered:
            return True
    return False


def _looks_like_repo_path_hint(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        return True
    if re.search(r"\b(readme|path|file)\b", text, flags=re.IGNORECASE):
        return True
    return False


def _env_list(name: str, default: str) -> list[str]:
    raw = (os.getenv(name, default) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        name = tool.get("name")
        return str(name) if name else ""
    for attr in ("name", "tool_name", "id"):
        value = getattr(tool, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value:
            return str(value)
    return ""
