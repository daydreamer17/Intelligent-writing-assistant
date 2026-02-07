from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from hello_agents.tools import MCPTool

logger = logging.getLogger("app.github")

_DEFAULT_KEYWORDS = (
    "github",
    "repo",
    "repository",
    "git",
    "open-source",
    "opensource",
    "issue",
    "pull request",
    "pr",
    "star",
    "fork",
    "开源",
    "仓库",
    "代码库",
    "项目",
    "issue",
    "pull request",
    "pr",
)


def maybe_fetch_github_context(
    tool: MCPTool | None,
    *,
    query: str,
    context: str = "",
    max_repos: int = 5,
    max_chars: int = 4000,
    keywords: Iterable[str] = _DEFAULT_KEYWORDS,
) -> str:
    if tool is None:
        return ""

    combined = f"{query}\n{context}".strip()
    if not _needs_github(combined, keywords):
        return ""

    tool_name = _pick_search_tool(tool)
    if not tool_name:
        return ""

    query_text = _normalize_query(query or combined)
    try:
        result = tool.run(
            {
                "action": "call_tool",
                "tool_name": tool_name,
                "arguments": {
                    "query": query_text,
                    "per_page": max_repos,
                },
            }
        )
    except Exception as exc:
        logger.warning("GitHub MCP call failed: %s", exc)
        return ""

    formatted = _format_result(result, max_repos=max_repos, max_chars=max_chars)
    if formatted:
        logger.info("Injected GitHub MCP context (%s chars).", len(formatted))
    return formatted


def _needs_github(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for kw in keywords:
        if not kw:
            continue
        if kw in lowered or kw in text:
            return True
    return False


def _normalize_query(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned


def _pick_search_tool(tool: MCPTool) -> str:
    candidates = (
        "github_search_repositories",
        "search_repositories",
        "search_repository",
        "search_repos",
        "search",
    )
    available = getattr(tool, "_available_tools", []) or []
    names = {t.get("name", "") for t in available if isinstance(t, dict)}
    for name in candidates:
        if name in names:
            return name
    for name in names:
        if "search_repositories" in name:
            return name
    return "github_search_repositories"


def _format_result(raw: str, *, max_repos: int, max_chars: int) -> str:
    if not raw:
        return ""
    text = raw
    try:
        data = json.loads(raw)
    except Exception:
        return raw[:max_chars]

    items = None
    if isinstance(data, dict):
        items = data.get("items")
        if items is None:
            items = data.get("data")
    if not isinstance(items, list):
        return raw[:max_chars]

    lines = []
    for item in items[:max_repos]:
        if not isinstance(item, dict):
            continue
        name = item.get("full_name") or item.get("name") or ""
        url = item.get("html_url") or item.get("url") or ""
        desc = item.get("description") or ""
        line = f"- {name}"
        if url:
            line += f" ({url})"
        if desc:
            line += f": {desc}"
        lines.append(line.strip())

    text = "\n".join(lines).strip()
    return text[:max_chars] if text else raw[:max_chars]
