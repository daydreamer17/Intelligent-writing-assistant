from __future__ import annotations

import json

from app.services.github_context import maybe_fetch_github_context
from app.services.tool_policy import decide_tools


class DummyTool:
    def __init__(self, tools: list[dict[str, str]], result: str) -> None:
        self._available_tools = tools
        self._result = result
        self.calls: list[dict] = []

    def run(self, payload: dict) -> str:
        self.calls.append(payload)
        return self._result


def test_github_context_injected_when_keyword_present():
    data = {
        "items": [
            {
                "full_name": "openai/openai-python",
                "html_url": "https://github.com/openai/openai-python",
                "description": "OpenAI Python SDK",
            },
            {
                "name": "example/repo",
                "url": "https://github.com/example/repo",
                "description": "Example repo",
            },
        ]
    }
    tool = DummyTool(
        tools=[{"name": "github_search_repositories", "description": "search repos"}],
        result=json.dumps(data),
    )

    text = maybe_fetch_github_context(
        tool,
        query="github openai sdk",
        context="",
        max_repos=1,
        max_chars=1000,
    )

    assert tool.calls, "expected MCP tool to be called"
    assert tool.calls[0]["tool_name"] == "github_search_repositories"
    assert "openai/openai-python" in text
    assert "Example repo" not in text


def test_github_context_skipped_without_keyword():
    tool = DummyTool(
        tools=[{"name": "github_search_repositories", "description": "search repos"}],
        result="{}",
    )
    text = maybe_fetch_github_context(tool, query="write a story", context="")
    assert text == ""
    assert not tool.calls


def test_stage_tool_policy_for_plan_and_draft():
    plan = decide_tools(stage="plan", topic="github repository overview")
    draft = decide_tools(
        stage="draft",
        topic="github repository overview",
        outline="read README.md from owner/repo",
    )
    assert plan.enabled is True
    assert "github_search_repositories" in plan.allowed_tools
    assert "github_get_file_contents" not in plan.allowed_tools
    assert draft.enabled is True
    assert "github_get_file_contents" in draft.allowed_tools


def test_stage_tool_policy_blocks_rewrite():
    rewrite = decide_tools(
        stage="rewrite",
        draft="rewrite this draft",
        guidance="improve style",
    )
    assert rewrite.enabled is False
    assert rewrite.allowed_tools == []


def test_stage_tool_policy_review_only_search_tools(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    monkeypatch.setenv("LLM_TOOL_POLICY_MODE", "rules")
    monkeypatch.setenv("LLM_TOOLS_REVIEW_STAGE", "github_search_repositories,github_search_code")
    review = decide_tools(
        stage="review",
        topic="review github code style",
        draft="请根据仓库代码审校本文",
        guidance="逻辑与结构",
    )
    assert review.enabled is True
    assert "github_search_repositories" in review.allowed_tools
    assert "github_search_code" in review.allowed_tools
    assert "github_get_file_contents" not in review.allowed_tools
