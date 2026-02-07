from __future__ import annotations

import json

from app.services.github_context import maybe_fetch_github_context


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
