from __future__ import annotations

from app.services.tool_policy import build_stage_tool_registry, decide_tools


class DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_stage_defaults_and_intent_routing(monkeypatch):
    monkeypatch.delenv("LLM_TOOLS_PLAN_STAGE", raising=False)
    monkeypatch.delenv("LLM_TOOLS_DRAFT_STAGE", raising=False)
    monkeypatch.delenv("LLM_TOOLS_REVIEW_STAGE", raising=False)
    monkeypatch.delenv("LLM_TOOLS_REWRITE_STAGE", raising=False)
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    monkeypatch.setenv("LLM_TOOL_POLICY_MODE", "rules")

    plan = decide_tools(stage="plan", topic="github repo status")
    assert plan.enabled is True
    assert "github_search_repositories" in plan.allowed_tools
    assert "github_search_code" in plan.allowed_tools

    draft = decide_tools(
        stage="draft",
        topic="github repository",
        outline="read README.md from owner/repo path docs",
    )
    assert draft.enabled is True
    assert "github_get_file_contents" in draft.allowed_tools

    review = decide_tools(stage="review", draft="github code review")
    assert review.enabled is True
    assert "github_get_file_contents" not in review.allowed_tools
    assert any("search" in name for name in review.allowed_tools)

    rewrite = decide_tools(stage="rewrite", draft="github code review")
    assert rewrite.enabled is False
    assert rewrite.allowed_tools == []


def test_disable_when_no_intent(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    monkeypatch.setenv("LLM_TOOL_POLICY_MODE", "rules")
    result = decide_tools(stage="plan", topic="写一篇校园爱情故事")
    assert result.enabled is False
    assert result.allowed_tools == []


def test_disable_when_rag_is_strong(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    monkeypatch.setenv("LLM_TOOL_POLICY_MODE", "rules")
    monkeypatch.setenv("LLM_TOOL_POLICY_DISABLE_WHEN_RAG_STRONG", "true")

    result = decide_tools(
        stage="draft",
        topic="普通写作任务",
        rag_enforced=True,
        source_count=6,
    )
    assert result.enabled is False
    assert "rag_strong_disable" in result.reason


def test_profile_id_is_stable(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    monkeypatch.setenv("LLM_TOOL_POLICY_MODE", "rules")

    a = decide_tools(stage="plan", topic="github repo")
    b = decide_tools(stage="plan", topic="github repo")
    c = decide_tools(stage="plan", topic="普通写作")
    assert a.profile_id == b.profile_id
    assert a.profile_id != c.profile_id


def test_build_stage_registry(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_BASED_TOOLS_ENABLED", "true")
    tools = [DummyTool("github_search_repositories"), DummyTool("github_get_file_contents")]
    reg = build_stage_tool_registry(
        tool_catalog=tools,
        allowed_tool_names=["github_search_repositories"],
    )
    assert reg is not None
    reg2 = build_stage_tool_registry(tool_catalog=tools, allowed_tool_names=["missing_tool"])
    assert reg2 is None
