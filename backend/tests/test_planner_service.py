from __future__ import annotations

from app.services.planner_service import PlanningService


def test_split_sections_handles_markdown_numbered_headings():
    text = """根据目前的工具执行结果，GitHub 上没有返回任何相关仓库信息。

### 1) Outline
- 引言
- 方法

### 2) Assumptions
- 假设一

### 3) Open Questions
- 问题一
"""
    outline, assumptions, questions = PlanningService._split_sections(text)

    assert "GitHub 上没有返回" not in outline
    assert outline == "- 引言\n- 方法"
    assert assumptions == "- 假设一"
    assert questions == "- 问题一"


def test_build_prompt_forbids_tool_diagnostics():
    service = PlanningService(agent=object())

    prompt = service._build_prompt(
        topic="大模型微调项目总结",
        audience="初学者",
        style="说明文",
        target_length="1500",
        constraints="不要跑题",
        key_points="案例与方法",
    )

    assert "Do not mention tool execution" in prompt
    assert "GitHub/MCP search results" in prompt
    assert "keyword suggestions" in prompt
