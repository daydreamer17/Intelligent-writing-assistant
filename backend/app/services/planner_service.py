from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Optional

from hello_agents.tools import ToolRegistry

from ..agents.writing_agent import WritingAgent


@dataclass(frozen=True)
class OutlinePlan:
    outline: str
    assumptions: str = ""
    open_questions: str = ""


class PlanningService:
    def __init__(self, agent: WritingAgent) -> None:
        self.agent = agent

    def plan_outline(
        self,
        *,
        topic: str,
        audience: str = "",
        style: str = "",
        target_length: str = "",
        constraints: str = "",
        key_points: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> OutlinePlan:
        prompt = self._build_prompt(
            topic=topic,
            audience=audience,
            style=style,
            target_length=target_length,
            constraints=constraints,
            key_points=key_points,
        )
        response = self.agent.run(
            prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )
        outline, assumptions, questions = self._split_sections(response)
        return OutlinePlan(outline=outline, assumptions=assumptions, open_questions=questions)

    def plan_outline_stream(
        self,
        *,
        topic: str,
        audience: str = "",
        style: str = "",
        target_length: str = "",
        constraints: str = "",
        key_points: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        on_chunk: Callable[[str], None] | None = None,
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> OutlinePlan:
        prompt = self._build_prompt(
            topic=topic,
            audience=audience,
            style=style,
            target_length=target_length,
            constraints=constraints,
            key_points=key_points,
        )
        chunks: list[str] = []
        for chunk in self.agent.stream(
            prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        ):
            if not chunk:
                continue
            chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
        text = "".join(chunks).strip()
        if not text:
            text = self.agent.run(
                prompt,
                session_id=session_id,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            )
            if on_chunk is not None and text:
                on_chunk(text)
        outline, assumptions, questions = self._split_sections(text)
        return OutlinePlan(outline=outline, assumptions=assumptions, open_questions=questions)

    def _build_prompt(
        self,
        *,
        topic: str,
        audience: str,
        style: str,
        target_length: str,
        constraints: str,
        key_points: str,
    ) -> str:
        parts = [
            f"Topic:\n{topic}",
        ]
        if audience:
            parts.append(f"Audience:\n{audience}")
        if style:
            parts.append(f"Style:\n{style}")
        if target_length:
            parts.append(f"Target length:\n{target_length}")
        if constraints:
            parts.append(f"Constraints:\n{constraints}")
        if key_points:
            parts.append(f"Key points:\n{key_points}")

        instructions = (
            "Create a clear outline for the requested document. "
            "Return three sections with labels:\n"
            "1) Outline\n2) Assumptions\n3) Open Questions\n"
            "Use bullet points for the outline.\n"
            "Do not mention tool execution, GitHub/MCP search results, retrieval diagnostics, "
            "missing search hits, API limitations, or keyword suggestions in the final answer. "
            "If concrete external examples are unavailable, still provide a useful generic outline "
            "and put only neutral content gaps in Open Questions."
        )
        return instructions + "\n\n" + "\n\n".join(parts)

    @staticmethod
    def _split_sections(text: str) -> tuple[str, str, str]:
        sections = {"outline": "", "assumptions": "", "open questions": ""}
        current_key: Optional[str] = None

        for line in text.splitlines():
            stripped = line.strip()
            normalized = PlanningService._normalize_section_heading(stripped)
            if normalized in sections:
                current_key = normalized
                continue
            if current_key:
                sections[current_key] += line + "\n"

        outline = sections["outline"].strip() or text.strip()
        assumptions = sections["assumptions"].strip()
        questions = sections["open questions"].strip()
        return outline, assumptions, questions

    @staticmethod
    def _normalize_section_heading(text: str) -> str:
        if not text:
            return ""
        normalized = text.strip()
        normalized = re.sub(r"^[#>\-\*\s]+", "", normalized)
        normalized = re.sub(r"^\d+\s*[\)\.\:\-、]\s*", "", normalized)
        normalized = normalized.strip().rstrip(":：").strip()
        normalized = re.sub(r"^[#>\-\*\s]+", "", normalized).strip()
        normalized = normalized.lower()
        aliases = {
            "outline": "outline",
            "**outline**": "outline",
            "大纲": "outline",
            "assumptions": "assumptions",
            "**assumptions**": "assumptions",
            "假设": "assumptions",
            "open questions": "open questions",
            "open question": "open questions",
            "**open questions**": "open questions",
            "待确认问题": "open questions",
            "开放问题": "open questions",
        }
        return aliases.get(normalized, "")
