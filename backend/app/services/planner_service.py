from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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
            "Use bullet points for the outline."
        )
        return instructions + "\n\n" + "\n\n".join(parts)

    @staticmethod
    def _split_sections(text: str) -> tuple[str, str, str]:
        sections = {"outline": "", "assumptions": "", "open questions": ""}
        current_key: Optional[str] = None

        for line in text.splitlines():
            stripped = line.strip()
            lower = stripped.lower().rstrip(":")
            if lower in sections:
                current_key = lower
                continue
            if current_key:
                sections[current_key] += line + "\n"

        outline = sections["outline"].strip() or text.strip()
        assumptions = sections["assumptions"].strip()
        questions = sections["open questions"].strip()
        return outline, assumptions, questions
