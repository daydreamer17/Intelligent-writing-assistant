from __future__ import annotations

from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional writing assistant. "
    "Create clear, structured, and well-argued drafts. "
    "Follow the user's constraints, keep a consistent tone, "
    "and avoid making up facts. If sources are missing, ask for them."
)


class WritingAgent(BaseWritingAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        name: str = "Writing Agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        config = AgentRuntimeConfig(
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )
        super().__init__(config, tool_registry=tool_registry)

    def draft(
        self,
        *,
        topic: str,
        outline: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
    ) -> str:
        parts = [
            f"Topic:\n{topic}",
            f"Outline:\n{outline}",
        ]
        if constraints:
            parts.append(f"Constraints:\n{constraints}")
        if style:
            parts.append(f"Style:\n{style}")
        if target_length:
            parts.append(f"Target length:\n{target_length}")

        prompt = (
            "Write a complete draft based on the information below.\n\n"
            + "\n\n".join(parts)
        )
        return self.run(prompt)

    @staticmethod
    def _split_outline(outline: str) -> list[str]:
        lines = [line.strip() for line in outline.splitlines() if line.strip()]
        sections: list[str] = []
        for line in lines:
            cleaned = line.lstrip("-*• \t")
            cleaned = cleaned.strip()
            cleaned = cleaned.lstrip("0123456789.、) ")
            cleaned = cleaned.strip()
            if cleaned:
                sections.append(cleaned)
        return sections

    @staticmethod
    def _parse_target_length(target_length: str) -> Optional[int]:
        digits = "".join(ch for ch in target_length if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    def draft_long(
        self,
        *,
        topic: str,
        outline: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
    ) -> str:
        sections = self._split_outline(outline)
        if len(sections) <= 1:
            return self.draft(
                topic=topic,
                outline=outline,
                constraints=constraints,
                style=style,
                target_length=target_length,
            )

        target_len = self._parse_target_length(target_length) or 0
        per_section_len = max(300, int(target_len / max(len(sections), 1))) if target_len else 0
        context_tail = ""
        outputs: list[str] = []

        for index, section in enumerate(sections, start=1):
            instructions = [
                f"Section {index}/{len(sections)}: {section}",
                "Write only this section. Do not include other sections.",
                "Keep continuity with previous content and avoid repetition.",
            ]
            if per_section_len:
                instructions.append(f"Target length for this section: ~{per_section_len} Chinese characters.")

            parts = [
                f"Topic:\n{topic}",
                f"Outline:\n{outline}",
                f"Instructions:\n" + "\n".join(instructions),
            ]
            if constraints:
                parts.append(f"Constraints:\n{constraints}")
            if style:
                parts.append(f"Style:\n{style}")
            if context_tail:
                parts.append(f"Previous context (do not repeat):\n{context_tail}")

            prompt = "Write the next section based on the information below.\n\n" + "\n\n".join(parts)

            max_tokens = int(per_section_len * 1.2) if per_section_len else None
            section_text = self.run(prompt, max_tokens=max_tokens)
            outputs.append(section_text.strip())
            context_tail = "\n\n".join(outputs)
            if len(context_tail) > 1200:
                context_tail = context_tail[-1200:]

        return "\n\n".join(outputs)
