from __future__ import annotations

from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional writing assistant. "
    "Create clear, structured, and well-argued drafts. "
    "Follow the user's constraints, keep a consistent tone, "
    "and avoid making up facts. If sources are missing, ask for them. "
    "When external facts are needed and tools are available, call tools first "
    "and incorporate their results."
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
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
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
        return self.run(
            prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )

    def draft_stream(
        self,
        *,
        topic: str,
        outline: str,
        constraints: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ):
        # 截断过长的 outline 以避免超过模型限制
        max_outline_chars = 10000  # 约 4000 tokens

        if len(outline) > max_outline_chars:
            outline = outline[:max_outline_chars] + "\n...(大纲过长已截断)"

        sections = self._split_outline(outline)
        if len(sections) <= 1:
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

            prompt = "Write a complete draft based on the information below.\n\n" + "\n\n".join(parts)
            yield from self.stream(
                prompt,
                session_id=session_id,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            )
            return

        target_len = self._parse_target_length(target_length) or 0
        per_section_len = max(300, int(target_len / max(len(sections), 1))) if target_len else 0
        context_tail = ""
        first_section = True

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
                "Instructions:\n" + "\n".join(instructions),
            ]
            if constraints:
                parts.append(f"Constraints:\n{constraints}")
            if style:
                parts.append(f"Style:\n{style}")
            if context_tail:
                parts.append(f"Previous context (do not repeat):\n{context_tail}")

            prompt = "Write the next section based on the information below.\n\n" + "\n\n".join(parts)

            section_max_tokens = int(per_section_len * 1.2) if per_section_len else None
            if max_tokens is not None and section_max_tokens is not None:
                section_max_tokens = min(max_tokens, section_max_tokens)
            elif max_tokens is not None:
                section_max_tokens = max_tokens
            section_chunks = []
            for chunk in self.stream(
                prompt,
                max_tokens=section_max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            ):
                section_chunks.append(chunk)
                yield chunk

            section_text = "".join(section_chunks).strip()
            if section_text:
                context_tail = (context_tail + "\n\n" + section_text).strip()
                if len(context_tail) > 1200:
                    context_tail = context_tail[-1200:]

            if first_section:
                first_section = False
            else:
                yield "\n\n"

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
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> str:
        sections = self._split_outline(outline)
        if len(sections) <= 1:
            return self.draft(
                topic=topic,
                outline=outline,
                constraints=constraints,
                style=style,
                target_length=target_length,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
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

            section_max_tokens = int(per_section_len * 1.2) if per_section_len else None
            if max_tokens is not None and section_max_tokens is not None:
                section_max_tokens = min(max_tokens, section_max_tokens)
            elif max_tokens is not None:
                section_max_tokens = max_tokens
            section_text = self.run(
                prompt,
                max_tokens=section_max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            )
            outputs.append(section_text.strip())
            context_tail = "\n\n".join(outputs)
            if len(context_tail) > 1200:
                context_tail = context_tail[-1200:]

        return "\n\n".join(outputs)
