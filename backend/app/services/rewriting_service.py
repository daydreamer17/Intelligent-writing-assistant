from __future__ import annotations

from dataclasses import dataclass
import re

from ..agents.editor_agent import EditorAgent


@dataclass(frozen=True)
class RewriteResult:
    revised: str


class RewritingService:
    def __init__(self, agent: EditorAgent) -> None:
        self.agent = agent

    def rewrite(
        self,
        *,
        draft: str,
        guidance: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
    ) -> RewriteResult:
        if _should_chunk(draft, target_length):
            revised = self._rewrite_long(
                draft=draft,
                guidance=guidance,
                style=style,
                target_length=target_length,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
            )
        else:
            revised = self.agent.rewrite(
                draft=draft,
                guidance=guidance,
                style=style,
                target_length=target_length,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
            )
        return RewriteResult(revised=revised)

    def _rewrite_long(
        self,
        *,
        draft: str,
        guidance: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
    ) -> str:
        chunks = _split_draft(draft, max_chars=1200)
        outputs: list[str] = []
        context_tail = ""

        for index, chunk in enumerate(chunks, start=1):
            section_guidance = guidance
            section_guidance = (section_guidance + "\n\n" if section_guidance else "")
            section_guidance += (
                f"Rewrite ONLY this section ({index}/{len(chunks)}). "
                "Keep continuity and avoid repetition."
            )
            if context_tail:
                section_guidance += f"\n\nPrevious context (do not repeat):\n{context_tail}"

            section_max_tokens = int(len(chunk) * 1.2)
            if max_tokens is not None:
                section_max_tokens = min(section_max_tokens, max_tokens)
            section_text = self.agent.rewrite(
                draft=chunk,
                guidance=section_guidance,
                style=style,
                target_length=target_length,
                max_tokens=section_max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
            )
            outputs.append(section_text.strip())
            context_tail = "\n\n".join(outputs)
            if len(context_tail) > 1200:
                context_tail = context_tail[-1200:]

        return "\n\n".join(outputs)


def _parse_target_length(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _should_chunk(draft: str, target_length: str) -> bool:
    target_len = _parse_target_length(target_length)
    if target_len is not None and target_len >= 1800:
        return True
    return len(draft) >= 1800


def _split_draft(draft: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in draft.split("\n") if p.strip()]
    if not paragraphs:
        return [draft]
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) + 1 > max_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer = [para]
            size = len(para)
        else:
            buffer.append(para)
            size += len(para) + 1
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks
