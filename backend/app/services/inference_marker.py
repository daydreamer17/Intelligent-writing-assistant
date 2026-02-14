from __future__ import annotations

import os
import re

_CITATION_PATTERN = re.compile(r"\[\d+\]")


def mark_inference_paragraphs(
    text: str,
    *,
    tag: str | None = None,
    min_paragraph_chars: int | None = None,
) -> str:
    value = (text or "").strip()
    if not value:
        return text

    resolved_tag = (tag or os.getenv("RAG_HYBRID_INFERENCE_TAG", "[推断]") or "[推断]").strip()
    if not resolved_tag:
        resolved_tag = "[推断]"

    if min_paragraph_chars is None:
        try:
            min_paragraph_chars = int(os.getenv("RAG_HYBRID_MIN_PARAGRAPH_CHARS", "12"))
        except ValueError:
            min_paragraph_chars = 12
    min_paragraph_chars = max(1, min_paragraph_chars)

    paragraphs = re.split(r"\n\s*\n", value)
    output: list[str] = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if _should_skip_mark(stripped, min_paragraph_chars):
            output.append(stripped)
            continue
        if _CITATION_PATTERN.search(stripped) or resolved_tag in stripped:
            output.append(stripped)
            continue
        output.append(f"{stripped} {resolved_tag}")
    return "\n\n".join(output)


def _should_skip_mark(paragraph: str, min_paragraph_chars: int) -> bool:
    if len(paragraph) < min_paragraph_chars:
        return True
    # Avoid labeling short headings/list titles as inference.
    if paragraph.startswith(("#", "- ", "* ")):
        return True
    if re.fullmatch(r"(第[一二三四五六七八九十0-9]+[章节部分]|[A-Za-z0-9\u4e00-\u9fff\s:：\-—]{1,40})", paragraph):
        return True
    return False
