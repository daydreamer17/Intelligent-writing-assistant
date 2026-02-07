from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from hello_agents import HelloAgentsLLM

logger = logging.getLogger("app.rag")


@dataclass(frozen=True)
class EvidenceConfig:
    max_items: int
    max_chars: int
    max_tokens: int

    @staticmethod
    def from_env() -> "EvidenceConfig":
        def _parse_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return EvidenceConfig(
            max_items=max(5, _parse_int("RAG_EVIDENCE_MAX_ITEMS", 12)),
            max_chars=max(500, _parse_int("RAG_EVIDENCE_MAX_CHARS", 4000)),
            max_tokens=max(128, _parse_int("RAG_EVIDENCE_MAX_TOKENS", 400)),
        )


class EvidenceExtractor:
    def __init__(self, llm: HelloAgentsLLM, config: EvidenceConfig) -> None:
        self._llm = llm
        self._config = config

    def extract(self, notes_text: str) -> str:
        if not notes_text.strip():
            return ""
        notes = _truncate(notes_text, self._config.max_chars)
        prompt = (
            "Extract up to {n} factual evidence statements from the notes below.\n"
            "Rules:\n"
            "- Only use facts that appear in the notes.\n"
            "- Keep each statement short and specific.\n"
            "- If a citation label like [1] exists in the notes, keep it.\n"
            "- Return one statement per line, no numbering, no extra text.\n\n"
            "Notes:\n{notes}"
        ).format(n=self._config.max_items, notes=notes)
        try:
            response = self._llm.invoke(
                [{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
            )
            text = response if isinstance(response, str) else str(response)
            return _truncate(text.strip(), self._config.max_chars)
        except Exception as exc:
            logger.warning("Evidence extraction failed: %s", exc)
            return ""


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
