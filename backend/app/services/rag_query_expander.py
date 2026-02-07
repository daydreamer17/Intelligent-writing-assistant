from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

from hello_agents import HelloAgentsLLM

logger = logging.getLogger("app.rag")


@dataclass(frozen=True)
class QueryVariant:
    text: str
    weight: float
    source: str


@dataclass(frozen=True)
class QueryExpansionConfig:
    hyde_enabled: bool
    max_query_chars: int
    max_hyde_chars: int
    max_hyde_tokens: int
    max_variants: int

    @staticmethod
    def from_env() -> "QueryExpansionConfig":
        def _parse_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _parse_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if not raw:
                return default
            return raw.strip().lower() in ("1", "true", "yes")

        return QueryExpansionConfig(
            hyde_enabled=_parse_bool("RAG_HYDE_ENABLED", False),
            max_query_chars=max(200, _parse_int("RAG_QUERY_MAX_CHARS", 800)),
            max_hyde_chars=max(300, _parse_int("RAG_HYDE_MAX_CHARS", 1500)),
            max_hyde_tokens=max(64, _parse_int("RAG_HYDE_MAX_TOKENS", 400)),
            max_variants=max(3, _parse_int("RAG_MAX_EXPANSION_QUERIES", 3)),
        )


class QueryExpander:
    def __init__(self, llm: HelloAgentsLLM, config: QueryExpansionConfig) -> None:
        self._llm = llm
        self._config = config

    def expand(self, query: str) -> List[QueryVariant]:
        base = _truncate(query, self._config.max_query_chars)
        variants = [QueryVariant(text=base, weight=1.0, source="original")]

        if self._config.hyde_enabled:
            hyde = self._generate_hyde(base)
            if hyde:
                variants.append(QueryVariant(text=hyde, weight=0.8, source="hyde"))

        deduped = _dedupe_variants(variants, self._config.max_variants)
        if len(deduped) > 1:
            logger.info(
                "RAG expansion enabled: %s variants (hyde=%s).",
                len(deduped),
                self._config.hyde_enabled,
            )
        return deduped

    def _generate_hyde(self, query: str) -> str:
        prompt = (
            "Write a concise hypothetical answer that would appear in documents relevant to the user request.\n"
            "Use plain text, 1-2 short paragraphs, no citations.\n\n"
            f"User request:\n{query}"
        )
        text = self._invoke(prompt, max_tokens=self._config.max_hyde_tokens)
        if not text:
            return ""
        return _truncate(text, self._config.max_hyde_chars)

    def _invoke(self, prompt: str, *, max_tokens: int) -> str:
        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}], max_tokens=max_tokens)
            if isinstance(response, str):
                return response.strip()
            return str(response).strip()
        except Exception as exc:
            logger.warning("RAG query expansion failed: %s", exc)
            return ""

def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _dedupe_variants(variants: Iterable[QueryVariant], max_count: int) -> List[QueryVariant]:
    seen = set()
    results: List[QueryVariant] = []
    for item in variants:
        key = " ".join(item.text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= max_count:
            break
    return results
