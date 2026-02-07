from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List

from hello_agents import HelloAgentsLLM

from .research_service import SourceDocument

logger = logging.getLogger("app.rag")


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool
    top_k: int
    max_candidates: int
    max_snippet_chars: int
    max_prompt_chars: int
    max_tokens: int

    @staticmethod
    def from_env() -> "RerankConfig":
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

        return RerankConfig(
            enabled=_parse_bool("RAG_RERANK_ENABLED", False),
            top_k=max(1, _parse_int("RAG_RERANK_TOP_K", 5)),
            max_candidates=max(5, _parse_int("RAG_RERANK_MAX_CANDIDATES", 8)),
            max_snippet_chars=max(200, _parse_int("RAG_RERANK_SNIPPET_CHARS", 600)),
            max_prompt_chars=max(1000, _parse_int("RAG_RERANK_MAX_PROMPT_CHARS", 6000)),
            max_tokens=max(64, _parse_int("RAG_RERANK_MAX_TOKENS", 300)),
        )


class Reranker:
    def __init__(self, llm: HelloAgentsLLM, config: RerankConfig) -> None:
        self._llm = llm
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def oversample_factor(self) -> int:
        return max(2, int(self._config.max_candidates / max(self._config.top_k, 1)))

    def rerank(self, query: str, docs: Iterable[SourceDocument], *, top_k: int) -> List[SourceDocument]:
        if not self._config.enabled:
            return list(docs)[:top_k]

        candidates = list(docs)[: self._config.max_candidates]
        if len(candidates) <= 1:
            return candidates

        prompt = self._build_prompt(query, candidates)
        ordered_ids = self._call_llm(prompt)
        if not ordered_ids:
            return candidates[:top_k]

        by_id = {doc.doc_id: doc for doc in candidates}
        ranked: List[SourceDocument] = []
        for doc_id in ordered_ids:
            doc = by_id.get(doc_id)
            if doc:
                ranked.append(doc)
        # Append any missing docs in original order
        for doc in candidates:
            if doc.doc_id not in {d.doc_id for d in ranked}:
                ranked.append(doc)
        return ranked[:top_k]

    def _build_prompt(self, query: str, docs: List[SourceDocument]) -> str:
        blocks: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            snippet = (doc.content or "").strip().replace("\n", " ")
            if len(snippet) > self._config.max_snippet_chars:
                snippet = snippet[: self._config.max_snippet_chars].rstrip()
            blocks.append(
                f"[{idx}] id={doc.doc_id}\nTitle: {doc.title}\nSnippet: {snippet}"
            )
        prompt = (
            "You are a retrieval reranker. Given the user query and candidate documents, "
            "return a JSON array of document ids ordered by relevance (most relevant first). "
            "Only return the JSON array, no extra text.\n\n"
            f"Query: {query}\n\nCandidates:\n" + "\n\n".join(blocks)
        )
        if len(prompt) > self._config.max_prompt_chars:
            prompt = prompt[: self._config.max_prompt_chars].rstrip()
        return prompt

    def _call_llm(self, prompt: str) -> List[str]:
        try:
            response = self._llm.invoke(
                [{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
            )
            text = response if isinstance(response, str) else str(response)
            return _parse_id_list(text)
        except Exception as exc:
            logger.warning("RAG rerank failed: %s", exc)
            return []


def _parse_id_list(text: str) -> List[str]:
    if not text:
        return []
    # Extract first JSON array
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    raw = match.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if item]
