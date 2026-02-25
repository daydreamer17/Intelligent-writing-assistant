from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from math import sqrt
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

from ..utils.tokenizer import tokenize, tokenize_list
from .rag_query_expander import QueryExpander, QueryVariant
from .rag_reranker import Reranker
from .research_service import SourceDocument
from .storage_service import StorageService
from .vector_store import VectorMatch, VectorStore

if TYPE_CHECKING:
    from ..models.schemas import RagEvalConfigOverride


@dataclass(frozen=True)
class UploadedDocument:
    doc_id: str
    title: str
    content: str
    url: str = ""


class RAGService:
    def __init__(
        self,
        storage: Optional[StorageService] = None,
        vector_store: Optional[VectorStore] = None,
        query_expander: Optional[QueryExpander] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self._documents: List[SourceDocument] = []
        self._storage = storage
        self._vector_store = vector_store
        self._query_expander = query_expander
        self._reranker = reranker
        self._logger = logging.getLogger("app.rag")

    def add_documents(self, docs: Iterable[UploadedDocument]) -> List[SourceDocument]:
        added: List[SourceDocument] = []
        for doc in docs:
            added.append(
                SourceDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                )
            )
        if self._storage:
            self._storage.upsert_documents(added)
        else:
            self._documents.extend(added)
        if self._vector_store:
            try:
                self._vector_store.upsert(added)
            except Exception as exc:
                self._logger.warning("Vector upsert failed, falling back to SQLite-only: %s", exc)
        return added

    def delete_document(self, doc_id: str) -> bool:
        deleted = False
        if self._storage:
            deleted = self._storage.delete_document(doc_id)
        if self._vector_store:
            try:
                self._vector_store.delete([doc_id])
            except Exception as exc:
                self._logger.warning("Vector delete failed: %s", exc)
        return deleted

    def list_documents(self) -> List[SourceDocument]:
        if self._storage:
            stored = self._storage.list_documents()
            return [
                SourceDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                )
                for doc in stored
            ]
        return list(self._documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        rag_eval_override: "RagEvalConfigOverride | None" = None,
    ) -> List[SourceDocument]:
        rerank_enabled_override = _get_override_bool(rag_eval_override, "rerank_enabled")
        hyde_enabled_override = _get_override_bool(rag_eval_override, "hyde_enabled")
        bilingual_rewrite_enabled_override = _get_override_bool(
            rag_eval_override, "bilingual_rewrite_enabled"
        )
        routed = self._route_query_strategy(query) if rag_eval_override is None else None
        if routed:
            rerank_enabled_override = (
                routed.get("rerank_enabled")
                if rerank_enabled_override is None
                else rerank_enabled_override
            )
            hyde_enabled_override = (
                routed.get("hyde_enabled")
                if hyde_enabled_override is None
                else hyde_enabled_override
            )
            bilingual_rewrite_enabled_override = (
                routed.get("bilingual_rewrite_enabled")
                if bilingual_rewrite_enabled_override is None
                else bilingual_rewrite_enabled_override
            )

        if rag_eval_override is not None:
            self._logger.info(
                "RAG eval override applied: rerank=%s hyde=%s bilingual=%s",
                rerank_enabled_override,
                hyde_enabled_override,
                bilingual_rewrite_enabled_override,
            )
        elif routed:
            self._logger.info(
                "RAG query strategy route: profile=%s strategy=%s query=%r",
                routed.get("profile"),
                routed.get("strategy"),
                _truncate_for_log(query, 160),
            )

        query_variants = self._expand_queries(
            query,
            hyde_enabled_override=hyde_enabled_override,
            bilingual_rewrite_enabled_override=bilingual_rewrite_enabled_override,
        )
        corpus_size = self._corpus_size()
        final_top_k, candidate_k = self._dynamic_search_plan(
            top_k=top_k,
            corpus_size=corpus_size,
            rerank_enabled_override=rerank_enabled_override,
        )
        if self._vector_store and self._storage and query:
            try:
                matches = self._vector_search_multi(query_variants, top_k=candidate_k)
                resolved = self._resolve_matches(matches)
                if resolved:
                    return self._rerank_if_needed(
                        query,
                        resolved,
                        top_k=final_top_k,
                        enabled_override=rerank_enabled_override,
                    )
            except Exception as exc:
                self._logger.warning("Vector search failed, falling back to keyword search: %s", exc)
        if self._storage:
            stored = self._storage.list_documents()
            candidates = [
                SourceDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                )
                for doc in stored
            ]
            ranked = self._rank_documents_multi(query_variants, candidates, top_k=candidate_k)
            return self._rerank_if_needed(
                query,
                ranked,
                top_k=final_top_k,
                enabled_override=rerank_enabled_override,
            )
        if not query:
            return self._documents[:final_top_k]
        ranked = self._rank_documents_multi(query_variants, self._documents, top_k=candidate_k)
        return self._rerank_if_needed(
            query,
            ranked,
            top_k=final_top_k,
            enabled_override=rerank_enabled_override,
        )

    def query_variants(self, query: str) -> List[QueryVariant]:
        """Return the active retrieval query variants for diagnostics/policy checks."""
        return self._expand_queries(query)

    def get_embedder(self):
        if self._vector_store and hasattr(self._vector_store, "embedder"):
            return getattr(self._vector_store, "embedder")
        return None

    def _candidate_k(self, top_k: int, *, rerank_enabled_override: bool | None = None) -> int:
        rerank_enabled = (
            bool(rerank_enabled_override)
            if rerank_enabled_override is not None
            else bool(self._reranker and self._reranker.enabled)
        )
        if self._reranker and rerank_enabled:
            factor = self._reranker.oversample_factor
            return max(top_k, top_k * factor)
        return top_k

    def _corpus_size(self) -> int:
        if self._storage:
            try:
                return len(self._storage.list_documents())
            except Exception:
                return len(self._documents)
        return len(self._documents)

    def _dynamic_search_plan(
        self,
        *,
        top_k: int,
        corpus_size: int,
        rerank_enabled_override: bool | None = None,
    ) -> tuple[int, int]:
        requested_top_k = max(1, top_k)
        if corpus_size <= 0:
            return requested_top_k, requested_top_k

        if not _parse_bool_env("RAG_DYNAMIC_TOPK_ENABLED", True):
            candidate_k = min(
                corpus_size,
                max(
                    requested_top_k,
                    self._candidate_k(
                        requested_top_k, rerank_enabled_override=rerank_enabled_override
                    ),
                ),
            )
            final_top_k = min(corpus_size, requested_top_k)
            return final_top_k, candidate_k

        small_threshold = max(1, _parse_int_env("RAG_DYNAMIC_SMALL_THRESHOLD", 50))
        large_threshold = max(small_threshold + 1, _parse_int_env("RAG_DYNAMIC_LARGE_THRESHOLD", 500))

        if corpus_size <= small_threshold:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_SMALL", 5))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_SMALL", 15))
            bucket = "small"
        elif corpus_size >= large_threshold:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_LARGE", 12))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_LARGE", 36))
            bucket = "large"
        else:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_MEDIUM", 10))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_MEDIUM", 24))
            bucket = "medium"

        # Dynamic target can expand the caller's requested top_k.
        final_top_k = min(corpus_size, max(requested_top_k, target_top_k))

        # For very small corpora, keep one document out so ranking still filters.
        if corpus_size < 10 and corpus_size > 1:
            final_top_k = max(1, min(final_top_k, corpus_size - 1))
            candidate_k = corpus_size
        else:
            candidate_k = min(corpus_size, max(final_top_k, target_candidates))

        self._logger.info(
            "RAG dynamic search plan: corpus=%s bucket=%s requested_top_k=%s target_top_k=%s final_top_k=%s candidates=%s",
            corpus_size,
            bucket,
            requested_top_k,
            target_top_k,
            final_top_k,
            candidate_k,
        )
        return final_top_k, candidate_k

    def _rerank_if_needed(
        self,
        query: str,
        docs: List[SourceDocument],
        *,
        top_k: int,
        enabled_override: bool | None = None,
    ) -> List[SourceDocument]:
        if not self._reranker:
            return docs[:top_k]
        return self._reranker.rerank(query, docs, top_k=top_k, enabled_override=enabled_override)

    def _expand_queries(
        self,
        query: str,
        *,
        hyde_enabled_override: bool | None = None,
        bilingual_rewrite_enabled_override: bool | None = None,
    ) -> List[QueryVariant]:
        base = (query or "").strip()
        if not base:
            return []
        if self._query_expander is None:
            return [QueryVariant(text=base, weight=1.0, source="original")]
        return self._query_expander.expand(
            base,
            hyde_enabled=hyde_enabled_override,
            bilingual_rewrite_enabled=bilingual_rewrite_enabled_override,
        )

    def _route_query_strategy(self, query: str) -> dict[str, object] | None:
        if not _parse_bool_env("RAG_QUERY_STRATEGY_ROUTING_ENABLED", False):
            return None
        text = (query or "").strip()
        if not text:
            return None

        profile = _profile_query(text)
        has_zh = bool(profile["has_zh"])
        has_en = bool(profile["has_en"])
        title_like = bool(profile["title_like"])
        cross_lingual = bool(profile["cross_lingual"])
        multi_concept = bool(profile["multi_concept"])
        multi_relevant = bool(profile["multi_relevant_hint"])

        strategy = "passthrough"
        rerank = None
        hyde = None
        bilingual = None

        # Low-cost path for title-like queries (exact/near-exact filenames/titles).
        if title_like and not cross_lingual and not multi_concept and not multi_relevant:
            strategy = "dense_only_title_like"
            rerank = False
            hyde = False
            bilingual = False
        # Hard mixed-language / multi-target queries: enable all three.
        elif cross_lingual and (multi_concept or multi_relevant):
            strategy = "dense_hyde_rerank_bilingual"
            rerank = True
            hyde = True
            bilingual = True
        # Cross-lingual without explicit multi-target: prefer bilingual + rerank.
        elif cross_lingual:
            strategy = "dense_rerank_bilingual"
            rerank = True
            hyde = False
            bilingual = True
        # Semantic/multi-concept monolingual queries: rerank usually gives best cost/quality tradeoff.
        elif multi_concept or multi_relevant or (has_zh or has_en):
            strategy = "dense_rerank_semantic"
            rerank = True
            hyde = False
            bilingual = False

        if strategy == "passthrough":
            return None
        return {
            "strategy": strategy,
            "profile": profile,
            "rerank_enabled": rerank,
            "hyde_enabled": hyde,
            "bilingual_rewrite_enabled": bilingual,
        }

    def _vector_search_multi(
        self,
        variants: List[QueryVariant],
        *,
        top_k: int,
    ) -> List[VectorMatch]:
        if not variants:
            return []
        per_k = max(2, int((top_k * 1.5) / max(len(variants), 1)))
        scores: Dict[str, float] = {}
        for variant in variants:
            matches = self._vector_store.search(variant.text, top_k=per_k) if self._vector_store else []
            for match in matches:
                score = match.score * variant.weight
                existing = scores.get(match.doc_id)
                if existing is None or score > existing:
                    scores[match.doc_id] = score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [VectorMatch(doc_id=doc_id, score=score) for doc_id, score in ranked[:top_k]]

    @staticmethod
    def _score(query: str, content: str, title: str) -> int:
        query_terms = RAGService._tokenize(query)
        if not query_terms:
            return 0
        content_terms = RAGService._tokenize(content)
        title_terms = RAGService._tokenize(title)
        return len(query_terms & content_terms) + 2 * len(query_terms & title_terms)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """使用统一的分词工具（支持中文）"""
        return tokenize(text, lowercase=True)

    @staticmethod
    def _vectorize(text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in RAGService._tokenize_list(text):
            counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _tokenize_list(text: str) -> List[str]:
        """使用统一的分词工具（支持中文，返回列表）"""
        return tokenize_list(text, lowercase=True)

    @staticmethod
    def _cosine_similarity(left: Dict[str, int], right: Dict[str, int]) -> float:
        if not left or not right:
            return 0.0
        dot = 0.0
        for token, value in left.items():
            dot += value * right.get(token, 0)
        left_norm = sqrt(sum(v * v for v in left.values()))
        right_norm = sqrt(sum(v * v for v in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _rank_documents_multi(
        self,
        variants: List[QueryVariant],
        docs: Iterable[SourceDocument],
        *,
        top_k: int,
    ) -> List[SourceDocument]:
        if not variants:
            return list(docs)[:top_k]
        query_vecs = [(variant, self._vectorize(variant.text)) for variant in variants]
        doc_text_cache: Dict[str, Dict[str, int]] = {}
        scored = sorted(
            docs,
            key=lambda doc: sum(
                variant.weight
                * (
                    self._cosine_similarity(
                        q_vec,
                        doc_text_cache.setdefault(
                            doc.doc_id, self._vectorize(doc.title + " " + doc.content)
                        ),
                    )
                    + 0.5 * self._score(variant.text, doc.content, doc.title)
                )
                for variant, q_vec in query_vecs
            ),
            reverse=True,
        )
        return scored[:top_k]

    def _resolve_matches(self, matches: List[VectorMatch]) -> List[SourceDocument]:
        if not self._storage:
            return []
        doc_ids = [match.doc_id for match in matches]
        docs = self._storage.get_documents_by_ids(doc_ids)
        by_id = {doc.doc_id: doc for doc in docs}
        return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_override_bool(override: object | None, attr: str) -> bool | None:
    if override is None:
        return None
    value: Any
    if isinstance(override, Mapping):
        value = override.get(attr)
    else:
        try:
            value = getattr(override, attr, None)
        except Exception:
            return None
    if value is None:
        return None
    return bool(value)


_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z]")
_TITLE_HINT_RE = re.compile(
    r"(?i)(^week\s*\d+|ebu6502|lecture|tutorial|introduction|advanced|hadoop|mapreduce|cuda|\.pdf\b)"
)
_MULTI_CONCEPT_RE = re.compile(
    r"(?i)(\+|/| and | both | together | rather than |\bvs\b|与|以及|同时|并且|而不是|两份|都算相关)"
)


def _profile_query(query: str) -> dict[str, bool]:
    text = (query or "").strip()
    has_zh = bool(_ZH_RE.search(text))
    has_en = bool(_EN_RE.search(text))
    normalized = f" {text.lower()} "
    title_like = bool(_TITLE_HINT_RE.search(text)) and len(text) <= 180
    multi_concept = bool(_MULTI_CONCEPT_RE.search(normalized))
    multi_relevant_hint = any(token in text for token in ("两份都算相关", "都算相关", "multiple", "both"))
    cross_lingual = (has_zh and has_en) or (
        has_zh
        and any(term in normalized for term in (" cuda", " hadoop", " mapreduce", " nosql", " graph", " sla", " gpu"))
    )
    return {
        "has_zh": has_zh,
        "has_en": has_en,
        "title_like": title_like,
        "cross_lingual": cross_lingual,
        "multi_concept": multi_concept,
        "multi_relevant_hint": multi_relevant_hint,
    }


def _truncate_for_log(text: str, max_chars: int) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."
