from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from math import sqrt
from typing import Dict, Iterable, List, Optional

from ..utils.tokenizer import tokenize, tokenize_list
from .rag_query_expander import QueryExpander, QueryVariant
from .rag_reranker import Reranker
from .research_service import SourceDocument
from .storage_service import StorageService
from .vector_store import VectorMatch, VectorStore


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

    def search(self, query: str, top_k: int = 5) -> List[SourceDocument]:
        query_variants = self._expand_queries(query)
        corpus_size = self._corpus_size()
        final_top_k, candidate_k = self._dynamic_search_plan(top_k=top_k, corpus_size=corpus_size)
        if self._vector_store and self._storage and query:
            try:
                matches = self._vector_search_multi(query_variants, top_k=candidate_k)
                resolved = self._resolve_matches(matches)
                if resolved:
                    return self._rerank_if_needed(query, resolved, top_k=final_top_k)
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
            return self._rerank_if_needed(query, ranked, top_k=final_top_k)
        if not query:
            return self._documents[:final_top_k]
        ranked = self._rank_documents_multi(query_variants, self._documents, top_k=candidate_k)
        return self._rerank_if_needed(query, ranked, top_k=final_top_k)

    def get_embedder(self):
        if self._vector_store and hasattr(self._vector_store, "embedder"):
            return getattr(self._vector_store, "embedder")
        return None

    def _candidate_k(self, top_k: int) -> int:
        if self._reranker and self._reranker.enabled:
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

    def _dynamic_search_plan(self, *, top_k: int, corpus_size: int) -> tuple[int, int]:
        if not _parse_bool_env("RAG_DYNAMIC_TOPK_ENABLED", True):
            return top_k, self._candidate_k(top_k)

        small_threshold = max(1, _parse_int_env("RAG_DYNAMIC_SMALL_THRESHOLD", 1000))
        large_threshold = max(small_threshold + 1, _parse_int_env("RAG_DYNAMIC_LARGE_THRESHOLD", 50000))

        if corpus_size <= small_threshold:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_SMALL", 8))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_SMALL", 30))
            bucket = "small"
        elif corpus_size >= large_threshold:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_LARGE", 12))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_LARGE", 120))
            bucket = "large"
        else:
            target_top_k = max(1, _parse_int_env("RAG_DYNAMIC_TOPK_MEDIUM", 10))
            target_candidates = max(target_top_k, _parse_int_env("RAG_DYNAMIC_CANDIDATES_MEDIUM", 60))
            bucket = "medium"

        # 动态策略优先：当调用方 top_k 偏小（如默认 5）时自动放大到动态目标。
        final_top_k = max(1, max(top_k, target_top_k))
        candidate_k = max(final_top_k, target_candidates)

        self._logger.info(
            "RAG dynamic search plan: corpus=%s bucket=%s top_k=%s candidates=%s",
            corpus_size,
            bucket,
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
    ) -> List[SourceDocument]:
        if not self._reranker or not self._reranker.enabled:
            return docs[:top_k]
        return self._reranker.rerank(query, docs, top_k=top_k)

    def _expand_queries(self, query: str) -> List[QueryVariant]:
        base = (query or "").strip()
        if not base:
            return []
        if self._query_expander is None:
            return [QueryVariant(text=base, weight=1.0, source="original")]
        return self._query_expander.expand(base)

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
