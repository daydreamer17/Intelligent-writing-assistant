from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional

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
    ) -> None:
        self._documents: List[SourceDocument] = []
        self._storage = storage
        self._vector_store = vector_store

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
            self._vector_store.upsert(added)
        return added

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
        if self._vector_store and self._storage and query:
            matches = self._vector_store.search(query, top_k=top_k)
            resolved = self._resolve_matches(matches)
            if resolved:
                return resolved
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
            return self._rank_documents(query, candidates, top_k=top_k)
        if not query:
            return self._documents[:top_k]
        return self._rank_documents(query, self._documents, top_k=top_k)

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
        return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token}

    @staticmethod
    def _vectorize(text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in RAGService._tokenize_list(text):
            counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _tokenize_list(text: str) -> List[str]:
        return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]

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

    def _rank_documents(
        self,
        query: str,
        docs: Iterable[SourceDocument],
        *,
        top_k: int,
    ) -> List[SourceDocument]:
        if not query:
            return list(docs)[:top_k]
        query_vec = self._vectorize(query)
        scored = sorted(
            docs,
            key=lambda doc: (
                self._cosine_similarity(query_vec, self._vectorize(doc.title + " " + doc.content)),
                self._score(query, doc.content, doc.title),
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
