from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from uuid import UUID
from math import sqrt
from typing import Iterable, List, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .research_service import SourceDocument


@dataclass(frozen=True)
class VectorMatch:
    doc_id: str
    score: float


class Embedder(Protocol):
    def embed_one(self, text: str) -> List[float]:
        ...

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        ...


class VectorStore(Protocol):
    def upsert(self, docs: Iterable[SourceDocument]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> List[VectorMatch]:
        ...


class HashEmbedding:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_single(self, text: str) -> List[float]:
        counts = [0.0] * self.dim
        for token in self._tokenize(text):
            idx = self._hash_token(token)
            counts[idx] += 1.0
        norm = sqrt(sum(v * v for v in counts))
        if norm == 0:
            return counts
        return [v / norm for v in counts]

    def _hash_token(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def embed_one(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_single(text) for text in texts]


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection: str,
        embed_dim: int = 256,
        distance: str = "cosine",
        timeout: float | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        self.collection = collection
        self.embedder = embedder or HashEmbedding(embed_dim)
        self._ensure_collection(embed_dim, distance)

    def _ensure_collection(self, embed_dim: int, distance: str) -> None:
        qdrant_distance = self._map_distance(distance)
        collections = self.client.get_collections().collections
        if any(col.name == self.collection for col in collections):
            info = self.client.get_collection(self.collection)
            size = info.config.params.vectors.size
            existing_distance = info.config.params.vectors.distance
            if size != embed_dim:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' vector size {size} "
                    f"does not match embed_dim {embed_dim}"
                )
            if existing_distance != qdrant_distance:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' distance {existing_distance} "
                    f"does not match distance {qdrant_distance}"
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=embed_dim, distance=qdrant_distance),
        )

    @staticmethod
    def _map_distance(distance: str) -> qmodels.Distance:
        value = (distance or "cosine").lower()
        if value in ("cosine", "cos"):
            return qmodels.Distance.COSINE
        if value in ("dot", "dotproduct", "inner"):
            return qmodels.Distance.DOT
        if value in ("euclid", "euclidean", "l2"):
            return qmodels.Distance.EUCLID
        raise ValueError(f"Unsupported Qdrant distance: {distance}")

    def upsert(self, docs: Iterable[SourceDocument]) -> None:
        points = []
        for doc in docs:
            vector = self.embedder.embed_one(doc.title + " " + doc.content)
            points.append(
                qmodels.PointStruct(
                    id=_normalize_point_id(doc.doc_id),
                    vector=vector,
                    payload={"title": doc.title, "url": doc.url},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int = 5) -> List[VectorMatch]:
        vector = self.embedder.embed_one(query)
        results = self._search_points(vector, top_k)
        return [VectorMatch(doc_id=str(item.id), score=float(item.score)) for item in results]

    def _search_points(self, vector: List[float], top_k: int):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
            )
        if hasattr(self.client, "search_points"):
            fn = getattr(self.client, "search_points")
            try:
                response = fn(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=top_k,
                )
            except TypeError:
                response = fn(
                    collection_name=self.collection,
                    vector=vector,
                    limit=top_k,
                )
            return getattr(response, "result", response)
        raise AttributeError("QdrantClient has no search method")


def _normalize_point_id(doc_id: str) -> int | str:
    raw = (doc_id or "").strip()
    if raw.isdigit():
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        UUID(raw)
        return raw
    except Exception:
        return raw
