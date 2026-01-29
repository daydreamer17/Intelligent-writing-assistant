from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import httpx


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    timeout: float = 20.0


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig) -> None:
        if not config.base_url:
            raise ValueError("Embedding base_url is required")
        if not config.model:
            raise ValueError("Embedding model is required")
        self._config = config
        self._client = httpx.Client(timeout=config.timeout)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        payload = {
            "input": list(texts),
            "model": self._config.model,
        }
        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        url = self._config.base_url.rstrip("/") + "/v1/embeddings"
        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        if len(embeddings) != len(payload["input"]):
            raise ValueError("Embedding response size mismatch")
        return embeddings

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
