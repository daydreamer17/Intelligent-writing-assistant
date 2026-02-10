import os
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

os.environ.setdefault("STORAGE_PATH", "data/test.db")

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.api.deps import get_services  # noqa: E402
from app.api.main import create_app  # noqa: E402
from app.services.research_service import SourceDocument  # noqa: E402


class _FakeRag:
    def __init__(self):
        self._docs = {
            "agent": [
                SourceDocument(doc_id="d1", title="A", content="..."),
                SourceDocument(doc_id="d2", title="B", content="..."),
                SourceDocument(doc_id="d3", title="C", content="..."),
            ],
            "mcp": [
                SourceDocument(doc_id="d4", title="M", content="..."),
                SourceDocument(doc_id="d2", title="B", content="..."),
            ],
        }

    def search(self, query: str, top_k: int = 5):
        return self._docs.get(query, [])[:top_k]


class _FakeStorage:
    def save_retrieval_eval_run(self, **kwargs):
        return 1

    def get_retrieval_eval_run(self, run_id: int):
        return SimpleNamespace(run_id=run_id, created_at="2026-02-09T00:00:00")


def test_rag_evaluate_endpoint_returns_metrics():
    app = create_app()
    app.dependency_overrides[get_services] = lambda: SimpleNamespace(rag=_FakeRag(), storage=_FakeStorage())
    client = TestClient(app)

    payload = {
        "cases": [
            {"query": "agent", "relevant_doc_ids": ["d1", "d3"], "query_id": "q1"},
            {"query": "mcp", "relevant_doc_ids": ["d2"], "query_id": "q2"},
        ],
        "k_values": [1, 3],
    }
    resp = client.post("/api/rag/evaluate", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["total_queries"] == 2
    assert body["queries_with_relevance"] == 2
    assert body["k_values"] == [1, 3]
    assert body["eval_run_id"] == 1
    assert len(body["macro_metrics"]) == 2
    assert len(body["per_query"]) == 2

    app.dependency_overrides.clear()
