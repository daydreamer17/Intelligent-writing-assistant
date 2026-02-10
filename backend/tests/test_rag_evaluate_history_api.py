import json
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


class _FakeStorageHistory:
    def __init__(self):
        self._rows = {
            7: SimpleNamespace(
                run_id=7,
                total_queries=2,
                queries_with_relevance=2,
                k_values_json=json.dumps([1, 3]),
                macro_metrics_json=json.dumps(
                    [
                        {
                            "k": 1,
                            "recall": 0.5,
                            "precision": 1.0,
                            "hit_rate": 1.0,
                            "mrr": 1.0,
                            "ndcg": 1.0,
                        }
                    ]
                ),
                per_query_json=json.dumps(
                    [
                        {
                            "query": "q",
                            "query_id": "qid",
                            "relevant_count": 1,
                            "retrieved_doc_ids": ["d1"],
                            "metrics": [
                                {
                                    "k": 1,
                                    "recall": 1.0,
                                    "precision": 1.0,
                                    "hit_rate": 1.0,
                                    "mrr": 1.0,
                                    "ndcg": 1.0,
                                }
                            ],
                        }
                    ]
                ),
                created_at="2026-02-09T00:00:00",
            )
        }

    def list_retrieval_eval_runs(self, limit: int = 20):
        return list(self._rows.values())[:limit]

    def get_retrieval_eval_run(self, run_id: int):
        return self._rows.get(run_id)

    def delete_retrieval_eval_run(self, run_id: int):
        return self._rows.pop(run_id, None) is not None


def test_rag_evaluation_history_endpoints():
    app = create_app()
    fake_storage = _FakeStorageHistory()
    app.dependency_overrides[get_services] = lambda: SimpleNamespace(storage=fake_storage)
    client = TestClient(app)

    list_resp = client.get("/api/rag/evaluations", params={"limit": 10})
    assert list_resp.status_code == 200
    list_body = list_resp.json()
    assert len(list_body["runs"]) == 1
    assert list_body["runs"][0]["run_id"] == 7

    detail_resp = client.get("/api/rag/evaluations/7")
    assert detail_resp.status_code == 200
    detail_body = detail_resp.json()
    assert detail_body["eval_run_id"] == 7
    assert detail_body["k_values"] == [1, 3]

    delete_resp = client.delete("/api/rag/evaluations/7")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    app.dependency_overrides.clear()
