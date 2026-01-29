import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("STORAGE_PATH", "data/test.db")

# Allow running this test file directly without needing PYTHONPATH setup.
backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.api.main import create_app  # noqa: E402


def test_health_check():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
