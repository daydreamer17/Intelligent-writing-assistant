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


def test_generation_mode_settings_compatibility():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/api/settings/generation-mode")
    assert resp.status_code == 200
    assert resp.json()["mode"] in {"rag_only", "hybrid", "creative"}
    assert isinstance(resp.json()["creative_mcp_enabled"], bool)

    resp = client.post("/api/settings/generation-mode", json={"mode": "hybrid"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "hybrid"
    assert resp.json()["citation_enforce"] is False
    assert resp.json()["mcp_allowed"] is True
    assert resp.json()["inference_mark_required"] is True
    assert isinstance(resp.json()["creative_mcp_enabled"], bool)

    resp = client.post(
        "/api/settings/generation-mode",
        json={"mode": "creative", "creative_mcp_enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "creative"
    assert resp.json()["creative_mcp_enabled"] is False
    assert resp.json()["mcp_allowed"] is False

    resp = client.get("/api/settings/citation")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    resp = client.post("/api/settings/citation", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True

    resp = client.get("/api/settings/generation-mode")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "rag_only"
