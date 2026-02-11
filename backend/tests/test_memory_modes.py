import os
import sys
from pathlib import Path

# Allow running this test file directly without needing PYTHONPATH setup.
backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import AppConfig  # noqa: E402


def test_retrieval_mode_falls_back_from_legacy_memory_mode(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("MEMORY_MODE", "long_term")

    cfg = AppConfig.from_env()

    assert cfg.retrieval_mode == "hybrid"
    assert cfg.memory_mode == "long_term"


def test_retrieval_mode_overrides_legacy_memory_mode(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "sqlite_only")
    monkeypatch.setenv("MEMORY_MODE", "long_term")

    cfg = AppConfig.from_env()

    assert cfg.retrieval_mode == "sqlite_only"
    # legacy compatibility value is derived from retrieval mode
    assert cfg.memory_mode == "short_term"


def test_conversation_memory_mode_validation(monkeypatch):
    monkeypatch.setenv("CONVERSATION_MEMORY_MODE", "invalid_value")

    cfg = AppConfig.from_env()

    assert cfg.conversation_memory_mode == "session"

