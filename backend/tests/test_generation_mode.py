from __future__ import annotations

from app.services.generation_mode import (
    citation_labels_enabled,
    get_generation_mode,
    inference_mark_required,
    mcp_allowed_for_mode,
    refusal_enabled_for_mode,
    set_generation_mode,
)


def test_default_mode_and_invalid_fallback(monkeypatch):
    monkeypatch.delenv("RAG_GENERATION_MODE", raising=False)
    monkeypatch.delenv("RAG_CITATION_ENFORCE", raising=False)
    assert get_generation_mode() == "rag_only"

    monkeypatch.setenv("RAG_GENERATION_MODE", "invalid")
    assert get_generation_mode() == "rag_only"


def test_legacy_env_compatibility(monkeypatch):
    monkeypatch.delenv("RAG_GENERATION_MODE", raising=False)
    monkeypatch.setenv("RAG_CITATION_ENFORCE", "true")
    assert get_generation_mode() == "rag_only"

    monkeypatch.setenv("RAG_CITATION_ENFORCE", "false")
    assert get_generation_mode() == "hybrid"


def test_mode_capabilities(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_MODE", "rag_only")
    assert citation_labels_enabled(get_generation_mode()) is True
    assert refusal_enabled_for_mode(get_generation_mode()) is True
    assert mcp_allowed_for_mode(get_generation_mode()) is False
    assert inference_mark_required(get_generation_mode()) is False

    monkeypatch.setenv("RAG_GENERATION_MODE", "hybrid")
    assert citation_labels_enabled(get_generation_mode()) is True
    assert refusal_enabled_for_mode(get_generation_mode()) is False
    assert mcp_allowed_for_mode(get_generation_mode()) is True
    assert inference_mark_required(get_generation_mode()) is True

    monkeypatch.setenv("RAG_GENERATION_MODE", "creative")
    monkeypatch.delenv("RAG_CREATIVE_MCP_ENABLED", raising=False)
    assert citation_labels_enabled(get_generation_mode()) is False
    assert refusal_enabled_for_mode(get_generation_mode()) is False
    assert mcp_allowed_for_mode(get_generation_mode()) is True
    assert inference_mark_required(get_generation_mode()) is False

    monkeypatch.setenv("RAG_CREATIVE_MCP_ENABLED", "false")
    assert mcp_allowed_for_mode(get_generation_mode()) is False


def test_set_generation_mode_updates_legacy_flag(monkeypatch):
    monkeypatch.delenv("RAG_GENERATION_MODE", raising=False)
    monkeypatch.delenv("RAG_CITATION_ENFORCE", raising=False)

    assert set_generation_mode("hybrid") == "hybrid"
    assert get_generation_mode() == "hybrid"

    assert set_generation_mode("creative") == "creative"
    assert get_generation_mode() == "creative"

    assert set_generation_mode("rag_only") == "rag_only"
    assert get_generation_mode() == "rag_only"
