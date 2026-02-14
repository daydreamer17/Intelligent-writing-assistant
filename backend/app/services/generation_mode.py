from __future__ import annotations

import os
from typing import Literal

GenerationMode = Literal["rag_only", "hybrid", "creative"]

_VALID_MODES = {"rag_only", "hybrid", "creative"}
_TRUE_VALUES = {"1", "true", "yes"}


def get_generation_mode() -> GenerationMode:
    raw = (os.getenv("RAG_GENERATION_MODE", "") or "").strip().lower()
    if raw in _VALID_MODES:
        return raw  # type: ignore[return-value]

    # Backward compatibility:
    # - RAG_CITATION_ENFORCE=true  => rag_only
    # - RAG_CITATION_ENFORCE=false => hybrid
    legacy = (os.getenv("RAG_CITATION_ENFORCE", "") or "").strip().lower()
    if legacy in {"1", "true", "yes"}:
        return "rag_only"
    if legacy in {"0", "false", "no"}:
        return "hybrid"
    return "rag_only"


def set_generation_mode(mode: str) -> GenerationMode:
    normalized = (mode or "").strip().lower()
    if normalized not in _VALID_MODES:
        normalized = "rag_only"
    os.environ["RAG_GENERATION_MODE"] = normalized
    # Keep legacy flag in sync for backward compatibility with old code paths.
    os.environ["RAG_CITATION_ENFORCE"] = "true" if normalized == "rag_only" else "false"
    return normalized  # type: ignore[return-value]


def get_creative_mcp_enabled() -> bool:
    raw = (os.getenv("RAG_CREATIVE_MCP_ENABLED", "true") or "").strip().lower()
    return raw in _TRUE_VALUES


def set_creative_mcp_enabled(enabled: bool) -> bool:
    os.environ["RAG_CREATIVE_MCP_ENABLED"] = "true" if enabled else "false"
    return enabled


def is_rag_only(mode: GenerationMode | None = None) -> bool:
    return (mode or get_generation_mode()) == "rag_only"


def is_hybrid(mode: GenerationMode | None = None) -> bool:
    return (mode or get_generation_mode()) == "hybrid"


def is_creative(mode: GenerationMode | None = None) -> bool:
    return (mode or get_generation_mode()) == "creative"


def citation_labels_enabled(mode: GenerationMode | None = None) -> bool:
    resolved = mode or get_generation_mode()
    return resolved in {"rag_only", "hybrid"}


def refusal_enabled_for_mode(mode: GenerationMode | None = None) -> bool:
    return is_rag_only(mode)


def mcp_allowed_for_mode(mode: GenerationMode | None = None) -> bool:
    resolved = mode or get_generation_mode()
    if resolved == "hybrid":
        return True
    if resolved == "creative":
        # Creative mode enables MCP by default. Users can explicitly disable it via env.
        return get_creative_mcp_enabled()
    return False


def inference_mark_required(mode: GenerationMode | None = None) -> bool:
    return is_hybrid(mode)


def conversation_memory_enabled_for_mode(mode: GenerationMode | None = None) -> bool:
    """Whether session/cold memory should participate in prompt construction.

    Defaults:
    - rag_only/hybrid: enabled
    - creative: disabled (to avoid topic contamination from previous tasks)

    Can be overridden for creative mode via:
    RAG_CREATIVE_MEMORY_ENABLED=true|false
    """
    resolved = mode or get_generation_mode()
    if resolved != "creative":
        return True
    raw = (os.getenv("RAG_CREATIVE_MEMORY_ENABLED", "false") or "").strip().lower()
    return raw in _TRUE_VALUES
