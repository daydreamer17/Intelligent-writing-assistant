from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_timeout: float = 60.0
    llm_max_tokens: Optional[int] = None
    storage_path: str = "data/app.db"
    memory_mode: str = "short_term"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "writing_memory"
    qdrant_embed_dim: int = 256
    qdrant_distance: str = "cosine"
    qdrant_timeout: float = 30.0
    embedding_provider: str = "openai_compatible"
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_timeout: float = 20.0
    embedding_probe: bool = True
    auto_update_env: bool = False
    upload_max_mb: int = 10
    github_token: str = ""
    github_mcp_enabled: bool = False

    @staticmethod
    def from_env() -> "AppConfig":
        def _parse_int(value: str) -> Optional[int]:
            value = value.strip()
            if not value:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip() or os.getenv("GITHUB_ACCESS_TOKEN", "").strip()
        github_mcp_enabled = os.getenv("MCP_GITHUB_ENABLED", "").strip().lower() in ("1", "true", "yes")
        if github_token and os.getenv("MCP_GITHUB_ENABLED") is None:
            github_mcp_enabled = True

        return AppConfig(
            llm_provider=os.getenv("LLM_PROVIDER", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_api_base=os.getenv("LLM_API_BASE", ""),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            llm_max_tokens=_parse_int(os.getenv("LLM_MAX_TOKENS", "")),
            storage_path=os.getenv("STORAGE_PATH", "data/app.db"),
            memory_mode=os.getenv("MEMORY_MODE", "short_term"),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "writing_memory"),
            qdrant_embed_dim=int(os.getenv("QDRANT_EMBED_DIM", "256")),
            qdrant_distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai_compatible"),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_api_base=os.getenv("EMBEDDING_API_BASE", ""),
            embedding_timeout=float(os.getenv("EMBEDDING_TIMEOUT", "20")),
            embedding_probe=os.getenv("EMBEDDING_PROBE", "true").lower() == "true",
            auto_update_env=os.getenv("AUTO_UPDATE_ENV", "false").lower() == "true",
            upload_max_mb=int(os.getenv("UPLOAD_MAX_MB", "10")),
            github_token=github_token,
            github_mcp_enabled=github_mcp_enabled,
        )
