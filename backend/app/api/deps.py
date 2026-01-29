from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path

from ..agents.editor_agent import EditorAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.writing_agent import WritingAgent
from ..config import AppConfig
from ..services.citation_service import CitationService
from ..services.drafting_service import DraftingService
from ..services.pipeline_service import WritingPipeline
from ..services.planner_service import PlanningService
from ..services.embedding_service import EmbeddingConfig, EmbeddingService
from ..services.rag_service import RAGService
from ..services.research_service import ResearchService
from ..services.reviewing_service import ReviewingService
from ..services.rewriting_service import RewritingService
from ..services.storage_service import StorageService
from ..services.vector_store import HashEmbedding, QdrantVectorStore, VectorStore


@dataclass(frozen=True)
class AppServices:
    planner: PlanningService
    reviewer: ReviewingService
    rewriter: RewritingService
    drafter: DraftingService
    pipeline: WritingPipeline
    rag: RAGService
    citations: CitationService
    storage: StorageService
    memory_mode: str
    storage_path: str
    qdrant_enabled: bool
    qdrant_url: str
    qdrant_collection: str
    qdrant_dim: int
    qdrant_distance: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int | None
    embedding_mode: str
    embedding_available: bool
    embedding_error: str | None
    qdrant_available: bool
    qdrant_error: str | None
    upload_max_bytes: int


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    logger = logging.getLogger("app.deps")
    config = AppConfig.from_env()

    def _mask(value: str) -> str:
        if not value:
            return "unset"
        if len(value) <= 6:
            return "*" * len(value)
        return value[:3] + "..." + value[-2:]

    logger.info(
        "Config summary: memory_mode=%s storage_path=%s llm_timeout=%ss llm_max_tokens=%s qdrant_url=%s qdrant_collection=%s "
        "qdrant_dim=%s qdrant_distance=%s qdrant_timeout=%ss embedding_provider=%s embedding_model=%s "
        "embedding_api_base=%s embedding_api_key=%s",
        config.memory_mode,
        config.storage_path,
        config.llm_timeout,
        config.llm_max_tokens if config.llm_max_tokens is not None else "unset",
        config.qdrant_url or "unset",
        config.qdrant_collection,
        config.qdrant_embed_dim,
        config.qdrant_distance,
        config.qdrant_timeout,
        config.embedding_provider or "hash",
        config.embedding_model or "unset",
        config.embedding_api_base or "unset",
        _mask(config.embedding_api_key),
    )

    writing_agent = WritingAgent(
        provider=config.llm_provider or None,
        model=config.llm_model or None,
        api_key=config.llm_api_key or None,
        base_url=config.llm_api_base or None,
        request_timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )
    reviewer_agent = ReviewerAgent(
        provider=config.llm_provider or None,
        model=config.llm_model or None,
        api_key=config.llm_api_key or None,
        base_url=config.llm_api_base or None,
        request_timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )
    editor_agent = EditorAgent(
        provider=config.llm_provider or None,
        model=config.llm_model or None,
        api_key=config.llm_api_key or None,
        base_url=config.llm_api_base or None,
        request_timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )

    storage = StorageService(config.storage_path)
    planner = PlanningService(writing_agent)
    reviewer = ReviewingService(reviewer_agent)
    rewriter = RewritingService(editor_agent)
    drafter = DraftingService(
        writing_agent=writing_agent,
        reviewing_service=reviewer,
        rewriting_service=rewriter,
    )
    pipeline = WritingPipeline(
        planner=planner,
        researcher=ResearchService(),
        drafter=drafter,
    )
    vector_store: VectorStore | None = None
    qdrant_dim = config.qdrant_embed_dim
    embedding_dim: int | None = None
    embedding_mode = "hash"
    embedding_error: str | None = None
    qdrant_error: str | None = None
    if config.memory_mode == "long_term" and config.qdrant_url:
        embedder = None
        base_url = config.embedding_api_base or config.llm_api_base
        missing = []
        if config.embedding_provider and config.embedding_provider != "hash":
            if not config.embedding_model:
                missing.append("EMBEDDING_MODEL")
            if not base_url:
                missing.append("EMBEDDING_API_BASE")
            if missing:
                logger.warning(
                    "Embedding config incomplete (%s). Falling back to hash embedding.",
                    ", ".join(missing),
                )
                embedder = HashEmbedding(config.qdrant_embed_dim)
                embedding_dim = config.qdrant_embed_dim
                embedding_mode = "hash"
            else:
                embedder = EmbeddingService(
                    EmbeddingConfig(
                        provider=config.embedding_provider,
                        model=config.embedding_model,
                        api_key=config.embedding_api_key,
                        base_url=base_url or "",
                        timeout=config.embedding_timeout,
                    )
                )
                if config.embedding_probe:
                    probe_dim, probe_error = _probe_embedding_dim(embedder, logger)
                    if probe_dim is None:
                        logger.warning("Embedding probe failed. Falling back to hash embedding.")
                        embedder = HashEmbedding(config.qdrant_embed_dim)
                        embedding_dim = config.qdrant_embed_dim
                        embedding_mode = "hash"
                        embedding_error = probe_error or "probe_failed"
                    elif probe_dim != config.qdrant_embed_dim:
                        logger.warning(
                            "Embedding dim mismatch (model=%s, qdrant_dim=%s). Using probed dim for Qdrant.",
                            probe_dim,
                            config.qdrant_embed_dim,
                        )
                        qdrant_dim = probe_dim
                        embedding_dim = probe_dim
                        embedding_mode = "embedding"
                        if config.auto_update_env:
                            _maybe_update_env_dim(logger, qdrant_dim)
                    else:
                        embedding_dim = probe_dim
                        embedding_mode = "embedding"
                else:
                    logger.info("Embedding probe disabled; using configured QDRANT_EMBED_DIM.")
                    embedding_dim = config.qdrant_embed_dim
                    embedding_mode = "embedding"
        else:
            embedder = HashEmbedding(config.qdrant_embed_dim)
            embedding_dim = config.qdrant_embed_dim
            embedding_mode = "hash"

        try:
            vector_store = QdrantVectorStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key or None,
                collection=config.qdrant_collection,
                embed_dim=qdrant_dim,
                distance=config.qdrant_distance,
                timeout=config.qdrant_timeout,
                embedder=embedder,
            )
            logger.info(
                "Qdrant enabled: collection=%s dim=%s distance=%s provider=%s",
                config.qdrant_collection,
                qdrant_dim,
                config.qdrant_distance,
                config.embedding_provider or "hash",
            )
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s). Falling back to SQLite-only RAG.", exc)
            vector_store = None
            qdrant_error = str(exc)
    rag = RAGService(storage=storage, vector_store=vector_store)
    citations = CitationService()

    return AppServices(
        planner=planner,
        reviewer=reviewer,
        rewriter=rewriter,
        drafter=drafter,
        pipeline=pipeline,
        rag=rag,
        citations=citations,
        storage=storage,
        memory_mode=config.memory_mode,
        storage_path=config.storage_path,
        qdrant_enabled=vector_store is not None,
        qdrant_url=config.qdrant_url or "",
        qdrant_collection=config.qdrant_collection,
        qdrant_dim=qdrant_dim,
        qdrant_distance=config.qdrant_distance,
        embedding_provider=config.embedding_provider or "hash",
        embedding_model=config.embedding_model or "",
        embedding_dim=embedding_dim,
        embedding_mode=embedding_mode,
        embedding_available=embedding_mode == "embedding",
        embedding_error=embedding_error,
        qdrant_available=vector_store is not None,
        qdrant_error=qdrant_error,
        upload_max_bytes=max(1, config.upload_max_mb) * 1024 * 1024,
    )


def _probe_embedding_dim(
    embedder: EmbeddingService, logger: logging.Logger
) -> tuple[int | None, str | None]:
    try:
        vector = embedder.embed_one("dimension check")
        dim = len(vector)
        logger.info("Embedding probe dim=%s", dim)
        return dim, None
    except Exception as exc:
        logger.warning("Embedding probe error: %s", exc)
        return None, str(exc)


def _maybe_update_env_dim(logger: logging.Logger, dim: int) -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read .env for update: %s", exc)
        return
    lines = content.splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith("QDRANT_EMBED_DIM="):
            lines[idx] = f"QDRANT_EMBED_DIM={dim}"
            updated = True
            break
    if not updated:
        lines.append(f"QDRANT_EMBED_DIM={dim}")
    try:
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Updated .env QDRANT_EMBED_DIM to %s", dim)
    except Exception as exc:
        logger.warning("Failed to update .env QDRANT_EMBED_DIM: %s", exc)
