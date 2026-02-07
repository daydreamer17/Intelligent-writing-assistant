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
from ..services.evidence_service import EvidenceConfig, EvidenceExtractor
from ..services.pipeline_service import WritingPipeline
from ..services.planner_service import PlanningService
from ..services.embedding_service import EmbeddingConfig, EmbeddingService
from ..services.rag_query_expander import QueryExpander, QueryExpansionConfig
from ..services.rag_reranker import RerankConfig, Reranker
from ..services.rag_service import RAGService
from ..services.research_service import ResearchService
from ..services.reviewing_service import ReviewingService
from ..services.rewriting_service import RewritingService
from ..services.storage_service import StorageService
from ..services.vector_store import HashEmbedding, QdrantVectorStore, VectorStore
from hello_agents.tools import ToolRegistry, MCPTool
import os


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
    github_mcp_enabled: bool
    github_mcp_tool: MCPTool | None


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
        "embedding_api_base=%s embedding_api_key=%s github_mcp=%s",
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
        "enabled" if config.github_mcp_enabled else "disabled",
    )

    tool_registry: ToolRegistry | None = None
    github_mcp_tool: MCPTool | None = None
    if config.github_mcp_enabled:
        tool_registry = ToolRegistry()
        try:
            if config.github_token:
                os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = config.github_token
            github_mcp_tool = MCPTool(
                name="github",
                server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": config.github_token} if config.github_token else None,
                auto_expand=True,
            )
            expanded = github_mcp_tool.get_expanded_tools()
            tools_to_register = expanded or []
            scope = os.getenv("MCP_GITHUB_TOOL_SCOPE", "search").strip().lower()
            if tools_to_register and scope != "all":
                allow = {
                    "github_search_repositories",
                    "github_search_code",
                    "github_search_issues",
                    "github_search_users",
                    "github_get_file_contents",
                }
                filtered = [tool for tool in tools_to_register if tool.get("name") in allow]
                if filtered:
                    tools_to_register = filtered
            if tools_to_register:
                for tool in tools_to_register:
                    tool_registry.register_tool(tool)
                logger.info(
                    "GitHub MCP tool enabled with %s tools (scope=%s).",
                    len(tools_to_register),
                    scope,
                )
            else:
                tool_registry.register_tool(github_mcp_tool)
                logger.info("GitHub MCP tool enabled with 1 tool (raw).")
        except Exception as exc:
            logger.warning("Failed to initialize GitHub MCP tool: %s", exc)
            tool_registry = None
            github_mcp_tool = None

    writing_agent = WritingAgent(
        tool_registry=tool_registry,
        provider=config.llm_provider or None,
        model=config.llm_model or None,
        api_key=config.llm_api_key or None,
        base_url=config.llm_api_base or None,
        request_timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )
    reviewer_agent = ReviewerAgent(
        tool_registry=tool_registry,
        provider=config.llm_provider or None,
        model=config.llm_model or None,
        api_key=config.llm_api_key or None,
        base_url=config.llm_api_base or None,
        request_timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
    )
    editor_agent = EditorAgent(
        tool_registry=tool_registry,
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
    evidence_extractor = EvidenceExtractor(writing_agent.llm, EvidenceConfig.from_env())
    drafter = DraftingService(
        writing_agent=writing_agent,
        reviewing_service=reviewer,
        rewriting_service=rewriter,
        evidence_extractor=evidence_extractor,
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
    expander_config = QueryExpansionConfig.from_env()
    query_expander = None
    if expander_config.hyde_enabled:
        query_expander = QueryExpander(writing_agent.llm, expander_config)
        logger.info(
            "RAG query expansion enabled (HyDE=%s).",
            expander_config.hyde_enabled,
        )

    rerank_config = RerankConfig.from_env()
    reranker = None
    if rerank_config.enabled:
        reranker = Reranker(writing_agent.llm, rerank_config)
        logger.info(
            "RAG rerank enabled (top_k=%s, max_candidates=%s).",
            rerank_config.top_k,
            rerank_config.max_candidates,
        )

    rag = RAGService(
        storage=storage,
        vector_store=vector_store,
        query_expander=query_expander,
        reranker=reranker,
    )
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
        github_mcp_enabled=config.github_mcp_enabled and github_mcp_tool is not None,
        github_mcp_tool=github_mcp_tool,
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
