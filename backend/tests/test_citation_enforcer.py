import os
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.api.routes.pipeline import _strict_citation_postcheck_failed as pipeline_postcheck_failed  # noqa: E402
from app.api.routes.writing import _strict_citation_postcheck_failed as writing_postcheck_failed  # noqa: E402
from app.services.citation_enforcer import CitationEnforcer, CoverageReport  # noqa: E402
from app.services.research_service import ResearchNote  # noqa: E402


class FakeEmbedder:
    def embed_one(self, text: str):
        value = (text or "").strip()
        if "云计算通过共享资源按需提供计算能力" in value:
            return [1.0, 0.0]
        if "Cloud computing provides shared on-demand configurable resources" in value:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed(self, texts):
        return [self.embed_one(text) for text in texts]


def test_hybrid_semantic_backfill_adds_citation_label(monkeypatch):
    monkeypatch.setenv("RAG_COVERAGE_SEMANTIC_ENABLED", "true")
    monkeypatch.setenv("RAG_COVERAGE_SEMANTIC_THRESHOLD", "0.2")

    enforcer = CitationEnforcer()
    note = ResearchNote(
        doc_id="doc-1",
        title="Cloud Computing Overview",
        summary="Cloud computing provides shared on-demand configurable resources.",
        url="",
    )

    text = "云计算通过共享资源按需提供计算能力。"
    enforced, report = enforcer.enforce(
        text,
        [note],
        apply_labels=True,
        strict_labels=False,
        embedder=FakeEmbedder(),
    )

    assert "[1]" in enforced
    assert "[推断]" not in enforced
    assert report.semantic_covered_paragraphs == 1


def test_strict_postcheck_accepts_semantic_evidence_for_rag_only(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_MODE", "rag_only")
    monkeypatch.setenv("RAG_CITATION_STRICT_POSTCHECK_ENABLED", "true")
    monkeypatch.setenv("RAG_CITATION_STRICT_MIN_SEMANTIC_COVERAGE", "0.6")

    report = CoverageReport(
        coverage=1.0,
        token_coverage=0.0,
        total_tokens=32,
        covered_tokens=0,
        total_paragraphs=1,
        covered_paragraphs=0,
        semantic_coverage=1.0,
        semantic_covered_paragraphs=1,
        semantic_total_paragraphs=1,
    )

    assert pipeline_postcheck_failed(report) is False
    assert writing_postcheck_failed(report) is False


def test_strict_postcheck_still_fails_without_semantic_evidence(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_MODE", "rag_only")
    monkeypatch.setenv("RAG_CITATION_STRICT_POSTCHECK_ENABLED", "true")
    monkeypatch.setenv("RAG_CITATION_STRICT_MIN_SEMANTIC_COVERAGE", "0.6")

    report = CoverageReport(
        coverage=0.0,
        token_coverage=0.0,
        total_tokens=32,
        covered_tokens=0,
        total_paragraphs=1,
        covered_paragraphs=0,
        semantic_coverage=0.1,
        semantic_covered_paragraphs=0,
        semantic_total_paragraphs=1,
    )

    assert pipeline_postcheck_failed(report) is True
    assert writing_postcheck_failed(report) is True


def test_default_lexical_threshold_is_less_harsh_for_paraphrased_rag(monkeypatch):
    monkeypatch.delenv("RAG_COVERAGE_THRESHOLD", raising=False)

    enforcer = CitationEnforcer()
    note = ResearchNote(
        doc_id="doc-1",
        title="Cloud",
        summary="cloud",
        url="",
    )

    text = "cloud pricing security governance operations migration backup analytics"
    _, report = enforcer.enforce(
        text,
        [note],
        apply_labels=False,
        strict_labels=False,
    )

    assert report.covered_paragraphs == 1
    assert report.total_paragraphs == 1
