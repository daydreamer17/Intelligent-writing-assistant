from __future__ import annotations

from app.services.rag_query_expander import QueryExpansionConfig, QueryExpander


class _DummyLLM:
    def invoke(self, messages, **kwargs):  # noqa: ANN001
        text = messages[0]["content"]
        if "English retrieval query" in text:
            return "cloud computing course summary"
        if "简洁的中文检索查询" in text:
            return "云计算课程总结"
        return ""


def test_bilingual_rewrite_for_chinese_query():
    expander = QueryExpander(
        _DummyLLM(),
        QueryExpansionConfig(
            hyde_enabled=False,
            bilingual_rewrite_enabled=True,
            max_query_chars=800,
            max_hyde_chars=1500,
            max_hyde_tokens=300,
            max_rewrite_chars=200,
            max_rewrite_tokens=64,
            max_variants=4,
        ),
    )

    variants = expander.expand("云计算知识总结")
    sources = [item.source for item in variants]
    assert "original" in sources
    assert "rewrite_en" in sources
    assert any("cloud computing" in item.text for item in variants)


def test_bilingual_rewrite_for_english_query():
    expander = QueryExpander(
        _DummyLLM(),
        QueryExpansionConfig(
            hyde_enabled=False,
            bilingual_rewrite_enabled=True,
            max_query_chars=800,
            max_hyde_chars=1500,
            max_hyde_tokens=300,
            max_rewrite_chars=200,
            max_rewrite_tokens=64,
            max_variants=4,
        ),
    )

    variants = expander.expand("cloud computing summary")
    sources = [item.source for item in variants]
    assert "original" in sources
    assert "rewrite_zh" in sources
    assert any("云计算" in item.text for item in variants)


def test_bilingual_rewrite_can_be_disabled():
    expander = QueryExpander(
        _DummyLLM(),
        QueryExpansionConfig(
            hyde_enabled=False,
            bilingual_rewrite_enabled=False,
            max_query_chars=800,
            max_hyde_chars=1500,
            max_hyde_tokens=300,
            max_rewrite_chars=200,
            max_rewrite_tokens=64,
            max_variants=4,
        ),
    )
    variants = expander.expand("云计算知识总结")
    assert len(variants) == 1
    assert variants[0].source == "original"
