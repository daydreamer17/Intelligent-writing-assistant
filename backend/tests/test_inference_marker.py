from __future__ import annotations

from app.services.inference_marker import mark_inference_paragraphs


def test_mark_inference_paragraphs_adds_tag_for_unlabeled_paragraphs():
    text = "这是第一段没有标注。\n\n这是第二段也没有标注。"
    out = mark_inference_paragraphs(text, tag="[推断]", min_paragraph_chars=4)
    assert out.count("[推断]") == 2


def test_mark_inference_paragraphs_keeps_existing_citations_and_tags():
    text = "事实段落 [1]\n\n推测段落 [推断]"
    out = mark_inference_paragraphs(text, tag="[推断]", min_paragraph_chars=4)
    assert out.count("[推断]") == 1
    assert "[1]" in out


def test_mark_inference_paragraphs_skips_short_headings():
    text = "## 小标题\n\n这是正文段落。"
    out = mark_inference_paragraphs(text, tag="[推断]", min_paragraph_chars=6)
    assert "## 小标题 [推断]" not in out
    assert "这是正文段落。 [推断]" in out


def test_mark_inference_paragraphs_marks_short_plain_phrase_in_hybrid_case():
    text = "量子烹饪龙语契约与海底火星税法"
    out = mark_inference_paragraphs(text, tag="[推断]", min_paragraph_chars=12)
    assert out.endswith("[推断]")
