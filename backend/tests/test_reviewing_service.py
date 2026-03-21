import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.services.reviewing_service import ReviewingService  # noqa: E402


class FakeReviewerAgent:
    def __init__(self) -> None:
        self.review_calls = []
        self.review_decision_calls = []
        self.review_text = "Minor wording polish only. No major changes needed."
        self.review_decision_text = (
            '{"review_text":"Structured review","needs_rewrite":true,'
            '"reason":"citations missing","score":0.82}'
        )

    def review(self, **kwargs):
        self.review_calls.append(kwargs)
        return self.review_text

    def review_decision(self, **kwargs):
        self.review_decision_calls.append(kwargs)
        return self.review_decision_text


def test_review_decision_parses_structured_json():
    agent = FakeReviewerAgent()
    service = ReviewingService(agent)

    result = service.review_decision(
        draft="draft",
        criteria="accurate",
        sources="notes",
        audience="beginner",
    )

    assert result.review_text == "Structured review"
    assert result.needs_rewrite is True
    assert result.reason == "citations missing"
    assert result.score == 0.82
    assert len(agent.review_decision_calls) == 1
    assert len(agent.review_calls) == 1


def test_review_decision_falls_back_to_heuristic_review_when_json_is_invalid():
    agent = FakeReviewerAgent()
    agent.review_decision_text = "Not JSON at all"
    agent.review_text = "Looks good. No major changes needed."
    service = ReviewingService(agent)

    result = service.review_decision(draft="draft")

    assert result.review_text == "Looks good. No major changes needed."
    assert result.needs_rewrite is False
    assert result.reason == "fallback_heuristic"
    assert result.score is None
    assert len(agent.review_decision_calls) == 1
    assert len(agent.review_calls) == 1


def test_review_decision_accepts_fenced_json_payload():
    agent = FakeReviewerAgent()
    agent.review_decision_text = """```json
{"review_text":"Need changes","needs_rewrite":true,"reason":"unsupported claims","score":0.55}
```"""
    service = ReviewingService(agent)

    result = service.review_decision(draft="draft")

    assert result.review_text == "Need changes"
    assert result.needs_rewrite is True
    assert result.reason == "unsupported claims"
    assert result.score == 0.55


def test_review_decision_falls_back_when_score_is_out_of_range():
    agent = FakeReviewerAgent()
    agent.review_decision_text = (
        '{"review_text":"Need changes","needs_rewrite":true,'
        '"reason":"unsupported claims","score":1.4}'
    )
    service = ReviewingService(agent)

    result = service.review_decision(draft="draft")

    assert result.review_text == agent.review_text
    assert result.needs_rewrite is False
    assert result.reason == "fallback_heuristic"
    assert result.score is None


def test_review_decision_falls_back_when_required_field_is_missing():
    agent = FakeReviewerAgent()
    agent.review_decision_text = '{"needs_rewrite":true,"reason":"unsupported claims","score":0.4}'
    service = ReviewingService(agent)

    result = service.review_decision(draft="draft")

    assert result.review_text == agent.review_text
    assert result.reason == "fallback_heuristic"
    assert result.score is None


def test_review_method_still_uses_plain_review_path():
    agent = FakeReviewerAgent()
    service = ReviewingService(agent)

    result = service.review(draft="draft")

    assert result.review == agent.review_text
    assert len(agent.review_calls) == 1
    assert len(agent.review_decision_calls) == 0
