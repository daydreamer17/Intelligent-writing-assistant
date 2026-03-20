from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from hello_agents.tools import ToolRegistry
from pydantic import BaseModel, ValidationError

from ..agents.reviewer_agent import ReviewerAgent

logger = logging.getLogger("app.reviewing")


@dataclass(frozen=True)
class ReviewResult:
    review: str


@dataclass(frozen=True)
class ReviewDecisionResult:
    review: str
    needs_rewrite: bool


class _ReviewDecisionPayload(BaseModel):
    needs_rewrite: bool


class ReviewingService:
    def __init__(self, agent: ReviewerAgent) -> None:
        self.agent = agent

    def review(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> ReviewResult:
        review_text = self.agent.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        )
        return ReviewResult(review=review_text)

    def review_decision(
        self,
        *,
        draft: str,
        criteria: str = "",
        sources: str = "",
        audience: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> ReviewDecisionResult:
        review = self.review(
            draft=draft,
            criteria=criteria,
            sources=sources,
            audience=audience,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
            tool_profile_id=tool_profile_id,
            tool_registry_override=tool_registry_override,
        ).review
        return ReviewDecisionResult(
            review=review,
            needs_rewrite=self.decide_rewrite(
                review=review,
                criteria=criteria,
                audience=audience,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            ),
        )

    def decide_rewrite(
        self,
        *,
        review: str,
        criteria: str = "",
        audience: str = "",
        max_tokens: int | None = None,
        max_input_chars: int | None = None,
        session_id: str = "",
        tool_profile_id: str | None = None,
        tool_registry_override: ToolRegistry | None = None,
    ) -> bool:
        decision_method = getattr(self.agent, "review_decision", None)
        if callable(decision_method):
            raw = decision_method(
                review=review,
                criteria=criteria,
                audience=audience,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
                tool_profile_id=tool_profile_id,
                tool_registry_override=tool_registry_override,
            )
            try:
                payload = _parse_review_decision(raw)
                return payload.needs_rewrite
            except ValueError as exc:
                logger.warning(
                    "review_decision structured parse failed; falling back to heuristic review. error=%s",
                    exc,
                )
        return _needs_rewrite_from_review(review)


_NO_REWRITE_PATTERNS = (
    r"\bno (major )?(changes|rewrite|revision)s? needed\b",
    r"\blooks good\b",
    r"\bready to publish\b",
    r"无需(重写|改写|修改|调整)",
    r"不需要(重写|改写|修改|调整)",
    r"可直接发布",
    r"无需进一步",
)


def _needs_rewrite_from_review(review: str) -> bool:
    text = (review or "").strip()
    if not text:
        return False
    lowered = text.lower()
    for pattern in _NO_REWRITE_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return False
    return True


def _parse_review_decision(raw: Any) -> _ReviewDecisionPayload:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty review decision output")
    json_text = _extract_json_object(text)
    try:
        payload = _ReviewDecisionPayload.model_validate(json.loads(json_text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid review decision payload: {exc}") from exc
    return payload


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no json object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated json object")
