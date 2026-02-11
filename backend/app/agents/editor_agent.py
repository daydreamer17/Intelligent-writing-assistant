from __future__ import annotations

import re
from typing import Optional

from hello_agents.tools import ToolRegistry

from .base import AgentRuntimeConfig, BaseWritingAgent

DEFAULT_SYSTEM_PROMPT = (
    "You are a rewriting editor. "
    "Return ONLY the final rewritten document text. "
    "Do NOT output review comments, critique, analysis, checklists, or meta explanations. "
    "Preserve factual meaning and avoid adding unsupported facts. "
    "Keep the same language as the draft unless explicitly requested otherwise. "
    "If factual grounding is needed and tools are available, call tools first."
)


class EditorAgent(BaseWritingAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        name: str = "Editor Agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        config = AgentRuntimeConfig(
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )
        super().__init__(config, tool_registry=tool_registry)

    def rewrite(
        self,
        *,
        draft: str,
        guidance: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
    ) -> str:
        # 截断过长的内容以避免超过模型限制
        max_draft_chars = 20000  # 约 8000 tokens
        max_guidance_chars = 5000  # 约 2000 tokens

        if len(draft) > max_draft_chars:
            draft = draft[:max_draft_chars] + "\n...(内容过长已截断)"

        if len(guidance) > max_guidance_chars:
            guidance = guidance[:max_guidance_chars] + "\n...(指导意见过长已截断)"

        prompt = self._build_rewrite_prompt(
            draft=draft,
            guidance=guidance,
            style=style,
            target_length=target_length,
        )
        rewritten = self.run(
            prompt,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
        )

        # If model accidentally returns review/feedback text, force a repair pass.
        if _looks_like_review_feedback(rewritten):
            repair_prompt = self._build_repair_prompt(
                draft=draft,
                guidance=guidance,
                bad_output=rewritten,
                style=style,
                target_length=target_length,
            )
            rewritten = self.run(
                repair_prompt,
                max_tokens=max_tokens,
                max_input_chars=max_input_chars,
                session_id=session_id,
            )

        return _clean_rewrite_output(rewritten)

    def rewrite_stream(
        self,
        *,
        draft: str,
        guidance: str = "",
        style: str = "",
        target_length: str = "",
        max_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        session_id: str = "",
    ):
        # 截断过长的内容以避免超过模型限制
        max_draft_chars = 20000  # 约 8000 tokens
        max_guidance_chars = 5000  # 约 2000 tokens

        if len(draft) > max_draft_chars:
            draft = draft[:max_draft_chars] + "\n...(内容过长已截断)"

        if len(guidance) > max_guidance_chars:
            guidance = guidance[:max_guidance_chars] + "\n...(指导意见过长已截断)"

        # Reuse the exact same quality gate as non-stream path, then chunk-yield.
        rewritten = self.rewrite(
            draft=draft,
            guidance=guidance,
            style=style,
            target_length=target_length,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            session_id=session_id,
        )
        chunk_size = 480
        for start in range(0, len(rewritten), chunk_size):
            yield rewritten[start : start + chunk_size]

    def _build_rewrite_prompt(
        self,
        *,
        draft: str,
        guidance: str,
        style: str,
        target_length: str,
    ) -> str:
        instruction = (
            "Task: Rewrite the draft into a final publishable document.\n"
            "Hard Rules:\n"
            "1. Output ONLY rewritten document text.\n"
            "2. Do NOT output comments such as suggestions/issues/analysis/assessment.\n"
            "3. Keep factual content grounded in draft and guidance; do not invent facts.\n"
            "4. Keep the original language.\n"
            "5. Keep structure when possible, but optimize readability and coherence.\n"
        )
        parts = [instruction, f"[DRAFT]\n{draft}\n[/DRAFT]"]
        if guidance:
            parts.append(f"[GUIDANCE]\n{guidance}\n[/GUIDANCE]")
        if style:
            parts.append(f"[STYLE]\n{style}\n[/STYLE]")
        if target_length:
            parts.append(f"[TARGET_LENGTH]\n{target_length}\n[/TARGET_LENGTH]")
        return "\n\n".join(parts)

    def _build_repair_prompt(
        self,
        *,
        draft: str,
        guidance: str,
        bad_output: str,
        style: str,
        target_length: str,
    ) -> str:
        parts = [
            "The previous output was review feedback, not rewritten text.",
            "Rewrite the draft into final text now.",
            "Return ONLY rewritten text. No analysis, no suggestions, no bullets about issues.",
            f"[DRAFT]\n{draft}\n[/DRAFT]",
        ]
        if guidance:
            parts.append(f"[GUIDANCE]\n{guidance}\n[/GUIDANCE]")
        if style:
            parts.append(f"[STYLE]\n{style}\n[/STYLE]")
        if target_length:
            parts.append(f"[TARGET_LENGTH]\n{target_length}\n[/TARGET_LENGTH]")
        parts.append(f"[INCORRECT_OUTPUT_EXAMPLE]\n{bad_output}\n[/INCORRECT_OUTPUT_EXAMPLE]")
        return "\n\n".join(parts)


def _looks_like_review_feedback(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return False
    strong_markers = [
        "feedback by severity",
        "high severity",
        "critical issues",
        "改进方向",
        "审校建议",
        "问题",
        "建议",
        "评估",
    ]
    hit = sum(1 for m in strong_markers if m in value)
    if hit >= 2:
        return True
    heading_count = len(re.findall(r"(?m)^(#{1,4}|[-*]\s+|\d+\.)", value))
    review_word_count = len(
        re.findall(r"(建议|问题|评估|应当|需要|feedback|issue|severity|critique)", value)
    )
    return heading_count >= 4 and review_word_count >= 4


def _clean_rewrite_output(text: str) -> str:
    value = (text or "").strip()
    # Strip fenced wrappers if the model encloses the output.
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
    # Remove common "label only" prefix.
    value = re.sub(r"^(改写后文本|重写结果|final rewritten text)\s*[:：]\s*", "", value, flags=re.I)
    return value.strip()
