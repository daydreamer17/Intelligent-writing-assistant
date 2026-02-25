from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    override: dict[str, bool]


@dataclass(frozen=True)
class RuleCheckResult:
    rule_id: str
    description: str
    passed: bool
    expected: str
    actual: str


@dataclass(frozen=True)
class AgentBehaviorSuiteResult:
    checks: list[RuleCheckResult]
    original_mode: str
    original_creative_mcp_enabled: bool
    restored_mode: str
    restored_creative_mcp_enabled: bool
    sample_outputs: dict[str, str]


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float


TARGET_REPORT_KS = (1, 3, 5)
DETAIL_REPORT_K = 5


def _default_eval_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "evals" / "retrieval_eval_small.json").resolve()


def _default_out_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "evals" / "baseline_report.md").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval baselines via /api/rag/evaluate")
    parser.add_argument("--eval", dest="eval_path", default=str(_default_eval_path()), help="Eval set JSON path")
    parser.add_argument("--out", dest="out_path", default=str(_default_out_path()), help="Markdown report output path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument(
        "--include-bilingual-baselines",
        action="store_true",
        help="Append bilingual rewrite baselines (D/E)",
    )
    parser.add_argument(
        "--skip-agent-regression",
        action="store_true",
        help="Skip Agent behavior regression mini-suite (mode switch/refusal/citation/inference-mark checks)",
    )
    parser.add_argument(
        "--inference-tag",
        default="[推断]",
        help="Expected inference marker tag in hybrid mode (default: [推断])",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP request timeout (seconds)")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each baseline N times and report mean/std (default: 1)",
    )
    return parser.parse_args()


def load_eval_set(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"评测集文件不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"评测集 JSON 解析失败: {path} (line={exc.lineno}, col={exc.colno})") from exc

    if not isinstance(data, dict):
        raise ValueError("评测集根对象必须是 JSON object")

    k_values = data.get("k_values")
    cases = data.get("cases")
    if not isinstance(k_values, list) or not all(isinstance(k, int) and k > 0 for k in k_values):
        raise ValueError("评测集字段 k_values 必须是正整数数组，例如 [1,3,5]")
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集字段 cases 必须是非空数组")

    for idx, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{idx}] 必须是对象")
        query = case.get("query")
        rel = case.get("relevant_doc_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"cases[{idx}].query 必须是非空字符串")
        if not isinstance(rel, list) or not all(isinstance(x, str) for x in rel):
            raise ValueError(f"cases[{idx}].relevant_doc_ids 必须是字符串数组")
        tags = case.get("tags")
        if tags is not None and (
            not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags)
        ):
            raise ValueError(f"cases[{idx}].tags 必须是字符串数组（可选）")
    return data


def baseline_specs(include_bilingual: bool) -> list[BaselineSpec]:
    specs = [
        BaselineSpec(
            name="dense_only",
            override={
                "rerank_enabled": False,
                "hyde_enabled": False,
                "bilingual_rewrite_enabled": False,
            },
        ),
        BaselineSpec(
            name="dense_rerank",
            override={
                "rerank_enabled": True,
                "hyde_enabled": False,
                "bilingual_rewrite_enabled": False,
            },
        ),
        BaselineSpec(
            name="dense_hyde_rerank",
            override={
                "rerank_enabled": True,
                "hyde_enabled": True,
                "bilingual_rewrite_enabled": False,
            },
        ),
    ]
    if include_bilingual:
        specs.extend(
            [
                BaselineSpec(
                    name="dense_rerank_bilingual",
                    override={
                        "rerank_enabled": True,
                        "hyde_enabled": False,
                        "bilingual_rewrite_enabled": True,
                    },
                ),
                BaselineSpec(
                    name="dense_hyde_rerank_bilingual",
                    override={
                        "rerank_enabled": True,
                        "hyde_enabled": True,
                        "bilingual_rewrite_enabled": True,
                    },
                ),
            ]
        )
    return specs


def call_eval_api(
    client: httpx.Client,
    *,
    base_url: str,
    eval_payload: dict[str, Any],
    baseline: BaselineSpec,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/rag/evaluate"
    payload = {
        "k_values": eval_payload["k_values"],
        "cases": eval_payload["cases"],
        "rag_config_override": baseline.override,
    }
    try:
        resp = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"请求超时: {url} | {exc}。当前 baseline 可能触发 Rerank/HyDE 的多次 LLM 调用，请增大 --timeout（例如 180 或 300）"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"请求失败（后端未启动或网络错误）: {url} | {exc}") from exc

    if resp.status_code != 200:
        body = resp.text
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        raise RuntimeError(
            f"/api/rag/evaluate 返回非 200: status={resp.status_code}\nResponse body:\n{body}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError("后端响应不是合法 JSON") from exc

    if not isinstance(data, dict):
        raise RuntimeError("后端响应 JSON 结构错误（预期 object）")
    if "macro_metrics" not in data or not isinstance(data["macro_metrics"], list):
        raise RuntimeError("后端响应缺少 macro_metrics 数组")
    return data


def _call_json_api(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        resp = client.request(method.upper(), url, json=json_payload)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"请求超时: {url} | {exc}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"请求失败: {url} | {exc}") from exc
    if resp.status_code != 200:
        body = resp.text
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        raise RuntimeError(f"{url} 返回非 200: status={resp.status_code}\nResponse body:\n{body}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"{url} 响应不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} 响应 JSON 结构错误（预期 object）")
    return data


def _add_rule(
    checks: list[RuleCheckResult],
    *,
    rule_id: str,
    description: str,
    passed: bool,
    expected: str,
    actual: Any,
) -> None:
    checks.append(
        RuleCheckResult(
            rule_id=rule_id,
            description=description,
            passed=bool(passed),
            expected=expected,
            actual=_safe_text(str(actual))[:400],
        )
    )


def _pick_evidence_query(eval_payload: dict[str, Any]) -> str:
    preferred_tag_orders = [
        ("title_like",),
        ("semantic_like",),
    ]
    cases = [case for case in (eval_payload.get("cases", []) or []) if isinstance(case, dict)]
    for tags_needed in preferred_tag_orders:
        for case in cases:
            tags = set(_case_tags(case))
            if not tags.issuperset(tags_needed):
                continue
            query = str(case.get("query", "")).strip()
            if query:
                return query
    for case in eval_payload.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        query = str(case.get("query", "")).strip()
        if query:
            return query
    return "cloud computing"


def _rewrite_payload(
    *,
    draft: str,
    guidance: str,
    style: str,
    target_length: str,
    session_prefix: str,
) -> dict[str, Any]:
    return {
        "draft": draft,
        "guidance": guidance,
        "style": style,
        "target_length": target_length,
        "session_id": f"{session_prefix}-{uuid4().hex[:8]}",
    }


def _run_rewrite_case(
    client: httpx.Client,
    *,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return _call_json_api(
        client,
        method="POST",
        url=f"{base_url.rstrip('/')}/api/rewrite",
        json_payload=payload,
    )


def run_agent_behavior_regression_suite(
    client: httpx.Client,
    *,
    base_url: str,
    eval_payload: dict[str, Any],
    inference_tag: str,
) -> AgentBehaviorSuiteResult:
    checks: list[RuleCheckResult] = []
    samples: dict[str, str] = {}
    settings_url = f"{base_url.rstrip('/')}/api/settings/generation-mode"
    refusal_message = "在提供的文档中，无法找到该问题的答案。"
    evidence_query = _pick_evidence_query(eval_payload)
    unrelated_query = "量子烹饪龙语契约与海底火星税法"

    original = _call_json_api(client, method="GET", url=settings_url)
    original_mode = str(original.get("mode", ""))
    original_creative_mcp_enabled = bool(original.get("creative_mcp_enabled", False))

    def set_mode(mode: str, creative_mcp_enabled: bool | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode}
        if creative_mcp_enabled is not None:
            payload["creative_mcp_enabled"] = creative_mcp_enabled
        return _call_json_api(client, method="POST", url=settings_url, json_payload=payload)

    restored_mode = original_mode
    restored_creative_mcp_enabled = original_creative_mcp_enabled
    try:
        # rag_only: settings flags
        resp = set_mode("rag_only")
        _add_rule(checks, rule_id="MODE_RAG_ONLY_SET", description="切换到 rag_only 成功", passed=resp.get("mode") == "rag_only", expected="mode=rag_only", actual=resp.get("mode"))
        _add_rule(
            checks,
            rule_id="MODE_RAG_ONLY_FLAGS",
            description="rag_only 模式标志正确（强制引用开 / MCP关 / 推断标注关）",
            passed=(
                bool(resp.get("citation_enforce")) is True
                and bool(resp.get("mcp_allowed")) is False
                and bool(resp.get("inference_mark_required")) is False
            ),
            expected="citation_enforce=true, mcp_allowed=false, inference_mark_required=false",
            actual=(
                f"citation_enforce={resp.get('citation_enforce')}, "
                f"mcp_allowed={resp.get('mcp_allowed')}, "
                f"inference_mark_required={resp.get('inference_mark_required')}"
            ),
        )

        # rag_only: insufficient evidence => refuse
        rag_only_refuse = _run_rewrite_case(
            client,
            base_url=base_url,
            payload=_rewrite_payload(
                draft=f"请说明：{unrelated_query}",
                guidance="仅基于知识库文档回答，不允许虚构。",
                style="简洁",
                target_length="120",
                session_prefix="eval-rag-only-refuse",
            ),
        )
        refuse_text = str(rag_only_refuse.get("revised", ""))
        samples["rag_only_refuse"] = refuse_text[:240]
        _add_rule(checks, rule_id="RAG_ONLY_REFUSE_MESSAGE", description="rag_only 证据不足时拒答", passed=refuse_text.strip() == refusal_message, expected=refusal_message, actual=refuse_text)
        _add_rule(
            checks,
            rule_id="RAG_ONLY_REFUSE_EMPTY_CITATION_AND_ZERO_COVERAGE",
            description="rag_only 拒答时无引用且 coverage=0",
            passed=(
                len(rag_only_refuse.get("citations", []) or []) == 0
                and float(rag_only_refuse.get("coverage", 0.0) or 0.0) == 0.0
            ),
            expected="citations=[], coverage=0.0",
            actual=(
                f"citations={len(rag_only_refuse.get('citations', []) or [])}, "
                f"coverage={rag_only_refuse.get('coverage')}"
            ),
        )

        # rag_only: evidence-rich => citations and labels
        rag_only_evidence = _run_rewrite_case(
            client,
            base_url=base_url,
            payload=_rewrite_payload(
                draft=f"请简要说明以下主题的核心内容：{evidence_query}",
                guidance=f"只基于知识库文档改写，保留关键术语：{evidence_query}",
                style="学术、简洁",
                target_length="180",
                session_prefix="eval-rag-only-evidence",
            ),
        )
        evidence_text = str(rag_only_evidence.get("revised", ""))
        samples["rag_only_evidence"] = evidence_text[:240]
        citations = rag_only_evidence.get("citations", []) or []
        has_numeric_label = bool(re.search(r"\[\d+\]", evidence_text))
        _add_rule(
            checks,
            rule_id="RAG_ONLY_EVIDENCE_NOT_REFUSE_AND_CITATIONS_EXIST",
            description="rag_only 有证据时不拒答且返回引用列表",
            passed=evidence_text.strip() != refusal_message and len(citations) > 0,
            expected="revised != refusal_message AND len(citations)>0",
            actual=f"revised_is_refusal={evidence_text.strip() == refusal_message}, citations={len(citations)}",
        )
        _add_rule(checks, rule_id="RAG_ONLY_EVIDENCE_LABEL_MARKED", description="rag_only 有证据时正文存在 [n] 或 citation_enforced=true", passed=has_numeric_label or bool(rag_only_evidence.get("citation_enforced")), expected="[n] in revised OR citation_enforced=true", actual=f"has_[n]={has_numeric_label}, citation_enforced={rag_only_evidence.get('citation_enforced')}")

        # hybrid: mode + inference marker
        resp = set_mode("hybrid")
        _add_rule(checks, rule_id="MODE_HYBRID_SET", description="切换到 hybrid 成功", passed=resp.get("mode") == "hybrid", expected="mode=hybrid", actual=resp.get("mode"))
        _add_rule(
            checks,
            rule_id="MODE_HYBRID_FLAGS",
            description="hybrid 模式标志正确（强制引用关 / 推断标注开）",
            passed=(
                bool(resp.get("citation_enforce")) is False
                and bool(resp.get("inference_mark_required")) is True
            ),
            expected="citation_enforce=false, inference_mark_required=true",
            actual=(
                f"citation_enforce={resp.get('citation_enforce')}, "
                f"inference_mark_required={resp.get('inference_mark_required')}"
            ),
        )

        hybrid_unrelated = _run_rewrite_case(
            client,
            base_url=base_url,
            payload=_rewrite_payload(
                draft=f"请说明：{unrelated_query}",
                guidance="可结合常识补全，但请明确不确定信息。",
                style="简洁",
                target_length="120",
                session_prefix="eval-hybrid-unrelated",
            ),
        )
        hybrid_text = str(hybrid_unrelated.get("revised", ""))
        samples["hybrid_unrelated"] = hybrid_text[:240]
        _add_rule(checks, rule_id="HYBRID_UNRELATED_NOT_REFUSE", description="hybrid 证据不足时不拒答", passed=hybrid_text.strip() != refusal_message, expected="revised != refusal_message", actual=hybrid_text[:120])
        _add_rule(checks, rule_id="HYBRID_UNRELATED_INFERENCE_TAG", description="hybrid 证据不足时包含推断标注", passed=inference_tag in hybrid_text, expected=f"revised contains {inference_tag}", actual=hybrid_text[:180])

        # creative: mode + creative_mcp toggle + no inference tag
        resp = set_mode("creative", creative_mcp_enabled=False)
        _add_rule(checks, rule_id="MODE_CREATIVE_SET", description="切换到 creative 成功", passed=resp.get("mode") == "creative", expected="mode=creative", actual=resp.get("mode"))
        _add_rule(
            checks,
            rule_id="MODE_CREATIVE_FLAGS",
            description="creative 模式标志正确（强制引用关 / 推断标注关）",
            passed=(
                bool(resp.get("citation_enforce")) is False
                and bool(resp.get("inference_mark_required")) is False
            ),
            expected="citation_enforce=false, inference_mark_required=false",
            actual=(
                f"citation_enforce={resp.get('citation_enforce')}, "
                f"inference_mark_required={resp.get('inference_mark_required')}"
            ),
        )
        _add_rule(
            checks,
            rule_id="MODE_CREATIVE_MCP_OFF_EFFECTIVE",
            description="creative_mcp_enabled=false 生效",
            passed=(
                bool(resp.get("creative_mcp_enabled")) is False
                and bool(resp.get("mcp_allowed")) is False
            ),
            expected="creative_mcp_enabled=false and mcp_allowed=false",
            actual=(
                f"creative_mcp_enabled={resp.get('creative_mcp_enabled')}, "
                f"mcp_allowed={resp.get('mcp_allowed')}"
            ),
        )

        creative_unrelated = _run_rewrite_case(
            client,
            base_url=base_url,
            payload=_rewrite_payload(
                draft=f"请写一个设定说明：{unrelated_query}",
                guidance="自由发挥，保持简洁，不需要引用。",
                style="创意",
                target_length="120",
                session_prefix="eval-creative-unrelated",
            ),
        )
        creative_text = str(creative_unrelated.get("revised", ""))
        samples["creative_unrelated"] = creative_text[:240]
        _add_rule(checks, rule_id="CREATIVE_UNRELATED_NOT_REFUSE", description="creative 不触发拒答", passed=creative_text.strip() != refusal_message, expected="revised != refusal_message", actual=creative_text[:120])
        _add_rule(checks, rule_id="CREATIVE_UNRELATED_NO_INFERENCE_TAG", description="creative 不自动追加 [推断]", passed=inference_tag not in creative_text, expected=f"revised not contains {inference_tag}", actual=creative_text[:180])

        resp = set_mode("creative", creative_mcp_enabled=True)
        _add_rule(
            checks,
            rule_id="MODE_CREATIVE_MCP_ON_EFFECTIVE",
            description="creative_mcp_enabled=true 生效",
            passed=(
                bool(resp.get("creative_mcp_enabled")) is True
                and bool(resp.get("mcp_allowed")) is True
            ),
            expected="creative_mcp_enabled=true and mcp_allowed=true",
            actual=(
                f"creative_mcp_enabled={resp.get('creative_mcp_enabled')}, "
                f"mcp_allowed={resp.get('mcp_allowed')}"
            ),
        )

    except Exception as exc:
        _add_rule(
            checks,
            rule_id="AGENT_REGRESSION_FATAL",
            description="Agent 行为回归小套件执行过程无未处理异常",
            passed=False,
            expected="no exception",
            actual=repr(exc),
        )
    finally:
        try:
            restored = set_mode(original_mode or "rag_only", creative_mcp_enabled=original_creative_mcp_enabled)
            restored_mode = str(restored.get("mode", ""))
            restored_creative_mcp_enabled = bool(restored.get("creative_mcp_enabled", False))
        except Exception as exc:
            _add_rule(
                checks,
                rule_id="MODE_RESTORE",
                description="回归套件结束后恢复原始生成模式",
                passed=False,
                expected=f"mode={original_mode}, creative_mcp_enabled={original_creative_mcp_enabled}",
                actual=repr(exc),
            )
        else:
            _add_rule(
                checks,
                rule_id="MODE_RESTORE",
                description="回归套件结束后恢复原始生成模式",
                passed=(
                    restored_mode == (original_mode or "rag_only")
                    and restored_creative_mcp_enabled is bool(original_creative_mcp_enabled)
                ),
                expected=f"mode={original_mode}, creative_mcp_enabled={original_creative_mcp_enabled}",
                actual=f"mode={restored_mode}, creative_mcp_enabled={restored_creative_mcp_enabled}",
            )

    return AgentBehaviorSuiteResult(
        checks=checks,
        original_mode=original_mode,
        original_creative_mcp_enabled=original_creative_mcp_enabled,
        restored_mode=restored_mode,
        restored_creative_mcp_enabled=restored_creative_mcp_enabled,
        sample_outputs=samples,
    )


def metric_at_k(report_json: dict[str, Any], k: int) -> dict[str, float]:
    macro = report_json.get("macro_metrics", [])
    for row in macro:
        if isinstance(row, dict) and int(row.get("k", 0)) == k:
            return {
                "recall": float(row.get("recall", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "hit_rate": float(row.get("hit_rate", 0.0)),
                "mrr": float(row.get("mrr", 0.0)),
                "ndcg": float(row.get("ndcg", 0.0)),
            }
    raise RuntimeError(f"macro_metrics 中未找到 k={k} 的指标（请确认 eval set 的 k_values 包含 {k}）")


def metrics_for_ks(report_json: dict[str, Any], ks: list[int]) -> dict[int, dict[str, float]]:
    return {k: metric_at_k(report_json, k) for k in ks}


def _mean_std(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(mean=0.0, std=0.0)
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return MetricStats(mean=mean, std=0.0)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return MetricStats(mean=mean, std=sqrt(var))


def aggregate_metric_runs(
    metric_runs: list[dict[int, dict[str, float]]],
    *,
    ks: list[int],
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, MetricStats]]]:
    mean_rows: dict[int, dict[str, float]] = {}
    stat_rows: dict[int, dict[str, MetricStats]] = {}
    metric_names = ("recall", "precision", "hit_rate", "mrr", "ndcg")
    for k in ks:
        mean_rows[k] = {}
        stat_rows[k] = {}
        for metric_name in metric_names:
            values = [float(run[k][metric_name]) for run in metric_runs if k in run and metric_name in run[k]]
            stats = _mean_std(values)
            mean_rows[k][metric_name] = stats.mean
            stat_rows[k][metric_name] = stats
    return mean_rows, stat_rows


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _details_out_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_details{out_path.suffix or '.md'}")


def _safe_text(value: str) -> str:
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def _first_hit_rank(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> int | None:
    for idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return idx
    return None


def _per_query_index(report_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in report_json.get("per_query", []) or []:
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        key = query_id or query
        if not key:
            continue
        metrics_by_k: dict[int, dict[str, float]] = {}
        for metric in row.get("metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            k = int(metric.get("k", 0))
            metrics_by_k[k] = {
                "recall": float(metric.get("recall", 0.0)),
                "precision": float(metric.get("precision", 0.0)),
                "hit_rate": float(metric.get("hit_rate", 0.0)),
                "mrr": float(metric.get("mrr", 0.0)),
                "ndcg": float(metric.get("ndcg", 0.0)),
            }
        out[key] = {
            "query_id": query_id,
            "query": query,
            "retrieved_doc_ids": [str(x) for x in (row.get("retrieved_doc_ids", []) or [])],
            "metrics_by_k": metrics_by_k,
        }
    return out


def _case_key(case: dict[str, Any]) -> str:
    query_id = str(case.get("query_id", "")).strip()
    query = str(case.get("query", "")).strip()
    return query_id or query


def _case_tags(case: dict[str, Any]) -> list[str]:
    raw = case.get("tags")
    if not isinstance(raw, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        tags.append(text)
    return tags


def _aggregate_metrics_from_per_query(
    per_query_rows: list[dict[str, Any]],
    *,
    report_ks: list[int],
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    if not per_query_rows:
        for k in report_ks:
            result[k] = {
                "recall": 0.0,
                "precision": 0.0,
                "hit_rate": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
            }
        return result

    for k in report_ks:
        rows_with_k: list[dict[str, float]] = []
        for row in per_query_rows:
            metrics_by_k = row.get("metrics_by_k", {})
            metric = metrics_by_k.get(k)
            if metric:
                rows_with_k.append(metric)
        if not rows_with_k:
            result[k] = {
                "recall": 0.0,
                "precision": 0.0,
                "hit_rate": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
            }
            continue
        n = len(rows_with_k)
        result[k] = {
            "recall": sum(m["recall"] for m in rows_with_k) / n,
            "precision": sum(m["precision"] for m in rows_with_k) / n,
            "hit_rate": sum(m["hit_rate"] for m in rows_with_k) / n,
            "mrr": sum(m["mrr"] for m in rows_with_k) / n,
            "ndcg": sum(m["ndcg"] for m in rows_with_k) / n,
        }
    return result


def build_tag_grouped_report_markdown(
    *,
    eval_payload: dict[str, Any],
    detailed_rows: list[tuple[BaselineSpec, dict[str, Any]]],
    report_ks: list[int],
) -> str:
    tag_to_case_keys: dict[str, list[str]] = {}
    tag_to_case_ids: dict[str, list[str]] = {}
    for case in eval_payload.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        key = _case_key(case)
        if not key:
            continue
        case_id = str(case.get("query_id", "")).strip() or key
        for tag in _case_tags(case):
            tag_to_case_keys.setdefault(tag, []).append(key)
            tag_to_case_ids.setdefault(tag, []).append(case_id)

    if not tag_to_case_keys:
        return ""

    per_baseline_index = {spec.name: _per_query_index(resp) for spec, resp in detailed_rows}

    lines = ["## Tag-Grouped Baseline Metrics", ""]
    for tag in sorted(tag_to_case_keys.keys()):
        case_keys = tag_to_case_keys[tag]
        case_ids = tag_to_case_ids.get(tag, [])
        lines.extend(
            [
                f"### Tag: `{tag}` (n={len(case_keys)})",
                "",
                f"- Cases: `{', '.join(case_ids)}`",
            ]
        )
        for k in report_ks:
            lines.extend(
                [
                    "",
                    f"#### @{k}",
                    "",
                    f"| Baseline | Recall@{k} | Precision@{k} | HitRate@{k} | MRR@{k} | nDCG@{k} |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for spec, _resp in detailed_rows:
                indexed = per_baseline_index.get(spec.name, {})
                selected_rows = [indexed[key] for key in case_keys if key in indexed]
                aggregated = _aggregate_metrics_from_per_query(selected_rows, report_ks=[k])[k]
                lines.append(
                    "| {name} | {recall} | {precision} | {hit_rate} | {mrr:.3f} | {ndcg:.3f} |".format(
                        name=spec.name,
                        recall=pct(aggregated["recall"]),
                        precision=pct(aggregated["precision"]),
                        hit_rate=pct(aggregated["hit_rate"]),
                        mrr=aggregated["mrr"],
                        ndcg=aggregated["ndcg"],
                    )
                )
        lines.append("")
    return "\n".join(lines)


def build_report_markdown(
    *,
    eval_path: Path,
    base_url: str,
    report_ks: list[int],
    rows: list[tuple[BaselineSpec, dict[int, dict[str, float]]]],
    tag_grouped_section: str = "",
    agent_regression_section: str = "",
    repeat_stats_section: str = "",
) -> str:
    lines = [
        "# Retrieval Baseline Report",
        "",
        f"Eval set: `{eval_path.as_posix()}`",
        f"Base URL: `{base_url}`",
    ]
    for k in report_ks:
        lines.extend(
            [
                "",
                f"## Metrics @K={k}",
                "",
                f"| Baseline | Recall@{k} | Precision@{k} | HitRate@{k} | MRR@{k} | nDCG@{k} |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for spec, by_k in rows:
            metrics = by_k[k]
            lines.append(
                "| {name} | {recall} | {precision} | {hit_rate} | {mrr:.3f} | {ndcg:.3f} |".format(
                    name=spec.name,
                    recall=pct(metrics["recall"]),
                    precision=pct(metrics["precision"]),
                    hit_rate=pct(metrics["hit_rate"]),
                    mrr=metrics["mrr"],
                    ndcg=metrics["ndcg"],
                )
            )
    lines.extend(
        [
            "",
            repeat_stats_section.strip(),
            "" if repeat_stats_section.strip() else "",
            tag_grouped_section.strip(),
            "" if tag_grouped_section.strip() else "",
            agent_regression_section.strip(),
            "" if agent_regression_section.strip() else "",
            "## Baselines",
            "",
            "- `dense_only`: rerank=off, hyde=off, bilingual_rewrite=off",
            "- `dense_rerank`: rerank=on, hyde=off, bilingual_rewrite=off",
            "- `dense_hyde_rerank`: rerank=on, hyde=on, bilingual_rewrite=off",
            "- `*_bilingual`（可选）: bilingual_rewrite=on",
            "",
            "> 说明：baseline 标签表示本次 `/api/rag/evaluate` 请求里的 `rag_config_override` 组合，不会修改后端默认配置。",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def build_repeat_stats_markdown(
    *,
    repeats: int,
    report_ks: list[int],
    rows: list[tuple[BaselineSpec, dict[int, dict[str, MetricStats]]]],
) -> str:
    if repeats <= 1 or not rows:
        return ""
    lines = [
        "## Repeated Runs Statistics (Mean ± Std)",
        "",
        f"- Repeats per baseline: `{repeats}`",
        "- 说明：上方主表展示均值；本节给出均值 ± 标准差（同一评测集重复运行，反映 LLM/HyDE/rerank 非确定性）。",
    ]
    for k in report_ks:
        lines.extend(
            [
                "",
                f"### @K={k}",
                "",
                f"| Baseline | Recall@{k} | Precision@{k} | HitRate@{k} | MRR@{k} | nDCG@{k} |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for spec, by_k in rows:
            stats = by_k[k]
            lines.append(
                "| {name} | {recall} | {precision} | {hit_rate} | {mrr} | {ndcg} |".format(
                    name=spec.name,
                    recall=f"{pct(stats['recall'].mean)} ± {pct(stats['recall'].std)}",
                    precision=f"{pct(stats['precision'].mean)} ± {pct(stats['precision'].std)}",
                    hit_rate=f"{pct(stats['hit_rate'].mean)} ± {pct(stats['hit_rate'].std)}",
                    mrr=f"{stats['mrr'].mean:.3f} ± {stats['mrr'].std:.3f}",
                    ndcg=f"{stats['ndcg'].mean:.3f} ± {stats['ndcg'].std:.3f}",
                )
            )
    lines.append("")
    return "\n".join(lines)


def build_agent_behavior_regression_markdown(
    suite: AgentBehaviorSuiteResult | None,
    *,
    inference_tag: str,
) -> str:
    if suite is None:
        return ""
    total = len(suite.checks)
    passed = sum(1 for c in suite.checks if c.passed)
    failed = total - passed
    lines = [
        "## Agent 行为回归小套件（规则判定）",
        "",
        "- 范围：引用存在性、证据不足时拒答/标注、生成模式切换是否生效",
        f"- 规则数：{total}（通过 {passed} / 失败 {failed}）",
        f"- Hybrid 推断标注匹配：`{inference_tag}`",
        f"- 模式恢复：`{suite.original_mode}` / creative_mcp={suite.original_creative_mcp_enabled} -> `"
        f"{suite.restored_mode}` / creative_mcp={suite.restored_creative_mcp_enabled}",
        "",
        "| Result | Rule ID | Description | Expected | Actual |",
        "|---|---|---|---|---|",
    ]
    for check in suite.checks:
        lines.append(
            "| {result} | `{rule_id}` | {desc} | `{expected}` | `{actual}` |".format(
                result="PASS" if check.passed else "FAIL",
                rule_id=check.rule_id,
                desc=check.description.replace("|", "\\|"),
                expected=(check.expected or "").replace("|", "\\|"),
                actual=(check.actual or "").replace("|", "\\|"),
            )
        )
    if suite.sample_outputs:
        lines.extend(["", "### Sample Outputs (truncated)", ""])
        for name, text in suite.sample_outputs.items():
            safe = (text or "").replace("\r", " ").strip()
            if len(safe) > 240:
                safe = safe[:240].rstrip() + "..."
            safe = safe.replace("`", "'")
            lines.append(f"- `{name}`: `{safe}`")
    lines.append("")
    return "\n".join(lines)


def build_details_report_markdown(
    *,
    eval_path: Path,
    base_url: str,
    detail_k: int,
    eval_payload: dict[str, Any],
    detailed_rows: list[tuple[BaselineSpec, dict[str, Any]]],
) -> str:
    case_list = eval_payload.get("cases", []) or []
    case_index: dict[str, dict[str, Any]] = {}
    case_order: list[str] = []
    for case in case_list:
        if not isinstance(case, dict):
            continue
        query_id = str(case.get("query_id", "")).strip()
        query = str(case.get("query", "")).strip()
        key = query_id or query
        if not key:
            continue
        case_index[key] = case
        case_order.append(key)

    per_baseline: dict[str, dict[str, Any]] = {}
    for spec, resp_json in detailed_rows:
        per_baseline[spec.name] = _per_query_index(resp_json)

    lines = [
        "# Retrieval Baseline Detailed Report",
        "",
        f"Eval set: `{eval_path.as_posix()}`",
        f"Base URL: `{base_url}`",
        f"Detail K: `{detail_k}`",
        "",
        "## Failure Samples",
        "",
        f"判定规则：在 top{detail_k} 内未命中任一 `relevant_doc_ids` 视为失败。",
        "",
    ]

    for spec, _ in detailed_rows:
        lines.extend([f"### {spec.name}", ""])
        failed = 0
        for key in case_order:
            case = case_index[key]
            row = per_baseline.get(spec.name, {}).get(key)
            relevant_ids = {str(x) for x in (case.get("relevant_doc_ids", []) or [])}
            if not row:
                failed += 1
                lines.append(f"- `{case.get('query_id') or key}`: 无 per_query 结果（接口返回缺失）")
                continue
            retrieved_ids = row.get("retrieved_doc_ids", [])
            rank = _first_hit_rank(retrieved_ids, relevant_ids, k=detail_k)
            if rank is not None:
                continue
            failed += 1
            preview = _safe_text(str(case.get("query", "")))[:120]
            lines.append(
                f"- `{case.get('query_id') or key}` 失败 | query=`{preview}` | relevant={list(relevant_ids)} | top{detail_k}={retrieved_ids[:detail_k]}"
            )
        if failed == 0:
            lines.append("- 无失败样本。")
        lines.append("")

    lines.extend(["## Per-Query Comparison", ""])

    for key in case_order:
        case = case_index[key]
        qid = str(case.get("query_id", "")).strip() or key
        query = _safe_text(str(case.get("query", "")))
        relevant_ids = [str(x) for x in (case.get("relevant_doc_ids", []) or [])]
        lines.extend(
            [
                f"### {qid}",
                "",
                f"- Query: `{query}`",
                f"- Relevant Doc IDs: `{', '.join(relevant_ids)}`",
                "",
                "| Baseline | Hit@1 | Hit@3 | Hit@5 | FirstHitRank@5 | Top5 Retrieved Doc IDs |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for spec, _ in detailed_rows:
            row = per_baseline.get(spec.name, {}).get(key)
            if not row:
                lines.append(f"| {spec.name} | N/A | N/A | N/A | N/A | (missing per_query row) |")
                continue
            metrics_by_k = row.get("metrics_by_k", {})
            m1 = metrics_by_k.get(1, {})
            m3 = metrics_by_k.get(3, {})
            m5 = metrics_by_k.get(5, {})
            retrieved_ids = row.get("retrieved_doc_ids", [])
            first_hit = _first_hit_rank(retrieved_ids, set(relevant_ids), k=detail_k)
            top5_text = ", ".join(retrieved_ids[:detail_k])
            top5_text = top5_text.replace("|", "\\|")
            lines.append(
                "| {name} | {hit1} | {hit3} | {hit5} | {rank} | {top5} |".format(
                    name=spec.name,
                    hit1="Y" if float(m1.get("hit_rate", 0.0)) > 0 else "N",
                    hit3="Y" if float(m3.get("hit_rate", 0.0)) > 0 else "N",
                    hit5="Y" if float(m5.get("hit_rate", 0.0)) > 0 else "N",
                    rank=first_hit if first_hit is not None else "-",
                    top5=top5_text,
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    eval_path = Path(args.eval_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()

    try:
        eval_payload = load_eval_set(eval_path)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if 5 not in eval_payload.get("k_values", []):
        print("[ERROR] 评测集 k_values 必须至少包含 5", file=sys.stderr)
        return 2
    missing_report_ks = [k for k in TARGET_REPORT_KS if k not in eval_payload.get("k_values", [])]
    if missing_report_ks:
        print(
            f"[ERROR] 评测集 k_values 缺少报告所需 K 值: {missing_report_ks}（脚本当前固定展示 @1/@3/@5）",
            file=sys.stderr,
        )
        return 2

    todo_count = 0
    for case in eval_payload.get("cases", []):
        todo_count += sum(1 for doc_id in case.get("relevant_doc_ids", []) if str(doc_id).startswith("TODO_"))
    if todo_count:
        print(f"[WARN] 检测到 {todo_count} 个 TODO_DOC_ID 占位符，结果仅用于流程校验，不代表真实检索质量。")

    specs = baseline_specs(args.include_bilingual_baselines)
    repeat_count = max(1, int(args.repeats or 1))
    rows: list[tuple[BaselineSpec, dict[int, dict[str, float]]]] = []
    repeat_stat_rows: list[tuple[BaselineSpec, dict[int, dict[str, MetricStats]]]] = []
    detailed_rows: list[tuple[BaselineSpec, dict[str, Any]]] = []
    agent_suite: AgentBehaviorSuiteResult | None = None

    # Disable proxy env inheritance for localhost calls. In some environments,
    # HTTP_PROXY/HTTPS_PROXY causes httpx to proxy 127.0.0.1 and return 502.
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        for spec in specs:
            print(f"[INFO] Running baseline: {spec.name} | override={spec.override}")
            run_metric_rows: list[dict[int, dict[str, float]]] = []
            run_resp_jsons: list[dict[str, Any]] = []
            for run_idx in range(repeat_count):
                if repeat_count > 1:
                    print(f"[INFO]   repeat {run_idx + 1}/{repeat_count}")
                try:
                    resp_json = call_eval_api(
                        client,
                        base_url=args.base_url,
                        eval_payload=eval_payload,
                        baseline=spec,
                    )
                    metrics = metrics_for_ks(resp_json, list(TARGET_REPORT_KS))
                except RuntimeError as exc:
                    print(f"[ERROR] baseline={spec.name} failed: {exc}", file=sys.stderr)
                    return 3
                run_metric_rows.append(metrics)
                run_resp_jsons.append(resp_json)

            mean_metrics, stat_metrics = aggregate_metric_runs(
                run_metric_rows,
                ks=list(TARGET_REPORT_KS),
            )
            rows.append((spec, mean_metrics))
            repeat_stat_rows.append((spec, stat_metrics))
            # Keep the last run for per-query/details report. Main report now shows means.
            detailed_rows.append((spec, run_resp_jsons[-1]))

        if not args.skip_agent_regression:
            print("[INFO] Running Agent behavior regression mini-suite (mode switch / refusal / citations / inference tag)")
            try:
                t0 = time.perf_counter()
                agent_suite = run_agent_behavior_regression_suite(
                    client,
                    base_url=args.base_url,
                    eval_payload=eval_payload,
                    inference_tag=args.inference_tag,
                )
                elapsed = time.perf_counter() - t0
                passed = sum(1 for item in agent_suite.checks if item.passed)
                print(
                    f"[INFO] Agent behavior regression done: {passed}/{len(agent_suite.checks)} passed in {elapsed:.1f}s"
                )
            except Exception as exc:
                print(f"[WARN] Agent behavior regression suite failed to run: {exc}", file=sys.stderr)

    report_md = build_report_markdown(
        eval_path=eval_path,
        base_url=args.base_url,
        report_ks=list(TARGET_REPORT_KS),
        rows=rows,
        repeat_stats_section=build_repeat_stats_markdown(
            repeats=repeat_count,
            report_ks=list(TARGET_REPORT_KS),
            rows=repeat_stat_rows,
        ),
        tag_grouped_section=build_tag_grouped_report_markdown(
            eval_payload=eval_payload,
            detailed_rows=detailed_rows,
            report_ks=list(TARGET_REPORT_KS),
        ),
        agent_regression_section=build_agent_behavior_regression_markdown(
            agent_suite,
            inference_tag=args.inference_tag,
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    details_path = _details_out_path(out_path)
    details_md = build_details_report_markdown(
        eval_path=eval_path,
        base_url=args.base_url,
        detail_k=DETAIL_REPORT_K,
        eval_payload=eval_payload,
        detailed_rows=detailed_rows,
    )
    details_path.write_text(details_md, encoding="utf-8")
    print(f"[OK] Report written to: {out_path}")
    print(f"[OK] Detailed report written to: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
