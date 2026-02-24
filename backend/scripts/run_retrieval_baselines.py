from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    override: dict[str, bool]


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
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP request timeout (seconds)")
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
            tag_grouped_section.strip(),
            "" if tag_grouped_section.strip() else "",
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
    rows: list[tuple[BaselineSpec, dict[int, dict[str, float]]]] = []
    detailed_rows: list[tuple[BaselineSpec, dict[str, Any]]] = []

    # Disable proxy env inheritance for localhost calls. In some environments,
    # HTTP_PROXY/HTTPS_PROXY causes httpx to proxy 127.0.0.1 and return 502.
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        for spec in specs:
            print(f"[INFO] Running baseline: {spec.name} | override={spec.override}")
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
            rows.append((spec, metrics))
            detailed_rows.append((spec, resp_json))

    report_md = build_report_markdown(
        eval_path=eval_path,
        base_url=args.base_url,
        report_ks=list(TARGET_REPORT_KS),
        rows=rows,
        tag_grouped_section=build_tag_grouped_report_markdown(
            eval_payload=eval_payload,
            detailed_rows=detailed_rows,
            report_ks=list(TARGET_REPORT_KS),
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
