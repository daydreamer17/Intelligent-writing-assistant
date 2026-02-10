from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Iterable, List

from .research_service import SourceDocument


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant_doc_ids: list[str]
    query_id: str = ""


@dataclass(frozen=True)
class MetricAtK:
    k: int
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    query_id: str
    relevant_count: int
    retrieved_doc_ids: list[str]
    metrics: list[MetricAtK]


@dataclass(frozen=True)
class EvalReport:
    total_queries: int
    queries_with_relevance: int
    k_values: list[int]
    macro_metrics: list[MetricAtK]
    per_query: list[EvalCaseResult]


class RetrievalEvalService:
    def evaluate(
        self,
        *,
        cases: Iterable[EvalCase],
        k_values: Iterable[int],
        search_fn: Callable[[str, int], List[SourceDocument]],
    ) -> EvalReport:
        case_list = [case for case in cases if case.query.strip()]
        ks = self._normalize_k_values(k_values)
        max_k = max(ks)

        per_query: list[EvalCaseResult] = []
        for case in case_list:
            retrieved = search_fn(case.query, max_k)
            retrieved_ids = _dedupe_preserve([doc.doc_id for doc in retrieved])[:max_k]
            metrics = self._case_metrics(retrieved_ids, set(case.relevant_doc_ids), ks)
            per_query.append(
                EvalCaseResult(
                    query=case.query,
                    query_id=case.query_id,
                    relevant_count=len(set(case.relevant_doc_ids)),
                    retrieved_doc_ids=retrieved_ids,
                    metrics=metrics,
                )
            )

        macro_metrics = self._macro_metrics(per_query, ks)
        queries_with_relevance = sum(1 for item in per_query if item.relevant_count > 0)
        return EvalReport(
            total_queries=len(per_query),
            queries_with_relevance=queries_with_relevance,
            k_values=ks,
            macro_metrics=macro_metrics,
            per_query=per_query,
        )

    @staticmethod
    def _normalize_k_values(k_values: Iterable[int]) -> list[int]:
        cleaned = sorted({int(k) for k in k_values if int(k) > 0})
        return cleaned or [1, 3, 5]

    def _case_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        ks: list[int],
    ) -> list[MetricAtK]:
        metrics: list[MetricAtK] = []
        relevance = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
        relevant_count = len(relevant_ids)

        for k in ks:
            rel_at_k = relevance[:k]
            hits = sum(rel_at_k)
            recall = (hits / relevant_count) if relevant_count > 0 else 0.0
            precision = hits / k
            hit_rate = 1.0 if hits > 0 else 0.0
            mrr = self._mrr_at_k(rel_at_k)
            ndcg = self._ndcg_at_k(rel_at_k, relevant_count, k)
            metrics.append(
                MetricAtK(
                    k=k,
                    recall=recall,
                    precision=precision,
                    hit_rate=hit_rate,
                    mrr=mrr,
                    ndcg=ndcg,
                )
            )
        return metrics

    def _macro_metrics(self, per_query: list[EvalCaseResult], ks: list[int]) -> list[MetricAtK]:
        if not per_query:
            return [
                MetricAtK(k=k, recall=0.0, precision=0.0, hit_rate=0.0, mrr=0.0, ndcg=0.0)
                for k in ks
            ]

        metrics: list[MetricAtK] = []
        for k in ks:
            rows = [row for row in per_query if any(item.k == k for item in row.metrics)]
            if not rows:
                metrics.append(MetricAtK(k=k, recall=0.0, precision=0.0, hit_rate=0.0, mrr=0.0, ndcg=0.0))
                continue
            recall = sum(next(item.recall for item in row.metrics if item.k == k) for row in rows) / len(rows)
            precision = sum(next(item.precision for item in row.metrics if item.k == k) for row in rows) / len(rows)
            hit_rate = sum(next(item.hit_rate for item in row.metrics if item.k == k) for row in rows) / len(rows)
            mrr = sum(next(item.mrr for item in row.metrics if item.k == k) for row in rows) / len(rows)
            ndcg = sum(next(item.ndcg for item in row.metrics if item.k == k) for row in rows) / len(rows)
            metrics.append(MetricAtK(k=k, recall=recall, precision=precision, hit_rate=hit_rate, mrr=mrr, ndcg=ndcg))
        return metrics

    @staticmethod
    def _mrr_at_k(rel_at_k: list[int]) -> float:
        for idx, rel in enumerate(rel_at_k, start=1):
            if rel > 0:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def _ndcg_at_k(rel_at_k: list[int], relevant_count: int, k: int) -> float:
        if not rel_at_k or relevant_count <= 0:
            return 0.0
        dcg = 0.0
        for idx, rel in enumerate(rel_at_k, start=1):
            if rel > 0:
                dcg += 1.0 / log2(idx + 1)
        ideal_len = min(relevant_count, k)
        idcg = sum(1.0 / log2(idx + 1) for idx in range(1, ideal_len + 1))
        if idcg <= 0.0:
            return 0.0
        return dcg / idcg


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
