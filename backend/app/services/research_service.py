from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from math import log
from typing import Iterable, List

from ..utils.tokenizer import tokenize, tokenize_list


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    content: str
    url: str = ""


@dataclass(frozen=True)
class ResearchNote:
    doc_id: str
    title: str
    summary: str
    url: str = ""


@dataclass(frozen=True)
class RelevanceReport:
    query_terms: int
    docs: int
    best_recall: float
    avg_recall: float
    lexical_best: float = 0.0
    lexical_avg: float = 0.0
    tfidf_best: float = 0.0
    tfidf_avg: float = 0.0


class ResearchService:
    def __init__(self, *, max_snippet_chars: int = 600) -> None:
        self.max_snippet_chars = max_snippet_chars

    def collect_notes(
        self,
        *,
        query: str,
        sources: Iterable[SourceDocument],
        top_k: int = 3,
    ) -> List[ResearchNote]:
        ranked = sorted(
            sources,
            key=lambda doc: self._score(query, doc.content, doc.title),
            reverse=True,
        )
        notes: List[ResearchNote] = []
        for doc in ranked[:top_k]:
            snippet = doc.content.strip().replace("\n", " ")
            summary = snippet[: self.max_snippet_chars].strip()
            notes.append(
                ResearchNote(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    summary=summary,
                    url=doc.url,
                )
            )
        return notes

    def format_notes(self, notes: Iterable[ResearchNote]) -> str:
        blocks = []
        for note in notes:
            header = f"- {note.title} ({note.doc_id})"
            if note.url:
                header += f" [{note.url}]"
            blocks.append(header + "\n  " + note.summary)
        return "\n".join(blocks)

    def relevance_report(
        self,
        query: str,
        sources: Iterable[SourceDocument],
    ) -> RelevanceReport:
        query_tokens = self._tokenize_list(query)
        query_terms = set(query_tokens)
        total_terms = len(query_terms)
        if total_terms == 0:
            return RelevanceReport(query_terms=0, docs=0, best_recall=0.0, avg_recall=0.0)

        doc_term_sets: List[set[str]] = []
        for doc in sources:
            terms = self._tokenize(doc.title + " " + doc.content)
            if terms:
                doc_term_sets.append(terms)
        if not doc_term_sets:
            return RelevanceReport(query_terms=total_terms, docs=0, best_recall=0.0, avg_recall=0.0)

        docs_count = len(doc_term_sets)
        query_tf = Counter(token for token in query_tokens if token in query_terms)
        doc_freq: dict[str, int] = {term: 0 for term in query_terms}
        for term in query_terms:
            for doc_terms in doc_term_sets:
                if term in doc_terms:
                    doc_freq[term] += 1

        query_weights: dict[str, float] = {}
        for term in query_terms:
            tf = float(query_tf.get(term, 1))
            idf = log((docs_count + 1) / (doc_freq.get(term, 0) + 1)) + 1.0
            query_weights[term] = tf * idf
        weight_sum = sum(query_weights.values()) or 1.0

        lexical_recalls: List[float] = []
        tfidf_recalls: List[float] = []
        mixed_recalls: List[float] = []
        for doc_terms in doc_term_sets:
            overlap_terms = query_terms.intersection(doc_terms)
            lexical = len(overlap_terms) / total_terms
            tfidf = sum(query_weights[t] for t in overlap_terms) / weight_sum
            mixed = (0.4 * lexical) + (0.6 * tfidf)
            lexical_recalls.append(lexical)
            tfidf_recalls.append(tfidf)
            mixed_recalls.append(mixed)

        best = max(mixed_recalls)
        avg = sum(mixed_recalls) / len(mixed_recalls)
        lexical_best = max(lexical_recalls)
        lexical_avg = sum(lexical_recalls) / len(lexical_recalls)
        tfidf_best = max(tfidf_recalls)
        tfidf_avg = sum(tfidf_recalls) / len(tfidf_recalls)
        return RelevanceReport(
            query_terms=total_terms,
            docs=docs_count,
            best_recall=best,
            avg_recall=avg,
            lexical_best=lexical_best,
            lexical_avg=lexical_avg,
            tfidf_best=tfidf_best,
            tfidf_avg=tfidf_avg,
        )

    @staticmethod
    def _score(query: str, content: str, title: str) -> int:
        query_terms = ResearchService._tokenize(query)
        if not query_terms:
            return 0
        content_terms = ResearchService._tokenize(content)
        title_terms = ResearchService._tokenize(title)
        return len(query_terms & content_terms) + 2 * len(query_terms & title_terms)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """使用统一的分词工具（支持中文）"""
        return tokenize(text, lowercase=True)

    @staticmethod
    def _tokenize_list(text: str) -> List[str]:
        """使用统一分词工具并保留词频信息"""
        return tokenize_list(text, lowercase=True)
