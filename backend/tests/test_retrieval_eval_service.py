from app.services.research_service import SourceDocument
from app.services.retrieval_eval_service import EvalCase, RetrievalEvalService


def test_retrieval_eval_metrics_macro_and_per_query():
    svc = RetrievalEvalService()

    index = {
        "q1": [
            SourceDocument(doc_id="d1", title="", content=""),
            SourceDocument(doc_id="d3", title="", content=""),
            SourceDocument(doc_id="d2", title="", content=""),
            SourceDocument(doc_id="d4", title="", content=""),
        ],
        "q2": [
            SourceDocument(doc_id="d5", title="", content=""),
            SourceDocument(doc_id="d2", title="", content=""),
            SourceDocument(doc_id="d1", title="", content=""),
        ],
    }

    def fake_search(query: str, top_k: int):
        return index.get(query, [])[:top_k]

    report = svc.evaluate(
        cases=[
            EvalCase(query="q1", relevant_doc_ids=["d1", "d2"], query_id="case-1"),
            EvalCase(query="q2", relevant_doc_ids=["d2"], query_id="case-2"),
        ],
        k_values=[1, 3],
        search_fn=fake_search,
    )

    assert report.total_queries == 2
    assert report.queries_with_relevance == 2
    assert report.k_values == [1, 3]

    by_k = {m.k: m for m in report.macro_metrics}

    # K=1: q1 hit(1/2), q2 miss(0/1)
    # recall = (0.5 + 0.0) / 2 = 0.25
    # precision = (1.0 + 0.0) / 2 = 0.5
    # hit_rate = (1 + 0) / 2 = 0.5
    # mrr = (1 + 0) / 2 = 0.5
    assert round(by_k[1].recall, 4) == 0.25
    assert round(by_k[1].precision, 4) == 0.5
    assert round(by_k[1].hit_rate, 4) == 0.5
    assert round(by_k[1].mrr, 4) == 0.5

    # K=3:
    # q1 rel@3 => [1,0,1] => recall=1.0 precision=2/3 hit=1 mrr=1.0
    # q2 rel@3 => [0,1,0] => recall=1.0 precision=1/3 hit=1 mrr=1/2
    # macro precision=(2/3 + 1/3)/2=0.5, macro mrr=(1+0.5)/2=0.75
    assert round(by_k[3].recall, 4) == 1.0
    assert round(by_k[3].precision, 4) == 0.5
    assert round(by_k[3].hit_rate, 4) == 1.0
    assert round(by_k[3].mrr, 4) == 0.75

    assert len(report.per_query) == 2
    first = report.per_query[0]
    assert first.query_id == "case-1"
    assert first.retrieved_doc_ids[:3] == ["d1", "d3", "d2"]
