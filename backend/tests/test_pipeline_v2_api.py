import os
import sys
import importlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

checkpoint_db = backend_root / "data" / "test_langgraph_v2_checkpoints.sqlite"
if checkpoint_db.exists():
    checkpoint_db.unlink()

os.environ.setdefault("STORAGE_PATH", "data/test.db")
os.environ["RAG_GENERATION_MODE"] = "creative"
os.environ["RAG_CREATIVE_MCP_ENABLED"] = "false"
os.environ.setdefault("LANGGRAPH_V2_CHECKPOINT_DB", str(checkpoint_db))

from app.api.deps import get_services  # noqa: E402
from app.api.main import create_app  # noqa: E402
from app.services.citation_enforcer import CoverageReport  # noqa: E402
from app.services.drafting_service import DraftResult  # noqa: E402
from app.services.planner_service import OutlinePlan  # noqa: E402
from app.services.research_service import RelevanceReport, ResearchNote  # noqa: E402
import app.api.routes.pipeline as pipeline_route  # noqa: E402
import app.api.routes.pipeline_v2 as pipeline_v2_route  # noqa: E402
import app.services.pipeline_langgraph_v2 as pipeline_v2_workflow  # noqa: E402


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def plan_outline(self, **kwargs) -> OutlinePlan:
        self.calls.append(kwargs)
        return OutlinePlan(
            outline="1) Outline\n- Intro\n- Body",
            assumptions="none",
            open_questions="none",
        )

    def plan_outline_stream(self, *, on_chunk=None, **kwargs) -> OutlinePlan:
        result = self.plan_outline(**kwargs)
        if on_chunk:
            for chunk in ["1) Outline\n", "- Intro\n", "- Body"]:
                on_chunk(chunk)
        return result


class FakeResearcher:
    def format_notes(self, notes) -> str:
        return "\n".join(f"- {note.title} ({note.doc_id})\n  {note.summary}" for note in notes)

    def relevance_report(self, query, sources) -> RelevanceReport:
        return RelevanceReport(
            query_terms=1,
            docs=len(list(sources)),
            best_recall=1.0,
            avg_recall=1.0,
            lexical_best=1.0,
            lexical_avg=1.0,
            tfidf_best=1.0,
            tfidf_avg=1.0,
        )


class FakePipeline:
    def __init__(self, planner: FakePlanner) -> None:
        self.planner = planner
        self.researcher = FakeResearcher()
        self.calls: list[dict[str, object]] = []
        self.fail_collect_once = False

    def collect_research_notes(self, topic, outline, sources):
        if self.fail_collect_once:
            self.fail_collect_once = False
            raise RuntimeError("collect failed")
        source_list = list(sources)
        self.calls.append(
            {
                "topic": topic,
                "outline": outline.outline,
                "source_count": len(source_list),
            }
        )
        if not source_list:
            return []
        return [
            ResearchNote(
                doc_id=source_list[0].doc_id,
                title=source_list[0].title,
                summary="summary from source",
                url=source_list[0].url,
            )
        ]


class FakeDrafter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []
        self.review_calls: list[dict[str, object]] = []
        self.rewrite_calls: list[dict[str, object]] = []
        self.fail_create_once = False
        self.fail_review_once = False
        self.writing_agent = SimpleNamespace(draft_stream=self._draft_stream)

    def _draft_stream(self, **kwargs):
        for chunk in ("draft ", "body"):
            yield chunk

    def create_draft(self, **kwargs) -> str:
        self.create_calls.append(kwargs)
        if self.fail_create_once:
            self.fail_create_once = False
            raise RuntimeError("draft failed")
        return "draft body"

    def review_draft(self, **kwargs) -> str:
        self.review_calls.append(kwargs)
        if self.fail_review_once:
            self.fail_review_once = False
            raise RuntimeError("review failed")
        return "review body"

    def revise_draft(self, **kwargs) -> str:
        self.rewrite_calls.append(kwargs)
        return "revised body"

    def extract_evidence(self, research_notes: str) -> str:
        return ""

    def build_constraints(self, **kwargs) -> str:
        return str(kwargs.get("constraints") or "")

    def run_full(self, **kwargs) -> DraftResult:
        self.calls.append(kwargs)
        return DraftResult(
            outline=str(kwargs.get("outline") or ""),
            research_notes=str(kwargs.get("research_notes") or ""),
            draft="draft body",
            review="review body",
            revised="revised body",
        )


class FakeReviewerService:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(review_stream=self._review_stream)
        self.review_calls: list[dict[str, object]] = []
        self.review_decision_calls: list[dict[str, object]] = []
        self.review_decision_from_review_calls: list[dict[str, object]] = []
        self.review_stream_calls: list[dict[str, object]] = []
        self.review_text = "review body"
        self.needs_rewrite = True
        self.reason = "needs more support"
        self.score = 0.88
        self.fail_once = False

    def _review_stream(self, **kwargs):
        self.review_stream_calls.append(kwargs)
        chunks = self.review_text.split(" ")
        for index, chunk in enumerate(chunks):
            suffix = " " if index < len(chunks) - 1 else ""
            yield chunk + suffix
 
    def review(self, **kwargs):
        self.review_calls.append(kwargs)
        return SimpleNamespace(review=self.review_text)

    def review_decision(self, **kwargs):
        self.review_decision_calls.append(kwargs)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("review decision failed")
        return SimpleNamespace(
            review=self.review_text,
            review_text=self.review_text,
            needs_rewrite=self.needs_rewrite,
            reason=self.reason,
            score=self.score,
        )

    def review_decision_from_review(self, **kwargs):
        self.review_decision_from_review_calls.append(kwargs)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("review decision failed")
        review_text = str(kwargs.get("review") or self.review_text)
        return SimpleNamespace(
            review=review_text,
            review_text=review_text,
            needs_rewrite=self.needs_rewrite,
            reason=self.reason,
            score=self.score,
        )

    def decide_rewrite(self, **kwargs):
        self.review_decision_calls.append(kwargs)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("review decision failed")
        return self.needs_rewrite


class FakeRewriterService:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(rewrite_stream=self._rewrite_stream)
        self.calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.fail_once = False
        self.revised_text = "revised body"

    def _rewrite_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        chunks = self.revised_text.split(" ")
        for index, chunk in enumerate(chunks):
            suffix = " " if index < len(chunks) - 1 else ""
            yield chunk + suffix

    def rewrite(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("rewrite failed")
        return SimpleNamespace(revised=self.revised_text)


class FakeRag:
    def search(self, query: str, top_k: int = 5):
        return []

    def query_variants(self, query: str):
        return []

    def get_embedder(self):
        return None


class FakeCitations:
    def build_citations(self, notes):
        return []

    def format_bibliography(self, citations):
        return ""


class FakeStorage:
    def save_draft_version(self, **kwargs) -> int:
        return 7


def _fake_services():
    planner = FakePlanner()
    return SimpleNamespace(
        planner=planner,
        pipeline=FakePipeline(planner),
        drafter=FakeDrafter(),
        reviewer=FakeReviewerService(),
        rewriter=FakeRewriterService(),
        rag=FakeRag(),
        citations=FakeCitations(),
        storage=FakeStorage(),
        github_mcp_tool=None,
        agent_tool_calling_enabled=False,
        agent_tools_catalog=(),
    )


def _build_client(monkeypatch, fake_services):
    global checkpoint_db
    checkpoint_db = backend_root / "data" / f"test_langgraph_v2_{uuid4().hex}.sqlite"
    monkeypatch.setenv("LANGGRAPH_V2_CHECKPOINT_DB", str(checkpoint_db))
    workflow_mod = importlib.reload(pipeline_v2_workflow)
    route_v2_mod = importlib.reload(pipeline_v2_route)
    route_pipeline_mod = importlib.reload(pipeline_route)
    routes_pkg = importlib.import_module("app.api.routes")
    importlib.reload(routes_pkg)
    api_main = importlib.import_module("app.api.main")
    api_main = importlib.reload(api_main)

    monkeypatch.setattr(workflow_mod, "get_services", lambda: fake_services)
    monkeypatch.setattr(
        route_pipeline_mod._citation_enforcer,
        "enforce",
        lambda text, notes, apply_labels, strict_labels, embedder: (
            text,
            CoverageReport(
                coverage=1.0,
                token_coverage=1.0,
                total_tokens=10,
                covered_tokens=10,
                total_paragraphs=1,
                covered_paragraphs=1,
                semantic_coverage=None,
                semantic_covered_paragraphs=0,
                semantic_total_paragraphs=0,
            ),
        ),
    )
    app = api_main.create_app()
    app.dependency_overrides[get_services] = lambda: fake_services
    return TestClient(app)


def _build_reloaded_client(monkeypatch, fake_services, db_path: Path):
    global checkpoint_db
    checkpoint_db = db_path
    monkeypatch.setenv("LANGGRAPH_V2_CHECKPOINT_DB", str(db_path))

    workflow_mod = importlib.reload(pipeline_v2_workflow)
    route_mod = importlib.reload(pipeline_v2_route)
    routes_pkg = importlib.import_module("app.api.routes")
    importlib.reload(routes_pkg)
    api_main = importlib.import_module("app.api.main")
    api_main = importlib.reload(api_main)

    monkeypatch.setattr(workflow_mod, "get_services", lambda: fake_services)
    route_pipeline_mod = importlib.import_module("app.api.routes.pipeline")
    route_pipeline_mod = importlib.reload(route_pipeline_mod)
    monkeypatch.setattr(
        route_pipeline_mod._citation_enforcer,
        "enforce",
        lambda text, notes, apply_labels, strict_labels, embedder: (
            text,
            CoverageReport(
                coverage=1.0,
                token_coverage=1.0,
                total_tokens=10,
                covered_tokens=10,
                total_paragraphs=1,
                covered_paragraphs=1,
                semantic_coverage=None,
                semantic_covered_paragraphs=0,
                semantic_total_paragraphs=0,
            ),
        ),
    )
    app = api_main.create_app()
    app.dependency_overrides[get_services] = lambda: fake_services
    return TestClient(app), workflow_mod


def _start_pipeline(client: TestClient) -> dict:
    start_resp = client.post(
        "/api/pipeline/v2",
        json=_pipeline_payload(),
    )
    assert start_resp.status_code == 200
    return start_resp.json()


def _resume_outline(client: TestClient, thread_id: str, outline_override: str = ""):
    return client.post(
        "/api/pipeline/v2/resume",
        json={"thread_id": thread_id, "outline_override": outline_override},
    )


def _resume_draft(client: TestClient, thread_id: str, draft_override: str = ""):
    return client.post(
        "/api/pipeline/v2/resume",
        json={"thread_id": thread_id, "draft_override": draft_override},
    )


def _pipeline_payload() -> dict:
    return {
        "topic": "测试主题",
        "audience": "初学者",
        "style": "说明文",
        "target_length": "800",
        "constraints": "不要跑题",
        "key_points": "要点A",
        "review_criteria": "准确",
        "sources": [
            {
                "doc_id": "doc-1",
                "title": "Doc 1",
                "content": "content",
                "url": "",
            }
        ],
    }


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for chunk in body.split("\n\n"):
        data_lines = [
            line.replace("data:", "", 1).strip()
            for line in chunk.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        try:
            events.append(json.loads("\n".join(data_lines)))
        except Exception:
            continue
    return events


def test_pipeline_v2_interrupt_and_resume(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    start_data = _start_pipeline(client)
    assert start_data["status"] == "interrupted"
    assert start_data["interrupt"]["kind"] == "outline_review"
    assert "outline" in start_data["interrupt"]["payload"]
    thread_id = start_data["thread_id"]
    assert thread_id
    assert fake_services.planner.calls[0]["session_id"] == thread_id

    draft_interrupt_resp = _resume_outline(client, thread_id, "1) Outline\n- Human Intro\n- Human Body")
    assert draft_interrupt_resp.status_code == 200
    draft_interrupt_data = draft_interrupt_resp.json()
    assert draft_interrupt_data["status"] == "interrupted"
    assert draft_interrupt_data["interrupt"]["kind"] == "draft_review"
    assert draft_interrupt_data["interrupt"]["payload"]["interrupt_stage"] == "draft_review"
    assert draft_interrupt_data["interrupt"]["payload"]["draft"] == "draft body"
    assert fake_services.drafter.create_calls[0]["outline"] == "1) Outline\n- Human Intro\n- Human Body"
    assert fake_services.drafter.create_calls[0]["session_id"] == thread_id

    review_interrupt_resp = _resume_draft(client, thread_id, "draft body\nhuman edited")
    assert review_interrupt_resp.status_code == 200
    review_interrupt_data = review_interrupt_resp.json()
    assert review_interrupt_data["status"] == "interrupted"
    assert review_interrupt_data["interrupt"]["kind"] == "review_confirmation"
    assert review_interrupt_data["interrupt"]["payload"]["interrupt_stage"] == "review_confirmation"
    assert review_interrupt_data["interrupt"]["payload"]["review_text"] == "review body"
    assert review_interrupt_data["interrupt"]["payload"]["needs_rewrite"] is True
    assert review_interrupt_data["interrupt"]["payload"]["reason"] == "needs more support"
    assert review_interrupt_data["interrupt"]["payload"]["score"] == 0.88

    resume_resp = _resume_draft(client, thread_id)
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()
    assert resume_data["status"] == "completed"
    assert resume_data["interrupt"] is None
    assert resume_data["result"]["outline"] == "1) Outline\n- Human Intro\n- Human Body"
    assert resume_data["result"]["draft"] == "draft body\nhuman edited"
    assert resume_data["result"]["review"] == "review body"
    assert resume_data["result"]["version_id"] == 7


def test_pipeline_v2_resume_returns_404_for_unknown_thread(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    resp = client.post(
        "/api/pipeline/v2/resume",
        json={"thread_id": "missing-thread", "outline_override": ""},
    )
    assert resp.status_code == 404


def test_pipeline_v2_resume_is_idempotent_for_same_thread(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]

    first_interrupt = _resume_outline(client, thread_id, "1) Outline\n- Human Intro")
    assert first_interrupt.status_code == 200
    assert first_interrupt.json()["status"] == "interrupted"

    review_interrupt = _resume_draft(client, thread_id, "draft body\nedited one")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"

    first = _resume_draft(client, thread_id)
    second = _resume_draft(client, thread_id)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "completed"
    assert second.json()["status"] == "completed"
    assert second.json()["result"]["draft"] == first.json()["result"]["draft"]


def test_pipeline_v2_resume_without_outline_override_uses_original_outline(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]
    original_outline = start_data["interrupt"]["payload"]["outline"]

    interrupt_resp = _resume_outline(client, thread_id)
    assert interrupt_resp.status_code == 200
    assert interrupt_resp.json()["status"] == "interrupted"

    review_interrupt = _resume_draft(client, thread_id)
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"

    resp = _resume_draft(client, thread_id)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["outline"] == original_outline


def test_pipeline_v2_resume_with_empty_outline_override_uses_original_outline(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]
    original_outline = start_data["interrupt"]["payload"]["outline"]

    interrupt_resp = _resume_outline(client, thread_id, "")
    assert interrupt_resp.status_code == 200
    assert interrupt_resp.json()["status"] == "interrupted"

    review_interrupt = _resume_draft(client, thread_id)
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"

    resp = _resume_draft(client, thread_id)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["outline"] == original_outline


def test_pipeline_v2_resume_requires_thread_id(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    resp = client.post(
        "/api/pipeline/v2/resume",
        json={"outline_override": "1) Outline\n- Human Intro"},
    )
    assert resp.status_code == 422


def test_pipeline_v2_resume_survives_module_reload_with_persistent_checkpoint(monkeypatch, tmp_path):
    db_path = tmp_path / "langgraph_v2.sqlite"
    fake_services = _fake_services()
    client, _ = _build_reloaded_client(monkeypatch, fake_services, db_path)

    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]
    interrupt_resp = _resume_outline(client, thread_id, "1) Outline\n- Reloaded Intro")
    assert interrupt_resp.status_code == 200
    assert interrupt_resp.json()["status"] == "interrupted"
    client.close()

    resumed_services = _fake_services()
    resumed_client, _ = _build_reloaded_client(monkeypatch, resumed_services, db_path)
    review_interrupt = _resume_draft(resumed_client, thread_id, "draft body\nreloaded edit")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"

    resp = _resume_draft(resumed_client, thread_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["outline"] == "1) Outline\n- Reloaded Intro"
    assert data["result"]["draft"] == "draft body\nreloaded edit"


def test_pipeline_v2_persistent_store_returns_404_for_unknown_thread(monkeypatch, tmp_path):
    db_path = tmp_path / "langgraph_v2.sqlite"
    fake_services = _fake_services()
    client, _ = _build_reloaded_client(monkeypatch, fake_services, db_path)

    resp = client.post(
        "/api/pipeline/v2/resume",
        json={"thread_id": "missing-thread", "outline_override": ""},
    )
    assert resp.status_code == 404


def test_pipeline_v2_checkpointer_raises_for_invalid_sqlite_path(tmp_path):
    invalid_path = tmp_path
    try:
        pipeline_v2_workflow._create_default_checkpointer(invalid_path)
    except sqlite3.Error:
        return
    raise AssertionError("expected sqlite3.Error for invalid checkpoint db path")


def test_pipeline_v2_stream_interrupts(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    with client.stream(
        "POST",
        "/api/pipeline/v2/stream",
        json={
            "topic": "测试主题",
            "audience": "初学者",
            "style": "说明文",
            "target_length": "800",
            "constraints": "不要跑题",
            "key_points": "要点A",
            "review_criteria": "准确",
            "sources": [],
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"type": "interrupt"' in body
    assert '"kind": "outline_review"' in body
    assert '"thread_id": "' in body


def test_pipeline_v2_resume_stream_completes(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id, "outline_override": "1) Outline\n- Stream Intro"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"type": "research"' in body
    assert '"type": "delta", "stage": "draft"' in body
    assert '"type": "interrupt"' in body
    assert '"kind": "draft_review"' in body
    assert '"type": "delta", "stage": "review"' not in body
    assert '"type": "result"' not in body

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id, "draft_override": "draft body\nstream edited"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"type": "delta", "stage": "review"' in body
    assert '"type": "review_decision"' in body
    assert '"kind": "review_confirmation"' in body
    assert '"review_text": "review body"' in body
    assert '"reason": "needs more support"' in body
    assert '"score": 0.88' in body
    assert '"type": "result"' not in body
    assert '"draft": "draft body\\nstream edited"' in body
    assert len(fake_services.reviewer.review_stream_calls) == 1
    assert len(fake_services.rewriter.stream_calls) == 0

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"type": "delta", "stage": "rewrite"' in body
    assert '"type": "result"' in body
    assert '"review": "review body"' in body
    assert '"revised": "revised body"' in body
    assert len(fake_services.rewriter.stream_calls) == 1


def test_pipeline_v2_sync_matches_pipeline_sync(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    payload = _pipeline_payload()

    pipeline_resp = client.post("/api/pipeline", json=payload)
    assert pipeline_resp.status_code == 200
    pipeline_data = pipeline_resp.json()

    start_resp = client.post("/api/pipeline/v2", json=payload)
    assert start_resp.status_code == 200
    thread_id = start_resp.json()["thread_id"]
    first_resume = _resume_outline(client, thread_id, start_resp.json()["interrupt"]["payload"]["outline"])
    assert first_resume.status_code == 200
    assert first_resume.json()["status"] == "interrupted"
    second_resume = _resume_draft(client, thread_id)
    assert second_resume.status_code == 200
    assert second_resume.json()["status"] == "interrupted"
    assert second_resume.json()["interrupt"]["kind"] == "review_confirmation"
    v2_resp = _resume_draft(client, thread_id)
    assert v2_resp.status_code == 200
    v2_data = v2_resp.json()["result"]

    assert v2_data["draft"] == pipeline_data["draft"]
    assert v2_data["review"] == pipeline_data["review"]
    assert v2_data["revised"] == pipeline_data["revised"]
    assert v2_data["coverage"] == pipeline_data["coverage"]
    assert v2_data["coverage_detail"] == pipeline_data["coverage_detail"]


def test_pipeline_v2_stream_matches_pipeline_stream_result_shape(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    payload = _pipeline_payload()

    with client.stream("POST", "/api/pipeline/stream", json=payload) as resp:
        assert resp.status_code == 200
        pipeline_events = _parse_sse_events("".join(resp.iter_text()))
    pipeline_result = next(event["payload"] for event in pipeline_events if event.get("type") == "result")

    with client.stream("POST", "/api/pipeline/v2/stream", json=payload) as resp:
        assert resp.status_code == 200
        start_events = _parse_sse_events("".join(resp.iter_text()))
    thread_id = next(event["payload"]["thread_id"] for event in start_events if event.get("type") == "interrupt")

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id, "outline_override": ""},
    ) as resp:
        assert resp.status_code == 200
        first_events = _parse_sse_events("".join(resp.iter_text()))
    assert any(event.get("type") == "interrupt" and event.get("kind") == "draft_review" for event in first_events)

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id, "draft_override": ""},
    ) as resp:
        assert resp.status_code == 200
        second_events = _parse_sse_events("".join(resp.iter_text()))
    assert any(event.get("type") == "interrupt" and event.get("kind") == "review_confirmation" for event in second_events)

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id},
    ) as resp:
        assert resp.status_code == 200
        v2_events = _parse_sse_events("".join(resp.iter_text()))
    v2_result = next(event["payload"] for event in v2_events if event.get("type") == "result")

    assert set(v2_result.keys()) == set(pipeline_result.keys())
    assert v2_result["draft"] == pipeline_result["draft"]
    assert v2_result["review"] == pipeline_result["review"]
    assert v2_result["revised"] == pipeline_result["revised"]


def test_pipeline_v2_delete_checkpoint_then_resume_returns_404(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    start_data = _start_pipeline(client)
    thread_id = start_data["thread_id"]

    delete_resp = client.delete(f"/api/pipeline/v2/checkpoints/{thread_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    resume_resp = client.post(
        "/api/pipeline/v2/resume",
        json={"thread_id": thread_id, "outline_override": ""},
    )
    assert resume_resp.status_code == 404


def test_pipeline_v2_cleanup_only_removes_old_completed_and_supports_dry_run(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    completed_start = _start_pipeline(client)
    completed_thread = completed_start["thread_id"]
    first_resume = _resume_outline(client, completed_thread, "")
    assert first_resume.status_code == 200
    assert first_resume.json()["status"] == "interrupted"
    second_resume = client.post("/api/pipeline/v2/resume", json={"thread_id": completed_thread, "draft_override": ""})
    assert second_resume.status_code == 200
    assert second_resume.json()["status"] == "interrupted"
    final_resume = client.post("/api/pipeline/v2/resume", json={"thread_id": completed_thread})
    assert final_resume.status_code == 200
    assert final_resume.json()["status"] == "completed"

    interrupted_thread = _start_pipeline(client)["thread_id"]

    with sqlite3.connect(checkpoint_db) as conn:
        conn.execute(
            "UPDATE langgraph_v2_checkpoint_index SET updated_at = '2000-01-01 00:00:00' WHERE thread_id = ?",
            (completed_thread,),
        )
        conn.execute(
            "UPDATE langgraph_v2_checkpoint_index SET updated_at = '2000-01-01 00:00:00' WHERE thread_id = ?",
            (interrupted_thread,),
        )
        conn.commit()

    dry_run = client.post("/api/pipeline/v2/checkpoints/cleanup", json={"dry_run": True})
    assert dry_run.status_code == 200
    dry_run_data = dry_run.json()
    assert dry_run_data["dry_run"] is True
    assert dry_run_data["matched"] == 1
    assert dry_run_data["deleted"] == 0
    assert dry_run_data["thread_ids"] == [completed_thread]

    cleanup = client.post("/api/pipeline/v2/checkpoints/cleanup", json={})
    assert cleanup.status_code == 200
    cleanup_data = cleanup.json()
    assert cleanup_data["matched"] == 1
    assert cleanup_data["deleted"] == 1
    assert cleanup_data["thread_ids"] == [completed_thread]

    completed_detail = client.get(f"/api/pipeline/v2/checkpoints/{completed_thread}")
    interrupted_detail = client.get(f"/api/pipeline/v2/checkpoints/{interrupted_thread}")
    assert completed_detail.status_code == 404
    assert interrupted_detail.status_code == 200


def test_pipeline_v2_checkpoint_list_and_detail(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)

    interrupted = _start_pipeline(client)
    completed_thread = interrupted["thread_id"]
    first_resume = _resume_outline(client, completed_thread, "")
    assert first_resume.status_code == 200
    assert first_resume.json()["status"] == "interrupted"
    second_resume = client.post("/api/pipeline/v2/resume", json={"thread_id": completed_thread, "draft_override": ""})
    assert second_resume.status_code == 200
    assert second_resume.json()["status"] == "interrupted"
    final_resume = client.post("/api/pipeline/v2/resume", json={"thread_id": completed_thread})
    assert final_resume.status_code == 200
    assert final_resume.json()["status"] == "completed"
    interrupted_thread = _start_pipeline(client)["thread_id"]

    list_resp = client.get("/api/pipeline/v2/checkpoints", params={"limit": 10})
    assert list_resp.status_code == 200
    checkpoints = list_resp.json()["checkpoints"]
    assert any(item["thread_id"] == completed_thread for item in checkpoints)
    assert any(item["thread_id"] == interrupted_thread for item in checkpoints)

    completed_detail = client.get(f"/api/pipeline/v2/checkpoints/{completed_thread}")
    interrupted_detail = client.get(f"/api/pipeline/v2/checkpoints/{interrupted_thread}")
    assert completed_detail.status_code == 200
    assert interrupted_detail.status_code == 200
    assert completed_detail.json()["can_resume"] is False
    assert interrupted_detail.json()["can_resume"] is True
    assert interrupted_detail.json()["outline"] != ""


def test_pipeline_v2_best_effort_resume_from_outline_accepted(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    fake_services.pipeline.fail_collect_once = True
    failed = _resume_outline(client, thread_id, "")
    assert failed.status_code == 500

    resume_state = pipeline_v2_workflow.load_pipeline_v2_resume_state(thread_id)
    assert resume_state is not None
    assert resume_state["current_stage"] == "outline_accepted"

    resumed = _resume_outline(client, thread_id, "")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"
    completed = _resume_draft(client, thread_id, "")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"


def test_pipeline_v2_best_effort_resume_from_research_done(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    fake_services.drafter.fail_create_once = True
    failed = _resume_outline(client, thread_id, "")
    assert failed.status_code == 500

    resume_state = pipeline_v2_workflow.load_pipeline_v2_resume_state(thread_id)
    assert resume_state is not None
    assert resume_state["current_stage"] == "research_done"
    research_call_count = len(fake_services.pipeline.calls)
    draft_call_count = len(fake_services.drafter.create_calls)

    resumed = _resume_outline(client, thread_id, "")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"
    completed = _resume_draft(client, thread_id, "")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert len(fake_services.pipeline.calls) == research_call_count
    assert len(fake_services.drafter.create_calls) == draft_call_count + 1


def test_pipeline_v2_best_effort_resume_from_draft_done(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    fake_services.reviewer.fail_once = True
    draft_interrupt = _resume_outline(client, thread_id, "")
    assert draft_interrupt.status_code == 200
    assert draft_interrupt.json()["status"] == "interrupted"

    failed = _resume_draft(client, thread_id, "")
    assert failed.status_code == 500

    resume_state = pipeline_v2_workflow.load_pipeline_v2_resume_state(thread_id)
    assert resume_state is not None
    assert resume_state["current_stage"] == "draft_done"
    create_call_count = len(fake_services.drafter.create_calls)
    review_call_count = len(fake_services.reviewer.review_decision_calls)

    resumed = _resume_draft(client, thread_id, "")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "interrupted"
    assert resumed.json()["interrupt"]["kind"] == "review_confirmation"
    completed = _resume_draft(client, thread_id, "")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert len(fake_services.drafter.create_calls) == create_call_count
    assert len(fake_services.reviewer.review_decision_calls) == review_call_count + 1


def test_pipeline_v2_tail_graph_skips_rewrite_when_review_says_no(monkeypatch):
    fake_services = _fake_services()
    fake_services.reviewer.review_text = "Looks good. No changes needed."
    fake_services.reviewer.needs_rewrite = False
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    first_resume = _resume_outline(client, thread_id, "")
    assert first_resume.status_code == 200
    assert first_resume.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"
    assert review_interrupt.json()["interrupt"]["payload"]["needs_rewrite"] is False
    resumed = _resume_draft(client, thread_id, "")
    assert resumed.status_code == 200
    result = resumed.json()["result"]
    assert result["review"] == "Looks good. No changes needed."
    assert result["revised"] == "draft body"
    assert fake_services.rewriter.calls == []


def test_pipeline_v2_tail_graph_resume_from_review_done(monkeypatch):
    fake_services = _fake_services()
    fake_services.rewriter.fail_once = True
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    first_resume = _resume_outline(client, thread_id, "")
    assert first_resume.status_code == 200
    assert first_resume.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    failed = _resume_draft(client, thread_id, "")
    assert failed.status_code == 500

    resume_state = pipeline_v2_workflow.load_pipeline_v2_resume_state(thread_id)
    assert resume_state is not None
    assert resume_state["current_stage"] == "review_done"
    review_call_count = len(fake_services.reviewer.review_decision_calls)

    resumed = _resume_draft(client, thread_id, "")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert len(fake_services.reviewer.review_decision_calls) == review_call_count


def test_pipeline_v2_review_confirmation_checkpoint_survives_reload(monkeypatch, tmp_path):
    db_path = tmp_path / "langgraph_v2_review_confirmation.sqlite"
    fake_services = _fake_services()
    client, _ = _build_reloaded_client(monkeypatch, fake_services, db_path)

    thread_id = _start_pipeline(client)["thread_id"]
    draft_interrupt = _resume_outline(client, thread_id, "")
    assert draft_interrupt.status_code == 200
    assert draft_interrupt.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "draft body\nedited before reload")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"
    assert review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"
    client.close()

    resumed_services = _fake_services()
    resumed_client, _ = _build_reloaded_client(monkeypatch, resumed_services, db_path)
    resp = _resume_draft(resumed_client, thread_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["draft"] == "draft body\nedited before reload"
    assert data["result"]["review"] == "review body"


def test_pipeline_v2_best_effort_resume_from_rewrite_done(monkeypatch):
    fake_services = _fake_services()
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    original_enforce = pipeline_route._citation_enforcer.enforce
    state = {"failed": False}

    def fail_once_enforce(*args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("post process failed")
        return original_enforce(*args, **kwargs)

    monkeypatch.setattr(pipeline_route._citation_enforcer, "enforce", fail_once_enforce)

    draft_interrupt = _resume_outline(client, thread_id, "")
    assert draft_interrupt.status_code == 200
    assert draft_interrupt.json()["status"] == "interrupted"
    review_interrupt = _resume_draft(client, thread_id, "")
    assert review_interrupt.status_code == 200
    assert review_interrupt.json()["status"] == "interrupted"

    failed = _resume_draft(client, thread_id, "")
    assert failed.status_code == 500

    resume_state = pipeline_v2_workflow.load_pipeline_v2_resume_state(thread_id)
    assert resume_state is not None
    assert resume_state["current_stage"] == "rewrite_done"
    rewrite_call_count = len(fake_services.rewriter.calls)

    resumed = _resume_draft(client, thread_id, "")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert len(fake_services.rewriter.calls) == rewrite_call_count


def test_pipeline_v2_full_graph_runner_drives_research_and_draft(monkeypatch):
    fake_services = _fake_services()
    _build_client(monkeypatch, fake_services)
    request_model = pipeline_v2_route.PipelineRequest.model_validate(_pipeline_payload())
    outline = OutlinePlan(outline="1) Outline\n- Intro\n- Body", assumptions="none", open_questions="none")

    full_state = pipeline_v2_workflow.run_pipeline_v2_full_sync(
        pipeline_v2_route._build_full_input(
            thread_id="graph-thread",
            mode="sync",
            request_model=request_model,
            resolved_session_id="graph-thread",
            effective_constraints=request_model.constraints,
            github_context="",
            outline=outline,
            research_notes=[],
            notes_text="",
            draft="",
            start_stage="outline_accepted",
            source_count=len(request_model.sources),
            services=fake_services,
        )
    )

    interrupts = full_state.get("__interrupt__")
    assert interrupts is not None
    interrupt_payload = dict(getattr(interrupts[0], "value", {}) or {})
    assert interrupt_payload["kind"] == "draft_review"
    assert interrupt_payload["draft"] == "draft body"
    assert len(fake_services.pipeline.calls) == 1
    assert len(fake_services.drafter.create_calls) == 1
    assert len(fake_services.reviewer.review_decision_calls) == 0


def test_pipeline_v2_resume_stream_skips_rewrite_consistently(monkeypatch):
    fake_services = _fake_services()
    fake_services.reviewer.review_text = "Looks good. No changes needed."
    fake_services.reviewer.needs_rewrite = False
    client = _build_client(monkeypatch, fake_services)
    thread_id = _start_pipeline(client)["thread_id"]

    first_sync = _resume_outline(client, thread_id, "")
    assert first_sync.status_code == 200
    assert first_sync.json()["status"] == "interrupted"
    sync_review_interrupt = _resume_draft(client, thread_id, "")
    assert sync_review_interrupt.status_code == 200
    assert sync_review_interrupt.json()["status"] == "interrupted"
    assert sync_review_interrupt.json()["interrupt"]["kind"] == "review_confirmation"
    sync_resp = _resume_draft(client, thread_id, "")
    assert sync_resp.status_code == 200
    sync_result = sync_resp.json()["result"]

    thread_id_stream = _start_pipeline(client)["thread_id"]
    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id_stream, "outline_override": ""},
    ) as resp:
        assert resp.status_code == 200
        first_events = _parse_sse_events("".join(resp.iter_text()))
    assert any(event.get("type") == "interrupt" and event.get("kind") == "draft_review" for event in first_events)

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id_stream, "draft_override": ""},
    ) as resp:
        assert resp.status_code == 200
        review_interrupt_events = _parse_sse_events("".join(resp.iter_text()))

    decision_event = next(event for event in review_interrupt_events if event.get("type") == "review_decision")
    interrupt_event = next(event for event in review_interrupt_events if event.get("type") == "interrupt")
    assert decision_event["payload"]["needs_rewrite"] is False
    assert interrupt_event["kind"] == "review_confirmation"

    with client.stream(
        "POST",
        "/api/pipeline/v2/resume/stream",
        json={"thread_id": thread_id_stream},
    ) as resp:
        assert resp.status_code == 200
        events = _parse_sse_events("".join(resp.iter_text()))

    result_event = next(event for event in events if event.get("type") == "result")
    assert result_event["payload"]["review"] == sync_result["review"]
    assert result_event["payload"]["revised"] == sync_result["revised"]
    assert len(fake_services.reviewer.review_stream_calls) >= 1
    assert len(fake_services.rewriter.stream_calls) == 0
