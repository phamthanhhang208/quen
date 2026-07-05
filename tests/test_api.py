"""FastAPI surface: every endpoint, DTO shapes frozen with
dashboard/src/lib/types.ts."""

import pytest
from fastapi.testclient import TestClient

from quen import fsrs
from quen.api import create_app
from quen.engine import QuenEngine
from quen.llm import ScriptedLLM


@pytest.fixture
def engine(cfg, store, embedder, clock):
    return QuenEngine(
        cfg,
        store=store,
        llm=ScriptedLLM.with_offline_defaults(),
        embedder=embedder,
        clock=clock.now,
    )


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def test_config_matches_fsrs_constants(client, cfg):
    data = client.get("/config").json()
    assert data["fsrs_version"] == fsrs.VERSION
    assert data["decay"] == fsrs.DECAY
    assert data["factor"] == pytest.approx(fsrs.FACTOR)
    assert data["eviction_r_threshold"] == cfg.eviction_r_threshold
    assert data["trust_threshold"] == cfg.trust_threshold
    assert data["default_token_budget"] == cfg.default_token_budget


def test_ingest_ask_trace_flow(client, clock):
    r = client.post(
        "/ingest",
        json={"text": "The team fetches data via useQuery.", "source_kind": "pr",
              "source_ref": "PR#42"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["stored"]) == 1
    assert body["stored"][0]["id"] and body["stored"][0]["snippet"]
    assert body["reinforced"] == [] and body["skipped"] == []

    clock.advance(days=1)
    r = client.post("/ask", json={"query": "How does the team fetch data?"})
    assert r.status_code == 200
    ask = r.json()
    assert set(ask) == {"trace_id", "answer", "answer_confidence", "abstained"}
    assert not ask["abstained"]

    trace = client.get(f"/recall/trace/{ask['trace_id']}").json()
    assert trace["query"] == "How does the team fetch data?"
    assert trace["tokens_used"] <= trace["token_budget"]
    used = trace["used"][0]
    for key in ("memory_id", "snippet", "relevance", "retrievability",
                "importance_norm", "score", "tokens", "trust", "confidence",
                "freshness_days"):
        assert key in used
    assert client.get("/recall/trace/latest").json()["trace_id"] == ask["trace_id"]

    r = client.post(f"/trace/{ask['trace_id']}/judge", json={"correct": True})
    assert r.status_code == 200
    assert client.post("/trace/nope/judge", json={"correct": True}).status_code == 404


def test_memory_list_detail_and_actions(client, clock):
    client.post("/ingest", json={"text": "Deploy target runs on Alibaba Cloud ECS."})
    rows = client.get("/memory").json()
    assert rows
    row = rows[0]
    for key in ("id", "snippet", "mtype", "status", "importance", "difficulty",
                "stability", "last_review_at", "review_count", "confidence",
                "freshness_days", "pinned", "valid_from", "valid_to",
                "superseded_by", "created_at", "triple"):
        assert key in row
    mem_id = row["id"]

    detail = client.get(f"/memory/{mem_id}").json()
    assert detail["content_verbatim"]
    assert isinstance(detail["reviews"], list)
    assert isinstance(detail["verifications"], list)
    assert isinstance(detail["audit"], list)
    assert client.get("/memory/nope").status_code == 404

    assert client.post(f"/memory/{mem_id}/pin", json={"pinned": True}).json()[
        "pinned"
    ] is True
    assert client.get(f"/memory/{mem_id}").json()["pinned"] is True

    clock.advance(days=3)
    st = client.post(f"/memory/{mem_id}/recall").json()
    assert set(st) == {"memory_id", "probe", "expected", "answer", "passed", "ds"}

    # no verifier configured -> unverifiable, audited
    ev = client.post(f"/memory/{mem_id}/verify").json()
    assert ev["outcome"] == "unverifiable"

    hint = client.post(
        f"/memory/{mem_id}/verify_hint",
        json={"result": "confirmed", "evidence": "grep hit"},
    ).json()
    assert hint["outcome"] == "confirmed"
    assert (
        client.post(
            f"/memory/{mem_id}/verify_hint", json={"result": "maybe"}
        ).status_code
        == 422
    )


def test_memory_filters(client):
    client.post("/ingest", json={"text": "The team prefers pnpm."})
    assert client.get("/memory", params={"status": "active"}).json()
    assert client.get("/memory", params={"q": "pnpm"}).json()
    assert client.get("/memory", params={"q": "zzz-none"}).json() == []


def test_dream_endpoints(client, clock):
    client.post("/ingest", json={"text": "The team fetches data via useApi."})
    clock.advance(days=10)
    client.post("/ingest", json={"text": "The team fetches data via useQuery."})
    clock.advance(days=1)
    run = client.post("/dream").json()
    assert run["run_id"] and run["stats"]["superseded"] == 1

    log = client.get("/dream/log").json()
    assert log[0]["run_id"] == run["run_id"]
    detail = client.get(f"/dream/{run['run_id']}").json()
    phases = [a["phase"] for a in detail["actions"]]
    assert "supersede" in phases
    assert client.get("/dream/nope").status_code == 404


def test_trace_latest_404_when_empty(client):
    assert client.get("/recall/trace/latest").status_code == 404


def test_vitals_shape(client, monkeypatch, tmp_path):
    monkeypatch.setenv("QUEN_EVAL_SUMMARY", str(tmp_path / "none.json"))
    client.post("/ingest", json={"text": "A fact for vitals."})
    v = client.get("/vitals").json()
    assert set(v["counts_by_status"]) == {"active", "deprecated", "superseded"}
    assert len(v["r_histogram"]) == 10
    assert v["kpis"]["fama"] is None
    assert isinstance(v["retention_calibration"], list)
    assert len(v["confidence_by_freshness_bucket"]) == 3


def test_recall_endpoint_budgeted_with_trust_fields(client):
    client.post("/ingest", json={"text": "The team fetches data via useQuery.",
                                 "source_kind": "pr", "source_ref": "PR#42"})
    r = client.post("/recall", json={"query": "data fetching hook",
                                     "token_budget": 300})
    assert r.status_code == 200
    got = r.json()
    assert got["tokens_used"] <= 300
    mem = got["memories"][0]
    for key in ("id", "content", "trust", "confidence", "freshness_days",
                "retrievability", "needs_verification"):
        assert key in mem


def test_dashboard_static_mount(engine, tmp_path, monkeypatch):
    """QUEN_DASHBOARD_DIST serves the built dashboard from the API origin;
    API routes are registered first and must keep winning."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>quen dashboard</html>")
    monkeypatch.setenv("QUEN_DASHBOARD_DIST", str(dist))
    client = TestClient(create_app(engine))
    assert "quen dashboard" in client.get("/").text
    assert client.get("/config").status_code == 200  # API still wins

    # without the env var (or a missing dir) nothing is mounted
    monkeypatch.setenv("QUEN_DASHBOARD_DIST", str(tmp_path / "nope"))
    bare = TestClient(create_app(engine))
    assert bare.get("/").status_code == 404
    assert bare.get("/config").status_code == 200
