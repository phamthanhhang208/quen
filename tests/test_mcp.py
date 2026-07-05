"""MCP server: eight tools on a server named "quen", JSON-safe returns,
thin over the same engine (spec §5)."""

import asyncio
import json

import pytest

import quen.engine as engine_mod
from quen.mcp_server import (
    ask,
    dream,
    inspect,
    judge,
    mcp,
    pin,
    recall,
    remember,
    verify_hint,
)

EXPECTED_TOOLS = {"remember", "recall", "ask", "judge", "dream",
                  "verify_hint", "pin", "inspect"}


@pytest.fixture
def offline_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("QUEN_OFFLINE", "1")
    monkeypatch.setenv("QUEN_DB_PATH", str(tmp_path / "mcp.db"))
    engine_mod.reset_engine()
    yield engine_mod.get_engine()
    engine_mod.reset_engine()


def test_server_name_and_tools():
    assert mcp.name == "quen"
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.description  # every tool documents itself to the harness


def test_remember_recall_roundtrip(offline_engine):
    out = remember("The team fetches data via useQuery.", source_kind="pr",
                   source_ref="PR#42")
    json.dumps(out)  # JSON-safe
    assert len(out["stored"]) == 1

    got = recall("How does the team fetch data?", token_budget=500)
    json.dumps(got)
    assert got["tokens_used"] <= 500
    mem = got["memories"][0]
    for key in ("id", "content", "trust", "confidence", "freshness_days",
                "retrievability", "needs_verification", "last_verified_at"):
        assert key in mem


def test_verify_hint_pin_inspect_dream(offline_engine):
    out = remember("MAX_RETRIES is 4 in the worker config.")
    mem_id = out["stored"][0]["id"]

    ev = verify_hint(mem_id, "confirmed", evidence="grepped worker config")
    json.dumps(ev)
    assert ev["outcome"] == "confirmed"
    assert ev["memory_id"] == mem_id

    assert pin(mem_id)["pinned"] is True
    rows = inspect(status="active")
    json.dumps(rows)
    assert any(r["id"] == mem_id and r["pinned"] for r in rows)

    run = dream()
    json.dumps(run)
    assert run["run_id"].startswith("dream-")
    assert "stats" in run


def test_verify_hint_rejects_invalid_outcome(offline_engine):
    out = remember("Some fact.")
    with pytest.raises(ValueError):
        verify_hint(out["stored"][0]["id"], "definitely")


def test_ask_judge_closes_the_use_review_loop(offline_engine):
    out = remember("The team fetches data via useQuery.", source_kind="pr",
                   source_ref="PR#42")
    mem_id = out["stored"][0]["id"]

    res = ask("The team fetches data via which hook?", token_budget=300)
    json.dumps(res)  # JSON-safe
    assert res["trace_id"]
    assert not res["abstained"]
    assert "useQuery" in res["answer"]

    got = judge(res["trace_id"], correct=True)
    assert got == {"trace_id": res["trace_id"], "judged_correct": True}
    reviews = offline_engine.store.reviews_for(mem_id)
    assert reviews and reviews[-1]["kind"] == "use_judged"
    events = offline_engine.store.calibration_events(kind="retention")
    assert any(e["memory_id"] == mem_id for e in events)


def test_judge_unknown_trace_raises(offline_engine):
    with pytest.raises(KeyError):
        judge("trace-nope", correct=True)


def test_inspect_q_filter(offline_engine):
    remember("MAX_RETRIES is 4 in the worker config.")
    remember("Team lunch was pho.")
    hits = inspect(q="MAX_RETRIES")
    assert hits and all("MAX_RETRIES" in r["snippet"] for r in hits)
    assert inspect(q="zzz-nothing") == []
