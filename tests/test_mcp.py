"""MCP server: six tools on a server named "quen", JSON-safe returns,
thin over the same engine (spec §5)."""

import asyncio
import json

import pytest

import quen.engine as engine_mod
from quen.mcp_server import (
    dream,
    inspect,
    mcp,
    pin,
    recall,
    remember,
    verify_hint,
)

EXPECTED_TOOLS = {"remember", "recall", "dream", "verify_hint", "pin", "inspect"}


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
