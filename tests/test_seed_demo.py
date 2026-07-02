"""The demo seeder is the standing dashboard-contract smoke test: it must
reproduce every beat of the spec §10 narrative offline, and the data it
leaves behind must have the shapes the dashboard panels read."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import seed_demo  # noqa: E402
from quen.store import MemoryStore  # noqa: E402


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("demo") / "demo.db")
    summary = seed_demo.main(db)
    store = MemoryStore(db)
    yield summary, store
    store.close()


def test_every_status_is_populated(seeded):
    summary, _ = seeded
    counts = summary["counts"]
    assert counts["superseded"] >= 1
    assert counts["deprecated"] >= 4  # pho eviction + refuted tombstones
    assert counts["active"] >= 2


def test_deterministic_supersession_beat(seeded):
    summary, store = seeded
    [(old_id, new_id, rule)] = summary["supersessions_dream2"]
    assert rule == "deterministic"
    old, new = store.get(old_id), store.get(new_id)
    assert "useApi" in old.content and "useQuery" in new.content
    assert old.status == "superseded" and old.superseded_by == new_id
    assert old.valid_to == new.valid_from


def test_eviction_beat(seeded):
    summary, store = seeded
    [evicted_id] = summary["evicted_dream3"]
    mem = store.get(evicted_id)
    assert "pho" in mem.content
    assert mem.status == "deprecated" and mem.valid_to is not None
    assert any(
        row["action"] == "evict" for row in store.audit_tail(memory_id=evicted_id)
    )


def test_refuted_beat_tombstones_uploadv1(seeded):
    summary, store = seeded
    refuted_ids = [m for m, o in summary["uploads_verifications"] if o == "refuted"]
    assert refuted_ids
    contents = [store.get(m).content for m in refuted_ids]
    assert any("uploadV1" in c for c in contents)
    for m in refuted_ids:
        assert store.get(m).status in ("deprecated", "superseded")


def test_final_trace_is_the_flagship(seeded):
    summary, store = seeded
    trace = store.get_trace(summary["final_trace"])
    assert trace == store.latest_trace()

    outcomes = {v["outcome"] for v in trace["verifications"]}
    assert {"confirmed", "refuted"} <= outcomes

    used_contents = [
        store.get(u["memory_id"]).content for u in trace["used"]
    ]
    assert any("useQuery" in c for c in used_contents)
    assert all("useApi" not in c for c in used_contents)

    # the superseded useApi fact is deliberately excluded, with a reason...
    assert any("superseded by" in e["reason"] for e in trace["excluded"])
    # ...while the counterfactual (append-only RAG) would have injected it
    cf_contents = [store.get(c["memory_id"]).content for c in trace["counterfactual"]]
    assert any("useApi" in c for c in cf_contents)

    assert trace["tokens_used"] <= trace["token_budget"]
    assert 0 < trace["answer_confidence"] <= 1
    assert not trace["abstained"]


def test_confirmed_memories_updated(seeded):
    summary, store = seeded
    confirmed = [m for m, o in summary["final_verifications"] if o == "confirmed"]
    assert confirmed
    for m in confirmed:
        mem = store.get(m)
        assert mem.last_verified_at is not None
        reviews = store.reviews_for(m)
        assert any(r["kind"] == "verification" and r["grade"] == 3 for r in reviews)


def test_calibration_events_seeded(seeded):
    _, store = seeded
    retention = store.calibration_events("retention")
    confidence = store.calibration_events("confidence")
    assert retention, "dream self-tests must leave retention calibration points"
    outcomes = {e["outcome"] for e in confidence}
    assert outcomes == {0, 1}, "both correct and incorrect judged answers seeded"


def test_dashboard_data_present(seeded):
    """Every dashboard panel has something to render."""
    _, store = seeded
    assert len(store.dream_runs()) == 3          # Dream log
    assert store.latest_trace() is not None      # Recall trace
    assert store.list(limit=100)                 # Memories table
    run = store.dream_run(store.dream_runs()[0]["run_id"])
    assert isinstance(run["actions"], list)      # per-run feed
