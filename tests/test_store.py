"""Store round-trips, validity transitions, and the never-delete guarantee."""

from pathlib import Path

import pytest

from quen.models import make_memory

SRC = Path(__file__).parent.parent / "src" / "quen"


def test_no_hard_delete_anywhere_in_engine_source():
    """Spec: tombstones only. No code path may DELETE memory rows."""
    for py in SRC.glob("*.py"):
        text = py.read_text().casefold()
        assert "delete from memories" not in text, f"{py.name} deletes memories"


def test_roundtrip_preserves_all_fields(store, mem_factory, clock):
    mem = mem_factory(
        "team fetches data via useApi",
        triple=("team", "fetches data via", "useApi"),
        mtype="semantic",
        importance=7.0,
        salience=0.9,
        confidence=0.8,
        source_ref="session-1",
        provenance=["abc", "def"],
    )
    store.add(mem, actor="test")
    got = store.get(mem.id)
    assert got is not None
    assert got.content == mem.content
    assert got.triple == ("team", "fetches data via", "useApi")
    assert isinstance(got.triple, tuple)
    assert got.embedding == pytest.approx(mem.embedding)
    assert got.mtype == "semantic"
    assert got.importance == 7.0
    assert got.provenance == ["abc", "def"]
    assert got.status == "active"
    assert got.valid_from == mem.valid_from
    assert got.last_review_at.tzinfo is not None


def test_update_writes_audit_row(store, mem_factory):
    mem = mem_factory("MAX_UPLOAD_MB is 10")
    store.add(mem, actor="test")
    mem.confidence = 0.95
    store.update(mem, actor="trust", action="verify", detail={"outcome": "confirmed"})
    tail = store.audit_tail(memory_id=mem.id)
    actions = [row["action"] for row in tail]
    assert "verify" in actions and "create" in actions
    verify_row = next(r for r in tail if r["action"] == "verify")
    assert verify_row["detail"] == {"outcome": "confirmed"}


def test_update_unknown_id_raises(store, mem_factory):
    mem = mem_factory("ghost")
    with pytest.raises(KeyError):
        store.update(mem, actor="test")


def test_validity_transition_supersede_persists(store, mem_factory, clock):
    old = mem_factory("team fetches data via useApi",
                      triple=("team", "fetches data via", "useApi"))
    store.add(old, actor="test")
    clock.advance(days=10)
    new = mem_factory("team fetches data via useQuery",
                      triple=("team", "fetches data via", "useQuery"))
    store.add(new, actor="test")

    old.status = "superseded"
    old.superseded_by = new.id
    old.valid_to = new.valid_from
    store.update(old, actor="dream", action="supersede")

    got = store.get(old.id)
    assert got.status == "superseded"
    assert got.superseded_by == new.id
    assert got.valid_to == new.valid_from
    # both rows still exist — tombstone, not delete
    assert store.get(new.id) is not None
    assert store.counts_by_status() == {"active": 1, "superseded": 1, "deprecated": 0}


def test_find_by_slot_uses_normalized_keys(store, mem_factory):
    a = mem_factory("Team fetches data via useApi",
                    triple=("Team ", "Fetches Data Via", "useApi"))
    store.add(a, actor="test")
    found = store.find_by_slot(("team", "fetches data via"))
    assert [m.id for m in found] == [a.id]


def test_find_by_triple_active_only_by_default(store, mem_factory):
    a = mem_factory("x uses y", triple=("x", "uses", "y"))
    store.add(a, actor="test")
    a.status = "deprecated"
    store.update(a, actor="test", action="deprecate")
    assert store.find_by_triple(("x", "uses", "y")) is None
    assert store.find_by_triple(("x", "uses", "y"), status=None) is not None


def test_touch_access_updates_timestamp(store, mem_factory, clock):
    mem = mem_factory("something")
    store.add(mem, actor="test")
    later = clock.advance(days=2)
    store.touch_access([mem.id], at=later)
    assert store.get(mem.id).last_accessed_at == later


def test_reviews_and_calibration_roundtrip(store, mem_factory, clock):
    mem = mem_factory("fact")
    store.add(mem, actor="test")
    store.record_review(
        mem.id, kind="self_test", grade=3, elapsed_days=2.0, r_before=0.8,
        before=(5.0, 1.0), after=(4.9, 3.0),
    )
    rows = store.reviews_for(mem.id)
    assert len(rows) == 1
    assert rows[0]["kind"] == "self_test"
    assert rows[0]["s_after"] == 3.0

    store.record_calibration("retention", predicted=0.8, outcome=True, memory_id=mem.id)
    store.record_calibration("confidence", predicted=0.6, outcome=False,
                             freshness_days=12.0, trace_id="t1")
    assert len(store.calibration_events("retention")) == 1
    conf = store.calibration_events("confidence")
    assert conf[0]["freshness_days"] == 12.0
    assert conf[0]["outcome"] == 0


def test_dream_run_lifecycle(store, clock):
    store.start_dream_run("run-1")
    store.log_dream_action("run-1", "supersede",
                           {"rule": "deterministic", "old": "a", "new": "b"})
    store.finish_dream_run("run-1", journal="did things", stats={"superseded": 1})
    runs = store.dream_runs()
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["stats"] == {"superseded": 1}
    detail = store.dream_run("run-1")
    assert detail["actions"][0]["phase"] == "supersede"
    assert detail["actions"][0]["detail"]["rule"] == "deterministic"


def test_trace_roundtrip_and_latest(store, clock):
    t1 = {
        "trace_id": "t1", "at": "2026-06-01T00:00:00+00:00", "query": "q1",
        "answer": "a1", "answer_confidence": 0.8, "abstained": False,
        "token_budget": 100, "tokens_used": 50,
        "used": [{"memory_id": "m1"}], "excluded": [], "counterfactual": [],
        "verifications": [],
    }
    t2 = dict(t1, trace_id="t2", at="2026-06-02T00:00:00+00:00", query="q2")
    store.save_trace(t1)
    store.save_trace(t2)
    assert store.get_trace("t1")["query"] == "q1"
    assert store.latest_trace()["trace_id"] == "t2"
    assert store.get_trace("t1")["used"] == [{"memory_id": "m1"}]


def test_list_filters(store, mem_factory):
    a = mem_factory("alpha useApi note", mtype="episodic")
    b = mem_factory("beta general rule", mtype="semantic")
    store.add(a, actor="test")
    store.add(b, actor="test")
    assert {m.id for m in store.list(mtype="semantic")} == {b.id}
    assert {m.id for m in store.list(q="useApi")} == {a.id}
    assert len(store.list()) == 2
