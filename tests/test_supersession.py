"""Two-stage supersession (spec §4.6.3): deterministic slot rule (zero LLM
calls, bi-temporal, tie-break stable), NLI fallback (confidence gate,
augmentation guard), pinned bars, and tombstone-chain preservation."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from quen import llm as llm_mod
from quen.supersession import (
    PINNED_NLI_BAR,
    SupersessionEvent,
    deterministic_pass,
    nli_pass,
    supersede,
)


def nli_json(label: str, confidence: float) -> str:
    return json.dumps({"label": label, "confidence": confidence})


# ------------------------------------------------------- deterministic stage


def test_deterministic_supersedes_older_valid_from_no_llm(
    store, mem_factory, scripted, clock
):
    old = mem_factory(
        "team fetches data via useApi", triple=("team", "fetches data via", "useApi")
    )
    store.add(old, actor="test")
    clock.advance(days=10)
    new = mem_factory(
        "team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(new, actor="test")
    clock.advance(days=1)

    events = deterministic_pass(store, now=clock.now(), run_id="run-1")

    [ev] = events
    assert isinstance(ev, SupersessionEvent)
    assert (ev.old_id, ev.new_id, ev.rule) == (old.id, new.id, "deterministic")
    assert ev.at == clock.now()
    got_old = store.get(old.id)
    assert got_old.status == "superseded"
    assert got_old.superseded_by == new.id
    assert got_old.valid_to == new.valid_from
    # tombstone, not delete: both rows still in the store
    assert store.get(new.id).status == "active"
    assert store.counts_by_status() == {"active": 1, "deprecated": 0, "superseded": 1}
    # the audit trail carries the rule
    [row] = store.audit_tail(memory_id=old.id, action="supersede")
    assert row["detail"]["rule"] == "deterministic"
    assert row["detail"]["new_id"] == new.id
    assert row["detail"]["run_id"] == "run-1"
    # the deterministic rule made ZERO LLM calls
    assert scripted.calls == []


def test_deterministic_is_ingest_order_independent(store, mem_factory, clock):
    """A late-INGESTED fact with an OLDER valid_from loses to the
    newer-valid_from fact regardless of ingest order (bi-temporal rule)."""
    t0 = clock.now()
    newer = mem_factory(
        "team deploys to vercel", triple=("team", "deploys to", "vercel")
    )
    store.add(newer, actor="test")  # ingested FIRST, newest valid_from
    clock.advance(days=5)
    older = mem_factory(
        "team deploys to heroku",
        triple=("team", "deploys to", "heroku"),
        valid_from=t0 - timedelta(days=30),
    )
    store.add(older, actor="test")  # ingested SECOND, older valid_from

    events = deterministic_pass(store, now=clock.now())

    [ev] = events
    assert (ev.old_id, ev.new_id) == (older.id, newer.id)
    assert store.get(older.id).status == "superseded"
    assert store.get(older.id).superseded_by == newer.id
    assert store.get(newer.id).status == "active"


def test_same_object_duplicates_are_not_contradictions(store, mem_factory, clock):
    a = mem_factory("ci runs on github actions", triple=("ci", "runs on", "github actions"))
    store.add(a, actor="test")
    clock.advance(days=2)
    b = mem_factory(
        "CI runs on GitHub Actions", triple=("CI", "Runs On", "GitHub  Actions")
    )  # same normalized (s, r, o)
    store.add(b, actor="test")

    assert deterministic_pass(store, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"


def test_winner_object_duplicates_untouched_while_loser_falls(
    store, mem_factory, clock
):
    loser = mem_factory(
        "team fetches data via useApi", triple=("team", "fetches data via", "useApi")
    )
    store.add(loser, actor="test")
    clock.advance(days=3)
    dup_of_winner = mem_factory(
        "team fetches data via USEQUERY",
        triple=("Team", "Fetches Data Via", "USEQUERY"),
    )
    store.add(dup_of_winner, actor="test")
    clock.advance(days=3)
    winner = mem_factory(
        "team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(winner, actor="test")

    events = deterministic_pass(store, now=clock.now())

    [ev] = events
    # bi-temporal chain: the loser is closed by its IMMEDIATE successor —
    # the first distinct-object arrival (day 3), not the final winner —
    # so the ledger records when the fact actually stopped being current
    assert ev.old_id == loser.id
    assert ev.new_id == dup_of_winner.id
    assert store.get(loser.id).valid_to == dup_of_winner.valid_from
    # same-o-as-winner member is a duplicate, NOT a contradiction
    assert store.get(dup_of_winner.id).status == "active"
    assert store.get(winner.id).status == "active"


def test_tiebreak_confidence_then_id_is_stable_across_runs(
    store, mem_factory, clock
):
    # identical valid_from and created_at → confidence decides
    hi = mem_factory(
        "team database is postgres", triple=("team", "database is", "postgres"),
        confidence=0.9,
    )
    lo = mem_factory(
        "team database is mysql", triple=("team", "database is", "mysql"),
        confidence=0.5,
    )
    store.add(hi, actor="test")
    store.add(lo, actor="test")

    events = deterministic_pass(store, now=clock.now())
    [ev] = events
    assert (ev.old_id, ev.new_id) == (lo.id, hi.id)

    # second run: settled — no flip-flop, no new events
    assert deterministic_pass(store, now=clock.now()) == []
    assert store.get(hi.id).status == "active"
    assert store.get(lo.id).superseded_by == hi.id


def test_tiebreak_full_tie_resolves_by_id_ascending(store, mem_factory, clock):
    a = mem_factory("cache ttl is 60", triple=("cache", "ttl is", "60"))
    b = mem_factory("cache ttl is 90", triple=("cache", "ttl is", "90"))
    store.add(a, actor="test")
    store.add(b, actor="test")
    expected_winner = min((a, b), key=lambda m: m.id)
    expected_loser = max((a, b), key=lambda m: m.id)

    [ev] = deterministic_pass(store, now=clock.now())
    assert (ev.old_id, ev.new_id) == (expected_loser.id, expected_winner.id)


def test_chain_preserved_settled_tombstones_never_rewritten(
    store, mem_factory, clock
):
    a = mem_factory("team uses redux", triple=("team", "state library is", "redux"))
    store.add(a, actor="test")
    clock.advance(days=5)
    b = mem_factory("team uses zustand", triple=("team", "state library is", "zustand"))
    store.add(b, actor="test")
    deterministic_pass(store, now=clock.now())
    assert store.get(a.id).superseded_by == b.id

    clock.advance(days=5)
    c = mem_factory("team uses jotai", triple=("team", "state library is", "jotai"))
    store.add(c, actor="test")
    [ev] = deterministic_pass(store, now=clock.now())

    assert (ev.old_id, ev.new_id) == (b.id, c.id)
    got_a = store.get(a.id)
    assert got_a.superseded_by == b.id  # A's tombstone still points at B
    assert got_a.valid_to == b.valid_from
    assert store.get(b.id).superseded_by == c.id
    assert store.get(c.id).status == "active"


def test_pinned_deterministic_blocked_below_confidence_bar(
    store, mem_factory, clock
):
    old = mem_factory(
        "prod region is eu-west-1",
        triple=("prod", "region is", "eu-west-1"),
        pinned=True,
        confidence=0.9,
    )
    store.add(old, actor="test")
    clock.advance(days=4)
    new = mem_factory(
        "prod region is us-east-1",
        triple=("prod", "region is", "us-east-1"),
        confidence=0.7,  # < old.confidence → blocked
    )
    store.add(new, actor="test")

    assert deterministic_pass(store, now=clock.now()) == []
    got = store.get(old.id)
    assert got.status == "active"
    assert got.superseded_by is None and got.valid_to is None
    [row] = store.audit_tail(action="supersede_blocked_pinned")
    assert row["memory_id"] == old.id
    assert row["detail"] == {"new_id": new.id, "rule": "deterministic"}


def test_pinned_deterministic_allowed_at_or_above_bar(store, mem_factory, clock):
    old = mem_factory(
        "prod region is eu-west-1",
        triple=("prod", "region is", "eu-west-1"),
        pinned=True,
        confidence=0.6,
    )
    store.add(old, actor="test")
    clock.advance(days=4)
    new = mem_factory(
        "prod region is us-east-1",
        triple=("prod", "region is", "us-east-1"),
        confidence=0.6,  # >= old.confidence → allowed
    )
    store.add(new, actor="test")

    [ev] = deterministic_pass(store, now=clock.now())
    assert (ev.old_id, ev.new_id) == (old.id, new.id)
    assert store.get(old.id).status == "superseded"


# ------------------------------------------------------------- NLI fallback


def add_pair(store, mem_factory, clock, *, pinned_old: bool = False):
    """Two non-slotted memories with clear entity-token overlap."""
    a = mem_factory("the office coffee machine is broken", pinned=pinned_old)
    store.add(a, actor="test")
    clock.advance(days=3)
    b = mem_factory("the office coffee machine works again after the repair")
    store.add(b, actor="test")
    return a, b


def test_nli_contradicts_above_gate_supersedes(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock)
    scripted.script(llm_mod.NLI, nli_json("contradicts", 0.9))

    events = nli_pass(
        store, scripted, embedder, cfg, now=clock.now(), run_id="run-n"
    )

    [ev] = events
    assert (ev.old_id, ev.new_id, ev.rule) == (a.id, b.id, "nli")
    assert ev.label == "contradicts"
    assert ev.nli_confidence == pytest.approx(0.9)
    got = store.get(a.id)
    assert got.status == "superseded"
    assert got.superseded_by == b.id
    assert got.valid_to == b.valid_from
    assert store.get(b.id).status == "active"
    [row] = store.audit_tail(memory_id=a.id, action="supersede")
    assert row["detail"]["rule"] == "nli"
    assert row["detail"]["label"] == "contradicts"


def test_nli_contradicts_below_gate_is_noop(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock)
    scripted.script(llm_mod.NLI, nli_json("contradicts", 0.5))  # < gate 0.7

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert len(scripted.calls) == 1  # the pair WAS evaluated, gate said no
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"


def test_nli_augments_keeps_both_and_audits(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock)
    scripted.script(llm_mod.NLI, nli_json("augments", 0.95))

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"
    [row] = store.audit_tail(action="augment_noted")
    assert row["memory_id"] == a.id
    assert row["detail"] == {"other": b.id}


@pytest.mark.parametrize("label", ["entails", "neutral"])
def test_nli_entails_and_neutral_are_noops(
    label, store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock)
    scripted.script(llm_mod.NLI, nli_json(label, 0.99))

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"
    assert store.audit_tail(action="augment_noted") == []
    assert store.audit_tail(action="supersede") == []


def test_nli_malformed_reply_audited_and_skipped(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock)
    scripted.script(llm_mod.NLI, "definitely not json")

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    [row] = store.audit_tail(action="error")
    assert row["memory_id"] == a.id
    assert row["detail"]["other"] == b.id


def test_nli_skips_pairs_sharing_a_slot(
    store, mem_factory, scripted, embedder, cfg, clock
):
    """Same-slot pairs belong to the deterministic rule — the NLI pass must
    not even call the LLM for them."""
    a = mem_factory(
        "team fetches data via useApi", triple=("team", "fetches data via", "useApi")
    )
    store.add(a, actor="test")
    clock.advance(days=2)
    b = mem_factory(
        "team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(b, actor="test")
    scripted.script(llm_mod.NLI, nli_json("contradicts", 0.99))

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert scripted.calls == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"


def test_nli_pinned_blocked_below_09_bar(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock, pinned_old=True)
    # 0.8 clears the normal gate (0.7) but not the pinned bar (0.9)
    assert cfg.nli_confidence_gate <= 0.8 < PINNED_NLI_BAR
    scripted.script(llm_mod.NLI, nli_json("contradicts", 0.8))

    assert nli_pass(store, scripted, embedder, cfg, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    [row] = store.audit_tail(action="supersede_blocked_pinned")
    assert row["memory_id"] == a.id
    assert row["detail"] == {"new_id": b.id, "rule": "nli"}


def test_nli_pinned_allowed_at_09_bar(
    store, mem_factory, scripted, embedder, cfg, clock
):
    a, b = add_pair(store, mem_factory, clock, pinned_old=True)
    scripted.script(llm_mod.NLI, nli_json("contradicts", 0.95))

    [ev] = nli_pass(store, scripted, embedder, cfg, now=clock.now())
    assert (ev.old_id, ev.new_id) == (a.id, b.id)
    assert store.get(a.id).status == "superseded"


def test_nli_fifo_pairs_ordered_and_each_pair_once(
    store, mem_factory, scripted, embedder, cfg, clock
):
    """Three overlapping memories → three deterministic pairs, oldest pair
    first; a FIFO script drives a different verdict per pair."""
    m0 = mem_factory("the roadmap lists payments first")
    store.add(m0, actor="test")
    clock.advance(days=1)
    m1 = mem_factory("the roadmap lists onboarding second")
    store.add(m1, actor="test")
    clock.advance(days=1)
    m2 = mem_factory("the roadmap got rewritten entirely last week")
    store.add(m2, actor="test")
    scripted.script(
        llm_mod.NLI,
        [  # newest-involving pairs first: (m0,m2), (m1,m2), then (m0,m1)
            nli_json("neutral", 0.6),
            nli_json("neutral", 0.6),
            nli_json("contradicts", 0.9),
        ],
    )

    events = nli_pass(store, scripted, embedder, cfg, now=clock.now())

    assert len(scripted.calls) == 3  # each pair evaluated exactly once
    [ev] = events
    assert (ev.old_id, ev.new_id) == (m0.id, m1.id)
    assert store.get(m2.id).status == "active"
    # pair ordering surfaced in the prompts: newest-involving first (the cap
    # must spend its budget reconciling fresh observations), A always older
    first_prompt = scripted.calls[0]["prompt"]
    assert f"A (older): {m0.content}" in first_prompt
    assert f"B (newer): {m2.content}" in first_prompt


def test_nli_superseded_memory_drops_out_of_later_pairs(
    store, mem_factory, scripted, embedder, cfg, clock
):
    """Once A is superseded mid-pass, its remaining pairs are skipped
    without burning LLM calls."""
    m0 = mem_factory("the roadmap lists payments first")
    store.add(m0, actor="test")
    clock.advance(days=1)
    m1 = mem_factory("the roadmap lists onboarding second")
    store.add(m1, actor="test")
    clock.advance(days=1)
    m2 = mem_factory("the roadmap got rewritten entirely last week")
    store.add(m2, actor="test")
    scripted.script(
        llm_mod.NLI,
        [  # (m0,m2) contradicts → m0 dies; (m1,m2) neutral; (m0,m1) skipped
            nli_json("contradicts", 0.9),
            nli_json("neutral", 0.6),
        ],
    )

    [ev] = nli_pass(store, scripted, embedder, cfg, now=clock.now())

    assert (ev.old_id, ev.new_id) == (m0.id, m2.id)
    assert len(scripted.calls) == 2
    assert store.get(m1.id).status == "active"


# ------------------------------------------------------------ supersede unit


def test_supersede_writes_tombstone_through_audited_update(
    store, mem_factory, clock
):
    old = mem_factory("old fact about the billing job")
    store.add(old, actor="test")
    clock.advance(days=7)
    new = mem_factory("new fact about the billing job")
    store.add(new, actor="test")

    ev = supersede(
        old, new, store=store, now=clock.now(), actor="tester", run_id="r-9"
    )

    assert ev == SupersessionEvent(
        old_id=old.id, new_id=new.id, rule="deterministic", at=clock.now()
    )
    got = store.get(old.id)
    assert got.status == "superseded"
    assert got.superseded_by == new.id
    assert got.valid_to == new.valid_from
    [row] = store.audit_tail(memory_id=old.id, action="supersede")
    assert row["actor"] == "tester"
    assert row["detail"]["run_id"] == "r-9"
