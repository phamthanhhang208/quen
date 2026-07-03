"""Regression tests for the algorithm-bias audit findings."""

import json

import pytest

from quen import llm as llm_mod
from quen.dream import run_dream
from quen.llm import ScriptedLLM
from quen.supersession import (
    deterministic_pass,
    is_functional_relation,
    nli_pass,
    supersede,
)
from quen.trust import penalize_derived, trust_score
from quen.write_pipeline import ingest_observation


def test_multi_valued_slots_never_hit_the_deterministic_rule(
    store, mem_factory, scripted, clock
):
    """(team, uses, Postgres) + (team, uses, Redis) is augmentation — the
    single-value rule superseding one of them was the over-forgetting bias."""
    a = mem_factory("team uses Postgres", triple=("team", "uses", "Postgres"))
    store.add(a, actor="test")
    clock.advance(days=5)
    b = mem_factory("team uses Redis", triple=("team", "uses", "Redis"))
    store.add(b, actor="test")

    assert deterministic_pass(store, now=clock.now()) == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"

    # ...and the NLI stage now owns the pair, where augments protects both
    scripted.script(llm_mod.NLI, json.dumps({"label": "augments", "confidence": 0.9}))
    cfg = __import__("quen.config", fromlist=["QuenConfig"]).QuenConfig()
    events = nli_pass(store, scripted, None, cfg, now=clock.now())
    assert events == []
    assert store.get(a.id).status == "active"
    assert store.get(b.id).status == "active"
    assert len(scripted.calls) == 1  # the pair reached NLI


def test_functional_relation_classifier():
    assert is_functional_relation("runs on")
    assert is_functional_relation("is set to")
    assert is_functional_relation("default branch is")   # trailing copula
    assert is_functional_relation("The database is")     # article + copula head
    assert not is_functional_relation("uses")
    assert not is_functional_relation("prefers")
    assert not is_functional_relation("supports")
    assert not is_functional_relation("depends on")


def test_authority_guard_blocks_casual_over_authoritative(
    store, mem_factory, clock
):
    """A casual chat mention must not silently retire a merged-PR fact."""
    pr_fact = mem_factory(
        "svc runs on Kubernetes", triple=("svc", "runs on", "Kubernetes"),
        confidence=0.9,
    )
    store.add(pr_fact, actor="test")
    clock.advance(days=5)
    gossip = mem_factory(
        "svc runs on EC2 again", triple=("svc", "runs on", "EC2"),
        confidence=0.6,
    )
    store.add(gossip, actor="test")

    assert deterministic_pass(store, now=clock.now()) == []
    assert store.get(pr_fact.id).status == "active"
    [row] = store.audit_tail(action="supersede_blocked_authority")
    assert row["memory_id"] == pr_fact.id


def test_corroboration_refreshes_freshness_and_confidence(
    store, embedder, cfg, scripted, clock
):
    """Re-observation is fresh evidence: freshness anchor moves, confidence
    accumulates — a fact re-stated daily must not decay to maximal hedging."""
    def ingest(kind):
        scripted.script(llm_mod.EXTRACT, json.dumps(
            [{"content": "team uses pnpm", "triple": ["team", "package manager is", "pnpm"],
              "mtype": "episodic"}]))
        scripted.script(llm_mod.SALIENCE, json.dumps(
            [{"salience": 0.9, "importance": 6}]))
        return ingest_observation(
            "team uses pnpm", store=store, llm=scripted, embedder=embedder,
            cfg=cfg, now=clock.now(), source_kind=kind,
        )

    r1 = ingest("chat")
    mem_id = r1.stored[0].id
    clock.advance(days=20)
    r2 = ingest("pr")  # re-observed from a stronger source
    assert r2.reinforced == [mem_id]

    got = store.get(mem_id)
    assert got.last_verified_at == clock.now()   # freshness anchor refreshed
    assert got.confidence > 0.9 - 1e-9 or got.confidence == pytest.approx(0.9)
    assert got.freshness_days(clock.now()) == pytest.approx(0.0)
    assert trust_score(got, clock.now(), cfg) > 0.8


def test_selftest_zombie_backoff_and_single_importance_nudge(
    store, mem_factory, cfg, embedder, clock
):
    """A persistently failing memory must not monopolize the sample slots,
    ratchet importance forever, or refresh its own eviction TTL."""
    mem = mem_factory("obscure fact nobody can recall")
    store.add(mem, actor="test")
    llm = ScriptedLLM({
        llm_mod.REABSTRACT: "[]",
        llm_mod.SELFTEST_PROBE: json.dumps({"probe": "?", "expected": "zzz"}),
        llm_mod.SELFTEST_ANSWER: "UNKNOWN",
        llm_mod.JUDGE: "NO",
        llm_mod.NLI: json.dumps({"label": "neutral", "confidence": 0.5}),
        llm_mod.COMPRESS: "short",
        llm_mod.JOURNAL: "j",
    })
    clock.advance(days=3)
    accessed_before = store.get(mem.id).last_accessed_at
    r1 = run_dream(store, llm, embedder, cfg, now=clock.now())
    assert len(r1.self_tests) == 1 and not r1.self_tests[0].passed
    got = store.get(mem.id)
    assert got.last_accessed_at == accessed_before  # introspection ≠ usage
    imp_after_first = got.importance

    clock.advance(days=1)  # inside the backoff window
    r2 = run_dream(store, llm, embedder, cfg, now=clock.now())
    assert r2.self_tests == []  # backoff: not re-tested the next day

    clock.advance(days=cfg.selftest_fail_backoff_days)
    r3 = run_dream(store, llm, embedder, cfg, now=clock.now())
    if r3.self_tests:  # eligible again after backoff...
        assert store.get(mem.id).importance == imp_after_first  # ...no ratchet


def test_provenance_penalty_on_supersession_and_refutation(
    store, mem_factory, cfg, clock
):
    """Derived knowledge loses confidence when its evidence is invalidated —
    consolidation must not launder refuted facts into trusted generalizations."""
    src = mem_factory("team fetches data via useApi",
                      triple=("team", "fetches data via", "useApi"))
    store.add(src, actor="test")
    gen = mem_factory("this team standardizes on useApi", mtype="semantic",
                      provenance=[src.id], confidence=0.8)
    store.add(gen, actor="test")
    clock.advance(days=5)
    new = mem_factory("team fetches data via useQuery",
                      triple=("team", "fetches data via", "useQuery"))
    store.add(new, actor="test")

    supersede(src, new, store=store, now=clock.now())

    got = store.get(gen.id)
    assert got.confidence == pytest.approx(0.4)  # halved
    [row] = store.audit_tail(action="provenance_invalidated")
    assert row["memory_id"] == gen.id
    assert row["detail"]["source"] == src.id


def test_penalize_derived_skips_unrelated(store, mem_factory, clock):
    other = mem_factory("independent generalization", mtype="semantic",
                        provenance=["nonexistent"], confidence=0.8)
    store.add(other, actor="test")
    assert penalize_derived("some-id", store=store, now=clock.now()) == []
    assert store.get(other.id).confidence == pytest.approx(0.8)
