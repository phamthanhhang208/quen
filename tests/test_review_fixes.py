"""Regression tests for the adversarial-review findings."""

import json

import pytest

from quen import llm as llm_mod
from quen.dream import run_dream
from quen.engine import QuenEngine
from quen.llm import ScriptedLLM
from quen.models import make_memory
from quen.trust import apply_verification
from quen.verifiers import RepoGrepVerifier
from quen.write_pipeline import ingest_observation


def _ingest(text, store, embedder, cfg, now, scripted, triple, source_kind="chat"):
    scripted.script(
        llm_mod.EXTRACT,
        json.dumps([{"content": text, "triple": triple, "mtype": "episodic"}]),
    )
    scripted.script(llm_mod.SALIENCE, json.dumps([{"salience": 0.9, "importance": 6}]))
    return ingest_observation(
        text, store=store, llm=scripted, embedder=embedder, cfg=cfg, now=now,
        source_kind=source_kind,
    )


def test_same_slot_update_survives_cosine_dedup(store, embedder, cfg, scripted, clock):
    """A same-(s,r)-new-o near-paraphrase must reach the store — cosine
    dedup swallowing it would reinforce the contradicted fact instead."""
    r1 = _ingest(
        "the api gateway upstream timeout is thirty seconds",
        store, embedder, cfg, clock.now(), scripted,
        ["api gateway", "upstream timeout is", "thirty seconds"],
    )
    clock.advance(days=5)
    r2 = _ingest(
        "the api gateway upstream timeout is sixty seconds",
        store, embedder, cfg, clock.now(), scripted,
        ["api gateway", "upstream timeout is", "sixty seconds"],
    )
    assert len(r2.stored) == 1, "update was swallowed by cosine dedup"
    assert r2.reinforced == []
    old = store.get(r1.stored[0].id)
    assert old.review_count == 0, "stale fact must not be reinforced"


def test_generalization_dated_by_evidence_not_dream_time(
    store, mem_factory, cfg, embedder, clock
):
    """A generalization distilled from old episodics must not out-recency
    a newer fact in the same slot."""
    for i in range(3):
        store.add(
            mem_factory(f"note {i}: the team fetches data via useApi"),
            actor="test",
        )
    clock.advance(days=10)
    newer = mem_factory(
        "team fetches data via useQuery",
        triple=("team", "fetches data via", "useQuery"),
        confidence=0.9,
        mtype="semantic",  # keep it out of the re-abstraction pool
    )
    store.add(newer, actor="test")
    clock.advance(days=2)

    gen_json = json.dumps(
        [{
            "content": "the team standardizes on useApi",
            "triple": ["team", "fetches data via", "useApi"],
            "source_indices": [0, 1, 2],
        }]
    )
    llm = ScriptedLLM(
        {
            llm_mod.REABSTRACT: gen_json,
            llm_mod.SELFTEST_PROBE: json.dumps({"probe": "?", "expected": "zz"}),
            llm_mod.SELFTEST_ANSWER: "UNKNOWN",
            llm_mod.JUDGE: "NO",
            llm_mod.NLI: json.dumps({"label": "neutral", "confidence": 0.5}),
            llm_mod.COMPRESS: "short",
            llm_mod.JOURNAL: "j",
        }
    )
    report = run_dream(store, llm, embedder, cfg, now=clock.now())

    assert store.get(newer.id).status == "active", (
        "stale generalization must never supersede the newer fact"
    )
    if report.generalization_ids:
        gen = store.get(report.generalization_ids[0])
        # dated by its newest evidence, not by the dream run
        assert gen.valid_from < newer.valid_from
        assert gen.status == "superseded"  # the deterministic rule caught it


def test_verify_on_tombstone_is_a_noop(store, mem_factory, cfg, clock):
    old = mem_factory("x uses y", triple=("x", "uses", "y"))
    store.add(old, actor="test")
    clock.advance(days=1)
    new = mem_factory("x uses z", triple=("x", "uses", "z"))
    store.add(new, actor="test")
    old.status = "superseded"
    old.superseded_by = new.id
    old.valid_to = new.valid_from
    store.update(old, actor="test", action="supersede")

    clock.advance(days=10)
    event = apply_verification(
        old, "refuted", store=store, cfg=cfg, now=clock.now(), evidence="late"
    )
    assert event.outcome == "refuted"  # the attempt is reported...
    got = store.get(old.id)
    assert got.valid_to == new.valid_from, "settled valid_to must not move"
    assert got.superseded_by == new.id
    assert got.status == "superseded"
    assert store.reviews_for(old.id) == []  # ...but no review lands on a tombstone


def test_verification_does_not_revert_touch_access(store, mem_factory, cfg, clock):
    mem = mem_factory("CI pool is ci-pool-7")
    store.add(mem, actor="test")
    stale_copy = store.get(mem.id)  # what a recall would be holding
    later = clock.advance(days=20)
    store.touch_access([mem.id], at=later)

    apply_verification(
        stale_copy, "confirmed", store=store, cfg=cfg, now=clock.now()
    )
    assert store.get(mem.id).last_accessed_at == later


def test_repogrep_word_boundary_and_negative_claims(tmp_path, mem_factory):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "api.ts").write_text("export function useApiV2() {}\n")
    v = RepoGrepVerifier(str(repo))

    renamed = mem_factory("team fetches data via useApi",
                          triple=("team", "fetches data via", "useApi"))
    outcome, evidence = v.verify(renamed)
    assert outcome == "refuted", "substring match must not confirm useApi via useApiV2"

    negative = mem_factory("the team no longer uses useApiV2")
    assert not v.can_verify(negative)
    assert v.verify(negative)[0] == "unverifiable"


def test_gate_loop_verifies_backfilled_memories(
    cfg, store, embedder, clock, tmp_path
):
    """After a refutation frees budget, the re-recalled replacements face
    the same verify-before-answer gate."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "up.ts").write_text("export const uploadV3 = 1;\n")
    engine = QuenEngine(
        cfg,
        store=store,
        llm=ScriptedLLM.with_offline_defaults(),
        embedder=embedder,
        clock=clock.now,
        verifiers=[RepoGrepVerifier(str(repo))],
    )
    engine.ingest("Uploads go through the uploadV1 helper.")
    engine.ingest("Uploads go through the uploadV2 helper.")
    clock.advance(days=40)  # both stale, trust < threshold

    # budget fits ~one memory: refuting the first must not let the second
    # slip through unverified
    # budget fits exactly one memory including its tag overhead
    res = engine.ask("What do uploads go through?", token_budget=25)
    refuted = {v.memory_id for v in res.verifications if v.outcome == "refuted"}
    assert len(refuted) == 2, "backfilled stale memory escaped the gate"
    used_ids = {sm.memory.id for sm in res.used}
    assert not (used_ids & refuted)
