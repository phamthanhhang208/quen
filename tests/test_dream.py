"""Dream pass: validity transitions, pinned floor, self-test reviews,
re-abstraction with provenance, compression keeps verbatim."""

import json

import pytest

from quen import llm as llm_mod
from quen.dream import run_dream
from quen.llm import ScriptedLLM


def dream_llm(**overrides) -> ScriptedLLM:
    """ScriptedLLM with safe defaults for every marker a dream may touch."""
    s = ScriptedLLM(
        {
            llm_mod.REABSTRACT: "[]",
            llm_mod.SELFTEST_PROBE: json.dumps(
                {"probe": "recall probe?", "expected": "zzz-no-match"}
            ),
            llm_mod.SELFTEST_ANSWER: "UNKNOWN",
            llm_mod.JUDGE: "NO",
            llm_mod.NLI: json.dumps({"label": "neutral", "confidence": 0.5}),
            llm_mod.COMPRESS: "short summary",
            llm_mod.JOURNAL: "dream journal entry",
        }
    )
    s.handlers.update(overrides)
    return s


def test_evict_unpinned_but_pinned_floor_holds(store, mem_factory, cfg, embedder, clock):
    pinned = mem_factory("pinned secret: rotate keys quarterly", pinned=True)
    trivia = mem_factory("team lunch was pho at the corner place")
    store.add(pinned, actor="test")
    store.add(trivia, actor="test")

    clock.advance(days=90)  # default stability 1.0 → R ≈ 0.2 < θ, TTL exceeded
    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())

    assert trivia.id in report.evicted_ids
    got_trivia = store.get(trivia.id)
    assert got_trivia.status == "deprecated"
    assert got_trivia.valid_to == clock.now()
    got_pinned = store.get(pinned.id)
    assert got_pinned.status == "active"
    assert got_pinned.valid_to is None


def test_recently_accessed_memory_survives_ttl(store, mem_factory, cfg, embedder, clock):
    mem = mem_factory("rarely reviewed but recently touched")
    store.add(mem, actor="test")
    clock.advance(days=90)
    store.touch_access([mem.id], at=clock.now())  # accessed now → TTL not exceeded
    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())
    assert mem.id not in report.evicted_ids
    assert store.get(mem.id).status == "active"


def test_dream_supersession_validity_transition(store, mem_factory, cfg, embedder, clock):
    old = mem_factory(
        "team fetches data via useApi", triple=("team", "fetches data via", "useApi")
    )
    store.add(old, actor="test")
    clock.advance(days=10)
    new = mem_factory(
        "team fetches data via useQuery", triple=("team", "fetches data via", "useQuery")
    )
    store.add(new, actor="test")
    clock.advance(days=1)

    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())

    assert any(
        ev.old_id == old.id and ev.new_id == new.id and ev.rule == "deterministic"
        for ev in report.supersessions
    )
    got = store.get(old.id)
    assert got.status == "superseded"
    assert got.superseded_by == new.id
    assert got.valid_to == new.valid_from
    # the action is inspectable in the dream log with its rule
    run = store.dream_run(report.run_id)
    supersede_actions = [a for a in run["actions"] if a["phase"] == "supersede"]
    assert supersede_actions and supersede_actions[0]["detail"]["rule"] == "deterministic"


def test_reabstract_creates_generalization_with_provenance(
    store, mem_factory, cfg, embedder, clock
):
    eps = [
        mem_factory("we picked useQuery for the feed page", importance=4, confidence=0.9),
        mem_factory("we picked useQuery for the settings page", importance=6, confidence=0.7),
        mem_factory("we picked useQuery for the profile page", importance=5, confidence=0.8),
    ]
    for m in eps:
        store.add(m, actor="test")

    gen_json = json.dumps(
        [
            {
                "content": "this team prefers useQuery for data fetching",
                "triple": ["team", "prefers", "useQuery"],
                "source_indices": [0, 1, 2],
            }
        ]
    )
    clock.advance(days=1)
    report = run_dream(
        store, dream_llm(**{llm_mod.REABSTRACT: gen_json}), embedder, cfg, now=clock.now()
    )

    assert len(report.generalization_ids) == 1
    gen = store.get(report.generalization_ids[0])
    assert gen.mtype == "semantic"
    assert sorted(gen.provenance) == sorted(m.id for m in eps)
    assert gen.confidence == pytest.approx(0.7)   # min of sources
    assert gen.importance == pytest.approx(6.0)   # max of sources
    assert gen.source_ref == f"dream:{report.run_id}"

    # idempotent: covered episodics are not re-consumed on the next run
    clock.advance(days=1)
    report2 = run_dream(
        store, dream_llm(**{llm_mod.REABSTRACT: gen_json}), embedder, cfg, now=clock.now()
    )
    assert report2.generalization_ids == []
    semantics = store.list(mtype="semantic", limit=100)
    assert len(semantics) == 1


def test_selftest_pass_reinforces(store, mem_factory, cfg, embedder, clock):
    mem = mem_factory(
        "team fetches data via useApi", triple=("team", "fetches data via", "useApi")
    )
    store.add(mem, actor="test")
    clock.advance(days=3)  # FSRS reviews at t=0 don't move S

    s_before = mem.stability
    report = run_dream(
        store,
        dream_llm(**{llm_mod.SELFTEST_ANSWER: "the value is useApi"}),
        embedder,
        cfg,
        now=clock.now(),
    )

    [result] = report.self_tests
    assert result.passed and result.expected == "useApi"
    got = store.get(mem.id)
    assert got.stability > s_before
    reviews = store.reviews_for(mem.id)
    # pass grades HARD with capped growth — retrieval health, not recall proof
    assert reviews[-1]["kind"] == "self_test" and reviews[-1]["grade"] == 2
    assert got.stability <= s_before * cfg.selftest_pass_growth_cap + 1e-9
    events = store.calibration_events("retrieval_health")
    assert events and events[-1]["outcome"] == 1


def test_selftest_fail_reexpands_verbatim_and_fails_review(
    store, mem_factory, cfg, embedder, clock
):
    mem = mem_factory("the deploy pipeline requires manual approval for prod releases")
    mem.content = "deploy needs approval"  # simulate an earlier compression
    store.add(mem, actor="test")
    clock.advance(days=3)

    s_before = mem.stability
    importance_before = mem.importance
    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())

    [result] = report.self_tests
    assert not result.passed
    got = store.get(mem.id)
    assert got.content == got.content_verbatim  # re-expanded
    assert got.importance == importance_before + 1  # do-not-forget nudge
    assert got.stability < s_before
    reviews = store.reviews_for(mem.id)
    assert reviews[-1]["kind"] == "self_test" and reviews[-1]["grade"] == 1
    assert store.calibration_events("retrieval_health")[-1]["outcome"] == 0


def test_compress_keeps_verbatim(store, mem_factory, cfg, embedder, clock):
    long_text = "the retro notes said " + "very " * 150 + "long discussion about caching"
    mem = mem_factory(long_text)
    store.add(mem, actor="test")
    clock.advance(days=1)

    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())

    assert mem.id in report.compressed_ids
    got = store.get(mem.id)
    assert got.content == "short summary"
    assert got.content_verbatim == long_text


def test_journal_and_stats_persisted(store, mem_factory, cfg, embedder, clock):
    store.add(mem_factory("a fact"), actor="test")
    clock.advance(days=1)
    report = run_dream(store, dream_llm(), embedder, cfg, now=clock.now())
    runs = store.dream_runs()
    assert runs[0]["run_id"] == report.run_id
    assert runs[0]["journal"] == "dream journal entry"
    assert runs[0]["stats"]["self_tests"] == len(report.self_tests)
    assert runs[0]["finished_at"] is not None
