"""Retrieval tests (spec §4.4): strict token budget, relevance-beats-decay,
active-only with the excluded-relevant panel, the append-only-RAG
counterfactual, and deterministic ordering."""

from __future__ import annotations

import pytest

from quen.retrieval import estimate_tokens, recall


def _used_ids(result) -> list[str]:
    return [sm.memory.id for sm in result.used]


def test_estimate_tokens_floor_and_ratio() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("x" * 40) == 10


def test_budget_cap_oversized_skip_and_continue(store, embedder, cfg, clock, mem_factory) -> None:
    """The top-scored item is too big for the budget: it is skipped and the
    scan continues — smaller items still fill, cap is never exceeded."""
    big = mem_factory(("alpha bravo charlie " * 12).strip(), importance=9.0)
    small_a = mem_factory("alpha bravo charlie delta", importance=5.0)
    small_b = mem_factory("alpha bravo charlie echo", importance=5.0)
    for m in (big, small_a, small_b):
        store.add(m, actor="test")

    # budget covers both small items INCLUDING their per-memory overhead
    budget = 2 * (7 + cfg.per_memory_overhead_tokens)
    assert estimate_tokens(big.content) + cfg.per_memory_overhead_tokens > budget

    result = recall(
        "alpha bravo charlie",
        token_budget=budget,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    ids = _used_ids(result)
    assert big.id not in ids
    assert set(ids) == {small_a.id, small_b.id}
    assert result.tokens_used == sum(
        sm.tokens + cfg.per_memory_overhead_tokens for sm in result.used
    )
    assert result.tokens_used <= budget
    assert result.token_budget == budget


def test_budget_too_small_nothing_fits(store, embedder, cfg, clock, mem_factory) -> None:
    mem = mem_factory("alpha bravo charlie delta echo foxtrot")
    store.add(mem, actor="test")
    result = recall(
        "alpha bravo charlie",
        token_budget=0,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    assert result.used == []
    assert result.tokens_used == 0
    assert result.counterfactual == []


def test_relevance_beats_decay(store, embedder, cfg, clock, mem_factory) -> None:
    """A 70-day-old memory with R < 0.3 but strong query overlap must outrank
    a fresh irrelevant one — decay never suppresses relevance."""
    old = mem_factory(
        "the team fetches data with the useApi hook for notifications",
        importance=5.0,
    )
    store.add(old, actor="test")

    clock.advance(days=70)
    fresh = mem_factory(
        "quarterly finance report totals were reconciled yesterday",
        importance=5.0,
    )
    store.add(fresh, actor="test")

    result = recall(
        "how does the team fetch data notifications useApi hook",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    ids = _used_ids(result)
    assert ids[0] == old.id
    top = result.used[0]
    assert top.retrievability < 0.3          # genuinely decayed
    assert top.relevance > 0.5               # but highly relevant
    # the fresh irrelevant memory doesn't merely rank lower — the inclusion
    # relevance floor keeps it out of the context entirely
    assert fresh.id not in ids


def test_active_only_and_excluded_relevant_reason(
    store, embedder, cfg, clock, mem_factory
) -> None:
    old = mem_factory(
        "the team fetches data with the useApi hook",
        triple=("team", "fetches data via", "useApi"),
    )
    store.add(old, actor="test")
    clock.advance(days=1)
    new = mem_factory(
        "the team fetches data with the useQuery hook",
        triple=("team", "fetches data via", "useQuery"),
    )
    store.add(new, actor="test")

    old.status = "superseded"
    old.superseded_by = new.id
    old.valid_to = clock.now()
    store.update(old, actor="test", action="supersede")

    result = recall(
        "how does the team fetch data useApi hook",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    ids = _used_ids(result)
    assert old.id not in ids
    assert new.id in ids

    excluded = {e["memory_id"]: e for e in result.excluded_relevant}
    assert old.id in excluded
    entry = excluded[old.id]
    assert entry["relevance"] >= 0.25
    assert entry["status"] == "superseded"
    assert entry["superseded_by"] == new.id
    assert "superseded by" in entry["reason"]
    assert new.id in entry["reason"]
    assert str(clock.now().date()) in entry["reason"]

    # counterfactual: an append-only RAG would have injected the stale memory
    cf_ids = [c["memory_id"] for c in result.counterfactual]
    assert old.id in cf_ids

    # historical mode may use the superseded memory again — with zero trust
    hist = recall(
        "how does the team fetch data useApi hook",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
        include_historical=True,
    )
    hist_ids = _used_ids(hist)
    assert old.id in hist_ids
    old_scored = next(sm for sm in hist.used if sm.memory.id == old.id)
    assert old_scored.trust == 0.0


def test_excluded_relevant_deprecated_reason(store, embedder, cfg, clock, mem_factory) -> None:
    mem = mem_factory("the staging cluster runs on kubernetes namespace shared")
    store.add(mem, actor="test")
    clock.advance(days=2)
    mem.status = "deprecated"
    mem.valid_to = clock.now()
    store.update(mem, actor="test", action="evict")

    result = recall(
        "which kubernetes namespace does the staging cluster use",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    assert result.used == []
    excluded = {e["memory_id"]: e for e in result.excluded_relevant}
    assert mem.id in excluded
    assert excluded[mem.id]["reason"].startswith("deprecated (evicted) on")
    assert str(clock.now().date()) in excluded[mem.id]["reason"]


def test_deterministic_ordering_and_tiebreak(store, embedder, cfg, clock, mem_factory) -> None:
    """Identical calls return identical results; exact ties break by id."""
    a = mem_factory("shared fact about the deploy pipeline tokens")
    b = mem_factory("shared fact about the deploy pipeline tokens")
    c = mem_factory("an unrelated grocery list entry")
    for m in (a, b, c):
        store.add(m, actor="test")

    kwargs = dict(token_budget=500, store=store, embedder=embedder, cfg=cfg, now=clock.now())
    r1 = recall("what is the deploy pipeline fact", **kwargs)
    r2 = recall("what is the deploy pipeline fact", **kwargs)

    assert _used_ids(r1) == _used_ids(r2)
    assert r1.excluded_relevant == r2.excluded_relevant
    assert r1.counterfactual == r2.counterfactual
    assert r1.tokens_used == r2.tokens_used
    assert r1.query_embedding == r2.query_embedding

    # a and b are byte-identical memories → same score → id ascending
    tied = [i for i in _used_ids(r1) if i in (a.id, b.id)]
    assert tied == sorted((a.id, b.id))


def test_retrieval_touches_access_but_is_not_a_review(
    store, embedder, cfg, clock, mem_factory
) -> None:
    mem = mem_factory("the api gateway uses rate limiting middleware")
    store.add(mem, actor="test")
    clock.advance(days=5)

    result = recall(
        "does the api gateway use rate limiting",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    assert mem.id in _used_ids(result)

    stored = store.get(mem.id)
    assert stored.last_accessed_at == clock.now()      # touched
    assert stored.last_review_at == mem.last_review_at  # NOT reviewed
    assert stored.stability == pytest.approx(mem.stability)
    assert store.reviews_for(mem.id) == []


def test_scored_memory_carries_trust_fields(store, embedder, cfg, clock, mem_factory) -> None:
    cfg.trust_stability_tempering = False  # pin the untempered formula here
    mem = mem_factory("the billing service retries webhooks three times", confidence=0.8)
    store.add(mem, actor="test")
    clock.advance(days=cfg.freshness_half_life_days)

    result = recall(
        "how many times does the billing service retry webhooks",
        token_budget=500,
        store=store,
        embedder=embedder,
        cfg=cfg,
        now=clock.now(),
    )
    sm = next(s for s in result.used if s.memory.id == mem.id)
    assert sm.confidence == pytest.approx(0.8)
    assert sm.freshness_days == pytest.approx(cfg.freshness_half_life_days)
    assert sm.trust == pytest.approx(0.8 * 0.5)  # one half-life elapsed
    assert sm.importance_norm == pytest.approx(mem.importance / 10.0)
    assert sm.tokens == estimate_tokens(mem.content)


def test_identifier_tokens_include_bare_numbers():
    """KU audit: '132 points' lost to the '132 meeples' distractor because
    pure-number tokens weren't lexical identifiers."""
    from quen.retrieval import _identifier_tokens

    assert "132" in _identifier_tokens("scored 132 points in Ticket to Ride")
    assert "220" in _identifier_tokens("now on page 220")
    # tiny numbers stay excluded — too common to be a signal
    assert _identifier_tokens("I have 2 cats") == set()
