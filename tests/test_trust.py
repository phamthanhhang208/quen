"""Trust gate tests (spec §4.7): freshness half-life, status-gated trust,
hedge bands, and answer-confidence aggregation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quen.trust import (
    answer_confidence,
    freshness_factor,
    hedge_phrase,
    hedging_instruction,
    trust_score,
)
from quen.verifiers import VerificationEvent


def _used(memory_id: str, *, score: float, trust: float) -> SimpleNamespace:
    """Duck-typed stand-in for a retrieval ScoredMemory."""
    return SimpleNamespace(score=score, trust=trust, memory=SimpleNamespace(id=memory_id))


def _event(memory_id: str, outcome: str, at) -> VerificationEvent:
    return VerificationEvent(
        memory_id=memory_id, verifier="repo_grep", outcome=outcome, evidence=None, at=at
    )


# ------------------------------------------------------------- freshness


def test_freshness_factor_is_half_at_half_life(clock, mem_factory) -> None:
    mem = mem_factory("the deploy target is region eu west")
    mem.last_verified_at = clock.now()          # anchor on verification
    clock.advance(days=30)
    assert freshness_factor(mem, clock.now(), 30.0) == pytest.approx(0.5)


def test_freshness_factor_one_when_just_verified(clock, mem_factory) -> None:
    mem = mem_factory("fact")
    clock.advance(days=45)                      # valid_from is long ago...
    mem.last_verified_at = clock.now()          # ...but verified right now
    assert freshness_factor(mem, clock.now(), 30.0) == pytest.approx(1.0)


def test_freshness_factor_falls_off_without_verification(clock, mem_factory) -> None:
    mem = mem_factory("fact")                   # anchor falls back to valid_from
    clock.advance(days=60)
    assert freshness_factor(mem, clock.now(), 30.0) == pytest.approx(0.25)


# ----------------------------------------------------------------- trust


def test_trust_is_confidence_times_freshness(clock, cfg, mem_factory) -> None:
    cfg.trust_stability_tempering = False  # the pure formula
    mem = mem_factory("fact", confidence=0.8)
    clock.advance(days=cfg.freshness_half_life_days)
    assert trust_score(mem, clock.now(), cfg) == pytest.approx(0.8 * 0.5)


def test_trust_stability_tempering_slows_decay_for_proven_facts(
    clock, cfg, mem_factory
) -> None:
    """Anti recency-bias: an old fact with EARNED stability keeps more trust
    than one with none, at the same age and confidence."""
    fragile = mem_factory("fact a", confidence=0.8, stability=1.0)
    proven = mem_factory("fact b", confidence=0.8, stability=100.0)
    clock.advance(days=60)
    t_fragile = trust_score(fragile, clock.now(), cfg)
    t_proven = trust_score(proven, clock.now(), cfg)
    assert t_proven > t_fragile
    cfg.trust_stability_tempering = False
    assert trust_score(proven, clock.now(), cfg) == pytest.approx(
        trust_score(fragile, clock.now(), cfg)
    )


@pytest.mark.parametrize("status", ["superseded", "deprecated"])
def test_trust_zero_for_non_active(status, clock, cfg, mem_factory) -> None:
    mem = mem_factory("fact", confidence=1.0)
    mem.status = status
    assert trust_score(mem, clock.now(), cfg) == 0.0


# ----------------------------------------------------------- hedge bands


def test_hedge_bands() -> None:
    assert hedge_phrase(0.9, None) == ""
    assert hedge_phrase(0.75, "PR#42") == ""                       # band edge: plain
    assert hedge_phrase(0.6, None) == "likely, but worth re-checking"
    assert hedge_phrase(0.5, "PR#42") == "likely, but worth re-checking"
    assert hedge_phrase(0.49, "PR#42") == "as of PR#42 — may have changed"
    assert hedge_phrase(0.1, None) == "as of an old observation — may have changed"


def test_hedging_instruction_mentions_trust_bands() -> None:
    text = hedging_instruction()
    assert isinstance(text, str) and text
    assert "trust" in text
    assert "0.75" in text and "0.5" in text


# ---------------------------------------------------- answer confidence


def test_answer_confidence_empty_used_is_floor() -> None:
    assert answer_confidence([], []) == pytest.approx(0.25)


def test_answer_confidence_is_score_share_weighted(clock) -> None:
    used = [_used("m1", score=3.0, trust=1.0), _used("m2", score=1.0, trust=0.2)]
    # (3*1.0 + 1*0.2) / 4
    assert answer_confidence(used, []) == pytest.approx(0.8)


def test_answer_confidence_confirmed_lifts_trust(clock) -> None:
    used = [_used("m1", score=1.0, trust=0.4)]
    events = [_event("m1", "confirmed", clock.now())]
    assert answer_confidence(used, events) == pytest.approx(0.9)
    # already-high trust is not lowered by the lift
    used_hi = [_used("m1", score=1.0, trust=0.95)]
    assert answer_confidence(used_hi, events) == pytest.approx(0.95)


def test_answer_confidence_refuted_drops_from_mean(clock) -> None:
    used = [_used("m1", score=5.0, trust=0.9), _used("m2", score=1.0, trust=0.3)]
    events = [_event("m1", "refuted", clock.now())]
    assert answer_confidence(used, events) == pytest.approx(0.3)


def test_answer_confidence_all_refuted_is_floor(clock) -> None:
    used = [_used("m1", score=2.0, trust=0.9)]
    events = [_event("m1", "refuted", clock.now())]
    assert answer_confidence(used, events) == pytest.approx(0.25)


def test_answer_confidence_unverifiable_leaves_trust_alone(clock) -> None:
    used = [_used("m1", score=1.0, trust=0.42)]
    events = [_event("m1", "unverifiable", clock.now())]
    assert answer_confidence(used, events) == pytest.approx(0.42)


def test_compact_hedge_bands() -> None:
    assert hedge_phrase(0.9, None, compact=True) == ""
    assert hedge_phrase(0.6, None, compact=True) == "re-check"
    assert hedge_phrase(0.4, "PR#42", compact=True) == "per PR#42; may be stale"
    assert hedge_phrase(0.4, None, compact=True) == "per old note; may be stale"


def test_compact_instruction_explains_tag_grammar() -> None:
    text = hedging_instruction(compact=True)
    assert "[t=<trust> <age>d]" in text
    assert "✓" in text
    assert "0.75" in text and "0.5" in text


def test_compact_strings_avoid_fama_negation_cues() -> None:
    """The offline reader echoes tags/hedges into scored answers — a cue
    word in a hedge would flip FAMA's negation forgiveness."""
    import re
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    import fama

    cues = fama.NEGATION_CUES_BEFORE | fama.NEGATION_CUES_AFTER
    # only tag + hedge ride along with answers; the hedging INSTRUCTION is
    # system text the reader never echoes (long form contains 'never' too)
    samples = [
        hedge_phrase(0.6, "PR#42", compact=True),
        hedge_phrase(0.4, "PR#42", compact=True),
        hedge_phrase(0.4, None, compact=True),
        "[t=0.82 3d ✓]",
    ]
    for s in samples:
        tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", s)}
        assert not (tokens & cues), (s, tokens & cues)
