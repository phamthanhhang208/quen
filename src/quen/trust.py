"""Trust gate (spec §4.7) — trust as a runtime decision, not a stored field.

Per-memory trust is ``confidence x freshness`` for active memories and zero
for tombstones. Answer wording hedges in proportion to trust
(`hedge_phrase`), the stated answer confidence tracks the trust of what was
actually used (`answer_confidence`), and verification outcomes close the
loop back into retention (`apply_verification`): confirmed = FSRS Good
review + confidence bump, refuted = FSRS Again review + confidence cut +
immediate tombstone, unverifiable = audit only.

Import discipline: this module may import quen.verifiers / fsrs / store /
models — never quen.retrieval (retrieval imports us).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from quen.config import QuenConfig
from quen.fsrs import Grade, retrievability, review
from quen.models import MemoryItem
from quen.store import MemoryStore
from quen.verifiers import VerificationEvent

_OUTCOMES = ("confirmed", "refuted", "unverifiable")


def freshness_factor(mem: MemoryItem, now: datetime, half_life_days: float) -> float:
    """Exponential freshness decay: 0.5 at exactly one half-life.

    Anchored on the evidence age (`MemoryItem.freshness_days`): time since
    last verification if any, else since the fact became valid.
    """
    if half_life_days <= 0:
        return 0.0
    return 2.0 ** (-mem.freshness_days(now) / half_life_days)


def trust_score(mem: MemoryItem, now: datetime, cfg: QuenConfig) -> float:
    """Per-memory trust = confidence x freshness; tombstones get zero.

    Freshness decay is tempered by FSRS stability (anti recency-bias): a raw
    exponential half-life treats an old-but-repeatedly-confirmed fact like
    day-old gossip. Stability is exactly the earned evidence of durability —
    each successful review (use judged good, self-test pass, verification)
    stretches the effective half-life logarithmically.
    """
    if mem.status != "active":
        return 0.0
    half_life = cfg.freshness_half_life_days
    if cfg.trust_stability_tempering:
        half_life *= 1.0 + cfg.trust_stability_gain * math.log1p(
            mem.stability / cfg.trust_stability_ref
        )
    return mem.confidence * freshness_factor(mem, now, half_life)


def hedge_phrase(trust: float, source_ref: Optional[str], *,
                 compact: bool = False) -> str:
    """Hedging qualifier proportional to trust; empty string = assert plainly.

    Compact forms exist because the trust channel is per-memory overhead
    paid on every ask; they must never contain FAMA negation-cue words
    (tests pin this) or the scorer would mistake a hedge for a negation.
    """
    if trust >= 0.75:
        return ""
    if trust >= 0.5:
        return "re-check" if compact else "likely, but worth re-checking"
    if compact:
        return f"per {source_ref or 'old note'}; may be stale"
    return f"as of {source_ref or 'an old observation'} — may have changed"


def hedging_instruction(compact: bool = False) -> str:
    """System-prompt text: calibrate wording to each memory's trust tag."""
    if compact:
        return (
            "Each memory carries a tag [t=<trust> <age>d] — trust = "
            "confidence x freshness, age in days; a trailing ✓ means it "
            "was verified against the live source just now. Calibrate your "
            "wording: t >= 0.75 state plainly; 0.5 <= t < 0.75 qualify it; "
            "t < 0.5 attribute it to its source and never assert with full "
            "assurance. Your stated answer confidence must track the trust "
            "of the memories you relied on; if nothing trustworthy supports "
            "an answer, say so."
        )
    return (
        "Each memory you are given carries a trust tag "
        "(trust = confidence x freshness). Calibrate your wording to it: "
        "trust >= 0.75 — state the fact plainly; "
        "0.5 <= trust < 0.75 — qualify it (e.g. 'likely, but worth re-checking'); "
        "trust < 0.5 — attribute it to its source ('as of <source> — may have "
        "changed') and never assert it with full assurance. Your stated answer "
        "confidence must track the trust of the memories you actually relied on; "
        "if nothing trustworthy supports an answer, say so instead of guessing."
    )


def answer_confidence(used: Sequence, verifications: Sequence) -> float:
    """Score-share-weighted mean of per-memory trust for the answer.

    Duck-typed so retrieval need not be imported: ``used`` items expose
    ``.score``, ``.trust`` and ``.memory.id`` (quen.retrieval.ScoredMemory
    fits); ``verifications`` expose ``.memory_id`` and ``.outcome``.

    A "confirmed" verification lifts that memory's trust to at least 0.9;
    "refuted" memories are dropped from the mean entirely. With nothing
    used (or everything refuted) the floor 0.25 is returned — an answer
    with no memory basis is a guess, not a zero.
    """
    if not used:
        return 0.25
    outcome_by_id: dict[str, str] = {v.memory_id: v.outcome for v in verifications}
    kept: list[tuple[float, float]] = []
    for item in used:
        outcome = outcome_by_id.get(item.memory.id)
        if outcome == "refuted":
            continue
        trust = item.trust
        if outcome == "confirmed":
            trust = max(trust, 0.9)
        kept.append((max(0.0, item.score), trust))
    if not kept:
        return 0.25
    total = sum(score for score, _ in kept)
    if total <= 0.0:
        conf = sum(trust for _, trust in kept) / len(kept)
    else:
        conf = sum(score * trust for score, trust in kept) / total
    return max(0.0, min(1.0, conf))


# Raw trust is systematically UNDER-confident as a probability (measured:
# empirical accuracy ~1.0 in bins predicted 0.55-0.87, overall ECE 0.227 on
# the canonical probe) — a single-source chat fact carries confidence ~0.58
# by AUTHORITY, which is not the probability the answer is right. This
# piecewise-linear map recalibrates the STATED number against the observed
# accuracy; hedging language stays tied to raw trust (epistemic honesty is
# about the memory's provenance, the number is about the answer).
_CALIBRATION_KNOTS: tuple[tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.25, 0.45),
    (0.50, 0.75),
    (0.75, 0.92),
    (1.00, 1.00),
)


def calibrate_confidence(raw: float) -> float:
    """Monotone piecewise-linear interpolation over _CALIBRATION_KNOTS."""
    raw = max(0.0, min(1.0, raw))
    for (x0, y0), (x1, y1) in zip(_CALIBRATION_KNOTS, _CALIBRATION_KNOTS[1:]):
        if raw <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
    return 1.0


def apply_verification(
    mem: MemoryItem,
    outcome: str,
    *,
    store: MemoryStore,
    cfg: QuenConfig,
    now: datetime,
    actor: str = "trust",
    evidence: Optional[str] = None,
    verifier: str = "hint",
    kind: str = "verification",
) -> VerificationEvent:
    """Close the loop: a verification outcome IS an FSRS review (spec §4.7).

    confirmed:
        Good review (S grows, more so at low R), confidence bumped by
        ``cfg.confidence_bump_on_confirm`` (capped at 1.0),
        ``last_verified_at`` refreshed.
    refuted:
        Again review (S collapses), confidence multiplied by
        ``cfg.confidence_cut_on_refute``, then tombstoned immediately:
        superseded by the newest active same-slot memory with a later
        ``valid_from`` if one exists, else deprecated; ``valid_to = now``.
    unverifiable:
        No state change; the attempt is audited so the trace shows the
        gate fired.

    Every mutation goes through `MemoryStore.update` (audited); the review
    row lands in the reviews table with ``kind``.
    """
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown verification outcome: {outcome!r}")

    # Work on the freshest row: the caller's instance may predate this ask's
    # touch_access (or a concurrent pin) and a full-row write would silently
    # revert those fields.
    fresh = store.get(mem.id)
    if fresh is not None:
        mem = fresh

    # Settled tombstones are never rewritten: verifying a superseded or
    # deprecated memory must not move its recorded valid_to, retarget
    # superseded_by, or land reviews on a dead row. Audit the attempt only.
    if mem.status != "active":
        store.audit(
            actor,
            "verify",
            memory_id=mem.id,
            detail={
                "outcome": outcome,
                "evidence": evidence,
                "verifier": verifier,
                "ignored": f"memory is {mem.status}; tombstones are settled",
            },
            at=now,
        )
        return VerificationEvent(mem.id, verifier, outcome, evidence, now)

    if outcome == "unverifiable":
        store.audit(
            actor,
            "verify",
            memory_id=mem.id,
            detail={"outcome": outcome, "evidence": evidence, "verifier": verifier},
            at=now,
        )
        return VerificationEvent(mem.id, verifier, outcome, evidence, now)

    grade = Grade.GOOD if outcome == "confirmed" else Grade.AGAIN
    elapsed = mem.elapsed_days(now)
    r_before = retrievability(elapsed, mem.stability)
    before = (mem.difficulty, mem.stability)
    d_after, s_after = review(mem.difficulty, mem.stability, elapsed, grade)

    mem.difficulty = d_after
    mem.stability = s_after
    mem.last_review_at = now
    mem.review_count += 1
    mem.last_verified_at = now

    if outcome == "confirmed":
        # scaled by verifier strength: a grep hit is weaker evidence than a
        # human confirmation — weak verifiers must not mint full confidence
        strength = _verifier_strength(store, verifier)
        mem.confidence = min(
            1.0, mem.confidence + cfg.confidence_bump_on_confirm * strength
        )
    else:  # refuted → tombstone on the spot
        mem.confidence = mem.confidence * cfg.confidence_cut_on_refute
        successor = _newest_active_slot_mate(mem, store)
        if successor is not None:
            mem.status = "superseded"
            mem.superseded_by = successor.id
        else:
            mem.status = "deprecated"
        mem.valid_to = now
        # the refutation propagates to derived knowledge: generalizations
        # citing this memory as provenance lose confidence (weakened, not
        # disproven — they may rest on other evidence too)
        penalize_derived(mem.id, store=store, now=now, actor=actor)

    store.record_review(
        mem.id,
        kind=kind,
        grade=int(grade),
        elapsed_days=elapsed,
        r_before=r_before,
        before=before,
        after=(d_after, s_after),
        at=now,
    )
    store.update(
        mem,
        actor=actor,
        action="verify",
        detail={"outcome": outcome, "evidence": evidence, "verifier": verifier},
    )
    return VerificationEvent(mem.id, verifier, outcome, evidence, now)


# Known verifier strengths: how much a "confirmed" from this verifier is
# worth. RepoGrepVerifier confirms identifier EXISTENCE, not claim truth
# (a changed value behind the same name still "confirms") — weak evidence.
_VERIFIER_STRENGTH = {"repo_grep": 0.7, "hint": 1.0, "none": 0.0}


def _verifier_strength(store: MemoryStore, verifier: str) -> float:
    return _VERIFIER_STRENGTH.get(verifier, 0.8)


DERIVED_PENALTY = 0.5


def penalize_derived(
    tombstoned_id: str,
    *,
    store: MemoryStore,
    now: datetime,
    actor: str = "trust",
) -> list[str]:
    """When a memory is invalidated, generalizations that cite it as
    provenance lose confidence (audited as ``provenance_invalidated``).
    Without this, consolidation launders refuted facts into fully-trusted
    derived knowledge that the closed loop never touches again."""
    hit: list[str] = []
    for mem in store.list(mtype="semantic", limit=100_000):
        if mem.status != "active" or tombstoned_id not in mem.provenance:
            continue
        mem.confidence *= DERIVED_PENALTY
        store.update(
            mem,
            actor=actor,
            action="provenance_invalidated",
            detail={"source": tombstoned_id},
        )
        hit.append(mem.id)
    return hit


def _newest_active_slot_mate(mem: MemoryItem, store: MemoryStore) -> Optional[MemoryItem]:
    """Newest ACTIVE memory sharing mem's (s,r) slot with a later valid_from."""
    slot = mem.slot_key
    if slot is None:
        return None
    mates = [
        m
        for m in store.find_by_slot(slot)  # active-only, newest valid_from first
        if m.id != mem.id and m.valid_from > mem.valid_from
    ]
    return mates[0] if mates else None
