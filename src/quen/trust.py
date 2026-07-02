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
    """Per-memory trust = confidence x freshness; tombstones get zero."""
    if mem.status != "active":
        return 0.0
    return mem.confidence * freshness_factor(mem, now, cfg.freshness_half_life_days)


def hedge_phrase(trust: float, source_ref: Optional[str]) -> str:
    """Hedging qualifier proportional to trust; empty string = assert plainly."""
    if trust >= 0.75:
        return ""
    if trust >= 0.5:
        return "likely, but worth re-checking"
    return f"as of {source_ref or 'an old observation'} — may have changed"


def hedging_instruction() -> str:
    """System-prompt text: calibrate wording to each memory's trust tag."""
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
        mem.confidence = min(1.0, mem.confidence + cfg.confidence_bump_on_confirm)
    else:  # refuted → tombstone on the spot
        mem.confidence = mem.confidence * cfg.confidence_cut_on_refute
        successor = _newest_active_slot_mate(mem, store)
        if successor is not None:
            mem.status = "superseded"
            mem.superseded_by = successor.id
        else:
            mem.status = "deprecated"
        mem.valid_to = now

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
