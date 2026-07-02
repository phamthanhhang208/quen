"""Write pipeline — store the delta (spec §4.3).

An observation goes through: EXTRACT (atomic facts, (s,r,o) triples where
slot-like) → SALIENCE gate (skip what a strong base model already knows) →
dedup (triple key first, cosine second — duplicates *reinforce* the existing
memory with an FSRS Good review instead of creating a new row) → init DSR →
store, with confidence set from source authority.

Input is never dropped silently: if extraction is unparseable after one
retry, the raw observation is stored verbatim as a triple-less episodic
memory and an ``error`` audit row is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quen import fsrs
from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.llm import ChatLLM, parse_json_block, render_extract, render_salience
from quen.models import MemoryItem, MType, _norm, make_memory
from quen.store import MemoryStore

_DEFAULT_SALIENCE = 0.5
_DEFAULT_IMPORTANCE = 5.0
_EXTRACT_ATTEMPTS = 2  # one call + one retry


@dataclass
class SkippedFact:
    """A candidate fact the pipeline chose not to store."""

    content: str
    reason: str                    # "salience" | "dup_triple" | "dup_cosine"
    existing_id: str | None = None


@dataclass
class WriteResult:
    """Outcome of one ingested observation."""

    stored: list[MemoryItem]
    skipped: list[SkippedFact]
    reinforced: list[str]          # ids of existing memories reinforced via dedup


@dataclass
class _Candidate:
    """A validated extracted fact flowing through the pipeline stages."""

    content: str
    triple: Optional[tuple[str, str, str]]
    mtype: MType
    salience: float = _DEFAULT_SALIENCE
    importance: float = _DEFAULT_IMPORTANCE


def ingest_observation(
    text: str,
    *,
    store: MemoryStore,
    llm: ChatLLM,
    embedder: Embedder,
    cfg: QuenConfig,
    now: datetime,
    source_ref: str | None = None,
    source_kind: str = "chat",
    actor: str = "write",
) -> WriteResult:
    """Ingest one observation through the write pipeline (spec §4.3).

    Args:
        text: The raw observation (chat turn, commit message, PR body, ...).
        store: Persistence layer; every mutation is audited through it.
        llm: Chat LLM used for extraction and salience rating.
        embedder: Embedding backend (one batched call per ingest).
        cfg: Thresholds — salience gate, cosine dedup, authority confidence.
        now: Injected current time; no wall clock is read here.
        source_ref: Optional pointer to the evidence (e.g. "PR#42").
        source_kind: Authority class of the source; maps to a confidence
            prior via ``cfg.authority_confidence`` ("default" when unknown).
        actor: Audit-log actor for every mutation this call makes.

    Returns:
        WriteResult with newly stored memories, skipped facts (with the
        reason), and the ids of existing memories reinforced by dedup.
    """
    if not text or not text.strip():
        return WriteResult(stored=[], skipped=[], reinforced=[])

    confidence = cfg.authority_confidence.get(
        source_kind, cfg.authority_confidence["default"]
    )
    d0, s0 = fsrs.initial_state(fsrs.Grade.GOOD)

    raw_facts = _extract(llm, text, now)
    if raw_facts is None:
        # Never drop input silently: keep the whole observation verbatim.
        mem = make_memory(
            text,
            embedding=embedder.embed([text])[0],
            mtype="episodic",
            triple=None,
            importance=_DEFAULT_IMPORTANCE,
            salience=_DEFAULT_SALIENCE,
            difficulty=d0,
            stability=s0,
            confidence=confidence,
            source_ref=source_ref,
            now=now,
        )
        store.add(
            mem,
            actor=actor,
            detail={
                "source_kind": source_kind,
                "source_ref": source_ref,
                "fallback": "extract_parse_failure",
            },
        )
        store.audit(
            actor,
            "error",
            memory_id=mem.id,
            detail={
                "stage": "extract",
                "reason": "unparseable after retry; stored raw text verbatim",
                "attempts": _EXTRACT_ATTEMPTS,
                "source_ref": source_ref,
            },
            at=now,
        )
        return WriteResult(stored=[mem], skipped=[], reinforced=[])

    facts = [c for c in (_validate_fact(f) for f in raw_facts) if c is not None]
    if not facts:
        return WriteResult(stored=[], skipped=[], reinforced=[])

    ratings = _rate_salience(llm, [f.content for f in facts])

    stored: list[MemoryItem] = []
    skipped: list[SkippedFact] = []
    reinforced: list[str] = []

    # --- salience gate: store the delta, not everything -----------------
    gated: list[_Candidate] = []
    for fact, (sal, imp) in zip(facts, ratings):
        if sal < cfg.salience_threshold:
            skipped.append(SkippedFact(content=fact.content, reason="salience"))
            continue
        fact.salience, fact.importance = sal, imp
        gated.append(fact)

    # --- dedup within the batch: keep the first fact per triple key -----
    seen_keys: set[tuple[str, str, str]] = set()
    batch: list[_Candidate] = []
    for fact in gated:
        key = _triple_key(fact.triple)
        if key is not None:
            if key in seen_keys:
                skipped.append(
                    SkippedFact(content=fact.content, reason="dup_triple")
                )
                continue
            seen_keys.add(key)
        batch.append(fact)

    # --- dedup vs store, triple-first: reinforce instead of re-store ----
    survivors: list[_Candidate] = []
    for fact in batch:
        existing = store.find_by_triple(fact.triple) if fact.triple else None
        if existing is not None:
            _reinforce(
                store, existing.id,
                now=now, actor=actor, source_ref=source_ref, reason="dup_triple",
            )
            reinforced.append(existing.id)
            skipped.append(
                SkippedFact(
                    content=fact.content,
                    reason="dup_triple",
                    existing_id=existing.id,
                )
            )
        else:
            survivors.append(fact)

    if not survivors:
        return WriteResult(stored=stored, skipped=skipped, reinforced=reinforced)

    # --- cosine dedup second (one batched embed call), then store -------
    # A fact carrying a DIFFERENT triple than the candidate is a distinct
    # claim, never a cosine-duplicate: same-(s,r)-new-o updates are exactly
    # what near-paraphrase embeddings cannot distinguish (MemStrata's
    # AUROC-0.59 failure) — they must reach the store so the deterministic
    # supersession rule can see them.
    embeddings = embedder.embed([f.content for f in survivors])
    actives = store.active()

    for fact, emb in zip(survivors, embeddings):
        fact_key = _triple_key(fact.triple)
        best: Optional[MemoryItem] = None
        best_cos = 0.0  # cosine can be negative for unrelated texts; clamp at 0
        for candidate in actives:
            if fact_key is not None and candidate.triple_key is not None \
                    and candidate.triple_key != fact_key:
                continue  # distinct claim — dedup would swallow a contradiction
            c = cosine(emb, candidate.embedding)
            if c > best_cos:
                best, best_cos = candidate, c
        if best is not None and best_cos >= cfg.dedup_cosine_threshold:
            _reinforce(
                store, best.id,
                now=now, actor=actor, source_ref=source_ref, reason="dup_cosine",
            )
            reinforced.append(best.id)
            skipped.append(
                SkippedFact(
                    content=fact.content,
                    reason="dup_cosine",
                    existing_id=best.id,
                )
            )
            continue

        mem = make_memory(
            fact.content,
            embedding=emb,
            mtype=fact.mtype,
            triple=fact.triple,
            importance=fact.importance,
            salience=fact.salience,
            difficulty=d0,
            stability=s0,
            confidence=confidence,
            source_ref=source_ref,
            now=now,
        )
        store.add(
            mem,
            actor=actor,
            detail={"source_kind": source_kind, "source_ref": source_ref},
        )
        stored.append(mem)
        actives.append(mem)  # later batch members dedup against this one too

    return WriteResult(stored=stored, skipped=skipped, reinforced=reinforced)


# ------------------------------------------------------------------ helpers

def _extract(llm: ChatLLM, text: str, now: datetime) -> Optional[list]:
    """Run the EXTRACT prompt; retry once on unparseable/non-list output.

    Returns the parsed fact list, or None when both attempts failed (the
    caller then stores the raw observation verbatim).
    """
    messages = render_extract(text, now.isoformat())
    for _ in range(_EXTRACT_ATTEMPTS):
        raw = llm.complete(messages, model_hint="fast")
        try:
            parsed = parse_json_block(raw)
        except ValueError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _validate_fact(item: object) -> Optional[_Candidate]:
    """Coerce one extracted entry into a _Candidate; None when unusable."""
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    triple_raw = item.get("triple")
    triple: Optional[tuple[str, str, str]] = None
    if (
        isinstance(triple_raw, (list, tuple))
        and len(triple_raw) == 3
        and all(isinstance(p, str) and p.strip() for p in triple_raw)
    ):
        triple = (triple_raw[0], triple_raw[1], triple_raw[2])
    mtype: MType = (
        item.get("mtype") if item.get("mtype") in ("episodic", "semantic")
        else "episodic"
    )
    return _Candidate(content=content.strip(), triple=triple, mtype=mtype)


def _rate_salience(llm: ChatLLM, contents: list[str]) -> list[tuple[float, float]]:
    """ONE batched salience+importance call; defaults fill any gaps.

    Returns clamped (salience in [0,1], importance in [1,10]) per content,
    same order. Parse failure or a short reply falls back to the defaults
    (0.5, 5.0) — fail-open so nothing is lost to a flaky rating call.
    """
    parsed: list = []
    try:
        raw = llm.complete(render_salience(contents), model_hint="fast")
        maybe = parse_json_block(raw)
        if isinstance(maybe, list):
            parsed = maybe
    except ValueError:
        parsed = []
    out: list[tuple[float, float]] = []
    for i in range(len(contents)):
        entry = parsed[i] if i < len(parsed) else None
        sal, imp = _DEFAULT_SALIENCE, _DEFAULT_IMPORTANCE
        if isinstance(entry, dict):
            sal = _as_float(entry.get("salience"), _DEFAULT_SALIENCE)
            imp = _as_float(entry.get("importance"), _DEFAULT_IMPORTANCE)
        out.append((_clamp(sal, 0.0, 1.0), _clamp(imp, 1.0, 10.0)))
    return out


def _reinforce(
    store: MemoryStore,
    memory_id: str,
    *,
    now: datetime,
    actor: str,
    source_ref: Optional[str],
    reason: str,
) -> None:
    """A duplicate observation is an implicit successful recall: apply an
    FSRS Good review to the existing memory instead of storing a new row."""
    mem = store.get(memory_id)
    if mem is None:  # pragma: no cover — caller just fetched this id
        return
    elapsed = mem.elapsed_days(now)
    before = (mem.difficulty, mem.stability)
    r_before = fsrs.retrievability(elapsed, mem.stability)
    d2, s2 = fsrs.review(mem.difficulty, mem.stability, elapsed, fsrs.Grade.GOOD)
    mem.difficulty, mem.stability = d2, s2
    mem.last_review_at = now
    mem.review_count += 1
    mem.last_accessed_at = now
    store.record_review(
        mem.id,
        kind="manual",
        grade=int(fsrs.Grade.GOOD),
        elapsed_days=elapsed,
        r_before=r_before,
        before=before,
        after=(d2, s2),
        at=now,
    )
    store.update(
        mem,
        actor=actor,
        action="dedup_reinforce",
        detail={"source_ref": source_ref, "reason": reason},
    )


def _triple_key(
    triple: Optional[tuple[str, str, str]],
) -> Optional[tuple[str, str, str]]:
    if triple is None:
        return None
    s, r, o = triple
    return (_norm(s), _norm(r), _norm(o))


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))
