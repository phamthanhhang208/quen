"""Two-stage supersession (spec §4.6.3).

Stage 1 — deterministic slot rule (MemStrata-style): among ACTIVE slotted
memories sharing a normalized ``(subject, relation)`` slot with more than one
distinct normalized object, the newest ``valid_from`` wins and every other
distinct-object member is superseded. Zero LLM calls — immune to the
AUROC-0.59 embedding failure. Bi-temporally correct: ingest order is
irrelevant, only ``valid_from`` decides.

Stage 2 — NLI fallback for non-slotted contradictions: candidate pairs come
from entity-token overlap ∪ cosine kNN, slot-sharing pairs are excluded
(stage 1 owns those), and Qwen classifies
``entails | neutral | contradicts | augments``. Only a confident
"contradicts" acts; augmentation NEVER supersedes (the Buddy/Scout guard).

Supersession is a tombstone, never a delete: ``old.valid_to =
new.valid_from``, ``old.status = "superseded"``, ``old.superseded_by =
new.id`` — all through the audited ``MemoryStore.update`` path. Pinned
memories are supersedable only at a higher bar (spec §4.8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quen import llm as llm_mod
from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.llm import ChatLLM, parse_json_block
from quen.models import MemoryItem
from quen.store import MemoryStore

# NLI pairs evaluated per pass — cost cap.
MAX_NLI_PAIRS = 50
# Pinned memories need at least this NLI confidence to be superseded.
PINNED_NLI_BAR = 0.9

# Relation glue and function words — not entities for candidate pairing.
_STOPWORDS = frozenset(
    {
        "about", "after", "all", "also", "and", "any", "are", "because",
        "been", "before", "being", "between", "both", "but", "can",
        "could", "did", "does", "doing", "down", "during", "each", "few",
        "for", "from", "had", "has", "have", "her", "here", "him", "his",
        "how", "into", "its", "just", "may", "might", "more", "most",
        "must", "not", "now", "off", "once", "only", "onto", "other",
        "our", "out", "over", "own", "per", "same", "she", "should",
        "some", "such", "than", "that", "the", "their", "them", "then",
        "there", "they", "this", "too", "under", "until", "use", "used",
        "uses", "using", "very", "via", "was", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with",
        "would", "you", "your",
    }
)


@dataclass
class SupersessionEvent:
    old_id: str
    new_id: str | None
    rule: str                       # "deterministic" | "nli"
    label: str | None = None        # NLI label when rule == "nli"
    nli_confidence: float | None = None
    at: datetime | None = None


def supersede(
    old: MemoryItem,
    new: MemoryItem,
    *,
    store: MemoryStore,
    now: datetime,
    actor: str = "dream",
    rule: str = "deterministic",
    run_id: Optional[str] = None,
    label: Optional[str] = None,
    nli_confidence: Optional[float] = None,
) -> SupersessionEvent:
    """Tombstone ``old`` as superseded by ``new`` (spec §4.6.3).

    Sets ``old.valid_to = new.valid_from`` (bi-temporal close), flips status
    to "superseded" and records the successor id. The mutation goes through
    the audited ``MemoryStore.update`` path; ``new`` is untouched.
    """
    old.valid_to = new.valid_from
    old.status = "superseded"
    old.superseded_by = new.id
    detail: dict = {"rule": rule, "new_id": new.id, "run_id": run_id}
    if label is not None:
        detail["label"] = label
    if nli_confidence is not None:
        detail["nli_confidence"] = nli_confidence
    store.update(old, actor=actor, action="supersede", detail=detail)
    return SupersessionEvent(
        old_id=old.id,
        new_id=new.id,
        rule=rule,
        label=label,
        nli_confidence=nli_confidence,
        at=now,
    )


# ------------------------------------------------------- deterministic stage

def deterministic_pass(
    store: MemoryStore,
    *,
    now: datetime,
    actor: str = "dream",
    run_id: Optional[str] = None,
) -> list[SupersessionEvent]:
    """Same-(s,r)-slot, different-object rule — zero LLM calls.

    Groups ACTIVE slotted memories by normalized slot key. In any group with
    more than one distinct normalized object the winner is the member with
    the newest ``valid_from`` (ties: ``created_at`` desc, then ``confidence``
    desc, then ``id`` asc — fully deterministic); every member holding a
    DIFFERENT object is superseded by it. Members sharing the winner's
    object are duplicates, not contradictions, and stay untouched.

    Only ACTIVE memories are considered, so settled tombstones (their
    ``superseded_by`` chains) are never rewritten.
    """
    groups: dict[tuple[str, str], list[MemoryItem]] = {}
    for mem in store.active():
        key = mem.slot_key
        if key is not None:
            groups.setdefault(key, []).append(mem)

    events: list[SupersessionEvent] = []
    for key in sorted(groups):
        members = groups[key]
        objects = {m.triple_key[2] for m in members}  # type: ignore[index]
        if len(objects) < 2:
            continue  # duplicates only — nothing to reconcile
        winner = _slot_winner(members)
        winner_o = winner.triple_key[2]  # type: ignore[index]
        losers = [m for m in members if m.triple_key[2] != winner_o]  # type: ignore[index]
        for old in sorted(losers, key=lambda m: m.id):
            if old.pinned and winner.confidence < old.confidence:
                store.audit(
                    actor,
                    "supersede_blocked_pinned",
                    memory_id=old.id,
                    run_id=run_id,
                    detail={"new_id": winner.id, "rule": "deterministic"},
                    at=now,
                )
                continue
            events.append(
                supersede(
                    old,
                    winner,
                    store=store,
                    now=now,
                    actor=actor,
                    rule="deterministic",
                    run_id=run_id,
                )
            )
    return events


def _slot_winner(members: list[MemoryItem]) -> MemoryItem:
    """Newest ``valid_from``; ties broken by ``created_at`` desc, then
    ``confidence`` desc, then ``id`` asc (stable sort keeps id order)."""
    ordered = sorted(members, key=lambda m: m.id)
    ordered.sort(
        key=lambda m: (m.valid_from, m.created_at, m.confidence), reverse=True
    )
    return ordered[0]


# ---------------------------------------------------------- NLI fallback stage

def nli_pass(
    store: MemoryStore,
    llm: ChatLLM,
    embedder: Embedder,
    cfg: QuenConfig,
    *,
    now: datetime,
    actor: str = "dream",
    run_id: Optional[str] = None,
) -> list[SupersessionEvent]:
    """NLI fallback for non-slotted contradictions (spec §4.6.3).

    Candidate pairs among actives = entity-token overlap ∪ top
    ``cfg.nli_knn_k`` cosine neighbors per memory, minus pairs sharing a
    slot key (the deterministic stage owns those). Each pair is ordered
    A = older ``valid_from``, B = newer, evaluated at most once, and the
    pass is capped at ``MAX_NLI_PAIRS`` LLM calls. Only
    ``label == "contradicts"`` at or above ``cfg.nli_confidence_gate``
    supersedes A by B; "augments" keeps both and is merely audited —
    augmentation is NOT contradiction.

    ``embedder`` is accepted for signature parity with the dream pipeline;
    kNN runs on the embeddings already stored on each memory.
    """
    del embedder  # stored embeddings are authoritative
    actives = store.active()
    events: list[SupersessionEvent] = []
    dead: set[str] = set()

    for a, b in _candidate_pairs(actives, cfg):
        if a.id in dead or b.id in dead:
            continue  # superseded earlier in this same pass
        raw = llm.complete(llm_mod.render_nli(a.content, b.content), model_hint="fast")
        try:
            parsed = parse_json_block(raw)
            label = str(parsed["label"]).strip().casefold()
            confidence = float(parsed["confidence"])
        except (ValueError, KeyError, TypeError):
            store.audit(
                actor,
                "error",
                memory_id=a.id,
                run_id=run_id,
                detail={"phase": "nli", "other": b.id, "raw": raw[:200]},
                at=now,
            )
            continue
        if label == "augments":
            # Buddy/Scout guard: augmentation adds a sibling, never DELETE+ADD.
            store.audit(
                actor,
                "augment_noted",
                memory_id=a.id,
                run_id=run_id,
                detail={"other": b.id},
                at=now,
            )
            continue
        if label != "contradicts" or confidence < cfg.nli_confidence_gate:
            continue  # entails / neutral / under-confident contradiction
        if a.pinned and confidence < PINNED_NLI_BAR:
            store.audit(
                actor,
                "supersede_blocked_pinned",
                memory_id=a.id,
                run_id=run_id,
                detail={"new_id": b.id, "rule": "nli"},
                at=now,
            )
            continue
        events.append(
            supersede(
                a,
                b,
                store=store,
                now=now,
                actor=actor,
                rule="nli",
                run_id=run_id,
                label=label,
                nli_confidence=confidence,
            )
        )
        dead.add(a.id)
    return events


def _entity_tokens(text: str) -> set[str]:
    """Non-stopword tokens of length >= 3 — crude entity fingerprint."""
    return {
        t
        for t in re.findall(r"[a-z0-9_]+", text.casefold())
        if len(t) >= 3 and t not in _STOPWORDS
    }


def _candidate_pairs(
    actives: list[MemoryItem], cfg: QuenConfig
) -> list[tuple[MemoryItem, MemoryItem]]:
    """Deterministic NLI candidate pairs: entity overlap ∪ cosine kNN,
    slot-sharing pairs excluded, each pair once, oldest-``valid_from``
    first within the pair, capped at ``MAX_NLI_PAIRS``."""
    order = sorted(actives, key=lambda m: (m.created_at, m.id))
    tokens = [_entity_tokens(m.content) for m in order]
    keys: set[tuple[int, int]] = set()

    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            if a.slot_key is not None and a.slot_key == b.slot_key:
                continue  # deterministic rule owns same-slot pairs
            if tokens[i] & tokens[j]:
                keys.add((i, j))

    for i, a in enumerate(order):
        sims: list[tuple[float, str, int]] = []
        for j, b in enumerate(order):
            if i == j or (a.slot_key is not None and a.slot_key == b.slot_key):
                continue
            sim = cosine(a.embedding, b.embedding)
            if sim > 0.0:  # HashingEmbedder cosine can go negative — clamp out
                sims.append((-sim, b.id, j))
        sims.sort()
        for _, _, j in sims[: cfg.nli_knn_k]:
            keys.add((min(i, j), max(i, j)))

    pairs: list[tuple[MemoryItem, MemoryItem]] = []
    for i, j in sorted(keys):
        a, b = order[i], order[j]
        if (b.valid_from, b.created_at, b.id) < (a.valid_from, a.created_at, a.id):
            a, b = b, a  # A = older valid_from, B = newer
        pairs.append((a, b))
    return pairs[:MAX_NLI_PAIRS]
