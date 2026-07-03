"""Memory schema (spec §4.2) and small shared value types.

All datetimes are timezone-aware UTC. Memories are never hard-deleted:
``status`` transitions active → deprecated | superseded are tombstones.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

MType = Literal["episodic", "semantic"]
Status = Literal["active", "deprecated", "superseded"]
VerificationOutcome = Literal["confirmed", "refuted", "unverifiable"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryItem:
    id: str
    content: str
    content_verbatim: str
    triple: Optional[tuple[str, str, str]]  # (subject, relation, object) if slot-extractable
    embedding: list[float]
    mtype: MType
    importance: float                        # 1–10 at write
    salience: float                          # 0–1 novelty vs model prior (write gate)
    # retention (FSRS)
    difficulty: float
    stability: float
    last_review_at: datetime
    review_count: int
    pinned: bool
    # bi-temporal validity
    valid_from: datetime
    valid_to: Optional[datetime]
    status: Status
    superseded_by: Optional[str]
    # trust
    confidence: float                        # 0–1; ↓ when challenged, ↑ when verified
    last_verified_at: Optional[datetime]
    source_ref: Optional[str]
    created_at: datetime
    # bookkeeping (extensions justified by §4.5 TTL and §4.6 provenance)
    last_accessed_at: datetime = field(default_factory=utcnow)
    provenance: list[str] = field(default_factory=list)  # source memory ids for generalizations

    # ---- derived-on-the-fly helpers ----

    def elapsed_days(self, now: Optional[datetime] = None) -> float:
        now = now or utcnow()
        return max(0.0, (now - self.last_review_at).total_seconds() / 86400.0)

    def freshness_days(self, now: Optional[datetime] = None) -> float:
        """Age of the evidence backing this memory: since last verification if
        any, else since the fact became valid."""
        now = now or utcnow()
        anchor = self.last_verified_at or self.valid_from
        return max(0.0, (now - anchor).total_seconds() / 86400.0)

    @property
    def triple_key(self) -> Optional[tuple[str, str, str]]:
        if self.triple is None:
            return None
        s, r, o = self.triple
        return (_norm(s), _norm(r), _norm(o))

    @property
    def slot_key(self) -> Optional[tuple[str, str]]:
        """(subject, relation) — the slot the deterministic supersession rule keys on."""
        if self.triple is None:
            return None
        s, r, _ = self.triple
        return (_norm(s), _norm(r))


# Leading noise words that make "the team" and "team" different slots —
# live extractors emit both phrasings for the same subject.
_LEAD_ARTICLES = frozenset({"the", "a", "an", "our", "this", "that", "its", "my"})
# Copulas at the edge of a relation: "stored in" vs "are stored in".
_EDGE_COPULAS = frozenset({"is", "are", "was", "were", "be", "been", "being"})


def _norm(part: str) -> str:
    """Canonicalize a triple part: casefold, collapse whitespace, strip
    punctuation edges, drop leading articles, and trim edge copulas (kept if
    they are the entire relation, e.g. plain "is")."""
    tokens = part.strip().casefold().split()
    tokens = [t.strip(".,;:!?'\"`") for t in tokens]
    tokens = [t for t in tokens if t]
    while len(tokens) > 1 and tokens[0] in _LEAD_ARTICLES:
        tokens = tokens[1:]
    # leading copula only: unifies "stored in"/"are stored in". A TRAILING
    # copula is the head of relations like "default branch is" — kept.
    while len(tokens) > 1 and tokens[0] in _EDGE_COPULAS:
        tokens = tokens[1:]
    return " ".join(tokens)


def make_memory(
    content: str,
    *,
    embedding: list[float],
    mtype: MType = "episodic",
    triple: Optional[tuple[str, str, str]] = None,
    importance: float = 5.0,
    salience: float = 0.5,
    difficulty: float = 5.0,
    stability: float = 1.0,
    confidence: float = 0.7,
    source_ref: Optional[str] = None,
    valid_from: Optional[datetime] = None,
    pinned: bool = False,
    provenance: Optional[list[str]] = None,
    now: Optional[datetime] = None,
) -> MemoryItem:
    now = now or utcnow()
    return MemoryItem(
        id=new_id(),
        content=content,
        content_verbatim=content,
        triple=triple,
        embedding=embedding,
        mtype=mtype,
        importance=importance,
        salience=salience,
        difficulty=difficulty,
        stability=stability,
        last_review_at=now,
        review_count=0,
        pinned=pinned,
        valid_from=valid_from or now,
        valid_to=None,
        status="active",
        superseded_by=None,
        confidence=confidence,
        last_verified_at=None,
        source_ref=source_ref,
        created_at=now,
        last_accessed_at=now,
        provenance=provenance or [],
    )
