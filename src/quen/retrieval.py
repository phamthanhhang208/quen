"""Retrieval — relevance-dominant, token-bounded, active-only (spec §4.4).

``rank = w_rel * max(0, cosine) + w_r * R + w_i * importance/10`` with
relevance dominant, so a decayed-but-relevant memory beats a fresh
irrelevant one (relevance-beats-decay). Results greedily fill a token
budget (skip-and-continue: an oversized item never blocks smaller ones).

Retrieval is NOT an FSRS review — the only side effect is a
``last_accessed_at`` touch on the memories actually used. Reviews come
from answer judging, self-tests and verification (spec §4.5).

Each result also carries the trace panels the dashboard shows: the
excluded-relevant list ("not used — superseded by X on DATE") and the
counterfactual pick (what an append-only RAG would have injected).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.fsrs import retrievability
from quen.models import MemoryItem
from quen.store import MemoryStore
from quen.trust import trust_score

_EXCLUDED_RELEVANCE_MIN = 0.25
_EXCLUDED_CAP = 5
_SNIPPET_LEN = 80
_ALL_LIMIT = 100_000


def estimate_tokens(text: str) -> int:
    """The single token estimator used everywhere: ~4 chars/token, min 1."""
    return max(1, len(text) // 4)


@dataclass
class ScoredMemory:
    """One retrieved memory with its full score breakdown and trust fields."""

    memory: MemoryItem
    relevance: float          # max(0, cosine(query, memory))
    retrievability: float     # FSRS R at `now`
    importance_norm: float    # importance / 10
    score: float
    tokens: int
    trust: float              # trust.trust_score(mem, now, cfg)
    confidence: float
    freshness_days: float


@dataclass
class RecallResult:
    """What recall used, what it deliberately skipped, and the counterfactual."""

    used: list[ScoredMemory]
    excluded_relevant: list[dict]
    counterfactual: list[dict]
    tokens_used: int
    token_budget: int
    query_embedding: list[float]


def recall(
    query: str,
    *,
    token_budget: int,
    store: MemoryStore,
    embedder: Embedder,
    cfg: QuenConfig,
    now: datetime,
    include_historical: bool = False,
) -> RecallResult:
    """Retrieve memories for `query` under a strict token budget.

    Active memories only unless ``include_historical`` (then deprecated and
    superseded memories may be used too — for explicitly historical
    questions). Candidates are pre-cut to ``cfg.candidate_pool`` by cosine,
    scored, then greedily packed by score with skip-and-continue so
    ``tokens_used <= token_budget`` always holds. Ties break
    deterministically: score desc, created_at desc, id asc.
    """
    query_embedding = embedder.embed([query])[0]

    pool = store.active()
    if include_historical:
        pool = (
            pool
            + store.list(status="deprecated", limit=_ALL_LIMIT)
            + store.list(status="superseded", limit=_ALL_LIMIT)
        )

    # Pre-cut to the candidate pool by cosine (deterministic tie-breaks).
    by_cosine = sorted(
        ((mem, cosine(query_embedding, mem.embedding)) for mem in pool),
        key=lambda mc: (-mc[1], -mc[0].created_at.timestamp(), mc[0].id),
    )[: cfg.candidate_pool]

    scored = [_score(mem, cos, now=now, cfg=cfg) for mem, cos in by_cosine]
    scored.sort(key=lambda sm: (-sm.score, -sm.memory.created_at.timestamp(), sm.memory.id))

    # Greedy fill, skip-and-continue: an oversized item is skipped, scanning
    # continues — the budget cap is a strict invariant.
    used: list[ScoredMemory] = []
    remaining = token_budget
    for sm in scored:
        if sm.tokens <= remaining:
            used.append(sm)
            remaining -= sm.tokens
    tokens_used = token_budget - remaining

    store.touch_access([sm.memory.id for sm in used], at=now)

    used_ids = {sm.memory.id for sm in used}
    excluded_relevant = _excluded_relevant(query_embedding, store, used_ids)
    counterfactual = _counterfactual(query_embedding, store, token_budget)

    return RecallResult(
        used=used,
        excluded_relevant=excluded_relevant,
        counterfactual=counterfactual,
        tokens_used=tokens_used,
        token_budget=token_budget,
        query_embedding=query_embedding,
    )


def _score(mem: MemoryItem, cos: float, *, now: datetime, cfg: QuenConfig) -> ScoredMemory:
    relevance = max(0.0, cos)
    r = retrievability(mem.elapsed_days(now), mem.stability)
    importance_norm = mem.importance / 10.0
    score = (
        cfg.w_relevance * relevance
        + cfg.w_retrievability * r
        + cfg.w_importance * importance_norm
    )
    return ScoredMemory(
        memory=mem,
        relevance=relevance,
        retrievability=r,
        importance_norm=importance_norm,
        score=score,
        tokens=estimate_tokens(mem.content),
        trust=trust_score(mem, now, cfg),
        confidence=mem.confidence,
        freshness_days=mem.freshness_days(now),
    )


def _excluded_relevant(
    query_embedding: list[float], store: MemoryStore, used_ids: set[str]
) -> list[dict]:
    """Tombstoned memories that WOULD have been relevant — shown, not used."""
    historical = store.list(status="deprecated", limit=_ALL_LIMIT) + store.list(
        status="superseded", limit=_ALL_LIMIT
    )
    relevant: list[tuple[MemoryItem, float]] = []
    for mem in historical:
        if mem.id in used_ids:
            continue
        rel = max(0.0, cosine(query_embedding, mem.embedding))
        if rel >= _EXCLUDED_RELEVANCE_MIN:
            relevant.append((mem, rel))
    relevant.sort(key=lambda mr: (-mr[1], -mr[0].created_at.timestamp(), mr[0].id))
    out: list[dict] = []
    for mem, rel in relevant[:_EXCLUDED_CAP]:
        if mem.status == "superseded":
            reason = f"superseded by {mem.superseded_by} on {_date_str(mem.valid_to)}"
        else:
            reason = f"deprecated (evicted) on {_date_str(mem.valid_to)}"
        out.append(
            {
                "memory_id": mem.id,
                "snippet": _snippet(mem.content),
                "relevance": rel,
                "reason": reason,
                "status": mem.status,
                "superseded_by": mem.superseded_by,
            }
        )
    return out


def _counterfactual(
    query_embedding: list[float], store: MemoryStore, token_budget: int
) -> list[dict]:
    """What an append-only RAG would inject: pure-cosine greedy fill over
    ALL memories regardless of status, to the same budget."""
    everything = store.list(limit=_ALL_LIMIT)
    ranked = sorted(
        ((mem, cosine(query_embedding, mem.embedding)) for mem in everything),
        key=lambda mc: (-mc[1], -mc[0].created_at.timestamp(), mc[0].id),
    )
    out: list[dict] = []
    remaining = token_budget
    for mem, _cos in ranked:
        tokens = estimate_tokens(mem.content)
        if tokens <= remaining:
            out.append(
                {"memory_id": mem.id, "snippet": _snippet(mem.content), "status": mem.status}
            )
            remaining -= tokens
    return out


def _snippet(text: str, length: int = _SNIPPET_LEN) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


def _date_str(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else "unknown"
