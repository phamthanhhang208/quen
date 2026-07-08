"""Retrieval — relevance-dominant, token-bounded, active-only (spec §4.4).

``rank = w_rel * (max(0, cosine) + lexical identifier bonus) + w_r * R +
w_i * importance/10`` with relevance dominant, so a decayed-but-relevant
memory beats a fresh irrelevant one (relevance-beats-decay). The lexical
bonus recovers exact identifier matches (MAX_UPLOAD_MB, camelCase names)
that dense embeddings smooth over — the classic hybrid-retrieval gap for
technical corpora. Results greedily fill a token budget (skip-and-continue:
an oversized item never blocks smaller ones), but ONLY memories clearing an
inclusion relevance floor: padding the budget with irrelevant memories both
pollutes the reader and — worse — used to touch their access time on every
ask, starving eviction's unaccessed-past-TTL condition (a 52-week simulation
evicted exactly nothing).

Retrieval is NOT an FSRS review — the only side effect is a
``last_accessed_at`` touch on the memories actually used. Reviews come
from answer judging, self-tests and verification (spec §4.5).

Each result also carries the trace panels the dashboard shows: the
excluded-relevant list ("not used — superseded by X on DATE") and the
counterfactual pick (what an append-only RAG would have injected).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quen.config import QuenConfig
from quen.embeddings import Embedder, cosine
from quen.fsrs import retrievability
from quen.models import MemoryItem
from quen.store import MemoryStore
from quen.trust import trust_score

# bare numbers count too: "132 points" vs "132 meeples" is exactly the
# distractor collision dense embeddings smooth over (KU failure audit)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|[0-9][0-9.]*")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")


def _identifier_tokens(text: str) -> set[str]:
    """Code-shaped tokens (camelCase / snake_case / dotted / digit-bearing) —
    the exact-match signal dense embeddings smooth over."""
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = raw.strip(".")
        if len(token) < 3:
            continue
        if "_" in token or "." in token or _CAMEL_RE.search(token) \
                or any(c.isdigit() for c in token):
            out.add(token.casefold())
    return out

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
    lexical: float            # exact identifier-overlap share with the query
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

    # Pre-cut to the candidate pool by cosine (deterministic tie-breaks),
    # UNIONED with exact-identifier hits: an identifier match outside the
    # cosine top-k would otherwise never even be scored — the lexical bonus
    # can't rescue what the pre-cut already dropped.
    query_idents = _identifier_tokens(query)
    ranked = sorted(
        ((mem, cosine(query_embedding, mem.embedding)) for mem in pool),
        key=lambda mc: (-mc[1], -mc[0].created_at.timestamp(), mc[0].id),
    )
    by_cosine = ranked[: cfg.candidate_pool]
    if query_idents:
        seen = {mem.id for mem, _ in by_cosine}
        by_cosine += [
            (mem, cos)
            for mem, cos in ranked[cfg.candidate_pool :]
            if mem.id not in seen
            and query_idents & _identifier_tokens(mem.content)
        ][: cfg.candidate_pool // 2]

    scored = [
        _score(mem, cos, query_idents=query_idents, now=now, cfg=cfg)
        for mem, cos in by_cosine
    ]
    scored.sort(key=lambda sm: (-sm.score, -sm.memory.created_at.timestamp(), sm.memory.id))

    # Greedy fill, skip-and-continue: an oversized item is skipped, scanning
    # continues — the budget cap is a strict invariant. Items below the
    # inclusion floor never enter (see module docstring: reader pollution +
    # eviction starvation).
    used: list[ScoredMemory] = []
    remaining = token_budget
    for sm in scored:
        if sm.relevance + sm.lexical < cfg.include_relevance_floor:
            continue
        cost = sm.tokens + cfg.per_memory_overhead_tokens
        if cost <= remaining:
            used.append(sm)
            remaining -= cost
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


def _score(
    mem: MemoryItem,
    cos: float,
    *,
    query_idents: set[str],
    now: datetime,
    cfg: QuenConfig,
) -> ScoredMemory:
    relevance = max(0.0, cos)
    lexical = 0.0
    if query_idents:
        hits = query_idents & _identifier_tokens(mem.content)
        lexical = len(hits) / len(query_idents)
    r = retrievability(mem.elapsed_days(now), mem.stability)
    importance_norm = mem.importance / 10.0
    score = (
        cfg.w_relevance * relevance
        + cfg.w_lexical * lexical
        + cfg.w_retrievability * r
        + cfg.w_importance * importance_norm
    )
    return ScoredMemory(
        memory=mem,
        relevance=relevance,
        lexical=lexical,
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
