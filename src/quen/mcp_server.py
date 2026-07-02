"""Quên as an MCP server — a thin FastMCP wrapper over the engine (spec §5).

Any MCP client (Claude Code, Cursor, OpenAI Agents SDK, LangGraph, ...) can
mount this and get trust-calibrated memory. The engine still talks only to
Qwen on Alibaba Cloud DashScope internally.

Note: `recall` returns memories WITH trust fields but does NOT verify — the
harness holds the live tools (grep, file read, URL fetch), verifies on its
side, and reports the outcome via `verify_hint`, which closes the loop
(confidence update + FSRS review).
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from quen.engine import _event_dict, _snippet, get_engine
from quen.retrieval import recall as _recall
from quen.store import iso

mcp = FastMCP("quen")


@mcp.tool()
def remember(
    observation: str,
    source_ref: Optional[str] = None,
    source_kind: str = "chat",
) -> dict:
    """Store an observation: facts are extracted, salience-gated, deduped
    (repeats reinforce retention instead of duplicating), and written with
    confidence derived from source authority (pr/commit > doc > chat)."""
    engine = get_engine()
    result = engine.ingest(
        observation, source_ref=source_ref, source_kind=source_kind
    )
    return {
        "stored": [{"id": m.id, "snippet": _snippet(m.content)} for m in result.stored],
        "reinforced": result.reinforced,
        "skipped": [{"content": s.content, "reason": s.reason} for s in result.skipped],
    }


@mcp.tool()
def recall(query: str, token_budget: int = 1500) -> dict:
    """Retrieve relevant active memories within a token budget. Every memory
    carries trust fields (confidence, freshness_days, retrievability, trust) —
    treat low-trust memories as hypotheses: verify them against the live
    source before acting, then report the outcome via verify_hint."""
    engine = get_engine()
    now = engine.clock()
    rr = _recall(
        query,
        token_budget=token_budget,
        store=engine.store,
        embedder=engine.embedder,
        cfg=engine.cfg,
        now=now,
    )
    return {
        "memories": [
            {
                "id": sm.memory.id,
                "content": sm.memory.content,
                "source_ref": sm.memory.source_ref,
                "tokens": sm.tokens,
                "score": round(sm.score, 4),
                "trust": round(sm.trust, 4),
                "confidence": round(sm.confidence, 4),
                "freshness_days": round(sm.freshness_days, 2),
                "retrievability": round(sm.retrievability, 4),
                "last_verified_at": iso(sm.memory.last_verified_at),
                "needs_verification": sm.trust < engine.cfg.trust_threshold,
            }
            for sm in rr.used
        ],
        "excluded_relevant": rr.excluded_relevant,
        "tokens_used": rr.tokens_used,
        "token_budget": rr.token_budget,
    }


@mcp.tool()
def dream() -> dict:
    """Run a consolidation pass: re-abstract episodics into generalizations,
    self-test retention, supersede contradicted facts (deterministic slot rule
    first, NLI fallback), evict decayed memories, compress."""
    report = get_engine().dream()
    return {"run_id": report.run_id, "journal": report.journal, "stats": report.stats}


@mcp.tool()
def verify_hint(memory_id: str, result: str, evidence: Optional[str] = None) -> dict:
    """Report a verification outcome you observed against the live source
    ('confirmed' | 'refuted' | 'unverifiable'). The engine closes the loop:
    confirmed bumps confidence + reinforces retention (FSRS good review);
    refuted invalidates the memory now (tombstone + fail review)."""
    event = get_engine().verify_hint(memory_id, result, evidence)
    return _event_dict(event)


@mcp.tool()
def pin(memory_id: str, pinned: bool = True) -> dict:
    """Pin a memory: it will never auto-evict and can only be superseded at a
    higher evidence bar."""
    get_engine().pin(memory_id, pinned)
    return {"id": memory_id, "pinned": pinned}


@mcp.tool()
def inspect(
    status: Optional[str] = None,
    mtype: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """List memories with retention/trust state (dashboard data): status,
    importance, difficulty, stability, confidence, freshness, validity."""
    return get_engine().inspect(status=status, mtype=mtype, limit=limit)


def main() -> None:  # quen-mcp console script (stdio transport)
    mcp.run()


if __name__ == "__main__":
    main()
