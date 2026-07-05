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
    return get_engine().recall_dict(query, token_budget=token_budget)


@mcp.tool()
def ask(query: str, token_budget: Optional[int] = None) -> dict:
    """Answer a question from memory through the full trust gate (low-trust
    memories get verified before the answer when a verifier is wired, stale
    facts are hedged, nothing relevant → abstains). Returns a trace_id —
    when you later learn whether the answer was right, report it via
    judge(trace_id, correct) so the outcome feeds retention. recall +
    verify_hint close the per-memory loop; ask + judge close the per-answer
    loop."""
    r = get_engine().ask(query, token_budget=token_budget)
    return {
        "trace_id": r.trace_id,
        "answer": r.answer,
        "answer_confidence": r.answer_confidence,
        "abstained": r.abstained,
    }


@mcp.tool()
def judge(trace_id: str, correct: bool) -> dict:
    """Close the per-answer loop: the memories that contributed to the
    answer get an FSRS use-judged review (good/fail) and the stated
    confidence lands in the calibration record."""
    get_engine().judge_answer(trace_id, correct)
    return {"trace_id": trace_id, "judged_correct": correct}


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
    q: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """List memories with retention/trust state (dashboard data): status,
    importance, difficulty, stability, confidence, freshness, validity.
    `q` filters by content substring."""
    return get_engine().inspect(status=status, mtype=mtype, q=q, limit=limit)


def main() -> None:  # quen-mcp console script (stdio transport)
    mcp.run()


if __name__ == "__main__":
    main()
