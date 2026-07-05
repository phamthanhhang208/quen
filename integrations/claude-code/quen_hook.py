#!/usr/bin/env python3
"""Claude Code lifecycle hooks for Quên — stdlib only, fail-silent.

Wire-up (see settings.example.json): SessionStart loads relevant memories
into the session as additionalContext; SessionEnd and PreCompact capture
the conversation tail into Quên, where the engine's own extraction /
salience / dedup pipeline decides what is worth keeping — the hook stays
dumb. This is the lifecycle-hook division of labor popularized by the
Mem0 × Claude Code integration (docs.mem0.ai/integrations/claude-code).

A memory hook must never break a session: any failure (API down, bad
JSON, no transcript) exits 0 with no output.

Env: QUEN_API_BASE (default http://localhost:8000), QUEN_HOOK_BUDGET
(recall token budget, default 600), QUEN_HOOK_TAIL_CHARS (default 4000).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API_BASE = os.environ.get("QUEN_API_BASE", "http://localhost:8000")
BUDGET = int(os.environ.get("QUEN_HOOK_BUDGET", "600"))
TAIL_CHARS = int(os.environ.get("QUEN_HOOK_TAIL_CHARS", "4000"))


def _post_json(path: str, payload: dict, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_context(recall_payload: dict) -> str:
    """Render a recall payload as the additionalContext block: content plus
    the trust story, so the agent knows what to double-check (and has the
    memory ids to verify_hint / judge later over MCP)."""
    memories = recall_payload.get("memories") or []
    if not memories:
        return ""
    lines = [
        "Persistent project memory (Quên). Low-trust lines are hypotheses —",
        "verify against the live repo before relying on them:",
    ]
    for m in memories:
        flag = " NEEDS-VERIFICATION" if m.get("needs_verification") else ""
        lines.append(
            f"- {m['content']}  "
            f"[id={m['id']} trust={m['trust']:.2f} "
            f"age={m['freshness_days']:.0f}d{flag}]"
        )
    return "\n".join(lines)


def tail_transcript(path: str, limit: int = TAIL_CHARS) -> str:
    """Last user/assistant text turns of a Claude Code transcript (JSONL),
    newest last, bounded to `limit` chars."""
    turns: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = row.get("message") or {}
            role = msg.get("role") or row.get("type")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if content and isinstance(content, str) and content.strip():
                turns.append(f"{role}: {content.strip()}")
    text = "\n".join(turns)
    return text[-limit:]


def cmd_session_start(hook_input: dict) -> None:
    project = os.path.basename(hook_input.get("cwd") or os.getcwd())
    payload = _post_json("/recall", {
        "query": f"{project} project conventions decisions preferences setup",
        "token_budget": BUDGET,
    })
    context = build_context(payload)
    if not context:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }))


def cmd_capture(hook_input: dict) -> None:
    path = hook_input.get("transcript_path")
    if not path or not os.path.exists(path):
        return
    tail = tail_transcript(path)
    if not tail.strip():
        return
    session = hook_input.get("session_id", "unknown")
    _post_json("/ingest", {
        "text": tail,
        "source_kind": "chat",
        "source_ref": f"claude-session:{session}",
    }, timeout=60.0)


def main(argv: list[str], stdin_text: str) -> int:
    try:
        hook_input = json.loads(stdin_text) if stdin_text.strip() else {}
        cmd = argv[1] if len(argv) > 1 else ""
        if cmd == "session-start":
            cmd_session_start(hook_input)
        elif cmd == "capture":
            cmd_capture(hook_input)
        return 0
    except Exception:
        return 0  # never break the session over a memory hook


if __name__ == "__main__":
    sys.exit(main(sys.argv, sys.stdin.read()))
