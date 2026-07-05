"""Claude Code hooks kit: pure-function units, no sockets (suite stays offline)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "integrations" / "claude-code"))

import quen_hook  # noqa: E402


RECALL_PAYLOAD = {
    "memories": [
        {"id": "m-1", "content": "The team fetches data via useQuery.",
         "trust": 0.91, "freshness_days": 2.0, "needs_verification": False},
        {"id": "m-2", "content": "MAX_UPLOAD_MB stays at 10 in src/config.ts.",
         "trust": 0.41, "freshness_days": 68.0, "needs_verification": True},
    ],
    "tokens_used": 30, "token_budget": 600, "excluded_relevant": [],
}


def test_build_context_carries_trust_story():
    ctx = quen_hook.build_context(RECALL_PAYLOAD)
    assert "useQuery" in ctx
    assert "id=m-2" in ctx  # ids present so the agent can verify_hint later
    assert "trust=0.41" in ctx
    assert "NEEDS-VERIFICATION" in ctx
    assert "hypotheses" in ctx  # instruction framing, not bare dump


def test_build_context_empty_store_is_silent():
    assert quen_hook.build_context({"memories": []}) == ""


def test_tail_transcript_reads_jsonl_turns(tmp_path):
    t = tmp_path / "t.jsonl"
    rows = [
        {"message": {"role": "user", "content": "We moved uploads to uploadV2."}},
        {"type": "tool_use", "message": {}},  # non-turn rows skipped
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Noted — uploadV2 it is."}]}},
        "not json at all",
    ]
    t.write_text("\n".join(
        r if isinstance(r, str) else json.dumps(r) for r in rows))
    tail = quen_hook.tail_transcript(str(t))
    assert "user: We moved uploads to uploadV2." in tail
    assert "assistant: Noted — uploadV2 it is." in tail


def test_tail_transcript_bounded(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps(
        {"message": {"role": "user", "content": "x" * 10000}}))
    assert len(quen_hook.tail_transcript(str(t), limit=500)) == 500


def test_main_never_raises_on_garbage():
    assert quen_hook.main(["quen_hook.py", "session-start"], "not-json{{") == 0
    assert quen_hook.main(["quen_hook.py", "capture"],
                          '{"transcript_path": "/nope/none.jsonl"}') == 0
    assert quen_hook.main(["quen_hook.py"], "{}") == 0
