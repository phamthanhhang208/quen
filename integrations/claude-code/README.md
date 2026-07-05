# Quên × Claude Code — lifecycle hooks

Give Claude Code persistent, trust-calibrated project memory: relevant
memories load at session start (with trust tags and NEEDS-VERIFICATION
flags), and the conversation tail is captured back into Quên at session end
and before context compaction — the engine's extraction/salience/dedup
pipeline decides what's worth keeping, the hook stays dumb. Same division
of labor as the [Mem0 × Claude Code
integration](https://docs.mem0.ai/integrations/claude-code); here the
memory layer is yours, local, and auditable.

## Setup (2 minutes)

1. Run the API next to your project (offline mode is fine and free):

   ```bash
   QUEN_OFFLINE=1 QUEN_DB_PATH=data/quen.db .venv/bin/quen-api   # :8000
   ```

2. Merge `settings.example.json` into your project's
   `.claude/settings.json` (or `~/.claude/settings.json` for all projects).

3. That's it. New sessions start with a "Persistent project memory" block;
   session ends and compactions feed the store. For the full loop, also
   mount the MCP server (`quen-mcp`) so the agent can `verify_hint` the
   flagged memories and `judge` answers — hooks handle ambient
   load/capture, MCP handles the deliberate calls.

Env knobs: `QUEN_API_BASE` (default `http://localhost:8000`),
`QUEN_HOOK_BUDGET` (recall budget, default 600), `QUEN_HOOK_TAIL_CHARS`
(capture tail size, default 4000).

Hooks are **fail-silent by design**: if the API is down they exit 0 and the
session proceeds without memory. A memory layer must never break the tool
it serves.

## Smoke test

```bash
QUEN_OFFLINE=1 QUEN_DB_PATH=data/quen.db .venv/bin/quen-api &
curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text":"Project convention: we use uv and pytest for the dev setup.","source_kind":"doc"}'

echo '{"session_id":"s1","cwd":"'$PWD'"}' \
  | python3 integrations/claude-code/quen_hook.py session-start
# → {"hookSpecificOutput": {..., "additionalContext": "Persistent project memory..."}}
# (offline hashing embedder needs SOME lexical overlap with the probe query;
#  live Qwen embeddings match far more loosely)

printf '%s\n' '{"message":{"role":"user","content":"We moved uploads to uploadV2 today"}}' > /tmp/t.jsonl
echo '{"session_id":"s1","transcript_path":"/tmp/t.jsonl"}' \
  | python3 integrations/claude-code/quen_hook.py capture
curl -s 'localhost:8000/memory?q=uploadV2'                 # → ingested
```
