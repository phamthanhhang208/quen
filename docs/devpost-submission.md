# Devpost submission text — copy-paste per field

**Project name:** Quên — trust-calibrated forgetting agent memory

**Tagline (one-liner):**
A Qwen-powered agent memory that knows what to forget, says how sure it is,
and verifies before it asserts.

---

## Project description (main text box)

### The problem

The sharpest failure mode of agent memory isn't forgetting too much — it's
**being confidently wrong from stale memory**. Stale memory rarely fails at
retrieval; it fails by making the agent act confidently on invalidated
assumptions. Memora/FAMA measured it: agents frequently reuse invalidated
memories.

### What Quên does

*Quên* (Vietnamese for "to forget", sounds like *Qwen*) treats **trust as a
runtime decision, not a stored property**:

- **Write the delta**: Qwen extracts facts and (s,r,o) triples; a salience
  gate skips what a base model already knows; confidence comes from source
  authority (PR > doc > chat).
- **Real forgetting math**: retention state is FSRS-4.5 (the
  spaced-repetition equations, imported verbatim). Reviews come from three
  places: use-in-answer judged good/bad, dream self-tests, and
  **verification outcomes** — the closed loop nobody else ships.
- **Dream consolidation**: episodics re-abstract into generalizations;
  contradictions resolve **deterministically first** (same-(s,r)-new-o slot
  rule for functional relations — zero LLM calls, fully auditable) with a
  Qwen NLI fallback where *augmentation never supersedes*. Unused memories
  evict (R<θ ∧ TTL ∧ not pinned). Tombstones only — never hard-delete.
- **Verify-before-answer**: at answer time,
  trust = f(confidence, freshness, status). Low-trust memories get checked
  against the live source *before* the agent asserts them: confirmed →
  confidence rises + FSRS good review; refuted → tombstoned on the spot +
  fail review; unverifiable → the answer hedges explicitly.
- **Measured honesty**: stated answer confidence is scored with
  freshness-stratified calibration (ECE), and the headline metric is FAMA =
  presence-of-valid ∧ absence-of-invalidated.

### Results (live on Qwen via Alibaba Cloud DashScope, canonical run v5)

- **Code-staleness probe (n=30)**: FAMA **0.933** [0.79, 0.98] vs 0.40
  append-only RAG / 0.37 full-context (paired McNemar 16–0, p = 3×10⁻⁵);
  absence of invalidated facts **1.00**; forgetting precision/recall
  **0.941/1.00**; confidence calibration ECE **0.116**.
- **LongMemEval (n=229, external anchor)**: Quên **beats turn-level
  append-only RAG overall — McNemar 49–17, p = 1×10⁻⁴** (LLM-judge
  scoring, the benchmark's own protocol, applied to every config
  symmetrically; conservative exact-match reported alongside).
  Knowledge-update 0.653 vs 0.569; temporal reasoning **0.409 vs 0.189**
  (31–3, p = 1×10⁻⁶) — and from 209 tokens/query vs 293. Abstention
  trails (0.733 vs 0.800) and we report that too.
- **Paraphrase-frozen probe** (untouched holdout): 0.90 — the mechanism,
  not our phrasing, carries the result. Model-generation stability: the
  qwen3.5-generation canonical run scored the same probe 0.933.
- Budget curve: baselines flat at every budget (their failures are trust
  failures); Quên 0.83 at a 30-token budget, 0.97 at 300.
- The whole canonical live eval cost ≈ **$1.8** (usage counters × DashScope
  pricing — the cost table is in the README).
- We also ran an adversarial audit of our own algorithms — 11 biases found,
  fixed, and regression-tested (over-forgetting multi-valued facts, recency
  bias in trust, eviction starvation, strawman baselines, self-serving
  scoring…). The full table is in the README.

### How it's built (all Qwen, all Alibaba Cloud)

`qwen-flash` (extraction/salience/judging) + `qwen3.5-plus`
(reader/NLI/re-abstraction) + `text-embedding-v4`, all through one client
(`alibaba_client.py`) on the DashScope international endpoint. Backend =
FastAPI + SQLite (WAL, audit-logged, tombstones only) on an **Alibaba Cloud
ECS** instance that also serves the React dashboard same-origin. A thin
**FastMCP server** (8 tools — remember/recall/ask/judge/dream/verify_hint/
pin/inspect — closing both the per-memory verification loop and the
per-answer retention loop) drops the same engine into any MCP harness, and
a **Claude Code hooks kit** (`integrations/claude-code/`) loads memories at
SessionStart and captures the conversation at SessionEnd/PreCompact.
Everything — 186 tests and the full eval — also runs 100% offline
(deterministic scripted LLM + hashing embedder), so judges can reproduce
without a key.

### What's next

Temporal-hierarchical digests (TiMem-style graduated compression), a
Letta-style sleep-time dream scheduler, FSRS weight re-fit from the
calibration events the store already records, and context-cache-aware
prompt ordering (DashScope bills cached prefixes at ~10% of fresh input —
our usage counters already track the hits). Full roadmap:
`docs/ROADMAP.md`. Recent flag-gated additions already measured offline:
compact trust tags (−35% delivered tokens/query at identical FAMA) and
spaced self-test scheduling (52-week sim: 221 vs 427 tokens/query against
append-only from an active store 6× smaller, 203 evictions/yr, stale-free
rate 1.0).

### Links

- Live demo (Alibaba Cloud ECS, Singapore): http://47.236.141.124/
- Repo: https://github.com/phamthanhhang208/quen
  (branch `claude/quen-memory-agent-spec-p3izss`)
- Demo video: <YouTube link — điền sau khi upload>

---

## "Built with" tags

`python` · `fastapi` · `qwen` · `alibaba-cloud` · `dashscope` · `sqlite` ·
`react` · `typescript` · `vite` · `mcp` · `fsrs`

## Testing instructions for judges (if the form asks)

1. Open http://47.236.141.124/ — the dashboard is pre-seeded with an 80-day
   narrative: watch retrievability decay in **Memories**, the deterministic
   supersession + eviction in **Dream log**, and the flagship **Recall
   trace** with live verification (confirmed + refuted) and the
   counterfactual toggle showing what a plain RAG would have injected.
2. Reproduce locally with zero keys: `uv venv .venv && uv pip install -e
   ".[dev]" && pytest` (186 offline tests), then
   `python scripts/seed_demo.py` + `quen-api` + `cd dashboard && npm run
   dev`.
3. Live mode: set `DASHSCOPE_API_KEY` in `.env`; every eval script takes
   `--live`.
