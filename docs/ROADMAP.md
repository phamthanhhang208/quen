# Roadmap — what's next after the hackathon

Ordered by expected impact on the trust-per-token frontier. Items marked
*(researched)* have literature anchors already vetted; the rest are
engineering debts we've disclosed in the README.

## Memory & consolidation

- **Temporal-hierarchical compression (TiMem-style)** *(researched:
  [arXiv 2601.02845](https://arxiv.org/abs/2601.02845))* — graduated
  digests instead of one-shot compression: recent memories stay verbatim,
  aging clusters roll up into weekly → monthly digests with provenance,
  and retrieval can descend from digest to detail. Replaces today's single
  `compress_min_chars` pass; biggest long-horizon token lever we haven't
  pulled.
- **Dream scheduler (Letta-style sleep-time compute)** *(researched:
  [letta.com/blog/sleep-time-compute](https://www.letta.com/blog/sleep-time-compute/))* —
  **step 1 shipped**: `maybe_dream()` write-count trigger
  (`dream_every_n_ingests`, used by the haystack eval at cadence 16).
  Remaining: idle triggers, a per-run consolidation budget, and
  re-abstract/compress scanning a dirty subset instead of the whole
  active set.
- **Retrieval-quality-aware confidence calibration** — measured on the
  haystack run: a map calibrated on oracle/probe workloads runs
  over-confident under heavy retrieval noise (accuracy 0.33–0.47 in the
  0.5–0.75 band). Condition the calibration on a noise signal (e.g.
  max relevance of used memories) instead of one global map.
- **FSRS weight re-fit** — the store already records every review and
  use-judged outcome in `calibration_events`; once enough accumulate,
  re-fit the 17 FSRS weights to *agent* forgetting instead of human
  flashcard priors, and report the before/after retention ECE.
- **Importance-weighted eviction** — a high-importance unpinned memory
  currently evicts on the same R/TTL rule as trivia; importance should
  stretch the TTL, not just the retrieval score.

## Harness integration

- **recall-side trace_id** — let a host correlate a specific recall with a
  later `judge`/`verify_hint` without going through `ask`.
- **Per-memory verifier suggestions** — `needs_verification` today is a
  boolean; recall should also say *how* to check (grep pattern, file path,
  URL) so the harness can verify mechanically.
- **MCP `traces` / `vitals` tools** — the REST API exposes them, MCP
  doesn't yet.
- **Hooks for more harnesses** — the Claude Code kit
  (`integrations/claude-code/`) generalizes: Cursor rules, OpenAI Agents
  SDK session callbacks, LangGraph checkpointers.

## Token economy

- **Context-cache-aware prompt ordering** *(researched: [DashScope Context
  Cache](https://www.alibabacloud.com/help/en/model-studio/context-cache))* —
  implicit prefix caching bills hits at ~10% of fresh input but needs a
  stable ≥1024-token prefix; at production budgets, ordering the stable
  instruction + pinned-memory block first makes consecutive asks cache-hit.
  `usage_summary()` already counts `cached_tokens` to measure it.
- **Real tokenizer** — `len//4` under-counts non-Latin scripts (disclosed
  in the README); swap in a proper tokenizer and re-derive
  `per_memory_overhead_tokens` from the actually-rendered trust tag.
- **Read-path compression** — long memories are packed verbatim until the
  budget skips them; a budget-aware summarize-on-read tier (LLMLingua-2
  family) would fit more evidence per token.

## Trust

- **Learned relation functionality** — the functional-relation list behind
  deterministic supersession is hand-written; learn it from the audit log
  (which relations' supersessions later get confirmed vs augmented).
- **Verifier plug-ins** — beyond repo-grep: URL fetch, file mtime, test
  runner. Each maps naturally onto the existing `verifier_strength` table.
