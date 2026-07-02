# Quên — trust-calibrated forgetting agent memory (Qwen Cloud, Track 1)

Project name: Quên (display, with diacritic); ASCII identifiers: quen (repo, Python package, MCP server name).
Contribution = closed loop (verify→FSRS review) + measured epistemic honesty (FAMA + freshness-stratified
confidence calibration), NOT novel storage. All Qwen via Alibaba DashScope (alibaba_client.py); never another provider.

Mechanisms (see docs/SPEC.md §4):
- WRITE: extract facts, (s,r,o) triples where slot-like; SALIENCE gate (skip what a base model knows);
  dedup triple-first then cosine; confidence from source authority.
- RETENTION: FSRS v4.5 imported verbatim from open-spaced-repetition; reviews = use-judged-good/bad,
  self-test pass/fail, VERIFICATION pass/fail. DSR = eviction prior, not scheduler.
- RETRIEVAL: wrel*cosine + wr*R + wi*importance, wrel dominant (low-R relevant beats high-R irrelevant);
  token_budget greedy fill; active-only; return trust fields per memory.
- DREAM: re-abstract into GENERALIZATIONS; self-test; SUPERSESSION two-stage: deterministic same-(s,r)-new-o
  rule FIRST (no LLM), NLI fallback (entails|neutral|contradicts|augments; augmentation NEVER supersedes);
  confidence-gate; tombstone only; decay+evict (R<theta, TTL, not pinned); compress keep verbatim; audit log.
- TRUST GATE: trust=f(confidence,freshness,status); low-trust + verifier available -> VERIFY-BEFORE-ANSWER
  (grep/file/user-confirm): confirmed->assert+bump confidence+good review; refuted->supersede now+fail review;
  unverifiable->hedge explicitly. Stated answer_confidence must track trust. Never assert stale unverified
  memory with full assurance.

TDD adds: deterministic supersession, verification-updates-retention, relevance-beats-decay.
DoD adds: /memory/{id}/verify; FAMA scorer; abstention subset in eval; MCP server (remember/recall/dream/
verify_hint/pin/inspect) as a thin FastMCP wrapper — engine unchanged.

## Repo layout
- `src/quen/` — engine (models, fsrs, store, write_pipeline, retrieval, supersession, dream, trust,
  verifiers, engine facade), `api.py` (FastAPI), `mcp_server.py` (FastMCP), `alibaba_client.py` (THE proof artifact).
- `tests/` — pytest; everything runs offline (QUEN_OFFLINE=1, hashing embedder + scripted LLM).
- `eval/` — LongMemEval (KU/Temporal/Abstention), FAMA-scored staleness probe, calibrations.
- `dashboard/` — Vite + React 18 + TS + Tailwind + shadcn/ui + recharts; SWR polling; no theme.

## Dev
- `uv pip install -e ".[dev]"` then `pytest`.
- Tests must never hit the network: construct `QuenEngine` with `HashingEmbedder` and a scripted LLM.
- Never hard-delete memories: tombstone (status change) only. Keep `content_verbatim` on every compression.
