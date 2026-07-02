# Quên

> **Quên** *(sounds like **Qwen**; Vietnamese for "to forget")* — a Qwen-powered
> memory that knows what to forget — and how sure to sound.
>
> *Also: Quen is the protective shield sign in The Witcher — it shields your
> agent from stale memory.*

**Trust-calibrated forgetting: an agent memory that knows what to forget, says
how sure it is, and verifies before it asserts.**

Built for the **Qwen Cloud Global AI Hackathon · Track 1: MemoryAgent**.
All LLM and embedding calls run on **Qwen via Alibaba Cloud DashScope**
([`src/quen/alibaba_client.py`](src/quen/alibaba_client.py) is the single
network path — no other provider, ever).

> ⚠️ **Not affiliated.** Quên is a community homage to the Qwen name; it is not
> affiliated with or endorsed by the Qwen / Alibaba Cloud teams.

---

## Why

The sharpest failure mode of agent memory isn't *forgetting too much* — it's
**being confidently wrong from stale memory**. Stale memory rarely fails at
retrieval; it fails by making the agent act confidently on invalidated
assumptions (arXiv 2605.26112). Memora/FAMA (arXiv 2604.20006) measured it:
agents *frequently reuse invalidated memories*.

Quên's answer, in one line:

> **Trust is a runtime decision, not a stored property.** Every answer's stated
> confidence tracks the memory's validity and freshness; low-trust memories
> trigger **verify-before-answer** against the live source; and the
> verification outcome **feeds back into retention as an FSRS review** —
> verification reinforces or invalidates memory. Then we *measure* the
> epistemic honesty: FAMA on a code-staleness probe, plus retention
> calibration and **freshness-stratified confidence calibration**.

To our knowledge no system closes the loop *verification → retention update*,
and none scores hedging against memory freshness. Everything else here —
decay-based forgetting, bi-temporal supersession, confidence-carrying memory —
exists in the literature and is cited below; we claim the closed loop and the
measurement, not the parts.

## How it works

```mermaid
flowchart LR
    subgraph write [WRITE — store the delta]
        W1[extract facts + s,r,o triples] --> W2[salience gate] --> W3[dedup: triple-first, cosine second] --> W4[init FSRS D/S + confidence from source authority]
    end
    subgraph store [MEMORY - SQLite, tombstones only]
        M[(episodic / semantic\nDSR + bi-temporal validity + trust)]
    end
    subgraph dream [DREAM — async consolidation]
        D1[re-abstract into generalizations] --> D2[self-test = FSRS review] --> D3[supersede: deterministic s,r rule FIRST, NLI fallback] --> D4[decay + evict R<θ ∧ TTL ∧ ¬pinned] --> D5[compress, keep verbatim]
    end
    subgraph ask [ASK — trust is a runtime decision]
        A1[relevance-dominant, token-budgeted recall] --> A2{trust = f confidence, freshness, status}
        A2 -- high --> A3[answer, confidence stated]
        A2 -- low + verifier --> A4[VERIFY against live source]
        A4 -- confirmed --> A5[assert + confidence↑ + FSRS good review]
        A4 -- refuted --> A6[supersede NOW + FSRS fail review + answer from corrected state]
        A4 -- unverifiable --> A7[hedge explicitly]
    end
    write --> store --> ask
    store <--> dream
```

The load-bearing mechanisms:

- **Retention = FSRS-4.5**, equations imported verbatim from
  [open-spaced-repetition](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler).
  DSR state is an *eviction/ranking prior*, not a scheduler. Reviews come from
  three places: use-in-answer judged good/bad, dream self-tests, and
  **verification outcomes** (the closed loop).
- **Supersession is deterministic first** (MemStrata-style same-(s,r)-new-o
  slot rule, zero LLM calls — embeddings *cannot* detect contradiction,
  AUROC ≈ 0.59), with an LLM-NLI fallback where `augments` **never**
  supersedes (the Buddy/Scout trap). Tombstones only; bi-temporal
  `valid_from/valid_to`; full audit log.
- **Retrieval is relevance-dominant and token-budgeted**: a low-R but highly
  relevant memory surfaces at full strength — decay never suppresses
  relevance. Every recall returns per-memory trust fields, an
  excluded-relevant list ("not used — superseded by X"), and the
  **counterfactual** an append-only RAG would have injected.
- **The trust gate**: `trust = confidence × 2^(−freshness/half-life)` (zero for
  tombstones). Below threshold, with a verifier available (repo-grep for the
  demo), the engine checks the claim against the live source *before*
  answering — and never asserts a stale, unverified memory with full
  assurance. When nothing relevant survives, it **abstains**.

## Results (offline dry-run — deterministic pipeline validation)

The whole eval runs **without any API key** (hashing embedder + scripted
extractive reader) — these numbers validate the *memory mechanics* under an
identical reader/budget, not LLM quality. Re-run everything with
`--live` on Qwen for the submission numbers.

**Code-staleness probe** (25 cases, invalidation by evidence — refactor PRs,
implicit reversals; ~20% augmentation distractors; abstention cases;
`live_files` cases exercising verify-before-answer). Headline metric:
**FAMA** = presence-of-valid ∧ absence-of-invalidated (Memora, 2604.20006):

| config | FAMA | presence | absence | tokens/query |
|---|---|---|---|---|
| append-only RAG | 0.28 | 0.88 | 0.40 | 16.2 |
| full-context (truncate oldest) | 0.28 | 0.88 | 0.40 | 16.2 |
| **Quên (ours)** | **1.00** | 1.00 | 1.00 | **11.1** |
| ours − verify (ablation) | 0.92 | 1.00 | 0.92 | 12.0 |

The ablation gap (0.92 → 1.00) is exactly the two cases where the *only*
signal that a memory went stale is the live repo — no ingested event ever
contradicted it. That's the verify-before-answer beat.

Charts: [`eval/out/`](eval/out) — FAMA by config, accuracy-vs-budget curve,
retention calibration (predicted R vs empirical recall), and the
**confidence-calibration-by-freshness** chart.

```bash
.venv/bin/python eval/run_probe.py            # probe, offline (default)
.venv/bin/python eval/run_probe.py --live     # probe on Qwen via DashScope
.venv/bin/python eval/run_longmemeval.py      # LongMemEval KU/Temporal/Abstention
.venv/bin/python eval/budget_curve.py
.venv/bin/python eval/calibration.py --db data/demo.db
.venv/bin/python eval/charts.py
```

## Quickstart

```bash
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/pytest                    # 137 offline, deterministic tests

# seed the full demo narrative (no API key needed) and serve it
.venv/bin/python scripts/seed_demo.py
QUEN_OFFLINE=1 QUEN_DB_PATH=data/demo.db \
  QUEN_VERIFY_REPO=scripts/demo_repo .venv/bin/quen-api    # :8000

cd dashboard && npm install && npm run dev                  # :5173
```

Live mode: copy `.env.example` → `.env`, set `DASHSCOPE_API_KEY`
(international endpoint: `dashscope-intl.aliyuncs.com`), unset `QUEN_OFFLINE`.
Smoke test the wiring: `.venv/bin/python -m quen.alibaba_client` → `quen-ok`.
Models: `qwen3.5-plus` (reader/NLI/reflection), `qwen-flash`
(extraction/judging), `text-embedding-v4`.

### The demo narrative (what the seeder builds)

180 days of team memory: `useApi` is learned, generalized in a dream pass,
then a migration PR lands → the **deterministic slot rule** supersedes it with
`useQuery` (no LLM call, inspectable in the Dream log). Trivia decays and is
**evicted** (R < θ, past TTL, not pinned). At day 180 the trust gate fires on
three aged memories: `uploadV1` is **refuted** against the live repo
(tombstoned on the spot + FSRS fail review), `MAX_UPLOAD_MB` and `useQuery`
are **confirmed** (confidence ↑, `last_verified_at`, FSRS good review). The
final recall trace shows the used memories with score breakdowns and trust
chips, the excluded `useApi` ("superseded by … on …"), and the counterfactual
toggle — what a plain RAG would have wrongly injected.

## MCP server — drop-in for any harness

A thin FastMCP wrapper (server name `quen`) over the same engine — mount it
from Claude Code, Cursor, OpenAI Agents SDK, LangGraph, or any MCP client:

```bash
.venv/bin/quen-mcp    # stdio transport
```

Tools: `remember(observation)` · `recall(query, token_budget)` (memories
**with trust fields** + `needs_verification` flags) · `dream()` ·
`verify_hint(memory_id, result, evidence)` (the harness verifies with *its*
live tools — grep, file read, URL fetch — and reports back; the engine applies
the confidence update + FSRS review) · `pin(memory_id)` · `inspect(filter)`.

Memory-as-MCP exists (MemMachine, Mem0, FSRS-memory servers) — the packaging
is adoption, not novelty. The wrapper stays thin; the eval never uses it.

## Dashboard

Vite + React 18 + TS + Tailwind + shadcn-style components + recharts, SWR
polling. Five panels: **Memories table** (live R computed client-side from
FSRS constants served by `/config`, decaying before your eyes), **Memory
detail** (forgetting curve with review markers and the eviction threshold θ,
validity timeline, verbatim toggle, Pin / Force self-test / Verify now),
**Dream log** (per-run consolidations, self-tests with ΔS, supersessions
badged `deterministic|NLI`, evictions, journal), **Recall trace** (token
meter, score breakdown, trust chips, excluded-relevant, counterfactual
toggle, verification events), **Vitals** (status counts, R/S histograms, KPI
tiles fed by real eval runs only — never fabricated, and the two calibration
charts).

## Honesty & limitations

- **Dry-run numbers measure mechanics, not models.** The offline reader is
  extractive; live Qwen numbers must be produced with `--live` before quoting
  accuracy anywhere.
- **Exact-match scoring under-reports paraphrases**; cross-check a sample with
  a second judge on live runs. All numbers are comparative-within-eval.
- **RepoGrepVerifier checks identifier presence, not claim truth**: a renamed
  identifier refutes; a changed *value* behind the same identifier confirms.
  Comment-only hits count as confirmed. It is the demo verifier; the MCP
  `verify_hint` path lets a real harness bring real verification.
- **The salience gate and NLI fallback inherit LLM judgment** — that's why
  supersession is deterministic-first and confidence-gated, contradictions
  never hard-delete, and every mutation is audited.
- FSRS default weights are used as published, not re-fit to agent-memory data.

## Research it stands on (cite generously, claim narrowly)

Retrieval scoring: Generative Agents (2304.03442). Retention: FSRS/DSR
(open-spaced-repetition). Consolidation: Letta sleep-time compute; A-MEM
(2502.12110). Supersession: MemStrata (2606.26511, deterministic (s,r,o) rule
+ the AUROC-0.59 finding); TOKI (2606.06240, bitemporal operators);
Knowledge-Conflicts survey (2403.08319); Memory-R1 (2508.19828, augmentation ≠
contradiction). Trust at answer time: Scaling-the-Harness (2605.26112); UAM
(2601.15703); Hindsight/CARA (2512.12818); abstention-aware retrieval for
coding agents (2604.27283). Measurement: Memora + FAMA (2604.20006);
LongMemEval (2410.10813). Efficiency: Mem0 (2504.19413). Anti-pattern:
MemPalace critique (2604.21284, why `content_verbatim` never dies).

## Repo map

```
src/quen/          engine: models · fsrs · store · llm · embeddings ·
                   write_pipeline · retrieval · supersession · dream ·
                   trust · verifiers · engine · api · mcp_server ·
                   alibaba_client (THE proof artifact)
tests/             137 offline deterministic tests (TDD list from the spec)
eval/              FAMA probe · LongMemEval · calibrations · budget curve
dashboard/         the glass box
scripts/           seed_demo.py + demo_repo fixture
docs/SPEC.md       master spec v3
```

## License

MIT.

---

*The line we defend: **memory that knows what to forget — and how sure to
sound — because it checks.***
