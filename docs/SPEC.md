# Quên — Master Spec v3 (definitive)

**Quên** *(pronounced like **Qwen**; Vietnamese for "to forget")* — **trust-calibrated forgetting: an agent memory that knows what to forget, says how sure it is, and verifies before it asserts.**

Qwen Cloud Global AI Hackathon · **Track 1: MemoryAgent** · Deadline **Jul 9, 2026, 2:00pm PDT**. Supersedes v2.

**Name & branding.** Display name **Quên** (keep the diacritic in README/logo/video); ASCII identifiers everywhere else: repo `quen`, Python package `quen`, MCP server name `quen`. Tagline: *"Quên (sounds like Qwen) — Vietnamese for 'to forget'. A Qwen-powered memory that knows what to forget — and how sure to sound."* README must state: a community homage to the Qwen name — not affiliated with or endorsed by the Qwen/Alibaba team. Optional nod: *Quen* is also the protective shield sign in The Witcher — "shields your agent from stale memory."

**Changelog vs v2:** (1) repositioned the differentiator against the H1-2026 literature explosion (Memora/FAMA, MemStrata, TOKI, UAM, CARA — see §1/§12); (2) contradiction pipeline upgraded to **deterministic (s,r,o) supersession first, LLM-NLI fallback** (MemStrata's AUROC-0.59 finding); (3) NEW mechanism: **verify-before-answer + verification-as-review**; (4) eval adopts **FAMA** and LongMemEval's **abstention** subset; (5) **MCP packaging** section added; (6) UI de-themed → **generic dashboard** (Tailwind + shadcn/ui + recharts + polling; UMAP/WebGL/graph/motion cut); (7) project named **Quên**.

---

## 0. TL;DR

**Quên** is a tiered memory engine for agents where every memory carries **retention state (FSRS D/S/R), bi-temporal validity, confidence, and freshness**. Writes are **salience-gated** (store the delta, not everything). Retrieval is **relevance-dominant and token-budgeted**, active-only. A **dream pass** re-abstracts episodics into generalizations, **self-tests** memories via active recall, supersedes contradicted facts (**deterministic (s,r,o) rule first, LLM-NLI fallback**), and decays/evicts the rest. At answer time the agent **propagates trust**: it hedges in proportion to confidence×freshness, and for low-trust memories it **verifies against the live source before asserting** — and the verification outcome is itself an FSRS review (reinforce or invalidate). Shipped as a FastAPI service **plus a thin MCP server** any agent harness can mount, with a **plain dashboard** for inspection. All LLM/embedding calls run on **Qwen via Alibaba Cloud DashScope**. Proven with numbers: LongMemEval (Knowledge-Update, Temporal, **Abstention**), a code-domain **autonomous-staleness probe scored with FAMA**, **retention calibration**, and **freshness-stratified confidence calibration**.

---

## 1. Positioning & the special thing (honest, post-research)

**The landscape (H1 2026) is now crowded — do not claim these as novel:**
- Forgetting *benchmarks* exist: **Memora + FAMA** (arXiv 2604.20006) penalizes reliance on invalidated memories; FiFA (privacy forgetting), GateMem (deletion-request forgetting), MemoryAgentBench (selective forgetting).
- Bi-temporal supersession exists, including for code: **MemStrata** (arXiv 2606.26511) retires stale facts via a deterministic (subject, relation, object) rule in a bi-temporal ledger; **TOKI** (arXiv 2606.06240) formalizes a bitemporal operator algebra for contradiction resolution.
- Confidence-in-memory exists: **UAM** (arXiv 2601.15703) retains verbalized confidence in context; **Hindsight/CARA** (arXiv 2512.12818) keeps confidence-scored opinions and evaluates abstention; abstention-aware memory *retrieval* for coding agents exists (arXiv 2604.27283).
- Decay-based forgetting exists: FadeMem (arXiv 2601.18642), FSRS-based memory MCPs, MemoryBank.

**What is still ours — the special thing, in one line:**

> **Trust-calibrated forgetting, closed-loop and measured.** Trust is a *runtime decision*, not a stored property: each answer's stated confidence tracks the memory's validity/freshness (hedging), low-trust memories trigger **verify-before-answer** against the live source, and **the verification outcome feeds back as an FSRS review** — verification reinforces or invalidates memory. We then *measure* the epistemic honesty: FAMA on a code-staleness probe + **freshness-stratified confidence calibration (ECE)** + retention calibration. To our knowledge no system closes the loop verification→retention-update, and none scores hedging against memory freshness.

This leans on the sharpest sentence in the recent harness literature (arXiv 2605.26112): *stale memory rarely fails at retrieval — it fails by making the agent act confidently on invalidated assumptions*; the fix is to treat retrieved memory as a hypothesis and make trust a runtime decision. We mechanize exactly that, then benchmark it.

Secondary strengths (adoption/presentation, not novelty): MCP packaging (drop-in for any harness), glass-box dashboard, all-Qwen-on-Alibaba stack.

---

## 2. Judging map

| Criterion | Weight | Where we earn it |
|---|---|---|
| Technical Depth | 30% | FSRS retention core; deterministic+NLI supersession; verify-before-answer loop; token-budgeted retrieval; MCP server; Qwen/DashScope |
| Innovation | 30% | Verification-as-review closed loop; freshness-stratified confidence calibration; FAMA-scored code-staleness probe |
| Impact | 25% | Kills "confidently wrong from stale memory" for real coding agents; drop-in via MCP |
| Presentation | 15% | Dashboard makes the lifecycle inspectable; honest eval with stated validity threats |

---

## 3. Research foundations (condensed; full table §12)

- **Retrieval scoring** — Generative Agents (2304.03442): importance (LLM-rated at write) + relevance (cosine); we replace naive recency with FSRS R and make relevance dominant.
- **Retention** — FSRS/DSR (canonical repo `open-spaced-repetition/free-spaced-repetition-scheduler`): R decays; success raises S (more at low R — spacing effect); failure lowers S, raises D. We use DSR state as an **eviction/ranking prior**, not a scheduler (we don't control review timing).
- **Consolidation/dream** — Letta sleep-time compute; Generative-Agents reflection; A-MEM memory evolution (2502.12110): new memories trigger updates to related old ones — the precedent for our reconcile-on-ingest.
- **Supersession** — MemStrata (2606.26511): cosine similarity separates *contradicted* from *duplicated* facts at **AUROC 0.59 (≈chance)** — embeddings structurally cannot detect supersession; a deterministic (s,r,o) slot rule can, with zero LLM calls. TOKI (2606.06240) gives the formal bitemporal operators. Knowledge-Conflicts survey (2403.08319): LLM contradiction detection is unreliable → confidence-gate + never hard-delete. Memory-R1 (2508.19828): the Buddy/Scout trap — augmentation ≠ contradiction.
- **Trust at answer time** — Scaling-the-Harness (2605.26112): staleness fails as confident action on invalidated assumptions; treat memory as hypothesis; re-verify against the live environment. UAM (2601.15703) and CARA (2512.12818) carry confidence; neither closes verification back into retention, neither calibrates hedging against freshness.
- **Measurement gap** — most benchmarks test shallow retrieval; Memora/FAMA (2604.20006) shows agents *frequently reuse invalid memories*; LongMemEval (2410.10813) has Knowledge-Update, Temporal, and **Abstention** abilities — our public anchor.
- **Efficiency** — Mem0 (2504.19413): 91% lower latency than full-context (verified paper claim; treat vendor "~94 LongMemEval" as self-report — independent evals put Mem0 ≈47–52).
- **Anti-pattern** — MemPalace critique (2604.21284): keep `content_verbatim`; don't let metaphors carry claims.

---

## 4. Architecture

### 4.1 Flow

```
 user/agent ─▶ Primary answer path (Qwen):
                retrieve(query, budget) → assemble → TRUST GATE →
                  high trust  → answer, confidence stated
                  low trust   → verify-before-answer (live source) → answer
                                └─ verification outcome = FSRS review (reinforce/invalidate)

 Memory store (tiered): working / episodic (raw, timestamped) / semantic (consolidated; DSR + validity + confidence)

 Write path (Qwen): extract (s,r,o where possible) → salience gate → importance → dedup → init DSR → store

 Dream pass (Qwen, async): re-abstract → self-test (active recall) → supersede (deterministic→NLI) → decay/evict → compress

 All LLM + embeddings ─▶ Qwen on Alibaba Cloud DashScope. Audit log on every mutation → dashboard.
```

### 4.2 Memory schema

```python
@dataclass
class MemoryItem:
    id: str
    content: str; content_verbatim: str
    triple: tuple[str,str,str] | None   # (subject, relation, object) if slot-extractable
    embedding: list[float]
    mtype: Literal["episodic","semantic"]
    importance: float                   # 1–10 at write
    salience: float                     # 0–1 novelty vs model prior (write gate)
    # retention (FSRS)
    difficulty: float; stability: float
    last_review_at: datetime; review_count: int
    pinned: bool
    # bi-temporal validity
    valid_from: datetime; valid_to: datetime | None
    status: Literal["active","deprecated","superseded"]
    superseded_by: str | None
    # trust
    confidence: float                   # 0–1; ↓ when challenged, ↑ when verified
    last_verified_at: datetime | None
    source_ref: str | None; created_at: datetime
    # derived on the fly: R(D,S,Δt); freshness = age since source_ref/last_verified_at
```

### 4.3 Write policy — store the delta

Extract candidate facts (as (s,r,o) triples where the fact is slot-like: "team fetches data via useApi"). **Salience-gate**: skip what a strong base model already knows; keep the surprising/personal/project-specific. Dedup by triple key first, cosine second. Init DSR defaults; set confidence from source authority (ingested PR/commit > casual mention).

### 4.4 Retrieval — relevance-dominant, token-bounded, active-only

`rank = wrel·cosine(q,mem) + wr·R + wi·norm(importance)`, wrel highest; a low-R but highly-relevant memory surfaces at full strength (decay never suppresses relevance). Greedy fill to `token_budget`; exclude deprecated/superseded unless the query is historical. Return memories + tokens + per-memory trust (confidence, freshness, R). Optional: short agentic re-query loop.

### 4.5 Retention — FSRS + review mapping

Import exact FSRS-v4.5/v6 equations + default weights from the canonical repo (v1–v4: `R=(1+t/9S)^-1`; v4.5+: `R=(1+F·t/S)^decay` — state the version). **Reviews** happen on: (a) use in an answer later judged correct/uncontradicted → good; (b) dream self-test pass/fail; (c) **verification pass/fail (new)**. Eviction: `R<θ` ∧ unaccessed past TTL ∧ ¬pinned → deprecated (tombstone, never hard-delete).

### 4.6 Dream pass

1. **Re-abstract** (primary value): episodics → reusable generalizations ("this team prefers X over Y"), provenance kept; dedupe/compress is secondary.
2. **Self-test (active recall):** probe → answer from active set → pass: reinforce; fail: re-expand verbatim + do-not-forget. Prefer exact-answer probes.
3. **Supersession — two-stage:**
   - **Deterministic first (MemStrata-style):** same (s,r) slot, different o, newer valid_from ⇒ supersede. Zero LLM calls, immune to the AUROC-0.59 embedding failure.
   - **NLI fallback (non-slotted facts):** candidates by entity overlap + kNN → Qwen classifies `entails|neutral|contradicts|augments` (augmentation adds a sibling — never DELETE+ADD) → confidence gate → arbitrate by recency+authority. Measure detection P/R on a labeled held-out set.
   - On supersede: `A.valid_to=B.valid_from; A.status="superseded"; A.superseded_by=B.id` (independent of R).
4. **Decay+evict**, 5. **Compress** (keep verbatim). Audit-log every action.

### 4.7 Trust gate + verify-before-answer (the special mechanism)

At answer time compute per-memory **trust = f(confidence, freshness, status)**. If the answer hinges on a memory below the trust threshold **and a verifier is available** (repo grep/file read via harness tools; a URL fetch; a user-confirm), run **verify-before-answer**: check the claim against the live source. Outcomes: **confirmed** → assert plainly; bump `confidence`, `last_verified_at`, FSRS good review. **Refuted** → supersede/invalidate on the spot; answer from the corrected state; FSRS fail review. **Unverifiable** → answer with explicit hedge ("as of PR#42 — may have changed"). Stated answer-confidence must track the trust of what it used. *Never* let a stale, unverified memory be asserted with full assurance.

### 4.8 Staleness signal & pinning

Staleness enters only via **newly ingested observations** (turns, commits/PRs); dream reconciles new vs existing (A-MEM evolution) — no magic. Memory beats re-reading because reconciliation is paid **once**, then all future queries stay clean within budget. Pinning: explicit | auto-pin classifier (security/PII rules) | earned floor; pinned never auto-evicts, supersedable only at a higher bar.

---

## 5. Packaging: MCP server (drop-in for any harness)

A **thin FastMCP wrapper** (server name `quen`) over the same engine — do not build a bespoke agent loop; let existing harnesses (Claude Code, Cursor, OpenAI Agents SDK, LangGraph — all MCP clients) mount it.

Tools: `remember(observation)` (write pipeline), `recall(query, token_budget)` (returns memories **with trust fields**), `dream()`, `verify_hint(memory_id, result)` (harness reports a verification outcome → engine applies the FSRS review + confidence update), `pin(id)`, `inspect(filter)` (dashboard data).

Honesty: memory-as-MCP exists (MemMachine, Mem0/Supermemory, FSRS-memory MCPs) — packaging is adoption, not novelty. Keep the wrapper thin; the **eval never uses a heavy harness** (minimal ingest→dream→ask loop only); the harness appears only in the demo. Qwen/Alibaba constraint is orthogonal: the server calls DashScope internally regardless of client.

---

## 6. Qwen on Alibaba Cloud — MANDATORY proof

All model+embedding calls via DashScope OpenAI-compatible endpoint; backend on Alibaba Cloud; **separate proof recording** + an obvious `alibaba_client.py`. Region keys not interchangeable (Singapore: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`). Models: `qwen3.5-plus` (reader/reflection/NLI), `qwen-flash` (extraction/judging) — confirm ids + embedding (`text-embedding-v4`, confirm dim). `DASHSCOPE_API_KEY` in `.env`; credits via the voucher form; **20-Q smoke run first** to extrapolate token cost.

---

## 7. Dashboard UI (generic, functional — no theme)

Plain admin dashboard (header text: "Quên · memory"). **Stack: Vite + React 18 + TS + Tailwind + shadcn/ui + recharts. Data via polling (SWR/React-Query, 2–5s); SSE optional later. No WebGL, no UMAP, no graph view, no animations, no custom theme (default shadcn zinc).**

Five panels (tabs or a simple grid):
1. **Memories table** — sortable/filterable: content snippet, type, status badge, importance, D, S, **live R** (client-computed), confidence, freshness, pinned, validity window, superseded_by. Row click → Detail.
2. **Memory detail** — recharts **forgetting curve** (R over time, review markers, eviction threshold θ, "forgotten in ~Nd"); validity timeline; verbatim toggle; provenance; verification history; buttons **Pin / Force self-test / Verify now**.
3. **Dream log** — per-run feed: consolidations (with the generalization produced), self-tests (probe, pass/fail, ΔS), supersessions (rule: deterministic|NLI), evictions; run diff; journal text (Qwen summary).
4. **Recall trace** — last `/ask`: query, answer + **answer_confidence**, token meter, used memories with score breakdown (relevance/R/importance) + trust chips (confidence·freshness), **excluded-relevant list** ("not used — superseded by X on DATE"), **counterfactual toggle** (what append-only RAG would have used), and any **verification event** that fired.
5. **Vitals** — counts by status, R & S histograms, KPI tiles (FAMA, forgetting P/R, avg tokens/query), **two calibration charts**: retention (predicted R vs empirical recall) and **confidence (stated confidence vs correctness, stratified by freshness)**.

Endpoints: `GET /memory, /memory/{id}, /dream/log, /dream/{run_id}, /recall/trace/{id}, /vitals; POST /memory/{id}/pin, /memory/{id}/recall, /memory/{id}/verify`.

---

## 8. Build plan (order; outer bound Jul 9)

**P0** setup (Alibaba key/credits, smoke test, deploy target, repo+license).
**P1** engine core: schema; write pipeline (triples + salience gate); FSRS module (imported eqns); relevance-dominant budgeted retrieval; `/ingest`,`/ask`.
**P2** dream pass (re-abstract → self-test → two-stage supersession → decay/evict) + audit log; **trust gate + verify-before-answer** + `verify_hint`. TDD: DSR monotonicity, validity transitions, deterministic-slot supersession, augmentation guard, pinned floor, budget cap, relevance-beats-decay, verification-updates-retention.
**P3** eval: 20-Q smoke → LongMemEval (KU+Temporal+**Abstention**) ×3 configs; **staleness probe scored with FAMA** (go/no-go gate — build first); retention + confidence calibration; charts.
**P4** MCP wrapper + dashboard + demo video + **separate Alibaba proof** + README/diagram/Track-1 submission (+optional blog).

---

## 9. Eval plan

**Datasets:** LongMemEval (`xiaowu0162/longmemeval-cleaned`): knowledge-update (78), temporal (133), **abstention** subset. Custom **code-staleness probe** (20–40 cases; facts invalidated by *evidence* — refactor PRs, implicit reversals — never by explicit "update" sentences).

**Configs (identical reader/judge/budget):** append-only RAG (top-k to same budget; no forgetting) · full-context Qwen (truncate oldest-first) · **ours** (and ours-minus-verify as an ablation if time allows).

**Metrics:**
- QA accuracy per subset (Qwen judge; prefer exact-match where possible; cross-check a sample with a second judge; numbers are comparative-within-eval).
- **FAMA** (adopted from Memora 2604.20006): presence-of-valid ∧ absence-of-invalidated — headline metric for the probe.
- Forgetting precision/recall (probe); tokens/query; accuracy-vs-budget curve.
- **Retention calibration**: predicted R vs empirical recall (reliability + log-loss).
- **Confidence calibration (ours):** stated answer-confidence vs correctness, **stratified by memory freshness** (ECE + selective-accuracy).

**Success gate:** ours > append-only on the probe (FAMA) by a clear margin at ≤ tokens; calibrations hold; KU tie vs a "use-latest" RAG is acceptable and expected. If the probe doesn't separate us → pivot early.

---

## 10. Demo (3 min, dashboard walkthrough)

1. Ingest sessions 1–3 → `useApi` memory visible in the table, R/importance rising. (0:25)
2. Ingest the migration PR → run dream → **Dream log** shows the deterministic supersession (`useApi` → superseded-by `useQuery`) + a produced generalization. (0:40)
3. Ask "implement notifications" → **Recall trace**: `useQuery` used; excluded panel shows `useApi` deliberately skipped; **counterfactual toggle** = the wrong answer a plain RAG gives. (0:40)
4. **The special beat:** ask about a config nobody re-stated → trust gate fires → **verify-before-answer** greps the live repo, confirms/refutes on camera → dashboard shows confidence + `last_verified_at` update and the FSRS review land. (0:40)
5. **Vitals:** FAMA bar vs baselines + the freshness-stratified confidence calibration chart. Close: "**Quên** — sounds like *Qwen*, Vietnamese for 'to forget': memory that knows what to forget — and how sure to sound." (0:35)

---

## 11. Risks & submission checklist

Risks: verify-before-answer needs a working verifier for the demo corpus (repo-grep — build it into the demo repo early); deterministic slot rule depends on decent triple extraction (fall back to NLI, measure both); the H1-2026 crowd means the README must cite generously and claim narrowly (the closed loop + the calibration measurement).

Checklist: ☐ repo `quen` + license ☐ README name story (Quên ≈ Qwen, Vietnamese "to forget") + not-affiliated disclaimer ☐ `alibaba_client.py` ☐ separate Alibaba proof ☐ diagram ☐ 3-min video ☐ Track-1 text ☐ reproducible eval (LongMemEval KU/Temporal/Abstention + FAMA probe + 2 calibrations + budget curve) ☐ MCP server ☐ dashboard ☐ optional blog.

---

*Quên, v3. The line to defend: **memory that knows what to forget — and how sure to sound — because it checks.***
