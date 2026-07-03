#!/usr/bin/env python3
"""Seed the spec §10 demo narrative into a fresh Quên database — fully
offline (HashingEmbedder + ScriptedLLM), deterministic mechanics, real dates.

The 180-day story (T0 = now − 180d):
  d0    session 1: "the team fetches data via useApi" (+ page-level facts)
  d1    session 2: MAX_UPLOAD_MB config fact, uploadV1 fact, pho trivia
  d2    ask about the data hook → judged correct (use-review, calibration)
  d3    dream #1 → generalization with provenance
  d10   ingest PR#42: migration to useQuery (authority: pr)
  d11   dream #2 → DETERMINISTIC SUPERSESSION useApi → useQuery
  d12   ask notifications (useQuery used, useApi excluded+counterfactual,
        judged correct) · ask lunch trivia (judged WRONG → retention drops)
  d180  dream #3 → the pho trivia EVICTS (R<θ ∧ TTL ∧ ¬pinned)
        verifier comes online (repo-grep over scripts/demo_repo), then:
        ask uploads   → uploadV1 REFUTED live (tombstoned + fail review)
        ask config    → MAX_UPLOAD_MB CONFIRMED live (confidence→1.0)
        ask data hook → useQuery CONFIRMED; stale page-facts refuted;
                        the final trace carries the whole trust story.

Run:  .venv/bin/python scripts/seed_demo.py  [db_path]
Then: QUEN_OFFLINE=1 QUEN_DB_PATH=data/demo.db \
        QUEN_VERIFY_REPO=scripts/demo_repo .venv/bin/quen-api
      cd dashboard && npm run dev
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quen import llm as llm_mod
from quen.config import QuenConfig
from quen.embeddings import HashingEmbedder
from quen.engine import QuenEngine
from quen.llm import ScriptedLLM
from quen.store import MemoryStore
from quen.verifiers import RepoGrepVerifier

DEMO_REPO = str(Path(__file__).parent / "demo_repo")
ASK_BUDGET = 80  # small budget → only the most relevant memories are used


class DemoClock:
    def __init__(self, start: datetime):
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance_to_day(self, t0: datetime, day: float) -> None:
        self.current = t0 + timedelta(days=day)

    def advance_hours(self, hours: float) -> None:
        self.current += timedelta(hours=hours)


def _reabstract_handler(prompt: str) -> str:
    """Generalize the useApi episodics (dream #1); nothing new afterwards —
    later runs dedup against the existing generalization anyway."""
    try:
        payload = json.loads(prompt[prompt.index("[{"):])
    except (ValueError, json.JSONDecodeError):
        return "[]"
    indices = [e["index"] for e in payload if "useApi" in e.get("content", "")]
    if not indices:
        return "[]"
    return json.dumps(
        [
            {
                "content": "This team standardizes its data fetching on the useApi hook.",
                "triple": None,
                "source_indices": indices,
            }
        ]
    )


def build_engine(db_path: str, clock: DemoClock, *, live: bool = False) -> QuenEngine:
    if live:
        # the realest integration test: Qwen does the extraction,
        # generalization, NLI and answering — the narrative mechanics
        # (supersession, eviction, verification) must still land
        from quen.embeddings import QwenEmbedder
        from quen.llm import QwenLLM

        llm = QwenLLM()
        embedder = QwenEmbedder()
    else:
        llm = ScriptedLLM.with_offline_defaults()
        llm.script(llm_mod.REABSTRACT, _reabstract_handler)
        embedder = HashingEmbedder()
    return QuenEngine(
        QuenConfig(db_path=db_path),
        store=MemoryStore(db_path, clock=clock.now),
        llm=llm,
        embedder=embedder,
        clock=clock.now,
    )


def main(db_path: str = "data/demo.db", *, live: bool = False) -> dict:
    # fresh demo DATABASE FILE (this deletes a db file, never a memory row)
    for suffix in ("", "-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)

    now = datetime.now(timezone.utc)
    t0 = now - timedelta(days=180)
    clock = DemoClock(t0)
    engine = build_engine(db_path, clock, live=live)

    # ---- d0-d1: three working sessions -------------------------------------
    engine.ingest("The team fetches data via useApi.", source_kind="chat",
                  source_ref="session-1")
    clock.advance_to_day(t0, 1)
    engine.ingest("Feed page fetches data with useApi.", source_kind="chat",
                  source_ref="session-2")
    engine.ingest("Settings page fetches data with useApi.", source_kind="chat",
                  source_ref="session-2")
    engine.ingest("Remember that MAX_UPLOAD_MB stays at 10 in src/config.ts.",
                  source_kind="doc", source_ref="session-2")
    engine.ingest("Uploads go through the uploadV1 helper.", source_kind="chat",
                  source_ref="session-2")
    engine.ingest("Team lunch was pho at the corner spot.", source_kind="chat",
                  source_ref="session-2")

    # ---- d2: a judged-good use builds review history ------------------------
    clock.advance_to_day(t0, 2)
    res = engine.ask("The team fetches data via which hook?", token_budget=ASK_BUDGET)
    engine.judge_answer(res.trace_id, correct=True)

    # ---- d3: dream #1 → generalization --------------------------------------
    clock.advance_to_day(t0, 3)
    dream1 = engine.dream()

    # ---- d10: the migration PR ----------------------------------------------
    clock.advance_to_day(t0, 10)
    engine.ingest("The team fetches data via useQuery.", source_kind="pr",
                  source_ref="PR#42")

    # ---- d11: dream #2 → deterministic supersession -------------------------
    clock.advance_to_day(t0, 11)
    dream2 = engine.dream()

    # ---- d12: judged asks (one right, one wrong) -----------------------------
    clock.advance_to_day(t0, 12)
    res = engine.ask("The team fetches data via which hook these days?",
                     token_budget=ASK_BUDGET)
    engine.judge_answer(res.trace_id, correct=True)
    # tiny budget → only the trivia memory is used; judging it wrong drops S
    res = engine.ask("What did the team have for lunch at the corner spot?",
                     token_budget=15)
    engine.judge_answer(res.trace_id, correct=False)

    # ---- d180 ("today"): dream #3 → eviction --------------------------------
    clock.advance_to_day(t0, 180)
    dream3 = engine.dream()

    # the verifier comes online (harness connected to the live repo)
    engine.verifiers.append(RepoGrepVerifier(DEMO_REPO))

    # verify-before-answer beats. The final ask is the flagship trace: it
    # confirms useQuery and MAX_UPLOAD against the live repo, refutes the
    # stale useApi generalization, excludes the superseded useApi fact, and
    # shows the counterfactual an append-only RAG would have injected.
    ask_uploads = engine.ask("What do uploads go through?", token_budget=ASK_BUDGET)
    clock.advance_hours(1)  # distinct timestamps → deterministic latest trace
    ask_final = engine.ask("The team fetches data via which hook these days?",
                           token_budget=ASK_BUDGET)

    summary = {
        "db_path": db_path,
        "counts": engine.store.counts_by_status(),
        "dreams": [dream1.run_id, dream2.run_id, dream3.run_id],
        "supersessions_dream2": [
            (ev.old_id, ev.new_id, ev.rule) for ev in dream2.supersessions
        ],
        "evicted_dream3": dream3.evicted_ids,
        "final_trace": ask_final.trace_id,
        "final_verifications": [
            (v.memory_id, v.outcome) for v in ask_final.verifications
        ],
        "uploads_verifications": [
            (v.memory_id, v.outcome) for v in ask_uploads.verifications
        ],
    }
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", default=None)
    parser.add_argument("--live", action="store_true",
                        help="use Qwen via DashScope instead of the offline stack")
    args = parser.parse_args()
    path = args.db_path or ("data/demo_live.db" if args.live else "data/demo.db")
    s = main(path, live=args.live)
    print(json.dumps(s, indent=2))
    print(
        "\nDemo seeded. Serve it:\n"
        f"  QUEN_OFFLINE=1 QUEN_DB_PATH={s['db_path']} "
        "QUEN_VERIFY_REPO=scripts/demo_repo .venv/bin/quen-api\n"
        "  cd dashboard && npm run dev   # → http://localhost:5173"
    )
