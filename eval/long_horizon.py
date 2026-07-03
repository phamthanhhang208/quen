#!/usr/bin/env python3
"""Long-horizon token economics: what forgetting saves as memory ages.

Simulates a year of team memory (new facts weekly, a share of slot facts
updated by evidence, throwaway notes), then measures — per week — answer
quality at a fixed token budget and the store footprint, for Quên vs an
append-only RAG over the SAME stream. Offline and deterministic (hashing
embedder + scripted extractive reader): this isolates the *memory
management* economics from LLM quality.

  .venv/bin/python eval/long_horizon.py [--weeks 52] [--budget 400]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import OUT_DIR, build_stack, write_json  # noqa: E402
from configs import AppendOnlyRAG, Quen  # noqa: E402
from fama import mentioned_positively, word_present  # noqa: E402
from quen.retrieval import estimate_tokens  # noqa: E402

NEW_SLOTS_PER_WEEK = 2
UPDATE_RATE = 0.15          # weekly share of live slots invalidated by evidence
TRIVIA_PER_WEEK = 3
QUERIES_PER_WEEK = 6


def _slot_fact(slot: int, version: int) -> tuple[str, str]:
    tech = f"stack{slot}v{version}"
    return f"service{slot} runs on {tech}.", tech


def simulate(weeks: int, budget: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    llm, embedder = build_stack(live=False)
    quen = Quen(llm, embedder, verify=False)
    rag = AppendOnlyRAG(*build_stack(live=False))

    current: dict[int, int] = {}      # slot -> live version
    history: dict[int, list[str]] = {}  # slot -> all past tech names
    timeline: list[dict] = []
    next_slot = 0

    for week in range(1, weeks + 1):
        day = week * 7
        events: list[str] = []
        for _ in range(NEW_SLOTS_PER_WEEK):
            current[next_slot] = 0
            text, tech = _slot_fact(next_slot, 0)
            history[next_slot] = [tech]
            events.append(text)
            next_slot += 1
        for slot in list(current):
            if rng.random() < UPDATE_RATE:
                current[slot] += 1
                text, tech = _slot_fact(slot, current[slot])
                history[slot].append(tech)
                events.append(text)
        for j in range(TRIVIA_PER_WEEK):
            events.append(
                f"standup week{week} note{j}: discussed sprint scope briefly."
            )
        for text in events:
            kind = "pr" if "runs on" in text else "chat"
            quen.ingest(text, day=day, kind=kind, source_ref=f"w{week}")
            rag.ingest(text, day=day, kind=kind, source_ref=f"w{week}")
        quen.day_boundary(day + 1)  # weekly dream: reconcile + decay

        # ---- weekly probe: ask about random live slots -------------------
        slots = rng.sample(sorted(current), min(QUERIES_PER_WEEK, len(current)))
        stats = {"quen": [0, 0], "append_only": [0, 0]}  # [correct, stale-free]
        tokens = {"quen": 0, "append_only": 0}
        quen._set_day(day + 2)
        for slot in slots:
            q = f"service{slot} runs on what?"
            valid = f"stack{slot}v{current[slot]}"
            invalid = history[slot][:-1]
            for name, system in (("quen", quen), ("append_only", rag)):
                ans = system.answer(q, budget)
                ok = word_present(valid, ans.text)
                clean = not any(mentioned_positively(t, ans.text) for t in invalid)
                stats[name][0] += ok and clean
                stats[name][1] += clean
                tokens[name] += ans.tokens_used

        store = quen.engine.store
        counts = store.counts_by_status()
        rag_tokens_total = sum(estimate_tokens(t) for t, _ in rag.entries)
        quen_active_tokens = sum(
            estimate_tokens(m.content) for m in store.active()
        )
        n = len(slots)
        timeline.append(
            {
                "week": week,
                "facts_total": len(rag.entries),
                "quen_active": counts["active"],
                "quen_superseded": counts["superseded"],
                "quen_deprecated": counts["deprecated"],
                "store_tokens_append_only": rag_tokens_total,
                "store_tokens_quen_active": quen_active_tokens,
                "fama_quen": round(stats["quen"][0] / n, 3),
                "fama_append_only": round(stats["append_only"][0] / n, 3),
                "stale_free_quen": round(stats["quen"][1] / n, 3),
                "stale_free_append_only": round(stats["append_only"][1] / n, 3),
                "tokens_per_q_quen": round(tokens["quen"] / n, 1),
                "tokens_per_q_append_only": round(tokens["append_only"] / n, 1),
            }
        )
        if week % 10 == 0:
            t = timeline[-1]
            print(f"week {week}: store {t['facts_total']} vs active "
                  f"{t['quen_active']} | FAMA {t['fama_append_only']} vs "
                  f"{t['fama_quen']}")
    return timeline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weeks", type=int, default=52)
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    timeline = simulate(args.weeks, args.budget)
    write_json(args.out / "long_horizon.json", timeline)

    last = timeline[-1]
    mid = timeline[len(timeline) // 2]
    print(json.dumps({"week_26": mid, "week_final": last}, indent=2))
    return timeline


if __name__ == "__main__":
    main()
