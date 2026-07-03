#!/usr/bin/env python3
"""The code-staleness probe, scored with FAMA (headline eval, spec §9).

Each case's history is walked day by day into each memory system (identical
reader/budget); the question is asked at question_day; FAMA scores the
answer. For the Quên configs, forgetting precision/recall are computed from
the store's tombstones vs the case's ground truth.

  .venv/bin/python eval/run_probe.py            # offline dry-run (default)
  .venv/bin/python eval/run_probe.py --live     # Qwen via DashScope
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import build_stack, load_probe_cases, make_parser, write_json  # noqa: E402
from configs import AppendOnlyRAG, FullContext, Quen  # noqa: E402
from fama import score_answer, word_present, mentioned_positively  # noqa: E402


def _materialize_live_files(case: dict) -> str | None:
    files = case.get("live_files")
    if not files:
        return None
    root = Path(tempfile.mkdtemp(prefix=f"probe-{case['id']}-"))
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return str(root)


def make_system(config: str, case: dict, live: bool):
    llm, embedder = build_stack(live)
    if config == "append_only":
        return AppendOnlyRAG(llm, embedder)
    if config == "full_context":
        return FullContext(llm, embedder)
    repo = _materialize_live_files(case)
    if config == "quen":
        return Quen(llm, embedder, verify=True, repo_path=repo)
    if config == "quen_no_verify":
        return Quen(llm, embedder, verify=False, repo_path=repo)
    raise ValueError(config)


def run_case(case: dict, config: str, *, budget: int, live: bool) -> dict:
    system = make_system(config, case, live)
    last_day = None
    for event in sorted(case["history"], key=lambda e: e["day"]):
        if last_day is not None and event["day"] != last_day:
            system.day_boundary(event["day"])
        system.ingest(
            event["text"],
            day=event["day"],
            kind=event.get("kind", "chat"),
            source_ref=event.get("source_ref"),
        )
        last_day = event["day"]
    q_day = case.get("question_day", (last_day or 0) + 1)
    system.day_boundary(q_day)

    ans = system.answer(case["question"], budget)
    score = score_answer(
        ans.text,
        valid_facts=case.get("valid_facts", []),
        invalidated_facts=case.get("invalidated_facts", []),
        expects_abstain=case.get("expects_abstain", False),
        abstained=ans.abstained,
    )
    row = {
        "case_id": case["id"],
        "config": config,
        "presence": score.presence,
        "absence": score.absence,
        "fama": score.fama,
        "abstain_ok": score.abstain_ok,
        "augmentation": case.get("augmentation", False),
        "expects_abstain": case.get("expects_abstain", False),
        "verify_case": bool(case.get("live_files")),
        "tokens_used": ans.tokens_used,
        "answer_confidence": ans.confidence,
        "freshness_max": ans.freshness_max,
        "answer": ans.text[:400],
    }
    if isinstance(system, Quen):
        row["forgetting"] = _forgetting_stats(system, case)
    return row


def _forgetting_stats(system: Quen, case: dict) -> dict:
    """Ground truth vs actual tombstones for this case's store."""
    store = system.engine.store
    invalidated = case.get("invalidated_facts", [])
    tombstoned = store.list(status="superseded", limit=1000) + [
        m for m in store.list(status="deprecated", limit=1000)
        # eviction is decay, not contradiction — exclude it from precision
        if not any(a["action"] == "evict"
                   for a in store.audit_tail(memory_id=m.id))
    ]
    active = store.active()
    # negation-aware, matching the recall side: a tombstoned note that only
    # MENTIONS an invalidated fact negatively ("useApi was removed") is not
    # a correct forgetting of that fact
    correct_tombstones = sum(
        1 for m in tombstoned
        if any(mentioned_positively(f, m.content_verbatim) for f in invalidated)
    )
    forgotten = sum(
        1 for f in invalidated
        if not any(mentioned_positively(f, m.content) for m in active)
    )
    return {
        "tombstoned": len(tombstoned),
        "correct_tombstones": correct_tombstones,
        "invalidated_total": len(invalidated),
        "invalidated_forgotten": forgotten,
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest uncertainty for small-n rates."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def mcnemar_exact(rows: list[dict], config_a: str, config_b: str) -> dict:
    """Exact McNemar on paired per-case FAMA outcomes — all configs answer
    the SAME cases, so the paired test is the honest comparison at n=25."""
    from math import comb

    by_case: dict[str, dict[str, bool]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["config"]] = bool(r["fama"])
    b = sum(1 for v in by_case.values()
            if v.get(config_a) and not v.get(config_b))
    c = sum(1 for v in by_case.values()
            if not v.get(config_a) and v.get(config_b))
    n = b + c
    if n == 0:
        return {"a_wins": b, "b_wins": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"a_wins": b, "b_wins": c, "p_value": round(min(1.0, p), 5)}


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    configs = sorted({r["config"] for r in rows})
    for config in configs:
        sub = [r for r in rows if r["config"] == config]
        n = len(sub)
        abstain_sub = [r for r in sub if r["expects_abstain"]]
        fama_hits = sum(r["fama"] for r in sub)
        # verify-before-answer cases need a live source by construction —
        # report the headline both with and without them so the comparison
        # to verifier-less baselines is transparent
        no_verify = [r for r in sub if not r["verify_case"]]
        agg = {
            "cases": n,
            "fama": round(fama_hits / n, 4),
            "fama_ci95": wilson_ci(fama_hits, n),
            "fama_excluding_verify_cases": (
                round(sum(r["fama"] for r in no_verify) / len(no_verify), 4)
                if no_verify else None
            ),
            "presence": round(sum(r["presence"] for r in sub) / n, 4),
            "absence": round(sum(r["absence"] for r in sub) / n, 4),
            "abstention_accuracy": (
                round(sum(bool(r["abstain_ok"]) for r in abstain_sub)
                      / len(abstain_sub), 4) if abstain_sub else None
            ),
            "avg_tokens_per_query": round(
                sum(r["tokens_used"] for r in sub) / n, 1
            ),
        }
        forgetting = [r["forgetting"] for r in sub if "forgetting" in r]
        if forgetting:
            tomb = sum(f["tombstoned"] for f in forgetting)
            correct = sum(f["correct_tombstones"] for f in forgetting)
            total_inv = sum(f["invalidated_total"] for f in forgetting)
            forgot = sum(f["invalidated_forgotten"] for f in forgetting)
            agg["forgetting_precision"] = round(correct / tomb, 4) if tomb else None
            agg["forgetting_recall"] = round(forgot / total_inv, 4) if total_inv else None
        out[config] = agg
    return out


def _confidence_by_freshness(quen_rows: list[dict]) -> list[dict]:
    """Stated answer-confidence vs FAMA correctness, stratified by the
    stalest memory relied upon — the headline epistemic-honesty measurement,
    computed from real eval outcomes (not demo judgments)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from quen.engine import _calibration_bins, _confidence_strata  # noqa: PLC0415

    events = [
        {
            "predicted": r["answer_confidence"],
            "outcome": 1 if r["fama"] else 0,
            "freshness_days": r.get("freshness_max"),
        }
        for r in quen_rows
        if r.get("answer_confidence") is not None
    ]
    strata = _confidence_strata([e for e in events if e["freshness_days"] is not None])
    overall = _calibration_bins(events)
    total = sum(b["count"] for b in overall)
    ece = (
        round(
            sum(abs(b["predicted_mean"] - b["empirical"]) * b["count"]
                for b in overall) / total, 4,
        )
        if total else None
    )
    return [{"overall_ece": ece, "events": len(events)}] + strata


CONFIGS = ["append_only", "full_context", "quen", "quen_no_verify"]


def main(argv: list[str] | None = None) -> dict:
    args = make_parser(__doc__).parse_args(argv)
    cases = load_probe_cases(args.limit, args.offset)
    rows = []
    for case in cases:
        for config in CONFIGS:
            row = run_case(case, config, budget=args.budget, live=args.live)
            rows.append(row)
            print(f"{case['id']:>10} {config:<14} fama={row['fama']} "
                  f"presence={row['presence']} absence={row['absence']}")
    summary_by_config = aggregate(rows)
    paired = {
        f"quen_vs_{base}": mcnemar_exact(rows, "quen", base)
        for base in ("append_only", "full_context", "quen_no_verify")
        if any(r["config"] == base for r in rows)
    }
    confidence_calibration = _confidence_by_freshness(
        [r for r in rows if r["config"] == "quen"]
    )

    # KPI tiles for /vitals read ONLY the headline (ours) numbers
    ours = summary_by_config.get("quen", {})
    kpis = {
        "fama": ours.get("fama"),
        "forgetting_precision": ours.get("forgetting_precision"),
        "forgetting_recall": ours.get("forgetting_recall"),
        "avg_tokens_per_query": ours.get("avg_tokens_per_query"),
        "mode": "live" if args.live else "dry-run",
        "embed_model": (
            __import__("os").environ.get("QUEN_EMBED_MODEL", "text-embedding-v4")
            if args.live else "hashing"
        ),
        "budget": args.budget,
        "by_config": summary_by_config,
        "paired_mcnemar": paired,
        "confidence_calibration_by_freshness": confidence_calibration,
    }
    write_json(args.out / "probe_results.json", rows)
    write_json(args.out / "summary.json", kpis)

    print("\n=== FAMA by config ===")
    for config, agg in summary_by_config.items():
        print(f"{config:<14} fama={agg['fama']:.2f} presence={agg['presence']:.2f} "
              f"absence={agg['absence']:.2f} tokens={agg['avg_tokens_per_query']}")
    return kpis


if __name__ == "__main__":
    main()
