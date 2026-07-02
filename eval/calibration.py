#!/usr/bin/env python3
"""Calibration metrics over a Quên database (spec §9):

- retention calibration: predicted R at self-test time vs empirical recall
  (reliability bins + log-loss);
- confidence calibration: stated answer-confidence vs judged correctness,
  stratified by memory freshness (per-stratum ECE + selective accuracy).

  .venv/bin/python eval/calibration.py --db data/demo.db
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import OUT_DIR, write_json  # noqa: E402
from quen.engine import _calibration_bins, _confidence_strata  # noqa: E402
from quen.store import MemoryStore  # noqa: E402

EPS = 1e-6


def log_loss(events: list[dict]) -> float | None:
    if not events:
        return None
    total = 0.0
    for e in events:
        p = min(1 - EPS, max(EPS, e["predicted"]))
        y = e["outcome"]
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return round(total / len(events), 4)


def ece(bins: list[dict]) -> float | None:
    total = sum(b["count"] for b in bins)
    if not total:
        return None
    return round(
        sum(abs(b["predicted_mean"] - b["empirical"]) * b["count"] for b in bins)
        / total,
        4,
    )


def selective_accuracy(events: list[dict], thresholds=(0.0, 0.25, 0.5, 0.75)) -> list[dict]:
    """Accuracy when only answering above a stated-confidence threshold —
    honest hedging should trade coverage for accuracy."""
    out = []
    for t in thresholds:
        kept = [e for e in events if e["predicted"] >= t]
        out.append(
            {
                "threshold": t,
                "coverage": round(len(kept) / len(events), 4) if events else None,
                "accuracy": (
                    round(sum(e["outcome"] for e in kept) / len(kept), 4)
                    if kept else None
                ),
            }
        )
    return out


def compute(db_path: str) -> dict:
    store = MemoryStore(db_path)
    try:
        retention = store.calibration_events("retention")
        confidence = store.calibration_events("confidence")
        retention_bins = _calibration_bins(retention)
        return {
            "db": db_path,
            "retention": {
                "events": len(retention),
                "bins": retention_bins,
                "ece": ece(retention_bins),
                "log_loss": log_loss(retention),
            },
            "confidence": {
                "events": len(confidence),
                "by_freshness": _confidence_strata(confidence),
                "overall_ece": ece(_calibration_bins(confidence)),
                "selective_accuracy": selective_accuracy(confidence),
            },
        }
    finally:
        store.close()


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/demo.db")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    result = compute(args.db)
    write_json(args.out / "calibration.json", result)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
