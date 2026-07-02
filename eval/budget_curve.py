#!/usr/bin/env python3
"""Accuracy-vs-budget curve: FAMA on the probe at several token budgets,
identical across configs (spec §9).

  .venv/bin/python eval/budget_curve.py [--limit N] [--live]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import load_probe_cases, make_parser, write_json  # noqa: E402
from run_probe import CONFIGS, run_case  # noqa: E402

BUDGETS = (30, 60, 120, 300, 600)


def main(argv=None) -> list[dict]:
    args = make_parser(__doc__).parse_args(argv)
    cases = load_probe_cases(args.limit)
    curve = []
    for budget in BUDGETS:
        for config in CONFIGS:
            rows = [
                run_case(case, config, budget=budget, live=args.live)
                for case in cases
            ]
            fama = sum(r["fama"] for r in rows) / len(rows)
            tokens = sum(r["tokens_used"] for r in rows) / len(rows)
            curve.append(
                {
                    "budget": budget,
                    "config": config,
                    "fama": round(fama, 4),
                    "avg_tokens": round(tokens, 1),
                }
            )
            print(f"budget={budget:>4} {config:<14} fama={fama:.2f} "
                  f"tokens={tokens:.1f}")
    write_json(args.out / "budget_curve.json", curve)
    return curve


if __name__ == "__main__":
    main()
