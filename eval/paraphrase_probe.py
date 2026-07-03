#!/usr/bin/env python3
"""Paraphrase-robustness variant of the staleness probe (anti-overfit).

The probe was authored by the same people who built the extractor — its
phrasings inevitably fit the system (researcher degrees of freedom). This
script measures that: Qwen paraphrases every history event ONCE (ground
truth facts untouched), the paraphrased dataset is committed for
reproducibility, and the probe re-runs on it. A large FAMA drop = the
system memorized our phrasing, not the mechanism.

  .venv/bin/python eval/paraphrase_probe.py --generate   # once, needs key
  .venv/bin/python eval/paraphrase_probe.py --live       # run on it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import make_parser, write_json  # noqa: E402
from run_probe import CONFIGS, aggregate, mcnemar_exact, run_case  # noqa: E402

PARAPHRASED = Path(__file__).parent / "data" / "probe_paraphrased.json"


def generate() -> list[dict]:
    """One-shot paraphrase of every history text via qwen-flash; committed
    so the held-out set is FROZEN before any scoring iteration sees it."""
    from quen import alibaba_client as ac
    from common import load_probe_cases

    cases = load_probe_cases()
    out = []
    for case in cases:
        clone = json.loads(json.dumps(case))
        for event in clone["history"]:
            resp = ac.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this engineering note in different words "
                            "with the same meaning. Keep every identifier, "
                            "number, and proper noun EXACTLY as written. "
                            "Reply with the rewritten sentence only.\n\n"
                            + event["text"]
                        ),
                    }
                ],
                model=ac.DEFAULT_FAST_MODEL,
                temperature=0.7,
                max_tokens=120,
                extra_body={"enable_thinking": False},
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                event["text"] = text
        out.append(clone)
    PARAPHRASED.parent.mkdir(parents=True, exist_ok=True)
    PARAPHRASED.write_text(json.dumps({"cases": out}, indent=2))
    print(f"wrote {PARAPHRASED} ({len(out)} cases)")
    return out


def main(argv=None) -> dict:
    parser = make_parser(__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="(re)generate the frozen paraphrased dataset")
    args = parser.parse_args(argv)
    if args.generate:
        generate()

    if not PARAPHRASED.exists():
        raise SystemExit("run with --generate first (needs DASHSCOPE_API_KEY)")
    cases = json.loads(PARAPHRASED.read_text())["cases"]
    if args.limit:
        cases = cases[args.offset:args.offset + args.limit]

    rows = []
    for case in cases:
        for config in CONFIGS:
            row = run_case(case, config, budget=args.budget, live=args.live)
            rows.append(row)
            print(f"{case['id']:>10} {config:<14} fama={row['fama']}")
    summary = {
        "by_config": aggregate(rows),
        "paired_mcnemar": {
            f"quen_vs_{b}": mcnemar_exact(rows, "quen", b)
            for b in ("append_only", "full_context")
        },
        "mode": "live" if args.live else "dry-run",
    }
    write_json(args.out / "paraphrase_results.json", rows)
    write_json(args.out / "paraphrase_summary.json", summary)
    print(json.dumps({k: v.get("fama") for k, v in summary["by_config"].items()},
                     indent=2))
    return summary


if __name__ == "__main__":
    main()
