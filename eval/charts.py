#!/usr/bin/env python3
"""Render the eval charts (matplotlib → eval/out/*.png).

Reads whatever eval/out JSONs exist (probe summary, budget curve,
calibration) and skips charts whose inputs are missing.

  uv pip install -e ".[eval]"   # matplotlib
  .venv/bin/python eval/charts.py [--db data/demo.db]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import OUT_DIR  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib missing — install with: uv pip install -e '.[eval]'")

# neutral, color-blind-safe series colors
COLORS = {
    "quen": "#4f46e5",
    "quen_no_verify": "#818cf8",
    "append_only": "#9ca3af",
    "full_context": "#d1d5db",
}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def chart_fama(out: Path) -> None:
    summary = _load(out / "summary.json")
    if not summary or "by_config" not in summary:
        return
    by_config = summary["by_config"]
    configs = list(by_config)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, offset in (("fama", -0.2), ("presence", 0.0), ("absence", 0.2)):
        vals = [by_config[c][metric] for c in configs]
        ax.bar(
            [i + offset for i in range(len(configs))],
            vals,
            width=0.2,
            label=metric,
        )
    ax.set_xticks(range(len(configs)), configs, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Staleness probe — FAMA ({summary.get('mode', '?')}, "
                 f"budget {summary.get('budget', '?')})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fama_by_config.png", dpi=150)
    print(f"wrote {out / 'fama_by_config.png'}")


def chart_budget_curve(out: Path) -> None:
    curve = _load(out / "budget_curve.json")
    if not curve:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    configs = sorted({r["config"] for r in curve})
    for config in configs:
        rows = sorted((r for r in curve if r["config"] == config),
                      key=lambda r: r["budget"])
        ax.plot(
            [r["budget"] for r in rows],
            [r["fama"] for r in rows],
            marker="o",
            label=config,
            color=COLORS.get(config),
        )
    ax.set_xscale("log")
    ax.set_xlabel("token budget")
    ax.set_ylabel("FAMA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs budget (probe)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "budget_curve.png", dpi=150)
    print(f"wrote {out / 'budget_curve.png'}")


def chart_calibration(out: Path) -> None:
    cal = _load(out / "calibration.json")
    if not cal:
        return
    # retention reliability
    bins = cal["retention"]["bins"]
    if bins:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfect")
        ax.scatter(
            [b["predicted_mean"] for b in bins],
            [b["empirical"] for b in bins],
            s=[20 + 6 * b["count"] for b in bins],
            color="#4f46e5",
        )
        ax.set_xlabel("predicted R at self-test")
        ax.set_ylabel("empirical recall")
        ax.set_title(
            f"Retention calibration (ECE {cal['retention']['ece']}, "
            f"log-loss {cal['retention']['log_loss']})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "retention_calibration.png", dpi=150)
        print(f"wrote {out / 'retention_calibration.png'}")
    # confidence by freshness
    strata = cal["confidence"]["by_freshness"]
    if any(s["count"] for s in strata):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.35
        labels = [s["freshness_bucket"] for s in strata]
        pred = [
            (sum(b["predicted_mean"] * b["count"] for b in s["bins"])
             / sum(b["count"] for b in s["bins"])) if s["count"] else 0
            for s in strata
        ]
        emp = [
            (sum(b["empirical"] * b["count"] for b in s["bins"])
             / sum(b["count"] for b in s["bins"])) if s["count"] else 0
            for s in strata
        ]
        x = range(len(labels))
        ax.bar([i - width / 2 for i in x], pred, width, label="stated confidence",
               color="#4f46e5")
        ax.bar([i + width / 2 for i in x], emp, width, label="empirical accuracy",
               color="#9ca3af")
        for i, s in enumerate(strata):
            ax.text(i, max(pred[i], emp[i]) + 0.03,
                    f"ECE {s['ece']}" if s["ece"] is not None else "—",
                    ha="center", fontsize=8)
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0, 1.15)
        ax.set_title("Confidence calibration by memory freshness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "confidence_by_freshness.png", dpi=150)
        print(f"wrote {out / 'confidence_by_freshness.png'}")


def chart_long_horizon(out: Path) -> None:
    data = _load(out / "long_horizon.json")
    if not data:
        return
    weeks = [r["week"] for r in data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(weeks, [r["store_tokens_append_only"] for r in data],
             label="append-only store", color="#9ca3af")
    ax1.plot(weeks, [r["store_tokens_quen_active"] for r in data],
             label="Quên active set", color="#4f46e5")
    ax1.set_xlabel("week")
    ax1.set_ylabel("store footprint (tokens)")
    ax1.set_title("What a full-context agent would carry")
    ax1.legend()
    ax2.plot(weeks, [r["fama_append_only"] for r in data],
             label="append-only", color="#9ca3af")
    ax2.plot(weeks, [r["fama_quen"] for r in data],
             label="Quên", color="#4f46e5")
    ax2.set_xlabel("week")
    ax2.set_ylabel("FAMA at fixed budget")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Answer quality as memory ages")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "long_horizon.png", dpi=150)
    print(f"wrote {out / 'long_horizon.png'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    chart_fama(args.out)
    chart_budget_curve(args.out)
    chart_calibration(args.out)
    chart_long_horizon(args.out)


if __name__ == "__main__":
    main()
