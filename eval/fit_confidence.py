#!/usr/bin/env python3
"""Fit the answer-confidence calibration knots from benchmark-scale data.

Reads quen rows (JSON or JSONL) that carry `answer_confidence` and the
LLM-judge verdict `correct`, pools them, runs isotonic regression (PAVA),
and interpolates 5 monotone knots for `trust._CALIBRATION_KNOTS`. Prints
knots plus ECE before/after ON THE FIT SET — the honest generalization
check is a HELD-OUT probe run, not this number.

  .venv/bin/python eval/fit_confidence.py eval/out/longmemeval_s_results.json \
      eval/out/abstention_recheck.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_pairs(paths: list[str]) -> list[tuple[float, int]]:
    pairs = []
    for p in paths:
        text = Path(p).read_text()
        rows = (json.loads(text) if text.lstrip().startswith("[")
                else [json.loads(l) for l in text.splitlines() if l.strip()])
        for r in rows:
            conf = r.get("answer_confidence")
            if r.get("config", "quen") == "quen" and conf is not None:
                pairs.append((float(conf), int(bool(r["correct"]))))
    return sorted(pairs)


def pava(pairs: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators: monotone fit of accuracy over confidence."""
    blocks = [[x, float(y), 1.0] for x, y in pairs]  # [x, mean, weight]
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            x2, m2, w2 = merged.pop()
            x1, m1, w1 = merged.pop()
            merged.append([x2, (m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return [(b[0], b[1]) for b in merged]


def knots_from_fit(fit: list[tuple[float, float]],
                   xs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[tuple[float, float]]:
    def value_at(x: float) -> float:
        prev = (0.0, 0.0)
        for fx, fy in fit:
            if fx >= x:
                return fy
            prev = (fx, fy)
        return prev[1]

    ys = [round(min(1.0, max(0.0, value_at(x))), 2) for x in xs]
    ys[0] = 0.0
    ys[-1] = 1.0
    for i in range(1, len(ys)):  # enforce monotone after rounding/anchoring
        ys[i] = max(ys[i], ys[i - 1])
    return list(zip(xs, ys))


def ece(pairs, transform=lambda x: x, bins: int = 10) -> float:
    grid: dict[int, list[tuple[float, int]]] = {}
    for c, y in pairs:
        grid.setdefault(min(bins - 1, int(transform(c) * bins)), []).append((transform(c), y))
    n = len(pairs)
    return sum(
        len(v) / n * abs(sum(c for c, _ in v) / len(v) - sum(y for _, y in v) / len(v))
        for v in grid.values()
    )


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    pairs = load_pairs(paths)
    print(f"{len(pairs)} (confidence, verdict) pairs from {len(paths)} files")
    fit = pava(pairs)
    knots = knots_from_fit(fit)

    def interp(x: float) -> float:
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return 1.0

    print("knots:", knots)
    print(f"ECE on fit set: raw={ece(pairs):.4f} calibrated={ece(pairs, interp):.4f}")
    print("(generalization check = held-out probe run, not this number)")


if __name__ == "__main__":
    main()
