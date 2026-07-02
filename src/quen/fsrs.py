"""FSRS retention core — equations imported verbatim from FSRS-4.5.

Source: the canonical scheduler repo,
https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler
(FSRS-4.5; power forgetting curve ``R = (1 + FACTOR*t/S) ** DECAY``).
Default weights are the published FSRS-4.5 defaults; they can be re-fit but
we use them as shipped.

Quên uses the DSR (Difficulty, Stability, Retrievability) state as an
*eviction and ranking prior*, not a scheduler — we do not control when
reviews happen. Reviews arrive from three places (spec §4.5):
  (a) a memory used in an answer later judged correct/uncontradicted → Good;
  (b) a dream self-test pass → Good / fail → Again;
  (c) a verification pass → Good / fail (refuted) → Again.
"""

from __future__ import annotations

import math
from enum import IntEnum

VERSION = "FSRS-4.5"

# FSRS-4.5 default weights w0..w16 (open-spaced-repetition defaults).
DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.4872, 1.4003, 3.7145, 13.8206, 5.1618, 1.2298, 0.8975, 0.031,
    1.6474, 0.1367, 1.0461, 2.1072, 0.0793, 0.3246, 1.587, 0.2272, 2.8755,
)

# Power forgetting curve constants (FSRS-4.5): R(t=S) = 0.9 by construction.
DECAY = -0.5
FACTOR = 19.0 / 81.0

D_MIN, D_MAX = 1.0, 10.0
S_MIN = 0.01


class Grade(IntEnum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


def retrievability(elapsed_days: float, stability: float) -> float:
    """R(t, S) = (1 + FACTOR * t / S) ** DECAY — probability of recall."""
    if stability <= 0:
        return 0.0
    t = max(0.0, elapsed_days)
    return (1.0 + FACTOR * t / stability) ** DECAY


def days_until_retrievability(target_r: float, stability: float) -> float:
    """Invert the forgetting curve: elapsed days at which R drops to target_r.

    Used for the dashboard's "forgotten in ~N days" and for eviction planning.
    """
    if not (0.0 < target_r < 1.0):
        raise ValueError("target_r must be in (0, 1)")
    return stability / FACTOR * (target_r ** (1.0 / DECAY) - 1.0)


def init_stability(grade: Grade, w: tuple[float, ...] = DEFAULT_WEIGHTS) -> float:
    """S0(G) = w[G-1]."""
    return max(S_MIN, w[int(grade) - 1])


def init_difficulty(grade: Grade, w: tuple[float, ...] = DEFAULT_WEIGHTS) -> float:
    """D0(G) = w4 - (G - 3) * w5, clamped to [1, 10]."""
    return _clamp_d(w[4] - (int(grade) - 3) * w[5])


def next_difficulty(
    d: float, grade: Grade, w: tuple[float, ...] = DEFAULT_WEIGHTS
) -> float:
    """D' = w7 * D0(Easy) + (1 - w7) * (D - w6 * (G - 3))  (mean reversion)."""
    delta = d - w[6] * (int(grade) - 3)
    return _clamp_d(w[7] * init_difficulty(Grade.EASY, w) + (1.0 - w[7]) * delta)


def next_stability_on_recall(
    d: float,
    s: float,
    r: float,
    grade: Grade,
    w: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> float:
    """Success: S' = S * (1 + e^w8 * (11-D) * S^-w9 * (e^(w10*(1-R)) - 1) * hard * easy).

    The (1-R) term is the spacing effect: success at low R grows S more.
    """
    hard_penalty = w[15] if grade == Grade.HARD else 1.0
    easy_bonus = w[16] if grade == Grade.EASY else 1.0
    growth = (
        math.exp(w[8])
        * (11.0 - d)
        * s ** (-w[9])
        * (math.exp(w[10] * (1.0 - r)) - 1.0)
        * hard_penalty
        * easy_bonus
    )
    return max(S_MIN, s * (1.0 + growth))


def next_stability_on_forget(
    d: float, s: float, r: float, w: tuple[float, ...] = DEFAULT_WEIGHTS
) -> float:
    """Failure: S'_f = w11 * D^-w12 * ((S+1)^w13 - 1) * e^(w14*(1-R)), capped at S."""
    s_forget = (
        w[11]
        * d ** (-w[12])
        * ((s + 1.0) ** w[13] - 1.0)
        * math.exp(w[14] * (1.0 - r))
    )
    return max(S_MIN, min(s_forget, s))


def review(
    d: float,
    s: float,
    elapsed_days: float,
    grade: Grade,
    w: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> tuple[float, float]:
    """Apply one review; returns (D', S').

    R is computed at review time from the elapsed interval, then D and S are
    stepped with the FSRS-4.5 update rules.
    """
    r = retrievability(elapsed_days, s)
    d2 = next_difficulty(d, grade, w)
    if grade == Grade.AGAIN:
        s2 = next_stability_on_forget(d, s, r, w)
    else:
        s2 = next_stability_on_recall(d, s, r, grade, w)
    return d2, s2


def initial_state(
    grade: Grade = Grade.GOOD, w: tuple[float, ...] = DEFAULT_WEIGHTS
) -> tuple[float, float]:
    """(D0, S0) for a first observation graded `grade`."""
    return init_difficulty(grade, w), init_stability(grade, w)


def _clamp_d(d: float) -> float:
    return min(D_MAX, max(D_MIN, d))
