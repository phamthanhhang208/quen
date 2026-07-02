"""DSR monotonicity and FSRS-4.5 behavior (spec §8 TDD: DSR monotonicity)."""

import math

import pytest

from quen import fsrs
from quen.fsrs import Grade


def test_retrievability_monotonically_decays():
    s = 10.0
    values = [fsrs.retrievability(t, s) for t in [0, 1, 5, 10, 30, 90, 365]]
    assert values[0] == 1.0
    for earlier, later in zip(values, values[1:]):
        assert later < earlier


def test_retrievability_is_090_at_t_equals_s():
    for s in [0.5, 1.0, 7.0, 42.0]:
        assert fsrs.retrievability(s, s) == pytest.approx(0.9, abs=1e-9)


def test_days_until_retrievability_inverts_curve():
    s = 12.0
    for target in [0.9, 0.7, 0.5, 0.3]:
        t = fsrs.days_until_retrievability(target, s)
        assert fsrs.retrievability(t, s) == pytest.approx(target, abs=1e-9)


def test_good_review_raises_stability():
    d, s = fsrs.initial_state(Grade.GOOD)
    d2, s2 = fsrs.review(d, s, elapsed_days=3.0, grade=Grade.GOOD)
    assert s2 > s


def test_again_review_lowers_stability_and_raises_difficulty():
    d, s = 5.0, 20.0
    d2, s2 = fsrs.review(d, s, elapsed_days=10.0, grade=Grade.AGAIN)
    assert s2 < s
    assert d2 > d


def test_spacing_effect_success_at_low_r_grows_s_more():
    d, s = 5.0, 10.0
    # low R: long elapsed; high R: short elapsed
    t_low_r = fsrs.days_until_retrievability(0.5, s)
    t_high_r = fsrs.days_until_retrievability(0.95, s)
    _, s_low = fsrs.review(d, s, elapsed_days=t_low_r, grade=Grade.GOOD)
    _, s_high = fsrs.review(d, s, elapsed_days=t_high_r, grade=Grade.GOOD)
    assert s_low > s_high > s


def test_same_instant_review_is_noop_for_stability():
    # documented FSRS property: R(0)=1 -> growth term is zero
    d, s = 5.0, 10.0
    _, s2 = fsrs.review(d, s, elapsed_days=0.0, grade=Grade.GOOD)
    assert s2 == pytest.approx(s)


def test_difficulty_stays_clamped():
    d = 10.0
    for _ in range(10):
        d, _ = fsrs.review(d, 5.0, elapsed_days=1.0, grade=Grade.AGAIN)
    assert fsrs.D_MIN <= d <= fsrs.D_MAX
    d = 1.0
    for _ in range(10):
        d, _ = fsrs.review(d, 5.0, elapsed_days=1.0, grade=Grade.EASY)
    assert fsrs.D_MIN <= d <= fsrs.D_MAX


def test_forget_stability_never_exceeds_previous():
    d, s = 5.0, 50.0
    for elapsed in [1.0, 10.0, 100.0]:
        s_f = fsrs.next_stability_on_forget(d, s, fsrs.retrievability(elapsed, s))
        assert s_f <= s


def test_initial_state_uses_published_weights():
    d0, s0 = fsrs.initial_state(Grade.GOOD)
    assert s0 == pytest.approx(fsrs.DEFAULT_WEIGHTS[2])
    assert d0 == pytest.approx(fsrs.DEFAULT_WEIGHTS[4])
    assert fsrs.VERSION == "FSRS-4.5"
    assert fsrs.DECAY == -0.5
    assert fsrs.FACTOR == pytest.approx(19.0 / 81.0)


def test_stability_floor_positive():
    d, s = 10.0, 0.02
    _, s2 = fsrs.review(d, s, elapsed_days=100.0, grade=Grade.AGAIN)
    assert s2 >= fsrs.S_MIN
    assert not math.isnan(s2)
