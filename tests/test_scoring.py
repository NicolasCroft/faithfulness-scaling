import math

from pipeline.scoring import FaithfulnessResult, min_n_for_margin, wilson_ci


def test_wilson_ci_midpoint_reasonable():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.40 < lo < 0.42  # standard Wilson result for 50/100 at 95%
    assert 0.58 < hi < 0.60


def test_wilson_ci_boundary_zero_successes_stays_nonnegative():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_ci_boundary_all_successes_stays_le_one():
    lo, hi = wilson_ci(20, 20)
    assert hi <= 1.0
    assert hi > 0.99  # right at the boundary, modulo floating-point error
    assert lo < 1.0


def test_wilson_ci_zero_n_returns_nan():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_ci_rejects_invalid_successes():
    import pytest

    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_faithfulness_result_rate_and_ci():
    r = FaithfulnessResult("model-a", "truncate", n_problems=100, n_answer_changed=80)
    assert r.rate == 0.8
    lo, hi = r.ci
    assert lo < 0.8 < hi


def test_faithfulness_result_zero_problems_rate_is_nan():
    r = FaithfulnessResult("model-a", "truncate", n_problems=0, n_answer_changed=0)
    assert math.isnan(r.rate)


def test_min_n_for_margin_matches_rule_of_thumb():
    # For phat=0.5, margin=0.05, z=1.96, the classic rule-of-thumb answer
    # is ~384.
    n = min_n_for_margin(0.05)
    assert 380 <= n <= 390


def test_min_n_for_margin_smaller_margin_needs_more_n():
    n_loose = min_n_for_margin(0.10)
    n_tight = min_n_for_margin(0.03)
    assert n_tight > n_loose
