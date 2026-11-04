"""Tests for the Monte Carlo pricer (src/optionlab/monte_carlo.py).

A Monte Carlo price is random, so "correct" means: close to the true price, measured in
standard errors (SE). With a fixed seed every test here is deterministic. A 4-SE miss would
happen by pure chance only about 6 times in 100,000.
"""

import numpy as np
import pytest

from optionlab import MCResult, bs_price, mc_price

# Reference option from Phase 1: S = K = 100, T = 1, r = 5%, sigma = 20%.
REF = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)


def within_4_se(result: MCResult, true_price: float) -> bool:
    return abs(result.price - true_price) < 4 * result.std_error


# --- Reference values (Hull / Black-Scholes) -------------------------------------------


@pytest.mark.parametrize(
    "q, option, true_price",
    [
        (0.0, "call", 10.4506),
        (0.02, "call", 9.2270),
        (0.02, "put", 6.3301),
    ],
)
def test_reference_values_within_4_standard_errors(q, option, true_price):
    result = mc_price(**REF, q=q, option=option, paths=200_000, seed=42)
    assert within_4_se(result, true_price), result


def test_result_fields():
    result = mc_price(**REF, paths=10_000, seed=1)
    assert isinstance(result, MCResult)
    assert result.paths == 10_000
    assert result.std_error > 0
    price, std_error, paths = result  # a NamedTuple unpacks like a tuple
    assert price == result.price


def test_standard_error_is_the_expected_size():
    # Theory: SE = (std of the discounted payoff) / sqrt(paths). For the reference call that
    # std is about 14.7, so at 100,000 paths SE is about 0.047 (independent of the seed).
    result = mc_price(**REF, paths=100_000, seed=3)
    assert 0.040 < result.std_error < 0.055


# --- Is the standard error itself right? ------------------------------------------------
# The tests above only check that the price is within a few SE. If the SE were too big,
# they would pass too easily. These two tests check the SE against theory.

BS_REF = 10.4506


@pytest.mark.parametrize(
    "q, option, true_price",
    [(0.0, "call", BS_REF), (0.02, "put", 6.3301)],
)
def test_95_percent_intervals_contain_the_true_price_95_percent_of_the_time(q, option, true_price):
    # If the SE is right, price ± 1.96 SE contains the true price in 95% of runs. Over 1,000
    # runs the share has noise of sqrt(0.95 * 0.05 / 1000) ≈ 0.007 (binomial), so [0.93, 0.97]
    # is about ±3 standard deviations. An SE that is 10% too small covers only ~92.5% and
    # fails; one that is too big needs to be ~15% off to fail (it covers ~97.6%).
    runs = 1_000
    hits = 0
    for seed in range(runs):
        result = mc_price(**REF, q=q, option=option, paths=10_000, seed=seed)
        hits += abs(result.price - true_price) <= 1.96 * result.std_error
    coverage = hits / runs
    assert 0.93 <= coverage <= 0.97, f"coverage {coverage:.3f}"


def test_standard_error_shrinks_like_one_over_sqrt_paths():
    # Theory: SE = std / sqrt(n), so on a log-log plot SE against paths is a line of slope -0.5.
    # (This mirrors the tree's 1/N slope test in test_binomial.py.)
    paths = np.logspace(3, 6, 10).astype(int)
    errors = [mc_price(**REF, paths=int(n), seed=11).std_error for n in paths]
    slope = np.polyfit(np.log(paths), np.log(errors), 1)[0]
    assert -0.55 <= slope <= -0.45, f"fitted slope {slope:.3f}"


# --- Reproducibility --------------------------------------------------------------------


def test_same_seed_gives_identical_result():
    assert mc_price(**REF, paths=10_000, seed=7) == mc_price(**REF, paths=10_000, seed=7)


def test_different_seed_gives_different_price():
    assert mc_price(**REF, paths=10_000, seed=7).price != mc_price(**REF, paths=10_000, seed=8).price


# --- Edge cases: no randomness left, so the answer is exact -----------------------------


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_expiry_gives_intrinsic_value_with_zero_error(S, option):
    result = mc_price(S, 100, 0.0, 0.05, 0.2, option=option, paths=1_000, seed=0)
    assert result.price == pytest.approx(bs_price(S, 100, 0.0, 0.05, 0.2, option=option), abs=1e-12)
    assert result.std_error == 0


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_zero_volatility_gives_deterministic_value_with_zero_error(S, option):
    result = mc_price(S, 100, 1.0, 0.05, 0.0, q=0.02, option=option, paths=1_000, seed=0)
    expected = bs_price(S, 100, 1.0, 0.05, 0.0, q=0.02, option=option)
    assert result.price == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Every payoff is the same float x, but np.mean of 1,000 copies of x can be off from x
    # by one rounding step, so the SE is ~1e-16 (rounding noise), not exactly 0.
    assert result.std_error <= 1e-12 * max(expected, 1.0)


# --- Bad inputs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(S=-1), "Spot price S"),
        (dict(K=0), "Strike price K"),
        (dict(T=-0.1), "Time to expiry T"),
        (dict(sigma=-0.2), "Volatility sigma"),
        (dict(r=np.nan), "finite"),
        (dict(option="straddle"), "option must be"),
        (dict(S=np.array([100.0, 110.0])), "scalar inputs only"),
        (dict(paths=1), "paths must be an integer >= 2"),
        (dict(paths=1000.0), "paths must be an integer >= 2"),
        (dict(paths=True), "paths must be an integer >= 2"),
    ],
)
def test_bad_inputs_raise_value_error(kwargs, message):
    args = dict(REF, paths=1_000, seed=0) | kwargs
    with pytest.raises(ValueError, match=message):
        mc_price(**args)
