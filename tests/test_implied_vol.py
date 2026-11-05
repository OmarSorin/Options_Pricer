"""Tests for implied_vol: find the sigma that makes bs_price match a given price.

References outside this project: Hull's implied volatility example, and exact put-call
parity (which turns a call price into a put price without using bs_price).
"""

import numpy as np
import pytest

from optionlab import bs_price, implied_vol, vega


def test_hull_example():
    # Hull, "Options, Futures, and Other Derivatives": a European call on a non-dividend
    # stock worth 1.875 with S = 21, K = 20, r = 10%, T = 0.25 has implied volatility 0.235
    # (23.5%, rounded). Checked: bs_price gives 1.8733 at 0.234 and 1.8766 at 0.235.
    sigma = implied_vol(1.875, S=21, K=20, T=0.25, r=0.1)
    assert sigma == pytest.approx(0.235, abs=5e-4)
    assert bs_price(21, 20, 0.25, 0.1, sigma) == pytest.approx(1.875, abs=1e-9)


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "sigma", "q", "option"),
    [
        (100, 100, 1, 0.05, 0.2, 0.0, "call"),  # reference case, price 10.4506
        (100, 100, 1, 0.05, 0.2, 0.02, "call"),  # 9.2270
        (100, 100, 1, 0.05, 0.2, 0.02, "put"),  # 6.3301
        (100, 150, 1, 0.05, 0.2, 0.0, "call"),  # out of the money: plain Newton fails here
        (100, 50, 1, 0.05, 0.2, 0.0, "put"),  # deep out of the money, price 0.00033
        (100, 80, 0.5, 0.05, 0.6, 0.0, "call"),  # in the money, high volatility
        (100, 100, 2, -0.01, 0.15, 0.03, "put"),  # negative rate
        (50, 55, 1 / 52, 0.03, 0.9, 0.0, "call"),  # one week, very high volatility
    ],
)
def test_round_trip_recovers_sigma(S, K, T, r, sigma, q, option):
    price = bs_price(S, K, T, r, sigma, q, option)
    assert implied_vol(price, S, K, T, r, q, option) == pytest.approx(sigma, abs=1e-8)


def test_reference_values_from_the_textbook_prices():
    # The 4-decimal reference prices from CLAUDE.md give back sigma = 20% to about 1e-5
    # (they are rounded, so not exactly 0.2).
    assert implied_vol(10.4506, 100, 100, 1, 0.05) == pytest.approx(0.2, abs=1e-5)
    assert implied_vol(9.2270, 100, 100, 1, 0.05, q=0.02) == pytest.approx(0.2, abs=1e-5)
    assert implied_vol(6.3301, 100, 100, 1, 0.05, q=0.02, option="put") == pytest.approx(0.2, abs=1e-5)


def test_call_and_put_give_the_same_sigma():
    # Put-call parity, put = call - (S*exp(-qT) - K*exp(-rT)), builds the put price without
    # bs_price. Both prices describe the same volatility, so both must invert to it.
    S, K, T, r, q = 100, 110, 0.75, 0.04, 0.01
    call = bs_price(S, K, T, r, 0.3, q, "call")
    put = call - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert implied_vol(put, S, K, T, r, q, "put") == pytest.approx(implied_vol(call, S, K, T, r, q, "call"), abs=1e-8)


def plain_newton(price, S, K, T, r):
    """Newton-Raphson with no bracket and no fallback, from the same starting guess."""
    sigma = np.sqrt(2 * np.pi / T) * price / S  # Brenner-Subrahmanyam: good at the money only
    for _ in range(50):
        sigma -= (bs_price(S, K, T, r, sigma) - price) / vega(S, K, T, r, sigma)
    return sigma


def test_bisection_fallback_rescues_newton():
    # For this out-of-the-money call the starting guess is about 0.009, far below 0.2. There
    # vega underflows to 0, so plain Newton divides by zero on its very first step.
    price = bs_price(100, 150, 1, 0.05, 0.2)
    with pytest.raises(ZeroDivisionError):
        plain_newton(price, 100, 150, 1, 0.05)
    assert implied_vol(price, 100, 150, 1, 0.05) == pytest.approx(0.2, abs=1e-8)


def test_returns_a_plain_float():
    assert type(implied_vol(10.4506, 100, 100, 1, 0.05)) is float


# --- Prices with no implied volatility -------------------------------------------------
# A call is worth between max(S*exp(-qT) - K*exp(-rT), 0) and S*exp(-qT); a put between
# max(K*exp(-rT) - S*exp(-qT), 0) and K*exp(-rT). Outside that range no sigma fits.


@pytest.mark.parametrize("price", [100.0, 150.0])  # call upper bound is S = 100
def test_call_price_at_or_above_upper_bound_raises(price):
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(price, 100, 100, 1, 0.05)


def test_put_price_at_or_above_upper_bound_raises():
    upper = 100 * np.exp(-0.05)  # K * exp(-rT)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(upper, 100, 100, 1, 0.05, option="put")


def test_price_exactly_on_lower_bound_raises():
    # Worth exactly its exercise value today: sigma would be 0, and the price says
    # nothing useful about volatility, so we refuse (like the Greeks at sigma == 0).
    lower = 120 - 100 * np.exp(-0.05)  # in-the-money call: S - K*exp(-rT)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(lower, 120, 100, 1, 0.05)


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_zero_or_negative_price_raises(price):
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(price, 100, 100, 1, 0.05)


# --- Other bad inputs ------------------------------------------------------------------


def test_expiry_zero_raises():
    with pytest.raises(ValueError, match="T == 0"):
        implied_vol(5.0, 100, 100, 0.0, 0.05)


@pytest.mark.parametrize("price", [np.nan, np.inf])
def test_non_finite_price_raises(price):
    with pytest.raises(ValueError, match="finite"):
        implied_vol(price, 100, 100, 1, 0.05)


@pytest.mark.parametrize(
    ("S", "K", "T"),
    [(0.0, 100, 1), (100, -1.0, 1), (100, 100, -1.0)],
)
def test_usual_input_rules_apply(S, K, T):
    with pytest.raises(ValueError):
        implied_vol(5.0, S, K, T, 0.05)


def test_unknown_option_type_raises():
    with pytest.raises(ValueError, match="option"):
        implied_vol(5.0, 100, 100, 1, 0.05, option="straddle")


def test_array_inputs_raise():
    with pytest.raises(ValueError, match="scalar"):
        implied_vol(np.array([5.0, 6.0]), 100, 100, 1, 0.05)


def test_not_converging_within_max_iter_raises():
    # One iteration cannot reach tol = 1e-10 from the starting guess.
    with pytest.raises(ValueError, match="did not converge"):
        implied_vol(10.4506, 100, 100, 1, 0.05, max_iter=1)
