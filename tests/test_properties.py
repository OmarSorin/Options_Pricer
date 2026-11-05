"""Property-based tests: hypothesis generates random valid inputs and checks rules
that must ALWAYS hold, instead of only the grid points we thought of."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from optionlab import bs_price, d1_d2, delta, gamma, implied_vol, mc_price, numerical_greeks, rho, theta, vega

pytestmark = pytest.mark.filterwarnings("error")

# Realistic input ranges. Extreme corners (e.g. sigma*sqrt(T) ~ 1e-6) are covered
# by the explicit edge-case tests; here we want broad, sensible coverage.
spot = st.floats(1.0, 1000.0)
strike = st.floats(1.0, 1000.0)
expiry = st.floats(1 / 365, 10.0)
rate = st.floats(-0.02, 0.15)
vol = st.floats(0.01, 2.0)
div = st.floats(0.0, 0.10)
option_type = st.sampled_from(["call", "put"])

SETTINGS = settings()  # profile from tests/conftest.py


@SETTINGS
@given(spot, strike, expiry, rate, vol, div)
def test_property_put_call_parity(S, K, T, r, sigma, q):
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    forward_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call - put == pytest.approx(forward_value, rel=1e-9, abs=1e-9 * (S + K))


@SETTINGS
@given(spot, strike, expiry, rate, vol, div)
def test_property_no_arbitrage_bounds(S, K, T, r, sigma, q):
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    tol = 1e-9 * (S + K)
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert max(spot_pv - strike_pv, 0.0) - tol <= call <= spot_pv + tol
    assert max(strike_pv - spot_pv, 0.0) - tol <= put <= strike_pv + tol


@SETTINGS
@given(spot, strike, expiry, rate, vol, div, option_type)
def test_property_black_scholes_pde(S, K, T, r, sigma, q, option):
    residual = (
        theta(S, K, T, r, sigma, q, option)
        + (r - q) * S * delta(S, K, T, r, sigma, q, option)
        + 0.5 * sigma**2 * S**2 * gamma(S, K, T, r, sigma, q)
        - r * bs_price(S, K, T, r, sigma, q, option)
    )
    assert residual == pytest.approx(0.0, abs=1e-9 * (S + K))


@SETTINGS
@given(spot, strike, expiry, rate, vol, div, option_type)
def test_property_analytic_matches_numerical(S, K, T, r, sigma, q, option):
    # Below sigma*sqrt(T) = 0.05 the price distribution is so narrow that no bump
    # size gives a reliable finite-difference gamma in double precision (a small
    # bump drowns in rounding error, a big one in truncation error). The analytic
    # Greeks are checked over the FULL range against QuantLib instead
    # (test_quantlib_crosscheck.py::test_property_matches_quantlib).
    assume(sigma * np.sqrt(T) >= 0.05)
    # Compare "cash Greeks" (in price units: delta*S, gamma*S^2) so that one
    # tolerance proportional to the size of the trade (S + K) fits every Greek.
    # Finite-difference rounding error grows with the price level, so a fixed
    # absolute tolerance is meaningless when, say, a put is worth 256.
    numeric = numerical_greeks(S, K, T, r, sigma, q, option)
    analytic = {
        "delta": delta(S, K, T, r, sigma, q, option),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, option),
        "rho": rho(S, K, T, r, sigma, q, option),
    }
    to_cash = {"delta": S, "gamma": S**2, "vega": 1.0, "theta": 1.0, "rho": 1.0}
    for name, value in analytic.items():
        assert value * to_cash[name] == pytest.approx(
            numeric[name] * to_cash[name], rel=1e-4, abs=1e-6 * (S + K)
        ), name


# --- Monte Carlo ------------------------------------------------------------------------
# A Monte Carlo price is random, so these properties hold "within a few standard errors".
# Fixed seeds keep every example deterministic; with 5 SE a pure-chance miss has
# probability about 6e-7 per example.

MC_PATHS = 20_000


def assume_enough_paths_in_the_money(S, K, T, r, sigma, q, option):
    # When fewer than 1% of paths finish in the money, the SE estimate itself is
    # unreliable (it can be 0 while the true price is > 0), so "within 5 SE" means
    # nothing there. Deep out-of-the-money options are left to the Black-Scholes tests.
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    assume(norm.cdf(d2 if option == "call" else -d2) >= 0.01)
    # Above sigma*sqrt(T) ~ 2 the stock price at expiry is so skewed (relative variance
    # e^(sigma^2 T) - 1 > 54) that a few rare, huge paths carry most of the price. A typical
    # run misses them, so the price comes out too low AND the sample SE is too small.
    # Example: sigma*sqrt(T) = 4.6, the top 0.01% of paths give 77% of the payoff, and
    # 2 million paths still give 0.64-0.89 against the true 0.978. This is a limit of plain
    # Monte Carlo, not a bug; it is listed in the README's Known limitations.
    assume(sigma * np.sqrt(T) <= 2)


@SETTINGS
@given(spot, strike, expiry, rate, vol, div, option_type, st.booleans())
def test_property_monte_carlo_matches_black_scholes(S, K, T, r, sigma, q, option, antithetic):
    assume_enough_paths_in_the_money(S, K, T, r, sigma, q, option)
    result = mc_price(S, K, T, r, sigma, q, option, paths=MC_PATHS, seed=0, antithetic=antithetic)
    exact = bs_price(S, K, T, r, sigma, q, option)
    assert abs(result.price - exact) <= 5 * result.std_error, (result, exact)


@SETTINGS
@given(spot, strike, expiry, rate, vol, div)
def test_property_monte_carlo_put_call_parity(S, K, T, r, sigma, q):
    assume_enough_paths_in_the_money(S, K, T, r, sigma, q, "call")
    assume_enough_paths_in_the_money(S, K, T, r, sigma, q, "put")
    # DIFFERENT seeds for the call and the put. With the same draws they would be
    # negatively correlated, and sqrt(SE_c^2 + SE_p^2) would understate the error of C - P.
    call = mc_price(S, K, T, r, sigma, q, "call", paths=MC_PATHS, seed=1)
    put = mc_price(S, K, T, r, sigma, q, "put", paths=MC_PATHS, seed=2)
    forward_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    combined_se = np.hypot(call.std_error, put.std_error)
    assert abs((call.price - put.price) - forward_value) <= 5 * combined_se


# --- Implied volatility -----------------------------------------------------------------

EPS = np.finfo(float).eps  # 2.2e-16, the relative rounding step of a float


@SETTINGS
@given(spot, strike, expiry, rate, vol, div, option_type)
def test_property_implied_vol_round_trip(S, K, T, r, sigma, q, option):
    price = bs_price(S, K, T, r, sigma, q, option)
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(spot_pv - strike_pv, 0.0) if option == "call" else max(strike_pv - spot_pv, 0.0)
    # Deep in the money the time value can be so small that the price rounds to exactly
    # its lower bound; then no sigma fits and implied_vol raises, correctly.
    assume(price > lower)

    found = implied_vol(price, S, K, T, r, q, option)

    # 1. Always: the volatility found reproduces the price.
    assert bs_price(S, K, T, r, found, q, option) == pytest.approx(price, rel=1e-9, abs=1e-9 * (S + K))
    # 2. Where the price still determines sigma: rounding the price (relative step EPS)
    #    moves sigma by about EPS * price / vega. Deep in the money (a large price that is
    #    almost all intrinsic value, vega ~1e-14) that is 0.01 or more, so the price simply
    #    does not contain sigma to 8 digits. There only check 1 is meaningful.
    assume(EPS * price / vega(S, K, T, r, sigma, q) < 1e-10)
    assert found == pytest.approx(sigma, abs=1e-8)
