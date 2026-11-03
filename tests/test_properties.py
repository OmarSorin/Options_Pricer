"""Property-based tests: hypothesis generates random valid inputs and checks rules
that must ALWAYS hold, instead of only the grid points we thought of."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from optionlab import bs_price, delta, gamma, numerical_greeks, rho, theta, vega

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
