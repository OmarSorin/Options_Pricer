"""Cross-check optionlab against QuantLib, an independent, widely used pricing library.

Skipped automatically if QuantLib is not installed (`pip install -e ".[validation]"`).

Conventions were checked before writing this file: QuantLib's AnalyticEuropeanEngine
returns vega and rho per 1.00 change, and theta() per YEAR as -dP/dT (thetaPerDay() is
theta / 365). These match optionlab exactly, so no conversion is applied.
QuantLib's expiry is a whole number of days, so T is always days / 365 here.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ql = pytest.importorskip("QuantLib")

from optionlab import bs_price, delta, gamma, rho, theta, vega  # noqa: E402

TODAY = ql.Date(3, 11, 2026)
DAY_COUNT = ql.Actual365Fixed()


def quantlib_option(S, K, days, r, sigma, q, option):
    ql.Settings.instance().evaluationDate = TODAY

    def flat_curve(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DAY_COUNT, ql.Continuous))

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        flat_curve(q),
        flat_curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, DAY_COUNT)),
    )
    kind = ql.Option.Call if option == "call" else ql.Option.Put
    instrument = ql.VanillaOption(ql.PlainVanillaPayoff(kind, K), ql.EuropeanExercise(TODAY + days))
    instrument.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return instrument


# Spots from deep OTM to deep ITM (relative to K=100), q > 0 included.
GRID = [
    (S, 100.0, days, r, sigma, q)
    for S in (40.0, 80.0, 100.0, 125.0, 250.0)
    for days in (30, 365, 1095)
    for r in (0.0, 0.05)
    for sigma in (0.1, 0.4)
    for q in (0.0, 0.03)
]


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S, K, days, r, sigma, q", GRID)
def test_matches_quantlib(S, K, days, r, sigma, q, option):
    T = days / 365
    ref = quantlib_option(S, K, days, r, sigma, q, option)
    ours = {
        "price": bs_price(S, K, T, r, sigma, q, option),
        "delta": delta(S, K, T, r, sigma, q, option),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, option),
        "rho": rho(S, K, T, r, sigma, q, option),
    }
    theirs = {
        "price": ref.NPV(),
        "delta": ref.delta(),
        "gamma": ref.gamma(),
        "vega": ref.vega(),
        "theta": ref.theta(),
        "rho": ref.rho(),
    }
    for name in ours:
        # rel=1e-8 is the pass criterion; abs covers values that are ~0 (deep OTM).
        assert ours[name] == pytest.approx(theirs[name], rel=1e-8, abs=1e-10), name


# The same comparison on random inputs over the FULL range, including very short,
# very low-vol options where finite differences can't check the Greeks.
@settings()  # profile from tests/conftest.py
@given(
    S=st.floats(1.0, 1000.0),
    K=st.floats(1.0, 1000.0),
    days=st.integers(1, 3650),
    r=st.floats(-0.02, 0.15),
    sigma=st.floats(0.01, 2.0),
    q=st.floats(0.0, 0.10),
    option=st.sampled_from(["call", "put"]),
)
def test_property_matches_quantlib(S, K, days, r, sigma, q, option):
    test_matches_quantlib(S, K, days, r, sigma, q, option)
