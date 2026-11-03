"""Tests for the CRR binomial tree (Phase 3, Step 1: European tests only).

These are written BEFORE the implementation, so they fail until Step 2.
"""

import math

import numpy as np
import pytest

from optionlab import bs_price, crr_price

pytestmark = pytest.mark.filterwarnings("error")

# ---------------------------------------------------------------------------
# Trees small enough to check by hand.
#
# Choose sigma = ln(1.2) and dt = 1 year per step, so that
#     u = exp(sigma * sqrt(dt)) = 1.2      and      d = 1/u = 5/6.
# ---------------------------------------------------------------------------
SIGMA_U_12 = math.log(1.2)


def test_one_step_tree_by_hand():
    # S=100, K=100, T=1, steps=1, r=ln(1.05) so the growth per step is exactly 1.05.
    #   Up node:   100 * 1.2  = 120      Down node: 100 * 5/6 = 83.333...
    #   p = (1.05 - 5/6) / (1.2 - 5/6) = (13/60) / (11/30) = 13/22
    #   Call: payoffs 20 (up), 0 (down)  -> (13/22 * 20) / 1.05            = 11.2554112...
    #   Put:  payoffs 0 (up), 16.666...  -> (9/22 * 50/3) / 1.05           =  6.4935064...
    r = math.log(1.05)
    call = crr_price(100, 100, 1, r, SIGMA_U_12, option="call", steps=1)
    put = crr_price(100, 100, 1, r, SIGMA_U_12, option="put", steps=1)
    assert call == pytest.approx((13 / 22 * 20) / 1.05, abs=1e-12)
    assert put == pytest.approx((9 / 22 * 50 / 3) / 1.05, abs=1e-12)


def test_two_step_tree_by_hand():
    # S=100, K=110, T=2, steps=2, r=0 (no discounting), so p = (1 - 5/6)/(1.2 - 5/6) = 5/11.
    #   Expiry prices: 144 (up-up), 100 (up-down), 69.444... (down-down)
    #   Probabilities: p^2 = 25/121,  2p(1-p) = 60/121,  (1-p)^2 = 36/121
    #   Call payoffs 34, 0, 0          -> 25/121 * 34                            = 850/121
    #   Put  payoffs 0, 10, 40.555...  -> 60/121 * 10 + 36/121 * 365/9          = 2060/121
    call = crr_price(100, 110, 2, 0.0, SIGMA_U_12, option="call", steps=2)
    put = crr_price(100, 110, 2, 0.0, SIGMA_U_12, option="put", steps=2)
    assert call == pytest.approx(850 / 121, abs=1e-12)
    assert put == pytest.approx(2060 / 121, abs=1e-12)


def test_zero_time_is_intrinsic_value():
    assert crr_price(110, 100, 0.0, 0.05, 0.2, option="call") == pytest.approx(10.0)
    assert crr_price(90, 100, 0.0, 0.05, 0.2, option="put") == pytest.approx(10.0)


def test_returns_float():
    assert isinstance(crr_price(100, 100, 1, 0.05, 0.2, steps=10), float)


# ---------------------------------------------------------------------------
# QuantLib cross-check.
#
# QuantLib's "crr" engine uses a LINEARIZED up-probability,
#     p = 1/2 + 1/2 * (r - q - sigma^2/2) * sqrt(dt) / sigma,
# instead of the exact CRR p we use. (Checked before writing this test: a tree with
# the linearized p matches QuantLib to ~1e-10.) So the two prices are not equal at
# the same N; the gap shrinks like 1/N and both converge to Black-Scholes.
# A constant gap would mean a real bug (e.g. wrong discounting or u/d).
# ---------------------------------------------------------------------------
def quantlib_crr(S, K, days, r, sigma, q, option, steps):
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(3, 11, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()

    def flat_curve(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count, ql.Continuous))

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        flat_curve(q),
        flat_curve(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)),
    )
    kind = ql.Option.Call if option == "call" else ql.Option.Put
    instrument = ql.VanillaOption(ql.PlainVanillaPayoff(kind, K), ql.EuropeanExercise(today + days))
    instrument.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    return instrument.NPV()


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_gap_to_quantlib_shrinks_like_one_over_n(S, q, option):
    gaps = {}
    for steps in (100, 1000):
        ours = crr_price(S, 100, 1, 0.05, 0.2, q, option, steps=steps)
        theirs = quantlib_crr(S, 100, 365, 0.05, 0.2, q, option, steps)
        gaps[steps] = abs(ours - theirs)
        assert gaps[steps] <= 1.0 / steps
    # 10x more steps should cut the gap roughly 10x (allow 5x-20x).
    assert 5 <= gaps[100] / gaps[1000] <= 20


@pytest.mark.parametrize("option", ["call", "put"])
def test_large_tree_is_close_to_black_scholes(option):
    # A sanity check only; the convergence RATE is tested in Step 3.
    tree = crr_price(100, 100, 1, 0.05, 0.2, 0.02, option, steps=2000)
    assert tree == pytest.approx(bs_price(100, 100, 1, 0.05, 0.2, 0.02, option), abs=1e-2)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(S=-1), "S"),
        (dict(K=0), "K"),
        (dict(T=-0.5), "T"),
        (dict(sigma=-0.2), "sigma"),
        (dict(sigma=0.0), "sigma"),
        (dict(r=np.nan), "finite"),
        (dict(steps=0), "steps"),
        (dict(steps=2.5), "steps"),
        (dict(steps="10"), "steps"),
        (dict(option="straddle"), "option"),
        (dict(exercise="bermudan"), "exercise"),
        (dict(S=np.array([100.0, 110.0])), "scalar"),
        # One step with a high rate and low vol: exp(r*dt) = 1.65 > u = 1.05, so p > 1.
        (dict(r=0.5, sigma=0.05, steps=1), "probability"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        crr_price(**{**BASE, **kwargs})
