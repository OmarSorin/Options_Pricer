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
def quantlib_crr(S, K, days, r, sigma, q, option, steps, exercise="european"):
    """QuantLib price on its CRR tree; steps=None uses its finite-difference engine instead."""
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
    if exercise == "european":
        exercise_rule = ql.EuropeanExercise(today + days)
    else:
        exercise_rule = ql.AmericanExercise(today, today + days)
    instrument = ql.VanillaOption(ql.PlainVanillaPayoff(kind, K), exercise_rule)
    if steps is None:
        engine = ql.FdBlackScholesVanillaEngine(process, 2000, 2000)  # 2000 time x 2000 space points
    else:
        engine = ql.BinomialVanillaEngine(process, "crr", steps)
    instrument.setPricingEngine(engine)
    return instrument.NPV()


@pytest.mark.parametrize("exercise", ["european", "american"])
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_gap_to_quantlib_shrinks_like_one_over_n(S, q, option, exercise):
    gaps = {}
    for steps in (100, 1000):
        ours = crr_price(S, 100, 1, 0.05, 0.2, q, option, steps=steps, exercise=exercise)
        theirs = quantlib_crr(S, 100, 365, 0.05, 0.2, q, option, steps, exercise)
        gaps[steps] = abs(ours - theirs)
        assert gaps[steps] <= 1.0 / steps
    if max(gaps.values()) < 1e-12:
        # Both trees agree exactly: an American option so deep in the money that
        # immediate exercise is optimal, so both are worth exactly the intrinsic value.
        return
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


# ===========================================================================
# Phase 3, Step 3: parity, no-arbitrage bounds and convergence.
# ===========================================================================
from hypothesis import assume, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def tree_or_reject(*args, **kwargs):
    """Price on the tree; skip hypothesis examples where p falls outside (0, 1)."""
    try:
        return crr_price(*args, **kwargs)
    except ValueError as error:
        assume("probability" not in str(error))
        raise


# --- Put-call parity holds EXACTLY on the tree, at any number of steps. ---
# Backward induction is linear, so the tree values (call - put), whose payoff is
# S_T - K, as  exp(-rT) * E[S_T - K].  The exact CRR p makes the tree's expected
# stock growth per step exactly exp((r - q) dt), so E[S_T] = S exp((r - q) T)
# with no approximation. Parity therefore tests p, not convergence.
@given(
    S=st.floats(1.0, 1000.0),
    K=st.floats(1.0, 1000.0),
    T=st.floats(0.01, 10.0),
    r=st.floats(-0.02, 0.15),
    sigma=st.floats(0.05, 1.0),
    q=st.floats(0.0, 0.10),
    steps=st.integers(1, 300),
)
def test_property_tree_put_call_parity_is_exact(S, K, T, r, sigma, q, steps):
    call = tree_or_reject(S, K, T, r, sigma, q, "call", steps=steps)
    put = tree_or_reject(S, K, T, r, sigma, q, "put", steps=steps)
    forward_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call - put == pytest.approx(forward_value, abs=1e-12 * (S + K) * steps)


# --- No-arbitrage bounds also hold exactly on the tree (Jensen's inequality). ---
@given(
    S=st.floats(1.0, 1000.0),
    K=st.floats(1.0, 1000.0),
    T=st.floats(0.01, 10.0),
    r=st.floats(-0.02, 0.15),
    sigma=st.floats(0.05, 1.0),
    q=st.floats(0.0, 0.10),
    steps=st.integers(1, 300),
    option=st.sampled_from(["call", "put"]),
)
def test_property_tree_no_arbitrage_bounds(S, K, T, r, sigma, q, steps, option):
    price = tree_or_reject(S, K, T, r, sigma, q, option, steps=steps)
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    tol = 1e-12 * (S + K) * steps
    if option == "call":
        assert max(spot_pv - strike_pv, 0.0) - tol <= price <= spot_pv + tol
    else:
        assert max(strike_pv - spot_pv, 0.0) - tol <= price <= strike_pv + tol


# --- Convergence to Black-Scholes. ---
# CRR does NOT converge smoothly: the error zig-zags between odd and even N and
# jumps with where the strike sits between nodes, sometimes passing through ~0.
# So we never test "the error shrinks at every N". Instead:
#   1. a bound   |error| <= S * sigma * sqrt(T) / N   at EVERY N from 50 to 1000
#      (S * sigma * sqrt(T) is the option's natural price scale; measured worst
#      cases sit 4-10x inside this bound), and
#   2. the convergence RATE: fit log(error) against log(N) using the largest error
#      in each window of 50 consecutive N (the "envelope"), which ignores the
#      near-zero crossings. CRR theory says the slope is -1.
CONVERGENCE_CASES = {
    "ATM call": (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call"),
    "OTM call, q>0": (100.0, 120.0, 1.0, 0.05, 0.2, 0.03, "call"),
    "ITM put, long": (100.0, 120.0, 2.0, 0.05, 0.3, 0.02, "put"),
    "OTM put, short": (100.0, 90.0, 0.25, 0.03, 0.25, 0.0, "put"),
}
STEP_RANGE = np.arange(50, 1001)


def tree_errors(S, K, T, r, sigma, q, option):
    exact = bs_price(S, K, T, r, sigma, q, option)
    return np.array([crr_price(S, K, T, r, sigma, q, option, steps=int(n)) - exact for n in STEP_RANGE])


@pytest.fixture(scope="module")
def convergence_errors():
    return {name: tree_errors(*args) for name, args in CONVERGENCE_CASES.items()}


@pytest.mark.parametrize("name", list(CONVERGENCE_CASES))
def test_error_is_bounded_by_c_over_n(name, convergence_errors):
    S, K, T, r, sigma, q, option = CONVERGENCE_CASES[name]
    bound = S * sigma * np.sqrt(T) / STEP_RANGE
    assert np.all(np.abs(convergence_errors[name]) <= bound)


@pytest.mark.parametrize("name", list(CONVERGENCE_CASES))
def test_convergence_rate_is_one_over_n(name, convergence_errors):
    abs_err = np.abs(convergence_errors[name])
    window = 50
    centers = [STEP_RANGE[i : i + window].mean() for i in range(0, len(STEP_RANGE) - window + 1, window)]
    envelope = [abs_err[i : i + window].max() for i in range(0, len(STEP_RANGE) - window + 1, window)]
    slope = np.polyfit(np.log(centers), np.log(envelope), 1)[0]
    assert -1.2 <= slope <= -0.8, f"fitted slope {slope:.3f}"


# ===========================================================================
# Phase 3, Step 4: American exercise.
#
# There is no closed-form price for an American put, so bs_price can't be the
# reference. Instead we use: Hull's published values, QuantLib (its CRR tree and,
# as a completely different method, its finite-difference PDE solver), and exact
# identities that must hold whatever the price is.
# ===========================================================================


# Hull, "Options, Futures, and Other Derivatives", binomial-trees-in-practice
# example: American put, S0 = K = 50, r = 10%, sigma = 40%, T = 5 months.
# Hull uses the same exact CRR tree, and quotes 4.49 for 5 steps and DerivaGem's
# 4.263, 4.272, 4.278 for 30, 50, 100 steps. Tolerance = half the last printed digit.
@pytest.mark.parametrize("steps, book_value, half_digit", [(5, 4.49, 0.005), (30, 4.263, 5e-4), (50, 4.272, 5e-4), (100, 4.278, 5e-4)])
def test_hull_american_put(steps, book_value, half_digit):
    price = crr_price(50, 50, 5 / 12, 0.10, 0.40, option="put", steps=steps, exercise="american")
    assert price == pytest.approx(book_value, abs=half_digit)


@pytest.mark.parametrize(
    "S, q, option",
    [(100.0, 0.0, "put"), (90.0, 0.0, "put"), (110.0, 0.0, "put"), (100.0, 0.03, "call"), (110.0, 0.05, "call")],
)
def test_american_matches_quantlib_finite_difference(S, q, option):
    # A finite-difference PDE solver is an independent method with its own small
    # error, so we allow the same O(1/N) bound as the tree's Black-Scholes test.
    steps = 2000
    ours = crr_price(S, 100, 1, 0.05, 0.2, q, option, steps=steps, exercise="american")
    reference = quantlib_crr(S, 100, 365, 0.05, 0.2, q, option, None, "american")
    assert abs(ours - reference) <= S * 0.2 * 1.0 / steps


def test_deep_in_the_money_american_put_is_exercised_immediately():
    # Waiting can only lose the interest on K, so the put is worth exactly K - S.
    assert crr_price(80, 100, 1, 0.05, 0.2, option="put", exercise="american") == pytest.approx(20.0, abs=1e-12)


def test_early_exercise_premium_is_positive_for_a_put():
    american = crr_price(100, 100, 1, 0.05, 0.2, option="put", exercise="american")
    european = crr_price(100, 100, 1, 0.05, 0.2, option="put", exercise="european")
    assert american > european + 0.1  # about 0.52 here


TREE_INPUTS = dict(
    S=st.floats(1.0, 1000.0),
    K=st.floats(1.0, 1000.0),
    T=st.floats(0.01, 10.0),
    r=st.floats(0.0, 0.15),
    sigma=st.floats(0.05, 1.0),
    q=st.floats(0.0, 0.10),
    steps=st.integers(1, 300),
)


@given(**TREE_INPUTS)
def test_property_american_call_without_dividends_equals_european(S, K, T, r, sigma, q, steps):
    # With q = 0 (and r >= 0) early exercise of a call is never optimal: exercising
    # gives S - K now, holding is worth at least S - K*exp(-rT) >= S - K.
    american = tree_or_reject(S, K, T, r, sigma, 0.0, "call", steps=steps, exercise="american")
    european = tree_or_reject(S, K, T, r, sigma, 0.0, "call", steps=steps, exercise="european")
    assert american == pytest.approx(european, rel=1e-12, abs=1e-12 * (S + K))


@given(**TREE_INPUTS, option=st.sampled_from(["call", "put"]))
def test_property_american_is_worth_at_least_european_and_intrinsic(S, K, T, r, sigma, q, steps, option):
    american = tree_or_reject(S, K, T, r, sigma, q, option, steps=steps, exercise="american")
    european = tree_or_reject(S, K, T, r, sigma, q, option, steps=steps, exercise="european")
    intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    tol = 1e-12 * (S + K) * steps
    assert american >= european - tol
    assert american >= intrinsic - tol


@given(**TREE_INPUTS)
def test_property_american_put_call_bounds(S, K, T, r, sigma, q, steps):
    # Exact parity fails once early exercise is allowed; it becomes two bounds:
    #     S*exp(-qT) - K  <=  C - P  <=  S - K*exp(-rT)
    call = tree_or_reject(S, K, T, r, sigma, q, "call", steps=steps, exercise="american")
    put = tree_or_reject(S, K, T, r, sigma, q, "put", steps=steps, exercise="american")
    tol = 1e-12 * (S + K) * steps
    assert S * np.exp(-q * T) - K - tol <= call - put <= S - K * np.exp(-r * T) + tol
