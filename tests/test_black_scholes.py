import numpy as np
import pytest

from optionlab import bs_price, d1_d2

# Fail any test that triggers a numpy/scipy RuntimeWarning (NaN, divide by zero, ...).
pytestmark = pytest.mark.filterwarnings("error")

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

# A grid of (S, K, T, r, sigma, q) used by several tests.
GRID = [
    (S, K, T, r, sigma, q)
    for S in (50.0, 100.0, 150.0)
    for K in (80.0, 100.0, 120.0)
    for T in (0.25, 1.0, 3.0)
    for r in (0.0, 0.05)
    for sigma in (0.1, 0.4)
    for q in (0.0, 0.03)
]


def test_reference_values():
    assert bs_price(**BASE, option="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(**BASE, option="put") == pytest.approx(5.5735, abs=1e-4)


def test_reference_values_with_dividend_yield():
    params = dict(**BASE, q=0.02)
    assert bs_price(**params, option="call") == pytest.approx(9.2270, abs=1e-4)
    assert bs_price(**params, option="put") == pytest.approx(6.3301, abs=1e-4)


def test_scalar_input_returns_float():
    assert isinstance(bs_price(**BASE), float)


@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9)


@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_no_arbitrage_bounds(S, K, T, r, sigma, q):
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert call >= max(spot_pv - strike_pv, 0.0) - 1e-12
    assert put >= max(strike_pv - spot_pv, 0.0) - 1e-12
    assert call <= spot_pv + 1e-12
    assert put <= strike_pv + 1e-12


def test_monotonicity_in_spot():
    spots = np.linspace(50, 150, 41)
    calls = bs_price(spots, 100, 1, 0.05, 0.2, option="call")
    puts = bs_price(spots, 100, 1, 0.05, 0.2, option="put")
    assert np.all(np.diff(calls) > 0)
    assert np.all(np.diff(puts) < 0)


@pytest.mark.parametrize("option", ["call", "put"])
def test_price_increases_with_sigma(option):
    sigmas = np.linspace(0.05, 0.8, 30)
    prices = bs_price(100, 100, 1, 0.05, sigmas, option=option)
    assert np.all(np.diff(prices) > 0)


@pytest.mark.parametrize(
    "S, K, option, expected",
    [(110, 100, "call", 10.0), (90, 100, "call", 0.0), (90, 100, "put", 10.0), (110, 100, "put", 0.0), (100, 100, "call", 0.0)],
)
def test_zero_time_is_intrinsic_value(S, K, option, expected):
    assert bs_price(S, K, 0.0, 0.05, 0.2, option=option) == pytest.approx(expected)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_zero_vol_is_discounted_deterministic_payoff(S, option):
    K, T, r, q = 100.0, 1.0, 0.05, 0.02
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    expected = max(spot_pv - strike_pv, 0.0) if option == "call" else max(strike_pv - spot_pv, 0.0)
    assert bs_price(S, K, T, r, 0.0, q, option) == pytest.approx(expected)


def test_deep_in_and_out_of_the_money():
    # Deep OTM call is worth ~0; deep ITM call is ~ S - K*exp(-rT).
    assert bs_price(10, 100, 1, 0.05, 0.2, option="call") == pytest.approx(0.0, abs=1e-8)
    assert bs_price(500, 100, 1, 0.05, 0.2, option="call") == pytest.approx(500 - 100 * np.exp(-0.05), abs=1e-6)
    assert bs_price(500, 100, 1, 0.05, 0.2, option="put") == pytest.approx(0.0, abs=1e-8)


def test_vectorized_matches_scalar_calls():
    spots = np.array([[80.0, 100.0], [120.0, 140.0]])
    result = bs_price(spots, 100, 1, 0.05, 0.2, 0.01, "call")
    assert result.shape == spots.shape
    for idx, S in np.ndenumerate(spots):
        assert result[idx] == pytest.approx(bs_price(S, 100, 1, 0.05, 0.2, 0.01, "call"))


@pytest.mark.parametrize("option", ["call", "put"])
def test_array_mixing_normal_and_edge_cases(option):
    # Entries: normal, T=0, sigma=0, at-the-money with T=0.
    T = np.array([1.0, 0.0, 1.0, 0.0])
    sigma = np.array([0.2, 0.2, 0.0, 0.2])
    S = np.array([100.0, 110.0, 105.0, 100.0])
    result = bs_price(S, 100, T, 0.05, sigma, 0.0, option)
    assert np.all(np.isfinite(result))
    for i in range(len(S)):
        assert result[i] == pytest.approx(bs_price(S[i], 100, T[i], 0.05, sigma[i], 0.0, option))


def test_d1_d2_values_and_degenerate_limits():
    d1, d2 = d1_d2(**BASE)
    assert d1 == pytest.approx(0.35)
    assert d2 == pytest.approx(0.15)
    assert d1_d2(110, 100, 0.0, 0.05, 0.2) == (np.inf, np.inf)
    assert d1_d2(90, 100, 0.0, 0.05, 0.2) == (-np.inf, -np.inf)
    assert d1_d2(100, 100, 0.0, 0.05, 0.2) == (0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(S=-1),
        dict(S=0),
        dict(K=-100),
        dict(K=0),
        dict(T=-0.1),
        dict(sigma=-0.2),
        dict(S=np.nan),
        dict(option="straddle"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs):
    params = {**BASE, "option": "call", **kwargs}
    with pytest.raises(ValueError):
        bs_price(**params)


def test_invalid_input_inside_array_raises():
    with pytest.raises(ValueError):
        bs_price(np.array([100.0, -5.0]), 100, 1, 0.05, 0.2)
