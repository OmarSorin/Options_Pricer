import numpy as np
import pytest

from optionlab import bs_price, delta, gamma, numerical_greeks, rho, theta, vega

pytestmark = pytest.mark.filterwarnings("error")

ANALYTIC = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
NEEDS_OPTION = {"delta", "theta", "rho"}  # gamma and vega are the same for calls and puts


def analytic(name, S, K, T, r, sigma, q, option):
    f = ANALYTIC[name]
    return f(S, K, T, r, sigma, q, option) if name in NEEDS_OPTION else f(S, K, T, r, sigma, q)


GRID = [
    (S, K, T, r, sigma, q)
    for S in (50.0, 100.0, 150.0)
    for K in (80.0, 120.0)
    for T in (0.25, 2.0)
    for r in (0.0, 0.05)
    for sigma in (0.1, 0.4)
    for q in (0.0, 0.03)
]


# 1. External reference: Hull, "Options, Futures, and Other Derivatives",
#    worked example S=49, K=50, r=5%, sigma=20%, T=20 weeks, call, no dividends.
#    Hull quotes vega per 1.00 of sigma and theta per year, matching our conventions.
@pytest.mark.parametrize(
    "name, expected, tol",
    [("delta", 0.522, 5e-4), ("gamma", 0.066, 5e-4), ("vega", 12.1, 0.05), ("theta", -4.31, 5e-3), ("rho", 8.91, 5e-3)],
)
def test_hull_textbook_example(name, expected, tol):
    value = analytic(name, 49, 50, 20 / 52, 0.05, 0.2, 0.0, "call")
    assert value == pytest.approx(expected, abs=tol)


# Widely published values for S=K=100, T=1, r=5%, sigma=20%, q=0.
@pytest.mark.parametrize(
    "name, option, expected",
    [
        ("delta", "call", 0.636831),
        ("delta", "put", -0.363169),
        ("gamma", "call", 0.018762),
        ("vega", "call", 37.524035),
        ("theta", "call", -6.414028),
        ("theta", "put", -1.657880),
        ("rho", "call", 53.232482),
        ("rho", "put", -41.890461),
    ],
)
def test_atm_reference_values(name, option, expected):
    assert analytic(name, 100, 100, 1, 0.05, 0.2, 0.0, option) == pytest.approx(expected, abs=1e-5)


# 2. Analytic vs bump-and-reprice: the main independent check.
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_analytic_matches_numerical(S, K, T, r, sigma, q, option):
    numeric = numerical_greeks(S, K, T, r, sigma, q, option)
    for name in ANALYTIC:
        assert analytic(name, S, K, T, r, sigma, q, option) == pytest.approx(
            numeric[name], rel=1e-5, abs=1e-6
        ), name


# 3. The Black-Scholes PDE ties theta, delta and gamma together.
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_black_scholes_pde_holds(S, K, T, r, sigma, q, option):
    price = bs_price(S, K, T, r, sigma, q, option)
    residual = (
        theta(S, K, T, r, sigma, q, option)
        + (r - q) * S * delta(S, K, T, r, sigma, q, option)
        + 0.5 * sigma**2 * S**2 * gamma(S, K, T, r, sigma, q)
        - r * price
    )
    assert residual == pytest.approx(0.0, abs=1e-9)


# 4. Call/put relations from parity. NOTE: like parity itself, these cannot
#    catch a bug that affects calls and puts equally (e.g. inside d1).
@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_call_put_relations(S, K, T, r, sigma, q):
    spot_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    args = (S, K, T, r, sigma, q)
    assert delta(*args, "call") - delta(*args, "put") == pytest.approx(np.exp(-q * T))
    assert rho(*args, "call") - rho(*args, "put") == pytest.approx(K * T * np.exp(-r * T))
    assert theta(*args, "call") - theta(*args, "put") == pytest.approx(q * spot_pv - r * strike_pv)


# 5. Signs and ranges.
@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_signs_and_ranges(S, K, T, r, sigma, q):
    args = (S, K, T, r, sigma, q)
    cap = np.exp(-q * T)
    assert gamma(*args) >= 0
    assert vega(*args) >= 0
    assert 0 <= delta(*args, "call") <= cap
    assert -cap <= delta(*args, "put") <= 0
    assert rho(*args, "call") >= 0
    assert rho(*args, "put") <= 0


def test_greeks_at_the_money_are_strictly_positive_where_expected():
    assert gamma(100, 100, 1, 0.05, 0.2) > 0
    assert vega(100, 100, 1, 0.05, 0.2) > 0
    assert theta(100, 100, 1, 0.05, 0.2, option="call") < 0


def test_deep_itm_european_put_can_have_positive_theta():
    # Deep ITM put ~ K*exp(-rT) - S, which GROWS as expiry approaches (interest on K).
    assert theta(20, 100, 1, 0.05, 0.2, option="put") > 0


# 6. Vectorization.
@pytest.mark.parametrize("name", list(ANALYTIC))
def test_vectorized_matches_scalar(name):
    spots = np.array([[70.0, 90.0], [110.0, 130.0]])
    result = analytic(name, spots, 100, 1, 0.05, 0.2, 0.01, "put")
    assert result.shape == spots.shape
    for idx, S in np.ndenumerate(spots):
        assert result[idx] == pytest.approx(analytic(name, S, 100, 1, 0.05, 0.2, 0.01, "put"))


def test_numerical_greeks_vectorized():
    spots = np.array([80.0, 100.0, 120.0])
    result = numerical_greeks(spots, 100, 1, 0.05, 0.2)
    assert all(v.shape == spots.shape for v in result.values())


def test_scalar_input_returns_float():
    assert isinstance(delta(100, 100, 1, 0.05, 0.2), float)
    assert all(isinstance(v, float) for v in numerical_greeks(100, 100, 1, 0.05, 0.2).values())


# 7. Errors.
ALL_FUNCS = [*ANALYTIC.values(), numerical_greeks]


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("T, sigma", [(0.0, 0.2), (1.0, 0.0), (np.array([1.0, 0.0]), 0.2), (1.0, np.array([0.2, 0.0]))])
def test_degenerate_inputs_raise(func, T, sigma):
    with pytest.raises(ValueError, match="Greeks undefined"):
        func(100, 100, T, 0.05, sigma)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("kwargs", [dict(S=-1), dict(S=0), dict(K=0), dict(T=-1), dict(sigma=-0.2), dict(r=np.nan)])
def test_invalid_inputs_raise(func, kwargs):
    params = {**dict(S=100, K=100, T=1, r=0.05, sigma=0.2), **kwargs}
    with pytest.raises(ValueError):
        func(**params)


@pytest.mark.parametrize("func", [delta, theta, rho, numerical_greeks])
def test_unknown_option_type_raises(func):
    with pytest.raises(ValueError, match="option"):
        func(100, 100, 1, 0.05, 0.2, option="straddle")
