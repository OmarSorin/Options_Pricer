"""Black-Scholes Greeks: analytic formulas and numerical (bump-and-reprice) versions.

The Greeks measure how the option price P changes when one input moves while
everything else stays fixed. Conventions used in this module:

    delta = dP/dS            per 1 unit of S
    gamma = d²P/dS²          per unit of S, squared
    vega  = dP/dsigma        per 1.00 change in sigma (NOT per 1%; divide by 100 for that)
    theta = -dP/dT           per YEAR of calendar time passing (divide by 365 for per day);
                             usually negative for a long option, because it loses time value
    rho   = dP/dr            per 1.00 change in r (divide by 100 for per 1%)

Inputs follow the same units, vectorization and ValueError rules as `bs_price`.
In addition, every Greek raises ValueError when T == 0 or sigma == 0 (for any element
of an array input), because the derivatives are undefined or infinite there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from optionlab.black_scholes import (
    FloatOrArray,
    _as_output,
    _d1_d2_arrays,
    _validate,
    _validate_option,
    bs_price,
)

Arrays = tuple[NDArray[np.float64], ...]


def _greeks_inputs(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> Arrays:
    """Validate inputs, reject T == 0 or sigma == 0, and return (S, K, T, r, sigma, q, d1, d2)."""
    S_, K_, T_, r_, sigma_, q_ = _validate(S, K, T, r, sigma, q)
    d1, d2, degenerate = _d1_d2_arrays(S_, K_, T_, r_, sigma_, q_)
    if np.any(degenerate):
        raise ValueError(
            "Greeks undefined when T == 0 or sigma == 0 (the option has no time value "
            "left, so the derivatives are zero or infinite)."
        )
    return S_, K_, T_, r_, sigma_, q_, d1, d2


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
) -> FloatOrArray:
    """Delta: how much the option price changes per 1 unit move in the spot, dP/dS.

    In plain English: delta is the number of shares you would hold to hedge one
    option. A call's delta is between 0 and exp(-qT); a put's is between -exp(-qT) and 0.

        call delta =  exp(-qT) * N(d1)
        put  delta = -exp(-qT) * N(-d1)

    Units: price units per 1 unit of S (dimensionless for same-currency S and P).

    Assumptions: same as `bs_price` (European exercise, constant sigma and r, continuous q).

    Raises:
        ValueError: on invalid inputs (as `bs_price`), unknown option type,
            or T == 0 / sigma == 0 anywhere in the inputs.
    """
    _validate_option(option)
    S_, K_, T_, r_, sigma_, q_, d1, d2 = _greeks_inputs(S, K, T, r, sigma, q)
    dividend_discount = np.exp(-q_ * T_)
    if option == "call":
        result = dividend_discount * norm.cdf(d1)
    else:
        result = -dividend_discount * norm.cdf(-d1)
    return _as_output(result)


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> FloatOrArray:
    """Gamma: how fast delta changes as the spot moves, d²P/dS².

    In plain English: gamma measures the curvature of the price in S. It is
    largest for at-the-money options close to expiry, and it is always positive
    for a long option.

        gamma = exp(-qT) * n(d1) / (S * sigma * sqrt(T))      (n = standard normal pdf)

    Gamma is identical for calls and puts. By put-call parity they differ only by
    S*exp(-qT) - K*exp(-rT), which is linear in S, so its second derivative is 0.
    That is why this function has no `option` argument.

    Units: per unit of S, squared.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), or T == 0 / sigma == 0.
    """
    S_, K_, T_, r_, sigma_, q_, d1, d2 = _greeks_inputs(S, K, T, r, sigma, q)
    result = np.exp(-q_ * T_) * norm.pdf(d1) / (S_ * sigma_ * np.sqrt(T_))
    return _as_output(result)


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> FloatOrArray:
    """Vega: how much the option price changes per 1.00 change in volatility, dP/dsigma.

    Scaling: per 1.00 change in sigma (sigma going from 0.20 to 1.20), NOT per 1%.
    For the market-quoted "per vol point" figure, divide by 100.

        vega = S * exp(-qT) * n(d1) * sqrt(T)

    In plain English: more volatility means a wider range of outcomes, and an
    option benefits from the upside while its downside is capped, so vega is
    always positive. It is identical for calls and puts (parity again: the
    difference S*exp(-qT) - K*exp(-rT) does not depend on sigma).

    Units: price units per 1.00 of sigma.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), or T == 0 / sigma == 0.
    """
    S_, K_, T_, r_, sigma_, q_, d1, d2 = _greeks_inputs(S, K, T, r, sigma, q)
    result = S_ * np.exp(-q_ * T_) * norm.pdf(d1) * np.sqrt(T_)
    return _as_output(result)


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
) -> FloatOrArray:
    """Theta: how much the option price changes as time passes, -dP/dT, per year.

    Sign convention: theta = -dP/dT, where T is time to expiry. As a day passes, T
    gets SMALLER, so the minus sign makes theta the change in value from time
    passing. It is usually negative for a long option (time decay). Divide by 365
    for a per-calendar-day figure.

        common   = -S * exp(-qT) * n(d1) * sigma / (2 * sqrt(T))
        call     = common - r*K*exp(-rT)*N(d2)  + q*S*exp(-qT)*N(d1)
        put      = common + r*K*exp(-rT)*N(-d2) - q*S*exp(-qT)*N(-d1)

    The first term is the loss of time value from volatility. The other two
    come from interest on the strike and dividends on the stock. Because of them,
    a deep in-the-money European put (or a call with a high q) can have
    POSITIVE theta.

    Units: price units per year.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), unknown option type,
            or T == 0 / sigma == 0 anywhere in the inputs.
    """
    _validate_option(option)
    S_, K_, T_, r_, sigma_, q_, d1, d2 = _greeks_inputs(S, K, T, r, sigma, q)
    spot_pv = S_ * np.exp(-q_ * T_)
    strike_pv = K_ * np.exp(-r_ * T_)
    common = -spot_pv * norm.pdf(d1) * sigma_ / (2.0 * np.sqrt(T_))
    if option == "call":
        result = common - r_ * strike_pv * norm.cdf(d2) + q_ * spot_pv * norm.cdf(d1)
    else:
        result = common + r_ * strike_pv * norm.cdf(-d2) - q_ * spot_pv * norm.cdf(-d1)
    return _as_output(result)


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
) -> FloatOrArray:
    """Rho: how much the option price changes per 1.00 change in the interest rate, dP/dr.

    Scaling: per 1.00 change in r (r going from 0.05 to 1.05). Divide by 100
    for the per-1% figure.

        call rho =  K * T * exp(-rT) * N(d2)
        put  rho = -K * T * exp(-rT) * N(-d2)

    In plain English: a higher rate makes the strike you pay later cheaper in
    today's money, which helps a call (positive rho) and hurts a put (negative rho).

    Units: price units per 1.00 of r.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), unknown option type,
            or T == 0 / sigma == 0 anywhere in the inputs.
    """
    _validate_option(option)
    S_, K_, T_, r_, sigma_, q_, d1, d2 = _greeks_inputs(S, K, T, r, sigma, q)
    strike_pv_times_T = K_ * T_ * np.exp(-r_ * T_)
    if option == "call":
        result = strike_pv_times_T * norm.cdf(d2)
    else:
        result = -strike_pv_times_T * norm.cdf(-d2)
    return _as_output(result)


def numerical_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
) -> dict[str, FloatOrArray]:
    """Estimate all five Greeks by bumping one input and repricing with `bs_price`.

    This is the method used for models with no closed-form Greeks (trees, Monte
    Carlo). Here it serves as an independent check of the analytic formulas.

    Method: central differences, f'(x) ≈ [f(x+h) - f(x-h)] / (2h), whose error
    shrinks like h². Gamma uses [f(x+h) - 2f(x) + f(x-h)] / h².

    Bump sizes and the trade-off: a smaller h reduces the truncation error
    (about h²) but magnifies floating-point rounding error (about eps*P/h, or
    eps*P/h² for gamma, where P is the option price).

    Rule: bump each input by a small fraction of its NATURAL SCALE, the change
    that moves d1 by about 1. A fixed bump fails when the price distribution is
    narrow (short expiry, low vol): the price then curves sharply, and the same
    h covers a much larger part of the curve. Property-based testing found exactly
    those failures for the fixed sizes used before.
        S:      h = 1e-3 * S * sigma * sqrt(T)   (S * sigma * sqrt(T) ~ width of the
                                                  price distribution at expiry)
        r:      h = 1e-3 * sigma / sqrt(T)       (d1 moves by sqrt(T)/sigma per unit of r)
        sigma:  h = 1e-4 * sigma                 (relative)
        T:      h = min(1e-5, T/2)               (so T - h stays positive)

    Accuracy limit: when sigma * sqrt(T) < ~0.05, the bump becomes small enough
    that rounding error dominates gamma in double precision. No bump size fixes
    that, so treat numerical gamma there as approximate. The analytic Greeks are
    exact at any sigma * sqrt(T) > 0.

    Returns:
        A dict with keys "delta", "gamma", "vega", "theta", "rho", using the
        same units and sign conventions as the analytic functions.

    Raises:
        ValueError: same rules as the analytic Greeks.
    """
    _validate_option(option)
    S_, K_, T_, r_, sigma_, q_, _, _ = _greeks_inputs(S, K, T, r, sigma, q)

    def price(S=S_, T=T_, r=r_, sigma=sigma_) -> NDArray[np.float64]:
        return np.asarray(bs_price(S, K_, T, r, sigma, q_, option))

    h_S = 1e-3 * S_ * sigma_ * np.sqrt(T_)
    h_sigma = 1e-4 * sigma_
    h_r = 1e-3 * sigma_ / np.sqrt(T_)
    h_T = np.minimum(1e-5, T_ / 2)

    p_mid = price()
    p_up, p_down = price(S=S_ + h_S), price(S=S_ - h_S)

    return {
        "delta": _as_output((p_up - p_down) / (2 * h_S)),
        "gamma": _as_output((p_up - 2 * p_mid + p_down) / h_S**2),
        "vega": _as_output((price(sigma=sigma_ + h_sigma) - price(sigma=sigma_ - h_sigma)) / (2 * h_sigma)),
        # theta = -dP/dT, hence the reversed order (down minus up).
        "theta": _as_output((price(T=T_ - h_T) - price(T=T_ + h_T)) / (2 * h_T)),
        "rho": _as_output((price(r=r_ + h_r) - price(r=r_ - h_r)) / (2 * h_r)),
    }
