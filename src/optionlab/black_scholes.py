"""Black-Scholes closed-form pricing for European options.

Conventions used everywhere in optionlab:
    S      spot price of the underlying (same currency unit as K)
    K      strike price
    T      time to expiry in YEARS (0.5 = six months)
    r      continuously compounded risk-free rate, annualised decimal (0.05 = 5%)
    sigma  annualised volatility, decimal (0.2 = 20%)
    q      continuous dividend yield, annualised decimal (default 0.0)
    option "call" or "put"

All functions accept Python scalars or numpy arrays (they broadcast against
each other) and return a float for scalar input or an ndarray otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatOrArray = float | NDArray[np.float64]


def _validate(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    """Convert inputs to float arrays and raise ValueError if any are invalid."""
    arrays = tuple(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    S_, K_, T_, r_, sigma_, q_ = arrays

    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("All inputs must be finite numbers (no NaN or infinity).")
    if np.any(S_ <= 0):
        raise ValueError("Spot price S must be strictly positive.")
    if np.any(K_ <= 0):
        raise ValueError("Strike price K must be strictly positive.")
    if np.any(T_ < 0):
        raise ValueError("Time to expiry T must not be negative.")
    if np.any(sigma_ < 0):
        raise ValueError("Volatility sigma must not be negative.")
    return arrays


def _as_output(x: NDArray[np.float64]) -> FloatOrArray:
    """Return a plain float for 0-dimensional results, otherwise the array."""
    return float(x) if x.ndim == 0 else x


def _d1_d2_arrays(
    S: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
    q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Core d1/d2 maths on already-validated arrays. Also returns the degenerate mask."""
    # Degenerate: no randomness left (T == 0 or sigma == 0), so sigma*sqrt(T) == 0.
    vol_sqrt_T = sigma * np.sqrt(T)
    degenerate = vol_sqrt_T == 0

    # Swap the zero denominator for 1 BEFORE dividing, so we never divide by zero.
    safe_denominator = np.where(degenerate, 1.0, vol_sqrt_T)
    numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T

    d1 = numerator / safe_denominator
    d2 = d1 - vol_sqrt_T

    # In the degenerate case the true limit of d1 and d2 is +inf or -inf
    # depending on whether the forward price is above or below the strike.
    log_forward_moneyness = np.log(S / K) + (r - q) * T
    limit = np.where(log_forward_moneyness > 0, np.inf, np.where(log_forward_moneyness < 0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)
    return d1, d2, degenerate


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[FloatOrArray, FloatOrArray]:
    """Compute the Black-Scholes quantities d1 and d2.

    In plain English: d2 measures how many standard deviations the expected
    log-price at expiry sits above the strike (under the risk-neutral measure),
    and d1 is d2 shifted up by one total-volatility unit (sigma * sqrt(T)).
    N(d2) is then the risk-neutral probability of finishing in the money.

        d1 = [ ln(S/K) + (r - q + sigma^2 / 2) * T ] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Units: S and K in the same currency; T in years; r, sigma, q annualised decimals.

    Assumptions: constant volatility and rates, lognormal prices, continuous dividend yield.

    Edge cases: when T == 0 or sigma == 0 the formula would divide by zero. There
    d1 and d2 are returned as +inf (forward above strike), -inf (forward below
    strike) or 0 (forward exactly at the strike), which are the true limits.

    Raises:
        ValueError: if S or K <= 0, T < 0, sigma < 0, or any input is not finite.
    """
    S_, K_, T_, r_, sigma_, q_ = _validate(S, K, T, r, sigma, q)
    d1, d2, _ = _d1_d2_arrays(S_, K_, T_, r_, sigma_, q_)
    return _as_output(d1), _as_output(d2)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
) -> FloatOrArray:
    """Price a European call or put with the Black-Scholes formula.

    In plain English: the price is the discounted expected payoff under
    risk-neutral probabilities. With F-style terms S*exp(-qT) (the dividend-adjusted
    spot) and K*exp(-rT) (the discounted strike):

        call = S*exp(-qT) * N(d1) - K*exp(-rT) * N(d2)
        put  = K*exp(-rT) * N(-d2) - S*exp(-qT) * N(-d1)

    where N is the standard normal CDF and d1, d2 come from `d1_d2`.

    Units: the price is in the same currency unit as S and K, for one unit of
    the underlying. T is in years; r, sigma, q are annualised decimals.

    Assumptions: European exercise only, constant volatility and interest rate,
    continuous dividend yield, frictionless markets, lognormal prices.

    Edge cases (no NaN or warnings, also inside arrays):
        T == 0:     intrinsic value, max(S - K, 0) for a call, max(K - S, 0) for a put.
        sigma == 0: discounted deterministic payoff,
                    max(S*exp(-qT) - K*exp(-rT), 0) for a call, put analogue.

    Args:
        option: "call" or "put".

    Returns:
        A float for scalar inputs, otherwise an ndarray (inputs are broadcast together).

    Raises:
        ValueError: if S or K <= 0, T < 0, sigma < 0, any input is not finite,
            or option is not "call" or "put".
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}.")

    S_, K_, T_, r_, sigma_, q_ = _validate(S, K, T, r, sigma, q)
    d1, d2, degenerate = _d1_d2_arrays(S_, K_, T_, r_, sigma_, q_)

    spot_pv = S_ * np.exp(-q_ * T_)  # dividend-adjusted spot
    strike_pv = K_ * np.exp(-r_ * T_)  # discounted strike

    if option == "call":
        regular = spot_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2)
        deterministic = np.maximum(spot_pv - strike_pv, 0.0)
    else:
        regular = strike_pv * norm.cdf(-d2) - spot_pv * norm.cdf(-d1)
        deterministic = np.maximum(strike_pv - spot_pv, 0.0)

    price = np.where(degenerate, deterministic, regular)
    return _as_output(price)
