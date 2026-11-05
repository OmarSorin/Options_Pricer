"""Implied volatility: the sigma that makes the Black-Scholes price match a given price.

Black-Scholes turns a volatility into a price. Markets quote the price, so traders run the
formula backwards: which sigma would make bs_price equal what the option actually costs?
That sigma is the volatility the market is "implying". There is no formula for it, so we
solve

    bs_price(sigma) - price = 0

numerically. Two facts make this safe:

1. The price always RISES with sigma (vega > 0), so there is at most one answer.
2. As sigma goes from 0 to infinity, the price runs over its whole no-arbitrage range:
   call from max(S*exp(-qT) - K*exp(-rT), 0) up to S*exp(-qT),
   put  from max(K*exp(-rT) - S*exp(-qT), 0) up to K*exp(-rT).
   So any price strictly inside that range has exactly one implied volatility.

Method: Newton-Raphson, kept inside a bracket, with bisection as the fallback.
- Newton jumps along the tangent: sigma_new = sigma - (bs_price(sigma) - price) / vega(sigma).
  Near the answer it is very fast (the number of correct digits roughly doubles each step).
- But far from the answer vega can be tiny, or even 0 in floating point (deep out of the
  money, or sigma far too small), and the step shoots off or divides by zero.
- So we keep a bracket [lo, hi] that is known to contain the answer, shrink it after every
  step, and whenever Newton would leave the bracket we take its midpoint instead
  (bisection). Bisection is slow, but it can't fail.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from optionlab.black_scholes import _validate, _validate_option, bs_price
from optionlab.greeks import vega

# A price that needs a volatility above this is, in practice, at its upper bound.
MAX_SIGMA = 1e6


def _price_bounds(S: float, K: float, T: float, r: float, q: float, option: str) -> tuple[float, float]:
    """The no-arbitrage range of a European option's price: (value at sigma = 0, limit as sigma -> inf)."""
    spot_pv = S * np.exp(-q * T)  # dividend-adjusted spot
    strike_pv = K * np.exp(-r * T)  # discounted strike
    if option == "call":
        return max(spot_pv - strike_pv, 0.0), spot_pv
    return max(strike_pv - spot_pv, 0.0), strike_pv


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find the volatility sigma at which `bs_price` equals `price`.

    Units and conventions match `bs_price`: T in years; r, q annualised decimals.
    The result is an annualised decimal (0.2 = 20%).

    Args:
        price: the option's price, in the same currency unit as S and K.
        option: "call" or "put".
        tol: stop when the repriced option is within tol of `price`, or when sigma is
            pinned down to within tol (the bracket is narrower than tol).
        max_iter: the maximum number of Newton or bisection steps.

    Returns:
        sigma as a float.

    Limitations: scalar inputs only (use a loop for many options). European exercise,
    like `bs_price`. Deep in or out of the money the price barely moves with sigma, so
    a tiny change in the price moves the implied volatility a lot.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), array inputs, unknown option type,
            a non-finite price, T == 0, a price not strictly inside the no-arbitrage
            bounds (including exactly on the lower bound, where sigma would be 0 and the
            price says nothing useful about volatility), or no convergence in max_iter steps.
    """
    # 1. Validate. sigma is what we are solving for, so pass a placeholder 0.0 for it.
    _validate_option(option)
    if any(np.ndim(x) != 0 for x in (price, S, K, T, r, q)):
        raise ValueError("implied_vol accepts scalar inputs only; use a loop to invert many prices.")
    if not np.isfinite(price):
        raise ValueError("price must be a finite number (no NaN or infinity).")
    S, K, T, r, _, q = (float(x) for x in _validate(S, K, T, r, 0.0, q))
    price = float(price)
    if T == 0:
        raise ValueError("Implied volatility is undefined at T == 0: the option is worth its intrinsic value whatever sigma is.")

    lower, upper = _price_bounds(S, K, T, r, q, option)
    if not lower < price < upper:
        raise ValueError(
            f"price {price:.6g} is not strictly inside the no-arbitrage bounds ({lower:.6g}, {upper:.6g}) "
            f"for this {option}, so no volatility fits it."
        )

    # 2. Bracket the answer: the price at sigma = 0 is `lower` (< price), so lo = 0 works.
    #    Double hi until the price there is above the target.
    lo, hi = 0.0, 1.0
    while bs_price(S, K, T, r, hi, q, option) < price:
        hi *= 2
        if hi > MAX_SIGMA:
            raise ValueError(f"price {price:.6g} is too close to its upper bound {upper:.6g} to invert.")

    # 3. Starting guess (Brenner-Subrahmanyam), good for at-the-money options:
    #    there price ≈ S * sigma * sqrt(T / (2*pi)), so sigma ≈ sqrt(2*pi / T) * price / S.
    sigma = np.sqrt(2 * np.pi / T) * price / S
    if not lo < sigma < hi:
        sigma = 0.5 * (lo + hi)

    # 4. Newton steps inside the bracket, bisection whenever Newton would leave it.
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, option) - price
        if abs(diff) < tol:
            return float(sigma)
        # The price rises with sigma: too expensive means the answer is below sigma.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < tol:
            return float(sigma)

        slope = vega(S, K, T, r, sigma, q)  # sigma > 0 here, so vega is defined
        newton = sigma - diff / slope if slope > 0 else None  # vega can underflow to 0
        if newton is not None and lo < newton < hi:
            sigma = newton
        else:
            sigma = 0.5 * (lo + hi)  # bisection: halve the bracket

    raise ValueError(f"implied_vol did not converge in {max_iter} steps (tol = {tol}).")
