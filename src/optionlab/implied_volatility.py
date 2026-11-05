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
- Newton jumps along the tangent. Near the answer it is very fast (the number of correct
  digits roughly doubles each step).
- We apply it to the LOGARITHM of the price, ln(bs_price(sigma)) = ln(price), whose slope is
  vega / bs_price:
      sigma_new = sigma - (ln bs_price(sigma) - ln price) * bs_price(sigma) / vega(sigma)
  Far out of the money the price is roughly exp(-c / sigma^2): it falls by dozens of orders
  of magnitude as sigma drops, so the price curve is extremely bent and plain Newton creeps
  along it (101 steps for a price of 1.6e-57). Its logarithm is nearly straight, so the
  tangent lands close to the answer (5 steps). Near the money both versions take 4-8.
- Far from the answer vega can be tiny, or even 0 in floating point, and a Newton step can
  shoot off. So we keep a bracket [lo, hi] that is known to contain the answer, shrink it
  after every step, and whenever Newton would leave the bracket we take its midpoint
  instead (bisection). Bisection is slow, but it can't fail.
- We stop when a step changes sigma by less than tol. A tolerance on the price would be
  wrong for cheap options: any sigma prices a 1e-20 option to within 1e-10.

Limit: deep in the money the price is almost all intrinsic value. Rounding the price by one
part in 1e16 then moves sigma by about 1e-16 * price / vega, which can be 0.01 or more, so
the price does not determine sigma to many digits. The answer still reproduces the price.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from optionlab.black_scholes import _validate, _validate_option, bs_price
from optionlab.greeks import vega

# A price that needs a volatility above this is, in practice, at its upper bound.
MAX_SIGMA = 1e6
# The starting guess is never below this (0.1%): see step 3 in implied_vol.
MIN_START_SIGMA = 1e-3


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
        tol: stop when a step changes sigma by less than tol (sigma is then known to
            about tol). A tolerance on sigma, not on the price, so it works the same for
            a $500 option and a $1e-20 one.
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
    #    For a very cheap option that formula gives an absurdly small sigma (1e-310 was seen),
    #    where d1 overflows, so start no lower than MIN_START_SIGMA. The search can still go
    #    below it if the answer really is lower, because lo stays at 0.
    sigma = max(np.sqrt(2 * np.pi / T) * price / S, MIN_START_SIGMA)
    if not lo < sigma < hi:
        sigma = 0.5 * (lo + hi)

    # 4. Newton steps inside the bracket, bisection whenever Newton would leave it.
    for _ in range(max_iter):
        model = bs_price(S, K, T, r, sigma, q, option)
        diff = model - price
        if diff == 0:
            return float(sigma)  # exact match
        # The price rises with sigma: too expensive means the answer is below sigma.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        # Newton on ln(price), not on the price: f(sigma) = ln(model) - ln(price), whose
        # slope is vega / model. Skipped when vega or the model price underflows to 0.
        slope = vega(S, K, T, r, sigma, q)  # sigma > 0 here, so vega is defined
        if slope > 0 and model > 0:
            newton = sigma - (np.log(model) - np.log(price)) * model / slope
        else:
            newton = None
        if newton is not None and lo < newton < hi:
            next_sigma = newton
        else:
            next_sigma = 0.5 * (lo + hi)  # bisection: halve the bracket

        # Stop on SIGMA, not on the price. "Price within 1e-10" is met by almost any sigma
        # when the option itself costs less than 1e-10, which gave answers off by 100%.
        if abs(next_sigma - sigma) < tol:
            return float(next_sigma)
        sigma = next_sigma

    raise ValueError(f"implied_vol did not converge in {max_iter} steps (tol = {tol}).")
