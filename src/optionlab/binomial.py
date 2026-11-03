"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

The tree splits the time to expiry T into `steps` equal intervals of length dt = T / steps.
In each interval the stock moves either up by a factor u or down by a factor d:

    u = exp(sigma * sqrt(dt)),   d = 1 / u

and the risk-neutral probability of an up move is chosen so that the stock's expected
growth over dt is exactly exp((r - q) * dt):

    p = (exp((r - q) * dt) - d) / (u - d)

This is the exact p from Cox, Ross & Rubinstein (1979), also used in Hull. (QuantLib's "crr"
engine uses a linearized approximation of p instead, so its prices differ from ours by
about 1/steps; both converge to Black-Scholes.)

The option is valued by backward induction: start from the payoffs at expiry, then step
back one interval at a time, discounting the probability-weighted average of the two
values that follow each node.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from optionlab.black_scholes import _validate, _validate_option

EXERCISE_TYPES = ("european", "american")


def _validate_tree_args(steps: int, exercise: str, *inputs: ArrayLike) -> None:
    """Checks specific to the tree, on top of the usual Black-Scholes input rules."""
    if any(np.ndim(x) != 0 for x in inputs):
        raise ValueError(
            "crr_price accepts scalar inputs only; use np.vectorize(crr_price) to price many options."
        )
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"steps must be an integer >= 1, got {steps!r}.")
    if exercise not in EXERCISE_TYPES:
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}.")


def crr_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
    steps: int = 500,
    exercise: str = "european",
) -> float:
    """Price a call or put on a Cox-Ross-Rubinstein binomial tree.

    Units and conventions match `bs_price`: T in years; r, sigma, q annualised decimals.

    Args:
        option: "call" or "put".
        steps: number of time steps in the tree (an integer >= 1). More steps means
            a more accurate price and a slower calculation; the error shrinks
            roughly like 1/steps.
        exercise: "european" (exercise only at expiry) or "american" (exercise at any
            node, including today). At every node an American option is worth the
            larger of exercising now and the discounted value of waiting.

    Returns:
        The option price as a float.

    Assumptions: same as Black-Scholes (constant sigma, r, q), with time divided into
    `steps` discrete intervals.

    Limitations: scalar inputs only (use np.vectorize to price many options);
    sigma must be > 0, because with sigma == 0 the up and down moves coincide.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), array inputs, sigma == 0,
            steps not an integer >= 1, unknown option or exercise type, or when the
            risk-neutral probability p falls outside (0, 1) (increase `steps`).
    """
    # 1. Validate: tree-specific rules first, then the same rules as bs_price.
    _validate_option(option)
    _validate_tree_args(steps, exercise, S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = (float(x) for x in _validate(S, K, T, r, sigma, q))
    if sigma == 0:
        raise ValueError("sigma must be > 0 for a binomial tree (with sigma == 0, u == d).")

    def payoff(stock: np.ndarray) -> np.ndarray:
        return np.maximum(stock - K, 0.0) if option == "call" else np.maximum(K - stock, 0.0)

    # 2. No time left: the option is worth its intrinsic value.
    if T == 0:
        return float(payoff(np.array(S)))

    # 3. Tree parameters.
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)  # expected growth of the stock over one step
    p = (growth - d) / (u - d)  # risk-neutral probability of an up move
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"Risk-neutral probability p = {p:.4f} is outside (0, 1): one step is too long "
            f"for these r, q and sigma. Increase steps."
        )
    discount = np.exp(-r * dt)

    # 4. Stock prices at expiry: node j has had j up moves and (steps - j) down moves.
    j = np.arange(steps + 1)
    stock_at_expiry = S * u**j * d ** (steps - j)
    values = payoff(stock_at_expiry)

    # 5. Backward induction, from step (steps - 1) down to step 0 (today).
    #    Each node is worth the discounted probability-weighted average of its two
    #    children: values[1:] are the "up" children and values[:-1] the "down" ones,
    #    so the array gets one element shorter every step, ending with a single value.
    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])  # value of waiting
        if exercise == "american":
            # The holder may also exercise now: the node is worth whichever is larger.
            j = np.arange(i + 1)
            stock_now = S * u**j * d ** (i - j)  # stock prices at the i + 1 nodes of step i
            values = np.maximum(values, payoff(stock_now))

    return float(values[0])
