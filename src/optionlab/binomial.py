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

from numpy.typing import ArrayLike


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
            node). American exercise is added in Phase 3, Step 4.

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
    raise NotImplementedError("crr_price is implemented in Phase 3, Step 2.")
