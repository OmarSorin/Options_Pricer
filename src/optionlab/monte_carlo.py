"""Monte Carlo pricing for European options, with a standard error.

Idea: under the risk-neutral measure the option price is the discounted EXPECTED payoff.
Monte Carlo estimates that expectation by simulating many possible stock prices at expiry,
computing the payoff for each, and averaging:

    price ≈ exp(-rT) * (1/n) * sum of payoff(S_T_i)

For a European option only the price AT EXPIRY matters, and under Black-Scholes it can be
sampled exactly in one step (no time-stepping, so no discretisation error):

    S_T = S * exp((r - q - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)

The only error left is statistical. Its size is measured by the standard error,

    SE = (sample standard deviation of the discounted payoffs) / sqrt(n)

so the true price lies within price ± 1.96 * SE about 95% of the time. Halving the error
costs four times as many paths.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from optionlab.black_scholes import _validate, _validate_option


class MCResult(NamedTuple):
    """A Monte Carlo estimate: the price, its standard error, and the number of paths used."""

    price: float
    std_error: float
    paths: int


def _validate_mc_args(paths: int, *inputs: ArrayLike) -> None:
    """Checks specific to Monte Carlo, on top of the usual Black-Scholes input rules."""
    if any(np.ndim(x) != 0 for x in inputs):
        raise ValueError("mc_price accepts scalar inputs only; use a loop to price many options.")
    # bool is a subclass of int in Python, so exclude it explicitly.
    # At least 2 paths, because the sample standard deviation needs two values.
    if isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths < 2:
        raise ValueError(f"paths must be an integer >= 2, got {paths!r}.")


def mc_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option: str = "call",
    paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Price a European call or put by Monte Carlo simulation.

    Units and conventions match `bs_price`: T in years; r, sigma, q annualised decimals.

    Args:
        option: "call" or "put".
        paths: number of simulated stock prices at expiry (an integer >= 2). The standard
            error shrinks like 1 / sqrt(paths).
        seed: seed for numpy's random generator. The same seed gives exactly the same result;
            None gives a different result every call.

    Returns:
        MCResult(price, std_error, paths). The true price is within price ± 1.96 * std_error
        about 95% of the time.

    Edge cases: at T == 0 or sigma == 0 every simulated price is the same number, so the
    result is the exact intrinsic or deterministic value and std_error is 0, up to
    floating-point rounding (about 1e-16 of the price). No special case is needed in the code.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), array inputs, unknown option type,
            or paths not an integer >= 2.
    """
    # 1. Validate: Monte Carlo rules first, then the same rules as bs_price.
    _validate_option(option)
    _validate_mc_args(paths, S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = (float(x) for x in _validate(S, K, T, r, sigma, q))

    # 2. Simulate the stock price at expiry, exactly, from standard normal draws Z.
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(paths)
    stock_at_expiry = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # 3. Discounted payoffs, then their average (the price) and its standard error.
    payoff = np.maximum(stock_at_expiry - K, 0.0) if option == "call" else np.maximum(K - stock_at_expiry, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(paths)  # ddof=1: the sample (unbiased) variance

    return MCResult(float(price), float(std_error), int(paths))
