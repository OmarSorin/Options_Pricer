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

Antithetic variates (antithetic=True) reduce the error for free: every draw Z is used twice,
as Z and as -Z. When Z gives a high stock price, -Z gives a low one, so the two payoffs tend
to move in opposite directions and their average is less noisy than two independent payoffs.
For calls and puts this always helps, because the payoff only goes one way as Z grows
(it is monotone in Z), which makes the covariance of f(Z) and f(-Z) negative.

The two payoffs of a pair are NOT independent, so the standard error must be computed from
the pair averages (n/2 independent numbers), never from all n payoffs as if they were
independent. The latter is a classic bug: it gets the price right and the error wrong.
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


def _validate_mc_args(paths: int, antithetic: bool, *inputs: ArrayLike) -> None:
    """Checks specific to Monte Carlo, on top of the usual Black-Scholes input rules."""
    if any(np.ndim(x) != 0 for x in inputs):
        raise ValueError("mc_price accepts scalar inputs only; use a loop to price many options.")
    # bool is a subclass of int in Python, so exclude it explicitly.
    # At least 2 paths, because the sample standard deviation needs two values.
    if isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths < 2:
        raise ValueError(f"paths must be an integer >= 2, got {paths!r}.")
    # Antithetic: each draw is used twice, and the SE needs at least two pairs.
    if antithetic and (paths % 2 != 0 or paths < 4):
        raise ValueError(f"antithetic=True needs an even number of paths >= 4, got {paths}.")


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
    antithetic: bool = False,
) -> MCResult:
    """Price a European call or put by Monte Carlo simulation.

    Units and conventions match `bs_price`: T in years; r, sigma, q annualised decimals.

    Args:
        option: "call" or "put".
        paths: number of simulated stock prices at expiry, i.e. payoff evaluations (an
            integer >= 2). The standard error shrinks like 1 / sqrt(paths). With antithetic
            variates it counts both halves of each pair, so plain and antithetic runs with
            the same `paths` cost the same.
        seed: seed for numpy's random generator. The same seed gives exactly the same result;
            None gives a different result every call.
        antithetic: if True, use paths / 2 draws Z, each as Z and -Z (see the module notes).
            Needs an even number of paths >= 4.

    Returns:
        MCResult(price, std_error, paths). The true price is within price ± 1.96 * std_error
        about 95% of the time.

    Edge cases: at T == 0 or sigma == 0 every simulated price is the same number, so the
    result is the exact intrinsic or deterministic value and std_error is 0, up to
    floating-point rounding (about 1e-16 of the price). No special case is needed in the code.

    Raises:
        ValueError: on invalid inputs (as `bs_price`), array inputs, unknown option type,
            paths not an integer >= 2, or (with antithetic=True) paths odd or < 4.
    """
    # 1. Validate: Monte Carlo rules first, then the same rules as bs_price.
    _validate_option(option)
    _validate_mc_args(paths, antithetic, S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = (float(x) for x in _validate(S, K, T, r, sigma, q))

    def discounted_payoff(Z: np.ndarray) -> np.ndarray:
        """Simulate the stock price at expiry exactly from draws Z, and discount its payoff."""
        stock_at_expiry = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoff = np.maximum(stock_at_expiry - K, 0.0) if option == "call" else np.maximum(K - stock_at_expiry, 0.0)
        return np.exp(-r * T) * payoff

    # 2. Draw standard normals and turn them into independent samples of the discounted payoff.
    rng = np.random.default_rng(seed)
    if antithetic:
        Z = rng.standard_normal(paths // 2)
        # One sample per PAIR: the average of the payoffs at Z and at -Z.
        samples = 0.5 * (discounted_payoff(Z) + discounted_payoff(-Z))
    else:
        samples = discounted_payoff(rng.standard_normal(paths))

    # 3. Price = average of the samples; SE = their spread / sqrt(number of samples).
    #    For antithetic runs that number is paths / 2 (the pairs), not paths.
    price = samples.mean()
    std_error = samples.std(ddof=1) / np.sqrt(len(samples))  # ddof=1: the sample (unbiased) variance

    return MCResult(float(price), float(std_error), int(paths))
