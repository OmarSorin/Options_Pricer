# optionlab

A Python library for pricing European and American options with several models
and computing the Greeks. Built as a learning project in derivatives pricing.

## Roadmap

- [x] Phase 1: Black-Scholes closed-form pricing (European call/put, dividend yield `q`)
- [x] Phase 2: Greeks, analytic and numerical (bumping)
- [x] Phase 3: Binomial tree (Cox-Ross-Rubinstein), including American options
- [ ] Phase 4: Monte Carlo pricing with standard error and antithetic variates
- [ ] Phase 5: Implied volatility solver (Newton-Raphson with bisection fallback)
- [ ] Phase 6: Streamlit demo page with plots

## Quick example

```python
from optionlab import bs_price, crr_price, delta, vega, theta

bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call")  # 10.4506...
delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call")     # 0.6368...
vega(S=100, K=100, T=1, r=0.05, sigma=0.2)                     # 37.52 per 1.00 of sigma
theta(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call")     # -6.414 per year

# Binomial tree (Cox-Ross-Rubinstein), European or American exercise
crr_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option="put", steps=500, exercise="american")  # 6.09
```

Conventions: `T` in years; `r`, `sigma`, `q` annualised decimals (0.05 = 5%);
`option` is `"call"` or `"put"`. Functions accept scalars and numpy arrays.
Greeks: vega and rho are per **1.00** change (divide by 100 for per 1%);
theta is `-dP/dT` **per year** (divide by 365 for per day).

## Install and test

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # add ",validation" for the QuantLib cross-check
pytest
```

The convergence plot is regenerated with `pip install -e ".[plots]"` and
`python scripts/plot_convergence.py`.

## Binomial tree convergence

The CRR tree converges to the Black-Scholes price, zig-zagging between odd and even
step counts, with an error that falls like 1/N:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/convergence-dark.png">
  <img alt="Left: CRR tree price for 10 to 200 steps, zig-zagging around the Black-Scholes price 10.4506 with shrinking amplitude. Right: log-log plot of the absolute error for 50 to 1000 steps; the envelope of the error follows a 1/N reference line with fitted slope -1.11." src="docs/convergence.png">
</picture>

## Docs

- [Black-Scholes notes](docs/black_scholes_notes.md): intuition behind d1, d2 and risk-neutral pricing
- [Greeks notes](docs/greeks_notes.md): intuition, scaling conventions, and how the formulas are verified
- [Binomial tree notes](docs/binomial_notes.md): u, d, p, backward induction, American exercise and convergence

## Known limitations

- Black-Scholes assumes constant volatility and constant interest rates (no smile or term structure).
- Black-Scholes prices European exercise only; use `crr_price(..., exercise="american")` for American options.
- Dividends are a continuous yield, not discrete cash dividends.
- Prices are lognormal: no jumps, no transaction costs, no borrowing constraints.
- `S` and `K` must be strictly positive.
- Greeks are undefined at `T == 0` or `sigma == 0` and raise `ValueError` (for an array input,
  if any element is degenerate). Prices still work there.
- `crr_price` accepts scalar inputs only (wrap it in `np.vectorize` to price many options),
  needs `sigma > 0`, and raises `ValueError` if a step is too long for the risk-neutral
  probability to lie in (0, 1); use more steps.
- The tree's error shrinks like 1/N but zig-zags between odd and even N: at the default
  N = 500 the at-the-money call above is off by about 0.004 (2 correct decimals).
  10x more steps gives about 10x less error, at 10x the cost.
