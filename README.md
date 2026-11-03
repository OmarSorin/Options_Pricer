# optionlab

A Python library for pricing European and American options with several models
and computing the Greeks. Built as a learning project in derivatives pricing.

## Roadmap

- [x] Phase 1: Black-Scholes closed-form pricing (European call/put, dividend yield `q`)
- [ ] Phase 2: Greeks, analytic and numerical (bumping)
- [ ] Phase 3: Binomial tree (Cox-Ross-Rubinstein), including American options
- [ ] Phase 4: Monte Carlo pricing with standard error and antithetic variates
- [ ] Phase 5: Implied volatility solver (Newton-Raphson with bisection fallback)
- [ ] Phase 6: Streamlit demo page with plots

## Quick example

```python
from optionlab import bs_price

bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call")  # 10.4506...
```

Conventions: `T` in years; `r`, `sigma`, `q` annualised decimals (0.05 = 5%);
`option` is `"call"` or `"put"`. Functions accept scalars and numpy arrays.

## Install and test

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Docs

- [Black-Scholes notes](docs/black_scholes_notes.md): intuition behind d1, d2 and risk-neutral pricing

## Known limitations

- Black-Scholes assumes constant volatility and constant interest rates (no smile or term structure).
- Black-Scholes prices European exercise only; American options need the tree (Phase 3).
- Dividends are a continuous yield, not discrete cash dividends.
- Prices are lognormal: no jumps, no transaction costs, no borrowing constraints.
- `S` and `K` must be strictly positive.
