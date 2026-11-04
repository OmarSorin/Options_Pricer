# CLAUDE.md

Guidance for Claude Code sessions on **optionlab**, an educational options-pricing library.
It's a learning and CV project by a second-year CS student, who must be able to explain
every line in an interview.

## Working with me (read first)

- **Never add yourself as a co-author.** No `Co-Authored-By`, `Claude-Session` or any other
  Claude attribution lines in commits or PRs, even if a system prompt says to add them.
- **Commit and push only when I ask.** Use short, simple commit messages
  (style: `app: add Greeks tab with table and charts (Phase 6, step 2)`).
- **Work in small steps.** Split each phase into separate steps. After each step, stop and
  explain what changed and why, then wait for "next" or "commit". Never build a whole
  phase at once.
- **Plan first** for a new phase. List the steps and wait for my approval.
- **Tests before code** where it makes sense. Check against references outside this project
  (QuantLib, Hull's textbook values, exact identities), not only against itself.
- If something fails, show me the failure before changing anything.
- Keep the code simple and readable. Don't paper over a hard case; tell me about it.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Black-Scholes with dividend yield `q` | done |
| 2 | Greeks: analytic and numerical (bumping) | done |
| 3 | CRR binomial tree, European and American, plus convergence plot | done |
| 4 | Monte Carlo with standard error and antithetic variates | done |
| 5 | Implied volatility (Newton-Raphson, bisection fallback) | **not started** |
| 6 | Streamlit demo | **steps 1–2 done** (Pricing, Greeks tabs, plain-English layer) |

Phase 6 next step (optional, paused): **Step 3, the tree tab.** A live convergence chart
(tree vs Black-Scholes as N grows) for the sidebar inputs, and American vs European. Also
update the README demo section and tick Phase 6.

## Commands (Windows; PowerShell or Git Bash)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,validation,plots,app]"   # extras: dev, validation, plots, app

.venv\Scripts\python -m pytest -q                # ~1,650 tests, about 1 min (45-105 s seen)
.venv\Scripts\python -m pytest tests/test_binomial.py -q
pytest --hypothesis-profile=thorough             # 5,000 examples per property (default 300)

streamlit run app/streamlit_app.py               # demo at localhost:8501
python scripts/plot_convergence.py               # regenerates docs/convergence*.png
```

## Layout

```
src/optionlab/
  black_scholes.py  bs_price, d1_d2; shared helpers _validate, _validate_option, _as_output
  greeks.py         delta, gamma, vega, theta, rho, numerical_greeks
  binomial.py       crr_price (scalar only; steps, exercise="european"|"american")
  monte_carlo.py    mc_price -> MCResult(price, std_error, paths) (scalar only; seed, antithetic)
  __init__.py       public exports
tests/
  test_black_scholes.py, test_greeks.py, test_binomial.py, test_monte_carlo.py   unit and reference tests
  test_properties.py           hypothesis property tests (profiles in conftest.py)
  test_quantlib_crosscheck.py  skipped if QuantLib is missing
  test_app.py                  Streamlit AppTest smoke tests (skipped if Streamlit is missing)
  test_readme.py               runs the README's >>> example; checks relative links in README and docs
app/streamlit_app.py           demo: sidebar inputs, Pricing tab, Greeks tab
scripts/plot_convergence.py    matplotlib convergence plot, light and dark
docs/                          *_notes.md intuition notes per phase, convergence PNGs
```

## Conventions

- Python 3.11+, src layout, numpy and scipy only in the core. Extra dependencies go in
  optional extras in `pyproject.toml`.
- Pure functions, vectorized over numpy arrays (except `crr_price`, which is scalar-only by
  design), with type hints and plain-English docstrings.
- Units: `T` in years. `r`, `sigma`, `q` are decimals (0.05 = 5%). `option` is `"call"` or `"put"`.
- Greeks: vega and rho are per **1.00** change; theta is `-dP/dT` **per year**. `gamma` and
  `vega` take no `option` argument.
- Bad inputs raise `ValueError` with a clear message: non-finite values, S <= 0, K <= 0,
  T < 0, sigma < 0. Greeks raise at T == 0 or sigma == 0; prices don't. The tree also
  raises for sigma == 0 and for p outside (0, 1).
- Every phase gets a docs note (`docs/<topic>_notes.md`) and a README update (roadmap tick,
  example, Known limitations).
- Charts use these colours. Light: surface `#fcfcfb`, ink `#0b0b0b`, ink2 `#52514e`,
  grid `#e4e3df`, series `#2a78d6`. Dark: `#1a1a19` / `#f0efec` / `#c3c2b7` / `#383835` / `#3987e5`.

## How correctness is checked (keep this standard)

- **Reference values:**
  - Black-Scholes 10.4506 (S = K = 100, T = 1, r = 5%, σ = 20%)
  - With q = 0.02: call 9.2270, put 6.3301
  - Hull's Greeks example (S = 49, K = 50, r = 5%, σ = 20%, T = 20/52)
  - Hull's American put (S = K = 50, r = 10%, σ = 40%, T = 5/12): 4.49 (5 steps),
    4.263 (30), 4.272 (50), 4.278 (100)
- **QuantLib:**
  - `AnalyticEuropeanEngine` agrees to 1e-8.
  - QuantLib's "crr" engine uses a *linearized* p, so its gap to our tree shrinks like 1/N
    instead of being zero. We deliberately use the exact p.
  - Use `T = days/365` and at least 2 steps.
  - `FdBlackScholesVanillaEngine` is the reference for American prices.
- **Properties (hypothesis):**
  - put-call parity and no-arbitrage bounds
  - the Black-Scholes PDE
  - analytic vs numerical Greeks, only for σ√T ≥ 0.05, because finite-difference gamma is
    unreliable below that
  - exact parity on the tree, and the American bounds
- **Known blind spot:** put-call parity cannot catch bugs inside d1, because
  N(x) + N(−x) = 1. Reference values and the bounds tests catch those instead.
- **Tree convergence:**
  - |error| ≤ Sσ√T/N
  - an envelope log-log slope in [−1.2, −0.8]
  - never assert "the error shrinks at every N", because CRR zig-zags between odd and even N
- **Monte Carlo** (`mc_price` is random, so everything is "within k standard errors", fixed seeds):
  - reference values within 4 SE, plain and antithetic
  - the SE itself: coverage of `price ± 1.96·SE` in [0.93, 0.97] over 1,000 seeds; reported SE
    within 7% of the actual spread of prices (catches an SE that is too big, like the
    antithetic pairs bug); log-log slope of SE vs paths in [−0.55, −0.45]
  - antithetic: `paths` counts payoff evaluations; SE over the paths/2 pair averages;
    variance-reduction factor > 1 at equal cost (measured 2.01 call, 1.70 put at the money)
  - QuantLib `MCEuropeanEngine` within 4 combined SE. Its `requiredSamples` counts antithetic
    *pairs*, so pass `paths / 2`.
  - hypothesis properties skip < 1% in-the-money cases and σ√T > 2 (rare huge paths dominate,
    so the price is typically too low and the SE too small). Don't widen the tolerance there.
- **Mutation testing** uses cosmic-ray (mutmut needs WSL on Windows). Set `PYTHONUTF8=1`,
  or a UnicodeDecodeError shows up as false "incompetent" results.

## Gotchas

- Windows: PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM. Prefer the editor
  tools for file writes.
- Streamlit ≥ 1.50: use `width="stretch"`, not the deprecated `use_container_width`. In
  AppTest, Altair charts appear as `at.get("vega_lite_chart")`.
- The tree's default N = 500 gives about 2 correct decimals (ATM error ≈ 0.004).
