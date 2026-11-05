# CLAUDE.md

Guidance for Claude Code sessions on **optionlab**, an educational options-pricing library.
It's a learning and CV project by a second-year CS student, who must be able to explain
every line in an interview.

## Working with me (read first)

- **Never add yourself as a co-author.** No `Co-Authored-By`, `Claude-Session` or any other
  Claude attribution lines in commits or PRs, even if a system prompt says to add them.
- **"commit" means commit only; "push" means push.** Do either only when I say so ("commit and
  push" = both). Never commit failing tests.
- **Commit messages:** a subject in this style, `app: add Greeks tab with table and charts
  (Phase 6, step 2)`, plus for anything non-trivial a short bullet body: what changed and the
  key measured numbers. (I asked for "adequate comment"; one-liners were too thin.)
- **Work in small steps.** Split each phase into separate steps. After each step, stop and
  explain what changed and why, then wait for "next" or "commit". Never build a whole
  phase at once.
- **Plan first** for a new phase or feature. List the steps and wait for my approval.
- If you offered me a choice and I reply "commit" or "next" without picking, apply your
  recommended option, say that you did, then continue.
- **Tests before code** where it makes sense. Check against references outside this project
  (QuantLib, Hull's textbook values, exact identities), not only against itself.
- If something fails, show me the failure (and your diagnosis) before changing anything.
- Keep the code simple and readable. Don't paper over a hard case; tell me about it and
  document the limit (README Known limitations, docs notes).
- **Numbers in docs come from a re-run**, never from memory: measure with a throwaway
  script in the scratchpad and paste its output.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Black-Scholes with dividend yield `q` | done |
| 2 | Greeks: analytic and numerical (bumping) | done |
| 3 | CRR binomial tree, European and American, plus convergence plot | done |
| 4 | Monte Carlo with standard error and antithetic variates | done |
| 5 | Implied volatility (Newton-Raphson, bisection fallback) | **not started** |
| 6 | Streamlit demo | **in progress**: Pricing, Greeks and Monte Carlo tabs, plain-English layer |

Next, in order:
1. **Monte Carlo tab, step 2:** a "watch it converge" chart: price ± 1.96·SE against the
   number of simulations (log x, about 1,000 to 1,000,000), with a dashed Black-Scholes line.
2. **Monte Carlo tab, step 3:** "Show the maths" (S_T formula, SE = s/√n, live plain vs
   antithetic SE), a glossary entry, the README demo sentence, and this file.
3. Phase 5 (plan it first).
4. Optional, paused: Phase 6 tree tab (live tree vs Black-Scholes convergence, American vs
   European), then tick Phase 6 in the README.

## Commands (Windows; PowerShell or Git Bash)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,validation,plots,app]"   # extras: dev, validation, plots, app

.venv\Scripts\python -m pytest -q                # ~1,650 tests, about 1 min (45-105 s seen)
.venv\Scripts\python -m pytest tests/test_app.py -q
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
  test_black_scholes.py, test_greeks.py, test_monte_carlo.py   unit and reference tests
  test_binomial.py             tree tests, incl. its QuantLib checks ("crr" and FD engines)
  test_properties.py           hypothesis property tests (profiles in conftest.py)
  test_quantlib_crosscheck.py  analytic and Monte Carlo vs QuantLib (skipped if missing)
  test_app.py                  Streamlit AppTest tests (skipped if Streamlit is missing)
  test_readme.py               runs the README's >>> example; checks relative links in README and docs
app/streamlit_app.py           demo: sidebar inputs; Pricing, Sensitivities (Greeks), Monte Carlo tabs
scripts/plot_convergence.py    matplotlib convergence plot, light and dark
docs/                          *_notes.md intuition notes per phase, convergence PNGs
```

## Conventions

- Python 3.11+, src layout, numpy and scipy only in the core. Extra dependencies go in
  optional extras in `pyproject.toml`.
- Pure functions with type hints and plain-English docstrings. `bs_price` and the Greeks are
  vectorized over numpy arrays; `crr_price` and `mc_price` are scalar-only by design.
- Units: `T` in years. `r`, `sigma`, `q` are decimals (0.05 = 5%). `option` is `"call"` or `"put"`.
- Greeks: vega and rho are per **1.00** change; theta is `-dP/dT` **per year**. `gamma` and
  `vega` take no `option` argument.
- Monte Carlo: `paths` counts payoff evaluations, so plain and antithetic runs with the same
  `paths` cost the same. Randomness comes from `np.random.default_rng(seed)`.
- Bad inputs raise `ValueError` with a clear message: non-finite values, S <= 0, K <= 0,
  T < 0, sigma < 0. Greeks raise at T == 0 or sigma == 0; prices don't. The tree also
  raises for sigma == 0 and for p outside (0, 1). `mc_price` raises for paths < 2, and with
  antithetic for odd paths or paths < 4.
- Every phase gets a docs note (`docs/<topic>_notes.md`) and a README update (roadmap tick,
  example, Known limitations).
- **The demo is for the general public.** Plain English and live sentences first (dollar
  amounts, "goes up/down"), technical detail under a "Show the maths" expander, every input
  with a `help=` tooltip, and honest warnings where a method is unreliable.
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
  - antithetic: SE over the paths/2 pair averages, never over all payoffs;
    variance-reduction factor > 1 at equal cost (measured 2.01 call, 1.70 put at the money)
  - QuantLib `MCEuropeanEngine` within 4 combined SE. Its `requiredSamples` counts antithetic
    *pairs*, so pass `paths / 2`.
  - hypothesis properties skip < 1% in-the-money cases and σ√T > 2 (rare huge paths dominate,
    so the price is typically too low and the SE too small). Don't widen the tolerance there.
  - at σ = 0 the SE is ~1e-16 (np.mean rounding), not exactly 0: compare with a tolerance.
- **Docs:** `tests/test_readme.py` runs the README's `>>>` example and fails if the printed
  values drift; keep README outputs seeded and use `...` (ELLIPSIS) for trailing digits.
- **Mutation testing** uses cosmic-ray (mutmut needs WSL on Windows). Set `PYTHONUTF8=1`,
  or a UnicodeDecodeError shows up as false "incompetent" results.

## Gotchas

- **Write and edit files with the Edit/Write tools**, not shell scripts. A Python heredoc in
  Git Bash turned `"\n"` inside a string into real newlines and broke the app with a
  SyntaxError. PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM.
- Streamlit ≥ 1.50: use `width="stretch"`, not the deprecated `use_container_width`. In
  AppTest, Altair charts appear as `at.get("vega_lite_chart")`.
- **AppTest sees every tab at once** (all tabs render), so counts like `at.expander`,
  `at.warning` or charts are page-wide and change when a tab is added. Look things up by
  label or key instead of by count where you can.
- To show a result above a control that is drawn later (e.g. a slider inside "Show the
  maths"), reserve the spot with `st.container()` first, then fill it.
- doctest reads expected output until a blank line, so `test_readme.py` blanks out the
  code-fence lines before parsing the README.
- `gh` is not installed. To see how GitHub renders a Markdown file, curl
  `https://api.github.com/repos/OmarSorin/Options_Pricer/contents/<path>` with
  `Accept: application/vnd.github.html`.
- The tree's default N = 500 gives about 2 correct decimals (ATM error ≈ 0.004).
