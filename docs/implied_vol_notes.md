# Implied volatility: the intuition

## The idea

Black-Scholes turns a volatility into a price. In a real market it is the other way round:
the **price** is what you can see (it's quoted on the screen), and the volatility is not. So
traders run the formula backwards: *which σ would make Black-Scholes give exactly this
price?* That σ is the **implied volatility**, the volatility the market is implying.

    find σ such that   bs_price(S, K, T, r, σ, q) = market price

It's how options are quoted in practice. "This call trades at 23.5% vol" means more than
"this call costs $1.875", because it can be compared across strikes, expiries and stocks.
Hull's example: a call with S = 21, K = 20, r = 10%, T = 3 months, priced at 1.875, has
implied volatility 0.2345 (23.5%).

    implied_vol(1.875, S=21, K=20, T=0.25, r=0.1)    # 0.2345129...  (tested in the README)

## Why there is exactly one answer, or none

Two facts make the problem well posed:

1. **The price always rises with σ** (vega > 0): more uncertainty helps an option, whose
   downside is capped. So at most one σ fits a given price.
2. **As σ runs from 0 to ∞, the price sweeps its whole no-arbitrage range.** With
   Se = S·e^(−qT) and Ke = K·e^(−rT):

   | | σ → 0 | σ → ∞ |
   |---|---|---|
   | call | max(Se − Ke, 0) | Se |
   | put | max(Ke − Se, 0) | Ke |

So a price strictly inside that range has exactly one implied volatility, and a price
outside it has none. `implied_vol` raises `ValueError` for the latter, and also for a price
exactly on the lower bound (σ would be 0, and the price says nothing useful about
volatility there), and at T = 0 (the option is worth its intrinsic value whatever σ is).

## The method: Newton, kept inside a bracket

There is no formula for σ, so we solve numerically.

**Newton-Raphson** follows the tangent. With the slope vega = dP/dσ:

    σ_new = σ − (bs_price(σ) − price) / vega(σ)

Near the answer it is very fast: the number of correct digits roughly doubles each step.
But it can fail badly when it starts far away. The usual starting guess (Brenner and
Subrahmanyam), σ₀ ≈ √(2π/T) · price / S, is accurate at the money and useless out of it.
For the call S = 100, K = 150, T = 1, σ = 20%, priced at 0.3596, it guesses **0.0090**.
There vega is exactly 0 in floating point, so plain Newton divides by zero on its first step.

**Bisection** is the opposite: slow but safe. If the price at `lo` is too low and the price
at `hi` too high, the answer is in between; check the midpoint and keep the half that still
brackets it. Each step halves the bracket, about 3 digits per 10 steps, and it can't fail.

`implied_vol` combines them:

1. **Bracket the answer.** At σ = 0 the price is the lower bound, below the target, so
   `lo = 0`. Start `hi = 1` and double it until the price there is above the target.
2. **Start** from the Brenner-Subrahmanyam guess, floored at 0.001. (For a very cheap option
   the formula gives absurd values like 1e-310, where d1 overflows.)
3. **Each step:** shrink the bracket using the sign of `bs_price(σ) − price`, then try a
   Newton step. If it lands inside the bracket, take it; otherwise take the midpoint.

## Two refinements that random testing forced on us

The first version passed every hand-written test. Random inputs broke it twice: first a
throwaway scan of 200,000 cases, then hypothesis.

**1. Stop on σ, not on the price.** The first version stopped when the price matched to
within 1e-10. That sounds strict, but it's an *absolute* amount of money. For the call
S = 100, K = 300, T = 3 months, the true price at σ = 20% is 1.4e-27, and **every σ from 0
to 0.33** prices it to within 1e-10 (at σ = 0.1 the price is 2.5e-105, still only 1.4e-27
away). So the solver accepted its first guess, and on 200,000 random cases σ was
off by up to **2.0**. Now it stops when a step changes σ by less than `tol` = 1e-10, which
means the same thing for a $500 option and a $1e-27 one.

**2. Newton on ln(price), not on the price.** Hypothesis then found a call (S = 1, K = 52,
T = 1, σ = 25%) priced at 1.6e-57 that didn't converge in 100 steps. Far out of the money
the price behaves like exp(−c/σ²): it falls by dozens of orders of magnitude as σ drops, so
the price curve is extremely bent. From above, each Newton step falls far short of the
answer, and the solver crept from σ = 0.5 down by about 0.007 per step (101 steps needed). The logarithm of
the price, about −c/σ², is nearly straight, so Newton on

    f(σ) = ln bs_price(σ) − ln price,    f'(σ) = vega / bs_price(σ)

lands close in one step. Measured with the same bracket and stopping rule:

| case | price | Newton on price | Newton on ln(price) |
|---|---|---|---|
| far OTM, S = 1, K = 52 | 1.6e-57 | 101 | 5 |
| very cheap, K = 300, T = 3 months | 1.4e-27 | 57 | 8 |
| deep OTM put, K = 50 | 3.3e-4 | 13 | 8 |
| at the money | 10.45 | 4 | 5 |
| 3,000 random cases, worst | | > 500 (2 never converged) | 52 |

Steps taken by the final version (each step is one `bs_price` and one `vega` call):

| case | price | steps | error in σ |
|---|---|---|---|
| at the money | 10.5 | 5 | 3e-17 |
| Hull's example | 1.87 | 5 | 2e-16 |
| out of the money, K = 150 | 0.36 | 8 | 1e-16 |
| deep OTM put, K = 50 | 3.3e-4 | 8 | 3e-16 |
| one week, 90% vol | 0.84 | 7 | 1e-15 |
| very cheap, K = 300, T = 3 months | 1.4e-27 | 8 | 3e-15 |
| far OTM, S = 1, K = 52 | 1.6e-57 | 5 | 0 |
| in the money, low vol (S = 100, K = 60, σ = 10%) | 45.7 | 15 | 3e-12 |

## The limit: deep in the money, the price doesn't contain σ

A deep in-the-money option is almost all intrinsic value. A float stores about 16
significant digits, so a price of 304.877058… has room for very little time value: changing
the price by one part in 10^16 changes σ by about

    ε · price / vega      (ε = 2.2e-16, the relative rounding step of a float)

Call with K = 100, T = 1, r = 5%, σ = 20%, measured:

| S | price | time value | vega | ε·price/vega | implied_vol | after the next float up |
|---|---|---|---|---|---|---|
| 300 | 204.88 | 2.6e-8 | 4.6e-6 | 1e-8 | 0.1999999951 | 0.2000000050 |
| 350 | 254.88 | 2.0e-10 | 4.4e-8 | 1e-6 | 0.1999998032 | 0.2000011010 |
| 400 | 304.88 | 1.8e-12 | 4.9e-10 | 1e-4 | 0.2000674461 | 0.2001376184 |
| 420 | 324.88 | 2.8e-13 | 8.5e-11 | 9e-4 | 0.2003588565 | 0.2006435228 |
| 450 | | rounds to 0 | | | raises | |

This isn't a solver bug: the information is gone before the solver starts. No method can do
better, and the answer still reproduces the price.

The same effect is much bigger with real quotes, which have only 2 decimals. For a
3-month call with K = 100, σ = 20%, adding **one cent** to the price moves the implied
volatility by 0.05 vol points at the money (S = 100, vega 19.6), 1.5 at S = 130 (vega 0.52)
and 9 at S = 150 (vega 0.004). This is one reason volatility surfaces are usually built
from out-of-the-money options.

## How it is tested

- **References outside the project:** Hull's example (0.235); the CLAUDE.md reference prices
  (10.4506, 9.2270, 6.3301) give back 20% to 1e-5 (they are rounded to 4 decimals); a put
  price built from the call price by put-call parity (without `bs_price`) gives the same σ.
- **The fallback matters:** a test runs plain Newton on the K = 150 call and checks that it
  divides by zero, while `implied_vol` converges.
- **Round trips** σ → price → σ to 1e-8 on hand-picked cases, including prices of 1e-10,
  1e-27 and 1e-57.
- **Hypothesis property:** on random inputs, the σ found always reproduces the price (to
  1e-9 relative), and equals the true σ to 1e-8 wherever ε·price/vega < 1e-10 (the price
  still determines σ) and the price is above 1e-300 (see the second limit below).
- **QuantLib:** `VanillaOption.impliedVolatility` on 166 grid cases and on random inputs,
  agreeing to 1e-8. Two things were checked first. Its default `accuracy` (1e-4) is loose,
  1e-5 off at the money, so the test passes 1e-12. And it is only a valid reference where
  vega ≥ 1e-4: below that its solver stops on the price error (the bug we fixed above).
  Against the true σ on 4,000 random cases:

  | vega | QuantLib, worst error | ours, worst error |
  |---|---|---|
  | < 1e-12 | 0.67 (it returns its minVol, 1e-7) | 8e-15 |
  | 1e-12 to 1e-8 | 1.1e-2 | 2e-15 |
  | 1e-8 to 1e-4 | 6.0e-7 | 2e-15 |
  | ≥ 1e-4 | 2.1e-10 | 9.9e-11 |

- **A random scan** (not part of the test suite; a throwaway script, 50,000 cases over the
  same ranges as the property tests, warnings turned into errors). No warnings. 1,059 cases
  raised, and every one was a price that rounds to exactly its lower bound. The σ found
  always reproduced the price, to within 4.3e-11 × (S + K). Error in σ, by vega:

  | vega | cases | worst error in σ |
  |---|---|---|
  | ≥ 1e-2 | 46,631 | 1.0e-10 |
  | 1e-6 to 1e-2 | 964 | 9.9e-8 |
  | 1e-20 to 1e-6 | 819 | 6.2e-2 |
  | 1e-100 to 1e-20 | 403 | 8.6e-11 |
  | 1e-300 to 1e-100 | 122 | 8.7e-11 |
  | < 1e-300 | 2 | 1.5e-4 |

  The big errors in the middle bands are all deep in the money: across the 1,783 cases with
  vega between 1e-20 and 1e-2, every error was below 1e-10 + 10·ε·price/vega (worst ratio
  0.87), i.e. within the limit above. Deep *out* of the money (tiny vega, tiny price) σ is
  accurate to 1e-10, because a tiny price still has 16 significant digits.

## A second limit: prices below 1e-300

The two cases in the last row are prices of 7.3e-309 and 7.7e-306, at the very bottom of
what a float can hold (the smallest normal float is 2.2e-308). There `bs_price` itself is
wrong. Its two terms, each around 1e-307, cancel, and what is left is rounding noise:
against a 60-digit mpmath reference, for the first of these options (inputs rounded to 4
digits) it returns 5.9e-309 where the exact price is 1.4e-311. Down to 7e-298 it is still
accurate to 2.5e-11. No σ can be "right" for a formula that is wrong, so the property test
skips prices below 1e-300.
Nobody quotes an option at 1e-300 dollars, so this only matters for random testing.

## Sources

- J. Hull, *Options, Futures, and Other Derivatives*, the implied volatility section of the
  Black-Scholes-Merton chapter (the 1.875 example).
- M. Brenner and M. Subrahmanyam (1988), "A simple formula to compute the implied standard
  deviation", *Financial Analysts Journal* 44(5). The at-the-money starting guess.
- P. Jäckel (2015), "Let's be rational", *Wilmott* 2015(75). The standard reference on
  computing implied volatility to full double precision, including far out of the money.
