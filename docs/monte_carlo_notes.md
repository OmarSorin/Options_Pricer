# Monte Carlo pricing: the intuition

## The idea

Under risk-neutral pricing, an option is worth its **discounted expected payoff**.
Black-Scholes computes that expectation with a formula and the tree with a grid. Monte
Carlo just **estimates** it: simulate many possible stock prices at expiry, compute the
payoff of each, and average.

    price ≈ e^(-rT) * (payoff_1 + payoff_2 + ... + payoff_n) / n

The answer is random, so every price comes with a **standard error** (SE) that says how
far off it is likely to be. `mc_price` returns both: `MCResult(price, std_error, paths)`.

## Exact sampling: one step, no discretisation error

A European option only cares about the stock price **at expiry**. Under Black-Scholes that
price is lognormal and can be drawn exactly from one standard normal number Z:

    S_T = S * exp((r - q - sigma²/2) T + sigma √T Z),    Z ~ N(0, 1)

The `- sigma²/2` term makes the average of S_T equal to the forward, S·e^((r−q)T), the same
risk-neutral rule as in the tree. There is no time-stepping, so unlike the tree there is no
discretisation error: the **only** error is statistical. That is why plain Monte Carlo
can be checked so precisely against Black-Scholes.

## Why the error falls like 1/√N

Let X be one discounted payoff, with standard deviation s. The average of n independent
copies has

    Var(mean) = s² / n,    so    SE = s / √n.

We don't know s, so we estimate it from the samples (`std(ddof=1)`, the unbiased sample
variance). For the reference call (S = K = 100, T = 1, r = 5%, σ = 20%), s ≈ 14.73, so:

| paths | SE |
|---|---|
| 10,000 | ≈ 0.147 |
| 100,000 | ≈ 0.047 |
| 1,000,000 | ≈ 0.015 |

**10× more accuracy costs 100× more paths.** Compare the tree, whose error falls like 1/N:
10× the steps for 10× the accuracy. Monte Carlo is slow to converge, but its rate does not
depend on the number of random inputs, which is why it wins for complex, many-asset products.

Rule of thumb: the true price lies in `price ± 1.96·SE` about 95% of the time.

## Antithetic variates: use every draw twice

For each draw Z, also use **−Z**. When Z gives a high stock price, −Z gives a low one, so the
two payoffs tend to move in opposite directions and their average is steadier than two
independent payoffs.

The maths, for a pair average `Y = (f(Z) + f(−Z)) / 2`, where ρ is the correlation of f(Z)
and f(−Z):

    Var(Y) = (s²/4) (1 + 1 + 2ρ) = (s²/2) (1 + ρ)

At the **same cost** (n payoff evaluations, i.e. n/2 pairs), plain Monte Carlo has variance
s²/n and antithetic has (s²/2)(1 + ρ) / (n/2) = s²(1 + ρ)/n. So

    variance-reduction factor = 1 / (1 + ρ)

It helps exactly when ρ < 0. For calls and puts that is guaranteed: the payoff only moves
one way as Z grows (it is **monotone** in Z), and a monotone function of Z and the same
function of −Z are negatively correlated.

Measured at the reference option:

| | ρ = corr(f(Z), f(−Z)) | predicted 1/(1+ρ) | measured factor (100k paths) |
|---|---|---|---|
| call | −0.501 | 2.00 | 2.01 |
| put | −0.414 | 1.71 | 1.70 |

A factor of 2 means antithetic gives the accuracy of 200,000 plain paths for the cost of 100,000.

In `mc_price`, `paths` counts **payoff evaluations** either way, so plain and antithetic runs
with the same `paths` cost the same and can be compared fairly. (QuantLib's
`MCEuropeanEngine` counts antithetic *pairs* instead, so the cross-check passes it `paths / 2`.)

## The pitfall: the antithetic SE comes from the pairs

f(Z) and f(−Z) are **not** independent, so the SE must be computed over the n/2 pair
averages Y, never over all n payoffs as if they were independent. The wrong way gets the
price exactly right and the error wrong. Here it is **29–39% too big**, because it ignores
the negative correlation that makes antithetic work. QuantLib does it the same way we do:
for 100,000 evaluations at the reference call its SE is 0.0329, ours 0.0331.

## How the standard error itself is tested

"Price within 4 SE of Black-Scholes" passes too easily if the SE is too big. So the tests
check the SE directly, in three ways:

- **Coverage.** Over 1,000 seeds × 10,000 paths, `price ± 1.96·SE` must contain the
  Black-Scholes price 93–97% of the time (theory: 95%). Measured: call 94.3%, put 95.0%;
  antithetic 93.6% for both.
- **Spread.** Over the same 1,000 runs, the reported SE must match the *actual* standard
  deviation of the prices to within 7%. Measured ratio: 0.984 (call) and 0.985 (put),
  antithetic. The buggy pairs SE gives 1.393 and 1.288 and fails. This check catches an SE
  that is too big as easily as one that is too small; coverage alone is weak on that side.
- **Rate.** The log-log slope of SE against paths (1,000 to 1,000,000) must lie in
  [−0.55, −0.45]. Measured: −0.4998 (theory: −0.5).

Plus the outside references: the Black-Scholes reference values (10.4506; 9.2270 and 6.3301
with q = 2%), QuantLib's `MCEuropeanEngine` (agreement within 4 combined SE on 132 cases), and
hypothesis properties (agreement with Black-Scholes and put-call parity within 5 SE on
random inputs). Every test uses fixed seeds, so it is deterministic.

## Where Monte Carlo breaks down

- **Deep out of the money.** If fewer than about 1% of paths finish in the money, a run
  sees only a handful of non-zero payoffs, and the SE estimate itself becomes unreliable.
  It can even be 0 while the true price is > 0. The property tests skip those cases.
- **Heavy tails: σ√T above about 2.** The relative variance of S_T is e^(σ²T) − 1: 1.7 at
  σ√T = 1, 54 at 2, about 8,100 at 3. Far out, a few rare, huge paths carry most of the
  price. Example: S = K = 1, T = 8 years, σ = 162.5% (σ√T = 4.6), true price 0.978.
  Only 1.1% of paths finish in the money, and the top 0.01% of paths produce 77% of the
  total payoff. A typical run misses them, so the price comes out **too low** and the SE
  **too small**: 20,000 paths give 0.262 ± 0.114, and 2 million paths still give 0.64–0.89
  (one seed gives 2.58). The estimate is unbiased on average, but its typical value is
  wrong. In practice σ√T = 2 means 200% volatility over one year, so this is rare.
- **European only.** Plain Monte Carlo doesn't know when to exercise early. American
  options need a regression method such as Longstaff-Schwartz; use `crr_price` instead.

## Sources

- P. Boyle (1977), "Options: A Monte Carlo approach", *Journal of Financial Economics* 4(3).
  The original Monte Carlo method for option pricing.
- P. Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, Springer. Exact
  simulation of geometric Brownian motion, and antithetic variates in the chapter on
  variance reduction.
