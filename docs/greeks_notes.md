# The Greeks: the intuition

A Greek is a **sensitivity**: how much the option price moves when one input
moves and everything else is held fixed. Traders use them to understand and
hedge risk. optionlab computes them with closed-form formulas (`delta`, `gamma`,
`vega`, `theta`, `rho`) and also by bump-and-reprice (`numerical_greeks`).

## Each Greek in one paragraph

**Delta (dP/dS).** If the stock moves up 1, the option moves about delta. It is
also the hedge ratio: short `delta` shares against one long call and small moves
cancel out. A call's delta goes from 0 (deep OTM) to e^(-qT) (deep ITM). A put's
delta is the call's minus e^(-qT).

**Gamma (d²P/dS²).** How fast delta changes. High gamma means your hedge goes
stale quickly. It peaks at the money near expiry. Long options always have
gamma ≥ 0: the payoff is convex.

**Vega (dP/dσ).** Sensitivity to volatility. More volatility means more chance of a big
payoff, and losses are capped at the premium, so vega ≥ 0 for long options.

**Theta (−dP/dT).** What the passing of time does to the option's value. It is usually
negative ("time decay"): the option loses time value as expiry approaches.

**Rho (dP/dr).** Sensitivity to interest rates. A higher r makes the strike paid
later cheaper today: good for calls, bad for puts.

## Scaling conventions (read these twice)

| Greek | optionlab returns | Market convention | Convert |
|---|---|---|---|
| vega | per **1.00** change in σ (0.20 → 1.20) | per 1 vol point (0.20 → 0.21) | divide by 100 |
| theta | **per year**, sign = −dP/dT | per calendar day | divide by 365 |
| rho | per **1.00** change in r | per 1% | divide by 100 |

So ATM vega of 37.52 means **0.375 per vol point**, not 37.52.

Why theta has a minus sign: T is *time remaining*. When a day passes, T goes
*down*, so the change from time passing is −dP/dT.

## Why gamma and vega are the same for calls and puts

Put-call parity says C − P = S e^(−qT) − K e^(−rT). The right-hand side is a
straight line in S, so its second derivative in S is 0, and it doesn't contain σ
at all. Differentiate both sides: gamma_C = gamma_P and vega_C = vega_P. That's
why `gamma` and `vega` have no `option` argument.

## Positive theta can happen

A deep in-the-money European put is worth roughly K e^(−rT) − S. As T shrinks,
K e^(−rT) *grows* toward K, so the put gains value as time passes: theta > 0.
(An American put would just be exercised early. That is Phase 3.)

## The Black-Scholes PDE as a consistency check

Every option price in the model satisfies

    theta + (r − q) S delta + ½ σ² S² gamma − r P = 0

In words: a delta-hedged option earns the risk-free rate. Time decay (theta)
is paid for by the gain from convexity (gamma). Theta, delta and gamma come from
three different formulas, so if any one is wrong the equation fails. The test suite
checks it across a grid.

## How we know the formulas are right

1. **External reference:** Hull's worked example (S=49, K=50, T=20 weeks): delta 0.522,
   gamma 0.066, vega 12.1, theta −4.31/yr, rho 8.91.
2. **Bump-and-reprice:** central differences on `bs_price`, which is already tested,
   must agree with the analytic formulas.
3. **The PDE above.**

Call/put relations (e.g. delta_C − delta_P = e^(−qT)) are also tested, but they
come from parity, so they share parity's blind spot. A bug that changes calls and
puts **equally** cancels out. When we put a wrong time-decay term into both
call and put theta, the relation tests all still passed while 328 other tests failed.

## Choosing the bump size

A central difference has two error sources: *truncation* (≈ h², shrinks as h
shrinks) and *rounding* (≈ machine-eps × price / h, grows as h shrinks).

`numerical_greeks` bumps each input by a small fraction of its **natural scale**,
the change that moves d1 by about 1:

| Input | Bump | Why that scale |
|---|---|---|
| S | 1e-3 × S·σ√T | S·σ√T ≈ width of the price distribution at expiry |
| r | 1e-3 × σ/√T | d1 moves by √T/σ per unit of r |
| σ | 1e-4 × σ | relative |
| T | min(1e-5, T/2) | keeps T − h positive |

How we got there: every fixed bump size we tried eventually failed. With a
fixed σ bump of 1e-4, deep out-of-the-money, low-vol vega was off by 1.5e-5 relative.
Shrinking the bump 10× cut the error exactly 100×, the signature of h² truncation error.
Property-based testing (`hypothesis`) then found the same problem for S
(3-day, 3%-vol option) and r (10-year, 3%-vol option), and found a rounding-error
case (a put worth 256 × S) when the bump was made too small.

**Accuracy limit:** when σ√T < about 0.05, no bump size gives a reliable numerical
gamma in double precision. The analytic Greeks are still exact there, and they are
checked against QuantLib over the full range (`tests/test_quantlib_crosscheck.py`).
