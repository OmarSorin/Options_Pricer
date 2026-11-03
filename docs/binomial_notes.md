# The binomial tree: the intuition

## The idea

Black-Scholes lets the stock move continuously. A binomial tree chops the time to
expiry into `N` short steps. In each step the stock either goes **up** by a factor `u`
or **down** by a factor `d`. After N steps there are N + 1 possible final prices.
As N grows, this "coin-flip" walk looks more and more like Black-Scholes' lognormal
stock, and the tree price converges to the Black-Scholes price.

## u, d and p (Cox, Ross & Rubinstein, 1979)

    dt = T / N
    u  = exp(sigma * sqrt(dt))       the up factor: one standard deviation of a step
    d  = 1 / u                       the down factor: up then down returns to the start
    p  = (exp((r - q) dt) - d) / (u - d)

`p` is the **risk-neutral** probability of an up move. It is chosen so that the stock's
expected growth over one step is exactly `exp((r - q) dt)`, the same rule as in
Black-Scholes (risk-neutral pricing). If a step is too long for the given r, q and
sigma, p falls outside (0, 1) and the tree is meaningless, so `crr_price` raises an
error and suggests more steps.

**Why this exact p, and not QuantLib's?** QuantLib's "crr" engine uses a linearized
approximation, `p = 1/2 + 1/2 (r - q - sigma²/2) sqrt(dt) / sigma`. Both converge
to the same price, but only the exact p makes the tree's expected stock price
exactly S·e^((r−q)T). That makes put-call parity hold *exactly* on the tree at
every N, which gives us a very sharp test (see below). Hull uses the exact p too,
which is why our prices match his textbook to the printed digit.

## Backward induction

1. **At expiry** every node's value is just the payoff, e.g. `max(S_T - K, 0)`.
2. **Step back.** A node's value is the discounted, probability-weighted average of
   its two children: `e^(-r dt) * (p * up_child + (1 - p) * down_child)`.
3. Repeat until one node is left. That's today's price.

In the code each step is one numpy operation over all the nodes:
`values = discount * (p * values[1:] + (1 - p) * values[:-1])`.

## American options: one extra line

An American option can be exercised at any node. So at each node, the holder
compares **waiting** (the value from step 2) with **exercising now** (the payoff at
that node's stock price) and takes the larger:

    values = np.maximum(waiting_value, payoff(stock_at_this_step))

That's the whole difference between European and American on a tree.

## Why we can't test American prices against Black-Scholes

Black-Scholes prices **European** options only. There's no closed-form formula for
an American put. So our references for American prices are:

- **Hull's published values** for his American put example (S = K = 50, r = 10%,
  σ = 40%, 5 months): 4.49 (5 steps), 4.263 (30), 4.272 (50), 4.278 (100).
  Ours match all four to the printed digit.
- **QuantLib**, in two ways: its own CRR tree (the gap shrinks like 1/N, because
  of the different p) and its finite-difference PDE solver, a completely
  different numerical method.
- **Exact rules** that must hold whatever the price is (next section).

## Rules every American price must obey

- **American ≥ European.** The right to exercise early can't have negative value.
- **American ≥ intrinsic value.** You could always exercise immediately.
- **A call without dividends is never exercised early**, so its American and
  European prices are equal. Exercising gives you `S - K` now. Holding the option
  is worth at least `S - K·e^(-rT)` (a European call is worth at least that), which is
  more than `S - K` when r ≥ 0. Killing the option early also throws away its
  remaining time value.
- **A deep in-the-money put is exercised immediately.** Waiting only delays
  receiving K, losing interest on it. With S = 80, K = 100 the tree gives exactly
  20 = K − S.
- **Put-call parity becomes two bounds.** With early exercise, C − P is no longer
  exactly S·e^(−qT) − K·e^(−rT), but it must satisfy
  `S·e^(−qT) − K ≤ C − P ≤ S − K·e^(−rT)`.

## Convergence: why the error zig-zags

The tree's error shrinks like 1/N, but **not smoothly**. It jumps between odd and
even N, and depending on whether the strike lands on a node or between two nodes.
Sometimes the error passes through almost zero. That's why the tests check:

- a **bound**, |error| ≤ S·σ·√T / N at every N, and
- the **rate**, a log-log slope of about −1, fitted to the largest error in each
  window of N (the "envelope"),

and never "the error shrinks at every N", which would fail on a correct tree.
