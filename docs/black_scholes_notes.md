# Black-Scholes: the intuition

## The one big idea: risk-neutral pricing

You can build a portfolio of the stock and cash that copies an option's payoff
at every moment. Since the copy and the option are worth the same, the option's
price is just the **cost of the copy**, and that cost does not depend on how
much investors like risk or on the stock's real-world expected growth.

That lets us do a trick: price as if the world were *risk-neutral*, where every
asset grows at the risk-free rate `r`. Then:

> option price = e^(-rT) x expected payoff (in the risk-neutral world)

Nobody claims the real world is risk-neutral. It is a calculation device that
gives the right price because of the copying argument.

## The model for the stock

In the risk-neutral world the stock follows a lognormal process: its log-price
at time T is normally distributed with

- mean:  ln(S) + (r - q - sigma^2/2) T
- std dev: sigma sqrt(T)

`q` is the continuous dividend yield: holders of the stock receive it, so the stock
itself grows at `r - q`. The `-sigma^2/2` is a convexity correction so that the
*average* price (not the average log-price) grows at `r - q`.

## Where d1 and d2 come from

A call pays `S_T - K` if `S_T > K`, otherwise 0. Taking the expectation splits into two pieces:

1. **Probability of finishing in the money.** The log-price is normal, so
   "log S_T > ln K" is a normal tail probability. Standardising it gives

   d2 = [ ln(S/K) + (r - q - sigma^2/2) T ] / (sigma sqrt(T))

   and P(S_T > K) = N(d2). Read d2 as: *how many standard deviations the
   expected log-price is above the strike.*

2. **Value received if in the money.** The expected stock price, *given* that you
   finish above K, is higher than the unconditional expectation, because you only
   count the good outcomes. Doing the integral shifts d2 up by one total-volatility
   unit:

   d1 = d2 + sigma sqrt(T) = [ ln(S/K) + (r - q + sigma^2/2) T ] / (sigma sqrt(T))

## What N(d1) and N(d2) mean

The call price is

    C = S e^(-qT) N(d1) - K e^(-rT) N(d2)

- **N(d2)** is the risk-neutral probability the call ends in the money. The second
  term is "pay the strike, discounted, but only when you exercise".
- **N(d1)** is the share-weighted version: it is the probability-like weight on
  receiving the stock. It is also the call's **delta**, the number of shares in
  the replicating portfolio. (Phase 2.)

So: a call is *long N(d1) shares, financed by borrowing K e^(-rT) N(d2).*

The put follows from put-call parity (C - P = S e^(-qT) - K e^(-rT)):

    P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

## Sanity checks to remember

- Deep in the money: N(d1), N(d2) -> 1, call -> S e^(-qT) - K e^(-rT).
- Deep out of the money: N(d1), N(d2) -> 0, call -> 0.
- sigma -> 0 or T -> 0: no randomness left, so the price is the discounted
  deterministic payoff max(S e^(-qT) - K e^(-rT), 0).
- Higher volatility raises both call and put prices (more upside is paid for, downside is capped).

## Why put-call parity can't catch a bug inside d1

Parity says C - P = S e^(-qT) - K e^(-rT). Subtract the two pricing formulas:

    C - P = S e^(-qT) [N(d1) + N(-d1)] - K e^(-rT) [N(d2) + N(-d2)]

Since N(x) + N(-x) = 1 for **any** x, d1 and d2 cancel out completely. Parity
holds whatever numbers you plug in for d1 and d2, so it only checks the discount
factors outside N(.), never the maths inside d1.

Financially: parity is model-free (long call + short put = a forward, under any
volatility model), so it can't test the model's probabilities.

Concrete case: dropping `q` from the drift in d1 still passes all 216 parity
cases, but gives a call of 9.1896 instead of 9.2270 (S=K=100, T=1, r=5%,
sigma=20%, q=2%). Two tests catch it:

- **Reference values with q != 0** (a q=0 reference can't see a q bug).
- **No-arbitrage bounds over a grid with q > 0 and low spots**: the buggy
  formula produces prices below the arbitrage lower bound (even negative ones).

Lesson: an identity test only checks what doesn't cancel out of the identity.
