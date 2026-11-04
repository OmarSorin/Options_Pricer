"""optionlab: a small options pricing library built for learning."""

from optionlab.binomial import crr_price
from optionlab.black_scholes import bs_price, d1_d2
from optionlab.greeks import delta, gamma, numerical_greeks, rho, theta, vega
from optionlab.monte_carlo import MCResult, mc_price

__all__ = [
    "bs_price",
    "d1_d2",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "numerical_greeks",
    "crr_price",
    "mc_price",
    "MCResult",
]
