"""optionlab demo: change the inputs with sliders and watch the prices react.

Run from the project root:

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Streamlit reruns this whole script from top to bottom every time a widget changes.
So the script is just: read the inputs, call the library, show the results.
"""

from __future__ import annotations

from collections.abc import Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from optionlab import bs_price, crr_price, delta, gamma, numerical_greeks, rho, theta, vega

SERIES = "#2a78d6"  # line colour, the same blue as the convergence plot
MUTED = "#8a8984"  # reference marks, readable on light and dark backgrounds

GLOSSARY = """
- **Call**: the right to *buy* the stock at the strike price. It pays off when the stock goes up.
- **Put**: the right to *sell* the stock at the strike price. It pays off when the stock goes down,
  a bit like insurance.
- **Strike price**: the fixed price written into the contract.
- **Expiry**: the date the option runs out. After that, the right is gone.
- **Volatility**: how much the stock tends to swing. More swing means a bigger chance of a big
  payoff, so the option is worth more.
- **Interest rate**: what risk-free money earns. It matters because paying the strike later
  is cheaper than paying it today.
- **Dividend yield**: cash the company pays its shareholders. An option holder doesn't get it,
  so dividends make calls cheaper and puts dearer.
- **European vs American**: a European option can only be used on the expiry date; an
  American one can be used on any day up to it.
"""


def sidebar_inputs() -> dict:
    """Draw the sidebar and return the option's inputs in the library's units.

    Rates and volatility are shown in percent because that's how people quote them,
    then divided by 100, because the library takes decimals (0.05 = 5%).
    """
    sb = st.sidebar
    sb.header("Your option")
    option = sb.radio(
        "Type", ["call", "put"], horizontal=True, key="option",
        help="A call is the right to BUY the stock at the strike price; a put is the right to SELL it.",
    )
    S = sb.number_input(
        "Stock price today (S)", min_value=0.01, value=100.0, step=1.0, key="S",
        help="What one share of the stock costs right now.",
    )
    K = sb.number_input(
        "Strike price (K)", min_value=0.01, value=100.0, step=1.0, key="K",
        help="The fixed price the option lets you buy (call) or sell (put) the stock at.",
    )
    T = sb.slider(
        "Time until expiry, years (T)", 0.0, 5.0, 1.0, 0.01, key="T",
        help="How long until the option runs out. 0.5 means six months.",
    )
    r = sb.slider(
        "Interest rate (r, %)", -2.0, 15.0, 5.0, 0.25, key="r",
        help="What a risk-free savings account would pay per year.",
    )
    sigma = sb.slider(
        "Volatility, how much the stock swings (σ, %)", 0.0, 100.0, 20.0, 0.5, key="sigma",
        help="Roughly how much the stock moves in a typical year. Calm stocks: 15-25%. Wild ones: 50% or more.",
    )
    q = sb.slider(
        "Dividend yield (q, %)", 0.0, 10.0, 0.0, 0.25, key="q",
        help="The share of the stock price paid out to shareholders each year as dividends.",
    )
    return dict(S=S, K=K, T=T, r=r / 100, sigma=sigma / 100, q=q / 100, option=option)


def try_price(pricer: Callable[..., float], **kwargs) -> tuple[float | None, str | None]:
    """Call a pricer and return (price, None), or (None, message) if it raises ValueError.

    The library rejects some inputs on purpose (e.g. the tree needs sigma > 0). Showing
    its message is more useful than crashing the page.
    """
    try:
        return float(pricer(**kwargs)), None
    except ValueError as err:
        return None, str(err)


def pricing_tab(inputs: dict) -> None:
    st.subheader("Prices")
    steps = st.slider("Tree steps N", 1, 2000, 500, key="steps")

    bs, bs_err = try_price(bs_price, **inputs)
    euro, euro_err = try_price(crr_price, **inputs, steps=steps, exercise="european")
    amer, amer_err = try_price(crr_price, **inputs, steps=steps, exercise="american")

    col1, col2, col3 = st.columns(3)
    col1.metric("Black-Scholes (European)", f"{bs:.4f}" if bs is not None else "n/a")
    col2.metric(
        "Tree, European",
        f"{euro:.4f}" if euro is not None else "n/a",
        # The gap to Black-Scholes is the tree's error; it shrinks like 1/N.
        delta=f"{euro - bs:+.4f} vs Black-Scholes" if euro is not None and bs is not None else None,
        delta_color="off",
    )
    col3.metric(
        "Tree, American",
        f"{amer:.4f}" if amer is not None else "n/a",
        # The gap between the two trees is what the right to exercise early is worth.
        delta=f"{amer - euro:+.4f} early-exercise premium" if amer is not None and euro is not None else None,
        delta_color="off",
    )

    for message in dict.fromkeys(m for m in (bs_err, euro_err, amer_err) if m):  # each message once
        st.error(message)

    if bs is not None:
        S, K = inputs["S"], inputs["K"]
        intrinsic = max(S - K, 0.0) if inputs["option"] == "call" else max(K - S, 0.0)
        st.caption(
            f"Intrinsic value (exercise now): {intrinsic:.4f}. "
            f"Time value (Black-Scholes minus intrinsic): {bs - intrinsic:.4f}."
        )


def analytic_greeks(S, K, T, r, sigma, q, option) -> dict:
    """All five analytic Greeks. S may be an array (the library is vectorized).

    gamma and vega take no `option` argument: they're the same for a call and a put.
    """
    return {
        "delta": delta(S, K, T, r, sigma, q, option),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, option),
        "rho": rho(S, K, T, r, sigma, q, option),
    }


# Library units, and how traders usually quote each Greek: (label, unit, divide by, quoted unit).
GREEK_UNITS = {
    "delta": ("Delta", "per 1.00 move in S", 1, "per 1.00 move in S"),
    "gamma": ("Gamma", "change in delta per 1.00 move in S", 1, "change in delta per 1.00 move in S"),
    "vega": ("Vega", "per 1.00 change in σ", 100, "per 1 vol point (σ + 1%)"),
    "theta": ("Theta", "per year", 365, "per calendar day"),
    "rho": ("Rho", "per 1.00 change in r", 100, "per 1% change in r"),
}


def spot_chart(df: pd.DataFrame, column: str, title: str, spot: float) -> alt.LayerChart:
    """One line against S, a dashed rule at today's S, and a hover readout.

    Hovering anywhere snaps to the nearest S on the curve and shows its value.
    """
    hover = alt.selection_point(fields=["S"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(df).encode(x=alt.X("S:Q", title="Spot S"))
    line = base.mark_line(color=SERIES, strokeWidth=2).encode(y=alt.Y(f"{column}:Q", title=None))
    points = (
        base.mark_point(color=SERIES, filled=True, size=60)
        .encode(
            y=f"{column}:Q",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),  # only the hovered point is visible
            tooltip=[alt.Tooltip("S:Q", format=".2f"), alt.Tooltip(f"{column}:Q", title=title, format=".4f")],
        )
        .add_params(hover)
    )
    today = alt.Chart(pd.DataFrame({"S": [spot]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="S:Q")
    return (line + points + today).properties(title=title, height=220)


def greeks_tab(inputs: dict) -> None:
    st.subheader("Greeks (Black-Scholes)")
    try:
        exact = analytic_greeks(**inputs)
        bumped = numerical_greeks(**inputs)
    except ValueError as err:  # Greeks are undefined at T == 0 or sigma == 0
        st.error(str(err))
        return

    # 1. Table: analytic vs finite-difference values, plus the usual trading units.
    rows = []
    for name, (label, unit, divisor, quoted_unit) in GREEK_UNITS.items():
        rows.append(
            {
                "Greek": label,
                "Analytic": exact[name],
                "Numerical (bumping)": bumped[name],
                "Library unit": unit,
                "As usually quoted": exact[name] / divisor,
                "Quoted unit": quoted_unit,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.6f") for c in ("Analytic", "Numerical (bumping)", "As usually quoted")},
    )
    st.caption(
        "Analytic = the closed-form formulas. Numerical = reprice with `bs_price` after a small bump "
        "(central differences). They should agree to about 5 digits; for very short expiries or low "
        "volatility (σ√T below about 0.05) the numerical gamma gets noisy."
    )

    # 2. Charts: price and each Greek as S moves, with everything else held fixed.
    S, K = inputs["S"], inputs["K"]
    spots = np.linspace(min(0.5 * K, 0.8 * S), max(1.5 * K, 1.2 * S), 201)
    others = {k: v for k, v in inputs.items() if k != "S"}
    df = pd.DataFrame({"S": spots, "price": bs_price(spots, **others), **analytic_greeks(spots, **others)})

    titles = {"price": "Price"} | {name: f"{label} ({unit})" for name, (label, unit, *_) in GREEK_UNITS.items()}
    st.caption(f"Each chart varies S and keeps K, T, r, σ, q fixed. The dashed line is today's S = {S:g}.")
    columns = st.columns(3)
    for i, (column, title) in enumerate(titles.items()):
        columns[i % 3].altair_chart(spot_chart(df, column, title, S), width="stretch")


def main() -> None:
    st.set_page_config(page_title="optionlab", layout="wide")
    st.title("optionlab")
    st.write(
        "An **option** is a contract that gives you the right, but not the obligation, to buy "
        "(a **call**) or sell (a **put**) a stock at a fixed price before a set date. "
        "This page works out what that right is worth today. "
        "Change the inputs on the left and every number below updates."
    )
    with st.expander("New to options? Start here"):
        st.markdown(GLOSSARY)

    inputs = sidebar_inputs()
    pricing, greeks = st.tabs(["Pricing", "Greeks"])
    with pricing:
        pricing_tab(inputs)
    with greeks:
        greeks_tab(inputs)


main()
