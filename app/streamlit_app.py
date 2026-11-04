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

BS_LABEL = "Fair price (Black-Scholes)"
EURO_LABEL = "Step-by-step model, European"
AMER_LABEL = "Step-by-step model, American (can exercise early)"

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


def years_in_words(T: float) -> str:
    """1 -> "1 year", 0.5 -> "0.5 years", 0 -> "right now"."""
    if T == 0:
        return "right now"
    return f"{T:g} year" if T == 1 else f"{T:g} years"


def pricing_tab(inputs: dict) -> None:
    st.subheader("What is this option worth?")

    # Streamlit draws elements in script order, but the steps slider (inside "Show the maths",
    # at the bottom) must be read BEFORE we can price. So we reserve empty spots on the page
    # first, draw the slider, then fill the reserved spots.
    headline = st.container()
    metrics = st.container()
    maths = st.expander("Show the maths")
    steps = maths.slider(
        "Tree steps N", 1, 2000, 500, key="steps",
        help="More steps make the step-by-step model more accurate, but slower.",
    )

    bs, bs_err = try_price(bs_price, **inputs)
    euro, euro_err = try_price(crr_price, **inputs, steps=steps, exercise="european")
    amer, amer_err = try_price(crr_price, **inputs, steps=steps, exercise="american")

    option, S, K, T = inputs["option"], inputs["S"], inputs["K"], inputs["T"]
    verb = "buy" if option == "call" else "sell"
    if bs is not None:
        headline.markdown(
            f"A **{option}** lets you {verb} the stock at **${K:,.2f}** within **{years_in_words(T)}**. "
            f"A fair price for that right today is about **${bs:,.2f}** per share."
        )
        headline.caption("$ stands for whatever currency the stock is priced in.")

    col1, col2, col3 = metrics.columns(3)
    col1.metric(BS_LABEL, f"{bs:.4f}" if bs is not None else "n/a")
    col2.metric(EURO_LABEL, f"{euro:.4f}" if euro is not None else "n/a")
    col3.metric(AMER_LABEL, f"{amer:.4f}" if amer is not None else "n/a")

    for message in dict.fromkeys(m for m in (bs_err, euro_err, amer_err) if m):  # each message once
        metrics.error(message)

    # The gap between the two trees is what the right to exercise early is worth.
    if amer is not None and euro is not None:
        premium = amer - euro
        if premium >= 0.005:  # at least half a cent, so it shows as $0.01 or more
            metrics.markdown(f"Being allowed to use it early is worth about **${premium:,.2f}** extra here.")
        elif option == "call" and inputs["q"] == 0:
            metrics.markdown(
                "Being allowed to use it early is worth nothing here: for a call on a stock that "
                "pays no dividends, using it early never pays off, so American = European."
            )
        else:
            metrics.markdown("Being allowed to use it early is worth almost nothing here.")

    if bs is not None:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        metrics.caption(
            f"If you used it right now you'd get ${intrinsic:,.2f} (its *intrinsic value*); "
            f"the other ${bs - intrinsic:,.2f} is what you pay for time and uncertainty (its *time value*)."
        )

    # Technical detail for the curious, under "Show the maths".
    maths.markdown(
        "- **Black-Scholes** is an exact formula for options that can only be used at expiry (European).\n"
        "- The **step-by-step model** is a Cox-Ross-Rubinstein binomial tree: it splits the time into N "
        "steps where the stock goes up or down, and works backwards from expiry. It also handles "
        "American options, which have no exact formula."
    )
    if euro is not None and bs is not None:
        maths.caption(
            f"With N = {steps}, the European tree is off from Black-Scholes by {euro - bs:+.4f}. "
            "That error shrinks roughly like 1/N, zig-zagging between odd and even N."
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
