"""optionlab demo: change the inputs with sliders and watch the prices react.

Run from the project root:

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Streamlit reruns this whole script from top to bottom every time a widget changes.
So the script is just: read the inputs, call the library, show the results.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from optionlab import bs_price, crr_price


def sidebar_inputs() -> dict:
    """Draw the sidebar and return the option's inputs in the library's units.

    Rates and volatility are shown in percent because that's how people quote them,
    then divided by 100, because the library takes decimals (0.05 = 5%).
    """
    st.sidebar.header("Option")
    option = st.sidebar.radio("Type", ["call", "put"], horizontal=True, key="option")
    S = st.sidebar.number_input("Spot price S", min_value=0.01, value=100.0, step=1.0, key="S")
    K = st.sidebar.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, key="K")
    T = st.sidebar.slider("Time to expiry T (years)", 0.0, 5.0, 1.0, 0.01, key="T")
    r = st.sidebar.slider("Interest rate r (%)", -2.0, 15.0, 5.0, 0.25, key="r")
    sigma = st.sidebar.slider("Volatility σ (%)", 0.0, 100.0, 20.0, 0.5, key="sigma")
    q = st.sidebar.slider("Dividend yield q (%)", 0.0, 10.0, 0.0, 0.25, key="q")
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


def main() -> None:
    st.set_page_config(page_title="optionlab", layout="wide")
    st.title("optionlab")
    st.write("Move the sliders on the left; every number below is recomputed by the library.")

    inputs = sidebar_inputs()
    (pricing,) = st.tabs(["Pricing"])
    with pricing:
        pricing_tab(inputs)


main()
