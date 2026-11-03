"""Plot how the CRR binomial tree converges to the Black-Scholes price.

Run from the project root:

    pip install -e ".[plots]"
    python scripts/plot_convergence.py

It writes docs/convergence.png (light) and docs/convergence-dark.png (dark), and
prints the fitted convergence slope, which should match the Step 3 test
(between -1.2 and -0.8; CRR theory says -1).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optionlab import bs_price, crr_price  # noqa: E402

# The option being priced: at-the-money European call (the Phase 1 reference case).
S, K, T, R, SIGMA, Q, OPTION = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call"

PRICE_STEPS = np.arange(10, 201)  # panel 1: small trees, where the zig-zag is visible
# Panel 2 uses the SAME range as the Step 3 test. Below N ~ 50 higher-order error
# terms still matter (fitting from N = 10 gives a slope of about -1.25); the 1/N rate
# is a large-N statement.
ERROR_STEPS = np.arange(50, 1001)
WINDOW = 50  # window size for the error envelope, same as the Step 3 test

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835", series="#3987e5"),
}

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs"


def tree_prices(steps: np.ndarray) -> np.ndarray:
    return np.array([crr_price(S, K, T, R, SIGMA, Q, OPTION, steps=int(n)) for n in steps])


def envelope_slope(steps: np.ndarray, abs_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Largest error in each window of N, and the log-log slope fitted to it (as in the tests)."""
    starts = range(0, len(steps) - WINDOW + 1, WINDOW)
    centers = np.array([steps[i : i + WINDOW].mean() for i in starts])
    peaks = np.array([abs_err[i : i + WINDOW].max() for i in starts])
    slope = np.polyfit(np.log(centers), np.log(peaks), 1)[0]
    return centers, peaks, slope


def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def draw(theme: str, exact: float, prices: np.ndarray, abs_err: np.ndarray, slope: float) -> pathlib.Path:
    t = THEMES[theme]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])

    # Panel 1: tree price vs N, with the Black-Scholes price as a reference line.
    left.axhline(exact, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)))
    left.plot(PRICE_STEPS, prices, color=t["series"], linewidth=1.5)
    left.annotate(
        f"Black-Scholes {exact:.4f}",
        xy=(PRICE_STEPS[-1], exact),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        color=t["ink2"],
        fontsize=9,
        bbox=dict(facecolor=t["surface"], edgecolor="none", pad=2),  # keep it readable over the lines
    )
    left.set_title("Tree price zig-zags toward Black-Scholes", loc="left", fontsize=11, color=t["ink"])
    left.set_xlabel("Number of steps N")
    left.set_ylabel("Call price")
    style_axes(left, t)

    # Panel 2: |error| vs N on log-log axes, with its envelope and a 1/N reference.
    centers, peaks, _ = envelope_slope(ERROR_STEPS, abs_err)
    right.loglog(ERROR_STEPS, abs_err, color=t["series"], linewidth=1.0, alpha=0.55)
    right.loglog(centers, peaks, color=t["series"], linewidth=2.0, marker="o", markersize=4)
    reference = peaks[0] * centers[0] / ERROR_STEPS  # c/N through the first envelope point
    right.loglog(ERROR_STEPS, reference, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)))
    right.annotate(
        "1/N reference",
        xy=(ERROR_STEPS[0], reference[0]),
        xytext=(4, 6),
        textcoords="offset points",
        ha="left",
        color=t["ink2"],
        fontsize=9,
    )
    right.annotate(
        f"envelope (max per {WINDOW} N), slope {slope:.2f}",
        xy=(centers[1], peaks[1]),
        xytext=(8, 8),
        textcoords="offset points",
        color=t["ink"],
        fontsize=9,
    )
    right.set_title("Error falls like 1/N", loc="left", fontsize=11, color=t["ink"])
    right.set_xlabel("Number of steps N")
    right.set_ylabel("|tree − Black-Scholes|")
    style_axes(right, t)

    fig.suptitle(
        f"CRR binomial tree vs Black-Scholes, European call  (S={S:g}, K={K:g}, T={T:g}, r={R:g}, σ={SIGMA:g})",
        x=0.01,
        ha="left",
        fontsize=10,
        color=t["ink2"],
    )
    fig.tight_layout()
    out = OUT_DIR / ("convergence.png" if theme == "light" else "convergence-dark.png")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def main() -> None:
    exact = bs_price(S, K, T, R, SIGMA, Q, OPTION)
    prices = tree_prices(PRICE_STEPS)
    abs_err = np.abs(tree_prices(ERROR_STEPS) - exact)
    _, _, slope = envelope_slope(ERROR_STEPS, abs_err)
    print(f"Fitted convergence slope (envelope): {slope:.3f}   (Step 3 test requires -1.2 to -0.8)")
    for theme in THEMES:
        print("wrote", draw(theme, exact, prices, abs_err, slope))


if __name__ == "__main__":
    main()
