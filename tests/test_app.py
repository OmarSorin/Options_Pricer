"""Smoke tests for the Streamlit demo (app/streamlit_app.py).

Streamlit's AppTest runs the script headlessly, with no browser, so we can check that the
page renders, shows the library's numbers, and turns library errors into messages
instead of crashing. Skipped if Streamlit isn't installed (pip install -e ".[app]").
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from optionlab import bs_price, crr_price  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_default_page_shows_library_prices():
    at = run_app()
    assert not at.exception
    # Defaults are the Phase 1 reference option: S = K = 100, T = 1, r = 5%, sigma = 20%, call.
    assert metric(at, "Black-Scholes (European)") == f"{bs_price(100, 100, 1, 0.05, 0.2):.4f}" == "10.4506"
    assert metric(at, "Tree, European") == f"{crr_price(100, 100, 1, 0.05, 0.2, steps=500):.4f}"


def test_put_shows_american_tree_price():
    at = run_app()
    at.radio(key="option").set_value("put").run()
    assert not at.exception
    expected = crr_price(100, 100, 1, 0.05, 0.2, option="put", steps=500, exercise="american")
    assert metric(at, "Tree, American") == f"{expected:.4f}"


def test_library_error_is_shown_not_raised():
    at = run_app()
    at.slider(key="sigma").set_value(0.0).run()  # the tree rejects sigma == 0
    assert not at.exception
    assert any("sigma must be > 0" in e.value for e in at.error)
    # Black-Scholes is still fine at sigma == 0, so its price is still shown.
    assert metric(at, "Black-Scholes (European)") != "n/a"
