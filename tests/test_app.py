"""Smoke tests for the Streamlit demo (app/streamlit_app.py).

Streamlit's AppTest runs the script headlessly, with no browser, so we can check that the
page renders, shows the library's numbers, and turns library errors into messages
instead of crashing. Skipped if Streamlit isn't installed (pip install -e ".[app]").
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from optionlab import bs_price, crr_price, delta, numerical_greeks, theta  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


BS_LABEL = "Fair price (Black-Scholes)"
EURO_LABEL = "Step-by-step model, European"
AMER_LABEL = "Step-by-step model, American (can exercise early)"


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_default_page_shows_library_prices():
    at = run_app()
    assert not at.exception
    # Defaults are the Phase 1 reference option: S = K = 100, T = 1, r = 5%, sigma = 20%, call.
    assert metric(at, BS_LABEL) == f"{bs_price(100, 100, 1, 0.05, 0.2):.4f}" == "10.4506"
    assert metric(at, EURO_LABEL) == f"{crr_price(100, 100, 1, 0.05, 0.2, steps=500):.4f}"


def test_headline_states_the_fair_price_in_words():
    text = markdown_text(run_app())
    assert "A **call** lets you buy the stock at **$100.00**" in text
    assert "about **$10.45** per share" in text


def test_early_exercise_sentence_for_a_put():
    at = run_app()
    at.radio(key="option").set_value("put").run()
    euro = crr_price(100, 100, 1, 0.05, 0.2, option="put", steps=500)
    amer = crr_price(100, 100, 1, 0.05, 0.2, option="put", steps=500, exercise="american")
    assert f"worth about **${amer - euro:.2f}** extra" in markdown_text(at)


def test_early_exercise_sentence_for_a_call_without_dividends():
    # Without dividends, early exercise of a call never pays off, so American = European.
    assert "never pays off" in markdown_text(run_app())


def test_tree_steps_slider_lives_in_show_the_maths():
    at = run_app()
    assert any(e.label == "Show the maths" for e in at.expander)
    at.slider(key="steps").set_value(50).run()
    assert metric(at, EURO_LABEL) == f"{crr_price(100, 100, 1, 0.05, 0.2, steps=50):.4f}"


def test_intro_explains_what_an_option_is():
    at = run_app()
    assert any("right, but not the obligation" in m.value for m in at.markdown)


def test_glossary_expander_is_there():
    at = run_app()
    assert any(e.label == "New to options? Start here" for e in at.expander)


def test_every_sidebar_input_has_a_help_tooltip():
    at = run_app()
    widgets = [at.radio(key="option"), at.number_input(key="S"), at.number_input(key="K")]
    widgets += [at.slider(key=k) for k in ("T", "r", "sigma", "q")]
    assert all(w.help for w in widgets)


def test_put_shows_american_tree_price():
    at = run_app()
    at.radio(key="option").set_value("put").run()
    assert not at.exception
    expected = crr_price(100, 100, 1, 0.05, 0.2, option="put", steps=500, exercise="american")
    assert metric(at, AMER_LABEL) == f"{expected:.4f}"


def test_library_error_is_shown_not_raised():
    at = run_app()
    at.slider(key="sigma").set_value(0.0).run()  # the tree rejects sigma == 0
    assert not at.exception
    assert any("sigma must be > 0" in e.value for e in at.error)
    # Black-Scholes is still fine at sigma == 0, so its price is still shown.
    assert metric(at, BS_LABEL) != "n/a"


def greeks_table(at: AppTest):
    return at.dataframe[0].value.set_index("Greek")


def test_greeks_table_shows_library_values():
    at = run_app()
    table = greeks_table(at)
    args = (100, 100, 1, 0.05, 0.2)
    assert table.loc["Delta", "Analytic"] == pytest.approx(delta(*args))
    assert table.loc["Delta", "Numerical (bumping)"] == pytest.approx(numerical_greeks(*args)["delta"])
    # Quoted units: theta per calendar day = theta per year / 365.
    assert table.loc["Theta", "As usually quoted"] == pytest.approx(theta(*args) / 365)


def test_greek_sentences_for_a_call():
    text = markdown_text(run_app())
    args = (100, 100, 1, 0.05, 0.2)
    assert f"the option's price goes up by about **${delta(*args):.2f}**" in text  # 0.64
    # theta is -dP/dT per year and negative here: the option loses value each day.
    assert f"the option loses about **${abs(theta(*args)) / 365:.3f}**" in text  # 0.018


def test_greek_sentences_flip_for_a_put():
    at = run_app()
    at.radio(key="option").set_value("put").run()
    assert f"the option's price goes down by about **${abs(delta(100, 100, 1, 0.05, 0.2, option='put')):.2f}**" in markdown_text(at)


def test_greeks_table_is_under_show_the_maths():
    at = run_app()
    assert sum(e.label == "Show the maths" for e in at.expander) == 2  # one per tab


def test_greeks_tab_draws_six_charts():
    at = run_app()
    assert len(at.get("vega_lite_chart")) == 6  # price + five Greeks


def test_greeks_error_at_expiry_is_shown_not_raised():
    at = run_app()
    at.slider(key="T").set_value(0.0).run()  # Greeks are undefined at T == 0
    assert not at.exception
    assert any("Greeks undefined" in e.value for e in at.error)
    assert len(at.dataframe) == 0
