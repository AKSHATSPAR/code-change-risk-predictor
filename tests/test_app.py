"""Run the Streamlit app headlessly on every page and check nothing crashes.

The environment variables point the app at files that don't exist, so it uses its fallbacks
(synthetic data + quick demo model) and the tests don't need the real dataset or trained model.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.setenv("RISKPREDICTOR_DATA", "missing/dataset.csv")
    monkeypatch.setenv("RISKPREDICTOR_BUNDLE", "missing/best_model.joblib")
    monkeypatch.setenv("RISKPREDICTOR_REPORTS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception
    return at


def open_page(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value == ("Code Change Risk Predictor" if page == "Overview" else page)
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_overview_uses_fallbacks(app):
    assert app.title[0].value == "Code Change Risk Predictor"
    assert any("synthetic demo data" in info.value for info in app.info)
    assert app.metric[0].value == "2,000"


def test_leaderboard_without_reports(app):
    at = open_page(app, "Model Leaderboard")
    assert any("riskpredictor.train" in info.value for info in at.info)


def test_leaderboard_with_reports(app, tmp_path):
    pd.DataFrame({  # same columns as the file written by riskpredictor.train
        "rank": [1, 2], "model": ["Logistic Regression", "Random Forest"], "cv_roc_auc": [0.74, 0.73],
        "test_roc_auc": [0.77, 0.78], "test_pr_auc": [0.36, 0.40], "test_f1": [0.39, 0.39], "note": [None, "tuned"],
    }).to_csv(tmp_path / "model_comparison.csv", index=False)
    at = open_page(app, "Model Leaderboard")
    assert "Logistic Regression" in at.success[0].value  # rank 1 (the demo model is not in the table)
    assert len(at.dataframe) >= 1


def test_score_local_rejects_bad_repo(app, tmp_path):
    pytest.importorskip("riskpredictor.explain")
    at = open_page(app, "Score a Local Commit")
    at.text_input[0].set_value(str(tmp_path))
    at.text_input[1].set_value("abc1234")
    at.button[0].click().run()
    assert not at.exception
    assert "not a git repository" in at.error[0].value


@pytest.mark.parametrize("page", ["Explain a Commit", "Global Explanations"])
def test_explanation_pages(app, page):
    pytest.importorskip("riskpredictor.explain")
    at = open_page(app, page)
    assert len(at.get("plotly_chart")) >= 1


def test_explain_page_lime(app):
    pytest.importorskip("riskpredictor.explain")
    at = open_page(app, "Explain a Commit")
    at.checkbox(key="lime_explain").check().run()
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_what_if_simulator(app):
    pytest.importorskip("riskpredictor.explain")
    at = open_page(app, "What-If Simulator")
    at.number_input(key="wi_la").set_value(5000.0).run()  # a huge change...
    assert not at.exception
    button(at, "Suggest how to lower risk").click().run()  # ...then ask how to make it safer
    assert not at.exception
    assert len(at.dataframe) >= 1


ROOT = Path(APP).parents[1]
PAGES = ["Overview", "Model Leaderboard", "Explain a Commit", "What-If Simulator",
         "Score a Local Commit", "Global Explanations"]


@pytest.mark.skipif(not (ROOT / "models" / "best_model.joblib").exists()
                    or not (ROOT / "data" / "processed" / "dataset.csv").exists(),
                    reason="real dataset / trained model not built yet")
def test_every_page_with_real_artifacts(monkeypatch):
    for var in ["RISKPREDICTOR_DATA", "RISKPREDICTOR_BUNDLE", "RISKPREDICTOR_REPORTS"]:
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert not any("demo" in w.value for w in at.sidebar.warning)  # the trained model was loaded
    for page in PAGES:
        open_page(at, page)
