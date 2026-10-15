"""Tests for the scoring layer and the command-line interface (fast: synthetic data, demo model)."""

import joblib
import pandas as pd
import pytest

from riskpredictor.cli import main
from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, make_synthetic_dataset
from riskpredictor.scoring import build_demo_bundle, risk_level, score_dataframe, top_risky
from riskpredictor.scoring import test_period as get_test_period  # renamed so pytest doesn't collect it


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(n=800, seed=1)


@pytest.fixture(scope="module")
def bundle(data) -> dict:
    return build_demo_bundle(data)


def test_risk_level_bands():
    assert risk_level(0.60, threshold=0.4) == "High"
    assert risk_level(0.40, threshold=0.4) == "High"  # the threshold itself is flagged
    assert risk_level(0.25, threshold=0.4) == "Medium"
    assert risk_level(0.20, threshold=0.4) == "Medium"
    assert risk_level(0.19, threshold=0.4) == "Low"


def test_demo_bundle_has_the_shared_format(bundle):
    assert {"model_name", "pipeline", "features", "threshold", "background", "metrics",
            "train_end_date", "trained_on"} <= set(bundle)
    assert bundle["features"] == FEATURE_COLUMNS
    assert 0 < bundle["threshold"] < 1
    assert len(bundle["background"]) <= 200
    assert 0.5 < bundle["metrics"]["roc_auc"] <= 1  # synthetic data has a learnable signal


def test_score_dataframe_adds_risk_columns(bundle, data):
    scored = score_dataframe(bundle, data.head(50))
    assert scored["risk"].between(0, 1).all()
    assert set(scored["risk_level"]) <= {"Low", "Medium", "High"}
    assert len(scored) == 50 and "risk" not in data.columns  # input is not modified


def test_test_period_is_after_training(bundle, data):
    test = get_test_period(data, bundle)
    assert len(test) > 0
    assert test[DATE_COLUMN].min() > pd.Timestamp(bundle["train_end_date"])
    assert len(get_test_period(data)) == len(data) - int(len(data) * 0.8)  # fallback: newest 20 %


def test_top_risky_is_sorted(bundle, data):
    top = top_risky(bundle, data, n=5)
    assert len(top) == 5
    assert top["risk"].is_monotonic_decreasing


def test_cli_top(tmp_path, bundle, data, capsys):
    data_path, bundle_path = tmp_path / "dataset.csv", tmp_path / "model.joblib"
    data.to_csv(data_path, index=False)
    joblib.dump(bundle, bundle_path)
    main(["top", "--data", str(data_path), "--bundle", str(bundle_path), "-n", "3"])
    out = capsys.readouterr().out
    assert "Top 3 riskiest commits" in out
    assert out.count("%") >= 3


def test_cli_missing_model_exits_with_help(tmp_path):
    with pytest.raises(SystemExit, match="riskpredictor.train"):
        main(["top", "--bundle", str(tmp_path / "missing.joblib")])
