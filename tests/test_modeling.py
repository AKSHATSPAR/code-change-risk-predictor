"""Fast tests for riskpredictor.modeling and riskpredictor.train (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN, make_synthetic_dataset
from riskpredictor.modeling import (
    BUNDLE_KEYS,
    best_f1_threshold,
    evaluate,
    load_bundle,
    make_pipeline,
    make_preprocessor,
    model_zoo,
    predict_risk,
    recall_at_effort,
    to_sklearn_pipeline,
    train_val_test_split,
)
from riskpredictor.train import run_training


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(600, seed=1)


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    """One quick end-to-end training run, shared by the tests below."""
    root = tmp_path_factory.mktemp("run")
    result = run_training(data, root / "models", root / "reports", trained_on="synthetic", quick=True, verbose=False)
    return root, result


# --- preprocessing ------------------------------------------------------------------------------


def test_preprocessor_maps_14_columns_one_to_one(data):
    prep = make_preprocessor().fit(data[FEATURE_COLUMNS])
    out = prep.transform(data[FEATURE_COLUMNS])
    assert out.shape == (len(data), 14)
    assert list(prep.get_feature_names_out()) == FEATURE_COLUMNS
    assert np.allclose(out.mean(axis=0), 0, atol=1e-8)
    # untouched-by-log columns are just standardised: perfectly correlated with the raw values
    for col in ("entropy", "fix"):
        i = FEATURE_COLUMNS.index(col)
        assert np.corrcoef(out[:, i], data[col])[0, 1] == pytest.approx(1.0)
    # log-transformed columns keep their order (monotonic) but are no longer linear in the raw values
    i = FEATURE_COLUMNS.index("la")
    assert pd.Series(out[:, i]).corr(data["la"].reset_index(drop=True), method="spearman") == pytest.approx(1.0)


def test_preprocessor_accepts_arrays_and_ignores_extra_columns(data):
    prep = make_preprocessor().fit(data[FEATURE_COLUMNS])
    from_frame = prep.transform(data)  # extra columns (dates, labels, ...) are ignored
    from_array = prep.transform(data[FEATURE_COLUMNS].to_numpy())
    assert np.allclose(from_frame, from_array)


# --- no time leakage ----------------------------------------------------------------------------


def test_splits_are_time_ordered(data):
    train, fit, val, test = train_val_test_split(data)
    assert len(train) + len(test) == len(data)
    assert len(fit) + len(val) == len(train)
    assert fit[DATE_COLUMN].max() <= val[DATE_COLUMN].min()
    assert train[DATE_COLUMN].max() <= test[DATE_COLUMN].min()
    assert len(test) == pytest.approx(0.2 * len(data), abs=1)


def test_cv_folds_never_train_on_the_future(data):
    train, *_ = train_val_test_split(data)
    for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(train):
        assert train_idx.max() < val_idx.min()


def test_smote_is_only_used_during_training(data):
    X, y = data[FEATURE_COLUMNS], data[LABEL_COLUMN]
    zoo = model_zoo(balanced=False)
    fitted = make_pipeline(zoo["Logistic Regression"].estimator, smote=True).fit(X, y)
    saved = to_sklearn_pipeline(fitted)
    assert [name for name, _ in saved.steps] == ["prep", "clf"]
    assert saved.predict_proba(X).shape == (len(X), 2)  # no resampling at prediction time


# --- metrics ------------------------------------------------------------------------------------


def test_recall_at_effort_hand_example():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    effort = np.array([10, 10, 70, 10])  # 20% budget = 20 lines -> review the first two commits only
    assert recall_at_effort(y, scores, effort) == pytest.approx(0.5)
    assert recall_at_effort(y, np.array([0.1, 0.2, 0.3, 0.4]), effort) == pytest.approx(0.0)


def test_threshold_and_metrics_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.7
    metrics = evaluate(y, scores, threshold, effort=np.ones(5))
    assert metrics["roc_auc"] == metrics["f1"] == metrics["mcc"] == pytest.approx(1.0)


def test_metric_ranges_on_random_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    metrics = evaluate(y, rng.random(300), 0.5, effort=rng.integers(1, 100, 300))
    assert -1 <= metrics["mcc"] <= 1
    assert all(0 <= v <= 1 for k, v in metrics.items() if k != "mcc")
    assert 0.35 < metrics["roc_auc"] < 0.65  # random scores ~ chance


# --- end-to-end training and the saved bundle (contract used by explain.py and the app) --------


def test_bundle_contract(trained, data):
    root, _ = trained
    bundle = load_bundle(root / "models" / "best_model.joblib")
    assert set(BUNDLE_KEYS) <= set(bundle)
    assert bundle["features"] == FEATURE_COLUMNS
    assert bundle["trained_on"] == "synthetic"

    pipe = bundle["pipeline"]
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["prep", "clf"]
    assert pipe.named_steps["prep"].transform(data[FEATURE_COLUMNS]).shape == (len(data), 14)

    proba = predict_risk(bundle, data)
    assert proba.shape == (len(data),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba, pipe.predict_proba(data[FEATURE_COLUMNS])[:, 1])
    assert predict_risk(bundle, data.iloc[0]).shape == (1,)  # a single commit as a Series

    assert isinstance(bundle["threshold"], float) and 0 < bundle["threshold"] < 1
    assert list(bundle["background"].columns) == FEATURE_COLUMNS and len(bundle["background"]) <= 200
    assert {"roc_auc", "pr_auc", "f1", "recall_at_20_effort"} <= set(bundle["metrics"])


def test_bundle_has_no_time_leakage(trained, data):
    _, result = trained
    train, _, _, test = train_val_test_split(data)
    end = pd.Timestamp(result["bundle"]["train_end_date"])
    assert end == train[DATE_COLUMN].max()
    assert end <= test[DATE_COLUMN].min()


def test_best_model_is_chosen_by_cv_not_test(trained):
    _, result = trained
    lb = result["leaderboard"]
    eligible = lb[lb["selectable"]]
    assert result["best"] == eligible.loc[eligible["cv_roc_auc"].idxmax(), "model"]
    assert "One-Class SVM (anomaly baseline)" in set(lb["model"])  # reported, but never selectable
    assert len(lb) == 13


def test_reports_and_figures_are_written(trained):
    root, result = trained
    reports = root / "reports"
    for name in ("model_comparison.csv", "model_comparison.md", "imbalance_comparison.csv", "split_info.json"):
        assert (reports / name).exists(), name
    for name in ("roc", "pr", "confusion_matrix", "comparison", "complexity_curve", "learning_curve"):
        assert (reports / "figures" / f"model_{name}.png").exists(), name

    imbalance = pd.read_csv(reports / "imbalance_comparison.csv")
    assert set(imbalance["strategy"]) == {"none", "balanced", "smote"}
    assert set(imbalance["model"]) == {"Logistic Regression", "Random Forest", "XGBoost"}

    split = json.loads((reports / "split_info.json").read_text())
    assert split["train"]["n"] + split["test"]["n"] == split["n_commits"]
    assert split["train"]["end"] <= split["test"]["start"]
