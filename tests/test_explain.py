"""Tests for the SHAP/LIME explanations, on two very different models (tree + linear).

The bundles are built here from synthetic data following the bundle contract, so these
tests do not depend on the real model having been trained.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from riskpredictor import explain
from riskpredictor.features import FEATURE_COLUMNS, LABEL_COLUMN, make_synthetic_dataset, time_split


def _bundle(df: pd.DataFrame, prep, clf, name: str) -> dict:
    train, _ = time_split(df)
    pipeline = Pipeline([("prep", prep), ("clf", clf)]).fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return {
        "model_name": name,
        "pipeline": pipeline,
        "features": list(FEATURE_COLUMNS),
        "threshold": 0.3,
        "background": train[FEATURE_COLUMNS].sample(200, random_state=0).reset_index(drop=True),
        "metrics": {},
        "train_end_date": str(train["author_date"].max()),
        "trained_on": "synthetic",
    }


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(n=1200, seed=1)


@pytest.fixture(scope="module")
def xgb_bundle(data):
    clf = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=0)
    return _bundle(data, SimpleImputer(strategy="median"), clf, "XGBoost")


@pytest.fixture(scope="module")
def logreg_bundle(data):
    prep = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    return _bundle(data, prep, LogisticRegression(max_iter=1000), "Logistic regression")


@pytest.fixture(params=["xgb", "logreg"])
def bundle(request, xgb_bundle, logreg_bundle):
    return {"xgb": xgb_bundle, "logreg": logreg_bundle}[request.param]


@pytest.fixture(scope="module")
def commits(data):
    return time_split(data)[1].head(5)


def test_shap_values_are_additive_and_sorted(bundle, commits):
    for _, row in commits.iterrows():
        expl = explain.explain_commit(bundle, row)
        total = expl.base_value + sum(c.shap_value for c in expl.contributions)
        assert total == pytest.approx(expl.risk, abs=0.02)
        assert expl.risk == pytest.approx(bundle["pipeline"].predict_proba(row[FEATURE_COLUMNS].to_frame().T.astype(float))[0, 1])
        assert len(expl.contributions) == len(FEATURE_COLUMNS)
        sizes = [abs(c.shap_value) for c in expl.contributions]
        assert sizes == sorted(sizes, reverse=True)
        assert expl.is_risky == (expl.risk >= bundle["threshold"])
        assert expl.risk_level in {"Low", "Medium", "High"}


def test_accepts_one_row_dataframe(xgb_bundle, commits):
    a = explain.explain_commit(xgb_bundle, commits.iloc[0])
    b = explain.explain_commit(xgb_bundle, commits.iloc[[0]])
    assert a.risk == pytest.approx(b.risk)


def test_tree_fast_path_matches_model_agnostic_shap(xgb_bundle, commits):
    X = commits[FEATURE_COLUMNS]
    tree_vals, tree_base = explain._tree_shap(xgb_bundle, X.reset_index(drop=True))
    perm_vals, perm_base = explain._shap_values(xgb_bundle, X, method="permutation")
    assert tree_base == pytest.approx(perm_base, abs=1e-6)
    assert np.corrcoef(tree_vals.ravel(), perm_vals.ravel())[0, 1] > 0.9


def test_reasons_and_text_report(bundle, commits):
    expl = explain.explain_commit(bundle, commits.iloc[0])
    reasons = expl.top_reasons(3)
    assert 1 <= len(reasons) <= 3
    assert all(r.endswith(".") and "percentage point" in r for r in reasons)
    text = expl.to_text()
    assert f"{expl.risk:.1%}" in text
    assert expl.risk_level in text


def test_bug_fix_sentence_is_short(xgb_bundle, commits):
    row = commits.iloc[0].copy()
    row["fix"] = 1.0
    expl = explain.explain_commit(xgb_bundle, row)
    fix = next(c for c in expl.contributions if c.feature == "fix")
    assert explain.describe_contribution(fix, xgb_bundle["background"]).startswith("It is a bug-fix commit")


def test_percentile_phrase():
    background = pd.DataFrame({"la": np.arange(100.0)})
    c = explain.Contribution("la", "Lines added", 95.0, 0.11, "lines of code added")
    sentence = explain.describe_contribution(c, background)
    assert sentence == "Adds 95 lines, more than 95% of past commits, which raises the risk by 11.0 percentage points."


def test_lime_returns_requested_number_of_features(bundle, commits):
    weights = explain.explain_commit_lime(bundle, commits.iloc[0], num_features=6)
    assert len(weights) == 6
    assert all(isinstance(cond, str) and isinstance(w, float) for cond, w in weights)


def test_global_importance(bundle, data):
    table = explain.global_importance(bundle, data, max_samples=40)
    assert list(table.columns) == ["feature", "label", "mean_abs_shap"]
    assert set(table["feature"]) == set(FEATURE_COLUMNS)
    assert table["mean_abs_shap"].is_monotonic_decreasing


def test_compare_shap_lime(xgb_bundle, commits):
    result = explain.compare_shap_lime(xgb_bundle, commits.head(3), k=3)
    assert len(result) == 3
    assert result["top_k_overlap"].between(0, 1).all()
    assert result["spearman"].between(-1, 1).all()


def test_plots_are_written(xgb_bundle, data, commits, tmp_path):
    expl = explain.explain_commit(xgb_bundle, commits.iloc[0])
    explain.plot_commit_explanation(expl, tmp_path / "commit.png")
    paths = explain.plot_global_importance(xgb_bundle, data, tmp_path, max_samples=40)
    explain.plot_lime_explanation(explain.explain_commit_lime(xgb_bundle, commits.iloc[0]), tmp_path / "lime.png")
    for path in [tmp_path / "commit.png", tmp_path / "lime.png", *paths]:
        assert path.exists() and path.stat().st_size > 1000
    assert {p.name for p in paths} == {"shap_summary.png", "shap_bar.png"}


def test_load_bundle_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        explain.load_bundle(tmp_path / "nope.joblib")
