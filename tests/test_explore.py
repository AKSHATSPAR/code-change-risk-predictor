"""Tests for the deeper exploratory analysis (small synthetic data, a few seconds)."""

import numpy as np
import pandas as pd
import pytest

from riskpredictor.explore import (
    SelfOrganizingMap,
    bayesian_comparison,
    contingency,
    data_types_table,
    em_impute,
    find_duplicates,
    intrinsic_dimension_mle,
    mark_missing,
    review_bandit,
    run_explore,
    size_bands,
)
from riskpredictor.features import make_synthetic_dataset


@pytest.fixture(scope="module")
def df():
    return make_synthetic_dataset(n=600, seed=3)


def test_em_beats_mean_on_correlated_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    data = np.column_stack([x + rng.normal(scale=0.2, size=500), x, rng.normal(size=500)])
    truth = data[:100, 0].copy()
    data[:100, 0] = np.nan
    filled, iterations = em_impute(data)
    em_error = np.sqrt(np.mean((filled[:100, 0] - truth) ** 2))
    mean_error = np.sqrt(np.mean((np.nanmean(data[:, 0]) - truth) ** 2))
    assert em_error < 0.5 * mean_error
    assert not np.isnan(filled).any() and iterations >= 1


def test_mark_missing_only_touches_commits_without_history(df):
    marked = mark_missing(df.assign(nuc=[0] * 10 + [5] * (len(df) - 10)))
    assert marked["age"].isna().sum() == 10
    assert marked["age"].iloc[10:].notna().all()


def test_size_bands_are_ordered():
    bands = size_bands(pd.Series([0, 3, 15, 50, 500]))
    assert list(bands.astype(str)) == ["0", "1-5", "6-20", "21-100", "100+"]
    assert bands.cat.ordered


def test_duplicates_found():
    small = make_synthetic_dataset(n=50, seed=1)
    doubled = pd.concat([small, small.iloc[:3].assign(commit_hash=lambda d: d["commit_hash"] + "x")])
    assert find_duplicates(doubled)["same_change_extra_copies"] == 3


def test_contingency_and_bayes(df):
    result = contingency(df, df["fix"].map({0: "other", 1: "fix"}), "type")
    assert result["table"][["clean", "buggy"]].to_numpy().sum() == len(df)
    assert 0 <= result["p"] <= 1
    skewed = pd.DataFrame({"buggy": [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80})
    mask = pd.Series([True] * 100 + [False] * 100)
    assert bayesian_comparison(skewed, mask, "a", "b")["p_a_higher"] > 0.99


def test_intrinsic_dimension_of_a_plane():
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(800, 2)) @ rng.normal(size=(2, 10))  # a 2-D sheet inside 10-D space
    assert 1.5 < intrinsic_dimension_mle(plane) < 2.6


def test_self_organizing_map_assigns_every_point():
    X = np.random.default_rng(0).normal(size=(200, 4))
    som = SelfOrganizingMap(rows=4, cols=4, iterations=500).fit(X)
    winners = som.winners(X)
    assert winners.shape == (200,) and winners.max() < 16


def test_bandit_learns_to_beat_random(df):
    result = review_bandit(df, rounds=800)
    assert result["curves"]["bandit"][-1] > result["curves"]["random"][-1]


def test_data_types_cover_every_column(df):
    table = data_types_table(df)
    assert set(df.columns) <= set(table["column"])
    assert set(table["scale"]) <= {"Nominal", "Ordinal", "Interval", "Ratio"}


def test_run_explore_end_to_end(df, tmp_path):
    result = run_explore(df, tmp_path, source="synthetic", bundle_path=None)
    assert len(result["figures"]) == 16
    assert all(path.exists() for path in result["figures"])
    summary = (tmp_path / "explore_summary.md").read_text()
    assert "Maximum likelihood (EM)" in summary and "contingency" not in summary.lower().split("##")[0]
