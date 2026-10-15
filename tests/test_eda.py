"""Smoke tests for the EDA / unsupervised-analysis module (small synthetic data, runs in seconds)."""

import numpy as np

from riskpredictor.eda import cliffs_delta, effect_magnitude, high_correlation_pairs, run_eda, spearman_correlation
from riskpredictor.features import make_synthetic_dataset


def test_cliffs_delta_known_values():
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0  # every x bigger than every y
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0  # identical groups
    # brute force check on random data
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 10, 50), rng.integers(0, 10, 70)
    brute = (np.sign(x[:, None] - y[None, :])).mean()
    assert np.isclose(cliffs_delta(x, y), brute)
    assert effect_magnitude(0.1) == "negligible" and effect_magnitude(-0.5) == "large"


def test_high_correlation_pairs_flags_duplicated_feature():
    df = make_synthetic_dataset(n=300, seed=1)
    pairs = high_correlation_pairs(spearman_correlation(df), threshold=0.8)
    assert set(pairs.columns) == {"feature_a", "feature_b", "rho"}
    assert (pairs["rho"].abs() > 0.8).all()


def test_run_eda_writes_figures_and_summary(tmp_path):
    df = make_synthetic_dataset(n=400, seed=0)
    results = run_eda(df, tmp_path, source="synthetic", max_tsne=300, max_kernel_pca=300, k_values=range(2, 5))

    expected = [
        "class_balance", "bug_rate_over_time", "feature_distributions", "correlation_heatmap",
        "pca_variance", "pca_scatter", "kernel_pca", "tsne", "kmeans_selection",
        "cluster_profiles", "clustering_comparison",
    ]
    for name in expected:
        path = tmp_path / "figures" / f"eda_{name}.png"
        assert path.exists() and path.stat().st_size > 1000, name

    summary = (tmp_path / "eda_summary.md").read_text()
    assert "Key numbers" in summary and "Commit archetypes" in summary
    assert len(results["feature_stats"]) == 14
    assert results["cluster_profiles"]["size"].sum() == len(df)
    assert 0 < results["bug_rate"] < 1
