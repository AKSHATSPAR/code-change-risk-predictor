"""A deeper exploratory pass over the commit dataset, going further than eda.py.

    uv run python -m riskpredictor.explore --data data/processed/dataset.csv --out reports
    uv run python -m riskpredictor.explore --synthetic --out /tmp/explore   # fake data, quick try

Figures go to <out>/figures/explore_*.png, tables to <out>/explore_*.csv and the findings to
<out>/explore_summary.md. The analysis is grouped into seven parts:

1. What kind of data we have: the data type and measurement scale of every column.
2. Preparing the data: duplicate commits, values that are really missing, three ways of filling
   them in (mean, median, and maximum likelihood with the EM algorithm), and binning commit size.
3. One, two and many variables at a time, and the commits as a time series (time-based indexing,
   resampling, rolling averages, grouping by year and weekday).
4. Summaries and tables: summary statistics, contingency tables with chi-square tests, a Bayesian
   take on the same questions, scatter plots, dot charts and bar plots.
5. More clustering: spectral clustering, Gaussian mixtures fitted with EM, a dendrogram, minimum
   spanning tree clustering, clustering the commit messages, and using clusters to find outliers.
6. More dimensionality reduction: SVD, factor analysis, intrinsic dimension, multidimensional
   scaling, Isomap and a small self-organizing map.
7. Models: a linear regression baseline (and why its accuracy looks better than it is), and a
   small reinforcement learning example, a reviewer that learns which commits to check by trial
   and feedback.

None of this changes the dataset or the trained model; it only looks at the data in more ways.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import beta, chi2_contingency, kurtosis, skew  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering  # noqa: E402
from sklearn.decomposition import FactorAnalysis, TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.manifold import MDS, Isomap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    adjusted_rand_score,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from riskpredictor.eda import (  # noqa: E402
    BUGGY_COLOR,
    CLEAN_COLOR,
    CLUSTER_COLORS,
    NEUTRAL_COLOR,
    _md_table,
    _p_text,
    _sample_index,
    _wilson_interval,
    dbscan_eps,
    prepare_matrix,
    separability_score,
)
from riskpredictor.features import (  # noqa: E402
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    LABEL_COLUMN,
    load_dataset,
    make_synthetic_dataset,
    time_split,
)

RANDOM_STATE = 42
SIZE_BINS = [-1, 0, 5, 20, 100, np.inf]
SIZE_LABELS = ["0", "1-5", "6-20", "21-100", "100+"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / f"explore_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _label(col: str) -> str:
    return FEATURE_INFO[col].label if col in FEATURE_INFO else col


# =============================================================================================
# 1. What kind of data we have
# =============================================================================================
# (data type, measurement scale, plain-English note) for every column, plus the derived ones.
COLUMN_TYPES: dict[str, tuple[str, str, str]] = {
    "repo": ("Categorical", "Nominal", "project name; a label with no order"),
    "commit_hash": ("Categorical (identifier)", "Nominal", "unique id of the commit"),
    "author_id": ("Categorical (identifier)", "Nominal", "anonymised author; only equal or not equal makes sense"),
    "author_date": ("Date-time", "Interval", "differences between dates are meaningful, but there is no true zero"),
    "message": ("Text", "Nominal", "free text, analysed separately as documents"),
    "ns": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "nd": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "nf": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "entropy": ("Numerical, continuous", "Ratio", "between 0 and 1; 0 means all changes in one file"),
    "la": ("Numerical, discrete", "Ratio", "a count; 200 lines really is twice 100 lines"),
    "ld": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "lt": ("Numerical, continuous", "Ratio", "an average of counts, so it can be fractional"),
    "fix": ("Categorical (binary)", "Nominal", "yes or no, stored as 1 or 0"),
    "ndev": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "age": ("Numerical, continuous", "Ratio", "days; can be fractional"),
    "nuc": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "exp": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "rexp": ("Numerical, continuous", "Ratio", "a weighted count, so it can be fractional"),
    "sexp": ("Numerical, discrete", "Ratio", "a count with a true zero"),
    "buggy": ("Categorical (binary)", "Nominal", "the target: 1 if the commit later caused a bug"),
    "size_band (derived)": ("Categorical", "Ordinal", "lines added grouped into bands; the order matters but gaps are uneven"),
    "weekday (derived)": ("Categorical", "Nominal", "day of the week the commit was made (UTC)"),
}


def data_types_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: data type, measurement scale, an example value and a short note."""
    rows = []
    for col, (kind, scale, note) in COLUMN_TYPES.items():
        if col == "size_band (derived)":
            example = size_bands(df["la"]).iloc[0]
        elif col == "weekday (derived)":
            example = df[DATE_COLUMN].iloc[0].day_name()
        elif col in df.columns:
            value = df[col].iloc[0]
            example = str(value)[:30] + ("..." if len(str(value)) > 30 else "")
        else:
            continue
        rows.append({"column": col, "data type": kind, "scale": scale, "example": str(example), "note": note})
    return pd.DataFrame(rows)


# =============================================================================================
# 2. Preparing the data: duplicates, missing values, binning
# =============================================================================================
def find_duplicates(df: pd.DataFrame) -> dict:
    """Exact duplicates (same commit id) and the same change saved twice.

    "The same change saved twice" means the same author, timestamp, message and size; this happens
    when a commit is copied onto another branch. Repeated messages alone (like "Fix typo") are
    normal and are *not* duplicates.
    """
    exact = int(df["commit_hash"].duplicated().sum())
    key = ["author_id", DATE_COLUMN, "message", "la", "ld", "nf"]
    same_change = df.duplicated(key, keep="first")
    repeated_message = int(df.duplicated(["message"], keep=False).sum())
    deduped = df[~same_change]
    return {
        "exact_duplicates": exact,
        "same_change_extra_copies": int(same_change.sum()),
        "rows_with_repeated_message": repeated_message,
        "rows_after_dedup": len(deduped),
        "bug_rate_before": float(df[LABEL_COLUMN].mean()),
        "bug_rate_after": float(deduped[LABEL_COLUMN].mean()),
        "examples": df[df.duplicated(key, keep=False)].sort_values(key)[["commit_hash", DATE_COLUMN, "message"]]
        .assign(commit_hash=lambda d: d["commit_hash"].str[:8]).head(6),
    }


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace values that really mean "unknown" with NaN.

    `age` (days since the touched files last changed) is stored as 0 when the files have no earlier
    history at all (nuc = 0). That 0 is not a real measurement, so here it becomes missing.
    """
    out = df.copy()
    out["age"] = out["age"].astype(float)
    out.loc[out["nuc"] == 0, "age"] = np.nan
    return out


def em_impute(X: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, int]:
    """Fill NaNs by maximum likelihood, assuming the columns follow a multivariate normal distribution.

    The EM algorithm repeats two steps until nothing changes:
    E-step: fill each missing value with its expected value given the observed values in the same
            row (a regression on the other columns, using the current mean and covariance);
    M-step: re-estimate the mean and covariance from the completed data, adding back the
            uncertainty of the filled-in values.
    """
    X = np.array(X, dtype=float)
    miss = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    X[miss] = np.take(mu, np.where(miss)[1])
    sigma = np.cov(X, rowvar=False, bias=True)
    rows = np.where(miss.any(axis=1))[0]
    for iteration in range(1, max_iter + 1):
        correction = np.zeros_like(sigma)
        for i in rows:
            m, o = miss[i], ~miss[i]
            coef = sigma[np.ix_(m, o)] @ np.linalg.pinv(sigma[np.ix_(o, o)])
            X[i, m] = mu[m] + coef @ (X[i, o] - mu[o])
            correction[np.ix_(m, m)] += sigma[np.ix_(m, m)] - coef @ sigma[np.ix_(o, m)]
        new_mu = X.mean(axis=0)
        new_sigma = np.cov(X, rowvar=False, bias=True) + correction / len(X)
        done = np.abs(new_mu - mu).max() < tol and np.abs(new_sigma - sigma).max() < tol
        mu, sigma = new_mu, new_sigma
        if done:
            break
    return X, iteration


IMPUTE_COLUMNS = ["exp", "rexp", "sexp", "nuc", "ndev", "la", "nf"]


def imputation_experiment(df: pd.DataFrame, column: str = "exp", share: float = 0.15,
                          random_state: int = RANDOM_STATE) -> dict:
    """Hide a random 15% of one column, fill the gaps three ways and see which guesses best.

    Everything is done on the log(1 + x) scale, where the counts look roughly normal. Because we hid
    the values ourselves, we know the true answers and can measure the error of each method.
    """
    cols = [column] + [c for c in IMPUTE_COLUMNS if c != column]
    logged = np.log1p(df[cols].clip(lower=0).to_numpy(dtype=float))
    rng = np.random.default_rng(random_state)
    hidden = rng.random(len(logged)) < share
    truth = logged[hidden, 0].copy()
    with_gaps = logged.copy()
    with_gaps[hidden, 0] = np.nan

    observed = with_gaps[~hidden, 0]
    guesses = {
        "Mean": np.full(hidden.sum(), observed.mean()),
        "Median": np.full(hidden.sum(), np.median(observed)),
    }
    filled, iterations = em_impute(with_gaps)
    guesses["Maximum likelihood (EM)"] = filled[hidden, 0]

    rows = []
    for method, guess in guesses.items():
        rmse = float(np.sqrt(np.mean((guess - truth) ** 2)))
        typical = float(np.median(np.abs(np.expm1(guess) - np.expm1(truth))))
        rows.append({"method": method, "rmse_log_scale": rmse, "median_abs_error": typical})
    return {
        "column": column, "hidden": int(hidden.sum()), "table": pd.DataFrame(rows),
        "truth": truth, "em_guess": guesses["Maximum likelihood (EM)"], "em_iterations": iterations,
    }


def impute_real_missing(df: pd.DataFrame) -> dict:
    """Fill the genuinely missing ages (commits on brand-new files) with each method."""
    marked = mark_missing(df)
    missing = marked["age"].isna()
    cols = ["age", "nuc", "ndev", "la", "nf", "exp"]
    logged = np.log1p(marked[cols].clip(lower=0).to_numpy(dtype=float))
    filled, _ = em_impute(logged)
    observed = marked.loc[~missing, "age"]
    return {
        "n_missing": int(missing.sum()),
        "share_missing": float(missing.mean()),
        "bug_rate_missing": float(marked.loc[missing, LABEL_COLUMN].mean()) if missing.any() else float("nan"),
        "mean_fill": float(observed.mean()),
        "median_fill": float(observed.median()),
        "em_fill_median": float(np.median(np.expm1(filled[missing.to_numpy(), 0]))) if missing.any() else float("nan"),
    }


def plot_imputation(experiment: dict, out_dir: Path) -> Path:
    table = experiment["table"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    colors = [NEUTRAL_COLOR, "#8a8983", BUGGY_COLOR]
    ax1.barh(table["method"], table["rmse_log_scale"], color=colors)
    for i, v in enumerate(table["rmse_log_scale"]):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Error on the hidden values (RMSE, log scale; lower is better)")
    ax1.set_title(f"Filling in {experiment['hidden']} hidden values of {_label(experiment['column']).lower()}")
    ax2.scatter(experiment["truth"], experiment["em_guess"], s=10, alpha=0.5, color=BUGGY_COLOR)
    lim = [0, max(experiment["truth"].max(), experiment["em_guess"].max()) * 1.05]
    ax2.plot(lim, lim, color=NEUTRAL_COLOR, lw=1, ls="--", label="perfect guess")
    ax2.set(xlim=lim, ylim=lim, xlabel="True value, log(1 + x)", ylabel="EM guess, log(1 + x)",
            title="EM uses the other columns, so its guesses follow the truth")
    ax2.legend(loc="upper left")
    return _save(fig, out_dir, "missing_imputation")


def size_bands(la: pd.Series) -> pd.Series:
    """Bin lines added into ordered bands: 0, 1-5, 6-20, 21-100, 100+."""
    return pd.cut(la, bins=SIZE_BINS, labels=SIZE_LABELS)


def binning_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bug rate per size band, next to equal-width and equal-frequency bins for comparison."""
    bands = size_bands(df["la"])
    table = df.groupby(bands, observed=False)[LABEL_COLUMN].agg(commits="size", buggy="sum")
    table["bug_rate"] = table["buggy"] / table["commits"].clip(lower=1)
    table.index.name = "lines added"
    return table.reset_index()


def compare_binning(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Equal-width bins put almost everything in the first bin; equal-frequency bins share commits evenly."""
    width = pd.cut(df["la"], bins=n_bins)
    freq = pd.qcut(df["la"].rank(method="first"), q=n_bins, labels=False)
    return pd.DataFrame({
        "method": ["Equal width", "Equal frequency", "Hand-picked bands"],
        "commits in the largest bin": [int(width.value_counts().max()), int(freq.value_counts().max()),
                                       int(size_bands(df["la"]).value_counts().max())],
        "commits in the smallest bin": [int(width.value_counts().min()), int(freq.value_counts().min()),
                                        int(size_bands(df["la"]).value_counts().min())],
    })


def plot_size_bands(table: pd.DataFrame, overall: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(table["lines added"].astype(str), table["bug_rate"] * 100, color=BUGGY_COLOR, width=0.6)
    for bar, n, rate in zip(bars, table["commits"], table["bug_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate * 100 + 1.2, f"{rate:.0%}\n({n:,} commits)",
                ha="center", fontsize=8)
    ax.axhline(overall * 100, color=NEUTRAL_COLOR, ls="--", lw=1, label=f"all commits ({overall:.0%})")
    ax.set(xlabel="Lines added (binned)", ylabel="Share of commits that are buggy (%)",
           title="Bigger commits are buggier: bug rate by size band", ylim=(0, max(table["bug_rate"]) * 100 + 14))
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "size_bands")


# =============================================================================================
# 3. One, two and many variables, and time series
# =============================================================================================
def plot_distribution_shape(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    """Univariate view: lines added is very skewed; log(1 + x) makes it far more symmetric."""
    raw = df["la"].astype(float)
    logged = np.log1p(raw)
    shape = {"skew_raw": float(skew(raw)), "skew_log": float(skew(logged))}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    ax1.hist(raw.clip(upper=raw.quantile(0.99)), bins=40, color=NEUTRAL_COLOR)
    ax1.set(title=f"Lines added as recorded (skewness {shape['skew_raw']:.1f})", xlabel="lines added (top 1% capped)",
            ylabel="commits")
    ax2.hist(logged, bins=40, color=CLEAN_COLOR)
    ax2.set(title=f"After log(1 + x) (skewness {shape['skew_log']:.1f})", xlabel="log(1 + lines added)")
    return _save(fig, out_dir, "distribution_shape"), shape


def plot_scatter_regression(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    """Bivariate view: lines added vs lines deleted, with a straight line fitted on the log scale."""
    x = np.log1p(df["la"].to_numpy(float)).reshape(-1, 1)
    y = np.log1p(df["ld"].to_numpy(float))
    model = LinearRegression().fit(x, y)
    fit = {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "r2": float(model.score(x, y))}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for value, color, name in [(0, CLEAN_COLOR, "Clean"), (1, BUGGY_COLOR, "Buggy")]:
        mask = df[LABEL_COLUMN].to_numpy() == value
        ax.scatter(x[mask, 0], y[mask], s=8, alpha=0.35, color=color, label=name, linewidths=0)
    grid = np.linspace(0, x.max(), 50).reshape(-1, 1)
    ax.plot(grid, model.predict(grid), color=NEUTRAL_COLOR, lw=2,
            label=f"linear fit: R² = {fit['r2']:.2f}")
    ax.set(xlabel="log(1 + lines added)", ylabel="log(1 + lines deleted)",
           title="Commits that add more also delete more")
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "scatter_regression"), fit


def plot_pairplot(df: pd.DataFrame, out_dir: Path, max_n: int = 1500,
                  random_state: int = RANDOM_STATE) -> Path:
    """Multivariate view: four key features against each other, coloured by the label."""
    cols = ["la", "nf", "nuc", "exp"]
    idx = _sample_index(len(df), max_n, random_state)
    data = np.log1p(df.iloc[idx][cols].clip(lower=0)).rename(columns={c: f"log {c}" for c in cols})
    data["label"] = np.where(df.iloc[idx][LABEL_COLUMN] == 1, "Buggy", "Clean")
    grid = sns.pairplot(data, hue="label", hue_order=["Clean", "Buggy"], palette=[CLEAN_COLOR, BUGGY_COLOR],
                        plot_kws={"s": 8, "alpha": 0.35, "linewidth": 0}, diag_kws={"common_norm": False},
                        height=1.9, corner=True)
    sns.move_legend(grid, "upper right", bbox_to_anchor=(0.92, 0.92), title=None)
    grid.figure.suptitle("Four features at once (log scale): no single view separates the classes", y=1.02)
    path = out_dir / "explore_pairplot.png"
    grid.figure.savefig(path)
    plt.close(grid.figure)
    return path


def time_series(df: pd.DataFrame) -> dict:
    """Time-based indexing, resampling, rolling averages and grouping."""
    ts = df.set_index(DATE_COLUMN).sort_index()[[LABEL_COLUMN]]
    monthly = ts[LABEL_COLUMN].resample("MS").agg(["count", "sum"])
    monthly.columns = ["commits", "buggy"]
    rolling = monthly.rolling(12, min_periods=6).sum()
    # only show the rate when the 12-month window holds at least 10 commits, otherwise it is noise
    monthly["rolling_bug_rate"] = rolling["buggy"] / rolling["commits"].where(rolling["commits"] >= 10)
    yearly = ts[LABEL_COLUMN].resample("YS").agg(["count", "mean"])
    yearly.columns = ["commits", "bug_rate"]
    yearly.index = yearly.index.year
    # time-based indexing: slice by date strings, like a dictionary of periods
    early, late = ts.loc[:"2014"], ts.loc["2015":]
    weekday = df.groupby(df[DATE_COLUMN].dt.day_name())[LABEL_COLUMN].agg(commits="size", buggy="sum")
    weekday = weekday.reindex([d for d in WEEKDAYS if d in weekday.index])
    weekday["bug_rate"] = weekday["buggy"] / weekday["commits"]
    return {
        "monthly": monthly, "yearly": yearly, "weekday": weekday,
        "months": len(monthly), "empty_months": int((monthly["commits"] == 0).sum()),
        "busiest_month": monthly["commits"].idxmax(), "busiest_month_commits": int(monthly["commits"].max()),
        "early_rate": float(early[LABEL_COLUMN].mean()) if len(early) else float("nan"),
        "late_rate": float(late[LABEL_COLUMN].mean()) if len(late) else float("nan"),
        "early_n": len(early), "late_n": len(late),
    }


def plot_time_series(series: dict, out_dir: Path) -> Path:
    monthly = series["monthly"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.bar(monthly.index, monthly["commits"], width=25, color="#b5b4ae")
    ax1.set(ylabel="commits per month", title="Commits resampled by month")
    ax2.plot(monthly.index, monthly["rolling_bug_rate"] * 100, color=BUGGY_COLOR, lw=2,
             label="bug rate, rolling 12 months")
    yearly = series["yearly"]
    ax2.scatter(pd.to_datetime(yearly.index.astype(str)) + pd.Timedelta(days=182), yearly["bug_rate"] * 100,
                color=NEUTRAL_COLOR, s=18, zorder=3, label="bug rate per calendar year")
    ax2.set(ylabel="buggy commits (%)", title="Bug rate over time, smoothed with a rolling window")
    ax2.legend(loc="upper right")
    return _save(fig, out_dir, "time_series")


# =============================================================================================
# 4. Summaries and tables
# =============================================================================================
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """1-D summary measures for every feature: centre, spread and shape."""
    rows = []
    for col in FEATURE_COLUMNS:
        x = df[col].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        rows.append({
            "feature": col, "mean": x.mean(), "median": x.median(), "std": x.std(), "min": x.min(),
            "max": x.max(), "IQR": q3 - q1, "skewness": skew(x), "kurtosis": kurtosis(x),
        })
    return pd.DataFrame(rows)


def contingency(df: pd.DataFrame, by: pd.Series, name: str) -> dict:
    """A contingency table of `by` against the label, with a chi-square test and Cramer's V."""
    table = pd.crosstab(by, df[LABEL_COLUMN])
    table = table.reindex(columns=[0, 1], fill_value=0)
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else float("nan")
    view = table.rename(columns={0: "clean", 1: "buggy"})
    view["bug rate"] = view["buggy"] / view.sum(axis=1)
    view.index.name = name
    return {"table": view.reset_index(), "chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": cramers_v}


def bayesian_comparison(df: pd.DataFrame, mask: pd.Series, name_a: str, name_b: str,
                        draws: int = 200_000, random_state: int = RANDOM_STATE) -> dict:
    """Bayesian version of "are group A commits buggier than group B commits?".

    With a flat Beta(1, 1) prior, the bug rate of each group has a Beta(1 + buggy, 1 + clean)
    posterior. Instead of a p-value we get a direct answer: the probability that A's rate is higher.
    """
    rng = np.random.default_rng(random_state)
    out = {}
    for key, part in (("a", df[mask]), ("b", df[~mask])):
        k, n = int(part[LABEL_COLUMN].sum()), len(part)
        post = beta(1 + k, 1 + n - k)
        out[key] = {"k": k, "n": n, "mean": float(post.mean()), "low": float(post.ppf(0.025)),
                    "high": float(post.ppf(0.975)), "draws": post.rvs(draws, random_state=rng)}
    out["p_a_higher"] = float(np.mean(out["a"]["draws"] > out["b"]["draws"]))
    out["names"] = (name_a, name_b)
    return out


def plot_bayesian(result: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    lo = min(result[k]["low"] for k in ("a", "b")) - 0.04
    hi = max(result[k]["high"] for k in ("a", "b")) + 0.04
    grid = np.linspace(max(lo, 0), min(hi, 1), 600)
    peak = 0.0
    for key, color in (("a", BUGGY_COLOR), ("b", CLEAN_COLOR)):
        part = result[key]
        density = beta(1 + part["k"], 1 + part["n"] - part["k"]).pdf(grid)
        name = result["names"][0 if key == "a" else 1]
        ax.plot(grid * 100, density, color=color, lw=2, label=f"{name}: {part['mean']:.1%} "
                f"(95% interval {part['low']:.1%} to {part['high']:.1%})")
        ax.fill_between(grid * 100, density, color=color, alpha=0.12)
        peak = max(peak, density.max())
    ax.set_ylim(0, peak * 1.45)  # room for the legend
    ax.set(xlabel="bug rate (%)", ylabel="how plausible (posterior density)",
           title=f"Bayesian view: P({result['names'][0]} riskier) = {result['p_a_higher']:.0%}")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "bayesian")


def plot_dot_charts(weekday: pd.DataFrame, bands: pd.DataFrame, overall: float, out_dir: Path) -> Path:
    """Dot charts with 95% intervals: one dot per group, easier to compare than bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, table, title in [(axes[0], weekday.reset_index(), "By weekday (UTC)"),
                             (axes[1], bands, "By lines added")]:
        names = table.iloc[:, 0].astype(str).tolist()
        k, n = table["buggy"].to_numpy(), table["commits"].to_numpy()
        low, high = _wilson_interval(k, np.clip(n, 1, None))  # an empty group would divide by zero
        rate = k / np.clip(n, 1, None)
        y = np.arange(len(names))
        ax.hlines(y, low * 100, high * 100, color="#b5b4ae", lw=2)
        ax.plot(rate * 100, y, "o", color=BUGGY_COLOR)
        ax.axvline(overall * 100, color=NEUTRAL_COLOR, ls="--", lw=1)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set(xlabel="buggy commits (%), with 95% interval", title=title)
    return _save(fig, out_dir, "dot_charts")


# =============================================================================================
# 5. More clustering
# =============================================================================================
def spectral_clusters(X: np.ndarray, y: np.ndarray, k: int = 3, max_n: int = 1500,
                      random_state: int = RANDOM_STATE) -> dict:
    """Spectral clustering on a nearest-neighbour graph, compared with K-Means on the same commits."""
    idx = _sample_index(len(X), max_n, random_state)
    Xs, ys = X[idx], y[idx]
    n_neighbors = max(5, min(15, len(Xs) - 1))
    spectral = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
                                  assign_labels="kmeans", random_state=random_state).fit_predict(Xs)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xs)
    rates = pd.Series(ys).groupby(spectral).mean()
    return {"labels": spectral, "idx": idx, "silhouette": float(silhouette_score(Xs, spectral)),
            "ari_vs_kmeans": float(adjusted_rand_score(kmeans, spectral)), "bug_rates": rates,
            "sizes": pd.Series(spectral).value_counts().sort_index()}


def gaussian_mixture(X: np.ndarray, y: np.ndarray, k_values=range(1, 13),
                     random_state: int = RANDOM_STATE) -> dict:
    """Model-based clustering: Gaussian mixtures fitted with the EM algorithm, k chosen by BIC."""
    bic = {}
    for k in k_values:
        gm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-3, random_state=random_state)
        bic[k] = gm.fit(X).bic(X)
    best_k = min(bic, key=bic.get)
    gm = GaussianMixture(n_components=best_k, covariance_type="full", reg_covar=1e-3,
                         random_state=random_state).fit(X)
    proba = gm.predict_proba(X)
    labels = proba.argmax(axis=1)
    rates = pd.Series(y).groupby(labels).mean()
    return {"bic": pd.Series(bic), "best_k": best_k, "labels": labels, "bug_rates": rates,
            "uncertain_share": float((proba.max(axis=1) < 0.8).mean()), "iterations": int(gm.n_iter_)}


def mst_clusters(X: np.ndarray, y: np.ndarray, max_n: int = 1200, random_state: int = RANDOM_STATE) -> dict:
    """Minimum spanning tree clustering: connect all commits with the shortest total edge length,
    then cut the unusually long edges (longer than mean + 3 standard deviations).

    The pieces that break off are small groups of unusual commits, so this doubles as outlier detection.
    """
    idx = _sample_index(len(X), max_n, random_state)
    Xs, ys = X[idx], y[idx]
    dist = squareform(pdist(Xs))
    tree = minimum_spanning_tree(dist).toarray()
    weights = tree[tree > 0]
    cutoff = weights.mean() + 3 * weights.std()
    tree[tree > cutoff] = 0
    n_parts, labels = connected_components(tree, directed=False)
    sizes = np.bincount(labels)
    small = sizes[labels] < 0.02 * len(Xs)  # pieces holding under 2% of commits
    return {"n_parts": int(n_parts), "largest": int(sizes.max()), "cut_edges": int((weights > cutoff).sum()),
            "outliers": int(small.sum()), "outlier_bug_rate": float(ys[small].mean()) if small.any() else float("nan"),
            "rest_bug_rate": float(ys[~small].mean())}


def clustering_outliers(X: np.ndarray, y: np.ndarray, k: int = 3, share: float = 0.05,
                        random_state: int = RANDOM_STATE) -> dict:
    """Outliers from clustering: the 5% of commits farthest from their K-Means centre, and DBSCAN noise."""
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    distance = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    far = distance >= np.quantile(distance, 1 - share)
    min_samples = 2 * X.shape[1]
    noise = DBSCAN(eps=dbscan_eps(X, min_samples), min_samples=min_samples).fit_predict(X) == -1
    return {"kmeans_far_rate": float(y[far].mean()), "kmeans_rest_rate": float(y[~far].mean()),
            "kmeans_far_n": int(far.sum()), "dbscan_noise_rate": float(y[noise].mean()) if noise.any() else float("nan"),
            "dbscan_rest_rate": float(y[~noise].mean()), "dbscan_noise_n": int(noise.sum())}


def plot_outliers(outliers: dict, mst: dict, overall: float, out_dir: Path) -> Path:
    names = ["Far from K-Means centre", "DBSCAN noise", "Cut off by the spanning tree"]
    unusual = [outliers["kmeans_far_rate"], outliers["dbscan_noise_rate"], mst["outlier_bug_rate"]]
    normal = [outliers["kmeans_rest_rate"], outliers["dbscan_rest_rate"], mst["rest_bug_rate"]]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.barh(y - 0.18, np.nan_to_num(unusual) * 100, height=0.36, color=BUGGY_COLOR, label="unusual commits")
    ax.barh(y + 0.18, np.array(normal) * 100, height=0.36, color=CLEAN_COLOR, label="everything else")
    ax.axvline(overall * 100, color=NEUTRAL_COLOR, ls="--", lw=1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel="buggy commits (%)", title="Three clustering methods agree: unusual commits are riskier")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "outliers")


def message_clusters(df: pd.DataFrame, k: int = 6, random_state: int = RANDOM_STATE) -> dict:
    """Document clustering: TF-IDF on the commit messages, then K-Means; SVD (LSA) for a 2-D map."""
    messages = df["message"].fillna("").str.lower()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=min(3, len(df)), token_pattern=r"[a-z]{3,}")
    tfidf = vectorizer.fit_transform(messages)
    k = max(2, min(k, messages.nunique(), tfidf.shape[1]))  # never more clusters than distinct messages
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(tfidf)
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for c in range(k):
        mask = km.labels_ == c
        top = terms[np.argsort(km.cluster_centers_[c])[::-1][:5]]
        rows.append({"cluster": f"M{c}", "top words": ", ".join(top), "commits": int(mask.sum()),
                     "bug rate": float(df.loc[mask, LABEL_COLUMN].mean())})
    table = pd.DataFrame(rows).sort_values("bug rate", ascending=False).reset_index(drop=True)
    n_comp = min(2, tfidf.shape[1] - 1) if tfidf.shape[1] > 2 else 1
    xy = TruncatedSVD(n_components=max(n_comp, 1), random_state=random_state).fit_transform(tfidf)
    if xy.shape[1] == 1:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    return {"table": table, "labels": km.labels_, "xy": xy, "vocabulary": len(terms)}


def plot_message_clusters(result: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    table = result["table"].set_index("cluster")
    for c in sorted(set(result["labels"])):
        mask = result["labels"] == c
        name = f"M{c}"
        words = ", ".join(table.loc[name, "top words"].split(", ")[:3])
        ax.scatter(result["xy"][mask, 0], result["xy"][mask, 1], s=8, alpha=0.5,
                   color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f"{name}: {words} ({table.loc[name, 'bug rate']:.0%} buggy)")
    ax.set(xlabel="SVD component 1", ylabel="SVD component 2",
           title="Commit messages clustered by the words they use")
    ax.legend(loc="best", fontsize=7, markerscale=2)
    return _save(fig, out_dir, "message_clusters")


def plot_more_clustering(X: np.ndarray, pca_xy: np.ndarray, spectral: dict, gmm: dict, out_dir: Path,
                         dendro_n: int = 400, random_state: int = RANDOM_STATE) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bic = gmm["bic"]
    axes[0].plot(bic.index, bic.values, "o-", color=NEUTRAL_COLOR)
    axes[0].plot(gmm["best_k"], bic[gmm["best_k"]], "o", color=BUGGY_COLOR, ms=10)
    axes[0].set(xlabel="number of mixture components", ylabel="BIC (lower is better)",
                title=f"Gaussian mixture (EM): best k = {gmm['best_k']}")
    idx = spectral["idx"]
    for c in sorted(set(spectral["labels"])):
        mask = spectral["labels"] == c
        axes[1].scatter(pca_xy[idx][mask, 0], pca_xy[idx][mask, 1], s=6, alpha=0.5,
                        color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                        label=f"S{c}: {spectral['bug_rates'][c]:.0%} buggy")
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Spectral clustering, shown on the PCA map")
    axes[1].legend(loc="best", markerscale=2)
    sample = X[_sample_index(len(X), dendro_n, random_state)]
    dendrogram(linkage(sample, method="ward"), ax=axes[2], truncate_mode="lastp", p=20, no_labels=True,
               color_threshold=None, above_threshold_color=NEUTRAL_COLOR)
    axes[2].set(title=f"Hierarchical clustering (Ward), {len(sample)} commits", ylabel="merge distance")
    axes[2].grid(False)
    return _save(fig, out_dir, "more_clustering")


# =============================================================================================
# 6. More dimensionality reduction
# =============================================================================================
def svd_spectrum(X: np.ndarray) -> dict:
    """Singular value decomposition of the standardised data. PCA is SVD under the hood:
    the variance explained by component i is s_i^2 / (n - 1)."""
    _, s, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    variance = s ** 2 / (len(X) - 1)
    share = variance / variance.sum()
    cumulative = np.cumsum(share)
    return {"singular_values": s, "share": share, "cumulative": cumulative,
            "n80": int(np.searchsorted(cumulative, 0.80) + 1), "n90": int(np.searchsorted(cumulative, 0.90) + 1),
            "n95": int(np.searchsorted(cumulative, 0.95) + 1)}


def intrinsic_dimension_mle(X: np.ndarray, k: int = 12) -> float:
    """Levina-Bickel maximum likelihood estimate of how many dimensions the data really lives in."""
    unique = np.unique(np.round(X, 6), axis=0)
    k = min(k, len(unique) - 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(unique).kneighbors(unique)
    dist = dist[:, 1:]  # drop the zero distance to itself
    dist = np.clip(dist, 1e-12, None)
    inverse = np.log(dist[:, -1:] / dist[:, :-1]).mean(axis=1)
    inverse = inverse[inverse > 0]
    return float(1.0 / inverse.mean())


def plot_intrinsic_dimension(svd: dict, mle_dim: float, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(np.arange(1, len(svd["singular_values"]) + 1), svd["singular_values"], color=NEUTRAL_COLOR)
    ax1.set(xlabel="component", ylabel="singular value", title="Singular values of the data (SVD)")
    ax2.plot(np.arange(1, len(svd["cumulative"]) + 1), svd["cumulative"] * 100, "o-", color=CLEAN_COLOR)
    for level, n in ((80, svd["n80"]), (90, svd["n90"]), (95, svd["n95"])):
        ax2.axhline(level, color="#b5b4ae", lw=1, ls="--")
        ax2.text(len(svd["cumulative"]) + 0.2, level, f"{level}%: {n} dims", va="center", fontsize=8)
    ax2.set(xlabel="number of components", ylabel="variance kept (%)",
            title=f"Intrinsic dimension: about {mle_dim:.1f} by the MLE estimate")
    return _save(fig, out_dir, "intrinsic_dimension")


def factor_analysis(X: np.ndarray, n_factors: int = 4, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Factor analysis with varimax rotation: a few hidden factors that explain the 14 features."""
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=random_state).fit(X)
    loadings = pd.DataFrame(fa.components_.T, index=FEATURE_COLUMNS,
                            columns=[f"F{i + 1}" for i in range(n_factors)])
    # name each factor after its strongest features, so the table reads on its own
    names = {}
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).index[:3]
        family = {"Diffusion": "spread", "Size": "size", "History": "file history",
                  "Experience": "experience", "Purpose": "purpose"}[FEATURE_INFO[top[0]].family]
        names[col] = f"{col} ({family}): " + ", ".join(top)
    loadings.attrs["names"] = names
    return loadings


def plot_factor_loadings(loadings: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap=sns.diverging_palette(250, 25, as_cmap=True), center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "loading"},
                yticklabels=[_label(c) if f"({c})" in _label(c) else f"{_label(c)} ({c})" for c in loadings.index])
    ax.set_title("Factor analysis (varimax): which features belong together")
    return _save(fig, out_dir, "factor_loadings")


class SelfOrganizingMap:
    """A small self-organizing map (Kohonen map) written with numpy.

    A grid of nodes, each holding a 14-feature "prototype" commit. For every training commit, the
    closest node (the best matching unit) and its neighbours on the grid move a little towards it.
    Over time neighbouring nodes end up holding similar commits, giving a 2-D map of the data.
    """

    def __init__(self, rows: int = 8, cols: int = 8, iterations: int = 6000, learning_rate: float = 0.5,
                 random_state: int = RANDOM_STATE):
        self.rows, self.cols, self.iterations = rows, cols, iterations
        self.learning_rate, self.rng = learning_rate, np.random.default_rng(random_state)

    def fit(self, X: np.ndarray) -> SelfOrganizingMap:
        n_nodes = self.rows * self.cols
        self.weights_ = X[self.rng.choice(len(X), n_nodes, replace=len(X) < n_nodes)].copy()
        grid = np.array([(r, c) for r in range(self.rows) for c in range(self.cols)], dtype=float)
        sigma0 = max(self.rows, self.cols) / 2
        for t in range(self.iterations):
            frac = t / self.iterations
            lr, sigma = self.learning_rate * (1 - frac), max(sigma0 * (1 - frac), 0.5)
            x = X[self.rng.integers(len(X))]
            bmu = np.argmin(((self.weights_ - x) ** 2).sum(axis=1))
            influence = np.exp(-((grid - grid[bmu]) ** 2).sum(axis=1) / (2 * sigma ** 2))
            self.weights_ += lr * influence[:, None] * (x - self.weights_)
        return self

    def winners(self, X: np.ndarray) -> np.ndarray:
        dist = ((X[:, None, :] - self.weights_[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1)


def nonlinear_maps(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE,
                   max_mds: int = 600, max_isomap: int = 1500) -> dict:
    """Multidimensional scaling, Isomap (manifold learning) and a self-organizing map."""
    idx_m = _sample_index(len(X), max_mds, random_state)
    mds = MDS(n_components=2, n_init=1, init="random", max_iter=300, random_state=random_state)
    mds_xy = mds.fit_transform(X[idx_m])
    original, embedded = pdist(X[idx_m]), pdist(mds_xy)
    stress1 = float(np.sqrt(((original - embedded) ** 2).sum() / (original ** 2).sum()))  # Kruskal stress-1
    idx_i = _sample_index(len(X), max_isomap, random_state)
    iso_xy = Isomap(n_neighbors=min(12, len(idx_i) - 1), n_components=2).fit_transform(X[idx_i])
    som = SelfOrganizingMap(random_state=random_state).fit(X)
    win = som.winners(X)
    hits = np.bincount(win, minlength=som.rows * som.cols)
    rates = np.bincount(win, weights=y, minlength=som.rows * som.cols) / np.clip(hits, 1, None)
    rates[hits < 5] = np.nan
    return {"mds_xy": mds_xy, "mds_idx": idx_m, "mds_stress": stress1,
            "iso_xy": iso_xy, "iso_idx": idx_i, "som": som, "som_hits": hits.reshape(som.rows, som.cols),
            "som_rates": rates.reshape(som.rows, som.cols),
            "separability": {"MDS": separability_score(mds_xy, y[idx_m]), "Isomap": separability_score(iso_xy, y[idx_i])}}


def plot_nonlinear_maps(maps: dict, y: np.ndarray, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, idx_key, title in [(axes[0], "mds_xy", "mds_idx", "Multidimensional scaling"),
                                    (axes[1], "iso_xy", "iso_idx", "Isomap (manifold learning)")]:
        xy, labels = maps[key], y[maps[idx_key]]
        for value, color, name in [(0, CLEAN_COLOR, "Clean"), (1, BUGGY_COLOR, "Buggy")]:
            mask = labels == value
            ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=0.45, color=color, label=name, linewidths=0)
        ax.set(title=f"{title}, {len(labels):,} commits", xticks=[], yticks=[])
        ax.legend(loc="best", markerscale=2)
    image = axes[2].imshow(maps["som_rates"] * 100, cmap=sns.color_palette("blend:#2a78d6,#f0efec,#eb6834", as_cmap=True),
                           vmin=0, vmax=60)
    axes[2].set(title="Self-organizing map: bug rate per node", xticks=[], yticks=[])
    axes[2].grid(False)
    fig.colorbar(image, ax=axes[2], fraction=0.046, label="buggy commits (%)")
    return _save(fig, out_dir, "nonlinear_maps")


# =============================================================================================
# 7. Models: a linear regression baseline and a reinforcement learning sketch
# =============================================================================================
def linear_regression_baseline(df: pd.DataFrame, bundle_path: Path | None = None) -> dict:
    """Fit an ordinary linear regression to the 0/1 label (a "linear probability model").

    It is trained on the oldest 80% of commits and evaluated on the newest 20%. The point is to see
    why accuracy alone is misleading: saying "clean" for every commit is already very accurate.
    """
    train, test = time_split(df)
    scaler = StandardScaler().fit(np.log1p(train[FEATURE_COLUMNS].clip(lower=0)))
    prep = lambda part: scaler.transform(np.log1p(part[FEATURE_COLUMNS].clip(lower=0)))  # noqa: E731
    model = LinearRegression().fit(prep(train), train[LABEL_COLUMN])
    pred = np.clip(model.predict(prep(test)), 0, 1)
    y = test[LABEL_COLUMN].to_numpy()
    rows = [{"model": "Always say clean", "accuracy": accuracy_score(y, np.zeros_like(y)), "recall": 0.0}]
    for cut in (0.5, 0.25):
        guess = (pred >= cut).astype(int)
        rows.append({"model": f"Linear regression, cut at {cut}", "accuracy": accuracy_score(y, guess),
                     "recall": recall_score(y, guess, zero_division=0)})
    if bundle_path is not None and Path(bundle_path).exists():
        bundle = joblib.load(bundle_path)
        proba = bundle["pipeline"].predict_proba(test[FEATURE_COLUMNS].astype(float))[:, 1]
        guess = (proba >= bundle["threshold"]).astype(int)
        rows.append({"model": f"{bundle['model_name']} (final model)", "accuracy": accuracy_score(y, guess),
                     "recall": recall_score(y, guess, zero_division=0)})
    coefs = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values(key=np.abs, ascending=False)
    return {"table": pd.DataFrame(rows), "r2": float(r2_score(y, pred)), "mse": float(mean_squared_error(y, pred)),
            "roc_auc": float(roc_auc_score(y, pred)) if 0 < y.sum() < len(y) else float("nan"),
            "coefficients": coefs, "test_bug_rate": float(y.mean())}


def plot_linear_regression(result: dict, out_dir: Path) -> Path:
    table = result["table"]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(table) + 1.6))
    ax.barh(y - 0.18, table["accuracy"] * 100, height=0.36, color=NEUTRAL_COLOR, label="accuracy")
    ax.barh(y + 0.18, table["recall"] * 100, height=0.36, color=BUGGY_COLOR, label="buggy commits caught (recall)")
    for yi, a, r in zip(y, table["accuracy"], table["recall"]):
        ax.text(a * 100 + 1, yi - 0.18, f"{a:.0%}", va="center", fontsize=8)
        ax.text(r * 100 + 1, yi + 0.18, f"{r:.0%}", va="center", fontsize=8)
    ax.set_yticks(y, table["model"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 110), xlabel="%", title="High accuracy can still mean catching almost no bugs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return _save(fig, out_dir, "linear_regression")


def review_bandit(df: pd.DataFrame, rounds: int = 1500, epsilon: float = 0.1,
                  random_state: int = RANDOM_STATE) -> dict:
    """A tiny reinforcement learning example: an epsilon-greedy multi-armed bandit.

    Each round, a reviewer picks one size band of commits ("arm"), reviews a random commit from it
    and gets a reward of 1 if that commit is buggy. It does not get the labels up front: it learns
    the value of each arm only from the rewards it receives, mostly picking the best arm so far
    (exploit) but sometimes trying another one (explore, with probability epsilon).
    """
    train, _ = time_split(df)
    bands = size_bands(train["la"])
    pools = {label: train.loc[bands == label, LABEL_COLUMN].to_numpy() for label in SIZE_LABELS}
    arms = [label for label in SIZE_LABELS if len(pools[label])]
    true_rates = {a: float(pools[a].mean()) for a in arms}
    rng = np.random.default_rng(random_state)

    def run(policy: str) -> tuple[np.ndarray, dict]:
        counts, values = {a: 0 for a in arms}, {a: 0.0 for a in arms}
        rewards = np.zeros(rounds)
        for t in range(rounds):
            if policy == "random" or (policy == "bandit" and (rng.random() < epsilon or t < len(arms))):
                arm = arms[t] if (policy == "bandit" and t < len(arms)) else arms[rng.integers(len(arms))]
            elif policy == "bandit":
                arm = max(values, key=values.get)
            else:  # oracle: knows the true rates from the start
                arm = max(true_rates, key=true_rates.get)
            reward = float(pools[arm][rng.integers(len(pools[arm]))])
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]  # running average
            rewards[t] = reward
        return np.cumsum(rewards), {"counts": counts, "values": values}

    curves, learned = {}, {}
    for policy in ("bandit", "random", "oracle"):
        curves[policy], learned[policy] = run(policy)
    best = max(true_rates, key=true_rates.get)
    return {"curves": curves, "learned": learned["bandit"], "true_rates": true_rates, "best_arm": best,
            "rounds": rounds, "epsilon": epsilon,
            "share_on_best": learned["bandit"]["counts"][best] / rounds}


def plot_bandit(result: dict, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    styles = {"oracle": (NEUTRAL_COLOR, "--", "knows the answer already"),
              "bandit": (BUGGY_COLOR, "-", "learning reviewer (epsilon-greedy)"),
              "random": (CLEAN_COLOR, ":", "picks at random")}
    for policy, (color, ls, name) in styles.items():
        ax1.plot(result["curves"][policy], color=color, ls=ls, lw=2, label=name)
    ax1.set(xlabel="commits reviewed", ylabel="bugs found so far", title="Learning from feedback finds more bugs")
    ax1.legend(loc="upper left")
    arms = list(result["true_rates"])
    x = np.arange(len(arms))
    ax2.bar(x - 0.18, [result["true_rates"][a] * 100 for a in arms], width=0.36, color="#b5b4ae", label="true bug rate")
    ax2.bar(x + 0.18, [result["learned"]["values"][a] * 100 for a in arms], width=0.36, color=BUGGY_COLOR,
            label="what the reviewer learned")
    ax2.set_xticks(x, arms)
    ax2.set(xlabel="lines added (arm)", ylabel="buggy commits (%)", title="Value of each arm after learning")
    ax2.legend(loc="upper left")
    return _save(fig, out_dir, "bandit")


# =============================================================================================
# Summary and entry point
# =============================================================================================
def write_summary(r: dict, path: Path) -> None:
    dup, miss, imp = r["duplicates"], r["real_missing"], r["imputation"]
    ts, lr, bandit = r["time_series"], r["linear_regression"], r["bandit"]
    bands = r["binning"]
    size_ct, fix_ct, wd_ct = r["contingency"]["size"], r["contingency"]["fix"], r["contingency"]["weekday"]
    bayes_fix, bayes_fri = r["bayes_fix"], r["bayes_friday"]
    spec, gmm, mst, outl = r["spectral"], r["gmm"], r["mst"], r["outliers"]
    msg, svd, maps, fa = r["messages"], r["svd"], r["maps"], r["factors"]
    imp_table = imp["table"].assign(rmse_log_scale=lambda d: d["rmse_log_scale"].map(lambda v: f"{v:.2f}"),
                                    median_abs_error=lambda d: d["median_abs_error"].map(lambda v: f"{v:.1f}"))
    band_view = bands.assign(bug_rate=bands["bug_rate"].map(lambda v: f"{v:.1%}"))
    msg_view = msg["table"].assign(**{"bug rate": msg["table"]["bug rate"].map(lambda v: f"{v:.1%}")})
    lr_view = lr["table"].assign(accuracy=lambda d: d["accuracy"].map(lambda v: f"{v:.1%}"),
                                 recall=lambda d: d["recall"].map(lambda v: f"{v:.1%}"))
    stats_view = r["summary_stats"].set_index("feature").round(2).add(0.0).reset_index()  # avoid "-0"
    fa_names = "; ".join(fa.attrs["names"].values())
    worst = bands.loc[bands["bug_rate"].idxmax()]
    text = f"""# Deeper exploratory analysis: summary

Generated by `python -m riskpredictor.explore` from `{r['source']}` in {r['runtime_s']:.0f} s.
Figures are in `figures/explore_*.png`. Nothing here changes the dataset or the trained model.

## 1. What kind of data we have

Every column has a data type and a measurement scale. Most features are counts (numerical, discrete,
ratio scale: zero means none, and 200 really is twice 100). The label and the bug-fix flag are binary
categories (nominal), the date is on an interval scale, and binning lines added creates an ordinal
variable.

{_md_table(r['data_types'][['column', 'data type', 'scale', 'note']])}

## 2. Preparing the data

**Duplicates.** {dup['exact_duplicates']} commits share an id. {dup['same_change_extra_copies']} commits are extra copies of the same
change (same author, time, message and size, usually copied onto another branch). Removing them would
leave {dup['rows_after_dedup']:,} commits and move the bug rate from {dup['bug_rate_before']:.2%} to {dup['bug_rate_after']:.2%}, so they don't
affect the results. Repeated messages on their own ({dup['rows_with_repeated_message']} commits, like "Fix typo") are normal and are kept.

**Replacing values and missing data.** "Days since the files were last changed" is stored as 0 when
the files have no earlier history. That 0 is not a real measurement, so it is replaced with a missing
value: {miss['n_missing']} commits ({miss['share_missing']:.1%}). Filling them in would use {miss['mean_fill']:.1f} days (mean),
{miss['median_fill']:.1f} days (median) or about {miss['em_fill_median']:.1f} days (maximum likelihood with EM, which also looks at
the other columns of each commit).

To compare the methods fairly, we hid {imp['hidden']} known values of {_label(imp['column']).lower()} at random and filled them
back in. EM converged in {imp['em_iterations']} iterations and was far closer to the truth, because experience is strongly
related to recent and subsystem experience:

{_md_table(imp_table)}

**Binning.** Lines added is binned into five ordered bands. The bug rate climbs steadily with size,
from {bands.loc[1, 'bug_rate']:.1%} for 1 to 5 lines to {worst['bug_rate']:.1%} for {worst['lines added']} lines. Commits that add nothing are
never labelled buggy, because the labelling method can only blame lines a commit added.

{_md_table(band_view)}

Equal-width bins would put almost every commit in the first bin, and equal-frequency bins split
them evenly but at odd cut points, which is why hand-picked bands are used:

{_md_table(r['binning_compare'])}

## 3. One, two and many variables, and time

- **One variable:** lines added has skewness {r['shape']['skew_raw']:.1f}; after log(1 + x) it is {r['shape']['skew_log']:.1f}. That is why
  the models and the clustering use the log scale.
- **Two variables:** a straight line fitted to log(lines deleted) against log(lines added) has slope
  {r['scatter']['slope']:.2f} and R² = {r['scatter']['r2']:.2f}: commits that add more also tend to delete more.
- **Many variables:** the pair plot shows four features at once. Buggy commits sit towards larger values,
  but no pair of features separates them cleanly.
- **Time series:** resampled by month there are {ts['months']} months, {ts['empty_months']} of them with no commits. The busiest month
  was {ts['busiest_month']:%B %Y} ({ts['busiest_month_commits']} commits). Indexing by date shows the drift directly: commits up to
  2014 were buggy {ts['early_rate']:.1%} of the time ({ts['early_n']:,} commits), commits from 2015 on {ts['late_rate']:.1%} ({ts['late_n']:,} commits).
  The dip at the very end is mostly label lag: bugs in the newest commits simply haven't been found yet.

## 4. Summaries and tables

Summary measures for each feature (1-D analysis):

{_md_table(stats_view, floatfmt='.4g')}

Contingency tables against the label (2-D analysis), with chi-square tests:

| Question | Chi-square | p-value | Cramer's V | Verdict |
|---|---|---|---|---|
| Does size band matter? | {size_ct['chi2']:.1f} | {_p_text(size_ct['p'])} | {size_ct['cramers_v']:.2f} | {'yes, clearly' if size_ct['p'] < 0.05 else 'no clear evidence'} |
| Are bug-fix commits buggier? | {fix_ct['chi2']:.1f} | {_p_text(fix_ct['p'])} | {fix_ct['cramers_v']:.2f} | {'yes' if fix_ct['p'] < 0.05 else 'not significant at 5%'} |
| Does the weekday matter? | {wd_ct['chi2']:.1f} | {_p_text(wd_ct['p'])} | {wd_ct['cramers_v']:.2f} | {'yes' if wd_ct['p'] < 0.05 else 'not significant at 5%'} |

**Classical vs Bayesian.** The chi-square test only says whether a difference is surprising under
"no difference". The Bayesian view (flat prior, Beta posterior) answers the question directly:

- Bug-fix commits are buggy {bayes_fix['a']['mean']:.1%} of the time vs {bayes_fix['b']['mean']:.1%}; the probability that fixes really are riskier is **{bayes_fix['p_a_higher']:.0%}**.
- Friday commits: {bayes_fri['a']['mean']:.1%} vs {bayes_fri['b']['mean']:.1%} on other days; probability Friday is riskier: **{bayes_fri['p_a_higher']:.0%}**.

So the evidence leans one way, but is not strong enough to pass a 5% test. Exploratory analysis sits
before both: it looks at the data first and suggests which questions are worth testing.

## 5. More clustering

- **Spectral clustering** (on a nearest-neighbour graph of {len(spec['idx']):,} commits): silhouette {spec['silhouette']:.2f}, agreement with
  K-Means ARI = {spec['ari_vs_kmeans']:.2f}, cluster bug rates {', '.join(f'{v:.0%}' for v in spec['bug_rates'])}.
- **Gaussian mixture fitted with EM** (model-based clustering): trying 1 to {gmm['bic'].index.max()} components, BIC is lowest at
  k = {gmm['best_k']}. Their bug rates range from {gmm['bug_rates'].min():.0%} to {gmm['bug_rates'].max():.0%}. Only {gmm['uncertain_share']:.0%} of commits are split between
  components (no component holds 80% of their probability).
- **Hierarchical clustering:** the Ward dendrogram shows two or three big branches merging late.
- **Minimum spanning tree:** cutting the {mst['cut_edges']} unusually long edges breaks the tree into {mst['n_parts']} pieces: one main
  group of {mst['largest']:,} commits, a few medium groups, and {mst['outliers']} commits left in tiny pieces (the outliers).
- **Outliers found by clustering:** the 5% of commits farthest from their K-Means centre are buggy {outl['kmeans_far_rate']:.0%} of the
  time vs {outl['kmeans_rest_rate']:.0%} for the rest; DBSCAN noise points {outl['dbscan_noise_rate']:.0%} vs {outl['dbscan_rest_rate']:.0%}; spanning-tree outliers
  {mst['outlier_bug_rate']:.0%} vs {mst['rest_bug_rate']:.0%}. All three agree that unusual commits are riskier.
- **Document clustering** of the commit messages (TF-IDF over {msg['vocabulary']} words, then K-Means):

{_md_table(msg_view)}

## 6. More dimensionality reduction

- **SVD:** PCA is SVD of the centred data. Keeping 80% of the variance needs {svd['n80']} components, 90% needs {svd['n90']}
  and 95% needs {svd['n95']}.
- **Intrinsic dimension:** the maximum likelihood estimate (Levina and Bickel) is about **{r['mle_dim']:.1f}**: the 14
  features really live in a space of roughly that many dimensions.
- **Factor analysis** (4 factors, varimax): {fa_names}.
- **Multidimensional scaling** (Kruskal stress {maps['mds_stress']:.2f}, where 0 is a perfect fit) and **Isomap** keep buggy and clean commits mixed:
  buggy-neighbour share {maps['separability']['MDS']:.0%} and {maps['separability']['Isomap']:.0%} against a base rate of {r['bug_rate']:.0%}.
- **Self-organizing map** (8 by 8 nodes): nodes range from {np.nanmin(maps['som_rates']):.0%} to {np.nanmax(maps['som_rates']):.0%} buggy, so
  some regions of the map are clearly riskier than others.

## 7. Models

**Linear regression baseline.** A plain linear regression on the 0/1 label, trained on the oldest 80%
and tested on the newest 20% (R² = {lr['r2']:.2f}, ROC-AUC = {lr['roc_auc']:.3f}). Only {lr['test_bug_rate']:.1%} of test commits are
buggy, so a model that says "clean" every time is already {1 - lr['test_bug_rate']:.1%} accurate while catching nothing:

{_md_table(lr_view)}

That is why the main results use ROC-AUC, PR-AUC, recall and MCC rather than accuracy.

**Supervised vs reinforcement learning.** Our models are supervised: they learn from past commits
whose answers we already know. A reinforcement learner is never shown the answers; it acts, gets a
reward, and adjusts. In a small simulation, a reviewer picks a size band each round, reviews a
random commit from it and is rewarded when it finds a bug. With epsilon = {bandit['epsilon']} it ends up spending
{bandit['share_on_best']:.0%} of its {bandit['rounds']:,} reviews on the riskiest band ({bandit['best_arm']} lines) and finds
{bandit['curves']['bandit'][-1]:.0f} bugs, against {bandit['curves']['random'][-1]:.0f} for random picking and {bandit['curves']['oracle'][-1]:.0f} for a reviewer who knew the answer
from the start. Real uses would include a review bot that learns from reviewers' feedback, or test
scheduling that learns which tests fail most.
"""
    path.write_text(text)


def run_explore(df: pd.DataFrame, out_dir: str | Path = "reports", source: str = "dataset",
                bundle_path: str | Path | None = "models/best_model.joblib",
                random_state: int = RANDOM_STATE) -> dict:
    """Run the deeper exploration and write figures, tables and a summary."""
    start = time.time()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    y = df[LABEL_COLUMN].to_numpy()
    X = prepare_matrix(df)
    overall = float(y.mean())
    figures: list[Path] = []
    r: dict = {"source": source, "bug_rate": overall}

    # 1. data types
    r["data_types"] = data_types_table(df)
    # 2. preparation
    r["duplicates"] = find_duplicates(df)
    r["real_missing"] = impute_real_missing(df)
    r["imputation"] = imputation_experiment(df, random_state=random_state)
    figures.append(plot_imputation(r["imputation"], fig_dir))
    r["binning"] = binning_table(df)
    r["binning_compare"] = compare_binning(df)
    figures.append(plot_size_bands(r["binning"], overall, fig_dir))
    # 3. one, two, many variables and time
    path, r["shape"] = plot_distribution_shape(df, fig_dir)
    figures.append(path)
    path, r["scatter"] = plot_scatter_regression(df, fig_dir)
    figures.append(path)
    figures.append(plot_pairplot(df, fig_dir, random_state=random_state))
    r["time_series"] = time_series(df)
    figures.append(plot_time_series(r["time_series"], fig_dir))
    # 4. summaries and tables
    r["summary_stats"] = summary_statistics(df)
    r["contingency"] = {
        "size": contingency(df, size_bands(df["la"]), "lines added"),
        "fix": contingency(df, df["fix"].map({0: "other", 1: "bug fix"}), "commit type"),
        "weekday": contingency(df, df[DATE_COLUMN].dt.day_name(), "weekday"),
    }
    r["bayes_fix"] = bayesian_comparison(df, df["fix"] == 1, "bug-fix commits", "other commits",
                                         random_state=random_state)
    r["bayes_friday"] = bayesian_comparison(df, df[DATE_COLUMN].dt.day_name() == "Friday", "Friday commits",
                                            "other days", random_state=random_state)
    figures.append(plot_bayesian(r["bayes_fix"], fig_dir))
    figures.append(plot_dot_charts(r["time_series"]["weekday"], r["binning"], overall, fig_dir))
    # 5. clustering
    pca_xy = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(X - X.mean(axis=0))
    r["spectral"] = spectral_clusters(X, y, random_state=random_state)
    r["gmm"] = gaussian_mixture(X, y, random_state=random_state)
    figures.append(plot_more_clustering(X, pca_xy, r["spectral"], r["gmm"], fig_dir, random_state=random_state))
    r["mst"] = mst_clusters(X, y, random_state=random_state)
    r["outliers"] = clustering_outliers(X, y, random_state=random_state)
    figures.append(plot_outliers(r["outliers"], r["mst"], overall, fig_dir))
    r["messages"] = message_clusters(df, random_state=random_state)
    figures.append(plot_message_clusters(r["messages"], fig_dir))
    # 6. dimensionality reduction
    r["svd"] = svd_spectrum(X)
    r["mle_dim"] = intrinsic_dimension_mle(X)
    figures.append(plot_intrinsic_dimension(r["svd"], r["mle_dim"], fig_dir))
    r["factors"] = factor_analysis(X, random_state=random_state)
    figures.append(plot_factor_loadings(r["factors"], fig_dir))
    r["maps"] = nonlinear_maps(X, y, random_state=random_state)
    figures.append(plot_nonlinear_maps(r["maps"], y, fig_dir))
    # 7. models
    r["linear_regression"] = linear_regression_baseline(df, Path(bundle_path) if bundle_path else None)
    figures.append(plot_linear_regression(r["linear_regression"], fig_dir))
    r["bandit"] = review_bandit(df, random_state=random_state)
    figures.append(plot_bandit(r["bandit"], fig_dir))

    # tables for other tools (the app reads some of these)
    r["data_types"].to_csv(out_dir / "explore_data_types.csv", index=False)
    r["summary_stats"].to_csv(out_dir / "explore_summary_stats.csv", index=False)
    r["binning"].to_csv(out_dir / "explore_size_bands.csv", index=False)
    r["imputation"]["table"].to_csv(out_dir / "explore_imputation.csv", index=False)
    r["messages"]["table"].to_csv(out_dir / "explore_message_clusters.csv", index=False)
    r["factors"].to_csv(out_dir / "explore_factor_loadings.csv")
    r["linear_regression"]["table"].to_csv(out_dir / "explore_linear_regression.csv", index=False)
    for name, result in r["contingency"].items():
        result["table"].to_csv(out_dir / f"explore_contingency_{name}.csv", index=False)

    r["figures"] = figures
    r["runtime_s"] = time.time() - start
    r["summary_path"] = out_dir / "explore_summary.md"
    write_summary(r, r["summary_path"])
    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/dataset.csv", help="dataset CSV")
    parser.add_argument("--out", default="reports", help="output folder (figures go in <out>/figures)")
    parser.add_argument("--bundle", default="models/best_model.joblib", help="trained model, for the accuracy comparison")
    parser.add_argument("--synthetic", action="store_true", help="use a generated fake dataset instead of --data")
    args = parser.parse_args(argv)
    if args.synthetic:
        df, source = make_synthetic_dataset(n=1200), "synthetic data"
    else:
        df, source = load_dataset(args.data), args.data
    r = run_explore(df, args.out, source=source, bundle_path=None if args.synthetic else args.bundle)
    print(f"Wrote {len(r['figures'])} figures to {Path(args.out) / 'figures'} and {r['summary_path']} "
          f"in {r['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
