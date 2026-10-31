"""Exploratory data analysis (EDA) and unsupervised learning on the commit dataset.

    uv run python -m riskpredictor.eda --data data/processed/dataset.csv --out reports
    uv run python -m riskpredictor.eda --synthetic --out /tmp/eda   # fake data, for a quick try

What it produces (in <out>/figures/eda_*.png plus <out>/eda_summary.md):

1. Class balance and the bug rate over time.
2. Each of the 14 features for buggy vs clean commits, with a Mann-Whitney U test and Cliff's delta.
3. A Spearman correlation heatmap (to spot features that carry the same information).
4. PCA, Kernel PCA and t-SNE: squash 14 dimensions into 2 so we can *look* at the data.
5. Clustering: K-Means (elbow + silhouette), hierarchical (Agglomerative) and density-based (DBSCAN),
   and "commit archetypes": what a typical commit in each K-Means cluster looks like and how often it is buggy.

Every unsupervised step uses the same input matrix: log(1 + x) of each feature (the counts are very
skewed, e.g. lines added ranges from 0 to 11,000), then standardised to mean 0 / std 1 so that no
feature dominates distances just because of its units.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA, KernelPCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from riskpredictor.features import (  # noqa: E402
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    LABEL_COLUMN,
    load_dataset,
    make_synthetic_dataset,
    time_split,
)

# ---------------------------------------------------------------------------------------------
# Plot style: one colour per meaning, the same as in the Streamlit app (colour-blind friendly).
# ---------------------------------------------------------------------------------------------
CLEAN_COLOR = "#2a78d6"  # blue
BUGGY_COLOR = "#eb6834"  # orange
NEUTRAL_COLOR = "#52514e"  # dark grey for reference lines / single-series bars
# Clusters skip blue/orange so a cluster is never confused with "clean"/"buggy".
CLUSTER_COLORS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300", "#e34948", "#52514e", "#2a78d6"]
NOISE_COLOR = "#b5b4ae"
# Diverging map for correlations: blue (negative) -> light grey (0) -> orange (positive)
CORR_CMAP = LinearSegmentedColormap.from_list("corr", [CLEAN_COLOR, "#f0efec", BUGGY_COLOR])
DPI = 150

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
    }
)


def _label(col: str) -> str:
    return FEATURE_INFO[col].label


def _long_label(col: str) -> str:
    """'Lines added (la)'; avoids 'Change spread (entropy) (entropy)'."""
    label = _label(col)
    return label if f"({col})" in label else f"{label} ({col})"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / f"eda_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------
# 0. Preprocessing shared by all unsupervised methods
# ---------------------------------------------------------------------------------------------
def prepare_matrix(df: pd.DataFrame) -> np.ndarray:
    """log(1 + x) to tame the long right tails, then standardise each feature to mean 0, std 1."""
    logged = np.log1p(df[FEATURE_COLUMNS].clip(lower=0).to_numpy(dtype=float))
    return StandardScaler().fit_transform(logged)


def _sample_index(n: int, max_n: int, random_state: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(random_state).choice(n, size=max_n, replace=False))


# ---------------------------------------------------------------------------------------------
# 1. Class balance and bug rate over time
# ---------------------------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(["Clean (0)", "Buggy (1)"], counts.values, color=[CLEAN_COLOR, BUGGY_COLOR], width=0.55)
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,}  ({n / len(df):.1%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_ylabel("Number of commits")
    ax.set_title("Class balance: clean vs bug-introducing commits")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "class_balance")


def _wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence interval for a proportion k/n (Wilson score interval; behaves well for small n)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def bug_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    year = df[DATE_COLUMN].dt.year.rename("year")
    table = df.groupby(year)[LABEL_COLUMN].agg(commits="size", buggy="sum").reset_index()
    table["bug_rate"] = table["buggy"] / table["commits"]
    table["ci_low"], table["ci_high"] = _wilson_interval(table["buggy"].to_numpy(), table["commits"].to_numpy())
    return table


def plot_bug_rate_over_time(df: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    table = bug_rate_by_year(df)
    overall = df[LABEL_COLUMN].mean()
    fig, (ax_n, ax_r) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    ax_n.bar(table["year"], table["commits"], color=NEUTRAL_COLOR, width=0.7)
    ax_n.set_ylabel("Code commits")
    ax_n.set_title("Activity and bug rate per year")
    ax_n.grid(axis="x", visible=False)

    yerr = [table["bug_rate"] - table["ci_low"], table["ci_high"] - table["bug_rate"]]
    ax_r.errorbar(table["year"], table["bug_rate"], yerr=yerr, color=BUGGY_COLOR, marker="o", ms=5,
                  lw=2, capsize=3, label="Bug rate (95% CI)")
    ax_r.axhline(overall, color=NEUTRAL_COLOR, ls="--", lw=1, label=f"Overall {overall:.1%}")
    ax_r.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_r.set_ylabel("Share of commits that are buggy")
    ax_r.set_xlabel("Year (author date)")
    ax_r.set_ylim(bottom=0)
    ax_r.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_r.legend(loc="upper right")
    return _save(fig, out_dir, "bug_rate_over_time"), table


# ---------------------------------------------------------------------------------------------
# 2. Buggy vs clean: distributions, Mann-Whitney U test and Cliff's delta
# ---------------------------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), a non-parametric effect size in [-1, 1].

    +1 means every x is bigger than every y, 0 means the groups overlap completely.
    Implemented with sorting + binary search instead of comparing all n*m pairs.
    """
    x, y = np.asarray(x, dtype=float), np.sort(np.asarray(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    y_smaller = np.searchsorted(y, x, side="left")  # for each x: how many y are < x
    y_bigger = len(y) - np.searchsorted(y, x, side="right")  # for each x: how many y are > x
    return float((y_smaller.sum() - y_bigger.sum()) / (len(x) * len(y)))


def effect_magnitude(delta: float) -> str:
    """Usual thresholds for |Cliff's delta| (Romano et al. 2006)."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def feature_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median of each feature for clean vs buggy commits, Mann-Whitney U p-value and Cliff's delta."""
    buggy = df[df[LABEL_COLUMN] == 1]
    clean = df[df[LABEL_COLUMN] == 0]
    rows = []
    for col in FEATURE_COLUMNS:
        b, c = buggy[col].to_numpy(), clean[col].to_numpy()
        same = np.all(np.concatenate([b, c]) == (b[0] if len(b) else 0))
        p = 1.0 if same or len(b) == 0 or len(c) == 0 else mannwhitneyu(b, c, alternative="two-sided").pvalue
        delta = cliffs_delta(b, c)
        rows.append(
            {
                "feature": col,
                "label": _label(col),
                "family": FEATURE_INFO[col].family,
                "median_clean": float(np.median(c)) if len(c) else np.nan,
                "median_buggy": float(np.median(b)) if len(b) else np.nan,
                "mean_clean": float(np.mean(c)) if len(c) else np.nan,
                "mean_buggy": float(np.mean(b)) if len(b) else np.nan,
                "mannwhitney_p": float(p),
                "cliffs_delta": delta,
                "effect": effect_magnitude(delta),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("cliffs_delta", key=np.abs, ascending=False).reset_index(drop=True)


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path, stats: pd.DataFrame) -> Path:
    """4 x 4 grid, one panel per feature, buggy vs clean.

    Most features get a box plot of log(1 + value). Features with only a few distinct values
    (e.g. `fix` is 0/1, `ns` is nearly always 1) make useless boxes, so they get a bar chart of
    the share of commits above the feature's minimum instead.
    """
    delta = stats.set_index("feature")["cliffs_delta"]
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    for ax, col in zip(axes.flat, FEATURE_COLUMNS):
        title = f"{_long_label(col)}\nCliff's δ = {delta[col]:+.2f}"
        if df[col].nunique() <= 5:
            low = df[col].min()
            shares = [(df.loc[df[LABEL_COLUMN] == k, col] > low).mean() for k in (0, 1)]
            ax.bar(["Clean", "Buggy"], shares, color=[CLEAN_COLOR, BUGGY_COLOR], width=0.55, alpha=0.85)
            for i, v in enumerate(shares):
                ax.annotate(f"{v:.0%}", (i, v), ha="center", va="bottom", fontsize=8,
                            xytext=(0, 2), textcoords="offset points")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_ylim(0, max(max(shares) * 1.25, 0.05))
            ax.set_ylabel(f"% of commits with {col} > {low:g}", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", visible=False)
            continue
        values = [np.log1p(df.loc[df[LABEL_COLUMN] == k, col].clip(lower=0)) for k in (0, 1)]
        box = ax.boxplot(values, tick_labels=["Clean", "Buggy"], widths=0.55, showfliers=False,
                         patch_artist=True, medianprops={"color": "black", "lw": 1.2})
        for patch, color in zip(box["boxes"], (CLEAN_COLOR, BUGGY_COLOR)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("log(1 + value)", fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(FEATURE_COLUMNS):]:
        ax.set_visible(False)
    fig.suptitle("Feature distributions: clean vs buggy commits (outliers hidden)", fontsize=12)
    fig.tight_layout()
    return _save(fig, out_dir, "feature_distributions")


# ---------------------------------------------------------------------------------------------
# 3. Correlation (multicollinearity)
# ---------------------------------------------------------------------------------------------
def spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].corr(method="spearman")


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    pairs = [
        {"feature_a": a, "feature_b": b, "rho": corr.loc[a, b]}
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]
        if abs(corr.loc[a, b]) > threshold
    ]
    out = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "rho"])
    return out.sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)


def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap=CORR_CMAP, vmin=-1, vmax=1)
    ticks = [_long_label(c) for c in corr.columns]
    ax.set_xticks(range(len(ticks)), ticks, rotation=60, ha="right")
    ax.set_yticks(range(len(ticks)), ticks)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.75 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Spearman correlation between the 14 change metrics")
    return _save(fig, out_dir, "correlation_heatmap")


# ---------------------------------------------------------------------------------------------
# 4. Dimensionality reduction: PCA, Kernel PCA, t-SNE
# ---------------------------------------------------------------------------------------------
def _scatter_by_label(ax: plt.Axes, xy: np.ndarray, y: np.ndarray) -> None:
    """Clean commits first, buggy on top so the minority class stays visible."""
    for k, color, name in ((0, CLEAN_COLOR, "Clean"), (1, BUGGY_COLOR, "Buggy")):
        m = y == k
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.45 if k == 0 else 0.7, color=color,
                   label=f"{name} (n={m.sum():,})", linewidths=0)
    ax.legend(loc="best", markerscale=2)


@dataclass
class PCAResult:
    explained_ratio: np.ndarray
    components_2d: np.ndarray
    loadings: pd.DataFrame  # feature x PC
    n_for_80: int  # components needed for 80% of the variance


def run_pca(X: np.ndarray) -> PCAResult:
    pca = PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLUMNS,
                            columns=[f"PC{i + 1}" for i in range(len(ratio))])
    return PCAResult(ratio, pca.transform(X)[:, :2], loadings, int(np.searchsorted(np.cumsum(ratio), 0.8) + 1))


def describe_component(loadings: pd.Series, top: int = 3) -> str:
    """Name a principal component by its largest loadings, e.g. '+exp, +rexp, +sexp'."""
    biggest = loadings.reindex(loadings.abs().sort_values(ascending=False).index)[:top]
    return ", ".join(f"{'+' if v > 0 else '-'}{k}" for k, v in biggest.items())


def plot_pca(result: PCAResult, y: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    n = len(result.explained_ratio)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(1, n + 1), result.explained_ratio, color=CLEAN_COLOR, width=0.7, label="Each component")
    ax.step(range(1, n + 1), np.cumsum(result.explained_ratio), where="mid", color=BUGGY_COLOR, lw=2,
            label="Cumulative")
    ax.axhline(0.8, color=NEUTRAL_COLOR, ls="--", lw=1, label="80% of variance")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title(f"PCA scree plot: {result.n_for_80} components keep 80% of the variance")
    ax.legend(loc="center right")
    ax.grid(axis="x", visible=False)
    scree = _save(fig, out_dir, "pca_variance")

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_by_label(ax, result.components_2d, y)
    r = result.explained_ratio
    ax.set_xlabel(f"PC1 ({r[0]:.0%}): {describe_component(result.loadings['PC1'])}")
    ax.set_ylabel(f"PC2 ({r[1]:.0%}): {describe_component(result.loadings['PC2'])}")
    ax.set_title("PCA: commits projected on the first two components")
    return scree, _save(fig, out_dir, "pca_scatter")


def run_kernel_pca(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """RBF Kernel PCA: like PCA but can follow curved structure. gamma = 1 / n_features (sklearn default)."""
    return KernelPCA(n_components=2, kernel="rbf", gamma=1.0 / X.shape[1], random_state=random_state).fit_transform(X)


def run_tsne(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    perplexity = min(30.0, max(5.0, (len(X) - 1) / 3.5))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state).fit_transform(X)


def plot_embedding(xy: np.ndarray, y: np.ndarray, out_dir: Path, name: str, title: str, axis: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_by_label(ax, xy, y)
    ax.set_xlabel(f"{axis} 1")
    ax.set_ylabel(f"{axis} 2")
    ax.set_title(title)
    return _save(fig, out_dir, name)


def separability_score(xy: np.ndarray, y: np.ndarray, k: int = 15) -> float:
    """Share of each buggy point's k nearest neighbours (in the 2-D map) that are also buggy.

    Compare with the overall bug rate: if it is about the same, buggy commits are not grouped together.
    """
    if y.sum() == 0 or len(y) <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy[y == 1])
    return float(y[idx[:, 1:]].mean())


# ---------------------------------------------------------------------------------------------
# 5. Clustering: K-Means, Agglomerative, DBSCAN
# ---------------------------------------------------------------------------------------------
def kmeans_scan(X: np.ndarray, k_values: range = range(2, 9), random_state: int = 42) -> pd.DataFrame:
    """Inertia (for the elbow plot) and silhouette score for each k."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(scan: pd.DataFrame, min_k: int = 3) -> int:
    """Best silhouette among k >= min_k. k = 2 nearly always 'wins' but just splits big vs small
    commits, which tells us little, so we require at least 3 archetypes."""
    candidates = scan[scan["k"] >= min_k]
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"]) if len(candidates) else int(scan["k"].iloc[0])


def plot_kmeans_scan(scan: pd.DataFrame, chosen_k: int, out_dir: Path) -> Path:
    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax_e.plot(scan["k"], scan["inertia"], marker="o", color=CLEAN_COLOR, lw=2)
    ax_e.set_title("Elbow method")
    ax_e.set_xlabel("Number of clusters k")
    ax_e.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_s.plot(scan["k"], scan["silhouette"], marker="o", color=BUGGY_COLOR, lw=2)
    ax_s.axvline(chosen_k, color=NEUTRAL_COLOR, ls="--", lw=1, label=f"chosen k = {chosen_k}")
    ax_s.set_title("Silhouette score (higher = better separated)")
    ax_s.set_xlabel("Number of clusters k")
    ax_s.set_ylabel("Mean silhouette")
    ax_s.legend()
    for ax in (ax_e, ax_s):
        ax.set_xticks(scan["k"])
    fig.suptitle("Choosing k for K-Means", fontsize=12)
    fig.tight_layout()
    return _save(fig, out_dir, "kmeans_selection")


def describe_cluster(z_means: pd.Series, top: int = 3, min_abs: float = 0.4) -> str:
    """Plain-English summary from the standardised cluster centre, e.g. 'high Author experience, low Files touched'."""
    strong = z_means[z_means.abs() >= min_abs]
    strong = strong.reindex(strong.abs().sort_values(ascending=False).index)[:top]
    if strong.empty:
        return "typical commit (close to the average on every feature)"
    return ", ".join(f"{'high' if v > 0 else 'low'} {_label(k).lower()}" for k, v in strong.items())


def cluster_profiles(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster: size, bug rate, a short description and the median of each feature."""
    z = pd.DataFrame(X, columns=FEATURE_COLUMNS).groupby(labels).mean()
    grouped = df.assign(cluster=labels).groupby("cluster")
    prof = grouped[FEATURE_COLUMNS].median()
    prof.insert(0, "description", [describe_cluster(z.loc[c]) for c in prof.index])
    prof.insert(0, "bug_rate", grouped[LABEL_COLUMN].mean())
    prof.insert(0, "share", grouped.size() / len(df))
    prof.insert(0, "size", grouped.size())
    return prof


def plot_cluster_profiles(profiles: pd.DataFrame, X: np.ndarray, labels: np.ndarray, overall_rate: float,
                          out_dir: Path) -> Path:
    """Left: heatmap of standardised cluster centres (what makes each cluster special).
    Right: bug rate per cluster vs the overall rate."""
    z = pd.DataFrame(X, columns=FEATURE_COLUMNS).groupby(labels).mean().loc[profiles.index]
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(12, 0.6 * len(profiles) + 2.6),
                                     gridspec_kw={"width_ratios": [3, 1.2]})
    lim = max(1.0, float(np.abs(z.to_numpy()).max()))
    im = ax_h.imshow(z.to_numpy(), cmap=CORR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax_h.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    ax_h.set_yticks(range(len(z)), [f"C{c} (n={profiles.loc[c, 'size']:,})" for c in z.index])
    ax_h.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax_h.text(j, i, f"{z.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax_h, shrink=0.9, label="Cluster mean (standard deviations from average)")
    ax_h.set_title("What each K-Means cluster looks like")

    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in profiles.index]
    ax_b.barh([f"C{c}" for c in profiles.index], profiles["bug_rate"], color=colors, height=0.6)
    ax_b.axvline(overall_rate, color=NEUTRAL_COLOR, ls="--", lw=1, label=f"Overall {overall_rate:.1%}")
    for i, r in enumerate(profiles["bug_rate"]):
        ax_b.annotate(f"{r:.1%}", (r, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax_b.invert_yaxis()
    ax_b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_b.set_xlim(0, max(profiles["bug_rate"].max() * 1.3, overall_rate * 1.3, 0.05))
    ax_b.set_title("Bug rate per cluster")
    ax_b.legend(loc="lower right")
    ax_b.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "cluster_profiles")


def dbscan_eps(X: np.ndarray, min_samples: int, quantile: float = 0.9) -> float:
    """Pick eps from the k-distance curve: the distance to the min_samples-th neighbour that 90% of
    points stay within. Points further out than that become 'noise' (unusual commits)."""
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return float(np.quantile(dist[:, -1], quantile))


@dataclass
class ClusteringComparison:
    table: pd.DataFrame
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    dbscan_eps: float = 0.0
    dbscan_noise_bug_rate: float = float("nan")
    dbscan_core_bug_rate: float = float("nan")


def compare_clustering(X: np.ndarray, y: np.ndarray, k: int, random_state: int = 42) -> ClusteringComparison:
    """Partitioning (K-Means) vs hierarchical (Agglomerative, Ward) vs density-based (DBSCAN)."""
    labels = {
        "K-Means": KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
    }
    min_samples = 2 * X.shape[1]  # common rule of thumb: 2 x number of features
    eps = dbscan_eps(X, min_samples)
    labels["DBSCAN"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    rows = []
    for name, lab in labels.items():
        clustered = lab != -1
        n_clusters = len(set(lab[clustered]))
        sil = silhouette_score(X[clustered], lab[clustered]) if n_clusters >= 2 else float("nan")
        rates = pd.Series(y[clustered]).groupby(lab[clustered]).mean()
        rows.append(
            {
                "method": name,
                "family": {"K-Means": "partitioning", "DBSCAN": "density-based"}.get(name, "hierarchical"),
                "n_clusters": n_clusters,
                "noise_share": float((~clustered).mean()),
                "silhouette": sil,
                "agreement_with_kmeans_ARI": adjusted_rand_score(labels["K-Means"], lab),
                "bug_rate_range": f"{rates.min():.1%} to {rates.max():.1%}" if len(rates) else "n/a",
            }
        )
    noise = labels["DBSCAN"] == -1
    return ClusteringComparison(
        table=pd.DataFrame(rows),
        labels=labels,
        dbscan_eps=eps,
        dbscan_noise_bug_rate=float(y[noise].mean()) if noise.any() else float("nan"),
        dbscan_core_bug_rate=float(y[~noise].mean()) if (~noise).any() else float("nan"),
    )


def plot_clustering_comparison(xy: np.ndarray, comparison: ClusteringComparison, out_dir: Path) -> Path:
    """The three clusterings drawn on the same 2-D PCA map."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, (name, lab) in zip(axes, comparison.labels.items()):
        row = comparison.table.set_index("method").loc[name]
        for c in sorted(set(lab)):
            m = lab == c
            color = NOISE_COLOR if c == -1 else CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, color=color, linewidths=0,
                       label="noise" if c == -1 else f"C{c}")
        sil = "n/a" if np.isnan(row["silhouette"]) else f"{row['silhouette']:.2f}"
        ax.set_title(f"{name}\n{row['n_clusters']} clusters, silhouette = {sil}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.legend(loc="best", markerscale=2, fontsize=7)
    axes[0].set_ylabel("PC2")
    fig.suptitle("Three families of clustering on the same data (shown on the PCA map)", fontsize=12)
    fig.tight_layout()
    return _save(fig, out_dir, "clustering_comparison")


# ---------------------------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def write_summary(results: dict, out_path: Path) -> None:
    r = results
    stats: pd.DataFrame = r["feature_stats"]
    top = stats.iloc[0]
    significant = int((stats["mannwhitney_p"] < 0.05).sum())
    by_year: pd.DataFrame = r["bug_rate_by_year"]
    profiles: pd.DataFrame = r["cluster_profiles"]
    riskiest, safest = profiles["bug_rate"].idxmax(), profiles["bug_rate"].idxmin()
    corr_pairs: pd.DataFrame = r["high_corr_pairs"]
    cmp_table: pd.DataFrame = r["clustering"].table

    stat_view = stats.assign(
        mannwhitney_p=stats["mannwhitney_p"].map(_p_text),
        cliffs_delta=stats["cliffs_delta"].map(lambda d: f"{d:+.3f}"),
    )[["feature", "label", "family", "median_clean", "median_buggy", "mannwhitney_p", "cliffs_delta", "effect"]]
    prof_view = profiles.reset_index()[["cluster", "size", "bug_rate", "description", "la", "ld", "nf", "exp", "nuc", "age"]]
    prof_view = prof_view.assign(bug_rate=prof_view["bug_rate"].map(lambda v: f"{v:.1%}"))
    corr_text = (
        "\n".join(f"- `{a}` and `{b}`: rho = {rho:+.2f}" for a, b, rho in corr_pairs.itertuples(index=False))
        if len(corr_pairs) else "- none above the threshold"
    )
    busiest = by_year.loc[by_year["commits"].idxmax()]

    text = f"""# EDA and unsupervised analysis: summary

Generated by `python -m riskpredictor.eda` from `{r['source']}` in {r['runtime_s']:.0f} s.
Figures are in `figures/eda_*.png`.

## Key numbers

- Commits: **{r['n_commits']:,}** ({r['first_date']} to {r['last_date']}); buggy: **{r['n_buggy']:,} ({r['bug_rate']:.1%})**, clean: {r['n_commits'] - r['n_buggy']:,}.
- Imbalance ratio clean : buggy = **{(r['n_commits'] - r['n_buggy']) / max(r['n_buggy'], 1):.1f} : 1**.
- Time-aware split (oldest 80% / newest 20%): bug rate **{r['train_bug_rate']:.1%}** in train vs **{r['test_bug_rate']:.1%}** in test.
- Features that differ significantly between buggy and clean commits (Mann-Whitney p < 0.05): **{significant} of {len(stats)}**.
- Strongest single signal: **{top['label']} (`{top['feature']}`)**, Cliff's delta = {top['cliffs_delta']:+.2f} ({top['effect']}).
- PCA: **{r['pca'].n_for_80} of 14** components are needed for 80% of the variance (PC1 alone: {r['pca'].explained_ratio[0]:.0%}).
- K-Means: chosen **k = {r['chosen_k']}** (silhouette {r['kmeans_scan'].set_index('k').loc[r['chosen_k'], 'silhouette']:.2f}); cluster bug rates range from **{profiles['bug_rate'].min():.1%} to {profiles['bug_rate'].max():.1%}**.

## Figures and one-line takeaways

| Figure | Takeaway |
|---|---|
| `eda_class_balance.png` | Only {r['bug_rate']:.1%} of commits are buggy, so accuracy is misleading; use PR-AUC / F1 / recall and class weights or SMOTE. |
| `eda_bug_rate_over_time.png` | Most commits are from {int(busiest['year'])} ({int(busiest['commits'])} commits); yearly bug rate ranges {by_year['bug_rate'].min():.0%}-{by_year['bug_rate'].max():.0%}, and later years have few commits (wide intervals), which is why we split by time. Train (oldest 80%) bug rate {r['train_bug_rate']:.1%} vs test (newest 20%) {r['test_bug_rate']:.1%}. |
| `eda_feature_distributions.png` | {significant}/{len(stats)} features differ significantly; the biggest effect is {top['label'].lower()} (delta {top['cliffs_delta']:+.2f}), but every box overlaps heavily, so no single metric separates the classes. |
| `eda_correlation_heatmap.png` | {len(corr_pairs)} feature pair(s) have abs(rho) > {r['corr_threshold']}; these measure nearly the same thing (see list below). |
| `eda_pca_variance.png` | Variance is spread out: {r['pca'].n_for_80} components for 80%, so the 14 metrics are not redundant overall. |
| `eda_pca_scatter.png` | PC1 = {describe_component(r['pca'].loadings['PC1'])}; PC2 = {describe_component(r['pca'].loadings['PC2'])}. Buggy and clean points overlap (buggy-neighbour share {r['separability']['PCA']:.0%} vs base rate {r['bug_rate']:.0%}). |
| `eda_kernel_pca.png` | RBF Kernel PCA bends the map but still does not separate buggy from clean commits (buggy-neighbour share {r['separability']['Kernel PCA']:.0%} vs base rate {r['bug_rate']:.0%}). |
| `eda_tsne.png` | t-SNE shows local groups of similar commits; buggy neighbours make up {r['separability']['t-SNE']:.0%} of a buggy point's neighbours vs {r['bug_rate']:.0%} by chance: weak, not clean, separation. |
| `eda_kmeans_selection.png` | Inertia falls smoothly (no sharp elbow); silhouette is highest at k = {int(r['kmeans_scan'].loc[r['kmeans_scan']['silhouette'].idxmax(), 'k'])} ({r['kmeans_scan']['silhouette'].max():.2f}), and among k >= 3 at k = {r['chosen_k']}, which we use. Silhouettes around 0.2-0.35 mean the clusters overlap: commits form a continuum, not crisp groups. |
| `eda_cluster_profiles.png` | Cluster C{riskiest} ({profiles.loc[riskiest, 'description']}) is buggy {profiles.loc[riskiest, 'bug_rate']:.1%} of the time vs {profiles.loc[safest, 'bug_rate']:.1%} for C{safest} ({profiles.loc[safest, 'description']}). |
| `eda_clustering_comparison.png` | K-Means, Ward and DBSCAN compared by silhouette (table below); DBSCAN noise points are buggy {r['clustering'].dbscan_noise_bug_rate:.1%} of the time vs {r['clustering'].dbscan_core_bug_rate:.1%} for points in dense regions. |

## Buggy vs clean: medians, Mann-Whitney U and Cliff's delta

Cliff's delta > 0 means buggy commits tend to have *larger* values. Magnitude thresholds: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.

{_md_table(stat_view)}

## Multicollinearity notes (Spearman abs(rho) > {r['corr_threshold']})

{corr_text}

## Commit archetypes (K-Means, k = {r['chosen_k']}, medians per cluster)

{_md_table(prof_view)}

## Clustering algorithms compared

DBSCAN: min_samples = 28 (2 x features), eps = {r['clustering'].dbscan_eps:.2f} (90th percentile of the 28-NN distance).

{_md_table(cmp_table)}
"""
    out_path.write_text(text)


# ---------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------
def run_eda(
    df: pd.DataFrame,
    out_dir: str | Path = "reports",
    source: str = "dataset",
    max_tsne: int = 3000,
    max_kernel_pca: int = 2000,
    k_values: range = range(2, 9),
    corr_threshold: float = 0.8,
    random_state: int = 42,
) -> dict:
    """Run the whole EDA, save every figure to <out_dir>/figures and a summary to <out_dir>/eda_summary.md."""
    start = time.time()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    y = df[LABEL_COLUMN].to_numpy()
    X = prepare_matrix(df)
    figures: list[Path] = []

    figures.append(plot_class_balance(df, fig_dir))
    path, by_year = plot_bug_rate_over_time(df, fig_dir)
    figures.append(path)

    stats = feature_comparison_table(df)
    figures.append(plot_feature_distributions(df, fig_dir, stats))

    corr = spearman_correlation(df)
    pairs = high_correlation_pairs(corr, corr_threshold)
    figures.append(plot_correlation_heatmap(corr, fig_dir))

    pca = run_pca(X)
    figures.extend(plot_pca(pca, y, fig_dir))

    idx = _sample_index(len(X), max_kernel_pca, random_state)
    kpca_xy = run_kernel_pca(X[idx], random_state)
    figures.append(plot_embedding(kpca_xy, y[idx], fig_dir, "kernel_pca",
                                  f"Kernel PCA (RBF), {len(idx):,} commits", "Kernel PC"))
    idx_t = _sample_index(len(X), max_tsne, random_state)
    tsne_xy = run_tsne(X[idx_t], random_state)
    figures.append(plot_embedding(tsne_xy, y[idx_t], fig_dir, "tsne", f"t-SNE, {len(idx_t):,} commits", "t-SNE"))
    separability = {
        "PCA": separability_score(pca.components_2d, y),
        "Kernel PCA": separability_score(kpca_xy, y[idx]),
        "t-SNE": separability_score(tsne_xy, y[idx_t]),
    }

    scan = kmeans_scan(X, k_values, random_state)
    k = choose_k(scan)
    figures.append(plot_kmeans_scan(scan, k, fig_dir))
    comparison = compare_clustering(X, y, k, random_state)
    profiles = cluster_profiles(df, X, comparison.labels["K-Means"])
    figures.append(plot_cluster_profiles(profiles, X, comparison.labels["K-Means"], float(y.mean()), fig_dir))
    figures.append(plot_clustering_comparison(pca.components_2d, comparison, fig_dir))

    stats.to_csv(out_dir / "eda_feature_tests.csv", index=False)
    profiles.to_csv(out_dir / "eda_cluster_profiles.csv")

    train, test = time_split(df)
    results = {
        "source": source,
        "train_bug_rate": float(train[LABEL_COLUMN].mean()),
        "test_bug_rate": float(test[LABEL_COLUMN].mean()),
        "n_commits": len(df),
        "n_buggy": int(y.sum()),
        "bug_rate": float(y.mean()),
        "first_date": str(df[DATE_COLUMN].min().date()),
        "last_date": str(df[DATE_COLUMN].max().date()),
        "bug_rate_by_year": by_year,
        "feature_stats": stats,
        "correlation": corr,
        "high_corr_pairs": pairs,
        "corr_threshold": corr_threshold,
        "pca": pca,
        "kernel_pca_2d": kpca_xy,
        "tsne_2d": tsne_xy,
        "separability": separability,
        "kmeans_scan": scan,
        "chosen_k": k,
        "cluster_profiles": profiles,
        "clustering": comparison,
        "figures": figures,
    }
    results["runtime_s"] = time.time() - start
    summary_path = out_dir / "eda_summary.md"
    write_summary(results, summary_path)
    results["summary_path"] = summary_path
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/dataset.csv", help="dataset CSV (features.ALL_COLUMNS)")
    parser.add_argument("--out", default="reports", help="output folder (figures go in <out>/figures)")
    parser.add_argument("--synthetic", action="store_true", help="use a generated fake dataset instead of --data")
    args = parser.parse_args(argv)

    if args.synthetic:
        df, source = make_synthetic_dataset(n=1500), "synthetic data"
    else:
        df, source = load_dataset(args.data), args.data
    results = run_eda(df, args.out, source=source)
    print(f"Wrote {len(results['figures'])} figures to {Path(args.out) / 'figures'} "
          f"and {results['summary_path']} in {results['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
