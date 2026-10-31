"""Explain *why* the model thinks a commit is risky.

Two explanation methods are used:

* **SHAP** (SHapley Additive exPlanations) - the main method. For one commit it splits the
  predicted risk into one contribution per feature, so that

      base_value + sum(contributions) = predicted risk

  where ``base_value`` is the average risk of the training commits. Everything is on the
  probability scale, so "+0.11" means "raises the risk by 11 percentage points".
* **LIME** (Local Interpretable Model-agnostic Explanations) - a second, independent method.
  It fits a small linear model around the commit. We compare it with SHAP to check that the
  explanations are stable (``compare_shap_lime``).

Both methods only call ``pipeline.predict_proba``, so they work for *any* classifier the
modeling step picks (XGBoost, random forest, logistic regression, SVM, KNN, ...).
For tree models we use SHAP's exact and much faster ``TreeExplainer`` instead.

Command line:
    uv run python -m riskpredictor.explain --commit 0477018            # one commit
    uv run python -m riskpredictor.explain --global                    # global plots + SHAP vs LIME
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from riskpredictor.features import FEATURE_COLUMNS, FEATURE_INFO, load_dataset, time_split  # noqa: E402
from riskpredictor.scoring import risk_level  # noqa: E402

DEFAULT_BUNDLE = Path("models/best_model.joblib")
FIGURES_DIR = Path("reports/figures")

RAISES_COLOR = "#d6453d"  # red: pushes the risk up
LOWERS_COLOR = "#2a78d6"  # blue: pushes the risk down
TEXT_COLOR = "#2b2b29"

# Model classes that SHAP's TreeExplainer can explain exactly on the probability scale.
TREE_MODELS = {"XGBClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
               "DecisionTreeClassifier", "GradientBoostingClassifier"}


# --------------------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------------------

@dataclass
class Contribution:
    feature: str  # e.g. "la"
    label: str  # human-readable name, e.g. "Lines added"
    value: float  # raw feature value of this commit
    shap_value: float  # contribution to risk on the probability scale (positive = raises risk)
    description: str  # one-sentence description of the feature


@dataclass
class CommitExplanation:
    risk: float  # predicted probability that the commit is buggy
    risk_level: str  # "Low" | "Medium" | "High"
    is_risky: bool  # risk >= the model's decision threshold
    base_value: float  # average predicted risk of the training (background) commits
    contributions: list[Contribution]  # all features, biggest |shap_value| first
    threshold: float = 0.5
    background: pd.DataFrame | None = None  # past commits, used to phrase percentiles

    def top_reasons(self, k: int = 3) -> list[str]:
        """Plain-English sentences for the ``k`` factors that raise the risk the most.

        If no factor raises the risk (a very safe commit), the biggest factors are used instead.
        """
        raising = [c for c in self.contributions if c.shap_value > 0]
        chosen = raising[:k] if raising else self.contributions[:k]
        return [describe_contribution(c, self.background) for c in chosen]

    def lowering_reasons(self, k: int = 2) -> list[str]:
        """Plain-English sentences for the ``k`` factors that lower the risk the most."""
        lowering = [c for c in self.contributions if c.shap_value < 0]
        return [describe_contribution(c, self.background) for c in lowering[:k]]

    def to_text(self) -> str:
        """A multi-line risk report for a terminal or a pull-request comment."""
        verdict = "FLAGGED as risky" if self.is_risky else "not flagged"
        lines = [
            "Commit risk report",
            "==================",
            f"Predicted risk : {self.risk:.1%}  ({self.risk_level} risk, {verdict}; "
            f"threshold {self.threshold:.1%})",
            f"Average commit : {self.base_value:.1%}",
            "",
            "Factors raising the risk:" if any(c.shap_value > 0 for c in self.contributions)
            else "Biggest factors:",
        ]
        lines += [f"  {i}. {reason}" for i, reason in enumerate(self.top_reasons(3), start=1)]
        lowering = self.lowering_reasons(2)
        if lowering:
            lines += ["", "Factors lowering the risk:"]
            lines += [f"  - {reason}" for reason in lowering]
        lines += ["", f"  {'Feature':<30}{'Value':>12}{'Effect (pts)':>15}"]
        for c in self.contributions:
            lines.append(f"  {c.label:<30}{_fmt(c.value):>12}{100 * c.shap_value:>+15.1f}")
        lines.append(f"  {'Average commit':<30}{'':>12}{100 * self.base_value:>15.1f}")
        lines.append(f"  {'= Predicted risk':<30}{'':>12}{100 * self.risk:>15.1f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Plain-English sentences
# --------------------------------------------------------------------------------------

# How to say "this commit has value v" for each feature. {v} is the formatted value.
_PHRASES = {
    "ns": "Touches {v} subsystem{s}",
    "nd": "Touches {v} director{ies}",
    "nf": "Modifies {v} file{s}",
    "entropy": "Spreads its changes across files with entropy {v}",
    "la": "Adds {v} line{s}",
    "ld": "Deletes {v} line{s}",
    "lt": "The modified files had {v} line{s} on average before the change",
    "ndev": "{v} developer{s} changed these files before",
    "age": "The files were last changed {v} day{s} ago on average",
    "nuc": "These files were changed in {v} earlier commit{s}",
    "exp": "The author has {v} earlier commit{s}",
    "rexp": "The author's recent-experience score is {v}",
    "sexp": "The author has {v} earlier commit{s} in these subsystems",
}
_COUNT_FEATURES = {"ns", "nd", "nf", "la", "ld", "lt", "ndev", "nuc", "exp", "sexp"}


def _fmt(value: float) -> str:
    """512.0 -> '512', 0.4567 -> '0.46'."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _effect(shap_value: float) -> str:
    points = abs(100 * shap_value)
    direction = "raises" if shap_value > 0 else "lowers"
    return f"{direction} the risk by {points:.1f} percentage point{'' if round(points, 1) == 1 else 's'}"


def describe_contribution(c: Contribution, background: pd.DataFrame | None = None) -> str:
    """One sentence, e.g. 'Adds 512 lines, more than 92% of past commits, which raises ...'."""
    if c.feature == "fix":
        what = "It is a bug-fix commit" if c.value >= 0.5 else "It is not a bug-fix commit"
        return f"{what}, which {_effect(c.shap_value)}."

    plural = c.value != 1
    what = _PHRASES.get(c.feature, c.label + " is {v}").format(
        v=_fmt(c.value), s="s" if plural else "", ies="ies" if plural else "y")

    comparison = ""
    if background is not None and c.feature in background:
        past = background[c.feature].to_numpy(dtype=float)
        below = 100 * np.mean(past < c.value)  # % of past commits with a smaller value
        above = 100 * np.mean(past > c.value)  # % of past commits with a larger value
        more, less = ("more", "fewer") if c.feature in _COUNT_FEATURES else ("higher", "lower")
        if below >= 99.5:  # beyond every sampled past commit; "more than 100%" would read oddly
            comparison = f", {more} than almost all past commits"
        elif above >= 99.5:
            comparison = f", {less} than almost all past commits"
        elif below >= 50:
            comparison = f", {more} than {below:.0f}% of past commits"
        elif above >= 50:
            comparison = f", {less} than {above:.0f}% of past commits"
        else:
            comparison = ", about typical for this project"
    return f"{what}{comparison}, which {_effect(c.shap_value)}."


# --------------------------------------------------------------------------------------
# Loading and SHAP computation
# --------------------------------------------------------------------------------------

def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict:
    """Load the model bundle saved by the training step."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model bundle at {path}. Train one first: uv run python -m riskpredictor.train")
    return joblib.load(path)


def _as_frame(row: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """One commit (Series or 1-row DataFrame) -> 1-row DataFrame with exactly FEATURE_COLUMNS."""
    frame = row.to_frame().T if isinstance(row, pd.Series) else row
    return frame[FEATURE_COLUMNS].astype(float).reset_index(drop=True)


def _predict_risk(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["pipeline"].predict_proba(X[FEATURE_COLUMNS])[:, 1]


def _shap_background(bundle: dict, max_rows: int = 100) -> pd.DataFrame:
    """The "average commit" SHAP compares against: up to 100 training commits (keeps it fast)."""
    background = bundle["background"][FEATURE_COLUMNS].astype(float)
    if len(background) > max_rows:
        background = background.sample(max_rows, random_state=0)
    return background.reset_index(drop=True)


def _tree_shap(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact SHAP values for tree models via TreeExplainer on the pre-processed features."""
    prep, clf = bundle["pipeline"].steps[0][1], bundle["pipeline"].steps[-1][1]
    background = np.asarray(prep.transform(_shap_background(bundle)), dtype=float)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.TreeExplainer(clf, data=masker, model_output="probability",
                                   feature_perturbation="interventional")
    values = explainer.shap_values(np.asarray(prep.transform(X), dtype=float), check_additivity=False)
    expected = explainer.expected_value
    if isinstance(values, list):  # older sklearn-style output: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes) -> the "buggy" class
        values = values[:, :, 1]
    expected = float(np.ravel(expected)[-1])
    return values, expected


def _permutation_shap(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Model-agnostic SHAP: only needs predict_proba, so it works for every classifier."""
    pipeline = bundle["pipeline"]

    def f(data: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(pd.DataFrame(data, columns=FEATURE_COLUMNS))[:, 1]

    background = _shap_background(bundle).to_numpy(dtype=float)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(f, masker, seed=0)
    result = explainer(X.to_numpy(dtype=float), max_evals=2 * len(FEATURE_COLUMNS) * 8 + 1, silent=True)
    return np.asarray(result.values), float(np.ravel(result.base_values)[0])


def _shap_values(bundle: dict, X: pd.DataFrame, method: str = "auto") -> tuple[np.ndarray, float]:
    """SHAP values (n, 14) and the base value, on the probability scale.

    ``method="auto"`` uses TreeExplainer for tree models and falls back to the model-agnostic
    permutation explainer if that fails or its values are not additive.
    """
    X = X[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    clf = bundle["pipeline"].steps[-1][1]
    if method == "auto" and type(clf).__name__ in TREE_MODELS:
        try:
            values, base = _tree_shap(bundle, X)
            if np.allclose(base + values.sum(axis=1), _predict_risk(bundle, X), atol=0.01):
                return values, base
        except Exception:  # unsupported model/version -> use the generic method below
            pass
    return _permutation_shap(bundle, X)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def explain_commit(bundle: dict, row: pd.Series | pd.DataFrame) -> CommitExplanation:
    """Predict the risk of one commit and explain it feature by feature."""
    X = _as_frame(row)
    risk = float(_predict_risk(bundle, X)[0])
    values, base = _shap_values(bundle, X)
    contributions = [
        Contribution(feature=f, label=FEATURE_INFO[f].label, value=float(X.at[0, f]),
                     shap_value=float(v), description=FEATURE_INFO[f].description)
        for f, v in zip(FEATURE_COLUMNS, values[0])
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    threshold = float(bundle.get("threshold", 0.5))
    return CommitExplanation(
        risk=risk,
        risk_level=risk_level(risk, threshold),
        is_risky=risk >= threshold,
        base_value=base,
        contributions=contributions,
        threshold=threshold,
        background=bundle["background"],
    )


def _sample(X: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    X = X[FEATURE_COLUMNS].astype(float)
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=0)
    return X.reset_index(drop=True)


def _importance_table(values: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "label": [FEATURE_INFO[f].label for f in FEATURE_COLUMNS],
        "mean_abs_shap": np.abs(values).mean(axis=0),
    })
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def global_importance(bundle: dict, X: pd.DataFrame, max_samples: int = 300) -> pd.DataFrame:
    """Which features matter most overall: mean |SHAP value| over (a sample of) commits."""
    values, _ = _shap_values(bundle, _sample(X, max_samples))
    return _importance_table(values)


def _lime_explanation(bundle: dict, row, num_features: int, num_samples: int = 2000):
    from lime.lime_tabular import LimeTabularExplainer  # imported here: only needed for LIME

    pipeline = bundle["pipeline"]
    explainer = LimeTabularExplainer(
        bundle["background"][FEATURE_COLUMNS].to_numpy(dtype=float),
        feature_names=FEATURE_COLUMNS,
        class_names=["clean", "buggy"],
        mode="classification",
        discretize_continuous=True,
        random_state=0,
    )
    predict_fn = lambda data: pipeline.predict_proba(pd.DataFrame(data, columns=FEATURE_COLUMNS))  # noqa: E731
    instance = _as_frame(row).to_numpy(dtype=float)[0]
    return explainer.explain_instance(instance, predict_fn, labels=(1,),
                                      num_features=num_features, num_samples=num_samples)


def explain_commit_lime(bundle: dict, row, num_features: int = 6) -> list[tuple[str, float]]:
    """LIME explanation: [(condition like 'la > 120.00', weight on the risk), ...]."""
    return _lime_explanation(bundle, row, num_features).as_list(label=1)


def compare_shap_lime(bundle: dict, X_sample: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Do SHAP and LIME agree on which features drive each commit's risk?

    For every commit we report
      * top_k_overlap - fraction of SHAP's top-k features that are also in LIME's top-k
      * spearman      - rank correlation of |SHAP| and |LIME| importance over all 14 features
    """
    X = X_sample[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    shap_vals, _ = _shap_values(bundle, X)
    risks = _predict_risk(bundle, X)
    rows = []
    for i in range(len(X)):
        lime_map = dict(_lime_explanation(bundle, X.iloc[i], len(FEATURE_COLUMNS)).as_map()[1])
        lime_abs = np.array([abs(lime_map.get(j, 0.0)) for j in range(len(FEATURE_COLUMNS))])
        shap_abs = np.abs(shap_vals[i])
        shap_top = [FEATURE_COLUMNS[j] for j in np.argsort(-shap_abs)[:k]]
        lime_top = [FEATURE_COLUMNS[j] for j in np.argsort(-lime_abs)[:k]]
        rho = pd.Series(shap_abs).corr(pd.Series(lime_abs), method="spearman")
        rows.append({
            "commit": X_sample["commit_hash"].iloc[i] if "commit_hash" in X_sample else i,
            "risk": round(float(risks[i]), 4),
            "shap_top": ", ".join(shap_top),
            "lime_top": ", ".join(lime_top),
            "top_k_overlap": len(set(shap_top) & set(lime_top)) / k,
            "spearman": round(float(rho), 4) if pd.notna(rho) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def plot_commit_explanation(expl: CommitExplanation, path, max_features: int = 10) -> None:
    """Horizontal waterfall: from the average risk, each feature pushes the risk up or down."""
    shown = expl.contributions[:max_features]
    rest = expl.contributions[max_features:]
    labels = [f"{c.label} = {_fmt(c.value)}" for c in shown]
    values = [c.shap_value for c in shown]
    if rest:
        labels.append(f"{len(rest)} other features")
        values.append(sum(c.shap_value for c in rest))

    # Draw bottom-up so the most important feature ends up at the top, next to the final risk.
    labels, values = labels[::-1], values[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(values) + 1.8))
    position = expl.base_value  # each bar starts where the previous one ended
    for y, (label, v) in enumerate(zip(labels, values)):
        color = RAISES_COLOR if v > 0 else LOWERS_COLOR
        ax.barh(y, v, left=position, color=color, height=0.6)
        ax.text(position + v + (0.002 if v >= 0 else -0.002), y, f"{100 * v:+.1f} pts",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=TEXT_COLOR)
        position += v

    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=TEXT_COLOR)
    ax.axvline(expl.base_value, color="#8a8984", linestyle="--", linewidth=1)
    ax.axvline(expl.risk, color=TEXT_COLOR, linewidth=1.2)
    # Read bottom-up: start at the average commit, end (top) at this commit's predicted risk.
    ax.text(expl.base_value, -0.9, f" average commit {expl.base_value:.0%}", fontsize=8, color="#52514e")
    ax.text(expl.risk, len(labels) - 0.3, f" this commit {expl.risk:.0%}", fontsize=8, color=TEXT_COLOR)
    ax.set_ylim(-1.2, len(labels) + 0.1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Predicted probability that the commit introduces a bug")
    ax.set_title(f"Commit risk {expl.risk:.0%} ({expl.risk_level}): red raises risk, blue lowers it",
                 fontsize=11, color=TEXT_COLOR, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    # leave room for the "+x pts" labels on both sides
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lime_explanation(weights: list[tuple[str, float]], path, title: str = "LIME explanation") -> None:
    """Bar chart of LIME's (condition, weight) pairs."""
    conditions, values = [w[0] for w in weights][::-1], [w[1] for w in weights][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(values) + 1.5))
    ax.barh(conditions, values, color=[RAISES_COLOR if v > 0 else LOWERS_COLOR for v in values], height=0.6)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set_xlabel("Effect on predicted risk (local linear weight)")
    ax.set_title(title, fontsize=11, loc="left", color=TEXT_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_global(X: pd.DataFrame, values: np.ndarray, base: float, out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(values=values, base_values=np.full(len(X), base),
                                   data=X.to_numpy(dtype=float),
                                   feature_names=[FEATURE_INFO[f].label for f in FEATURE_COLUMNS])
    paths = []
    plots = [
        ("shap_summary.png", shap.plots.beeswarm, "SHAP value (change in predicted probability of a bug)"),
        ("shap_bar.png", shap.plots.bar, "Mean |SHAP value| (average change in predicted probability)"),
    ]
    for name, draw, xlabel in plots:
        plt.figure()
        draw(explanation, max_display=len(FEATURE_COLUMNS), show=False)
        fig = plt.gcf()
        fig.axes[0].set_xlabel(xlabel)
        fig.suptitle("SHAP: impact of each feature on predicted risk (probability)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_dir / name)
    return paths


def plot_global_importance(bundle: dict, X: pd.DataFrame, out_dir, max_samples: int = 300) -> list[Path]:
    """Save the SHAP beeswarm (shap_summary.png) and mean |SHAP| bar chart (shap_bar.png)."""
    X = _sample(X, max_samples)
    values, base = _shap_values(bundle, X)
    return _plot_global(X, values, base, out_dir)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explain commit risk predictions with SHAP and LIME.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--commit", help="hash (or prefix) of the commit to explain")
    parser.add_argument("--global", dest="global_", action="store_true",
                        help="global SHAP plots + SHAP-vs-LIME agreement on test-period commits")
    parser.add_argument("--out", default=str(FIGURES_DIR), help="folder for figures")
    parser.add_argument("--agreement-samples", type=int, default=50)
    args = parser.parse_args(argv)
    if not args.commit and not args.global_:
        parser.error("give --commit HASH and/or --global")

    bundle = load_bundle(args.bundle)
    df = load_dataset(args.data)
    out_dir = Path(args.out)
    print(f"Model: {bundle.get('model_name', '?')}  (threshold {bundle.get('threshold', 0.5):.2f})")

    if args.commit:
        matches = df[df["commit_hash"].str.startswith(args.commit)]
        if matches.empty:
            raise SystemExit(f"No commit starting with {args.commit!r} in {args.data}")
        row = matches.iloc[0]
        short = row["commit_hash"][:8]
        print(f"\nCommit {short}: {str(row['message']).splitlines()[0][:72]}")
        if "buggy" in row:
            print(f"(actual label: {'buggy' if row['buggy'] == 1 else 'clean'})\n")
        expl = explain_commit(bundle, row)
        print(expl.to_text())
        plot_commit_explanation(expl, out_dir / f"commit_{short}.png")
        plot_lime_explanation(explain_commit_lime(bundle, row), out_dir / f"lime_{short}.png",
                              title=f"LIME explanation for commit {short}")
        print(f"\nSaved {out_dir / f'commit_{short}.png'} and {out_dir / f'lime_{short}.png'}")

    if args.global_:
        _, test = time_split(df)
        X = _sample(test, 300)
        values, base = _shap_values(bundle, X)
        print("\nGlobal feature importance (mean |SHAP|, percentage points):")
        table = _importance_table(values)
        print(table.assign(mean_abs_shap=(100 * table["mean_abs_shap"]).round(2)).to_string(index=False))
        for path in _plot_global(X, values, base, out_dir):
            print(f"Saved {path}")

        sample = test.sample(min(args.agreement_samples, len(test)), random_state=0).reset_index(drop=True)
        agreement = compare_shap_lime(bundle, sample, k=3)
        csv_path = out_dir.parent / "shap_lime_agreement.csv"
        agreement.to_csv(csv_path, index=False)
        print(f"\nSHAP vs LIME on {len(agreement)} test commits: mean top-3 overlap "
              f"{agreement['top_k_overlap'].mean():.2f}, mean Spearman {agreement['spearman'].mean():.2f}")
        print(f"Saved {csv_path}")

        riskiest = sample.iloc[int(agreement["risk"].idxmax())]
        plot_lime_explanation(explain_commit_lime(bundle, riskiest), out_dir / "lime_example.png",
                              title=f"LIME explanation for riskiest sampled commit {riskiest['commit_hash'][:8]}")
        print(f"Saved {out_dir / 'lime_example.png'}")


if __name__ == "__main__":
    main()
