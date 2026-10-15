"""Train, compare and save the code change risk models.

    uv run python -m riskpredictor.train --data data/processed/dataset.csv \\
        --models-dir models --reports-dir reports [--quick] [--synthetic]

Every decision is made on training data only; the test set is used once per model, to report results.

1. Split by time: train = oldest 80% of commits, test = newest 20%. The newest 20% of the training
   period is the validation slice used to choose decision thresholds.
2. Cross-validate every algorithm on the training period with TimeSeriesSplit (each fold trains on
   the past and validates on the future). Tune Random Forest, XGBoost, Logistic Regression and the
   RBF SVM with RandomizedSearchCV.
3. Choose each model's threshold on the validation slice (max F1), refit it on the whole training
   period and score it on the test set.
4. Class-imbalance study: no handling vs class weights vs SMOTE, for LR, Random Forest and XGBoost.
5. Select the model with the highest mean CV ROC-AUC (not the highest test score) and save it to
   models/best_model.joblib.
6. Bias-variance figures: decision-tree depth vs ROC-AUC, and a learning curve of the best model.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate, validation_curve  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from riskpredictor.features import (  # noqa: E402
    DATE_COLUMN,
    DEFAULT_DATASET,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    load_dataset,
    make_synthetic_dataset,
)
from riskpredictor.modeling import (  # noqa: E402
    IMBALANCE_STRATEGIES,
    METRIC_INFO,
    RANDOM_STATE,
    churn,
    evaluate,
    fit_and_choose_threshold,
    make_bundle,
    make_pipeline,
    model_zoo,
    risk_scores,
    save_bundle,
    train_val_test_split,
)

IMBALANCE_MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]
IMBALANCE_LABELS = {
    "none": "no handling",
    "balanced": "class weights (balanced / scale_pos_weight)",
    "smote": "SMOTE oversampling (training folds only)",
}


@dataclass
class TrainConfig:
    n_splits: int = 5  # TimeSeriesSplit folds inside the training period
    n_iter: int = 25  # RandomizedSearchCV candidates per tuned model
    imbalance: str = "balanced"  # strategy for the main comparison: none / balanced / smote
    random_state: int = RANDOM_STATE
    learning_curve_points: int = 8

    @classmethod
    def quick(cls, **kwargs) -> TrainConfig:
        return cls(n_splits=3, n_iter=4, learning_curve_points=4, **kwargs)


def _pos_weight(df: pd.DataFrame) -> float:
    """#clean / #buggy, XGBoost's equivalent of class_weight="balanced"."""
    n_pos = max(int(df[LABEL_COLUMN].sum()), 1)
    return float((len(df) - n_pos) / n_pos)


def _clean_params(params: dict) -> dict:
    """{"clf__estimator__C": np.float64(3.2)} -> {"C": 3.2} for readable reports."""
    out = {}
    for key, value in params.items():
        key = key.removeprefix("clf__").removeprefix("estimator__")
        value = value.item() if hasattr(value, "item") else value
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------


def compare_models(train, fit, val, test, cfg: TrainConfig, log=print) -> tuple[pd.DataFrame, dict]:
    """Cross-validate (and tune) every model, then score it once on the test set.

    Returns the leaderboard (sorted by mean CV ROC-AUC) and a dict ``runs[name]`` holding the
    unfitted pipeline, the final fitted pipeline, its threshold, test scores and test metrics.
    """
    X, y = train[FEATURE_COLUMNS], train[LABEL_COLUMN]
    cv = TimeSeriesSplit(n_splits=cfg.n_splits)
    zoo = model_zoo(balanced=cfg.imbalance == "balanced", pos_weight=_pos_weight(train), random_state=cfg.random_state)
    rows, runs = [], {}
    for name, spec in zoo.items():
        start = time.perf_counter()
        use_smote = cfg.imbalance == "smote" and spec.selectable  # SMOTE is pointless for the clean-only baseline
        pipe = make_pipeline(spec.estimator, smote=use_smote, random_state=cfg.random_state)

        cv_scores = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)["test_score"]
        default_cv = float(cv_scores.mean())
        cv_mean, cv_std, params, tuning = default_cv, float(cv_scores.std()), {}, "not tuned"
        if spec.search_space:
            search = RandomizedSearchCV(pipe, spec.search_space, n_iter=cfg.n_iter, scoring="roc_auc", cv=cv,
                                        n_jobs=-1, random_state=cfg.random_state, refit=False).fit(X, y)
            tuning = "defaults kept"  # unless a candidate beats the default settings in CV
            if search.best_score_ > default_cv:
                params, tuning = search.best_params_, "tuned"
                cv_mean = float(search.best_score_)
                cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
                pipe.set_params(**params)

        final, threshold = fit_and_choose_threshold(pipe, fit, val, train)
        test_scores = risk_scores(final, test[FEATURE_COLUMNS])
        metrics = evaluate(test[LABEL_COLUMN], test_scores, threshold, churn(test))
        seconds = time.perf_counter() - start

        note = "" if spec.selectable else "anomaly detector fitted on clean commits only; risk = -decision_function; not eligible as best model"
        rows.append({
            "model": name, "family": spec.family, "tuning": tuning,
            "cv_roc_auc": cv_mean, "cv_roc_auc_std": cv_std, "cv_roc_auc_default": default_cv,
            "threshold": threshold, **{f"test_{k}": v for k, v in metrics.items()},
            "seconds": round(seconds, 1), "params": json.dumps(_clean_params(params)),
            "selectable": spec.selectable, "note": note,
        })
        runs[name] = {"pipeline": pipe, "final": final, "threshold": threshold,
                      "test_scores": test_scores, "metrics": metrics, "params": _clean_params(params)}
        log(f"  {name:<34} CV ROC-AUC {cv_mean:.3f} ± {cv_std:.3f}   test ROC-AUC {metrics['roc_auc']:.3f}   ({seconds:.1f}s)")

    leaderboard = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False, kind="stable").reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard, runs


def select_best(leaderboard: pd.DataFrame) -> str:
    """Highest mean CV ROC-AUC on the training period, among models that output probabilities."""
    eligible = leaderboard[leaderboard["selectable"]]
    return str(eligible.loc[eligible["cv_roc_auc"].idxmax(), "model"])


def compare_imbalance(train, fit, val, test, cfg: TrainConfig, log=print) -> pd.DataFrame:
    """None vs class weights vs SMOTE on LR, Random Forest and XGBoost.

    A controlled experiment: every run uses the same default hyperparameters, only the imbalance
    handling changes. We also report recall/F1 at the naive 0.5 threshold, because that is where
    imbalance hurts most (an unweighted model rarely predicts the minority class).
    """
    X, y = train[FEATURE_COLUMNS], train[LABEL_COLUMN]
    cv = TimeSeriesSplit(n_splits=cfg.n_splits)
    zoos = {s: model_zoo(balanced=s == "balanced", pos_weight=_pos_weight(train), random_state=cfg.random_state)
            for s in IMBALANCE_STRATEGIES}
    rows = []
    for name in IMBALANCE_MODELS:
        for strategy in IMBALANCE_STRATEGIES:
            pipe = make_pipeline(zoos[strategy][name].estimator, smote=strategy == "smote", random_state=cfg.random_state)
            cv_res = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=-1)
            final, threshold = fit_and_choose_threshold(pipe, fit, val, train)
            scores = risk_scores(final, test[FEATURE_COLUMNS])
            metrics = evaluate(test[LABEL_COLUMN], scores, threshold, churn(test))
            at_half = evaluate(test[LABEL_COLUMN], scores, 0.5, churn(test))
            rows.append({
                "model": name, "strategy": strategy, "description": IMBALANCE_LABELS[strategy],
                "cv_roc_auc": cv_res["test_roc_auc"].mean(), "cv_pr_auc": cv_res["test_average_precision"].mean(),
                "threshold": threshold, **{f"test_{k}": v for k, v in metrics.items()},
                "test_recall_at_0.5": at_half["recall"], "test_f1_at_0.5": at_half["f1"],
            })
            log(f"  {name:<20} {strategy:<9} CV ROC-AUC {rows[-1]['cv_roc_auc']:.3f}  CV PR-AUC {rows[-1]['cv_pr_auc']:.3f}"
                f"  test recall@0.5 {at_half['recall']:.2f}")
    return pd.DataFrame(rows)


def complexity_curve(train, cfg: TrainConfig, max_depth: int = 15) -> pd.DataFrame:
    """Bias-variance: decision-tree max_depth vs train and cross-validated ROC-AUC."""
    tree = DecisionTreeClassifier(class_weight="balanced" if cfg.imbalance == "balanced" else None,
                                  random_state=cfg.random_state)
    pipe = make_pipeline(tree, smote=cfg.imbalance == "smote", random_state=cfg.random_state)
    depths = np.arange(1, max_depth + 1)
    train_scores, val_scores = validation_curve(
        pipe, train[FEATURE_COLUMNS], train[LABEL_COLUMN], param_name="clf__max_depth", param_range=depths,
        cv=TimeSeriesSplit(n_splits=cfg.n_splits), scoring="roc_auc", n_jobs=-1)
    return pd.DataFrame({
        "max_depth": depths,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1), "val_std": val_scores.std(axis=1),
    })


def learning_curve(pipeline, fit, val, cfg: TrainConfig) -> pd.DataFrame:
    """Train on the n most recent commits before the validation slice, for growing n.

    Every subset ends right before the validation slice, so the only thing that changes is how much
    history the model sees. A large train/validation gap means high variance (more data helps);
    two low, close curves mean high bias (a more flexible model is needed).
    """
    rows = []
    for frac in np.linspace(0.1, 1.0, cfg.learning_curve_points):
        part = fit.iloc[-int(round(frac * len(fit))):]
        if part[LABEL_COLUMN].sum() < 15:  # too few buggy commits to fit (SMOTE / calibration need a few)
            continue
        model = clone(pipeline).fit(part[FEATURE_COLUMNS], part[LABEL_COLUMN])
        rows.append({
            "n_train": len(part),
            "train_roc_auc": roc_auc_score(part[LABEL_COLUMN], risk_scores(model, part[FEATURE_COLUMNS])),
            "val_roc_auc": roc_auc_score(val[LABEL_COLUMN], risk_scores(model, val[FEATURE_COLUMNS])),
        })
    return pd.DataFrame(rows)


def describe_split(df, train, fit, val, test, cfg: TrainConfig, data_path: str) -> dict:
    def part(d: pd.DataFrame) -> dict:
        return {"n": len(d), "start": d[DATE_COLUMN].min().isoformat(), "end": d[DATE_COLUMN].max().isoformat(),
                "n_buggy": int(d[LABEL_COLUMN].sum()), "bug_rate": round(float(d[LABEL_COLUMN].mean()), 4)}

    return {
        "dataset": data_path, "n_commits": len(df),
        "train": part(train), "fit": part(fit), "validation": part(val), "test": part(test),
        "explanation": "train = oldest 80% of commits, test = newest 20%. Inside train, 'validation' is the newest "
                       "20% (used only to choose decision thresholds) and 'fit' is the rest. "
                       "Cross-validation and tuning use TimeSeriesSplit on train.",
        "cv": f"TimeSeriesSplit(n_splits={cfg.n_splits}) on train",
        "tuning": f"RandomizedSearchCV(n_iter={cfg.n_iter}, scoring='roc_auc')",
        "imbalance_strategy": cfg.imbalance,
    }


# ---------------------------------------------------------------------------------------------
# Figures (colour-blind-safe categorical palette, hairline grid, text in neutral ink)
# ---------------------------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]  # second channel, so lines differ without colour
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f7fd", "#9ec5f4", "#2a78d6", "#104281"])
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK_SOFT, "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT,
    "legend.frameon": False, "legend.labelcolor": INK_SOFT, "lines.solid_capstyle": "round",
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(runs: dict, names: list[str], y_test, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, label="Random guessing (AUC 0.50)")
        for i, name in enumerate(names):
            fpr, tpr, _ = roc_curve(y_test, runs[name]["test_scores"])
            ax.plot(fpr, tpr, color=PALETTE[i], ls=LINESTYLES[i], lw=2,
                    label=f"{name} (AUC {runs[name]['metrics']['roc_auc']:.3f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (clean commits flagged)",
               ylabel="True positive rate (buggy commits caught)",
               title="ROC curves on the test set\n(top models by cross-validated ROC-AUC)")
        ax.legend(loc="lower right", fontsize=8.5)
        _save(fig, path)


def plot_pr(runs: dict, names: list[str], y_test, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        bug_rate = float(np.mean(y_test))
        ax.axhline(bug_rate, color=MUTED, lw=1, label=f"Random guessing (bug rate {bug_rate:.2f})")
        for i, name in enumerate(names):
            precision, recall, _ = precision_recall_curve(y_test, runs[name]["test_scores"])
            ax.plot(recall, precision, color=PALETTE[i], ls=LINESTYLES[i], lw=2,
                    label=f"{name} (AP {runs[name]['metrics']['pr_auc']:.3f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall (share of buggy commits caught)",
               ylabel="Precision (share of flagged commits that are buggy)",
               title="Precision-recall curves on the test set\n(top models by cross-validated ROC-AUC)")
        ax.legend(loc="upper right", fontsize=8.5)
        _save(fig, path)


def plot_confusion(y_test, scores, threshold: float, name: str, path: Path) -> None:
    pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    row_share = cm / cm.sum(axis=1, keepdims=True)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(row_share, cmap=BLUES, vmin=0, vmax=1)
        labels = ["Clean", "Buggy"]
        for i in range(2):
            for j in range(2):
                colour = "white" if row_share[i, j] > 0.55 else INK
                ax.text(j, i, f"{cm[i, j]}\n{row_share[i, j]:.0%} of actual {labels[i].lower()}",
                        ha="center", va="center", color=colour, fontsize=10)
        ax.set_xticks([0, 1], [f"Predicted {lab.lower()}" for lab in labels])
        ax.set_yticks([0, 1], [f"Actually {lab.lower()}" for lab in labels])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"Confusion matrix on the test set\n{name}, threshold {threshold:.3f}")
        _save(fig, path)


def plot_comparison(leaderboard: pd.DataFrame, best: str, path: Path) -> None:
    """Bars = test ROC-AUC measured from 0.5 (random guessing); diamonds = CV ROC-AUC used for selection."""
    lb = leaderboard.iloc[::-1]  # best CV at the top
    y = np.arange(len(lb))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(lb) + 1.6))
        colours = [PALETTE[0] if sel else MUTED for sel in lb["selectable"]]
        ax.barh(y, lb["test_roc_auc"] - 0.5, left=0.5, height=0.55, color=colours)
        ax.scatter(lb["cv_roc_auc"], y, marker="D", s=30, color=PALETTE[1], edgecolor="white", linewidth=1, zorder=3)
        for yi, value in zip(y, lb["test_roc_auc"]):
            ax.text(max(value, 0.5) + 0.006, yi, f"{value:.3f}", va="center", fontsize=8, color=INK_SOFT)
        names = [f"{m} (selected)" if m == best else m for m in lb["model"]]
        ax.set_yticks(y, names)
        ax.axvline(0.5, color=AXIS, lw=1)
        low = min(0.45, float(lb[["test_roc_auc", "cv_roc_auc"]].min().min()) - 0.03)
        high = max(float(lb[["test_roc_auc", "cv_roc_auc"]].max().max()) + 0.06, 0.6)
        ax.set(xlim=(low, high), xlabel="ROC-AUC (0.5 = random guessing)",
               title="Model comparison, sorted by cross-validated ROC-AUC")
        ax.grid(axis="y", visible=False)
        handles = [Patch(color=PALETTE[0], label="Test ROC-AUC (newest 20% of commits)"),
                   Patch(color=MUTED, label="Test ROC-AUC, anomaly baseline"),
                   Line2D([], [], marker="D", ls="", color=PALETTE[1], markeredgecolor="white",
                          label="CV ROC-AUC on training period (used to select)")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=2, fontsize=8)
        _save(fig, path)


def plot_complexity(curve: pd.DataFrame, path: Path) -> None:
    best = curve.loc[curve["val_mean"].idxmax()]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        for col, label, colour, ls in [("train", "Training folds", PALETTE[0], "-"),
                                       ("val", "Validation folds (TimeSeriesSplit)", PALETTE[1], "--")]:
            ax.plot(curve["max_depth"], curve[f"{col}_mean"], color=colour, ls=ls, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(curve["max_depth"], curve[f"{col}_mean"] - curve[f"{col}_std"],
                            curve[f"{col}_mean"] + curve[f"{col}_std"], color=colour, alpha=0.1, lw=0)
        ax.axvline(best["max_depth"], color=AXIS, lw=1)
        ax.annotate(f"best depth = {int(best['max_depth'])}\n(validation AUC {best['val_mean']:.3f})",
                    (best["max_depth"], best["val_mean"]), xytext=(8, -36), textcoords="offset points",
                    fontsize=8.5, color=INK_SOFT)
        ax.text(0.02, 0.98, "← underfitting (high bias)", transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.text(0.98, 0.98, "overfitting (high variance) →", transform=ax.transAxes, va="top", ha="right",
                fontsize=8.5, color=MUTED)
        low = float((curve[["train_mean", "val_mean"]] - curve[["train_std", "val_std"]].values).min().min())
        ax.set(xlabel="Decision tree max_depth (model complexity)", ylabel="ROC-AUC", ylim=(low - 0.02, 1.1),
               title="Bias-variance trade-off: decision tree complexity curve")
        ax.set_xticks(curve["max_depth"])
        ax.set_yticks([t for t in ax.get_yticks() if low - 0.02 <= t <= 1.0])  # AUC never exceeds 1
        ax.legend(loc="lower left")
        _save(fig, path)


def plot_learning(curve: pd.DataFrame, name: str, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.plot(curve["n_train"], curve["train_roc_auc"], color=PALETTE[0], lw=2, marker="o", ms=4,
                label="Training data")
        ax.plot(curve["n_train"], curve["val_roc_auc"], color=PALETTE[1], ls="--", lw=2, marker="o", ms=4,
                label="Validation slice (newest 20% of training period)")
        ax.set(xlabel="Number of training commits (most recent before the validation slice)", ylabel="ROC-AUC",
               title=f"Learning curve: {name}")
        ax.legend(loc="lower right")
        _save(fig, path)


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_markdown(path: Path, leaderboard, imbalance, best: str, runs: dict, split: dict,
                   complexity: pd.DataFrame, cfg: TrainConfig) -> None:
    lb = leaderboard.assign(cv=[f"{m:.3f} ± {s:.3f}" for m, s in zip(leaderboard["cv_roc_auc"], leaderboard["cv_roc_auc_std"])])
    table = lb[["rank", "model", "family", "tuning", "cv", "test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
                "test_recall", "test_mcc", "test_balanced_accuracy", "test_recall_at_20_effort"]]
    table.columns = ["#", "Model", "Type", "Tuning", "CV ROC-AUC (train)", "Test ROC-AUC", "Test PR-AUC", "F1",
                     "Precision", "Recall", "MCC", "Bal. acc.", "Recall@20%Effort"]
    imb = imbalance[["model", "description", "cv_roc_auc", "cv_pr_auc", "test_roc_auc", "test_pr_auc", "test_f1",
                     "test_recall", "test_recall_at_0.5", "test_f1_at_0.5"]]
    imb.columns = ["Model", "Imbalance handling", "CV ROC-AUC", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC",
                   "F1 (tuned thr.)", "Recall (tuned thr.)", "Recall @0.5", "F1 @0.5"]
    searched = leaderboard[leaderboard["tuning"] != "not tuned"]
    tuning = pd.DataFrame({"Model": searched["model"], "CV ROC-AUC (defaults)": searched["cv_roc_auc_default"],
                           "CV ROC-AUC (after search)": searched["cv_roc_auc"], "Result": searched["tuning"],
                           "Best hyperparameters": [f"`{p}`" for p in searched["params"]]})
    run, row = runs[best], leaderboard.set_index("model").loc[best]
    eligible = leaderboard[leaderboard["selectable"]]
    tied = eligible.loc[eligible["cv_roc_auc"] >= row["cv_roc_auc"] - row["cv_roc_auc_std"], "model"].tolist()
    best_depth = complexity.loc[complexity["val_mean"].idxmax()]
    deepest = complexity.iloc[-1]
    tr, te = split["train"], split["test"]
    text = f"""# Model comparison

Dataset: `{split['dataset']}`, {split['n_commits']} commits.
Time-ordered split: **train** {tr['n']} commits ({tr['start'][:10]} → {tr['end'][:10]}, bug rate {tr['bug_rate']:.1%}),
**test** {te['n']} commits ({te['start'][:10]} → {te['end'][:10]}, bug rate {te['bug_rate']:.1%}).
Cross-validation: {split['cv']}; tuning: {split['tuning']}. Thresholds are chosen on the validation slice
(newest 20% of the training period, {split['validation']['n']} commits) by maximising F1, never on the test set.
Class-imbalance handling for this comparison: **{IMBALANCE_LABELS[cfg.imbalance]}**.

## Leaderboard (sorted by mean CV ROC-AUC on the training period)

{_md_table(table)}

## Selected model: {best}

Chosen because it has the highest mean cross-validated ROC-AUC on the training period
({row['cv_roc_auc']:.3f} ± {row['cv_roc_auc_std']:.3f}); test scores played no part in the choice.
Decision threshold {run['threshold']:.3f} (max F1 on the validation slice). Hyperparameters: `{json.dumps(run['params']) if run['params'] else 'defaults'}`.

Models whose CV ROC-AUC is within one standard deviation of the winner: {', '.join(tied)}.
These are practically tied, so simplicity, speed and explainability are fair tie-breakers.

Test set: ROC-AUC {run['metrics']['roc_auc']:.3f}, PR-AUC {run['metrics']['pr_auc']:.3f} (random = {te['bug_rate']:.3f}),
F1 {run['metrics']['f1']:.3f}, precision {run['metrics']['precision']:.3f}, recall {run['metrics']['recall']:.3f},
MCC {run['metrics']['mcc']:.3f}, balanced accuracy {run['metrics']['balanced_accuracy']:.3f},
Recall@20%Effort {run['metrics']['recall_at_20_effort']:.3f}.

## Hyperparameter optimisation (RandomizedSearchCV, TimeSeriesSplit, scoring = ROC-AUC)

{_md_table(tuning)}

## Class imbalance (default hyperparameters, only the imbalance handling changes)

{_md_table(imb)}

With the threshold tuned on validation data, the three strategies end up close; the difference shows at the
naive 0.5 threshold, where an unweighted model flags far fewer buggy commits.

## Bias-variance trade-off

* Decision tree complexity curve (`figures/model_complexity_curve.png`): as max_depth grows, training ROC-AUC
  keeps rising (bias falls) while validation ROC-AUC peaks at max_depth = {int(best_depth['max_depth'])} ({best_depth['val_mean']:.3f}) and then
  drops (variance grows). At depth {int(deepest['max_depth'])} the tree scores {deepest['train_mean']:.3f} on its training folds but only
  {deepest['val_mean']:.3f} on validation folds: it memorises the training commits (overfitting).
* Learning curve of the selected model: `figures/model_learning_curve.png`.

## Notes and caveats

* PR-AUC of random guessing equals the test bug rate ({te['bug_rate']:.3f}), so compare PR-AUC against that.
* For tuned models the CV score is the best of {cfg.n_iter} random candidates, so it is slightly optimistic
  (nested cross-validation would remove this bias but costs {cfg.n_splits}x more time).
* The One-Class SVM never sees buggy commits; it is an anomaly-detection baseline and cannot be selected.
* Recall@20%Effort rewards models that rank *small* risky commits first. Big changes are risky but use up
  the review budget quickly, so a model can have a good ROC-AUC and a low Recall@20%Effort.
* Labels come from the SZZ algorithm, which is noisy: some "clean" commits contain bugs nobody has fixed yet.

## Metric glossary

""" + "\n".join(f"* {desc}" for desc in METRIC_INFO.values()) + "\n"
    path.write_text(text)


def print_leaderboard(leaderboard: pd.DataFrame, best: str) -> None:
    cols = {"rank": "#", "model": "model", "cv_roc_auc": "CV AUC", "cv_roc_auc_std": "± std",
            "test_roc_auc": "test AUC", "test_pr_auc": "test PR-AUC", "test_f1": "test F1",
            "test_mcc": "test MCC", "test_recall_at_20_effort": "R@20%E"}
    table = leaderboard[list(cols)].rename(columns=cols)
    table["model"] = [f"{m} *" if m == best else m for m in table["model"]]
    print("\nLeaderboard (sorted by CV ROC-AUC on the training period; * = selected model)\n")
    print(table.to_string(index=False, float_format="{:.3f}".format))


# ---------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------


def run_training(df: pd.DataFrame, models_dir: str | Path = "models", reports_dir: str | Path = "reports",
                 trained_on: str = str(DEFAULT_DATASET), quick: bool = False, imbalance: str = "balanced",
                 seed: int = RANDOM_STATE, verbose: bool = True) -> dict:
    """Run the whole experiment and write the model bundle, reports and figures."""
    log = print if verbose else (lambda *args, **kwargs: None)
    cfg = (TrainConfig.quick if quick else TrainConfig)(imbalance=imbalance, random_state=seed)
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train, fit, val, test = train_val_test_split(df)
    split = describe_split(df, train, fit, val, test, cfg, trained_on)
    (reports_dir / "split_info.json").write_text(json.dumps(split, indent=2))
    log(f"Train {len(train)} commits (bug rate {split['train']['bug_rate']:.1%}) | "
        f"test {len(test)} commits (bug rate {split['test']['bug_rate']:.1%}) | "
        f"{'quick' if quick else 'full'} run, imbalance = {cfg.imbalance}")

    log("\n[1/4] Cross-validating and tuning every model ...")
    leaderboard, runs = compare_models(train, fit, val, test, cfg, log)
    best = select_best(leaderboard)
    run = runs[best]
    bundle = make_bundle(best, run["final"], run["threshold"], train, run["metrics"], trained_on,
                         cv_roc_auc=float(leaderboard.set_index("model").loc[best, "cv_roc_auc"]),
                         imbalance=cfg.imbalance, params=run["params"])
    bundle_path = save_bundle(bundle, models_dir / "best_model.joblib")
    leaderboard.to_csv(reports_dir / "model_comparison.csv", index=False)
    log(f"Saved {best} to {bundle_path}")

    log("\n[2/4] Class-imbalance study ...")
    imbalance_df = compare_imbalance(train, fit, val, test, cfg, log)
    imbalance_df.to_csv(reports_dir / "imbalance_comparison.csv", index=False)

    log("\n[3/4] Bias-variance curves ...")
    complexity = complexity_curve(train, cfg)
    learning = learning_curve(run["pipeline"], fit, val, cfg)

    log("\n[4/4] Figures and reports ...")
    top = [m for m in leaderboard["model"] if leaderboard.set_index("model").loc[m, "selectable"]][:5]
    y_test = test[LABEL_COLUMN].to_numpy()
    plot_roc(runs, top, y_test, fig_dir / "model_roc.png")
    plot_pr(runs, top, y_test, fig_dir / "model_pr.png")
    plot_confusion(y_test, run["test_scores"], run["threshold"], best, fig_dir / "model_confusion_matrix.png")
    plot_comparison(leaderboard, best, fig_dir / "model_comparison.png")
    plot_complexity(complexity, fig_dir / "model_complexity_curve.png")
    plot_learning(learning, best, fig_dir / "model_learning_curve.png")
    write_markdown(reports_dir / "model_comparison.md", leaderboard, imbalance_df, best, runs, split, complexity, cfg)

    if verbose:
        print_leaderboard(leaderboard, best)
        m = run["metrics"]
        print(f"\nSelected: {best} (CV ROC-AUC {bundle['cv_roc_auc']:.3f}), threshold {run['threshold']:.3f}")
        print(f"Test: ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | F1 {m['f1']:.3f} | "
              f"precision {m['precision']:.3f} | recall {m['recall']:.3f} | MCC {m['mcc']:.3f} | "
              f"Recall@20%Effort {m['recall_at_20_effort']:.3f}")
        print(f"Reports in {reports_dir}/, model in {bundle_path}  ({time.perf_counter() - started:.0f}s)")
    return {"leaderboard": leaderboard, "imbalance": imbalance_df, "bundle": bundle, "best": best,
            "split": split, "complexity": complexity, "learning": learning}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare code change risk models.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="mined dataset CSV")
    parser.add_argument("--models-dir", default="models", help="where best_model.joblib is written")
    parser.add_argument("--reports-dir", default="reports", help="where tables and figures are written")
    parser.add_argument("--quick", action="store_true", help="fewer CV folds and tuning candidates (fast check)")
    parser.add_argument("--synthetic", action="store_true", help="use generated data instead of --data")
    parser.add_argument("--imbalance", choices=IMBALANCE_STRATEGIES, default="balanced",
                        help="class-imbalance handling for the main comparison (default: balanced class weights)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    if args.synthetic:
        df, trained_on = make_synthetic_dataset(2000, seed=args.seed), "synthetic"
    else:
        df, trained_on = load_dataset(args.data), str(args.data)
    run_training(df, args.models_dir, args.reports_dir, trained_on=trained_on, quick=args.quick,
                 imbalance=args.imbalance, seed=args.seed)


if __name__ == "__main__":
    main()
