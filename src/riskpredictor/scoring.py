"""Turn a trained model into risk scores for real commits.

This is the bridge between the ML pipeline and the applications (CLI and Streamlit app):

* ``score_commit``    - mine one commit from a local git repo and explain its risk.
* ``score_dataframe`` - score many already-mined commits at once (vectorised).
* ``test_period``     - the commits the model never saw during training.
* ``build_demo_bundle`` - a small model trained in seconds, so the app still works
  before the real training pipeline has been run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN, time_split


def risk_level(risk: float, threshold: float) -> str:
    """Map a probability to Low / Medium / High.

    High   = at or above the model's decision threshold (the commit is flagged as risky)
    Medium = at least half the threshold (worth a second look)
    Low    = everything else
    """
    if risk >= threshold:
        return "High"
    if risk >= threshold / 2:
        return "Medium"
    return "Low"


def score_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``risk`` (probability of being buggy) and ``risk_level`` columns."""
    out = df.copy()
    X = out[FEATURE_COLUMNS].astype(float)
    out["risk"] = bundle["pipeline"].predict_proba(X)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))
    out["risk_level"] = [risk_level(r, threshold) for r in out["risk"]]
    return out


def score_commit(repo_path: str, commit_hash: str, bundle: dict):
    """Mine one commit from a local repository and explain its predicted risk.

    Returns ``(row, explanation)`` where ``row`` holds the mined metadata + 14 change metrics
    and ``explanation`` is a ``riskpredictor.explain.CommitExplanation``.
    Mining walks the history up to the commit, so this can take a minute on large repos.
    """
    from riskpredictor.explain import explain_commit  # imported here: heavy (shap)
    from riskpredictor.mining import extract_commit_features  # imported here: heavy (pydriller)

    row = pd.Series(extract_commit_features(repo_path, commit_hash))
    row[FEATURE_COLUMNS] = row[FEATURE_COLUMNS].astype(float)
    return row, explain_commit(bundle, row)


def test_period(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Commits after the model's training period (falls back to the last 20 % of commits)."""
    end = (bundle or {}).get("train_end_date")
    if end is not None:
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        later = df[pd.to_datetime(df[DATE_COLUMN], utc=True) > end]
        if len(later):
            return later.reset_index(drop=True)
    return time_split(df)[1]


def top_risky(bundle: dict, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` riskiest commits of the test period, highest risk first."""
    scored = score_dataframe(bundle, test_period(df, bundle))
    return scored.sort_values("risk", ascending=False).head(n).reset_index(drop=True)


def build_demo_bundle(df: pd.DataFrame, trained_on: str = "synthetic demo data") -> dict:
    """Train a small XGBoost model in a few seconds, packaged like ``models/best_model.joblib``.

    Only used as a fallback by the app and tests; the real model comes from ``riskpredictor.train``.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    train, _ = time_split(df)
    fit_part, val_part = time_split(train, test_size=0.2)  # validation slice to pick the threshold

    def make_pipeline() -> Pipeline:
        return Pipeline([
            ("prep", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("clf", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                  subsample=0.9, eval_metric="logloss", random_state=0)),
        ])

    # 1) choose the decision threshold on data the model did not see
    pipe = make_pipeline().fit(fit_part[FEATURE_COLUMNS], fit_part[LABEL_COLUMN])
    val_proba = pipe.predict_proba(val_part[FEATURE_COLUMNS])[:, 1]
    candidates = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(val_part[LABEL_COLUMN], val_proba >= t, zero_division=0) for t in candidates]
    threshold = float(candidates[int(np.argmax(f1s))])
    metrics = {
        "roc_auc": float(roc_auc_score(val_part[LABEL_COLUMN], val_proba)),
        "pr_auc": float(average_precision_score(val_part[LABEL_COLUMN], val_proba)),
        "f1": float(max(f1s)),
    }

    # 2) refit on the whole training period
    pipe = make_pipeline().fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return {
        "model_name": "XGBoost (demo)",
        "pipeline": pipe,
        "features": list(FEATURE_COLUMNS),
        "threshold": threshold,
        "background": train[FEATURE_COLUMNS].sample(min(200, len(train)), random_state=0).reset_index(drop=True),
        "metrics": metrics,
        "train_end_date": str(train[DATE_COLUMN].max()),
        "trained_on": trained_on,
    }
