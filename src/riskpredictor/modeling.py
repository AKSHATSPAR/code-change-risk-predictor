"""Models, preprocessing, metrics and the saved-model bundle for the code change risk predictor.

The module is organised in five small sections:

1. Preprocessing - log(1 + x) on the skewed counts, then standardisation (14 columns in -> 14 out).
2. Model zoo     - one entry per algorithm we compare, plus search spaces for tuning.
3. Pipelines     - preprocessing + (optional SMOTE) + classifier, and the time-aware helpers.
4. Metrics       - ROC-AUC, PR-AUC, F1, ..., Recall@20%Effort, and decision-threshold selection.
5. Bundle        - the dict saved to ``models/best_model.joblib`` that the explainer and app load.

The experiment itself (which model wins, figures, reports) lives in ``riskpredictor.train``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN, time_split

RANDOM_STATE = 42
DEFAULT_BUNDLE_PATH = Path("models/best_model.joblib")
IMBALANCE_STRATEGIES = ("none", "balanced", "smote")


# ---------------------------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------------------------

# Counts and sizes are heavily right-skewed (most commits add a few lines, a handful add 10,000+),
# so they are compressed with log(1 + x). `entropy` is already in [0, 1] and `fix` is 0/1.
LOG_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in ("entropy", "fix")]
_LOG_IDX = [FEATURE_COLUMNS.index(c) for c in LOG_FEATURES]


def log_transform(X) -> np.ndarray:
    """Apply log1p to the skewed columns and return an (n, 14) array in FEATURE_COLUMNS order.

    Accepts a DataFrame (columns are picked by name) or an array already in FEATURE_COLUMNS order.
    """
    if isinstance(X, pd.DataFrame):
        X = X[FEATURE_COLUMNS]
    X = np.array(X, dtype=float)  # copy, so the caller's data is never modified
    X[:, _LOG_IDX] = np.log1p(np.clip(X[:, _LOG_IDX], 0, None))
    return X


class LogScaler(TransformerMixin, BaseEstimator):
    """The preprocessing step: log1p on the skewed columns, then StandardScaler on all 14.

    Output column i is always feature ``FEATURE_COLUMNS[i]`` (a 1:1 mapping), so SHAP / LIME
    explanations computed on the transformed data map straight back to the original features.
    """

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(log_transform(X))
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    def transform(self, X) -> np.ndarray:
        return self.scaler_.transform(log_transform(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_COLUMNS, dtype=object)


def make_preprocessor() -> LogScaler:
    return LogScaler()


# ---------------------------------------------------------------------------------------------
# 2. Model zoo
# ---------------------------------------------------------------------------------------------


class OneClassSVMRiskScorer(ClassifierMixin, BaseEstimator):
    """Anomaly-detection baseline: learn what *clean* commits look like, flag unusual ones.

    The One-Class SVM is fitted on the clean (label 0) commits only. Its risk score is
    ``-decision_function``: the further a commit lies outside the "normal" region, the riskier.
    It produces scores, not probabilities, so it is reported but never saved as the best model.
    """

    def __init__(self, nu: float = 0.1, gamma: str | float = "scale"):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.ocsvm_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma).fit(X[np.asarray(y) == 0])
        return self

    def decision_function(self, X) -> np.ndarray:
        return -self.ocsvm_.decision_function(X)

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)


@dataclass
class ModelSpec:
    """One algorithm in the comparison."""

    name: str
    estimator: BaseEstimator
    family: str  # rough kind of model, shown in the leaderboard
    search_space: dict = field(default_factory=dict)  # RandomizedSearchCV distributions; empty = not tuned
    selectable: bool = True  # False for the anomaly baseline (no predict_proba)


def model_zoo(balanced: bool = True, pos_weight: float = 1.0, random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """All candidate models, keyed by display name.

    balanced:   use class_weight="balanced" (scale_pos_weight for XGBoost) where the model supports it.
                Naive Bayes, KNN, MLP and AdaBoost have no such option and are left as they are.
    pos_weight: #clean / #buggy in the training data, used by XGBoost when ``balanced`` is True.

    Perceptron and the SVMs output scores, not probabilities, so they are wrapped in
    CalibratedClassifierCV, which fits a sigmoid (Platt scaling) to turn scores into probabilities.
    """
    cw = "balanced" if balanced else None
    rs = random_state

    def calibrated(model):
        return CalibratedClassifierCV(model, method="sigmoid", cv=3)

    specs = [
        ModelSpec("Naive Bayes", GaussianNB(), "Probabilistic"),
        ModelSpec("Decision Tree (CART, gini)",
                  DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, class_weight=cw, random_state=rs),
                  "Tree"),
        ModelSpec("Decision Tree (ID3-style, entropy)",
                  DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=10, class_weight=cw, random_state=rs),
                  "Tree"),
        ModelSpec("KNN", KNeighborsClassifier(n_neighbors=25), "Instance-based"),
        ModelSpec("Logistic Regression", LogisticRegression(C=1.0, max_iter=2000, class_weight=cw), "Linear",
                  {"clf__C": loguniform(1e-3, 1e2)}),
        ModelSpec("Perceptron", calibrated(Perceptron(class_weight=cw, random_state=rs)), "Linear"),
        ModelSpec("SVM (linear)", calibrated(LinearSVC(C=0.1, class_weight=cw, max_iter=5000, random_state=rs)), "Linear"),
        ModelSpec("SVM (RBF)", calibrated(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw)), "Kernel",
                  {"clf__estimator__C": loguniform(0.1, 100), "clf__estimator__gamma": loguniform(1e-3, 1)}),
        ModelSpec("MLP (neural network)",
                  MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=1000, random_state=rs),
                  "Neural network"),
        ModelSpec("Random Forest",
                  RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=cw, n_jobs=1, random_state=rs),
                  "Ensemble (bagging)",
                  {"clf__n_estimators": [200, 300, 500],
                   "clf__max_depth": [None, 4, 6, 8, 12, 16],
                   "clf__min_samples_leaf": [1, 2, 5, 10, 20],
                   "clf__max_features": ["sqrt", 0.5, 1.0]}),
        ModelSpec("AdaBoost", AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=rs), "Ensemble (boosting)"),
        ModelSpec("XGBoost",
                  XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pos_weight if balanced else 1.0, eval_metric="logloss",
                                n_jobs=1, random_state=rs),
                  "Ensemble (boosting)",
                  {"clf__n_estimators": randint(100, 600),
                   "clf__max_depth": randint(2, 7),
                   "clf__learning_rate": loguniform(0.01, 0.3),
                   "clf__subsample": uniform(0.6, 0.4),
                   "clf__colsample_bytree": uniform(0.5, 0.5),
                   "clf__min_child_weight": randint(1, 10),
                   "clf__reg_lambda": loguniform(0.1, 10)}),
        ModelSpec("One-Class SVM (anomaly baseline)", OneClassSVMRiskScorer(nu=0.1), "Anomaly detection", selectable=False),
    ]
    return {spec.name: spec for spec in specs}


# ---------------------------------------------------------------------------------------------
# 3. Pipelines and time-aware helpers
# ---------------------------------------------------------------------------------------------


def make_pipeline(clf: BaseEstimator, smote: bool = False, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """prep -> [SMOTE] -> clf.

    imblearn's Pipeline applies SMOTE only inside ``fit``: synthetic buggy commits are added to the
    (already transformed) training data, while validation/test data is never resampled. Inside
    cross-validation this means SMOTE only ever sees the training folds.
    """
    steps = [("prep", make_preprocessor())]
    if smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def to_sklearn_pipeline(fitted: ImbPipeline) -> Pipeline:
    """Drop the sampler: the saved model is just the fitted ("prep", "clf") pair."""
    return Pipeline([("prep", fitted.named_steps["prep"]), ("clf", fitted.named_steps["clf"])])


def train_val_test_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Time-ordered split into train / fit / validation / test.

    train = oldest 80% of commits, test = newest 20% (``features.time_split``).
    Inside train, the newest 20% is the *validation* slice used to pick decision thresholds,
    and the older part is called *fit*. So: fit < validation < test in time, and train = fit + validation.
    """
    train, test = time_split(df, test_size)
    fit, val = time_split(train, val_size)
    return train, fit, val, test


def risk_scores(model, X) -> np.ndarray:
    """Probability of "buggy" when available, otherwise the model's decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def fit_and_choose_threshold(model, fit_df: pd.DataFrame, val_df: pd.DataFrame, train_df: pd.DataFrame):
    """Pick a decision threshold without touching the test set, then refit on all training data.

    1. fit on the older part of the training period, score the validation slice,
       choose the threshold that maximises F1 there;
    2. refit a fresh copy on the whole training period (fit + validation), so the final model
       also learns from the most recent training commits.
    """
    first = clone(model).fit(fit_df[FEATURE_COLUMNS], fit_df[LABEL_COLUMN])
    threshold = best_f1_threshold(val_df[LABEL_COLUMN], risk_scores(first, val_df[FEATURE_COLUMNS]))
    final = clone(model).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    return final, threshold


# ---------------------------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------------------------

METRIC_INFO: dict[str, str] = {
    "roc_auc": "ROC-AUC: chance that a random buggy commit gets a higher risk score than a random clean one (0.5 = guessing).",
    "pr_auc": "PR-AUC (average precision): area under the precision-recall curve; guessing scores the bug rate.",
    "f1": "F1: harmonic mean of precision and recall at the chosen threshold.",
    "precision": "Precision: of the commits we flag as risky, the fraction that really are buggy.",
    "recall": "Recall: of the truly buggy commits, the fraction we flag.",
    "mcc": "MCC: correlation between predictions and truth, -1..1; stays honest on imbalanced data.",
    "balanced_accuracy": "Balanced accuracy: average of recall on buggy commits and recall on clean commits.",
    "recall_at_20_effort": "Recall@20%Effort: share of buggy commits found when reviewing the riskiest commits first "
                           "until 20% of all changed lines (la + ld) have been read.",
}


def churn(df: pd.DataFrame) -> np.ndarray:
    """Review effort of each commit: lines added + lines deleted."""
    return (df["la"] + df["ld"]).to_numpy(dtype=float)


def recall_at_effort(y_true, scores, effort, budget: float = 0.2) -> float:
    """Effort-aware recall (Kamei et al. 2013).

    Sort commits by predicted risk (highest first) and "review" them in that order while the
    reviewed lines stay within ``budget`` (20%) of all changed lines. Return the fraction of all
    buggy commits that were reviewed. A model that ranks small risky commits first scores well.
    """
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="stable")
    reviewed = np.cumsum(np.asarray(effort, dtype=float)[order]) <= budget * np.sum(effort)
    return float(y_true[order][reviewed].sum() / y_true.sum())


def best_f1_threshold(y_true, scores) -> float:
    """Threshold on ``scores`` that gives the highest F1 (computed on a validation set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true, scores, threshold: float, effort) -> dict[str, float]:
    """All test metrics (see METRIC_INFO). ROC-AUC, PR-AUC and Recall@20%Effort use the raw scores;
    the others use the yes/no predictions ``scores >= threshold``."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall_at_20_effort": recall_at_effort(y_true, scores, effort, budget=0.2),
    }


# ---------------------------------------------------------------------------------------------
# 5. The saved bundle (contract shared with explain.py, scoring.py and the app)
# ---------------------------------------------------------------------------------------------

BUNDLE_KEYS = ("model_name", "pipeline", "features", "threshold", "background", "metrics",
               "train_end_date", "trained_on")


def make_bundle(model_name: str, fitted: ImbPipeline, threshold: float, train_df: pd.DataFrame,
                metrics: dict, trained_on: str, **extra) -> dict:
    """Package a fitted model for the explainer and the app.

    ``background`` is a fixed-seed sample of at most 200 raw training rows, used as the
    reference data for SHAP / LIME.
    """
    background = train_df[FEATURE_COLUMNS].sample(n=min(200, len(train_df)), random_state=RANDOM_STATE)
    return {
        "model_name": model_name,
        "pipeline": to_sklearn_pipeline(fitted),
        "features": list(FEATURE_COLUMNS),
        "threshold": float(threshold),
        "background": background.reset_index(drop=True),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "train_end_date": pd.Timestamp(train_df[DATE_COLUMN].max()).isoformat(),
        "trained_on": str(trained_on),
        **extra,
    }


def save_bundle(bundle: dict, path: str | Path = DEFAULT_BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path = DEFAULT_BUNDLE_PATH) -> dict:
    """Load the dict saved by ``riskpredictor.train`` and check it has the agreed keys."""
    bundle = joblib.load(path)
    missing = [k for k in BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"{path} is not a model bundle; missing keys: {missing}")
    return bundle


def predict_risk(bundle: dict, X) -> np.ndarray:
    """Probability that each commit is buggy.

    X can be a DataFrame (extra columns are ignored), a Series for one commit, or an array whose
    columns are in FEATURE_COLUMNS order.
    """
    features = bundle["features"]
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    if isinstance(X, pd.DataFrame):
        X = X[features].astype(float)
    else:
        X = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)), columns=features)
    return bundle["pipeline"].predict_proba(X)[:, 1]
