"""Shared contract for the whole project: feature names, dataset I/O and the train/test split.

Every other module (mining, modeling, explainability, app) imports its column names from here,
so the dataset schema is defined in exactly one place.

The 14 change metrics follow Kamei et al. (2013), "A Large-Scale Empirical Study of
Just-in-Time Quality Assurance", IEEE Transactions on Software Engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureInfo:
    family: str
    label: str  # short human-readable name for plots / UI
    description: str  # one sentence used in plain-English explanations


FEATURE_INFO: dict[str, FeatureInfo] = {
    # Diffusion: how spread out the change is
    "ns": FeatureInfo("Diffusion", "Subsystems touched", "number of top-level subsystems (directories) modified"),
    "nd": FeatureInfo("Diffusion", "Directories touched", "number of directories modified"),
    "nf": FeatureInfo("Diffusion", "Files touched", "number of code files modified"),
    "entropy": FeatureInfo("Diffusion", "Change spread (entropy)", "how evenly the modified lines are spread across files (0 = one file, 1 = evenly spread)"),
    # Size: how big the change is
    "la": FeatureInfo("Size", "Lines added", "lines of code added"),
    "ld": FeatureInfo("Size", "Lines deleted", "lines of code deleted"),
    "lt": FeatureInfo("Size", "File size before change", "average lines of code in the modified files before the change"),
    # Purpose
    "fix": FeatureInfo("Purpose", "Is a bug fix", "whether the commit message says it fixes a bug (1 = yes)"),
    # History of the modified files
    "ndev": FeatureInfo("History", "Past developers on files", "number of distinct developers who previously changed these files"),
    "age": FeatureInfo("History", "Days since files last changed", "average days since the modified files were last changed"),
    "nuc": FeatureInfo("History", "Past changes to files", "number of unique earlier commits that changed these files"),
    # Experience of the author
    "exp": FeatureInfo("Experience", "Author experience", "number of earlier commits by this author"),
    "rexp": FeatureInfo("Experience", "Author recent experience", "earlier commits by this author, weighted towards recent years"),
    "sexp": FeatureInfo("Experience", "Author subsystem experience", "earlier commits by this author in the same subsystems"),
}

FEATURE_COLUMNS: list[str] = list(FEATURE_INFO)
LABEL_COLUMN = "buggy"  # 1 = this commit later turned out to introduce a bug (SZZ), 0 = clean
DATE_COLUMN = "author_date"
META_COLUMNS: list[str] = ["repo", "commit_hash", "author_id", DATE_COLUMN, "message"]
ALL_COLUMNS: list[str] = META_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]

DEFAULT_DATASET = Path("data/processed/dataset.csv")


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load a mined dataset, parse dates and sort commits from oldest to newest."""
    df = pd.read_csv(path)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], utc=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: train on the oldest commits, test on the newest.

    A random split would leak the future into training (a model could learn from commits made
    after the ones it is tested on), which never happens when the predictor is used for real.
    """
    df = df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def make_synthetic_dataset(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Realistic-looking fake dataset with the real schema, for tests and development only."""
    rng = np.random.default_rng(seed)
    nf = rng.geometric(0.45, n).astype(float)
    nd = np.minimum(nf, rng.geometric(0.6, n)).astype(float)
    ns = np.minimum(nd, rng.geometric(0.75, n)).astype(float)
    la = np.round(rng.lognormal(3.0, 1.4, n) * np.sqrt(nf))
    ld = np.round(la * rng.beta(1.2, 2.5, n))
    lt = np.round(rng.lognormal(5.3, 0.9, n))
    entropy = np.where(nf > 1, rng.beta(2, 1.5, n), 0.0)
    fix = (rng.random(n) < 0.3).astype(float)
    exp = np.round(rng.lognormal(3.5, 1.6, n))
    rexp = np.round(exp * rng.uniform(0.1, 0.8, n), 2)
    sexp = np.round(exp * rng.uniform(0.2, 1.0, n))
    nuc = np.round(rng.lognormal(2.5, 1.2, n))
    ndev = np.minimum(nuc, np.round(rng.lognormal(1.2, 0.7, n)))
    age = np.round(rng.lognormal(3.0, 1.5, n), 2)

    # Risk grows with size, spread, fixing and file churn; falls with author experience.
    logit = (
        -2.6
        + 0.45 * np.log1p(la + ld)
        + 0.35 * np.log1p(nf)
        + 0.6 * entropy
        + 0.5 * fix
        + 0.15 * np.log1p(ndev)
        - 0.25 * np.log1p(exp)
        + rng.normal(0, 0.6, n)
    )
    buggy = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    dates = pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 3000, n)), unit="D")
    df = pd.DataFrame(
        {
            "repo": "synthetic",
            "commit_hash": [f"{rng.integers(0, 2**63):016x}{i:024x}"[:40] for i in range(n)],
            "author_id": [f"a{a:03d}" for a in rng.integers(0, 40, n)],
            DATE_COLUMN: dates,
            "message": np.where(fix == 1, "Fix bug in handler", "Add feature"),
            "ns": ns, "nd": nd, "nf": nf, "entropy": np.round(entropy, 4),
            "la": la, "ld": ld, "lt": lt, "fix": fix,
            "ndev": ndev, "age": age, "nuc": nuc,
            "exp": exp, "rexp": rexp, "sexp": sexp,
            LABEL_COLUMN: buggy,
        }
    )
    return df[ALL_COLUMNS]
