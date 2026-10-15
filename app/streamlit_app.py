"""Streamlit app: the real-time face of the code change risk predictor.

Run it from the project root with::

    uv run streamlit run app/streamlit_app.py

Pages (sidebar):
    Overview, Model Leaderboard, Explain a Commit, What-If Simulator,
    Score a Local Commit, Global Explanations.

If the trained model or the dataset is missing, the app explains how to create them and falls back to a
small demo model (trained in seconds) so every page still works.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "import charts" works from any cwd

import charts  # noqa: E402

from riskpredictor.features import (  # noqa: E402
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    LABEL_COLUMN,
    load_dataset,
    make_synthetic_dataset,
)
from riskpredictor.scoring import (  # noqa: E402
    build_demo_bundle,
    risk_level,
    score_commit,
    score_dataframe,
    test_period,
)

# ----------------------------------------------------------------------------------------------
# Paths (relative paths are resolved against the project root, so the app works from any folder).
# Environment variables let the tests point the app at missing files to exercise the fallbacks.
# ----------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def _path(env_var: str, default: str) -> Path:
    path = Path(os.environ.get(env_var, default))
    return path if path.is_absolute() else ROOT / path


DATASET_PATH = _path("RISKPREDICTOR_DATA", "data/processed/dataset.csv")
BUNDLE_PATH = _path("RISKPREDICTOR_BUNDLE", "models/best_model.joblib")
REPORTS_DIR = _path("RISKPREDICTOR_REPORTS", "reports")
DEFAULT_REPO = "data/raw/repos/requests"

BUILD_CMD = "uv run python -m riskpredictor.build_dataset --repo data/raw/repos/requests"
TRAIN_CMD = "uv run python -m riskpredictor.train"
EXPLAIN_CMD = "uv run python -m riskpredictor.explain --global"

# Features that only take whole-number values (the rest are ratios / averages).
CONTINUOUS_FEATURES = {"entropy", "rexp", "age", "lt"}
FEATURE_FAMILIES = ["Size", "Diffusion", "Purpose", "History", "Experience"]

st.set_page_config(page_title="Code Change Risk Predictor", page_icon=":material/bug_report:", layout="wide")


# ----------------------------------------------------------------------------------------------
# Cached loading
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(path: str) -> tuple[pd.DataFrame, bool]:
    """Return ``(dataset, is_real)``. Falls back to synthetic data if no mined dataset exists."""
    if Path(path).exists():
        return load_dataset(path), True
    return make_synthetic_dataset(n=2000, seed=0), False


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(bundle_path: str, dataset_path: str) -> tuple[dict, bool]:
    """Return ``(bundle, is_trained_model)``. Falls back to a quick demo model if none was trained."""
    if Path(bundle_path).exists():
        return joblib.load(bundle_path), True
    df, is_real = get_dataset(dataset_path)
    trained_on = "mined dataset (demo model)" if is_real else "synthetic demo data"
    return build_demo_bundle(df, trained_on=trained_on), False


def bundle_key(bundle: dict) -> str:
    """A short string that identifies a bundle, used as part of cache keys."""
    return f"{bundle.get('model_name')}|{bundle.get('trained_on')}|{bundle.get('train_end_date')}"


@st.cache_data(show_spinner=False)
def scored_test_period(_bundle: dict, key: str, df: pd.DataFrame) -> pd.DataFrame:
    """All test-period commits with their predicted risk, riskiest first."""
    scored = score_dataframe(_bundle, test_period(df, _bundle))
    return scored.sort_values("risk", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Computing SHAP explanation...")
def explain_values(_bundle: dict, key: str, values: tuple[float, ...]):
    from riskpredictor.explain import explain_commit

    return explain_commit(_bundle, pd.Series(values, index=FEATURE_COLUMNS, dtype=float))


@st.cache_data(show_spinner="Running LIME...")
def lime_values(_bundle: dict, key: str, values: tuple[float, ...]) -> list[tuple[str, float]]:
    from riskpredictor.explain import explain_commit_lime

    return explain_commit_lime(_bundle, pd.Series(values, index=FEATURE_COLUMNS, dtype=float))


@st.cache_data(show_spinner="Computing global SHAP importance...")
def global_importance_cached(_bundle: dict, key: str, X: pd.DataFrame) -> pd.DataFrame:
    from riskpredictor.explain import global_importance

    return global_importance(_bundle, X, max_samples=300)


def feature_values(row: pd.Series) -> tuple[float, ...]:
    """The 14 features of a commit as a hashable tuple (used as a cache key)."""
    return tuple(float(row[f]) for f in FEATURE_COLUMNS)


def explain_available() -> bool:
    try:
        import riskpredictor.explain  # noqa: F401
    except ImportError:
        st.warning("The explainability module (`riskpredictor/explain.py`) is not available yet, "
                   "so commits can be scored but not explained.")
        return False
    return True


# ----------------------------------------------------------------------------------------------
# Small display helpers
# ----------------------------------------------------------------------------------------------
def first_line(message, width: int = 70) -> str:
    text = str(message).strip().splitlines()[0] if str(message).strip() else "(no message)"
    return text if len(text) <= width else text[: width - 1] + "…"


def commit_label(row: pd.Series) -> str:
    return f"{row['commit_hash'][:8]}  ·  {row['risk']:.0%}  ·  {first_line(row['message'], 60)}"


def fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


LEVEL_BADGE = {"Low": "green", "Medium": "orange", "High": "red"}


def show_explanation(bundle: dict, row: pd.Series, df: pd.DataFrame, reference: float | None = None,
                     key: str = "main") -> None:
    """The full explanation view shared by the 'Explain', 'What-If' and 'Score a Local Commit' pages."""
    explanation = explain_values(bundle, bundle_key(bundle), feature_values(row))
    threshold = float(bundle.get("threshold", 0.5))

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.plotly_chart(charts.risk_gauge(explanation.risk, threshold, reference), width="stretch",
                        key=f"gauge_{key}")
    with right:
        st.badge(f"{explanation.risk_level} risk", color=LEVEL_BADGE.get(explanation.risk_level, "gray"))
        verdict = "flagged as **risky**" if explanation.is_risky else "**not flagged**"
        st.markdown(f"Predicted probability of introducing a bug: **{explanation.risk:.1%}**, so it is {verdict} "
                    f"(the model flags commits at or above **{threshold:.0%}**).")
        st.markdown("**Why?**")
        st.markdown("\n".join(f"- {reason}" for reason in explanation.top_reasons(3)))
        lowering = explanation.lowering_reasons(2)
        if lowering:
            st.markdown("**What keeps it lower**")
            st.markdown("\n".join(f"- {reason}" for reason in lowering))
        if LABEL_COLUMN in row and pd.notna(row[LABEL_COLUMN]):
            if int(row[LABEL_COLUMN]) == 1:
                st.error("Ground truth (SZZ): this commit **did** later turn out to introduce a bug.",
                         icon=":material/bug_report:")
            else:
                st.success("Ground truth (SZZ): no later bug fix was traced back to this commit.",
                           icon=":material/check_circle:")
            st.caption("The SZZ label is shown for comparison only. The model never sees it when scoring.")

    st.plotly_chart(charts.shap_bar(explanation.contributions), width="stretch", key=f"shap_{key}")
    st.caption(f"The model starts from the average risk of past commits ({explanation.base_value:.1%}); each bar "
               f"adds or removes percentage points, and together they sum to this commit's risk.")

    with st.expander("LIME explanation (a second, model-agnostic opinion)"):
        st.write("LIME fits a simple linear model around this commit by perturbing its features. "
                 "If SHAP and LIME agree on the top features, the explanation is more trustworthy.")
        if st.checkbox("Compute LIME weights (takes a few seconds)", key=f"lime_{key}"):
            weights = lime_values(bundle, bundle_key(bundle), feature_values(row))
            st.dataframe(pd.DataFrame(weights, columns=["Rule", "Weight"]).style.format({"Weight": "{:+.3f}"})
                         .map(lambda w: f"color: {charts.RAISES_RISK if w > 0 else charts.LOWERS_RISK}",
                              subset=["Weight"]),
                         hide_index=True, width="stretch")

    with st.expander("Feature values compared with a typical commit", expanded=False):
        medians = df[FEATURE_COLUMNS].median()
        table = pd.DataFrame({
            "Feature": [FEATURE_INFO[f].label for f in FEATURE_COLUMNS],
            "This commit": [fmt(row[f]) for f in FEATURE_COLUMNS],
            "Dataset median": [fmt(medians[f]) for f in FEATURE_COLUMNS],
            "Higher than % of commits": [f"{(df[f] < row[f]).mean():.0%}" for f in FEATURE_COLUMNS],
            "Meaning": [FEATURE_INFO[f].description for f in FEATURE_COLUMNS],
        })
        st.dataframe(table, hide_index=True, width="stretch")


def missing_model_notice(is_trained: bool, is_real: bool) -> None:
    """Explain how to create the missing artifacts (shown at the top of pages that need them)."""
    if not is_real:
        st.info(f"No mined dataset found, so the app is using **synthetic demo data**. Create the real one with:\n\n"
                f"```\n{BUILD_CMD}\n```", icon=":material/info:")
    if not is_trained:
        st.info(f"No trained model found at `{BUNDLE_PATH.relative_to(ROOT) if BUNDLE_PATH.is_relative_to(ROOT) else BUNDLE_PATH}`, "
                f"so the app is using a **quick demo XGBoost model**. Train the real one with:\n\n"
                f"```\n{TRAIN_CMD}\n```", icon=":material/info:")


def commit_picker(scored: pd.DataFrame, key: str) -> pd.Series:
    """Searchable select box over the test-period commits (type a hash or words of the message)."""
    index = st.selectbox(
        "Commit from the test period (riskiest first, type to search by hash or message)",
        options=range(len(scored)), format_func=lambda i: commit_label(scored.iloc[i]), key=key,
    )
    return scored.iloc[index]


# ----------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, is_real: bool, bundle: dict) -> None:
    st.title("Code Change Risk Predictor")
    st.markdown(
        "Every commit gets a **probability of introducing a bug**, computed from 14 change metrics "
        "(size, spread, history and author experience, from Kamei et al. 2013), plus a plain-English "
        "**explanation** of what drove the prediction. Labels come from the **SZZ algorithm**: a commit is "
        "*buggy* if a later bug-fix commit changed lines it wrote."
    )
    if not is_real:
        st.info(f"Showing **synthetic demo data**. Mine the real dataset with:\n\n```\n{BUILD_CMD}\n```",
                icon=":material/info:")

    dates = pd.to_datetime(df[DATE_COLUMN], utc=True)
    cols = st.columns(4)
    cols[0].metric("Commits", f"{len(df):,}")
    cols[1].metric("Bug-introducing", f"{df[LABEL_COLUMN].mean():.1%}", help="Share of commits labelled buggy by SZZ")
    cols[2].metric("Period", f"{dates.min():%Y}-{dates.max():%y}", help=f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}")
    cols[3].metric("Authors", f"{df['author_id'].nunique():,}")

    st.plotly_chart(charts.bug_rate_over_time(df), width="stretch")
    st.caption("Recent months look cleaner partly because their bugs have not been found and fixed yet "
               "and that is one reason the model is tested on the newest commits.")

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(charts.feature_difference(df), width="stretch")
    with right:
        feature = st.selectbox("Compare one feature", FEATURE_COLUMNS, format_func=lambda f: FEATURE_INFO[f].label)
        log_scale = st.toggle("Log scale", value=feature not in {"entropy", "fix"})
        st.plotly_chart(charts.feature_box(df, feature, log_scale), width="stretch")

    with st.expander("What do the 14 features mean?"):
        st.dataframe(pd.DataFrame([
            {"Feature": f, "Name": info.label, "Family": info.family, "Meaning": info.description}
            for f, info in FEATURE_INFO.items()
        ]), hide_index=True, width="stretch")


# Friendly names for the columns of reports/model_comparison.csv (others are hidden from the table).
METRIC_NAMES = {
    "cv_roc_auc": "CV ROC-AUC", "cv_roc_auc_std": "CV std",
    "test_roc_auc": "ROC-AUC", "test_pr_auc": "PR-AUC", "test_f1": "F1",
    "test_precision": "Precision", "test_recall": "Recall", "test_mcc": "MCC",
    "test_balanced_accuracy": "Balanced acc.", "test_recall_at_20_effort": "Recall@20% effort",
    "threshold": "Threshold",
}
CHART_METRICS = ["ROC-AUC", "PR-AUC", "F1", "MCC", "Recall@20% effort"]


def page_leaderboard(bundle: dict, is_trained: bool) -> None:
    st.title("Model Leaderboard")
    st.markdown(
        "Every classifier was tuned and compared with **time-aware cross-validation on the training period** "
        "(oldest 80 % of commits) and ranked by **CV ROC-AUC**. The winner was then evaluated on the "
        "**newest 20 %** of commits, which no model saw during training or selection. That is the same situation "
        "as scoring tomorrow's commits. Test-set columns are shown for every model for comparison only."
    )

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        st.info(f"No model comparison found at `reports/model_comparison.csv`. Create it with:\n\n"
                f"```\n{TRAIN_CMD}\n```", icon=":material/info:")
        st.subheader(f"Current model: {bundle.get('model_name', 'unknown')}")
        st.json(bundle.get("metrics", {}))
        return

    comparison = pd.read_csv(comparison_path)
    name_col = "model" if "model" in comparison.columns else comparison.columns[0]
    if "rank" in comparison.columns:
        comparison = comparison.sort_values("rank").reset_index(drop=True)
    names = list(comparison[name_col])
    best_name = bundle.get("model_name") if bundle.get("model_name") in names else names[0]

    st.success(f"Selected model: **{best_name}**" + ("" if is_trained else " (but the app is running a demo model)"),
               icon=":material/emoji_events:")
    if is_trained:
        m = bundle.get("metrics", {})
        cols = st.columns(4)
        for col, (key, label) in zip(cols, [("roc_auc", "Test ROC-AUC"), ("pr_auc", "Test PR-AUC"),
                                            ("recall", "Test recall"), ("precision", "Test precision")]):
            if key in m:
                col.metric(label, f"{m[key]:.3f}")

    text_cols = [c for c in ["rank", name_col, "family", "tuning"] if c in comparison.columns]
    metric_cols = [c for c in METRIC_NAMES if c in comparison.columns]
    if not metric_cols:  # unknown format: show every numeric column
        metric_cols = [c for c in comparison.columns if c not in text_cols and pd.api.types.is_numeric_dtype(comparison[c])]
    table = comparison[text_cols + metric_cols].rename(
        columns={**METRIC_NAMES, "rank": "Rank", name_col: "Model", "family": "Type", "tuning": "Tuning"})
    if "Tuning" in table.columns:
        table["Tuning"] = table["Tuning"].fillna("default")
    shown_metrics = [METRIC_NAMES.get(c, c) for c in metric_cols]

    def highlight(row: pd.Series) -> list[str]:
        style = "background-color: rgba(42,120,214,0.18); font-weight: 600" if row["Model"] == best_name else ""
        return [style] * len(row)

    st.dataframe(table.style.apply(highlight, axis=1).format({c: "{:.3f}" for c in shown_metrics}, na_rep="-"),
                 hide_index=True, width="stretch")
    if "note" in comparison.columns and comparison["note"].notna().any():
        for _, r in comparison[comparison["note"].notna()].iterrows():
            st.caption(f"**{r[name_col]}**: {r['note']}")

    chart_metrics = [m for m in CHART_METRICS if m in table.columns]
    if chart_metrics:
        st.plotly_chart(charts.metric_bars(table, "Model", chart_metrics), width="stretch")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **ROC-AUC**: the chance a random buggy commit is ranked above a random clean one (0.5 = guessing).\n"
            "- **PR-AUC**: the precision/recall trade-off. It is more informative than ROC-AUC when bugs are rare.\n"
            "- **Precision / Recall / F1**: of the flagged commits, how many were buggy / of the buggy commits, "
            "how many were flagged / their harmonic mean.\n"
            "- **MCC**: a balanced correlation between predictions and truth (-1 to 1).\n"
            "- **Recall@20% effort**: the share of bugs caught by reviewing the riskiest commits until 20 % of all "
            "changed lines have been read (rewards models that find bugs in *small* commits).\n"
            "- **Threshold**: the probability above which a commit is flagged, chosen on a validation slice."
        )

    split_path = REPORTS_DIR / "split_info.json"
    if split_path.exists():
        with st.expander("Train / test split"):
            split = json.loads(split_path.read_text())
            parts = [k for k in ["train", "fit", "validation", "test"] if isinstance(split.get(k), dict)]
            if parts:
                st.dataframe(pd.DataFrame([
                    {"Part": k, "Commits": split[k].get("n"), "From": str(split[k].get("start", ""))[:10],
                     "To": str(split[k].get("end", ""))[:10], "Bug rate": f"{split[k].get('bug_rate', 0):.1%}"}
                    for k in parts
                ]), hide_index=True, width="stretch")
            for key in ["explanation", "cv", "tuning"]:
                if key in split:
                    st.caption(f"**{key.title()}:** {split[key]}")

    imbalance_path = REPORTS_DIR / "imbalance_comparison.csv"
    if imbalance_path.exists():
        with st.expander("Handling class imbalance (buggy commits are the minority)"):
            imbalance = pd.read_csv(imbalance_path)
            numeric = [c for c in imbalance.columns if pd.api.types.is_numeric_dtype(imbalance[c])]
            st.dataframe(imbalance.style.format({c: "{:.3f}" for c in numeric}), hide_index=True, width="stretch")

    figures = sorted((REPORTS_DIR / "figures").glob("model_*.png"))
    if figures:
        st.subheader("Evaluation figures")
        tabs = st.tabs([f.stem.removeprefix("model_").replace("_", " ").capitalize() for f in figures])
        for tab, figure in zip(tabs, figures):
            tab.image(str(figure), width="stretch")


def page_explain(bundle: dict, df: pd.DataFrame) -> None:
    st.title("Explain a Commit")
    st.markdown("Pick a commit the model **never saw during training** and see why it was scored the way it was.")
    if not explain_available():
        return
    scored = scored_test_period(bundle, bundle_key(bundle), df)
    row = commit_picker(scored, key="explain_commit")
    st.caption(f"`{row['commit_hash']}` · {pd.Timestamp(row[DATE_COLUMN]):%d %b %Y} · "
               f"{first_line(row['message'], 200)}")
    show_explanation(bundle, row, df, key="explain")


def _reset_whatif(row: pd.Series) -> None:
    for f in FEATURE_COLUMNS:
        st.session_state[f"wi_{f}"] = float(row[f])


def suggestions(bundle: dict, values: dict[str, float], df: pd.DataFrame) -> pd.DataFrame:
    """Try a few simple, actionable changes to the commit and re-score each one."""
    def split(v: dict) -> dict:
        v = dict(v)
        for f in ["la", "ld"]:
            v[f] = np.floor(v[f] / 2)
        for f in ["nf", "nd", "ns"]:
            v[f] = max(1.0, np.ceil(v[f] / 2)) if v[f] > 0 else 0.0
        return v

    def experienced(v: dict) -> dict:
        v = dict(v)
        for f in ["exp", "rexp", "sexp"]:  # an author at the 75th percentile of experience
            v[f] = max(v[f], float(df[f].quantile(0.75)))
        return v

    scenarios = {
        "Current commit": values,
        "Split it into two smaller commits": split(values),
        "Have an experienced author make it": experienced(values),
        "Both": experienced(split(values)),
    }
    table = pd.DataFrame(list(scenarios.values()), index=list(scenarios))
    risks = bundle["pipeline"].predict_proba(table[FEATURE_COLUMNS].astype(float))[:, 1]
    return pd.DataFrame({"Scenario": list(scenarios), "Risk": risks, "Change": risks - risks[0]})


def page_whatif(bundle: dict, df: pd.DataFrame) -> None:
    st.title("What-If Simulator")
    st.markdown("Start from a real commit, change its metrics and watch the risk and the explanation update live.")
    st.warning("This is a **simulation**: it shows how the *model* reacts, not a guarantee that a "
               "real change would be safer.", icon=":material/science:")
    if not explain_available():
        return

    scored = scored_test_period(bundle, bundle_key(bundle), df)
    row = commit_picker(scored, key="whatif_commit")
    if st.session_state.get("wi_base") != row["commit_hash"]:
        st.session_state["wi_base"] = row["commit_hash"]
        _reset_whatif(row)
    st.button("Reset to the original values", on_click=_reset_whatif, args=(row,))

    # Widgets write to session_state before the rerun, so the live score can be shown above the inputs.
    values = {f: float(st.session_state[f"wi_{f}"]) for f in FEATURE_COLUMNS}
    sim_risk = float(bundle["pipeline"].predict_proba(pd.DataFrame([values])[FEATURE_COLUMNS])[:, 1][0])
    threshold = float(bundle.get("threshold", 0.5))
    cols = st.columns(3)
    cols[0].metric("Original risk", f"{row['risk']:.1%}")
    cols[1].metric("Simulated risk", f"{sim_risk:.1%}", delta=f"{(sim_risk - row['risk']) * 100:+.1f} pts",
                   delta_color="inverse")
    cols[2].metric("Simulated level", risk_level(sim_risk, threshold))

    with st.container(border=True):
        for family in FEATURE_FAMILIES:
            features = [f for f in FEATURE_COLUMNS if FEATURE_INFO[f].family == family]
            cols = st.columns(4)
            for i, f in enumerate(features):
                info = FEATURE_INFO[f]
                continuous = f in CONTINUOUS_FEATURES
                cols[i % 4].number_input(
                    info.label, key=f"wi_{f}", min_value=0.0,
                    max_value=1.0 if f in {"entropy", "fix"} else None,
                    step=0.05 if f == "entropy" else 1.0,
                    format="%.2f" if continuous else "%.0f", help=f"`{f}` ({family}): {info.description}",
                )

    if st.button("Suggest how to lower risk", type="primary", icon=":material/lightbulb:"):
        table = suggestions(bundle, values, df)
        st.dataframe(table.style.format({"Risk": "{:.1%}", "Change": "{:+.1%}"}), hide_index=True, width="stretch")
        best = table.iloc[1:].loc[lambda t: t["Change"].idxmin()]
        if best["Change"] < -0.005:
            st.success(f"Simulated best option: **{best['Scenario'].lower()}** "
                       f"→ risk {best['Risk']:.1%} ({best['Change']:+.1%}).")
        else:
            st.info("None of these simple changes lowers the simulated risk for this commit.")
        st.caption("Simulation only. *Split* halves lines added/deleted and files/directories/subsystems touched; "
                   "*experienced author* raises the three experience metrics to the 75th percentile of the dataset.")

    st.subheader("Explanation of the simulated commit")
    show_explanation(bundle, pd.Series(values), df, reference=float(row["risk"]), key="whatif")


def page_score_local(bundle: dict, df: pd.DataFrame) -> None:
    st.title("Score a Local Commit")
    st.markdown("Mine a commit straight from a git repository on this computer, score it and explain it, "
                "the same way the predictor would run in a code-review bot or a pre-merge check.")
    if not explain_available():
        return

    with st.form("score_local"):
        repo = st.text_input("Repository path", value=DEFAULT_REPO,
                             help="Absolute, or relative to the project folder")
        commit = st.text_input("Commit hash (full or short)", placeholder="e.g. 1a2b3c4")
        submitted = st.form_submit_button("Score commit", type="primary")
    st.caption("Mining replays the repository history up to the commit, so this can take **30 to 60 seconds**.")

    if submitted:
        repo_path = Path(repo).expanduser()
        repo_path = repo_path if repo_path.is_absolute() else ROOT / repo_path
        if not commit.strip():
            st.error("Please enter a commit hash.")
        elif not (repo_path / ".git").exists() and not (repo_path / "HEAD").exists():
            st.error(f"`{repo_path}` is not a git repository.")
        else:
            with st.spinner(f"Mining {commit.strip()} …"):
                try:
                    mined, _ = score_commit(str(repo_path), commit.strip(), bundle)
                    st.session_state["local_result"] = mined
                except Exception as exc:  # bad hash, merge commit, no code files changed, ...
                    st.session_state.pop("local_result", None)
                    st.error(f"Could not score this commit: {exc}")

    mined = st.session_state.get("local_result")
    if mined is not None:
        st.subheader(f"{str(mined.get('commit_hash', ''))[:10]}: {first_line(mined.get('message', ''), 90)}")
        show_explanation(bundle, mined, df, key="local")


def page_global(bundle: dict, df: pd.DataFrame) -> None:
    st.title("Global Explanations")
    st.markdown("Which change metrics drive the model's predictions **across all test commits**? "
                "Importance = the average size of a feature's SHAP contribution.")
    if not explain_available():
        return
    sample = test_period(df, bundle)[FEATURE_COLUMNS]
    sample = sample.sample(min(300, len(sample)), random_state=0).reset_index(drop=True)
    importance = global_importance_cached(bundle, bundle_key(bundle), sample)
    st.plotly_chart(charts.importance_bar(importance), width="stretch")
    top = importance.sort_values("mean_abs_shap", ascending=False)["label"].head(3).tolist()
    st.markdown(f"The three most influential features are **{', '.join(top)}**.")

    summary = REPORTS_DIR / "figures" / "shap_summary.png"
    if summary.exists():
        st.subheader("SHAP summary plot")
        st.caption("Each dot is a commit. Red = high feature value, blue = low. Dots to the right raised the risk.")
        st.image(str(summary), width="stretch")
    else:
        st.info(f"The SHAP summary plot (`reports/figures/shap_summary.png`) has not been generated yet. "
                f"Create it with:\n\n```\n{EXPLAIN_CMD}\n```", icon=":material/info:")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
PAGES = {
    "Overview": ":material/dashboard:",
    "Model Leaderboard": ":material/leaderboard:",
    "Explain a Commit": ":material/manage_search:",
    "What-If Simulator": ":material/tune:",
    "Score a Local Commit": ":material/terminal:",
    "Global Explanations": ":material/insights:",
}


def main() -> None:
    df, is_real = get_dataset(str(DATASET_PATH))
    bundle, is_trained = get_bundle(str(BUNDLE_PATH), str(DATASET_PATH))

    with st.sidebar:
        st.header("Risk Predictor")
        # ?page=explain (or any other page name) opens that page directly, handy for links in docs
        wanted = st.query_params.get("page", "").lower()
        start = next((i for i, p in enumerate(PAGES) if wanted and wanted in p.lower()), 0)
        page = st.radio("Page", list(PAGES), index=start, format_func=lambda p: f"{PAGES[p]} {p}",
                        label_visibility="collapsed")
        st.divider()
        st.caption("MODEL")
        st.markdown(f"**{bundle.get('model_name', 'unknown')}**  \nTrained on {bundle.get('trained_on', '?')}  \n"
                    f"Decision threshold: {float(bundle.get('threshold', 0.5)):.0%}")
        if not is_trained or not is_real:
            st.warning("Running in demo mode. See the Overview page for how to build the real model.",
                       icon=":material/warning:")

    if page != "Overview" and page != "Model Leaderboard":
        missing_model_notice(is_trained, is_real)

    if page == "Overview":
        page_overview(df, is_real, bundle)
        if not is_trained:
            missing_model_notice(is_trained, True)
    elif page == "Model Leaderboard":
        page_leaderboard(bundle, is_trained)
    elif page == "Explain a Commit":
        page_explain(bundle, df)
    elif page == "What-If Simulator":
        page_whatif(bundle, df)
    elif page == "Score a Local Commit":
        page_score_local(bundle, df)
    elif page == "Global Explanations":
        page_global(bundle, df)


main()
