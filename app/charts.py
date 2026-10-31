"""Plotly figures used by the Streamlit app (kept separate so the page code stays readable)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, FEATURE_INFO, LABEL_COLUMN
from riskpredictor.scoring import risk_level

# One colour per meaning, used consistently on every page.
RAISES_RISK = "#e34948"  # red  - pushes the prediction towards "buggy"
LOWERS_RISK = "#2a78d6"  # blue - pushes the prediction towards "clean"
BUGGY = "#eb6834"
CLEAN = "#2a78d6"
LEVEL_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    kwargs.setdefault("margin", dict(l=10, r=10, t=40, b=10))
    fig.update_layout(height=height, **kwargs)
    return fig


def risk_gauge(risk: float, threshold: float, reference: float | None = None) -> go.Figure:
    """Speedometer-style gauge: 0-100 % risk, coloured Low / Medium / High bands."""
    steps = [
        dict(range=[0, threshold * 50], color="rgba(12,163,12,0.18)"),
        dict(range=[threshold * 50, threshold * 100], color="rgba(250,178,25,0.22)"),
        dict(range=[threshold * 100, 100], color="rgba(208,59,59,0.20)"),
    ]
    indicator = go.Indicator(
        mode="gauge+number" + ("+delta" if reference is not None else ""),
        value=risk * 100,
        number=dict(suffix="%", valueformat=".1f"),
        delta=(dict(reference=reference * 100, valueformat=".1f", suffix=" pts",
                    increasing=dict(color=RAISES_RISK), decreasing=dict(color=LOWERS_RISK))
               if reference is not None else None),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color=LEVEL_COLORS[risk_level(risk, threshold)], thickness=0.3),
            steps=steps,
            threshold=dict(line=dict(color=LEVEL_COLORS["High"], width=3), value=threshold * 100),
        ),
        title=dict(text="Bug risk"),
    )
    return _layout(go.Figure(indicator), height=260, margin=dict(l=30, r=30, t=60, b=10))


def shap_bar(contributions, top_k: int = 10) -> go.Figure:
    """Horizontal bars: how much each feature pushed this commit's risk up (red) or down (blue)."""
    contribs = list(contributions)[:top_k][::-1]  # largest at the top
    labels = [f"{c.label} = {c.value:g}" for c in contribs]
    values = [c.shap_value for c in contribs]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=[RAISES_RISK if v > 0 else LOWERS_RISK for v in values]),
        customdata=[c.description for c in contribs],
        hovertemplate="%{y}<br>Effect on risk: %{x:+.1%}<br>%{customdata}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#898781")
    return _layout(fig, height=max(260, 34 * len(contribs) + 60),
                   title="Feature contributions (SHAP): red raises risk, blue lowers it",
                   xaxis=dict(title="Change in predicted risk (SHAP)", tickformat="+.0%"), showlegend=False)


def bug_rate_over_time(df: pd.DataFrame, min_commits: int = 10) -> go.Figure:
    """Quarterly share of commits that SZZ later labelled as bug-introducing.

    Quarters with fewer than ``min_commits`` commits are hidden: with 2 commits a single bug is "50 %".
    """
    quarterly = (
        df.set_index(pd.to_datetime(df[DATE_COLUMN], utc=True).dt.tz_localize(None))[LABEL_COLUMN]
        .resample("QS").agg(["mean", "count"]).query("count >= @min_commits").reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quarterly[DATE_COLUMN], y=quarterly["count"], name="Commits", yaxis="y2",
        marker=dict(color="rgba(137,135,129,0.25)"),
        hovertemplate="Commits: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=quarterly[DATE_COLUMN], y=quarterly["mean"] * 100, mode="lines+markers", name="Bug rate",
        line=dict(color=BUGGY, width=2), marker=dict(size=5),
        hovertemplate="Bug rate: %{y:.1f}%<extra></extra>",
    ))
    return _layout(fig, title="Bug-introducing commits per quarter",
                   yaxis=dict(title="Bug rate (%)", rangemode="tozero"),
                   yaxis2=dict(title="Commits", overlaying="y", side="right", showgrid=False, rangemode="tozero",
                               tickformat=",.0f"),
                   hovermode="x unified", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))


def feature_difference(df: pd.DataFrame) -> go.Figure:
    """For every feature: how much higher (red) or lower (blue) the average is in buggy commits."""
    means = df.groupby(LABEL_COLUMN)[FEATURE_COLUMNS].mean()
    clean, buggy = means.loc[0], means.loc[1]
    diff = ((buggy - clean) / clean.replace(0, float("nan")) * 100).fillna(0).sort_values()
    fig = go.Figure(go.Bar(
        x=diff.values, y=[FEATURE_INFO[f].label for f in diff.index], orientation="h",
        marker=dict(color=[RAISES_RISK if v > 0 else LOWERS_RISK for v in diff.values]),
        hovertemplate="%{y}<br>Buggy vs clean: %{x:+.0f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#898781")
    return _layout(fig, height=480, title="Buggy vs clean: difference of the mean",
                   xaxis_title="How much higher in buggy commits (%)")


def feature_box(df: pd.DataFrame, feature: str, log_scale: bool) -> go.Figure:
    """Distribution of one feature for buggy vs clean commits."""
    fig = go.Figure()
    for value, name, color in [(0, "Clean", CLEAN), (1, "Buggy", BUGGY)]:
        values = df.loc[df[LABEL_COLUMN] == value, feature]
        fig.add_trace(go.Box(y=values + (1 if log_scale else 0), name=name, marker_color=color,
                             boxmean=True))
    title = FEATURE_INFO[feature].label + (" (log scale, +1)" if log_scale else "")
    fig.update_yaxes(type="log" if log_scale else "linear")
    return _layout(fig, height=480, title=title, showlegend=False)


def importance_bar(importance: pd.DataFrame) -> go.Figure:
    """Global importance = mean |SHAP| over many commits."""
    imp = importance.sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=imp["mean_abs_shap"], y=imp["label"], orientation="h", marker=dict(color=LOWERS_RISK),
        hovertemplate="%{y}<br>Average effect on risk: %{x:.1%}<extra></extra>",
    ))
    return _layout(fig, height=max(320, 30 * len(imp) + 60),
                   title="Which features matter most overall",
                   xaxis=dict(title="Average effect on predicted risk (mean |SHAP|)", tickformat=".0%"))


def metric_bars(comparison: pd.DataFrame, name_col: str, metrics: list[str]) -> go.Figure:
    """Grouped bars comparing models on a few metrics."""
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(name=metric, x=comparison[name_col], y=comparison[metric],
                             marker_color=palette[i % len(palette)],
                             hovertemplate="%{x}<br>" + metric + ": %{y:.3f}<extra></extra>"))
    return _layout(fig, height=380, barmode="group", title="Test-set metrics by model",
                   legend=dict(orientation="h", y=-0.2))


# --------------------------------------------------------------------------------------------
# Data Explorer page
# --------------------------------------------------------------------------------------------
def size_band_chart(table: pd.DataFrame, overall: float) -> go.Figure:
    """Bug rate per size band (lines added binned into ordered bands)."""
    fig = go.Figure(go.Bar(
        x=table["lines added"].astype(str), y=table["bug_rate"] * 100, marker_color=BUGGY,
        text=[f"{r:.0%}" for r in table["bug_rate"]], textposition="outside",
        customdata=table["commits"], hovertemplate="%{x} lines<br>Bug rate: %{y:.1f}%<br>Commits: %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=overall * 100, line_dash="dash", line_color="#898781",
                  annotation_text=f"all commits ({overall:.0%})", annotation_position="top left")
    return _layout(fig, title="Bug rate by size band", xaxis_title="Lines added",
                   yaxis_title="Buggy commits (%)", yaxis_range=[0, max(table["bug_rate"]) * 100 + 12])


def rolling_rate_chart(monthly: pd.DataFrame) -> go.Figure:
    """Monthly commits (bars) and the 12-month rolling bug rate (line)."""
    index = monthly.index.tz_localize(None) if getattr(monthly.index, "tz", None) else monthly.index
    fig = go.Figure()
    fig.add_trace(go.Bar(x=index, y=monthly["commits"], name="commits per month", marker_color="#c9c8c2",
                         yaxis="y2", hovertemplate="%{x|%b %Y}: %{y} commits<extra></extra>"))
    fig.add_trace(go.Scatter(x=index, y=monthly["rolling_bug_rate"] * 100, name="bug rate, rolling 12 months",
                             line=dict(color=BUGGY, width=2.5), hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Buggy commits (%)"),
                      yaxis2=dict(title="Commits", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    return _layout(fig, title="Commits and bug rate over time")


def dot_chart(table: pd.DataFrame, label_col: str, title: str, overall: float) -> go.Figure:
    """One dot per group with a 95% interval: easier to compare than bars."""
    k, n = table["buggy"].to_numpy(), table["commits"].clip(lower=1).to_numpy()
    rate = k / n
    z = 1.96
    centre = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * ((rate * (1 - rate) / n + z**2 / (4 * n**2)) ** 0.5) / (1 + z**2 / n)
    fig = go.Figure(go.Scatter(
        x=rate * 100, y=table[label_col].astype(str), mode="markers", marker=dict(color=BUGGY, size=11),
        error_x=dict(type="data", symmetric=False, array=(centre + half - rate) * 100,
                     arrayminus=(rate - (centre - half)) * 100, color="#b5b4ae", thickness=2),
        customdata=table["commits"], hovertemplate="%{y}: %{x:.1f}% buggy (%{customdata} commits)<extra></extra>",
    ))
    fig.add_vline(x=overall * 100, line_dash="dash", line_color="#898781")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title=title, xaxis_title="Buggy commits (%), with 95% interval", showlegend=False)


def bandit_chart(curves: dict) -> go.Figure:
    """Cumulative bugs found by three reviewing strategies."""
    styles = {"oracle": ("#52514e", "dash", "knows the answer already"),
              "bandit": (BUGGY, "solid", "learning reviewer"),
              "random": (CLEAN, "dot", "picks at random")}
    fig = go.Figure()
    for policy, (color, dash, name) in styles.items():
        fig.add_trace(go.Scatter(y=curves[policy], name=name, line=dict(color=color, dash=dash, width=2.5),
                                 hovertemplate="after %{x} reviews: %{y:.0f} bugs<extra>" + name + "</extra>"))
    return _layout(fig, title="Bugs found so far", xaxis_title="Commits reviewed", yaxis_title="Bugs found",
                   legend=dict(orientation="h", y=-0.2))
