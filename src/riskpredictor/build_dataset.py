"""Build the training dataset: mine change metrics + SZZ labels for one or more repositories.

    uv run python -m riskpredictor.build_dataset --repo data/raw/repos/requests

Writes data/processed/dataset.csv (schema: features.ALL_COLUMNS) and a JSON summary next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from riskpredictor.features import ALL_COLUMNS, DATE_COLUMN, DEFAULT_DATASET, LABEL_COLUMN
from riskpredictor.labeling import find_bug_introducing_commits
from riskpredictor.mining import mine_repository

# Bugs in the newest commits haven't been found and fixed yet, so their "clean" label can't be
# trusted. We drop commits from the last N days of each repository's history.
DEFAULT_EXCLUDE_RECENT_DAYS = 180


def build_repo_dataset(
    repo_path: str | Path,
    exclude_recent_days: int = DEFAULT_EXCLUDE_RECENT_DAYS,
    max_commits: int | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = mine_repository(repo_path, max_commits=max_commits, progress=progress)
    fixes = df.loc[df["fix"] == 1, "commit_hash"].tolist()
    introducers = find_bug_introducing_commits(repo_path, fixes, progress=progress)
    df[LABEL_COLUMN] = df["commit_hash"].isin(introducers).astype(int)

    cutoff = df[DATE_COLUMN].max() - pd.Timedelta(days=exclude_recent_days)
    labelled = df[df[DATE_COLUMN] <= cutoff].reset_index(drop=True)

    summary = {
        "repo": Path(repo_path).name,
        "code_commits_mined": len(df),
        "fix_commits": len(fixes),
        "bug_introducing_commits": int(df[LABEL_COLUMN].sum()),
        "dropped_recent_commits": len(df) - len(labelled),
        "rows": len(labelled),
        "buggy_rate": round(float(labelled[LABEL_COLUMN].mean()), 4) if len(labelled) else None,
        "first_commit": str(labelled[DATE_COLUMN].min()),
        "last_commit": str(labelled[DATE_COLUMN].max()),
    }
    return labelled[ALL_COLUMNS], summary


def build_dataset(
    repo_paths: list[str | Path],
    out: str | Path = DEFAULT_DATASET,
    exclude_recent_days: int = DEFAULT_EXCLUDE_RECENT_DAYS,
    max_commits: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    frames, summaries = [], []
    for repo in repo_paths:
        df, summary = build_repo_dataset(repo, exclude_recent_days, max_commits, progress)
        frames.append(df)
        summaries.append(summary)

    data = pd.concat(frames, ignore_index=True).sort_values(DATE_COLUMN, kind="stable")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summaries, indent=2))
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", nargs="+", required=True, help="path(s) to local git clones")
    parser.add_argument("--out", default=str(DEFAULT_DATASET))
    parser.add_argument("--exclude-recent-days", type=int, default=DEFAULT_EXCLUDE_RECENT_DAYS)
    parser.add_argument("--max-commits", type=int, default=None, help="only mine the first N commits (for quick tests)")
    args = parser.parse_args(argv)

    data = build_dataset(args.repo, args.out, args.exclude_recent_days, args.max_commits)
    print(f"\nWrote {len(data)} commits to {args.out}")
    print(data.groupby("repo")[LABEL_COLUMN].agg(commits="size", buggy="sum", buggy_rate="mean").round(3))


if __name__ == "__main__":
    main()
