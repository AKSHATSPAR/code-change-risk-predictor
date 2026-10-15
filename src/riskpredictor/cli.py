"""Command-line interface.

Examples
--------
Score and explain one commit of a local repository::

    uv run python -m riskpredictor.cli score --repo data/raw/repos/requests --commit 1a2b3c4

List the riskiest commits of the test period::

    uv run python -m riskpredictor.cli top --data data/processed/dataset.csv -n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_BUNDLE = "models/best_model.joblib"


def _load_bundle(path: str) -> dict:
    if not Path(path).exists():
        sys.exit(f"Model not found at {path}. Train one first: uv run python -m riskpredictor.train")
    return joblib.load(path)


def cmd_score(args: argparse.Namespace) -> None:
    from riskpredictor.scoring import score_commit

    bundle = _load_bundle(args.bundle)
    print(f"Mining {args.commit} in {args.repo} (this can take a minute)...", file=sys.stderr)
    row, explanation = score_commit(args.repo, args.commit, bundle)
    print(f"Commit : {row.get('commit_hash', args.commit)}")
    print(f"Message: {str(row.get('message', '')).splitlines()[0] if row.get('message') else ''}")
    print()
    print(explanation.to_text())


def cmd_top(args: argparse.Namespace) -> None:
    from riskpredictor.features import LABEL_COLUMN, load_dataset
    from riskpredictor.scoring import top_risky

    bundle = _load_bundle(args.bundle)
    top = top_risky(bundle, load_dataset(args.data), n=args.n)
    table = pd.DataFrame({
        "commit": top["commit_hash"].str[:10],
        "date": pd.to_datetime(top["author_date"]).dt.strftime("%Y-%m-%d"),
        "risk": top["risk"].map("{:.1%}".format),
        "level": top["risk_level"],
        "buggy (SZZ)": top[LABEL_COLUMN].map({1: "yes", 0: "no"}),
        "message": top["message"].astype(str).str.split("\n").str[0].str.slice(0, 60),
    })
    print(f"Top {len(table)} riskiest commits in the test period ({bundle.get('model_name', 'model')}):\n")
    print(table.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="riskpredictor", description="Explainable code change risk predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score and explain one commit of a local git repository")
    score.add_argument("--repo", required=True, help="path to a local git repository")
    score.add_argument("--commit", required=True, help="commit hash (full or short)")
    score.add_argument("--bundle", default=DEFAULT_BUNDLE, help=f"trained model (default: {DEFAULT_BUNDLE})")
    score.set_defaults(func=cmd_score)

    top = sub.add_parser("top", help="list the riskiest commits of the test period")
    top.add_argument("--data", default="data/processed/dataset.csv", help="mined dataset CSV")
    top.add_argument("-n", type=int, default=10, help="how many commits to show (default: 10)")
    top.add_argument("--bundle", default=DEFAULT_BUNDLE, help=f"trained model (default: {DEFAULT_BUNDLE})")
    top.set_defaults(func=cmd_top)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
