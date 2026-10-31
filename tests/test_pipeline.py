"""End-to-end test of the data pipeline on a tiny hand-made git repository.

History of the fake repo:
    c1  alice  2024-01-01  add src/calc.py with a bug (a - b) + README.md
    c2  bob    2024-01-05  add src/util.py and lib/io.py (two subsystems)
    c3  alice  2024-01-10  docs only -> not a code commit, skipped
    c4  bob    2024-02-01  "Fix bug in add": changes the buggy line in src/calc.py

SZZ should therefore blame c1 as the bug-introducing commit.
"""

import os
import subprocess
from pathlib import Path

import pytest

from riskpredictor.build_dataset import build_dataset
from riskpredictor.features import ALL_COLUMNS, load_dataset
from riskpredictor.labeling import find_bug_introducing_commits
from riskpredictor.mining import change_entropy, extract_commit_features, is_fix_commit, mine_repository

CALC_BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
CALC_FIXED = CALC_BUGGY.replace("a - b", "a + b")


def _commit(repo: Path, files: dict[str, str], message: str, author: str, date: str) -> str:
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content)
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author}@example.com", "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author}@example.com", "GIT_COMMITTER_DATE": date,
    }
    run = lambda *args: subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True)
    run("add", "-A")
    run("-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    return run("rev-parse", "HEAD").stdout.strip()


@pytest.fixture(scope="module")
def toy_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("toy_repo")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    hashes = {
        "c1": _commit(repo, {"src/calc.py": CALC_BUGGY, "README.md": "# toy\n"}, "Add calculator", "alice", "2024-01-01T10:00:00+00:00"),
        "c2": _commit(repo, {"src/util.py": "X = 1\nY = 2\n", "lib/io.py": "def read():\n    pass\n"}, "Add utils and io", "bob", "2024-01-05T10:00:00+00:00"),
        "c3": _commit(repo, {"README.md": "# toy\nmore docs\n"}, "Update docs", "alice", "2024-01-10T10:00:00+00:00"),
        "c4": _commit(repo, {"src/calc.py": CALC_FIXED}, "Fix bug in add", "bob", "2024-02-01T10:00:00+00:00"),
    }
    return repo, hashes


def test_helpers():
    assert is_fix_commit("Fixes #123: crash on empty body")
    assert is_fix_commit("bugfix for redirects")
    assert not is_fix_commit("Support HTTP PATCH requests")
    assert not is_fix_commit("Add prefix handling")  # 'prefix' must not count as 'fix'
    assert change_entropy([10]) == 0.0
    assert change_entropy([5, 5]) == pytest.approx(1.0)
    assert 0 < change_entropy([9, 1]) < 1


def test_mining_metrics(toy_repo):
    repo, h = toy_repo
    df = mine_repository(repo, progress=False).set_index("commit_hash")
    assert list(df.index) == [h["c1"], h["c2"], h["c4"]]  # docs-only c3 is skipped

    row = lambda c, *cols: tuple(df.loc[h[c], col] for col in cols)  # noqa: E731
    assert row("c1", "nf", "la", "ld", "lt", "exp", "ndev", "fix") == (1, 6, 0, 0, 0, 0, 0)
    assert row("c2", "nf", "ns", "nd", "exp") == (2, 2, 2, 0)
    assert df.loc[h["c2"], "entropy"] == pytest.approx(1.0)  # 2 lines in each file -> perfectly spread
    assert row("c4", "fix", "la", "ld", "lt") == (1, 1, 1, 6)
    assert row("c4", "ndev", "nuc") == (1, 1)  # only alice's c1 touched calc.py before
    assert row("c4", "exp", "sexp") == (1, 1)  # bob's c2 also touched the src/ subsystem
    assert df.loc[h["c4"], "age"] == pytest.approx(31.0)


def test_szz_blames_bug_introducing_commit(toy_repo):
    repo, h = toy_repo
    introducers = find_bug_introducing_commits(repo, [h["c4"]], progress=False)
    assert introducers == {h["c1"]: {h["c4"]}}


def test_extract_single_commit_matches_full_mining(toy_repo):
    repo, h = toy_repo
    single = extract_commit_features(repo, h["c4"][:8])
    full = mine_repository(repo, progress=False).set_index("commit_hash").loc[h["c4"]]
    assert single["commit_hash"] == h["c4"]
    for feature in ("nf", "la", "ld", "lt", "ndev", "nuc", "exp", "sexp", "age"):
        assert single[feature] == full[feature]


def test_build_dataset_writes_valid_csv(toy_repo, tmp_path):
    repo, h = toy_repo
    out = tmp_path / "dataset.csv"
    build_dataset([repo], out, exclude_recent_days=0, progress=False)
    df = load_dataset(out)
    assert list(df.columns) == ALL_COLUMNS
    assert df.set_index("commit_hash")["buggy"].to_dict() == {h["c1"]: 1, h["c2"]: 0, h["c4"]: 0}
    assert (tmp_path / "dataset_summary.json").exists()
