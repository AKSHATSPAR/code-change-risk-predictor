"""Walk a git repository from its first commit to its last and compute the 14 just-in-time (JIT)
change metrics (see features.py) for every commit that touches source code.

History and experience metrics only use commits that came *before* the one being measured,
so the features never peek into the future.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
from pydriller import Git, Repository
from pydriller.domain.commit import Commit, ModificationType, ModifiedFile
from tqdm import tqdm

from riskpredictor.features import DATE_COLUMN, FEATURE_COLUMNS, META_COLUMNS

CODE_EXTENSIONS = (
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".cs", ".go", ".rb", ".rs", ".php", ".kt", ".scala", ".swift", ".m", ".sh",
)

# Keywords that mark a commit as a bug fix. "patch" is left out on purpose: in HTTP libraries
# such as requests it usually means the PATCH method, not a bug fix.
FIX_PATTERN = re.compile(r"\b(fix(e[sd]|ing)?|bugs?|bugfix|hotfix|defects?)\b", re.IGNORECASE)


def is_code_file(path: str | None, extensions: tuple[str, ...] = CODE_EXTENSIONS) -> bool:
    return path is not None and path.lower().endswith(extensions)


def is_fix_commit(message: str) -> bool:
    return FIX_PATTERN.search(message) is not None


def change_entropy(changed_lines: list[int]) -> float:
    """Normalised Shannon entropy of modified lines across files: 0 = all in one file, 1 = evenly spread."""
    total = sum(changed_lines)
    if total == 0 or len(changed_lines) < 2:
        return 0.0
    h = -sum(n / total * math.log2(n / total) for n in changed_lines if n > 0)
    return h / math.log2(len(changed_lines))


def _subsystem(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "<root>"


def _author_id(commit: Commit) -> str:
    """Short anonymous id, so the dataset doesn't store anyone's email address."""
    who = (commit.author.email or commit.author.name or "unknown").strip().lower()
    return hashlib.sha1(who.encode()).hexdigest()[:10]


def _loc_before(f: ModifiedFile) -> int:
    if f.change_type == ModificationType.ADD:
        return 0
    try:
        source = f.source_code_before
    except (ValueError, UnicodeDecodeError):  # binary or undecodable file
        return 0
    return len(source.splitlines()) if source else 0


class _RepoHistory:
    """Everything we know about the repository *before* the current commit."""

    def __init__(self) -> None:
        self.file_commits: dict[str, set[str]] = defaultdict(set)  # file -> commits that changed it
        self.file_authors: dict[str, set[str]] = defaultdict(set)  # file -> authors who changed it
        self.file_last_change: dict[str, datetime] = {}
        self.author_dates: dict[str, list[datetime]] = defaultdict(list)  # author -> dates of their commits
        self.author_subsystems: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def metrics(self, commit: Commit, files: list[ModifiedFile]) -> dict[str, float]:
        author, when = _author_id(commit), commit.author_date
        old_paths = [f.old_path or f.new_path for f in files]  # history is stored under the pre-change name
        new_paths = [f.new_path or f.old_path for f in files]
        subsystems = {_subsystem(p) for p in new_paths}
        directories = {str(PurePosixPath(p).parent) for p in new_paths}

        prior_commits = set().union(*(self.file_commits.get(p, set()) for p in old_paths))
        prior_devs = set().union(*(self.file_authors.get(p, set()) for p in old_paths))
        ages = [
            max((when - self.file_last_change[p]).total_seconds(), 0) / 86_400
            for p in old_paths
            if p in self.file_last_change
        ]
        past = self.author_dates.get(author, [])
        by_subsystem = self.author_subsystems.get(author, {})
        subsystem_commits = set().union(*(by_subsystem.get(s, set()) for s in subsystems))

        return {
            "ns": len(subsystems),
            "nd": len(directories),
            "nf": len(files),
            "entropy": round(change_entropy([f.added_lines + f.deleted_lines for f in files]), 4),
            "la": sum(f.added_lines for f in files),
            "ld": sum(f.deleted_lines for f in files),
            "lt": round(sum(_loc_before(f) for f in files) / len(files), 1) if files else 0.0,
            "fix": int(is_fix_commit(commit.msg)),
            "ndev": len(prior_devs),
            "age": round(sum(ages) / len(ages), 2) if ages else 0.0,
            "nuc": len(prior_commits),
            "exp": len(past),
            # a commit from this year counts 1, from last year 1/2, two years ago 1/3, ...
            "rexp": round(sum(1 / (1 + max((when - d).days, 0) // 365) for d in past), 2),
            "sexp": len(subsystem_commits),
        }

    def update(self, commit: Commit, files: list[ModifiedFile]) -> None:
        author, when = _author_id(commit), commit.author_date
        for f in files:
            if f.change_type == ModificationType.RENAME:
                for table in (self.file_commits, self.file_authors, self.file_last_change):
                    if f.old_path in table:
                        table[f.new_path] = table.pop(f.old_path)
            if f.change_type == ModificationType.DELETE:
                for table in (self.file_commits, self.file_authors, self.file_last_change):
                    table.pop(f.old_path, None)
                continue
            self.file_commits[f.new_path].add(commit.hash)
            self.file_authors[f.new_path].add(author)
            self.file_last_change[f.new_path] = when
        self.author_dates[author].append(when)
        for path in (f.new_path or f.old_path for f in files):
            self.author_subsystems[author][_subsystem(path)].add(commit.hash)


def _count_commits(repo_path: str | Path, until: str = "HEAD") -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "--no-merges", until],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def iter_commit_metrics(
    repo_path: str | Path,
    to_commit: str | None = None,
    extensions: tuple[str, ...] = CODE_EXTENSIONS,
    progress: bool = False,
    always_include: str | None = None,
) -> Iterator[dict]:
    """Yield one row (metadata + 14 metrics) per non-merge commit that changes code, oldest first.

    `always_include` forces a row for that commit hash even if it touches no code files.
    """
    history = _RepoHistory()
    commits = Repository(str(repo_path), only_no_merge=True, to_commit=to_commit).traverse_commits()
    if progress:
        total = _count_commits(repo_path, to_commit or "HEAD")
        commits = tqdm(commits, total=total, unit="commit", desc=f"Mining {Path(repo_path).name}")
    for commit in commits:
        files = [f for f in commit.modified_files if is_code_file(f.new_path or f.old_path, extensions)]
        if not files and commit.hash != always_include:
            continue
        row = {
            "repo": Path(repo_path).name,
            "commit_hash": commit.hash,
            "author_id": _author_id(commit),
            DATE_COLUMN: commit.author_date.astimezone(timezone.utc),
            "message": commit.msg.strip().splitlines()[0][:120] if commit.msg.strip() else "",
            **history.metrics(commit, files),
        }
        history.update(commit, files)
        yield row


def mine_repository(
    repo_path: str | Path,
    extensions: tuple[str, ...] = CODE_EXTENSIONS,
    max_commits: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Compute the metrics for every code-changing commit of a repository (no labels yet)."""
    rows = []
    for row in iter_commit_metrics(repo_path, extensions=extensions, progress=progress):
        rows.append(row)
        if max_commits and len(rows) >= max_commits:
            break
    df = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_COLUMNS)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], utc=True)
    return df


def extract_commit_features(repo_path: str | Path, commit_hash: str) -> dict:
    """Metrics for a single commit (used to score a new commit). Replays history up to that commit."""
    full_hash = Git(str(repo_path)).get_commit(commit_hash).hash
    last = None
    for row in iter_commit_metrics(repo_path, to_commit=full_hash, always_include=full_hash):
        last = row
    if last is None or last["commit_hash"] != full_hash:
        raise ValueError(f"Commit {commit_hash} is a merge commit or could not be found in {repo_path}")
    return last
