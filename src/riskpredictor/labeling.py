"""Label commits as bug-introducing with the SZZ algorithm (Śliwerski, Zimmermann & Zeller, 2005).

1. Find bug-fixing commits: the commit message mentions a fix or bug (mining.is_fix_commit).
2. For every line a fix deletes or changes, `git blame` the fix's parent to find the commit that
   last touched that line (PyDriller does this, ignoring whitespace, blank lines and comments).
3. Those commits introduced the bug: label 1. Every other commit: label 0.

Known limitations (worth discussing in the report): fixes that only *add* lines can't be traced,
keyword matching misses some fixes and flags some non-fixes, and the newest commits look clean
only because their bugs haven't been found yet (build_dataset drops them for that reason).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydriller import Git
from pydriller.domain.commit import ModificationType
from tqdm import tqdm

from riskpredictor.mining import CODE_EXTENSIONS, is_code_file

# Fixes that touch more files than this are usually refactorings or mass reformatting rather than
# real bug fixes, and blaming them would wrongly flag lots of innocent commits.
MAX_FILES_PER_FIX = 30


def find_bug_introducing_commits(
    repo_path: str | Path,
    fix_hashes: list[str],
    extensions: tuple[str, ...] = CODE_EXTENSIONS,
    max_files_per_fix: int = MAX_FILES_PER_FIX,
    progress: bool = True,
) -> dict[str, set[str]]:
    """Map each bug-introducing commit hash to the set of fix commits that pointed at it."""
    git = Git(str(repo_path))
    introducers: dict[str, set[str]] = defaultdict(set)
    for fix_hash in tqdm(fix_hashes, desc="SZZ blame", unit="fix", disable=not progress):
        fix = git.get_commit(fix_hash)
        files = [
            f for f in fix.modified_files
            if is_code_file(f.new_path or f.old_path, extensions) and f.change_type != ModificationType.ADD
        ]
        if not files or len(files) > max_files_per_fix:
            continue
        for f in files:
            for blamed in git.get_commits_last_modified_lines(fix, f).values():
                for introducer in blamed:
                    introducers[introducer].add(fix_hash)
    git.clear()
    return dict(introducers)
