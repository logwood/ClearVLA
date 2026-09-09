from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "repository_convergence_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_repository_convergence.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_parse_worktree_porcelain_preserves_branch_and_flags():
    rows = audit.parse_worktree_porcelain(
        "worktree C:/repo\n"
        "HEAD abcdef\n"
        "branch refs/heads/main\n\n"
        "worktree C:/repo/detached\n"
        "HEAD 123456\n"
        "detached\n"
        "locked running experiment\n\n"
    )
    assert rows == [
        {"worktree": "C:/repo", "HEAD": "abcdef", "branch": "main"},
        {
            "worktree": "C:/repo/detached",
            "HEAD": "123456",
            "detached": True,
            "locked": "running experiment",
        },
    ]


def test_parse_status_porcelain_counts_rename_once():
    output = (
        b" M tracked.txt\0"
        b"A  staged.txt\0"
        b"R  renamed.txt\0old.txt\0"
        b"?? new.txt\0"
        b"UU conflict.txt\0"
    )
    assert audit.parse_status_porcelain(output) == {
        "tracked": 4,
        "staged": 3,
        "unstaged": 2,
        "untracked": 1,
        "conflicts": 1,
        "dirty": True,
    }


def test_collect_inventory_reports_real_worktrees_without_writing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "tracked.txt").write_text("base", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "branch", "--show-current")

    topic = tmp_path / "topic"
    _git(repo, "worktree", "add", "-b", "topic", str(topic), "HEAD")
    (topic / "tracked.txt").write_text("changed", encoding="utf-8")
    (topic / "new.txt").write_text("untracked", encoding="utf-8")
    _git(topic, "stash", "push", "-u", "-m", "protected test state")
    (topic / "tracked.txt").write_text("changed again", encoding="utf-8")
    (topic / "new.txt").write_text("untracked again", encoding="utf-8")

    before = {path.relative_to(topic) for path in topic.rglob("*") if path.is_file()}
    inventory = audit.collect_inventory(
        repo, base, include_remotes=False, untracked_files="all"
    )
    after = {path.relative_to(topic) for path in topic.rglob("*") if path.is_file()}

    assert before == after
    assert inventory["complete"] is True
    assert inventory["base"]["ref"] == base
    assert {row["ref"] for row in inventory["refs"]["local"]} == {base, "topic"}
    assert len(inventory["stashes"]) == 1
    assert "protected test state" in inventory["stashes"][0]["subject"]
    topic_row = next(row for row in inventory["worktrees"] if row.get("branch") == "topic")
    assert topic_row["status"]["tracked"] == 1
    assert topic_row["status"]["untracked"] == 1
    assert topic_row["status"]["dirty"] is True

    report = audit.render_markdown(inventory)
    assert "This is a read-only fact report" in report
    assert "0 unreadable" in report
    assert "| topic |" in report
