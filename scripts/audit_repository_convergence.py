#!/usr/bin/env python3
"""Report branch and worktree convergence facts without changing Git state.

This tool intentionally stops at inventory.  It does not fetch, switch, merge,
tag, stash, prune, remove, or delete anything.  Human review still owns branch
disposition and the choice of a formal trunk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class GitCommandError(RuntimeError):
    """A read-only Git query could not be completed."""


def _run_git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    trusted_path = str(repo.resolve()).replace("\\", "/")
    result = subprocess.run(
        ["git", "-c", f"safe.directory={trusted_path}", "-C", str(repo), *args],
        capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        text=not binary,
        check=False,
        timeout=120,
    )
    if result.returncode:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise GitCommandError(stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _optional_git(repo: Path, *args: str) -> str | None:
    try:
        return str(_run_git(repo, *args)).strip()
    except GitCommandError:
        return None


def repository_root(repo: Path) -> Path:
    root = str(_run_git(repo, "rev-parse", "--show-toplevel")).strip()
    return Path(root).resolve()


def parse_worktree_porcelain(output: str) -> list[dict[str, Any]]:
    worktrees: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in [*output.splitlines(), ""]:
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key in {"bare", "detached"}:
            current[key] = True
        elif key == "branch":
            current[key] = value.removeprefix("refs/heads/")
        else:
            current[key] = value
    return worktrees


def parse_status_porcelain(output: bytes) -> dict[str, int | bool]:
    records = output.split(b"\0")
    tracked = staged = unstaged = untracked = conflicts = 0
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code = record[:2].decode("ascii", errors="replace")
        if code == "??":
            untracked += 1
            continue
        if code == "!!":
            continue
        tracked += 1
        x, y = code[0], code[1]
        staged += int(x != " ")
        unstaged += int(y != " ")
        conflicts += int(code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"})
        if "R" in code or "C" in code:
            index += 1  # Porcelain v1 -z appends the original rename/copy path.
    return {
        "tracked": tracked,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "conflicts": conflicts,
        "dirty": bool(tracked or untracked),
    }


def worktree_status(path: Path, untracked_files: str) -> dict[str, Any]:
    try:
        raw = _run_git(
            path,
            "status",
            "--porcelain=v1",
            "-z",
            f"--untracked-files={untracked_files}",
            binary=True,
        )
    except GitCommandError as error:
        return {"error": str(error), "dirty": None}
    assert isinstance(raw, bytes)
    return parse_status_porcelain(raw)


def _parse_ref_rows(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        name, commit, date, subject, symref = line.split("\0", 4)
        if symref:
            continue
        rows.append(
            {"ref": name, "commit": commit, "committer_date": date, "subject": subject}
        )
    return rows


def _divergence(repo: Path, base: str, ref: str) -> dict[str, Any]:
    counts = str(_run_git(repo, "rev-list", "--left-right", "--count", f"{base}...{ref}"))
    base_only, ref_only = (int(value) for value in counts.split())
    cherry = str(_run_git(repo, "cherry", base, ref))
    patch_equivalent = unique_patch = 0
    for line in cherry.splitlines():
        if line.startswith("- "):
            patch_equivalent += 1
        elif line.startswith("+ "):
            unique_patch += 1
    return {
        "base_only_commits": base_only,
        "ref_only_commits": ref_only,
        "patch_equivalent_commits": patch_equivalent,
        "unique_patch_commits": unique_patch,
        "merge_base": _optional_git(repo, "merge-base", base, ref),
    }


def collect_refs(repo: Path, base: str, namespace: str) -> list[dict[str, Any]]:
    format_string = (
        "%(refname:short)%00%(objectname)%00%(committerdate:iso-strict)"
        "%00%(subject)%00%(symref)"
    )
    output = str(_run_git(repo, "for-each-ref", f"--format={format_string}", namespace))
    rows = _parse_ref_rows(output)
    for row in rows:
        row.update(_divergence(repo, base, row["ref"]))
    return rows


def collect_stashes(repo: Path) -> list[dict[str, str]]:
    # ``stash list`` uses log pretty-format escapes, unlike ``for-each-ref``.
    output = str(_run_git(repo, "stash", "list", "--format=%gd%x00%H%x00%gs"))
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if line:
            name, commit, subject = line.split("\0", 2)
            rows.append({"name": name, "commit": commit, "subject": subject})
    return rows


def collect_inventory(
    repo: Path,
    base: str,
    *,
    include_remotes: bool = True,
    untracked_files: str = "all",
) -> dict[str, Any]:
    root = repository_root(repo)
    base_commit = str(_run_git(root, "rev-parse", "--verify", f"{base}^{{commit}}")).strip()
    worktrees = parse_worktree_porcelain(str(_run_git(root, "worktree", "list", "--porcelain")))
    for worktree in worktrees:
        path = Path(worktree["worktree"])
        worktree["status"] = worktree_status(path, untracked_files)

    local_refs = collect_refs(root, base, "refs/heads")
    remote_refs = collect_refs(root, base, "refs/remotes") if include_remotes else []
    complete = all("error" not in worktree["status"] for worktree in worktrees)
    return {
        "schema_version": 1,
        "complete": complete,
        "repository": str(root),
        "base": {"ref": base, "commit": base_commit},
        "worktrees": worktrees,
        "refs": {"local": local_refs, "remote_tracking": remote_refs},
        "stashes": collect_stashes(root),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(inventory: dict[str, Any]) -> str:
    worktrees = inventory["worktrees"]
    dirty = sum(item["status"].get("dirty") is True for item in worktrees)
    unreadable = sum("error" in item["status"] for item in worktrees)
    lines = [
        "# Repository convergence inventory",
        "",
        f"Repository: `{_cell(inventory['repository'])}`",
        f"Base: `{_cell(inventory['base']['ref'])}` "
        f"(`{_cell(inventory['base']['commit'])}`)",
        f"Worktrees: {len(worktrees)} total, {dirty} dirty, {unreadable} unreadable",
        "",
        "This is a read-only fact report. It does not select, merge, or remove a ref.",
        "",
        "## Worktrees",
        "",
        "| Branch | HEAD | Tracked | Untracked | Conflicts | Path |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in worktrees:
        status = item["status"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(item.get("branch", "(detached)")),
                    _cell(item.get("HEAD", "")[:12]),
                    _cell(status.get("tracked", "error")),
                    _cell(status.get("untracked", "error")),
                    _cell(status.get("conflicts", "error")),
                    f"`{_cell(item['worktree'])}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Refs relative to base",
            "",
            "| Scope | Ref | Base-only | Ref-only | Patch-equivalent | Unique patches | Merge base | Tip | Subject |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for scope, rows in inventory["refs"].items():
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(scope),
                        _cell(row["ref"]),
                        _cell(row["base_only_commits"]),
                        _cell(row["ref_only_commits"]),
                        _cell(row["patch_equivalent_commits"]),
                        _cell(row["unique_patch_commits"]),
                        _cell((row.get("merge_base") or "")[:12]),
                        _cell(row["commit"][:12]),
                        _cell(row["subject"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Stashes", ""])
    if inventory["stashes"]:
        lines.extend(["| Name | Commit | Subject |", "|---|---|---|"])
        for row in inventory["stashes"]:
            lines.append(
                f"| {_cell(row['name'])} | {_cell(row['commit'][:12])} | "
                f"{_cell(row['subject'])} |"
            )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--no-remotes", action="store_true")
    parser.add_argument(
        "--untracked-files", choices=("no", "normal", "all"), default="all"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = collect_inventory(
        args.repo,
        args.base,
        include_remotes=not args.no_remotes,
        untracked_files=args.untracked_files,
    )
    if args.format == "json":
        json.dump(inventory, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(inventory))
    return 0 if inventory["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
