from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "workspace_tool", Path(__file__).resolve().parents[1] / "scripts/clearvla_workspace.py"
)
workspace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workspace)


@pytest.mark.parametrize("name", ["../pen", "/tmp", "..", "pen/name", "-bad", "A" * 100])
def test_run_name_cannot_escape_namespace(name):
    with pytest.raises(ValueError):
        workspace.segment(name)


def test_readonly_list_creates_nothing(tmp_path, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["workspace", "--root", str(root), "list"])
    assert workspace.main() == 0
    assert not root.exists()


def test_clone_requires_explicit_branch_before_creating_directories(tmp_path, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["workspace", "--root", str(root), "init", "--clone"])
    with pytest.raises(ValueError, match="requires --branch"):
        workspace.main()
    assert not root.exists()


def require_symlinks(tmp_path):
    trial = tmp_path / "link-test"
    try:
        trial.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("this host cannot create symlinks")
    trial.unlink()


def test_link_refuses_overwrite_and_is_idempotent(tmp_path):
    require_symlinks(tmp_path)
    target = tmp_path / "real.log"
    target.write_text("retained")
    alias = tmp_path / "console.log"
    workspace.link(alias, target)
    workspace.link(alias, target)
    other = tmp_path / "another.log"
    with pytest.raises(FileExistsError):
        workspace.link(alias, other)
    assert alias.read_text() == "retained"


def test_import_checks_context_and_preserves_originals(tmp_path):
    require_symlinks(tmp_path)
    root = workspace.layout(tmp_path / "managed")
    code = tmp_path / "old-code"
    output = tmp_path / "old-run"
    code.mkdir()
    output.mkdir()
    context = {"config": {"data": {"output_dir": str(output)}}}
    (output / "run_context.json").write_text(json.dumps(context))
    (output / "metrics.jsonl").write_text('{"step":20}\n')
    console = tmp_path / "old.log"
    console.write_text("original")
    args = argparse.Namespace(
        output=str(output),
        code=str(code),
        console=str(console),
        outlet="pen",
        name="20260906-test",
        pid=None,
    )
    view = workspace.import_run(root, args)
    workspace.import_run(root, args)
    assert (view / "console.log").resolve() == console
    assert (view / "metrics.jsonl").resolve() == output / "metrics.jsonl"
    assert console.read_text() == "original"
    context["config"]["data"]["output_dir"] = str(tmp_path / "wrong")
    (output / "run_context.json").write_text(json.dumps(context))
    with pytest.raises(ValueError, match="output differs"):
        workspace.import_run(root, args)


def test_active_pointer_cannot_abandon_live_or_unknown_run(tmp_path, monkeypatch):
    require_symlinks(tmp_path)
    root = workspace.layout(tmp_path / "managed")
    for name in ("old", "new"):
        view = workspace.run_path(root, "pen", name)
        view.mkdir(parents=True)
        (view / "run_context.json").write_text("{}")
    workspace.activate(root, "pen", "old")
    with pytest.raises(ValueError, match="unverified"):
        workspace.activate(root, "pen", "new")
    workspace.write_json(
        workspace.run_path(root, "pen", "old") / "entry.json", {"status": "running"}
    )
    monkeypatch.setattr(workspace, "is_running", lambda _: True)
    with pytest.raises(ValueError, match="running"):
        workspace.activate(root, "pen", "new")
    assert (root / "active/pen").resolve().name == "old"


@pytest.mark.skipif(
    os.name != "posix", reason="real remote lifecycle requires POSIX open-file rename"
)
@pytest.mark.parametrize("fail_before_context", [False, True])
def test_new_run_log_lifecycle_and_no_overwrite(tmp_path, fail_before_context):
    code = tmp_path / "code"
    code.mkdir()
    output = tmp_path / "experiment"
    script = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "assert not list(p.iterdir()), 'must enter an empty experiment'; "
        "print('startup',flush=True); "
        + (
            "sys.exit(7)"
            if fail_before_context
            else "(p/'run_context.json').write_text('{}'); time.sleep(.5); print('finished',flush=True)"
        )
    )
    command = [sys.executable, "-c", script, str(output)]
    result = workspace.run_logged(command, code, output, {}, {"kind": "test"})
    assert result == (7 if fail_before_context else 0)
    assert "startup" in (output / "console.log").read_text()
    assert (output / "code").resolve() == code
    assert workspace.read_entry(output)["exit_code"] == result
    assert not list(tmp_path.glob(".*-starting-*.log"))
    with pytest.raises(FileExistsError):
        workspace.run_logged(command, code, output, {}, {})


def test_checkout_reuses_git_worktree_and_rejects_dirty(tmp_path):
    root = workspace.layout(tmp_path / "managed")
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "tracked.txt").write_text("source")
    workspace.git(repo, "add", "tracked.txt")
    workspace.git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "test",
    )
    worktree = workspace.checkout(root, "HEAD")
    assert workspace.checkout(root, "HEAD") == worktree
    (worktree / "tracked.txt").write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        workspace.checkout(root, "HEAD")
    assert (worktree / "tracked.txt").read_text() == "dirty"


def test_launch_environment_does_not_allow_auth_tokens():
    assert "ANTHROPIC_AUTH_TOKEN" not in workspace.ENVIRONMENT
    assert "OPENAI_API_KEY" not in workspace.ENVIRONMENT
    assert "HF_TOKEN" not in workspace.ENVIRONMENT
