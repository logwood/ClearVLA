#!/usr/bin/env python3
"""One remote experiment namespace; never delete or move legacy run data.

Standard-library operational tool, independent of model imports and its ABI.
Existing runs are link-only views. New runs use an immutable Git worktree and
write their actual outputs to experiments/<outlet>/<name>.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_ROOT = "/data/senwang/clearvla"
DEFAULT_REMOTE = "git@github.com:logwood/ClearVLA.git"
ENVIRONMENT = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "CONDA_PREFIX",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
}
MODEL_CONTRACT_MIGRATIONS = (
    "world_camera_coordinate_role_v1",
    "world_action_sequence_prefix_v1",
)


def segment(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise ValueError("use a short lowercase outlet/name, not a path or version tree")
    return value


def directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"managed directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def layout(root: Path) -> Path:
    root = root.expanduser().absolute()
    if root == Path(root.anchor) or len(root.parts) < 3:
        raise ValueError("workspace root must be a dedicated directory")
    directory(root)
    for name in ("experiments", "active", "archive", "checkouts"):
        directory(root / name)
    return root


def run_path(root: Path, outlet: str, name: str) -> Path:
    parent = root / "experiments" / segment(outlet)
    if parent.is_symlink():
        raise ValueError("outlet directory cannot redirect outside the workspace")
    return parent / segment(name)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def link(path: Path, target: Path) -> None:
    target = target.absolute()
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing entry: {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def write_json(path: Path, data: dict) -> None:
    # Operational metadata only; no environment dump, credentials or model state.
    handle, temp = tempfile.mkstemp(prefix=".entry-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_entry(path: Path) -> dict:
    entry = path / "entry.json"
    return json.loads(entry.read_text()) if entry.is_file() else {}


def process_identity(pid: int) -> dict | None:
    try:
        proc = Path("/proc") / str(pid)
        # Account for spaces in comm; starttime prevents reused PID confusion.
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": stat[19],
            "cwd": str((proc / "cwd").resolve(strict=True)),
        }
    except (OSError, IndexError):
        return None


def is_running(entry: dict) -> bool:
    saved = entry.get("process")
    return bool(saved and process_identity(saved["pid"]) == saved)


def process_unverified(entry: dict) -> bool:
    saved = entry.get("process")
    if not saved:
        return entry.get("status") not in {"complete", "failed"}
    return process_identity(saved["pid"]) is None and (Path("/proc") / str(saved["pid"])).exists()


def activate(root: Path, outlet: str, name: str) -> None:
    target = run_path(root, outlet, name)
    if not (target / "run_context.json").is_file():
        raise ValueError("only a run with a serialized context can become active")
    pointer = root / "active" / segment(outlet)
    if pointer.is_symlink() and pointer.resolve() == target.resolve():
        return
    if pointer.exists() or pointer.is_symlink():
        if not pointer.is_symlink():
            raise ValueError("active entry is not a managed pointer")
        previous = read_entry(pointer)
        if not previous or process_unverified(previous) or is_running(previous):
            raise ValueError("previous active run is running or unverified; do not replace it")
    temporary = pointer.with_name(f".{outlet}-{os.getpid()}")
    link(temporary, target)
    os.replace(temporary, pointer)


def import_run(root: Path, args: argparse.Namespace) -> Path:
    source, code, console = (
        Path(value).resolve(strict=True) for value in (args.output, args.code, args.console)
    )
    context = json.loads((source / "run_context.json").read_text())
    if not (source / "metrics.jsonl").is_file() or not console.is_file():
        raise ValueError("run must have context, metrics and a verified console log")
    configured = Path(context["config"]["data"]["output_dir"])
    configured = configured if configured.is_absolute() else code / configured
    if configured.resolve() != source:
        raise ValueError("output differs from the run context")
    process = process_identity(args.pid) if args.pid else None
    if args.pid:
        if not process or process["cwd"] != str(code):
            raise ValueError("process cwd differs from the supplied code directory")
        stdout = (Path("/proc") / str(args.pid) / "fd/1").resolve(strict=True)
        if stdout != console:
            raise ValueError("console differs from the process stdout file")
    view = run_path(root, args.outlet, args.name)
    directory(view.parent)
    directory(view)
    existing = read_entry(view)
    if existing and (existing.get("output") != str(source) or existing.get("code") != str(code)):
        raise ValueError("existing experiment entry belongs to a different run")
    for name, target in (
        ("console.log", console),
        ("metrics.jsonl", source / "metrics.jsonl"),
        ("run_context.json", source / "run_context.json"),
        ("checkpoints", source / "checkpoints"),
        ("code", code),
    ):
        link(view / name, target)
    write_json(
        view / "entry.json",
        {
            "kind": "legacy-view",
            "output": str(source),
            "code": str(code),
            "console": str(console),
            "process": process,
            "source_digest": context.get("identity", {}).get("source", {}).get("digest"),
        },
    )
    return view


def checkout(root: Path, ref: str) -> Path:
    if ref.startswith("-"):
        raise ValueError("Git ref cannot be an option")
    repo = root / "repo"
    commit = git(repo, "rev-parse", "--verify", ref + "^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected an exact Git commit")
    target = root / "checkouts" / commit
    if target.exists() or target.is_symlink():
        if target.is_symlink() or git(target, "rev-parse", "HEAD") != commit:
            raise ValueError("existing checkout has the wrong identity")
        common = Path(git(target, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        if common.resolve() != (repo / ".git").resolve():
            raise ValueError("checkout is not owned by the central repository")
    else:
        git(repo, "worktree", "add", "--detach", str(target), commit)
    if git(target, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("refusing to launch from a dirty checkout")
    return target


def run_logged(command: list[str], code: Path, output: Path, env: dict, metadata: dict) -> int:
    """Keep the trainer's empty-output check intact, including startup failure."""
    output.mkdir()  # Exclusive reservation; no reuse or overwrite.
    handle, name = tempfile.mkstemp(
        prefix=f".{output.name}-starting-", suffix=".log", dir=output.parent
    )
    staged = Path(name)
    published = False
    entry = dict(metadata, code=str(code), output=str(output), status="starting")

    def publish() -> None:
        nonlocal published
        if published:
            return
        if (output / "console.log").exists() or (output / "console.log").is_symlink():
            raise FileExistsError("trainer unexpectedly owns console.log")
        os.replace(staged, output / "console.log")
        link(output / "code", code)
        write_json(output / "entry.json", entry)
        published = True

    try:
        with os.fdopen(handle, "ab", buffering=0) as console:
            process = subprocess.Popen(
                command,
                cwd=code,
                env=env,
                stdout=console,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            entry["process"] = process_identity(process.pid)
            entry["training_pid"] = process.pid
            while process.poll() is None:
                # The context is written only after the trainer's fresh-output
                # validation. Never create metadata inside its empty folder early.
                if (output / "run_context.json").is_file() and not published:
                    entry["status"] = "running"
                    publish()
                time.sleep(0.25)
            entry.update(
                status="complete" if process.returncode == 0 else "failed",
                exit_code=process.returncode,
            )
            publish()
            write_json(output / "entry.json", entry)
            return int(process.returncode)
    except Exception as error:
        entry.update(status="launcher_failed", error=str(error))
        # Preserve the log on failure, including a missing interpreter.
        publish()
        write_json(output / "entry.json", entry)
        raise


def initialization_arguments(args: argparse.Namespace) -> list[str]:
    """Build an unambiguous, read-only model-initialization request."""
    checkpoint = getattr(args, "init_checkpoint", None)
    migration = getattr(args, "init_model_contract_migration", None)
    if migration is not None and migration not in MODEL_CONTRACT_MIGRATIONS:
        raise ValueError("unsupported model-contract migration")
    if migration is not None and checkpoint is None:
        raise ValueError("--init-model-contract-migration requires --init-checkpoint")
    if checkpoint is None:
        return []
    requested = Path(checkpoint).expanduser()
    if not requested.is_absolute():
        raise ValueError("--init-checkpoint must be an absolute checkpoint path")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("--init-checkpoint must identify a checkpoint file")
    result = ["--init-checkpoint", str(resolved)]
    if migration is not None:
        result += ["--init-model-contract-migration", migration]
    return result


def launch(root: Path, args: argparse.Namespace) -> int:
    if os.name != "posix":
        raise ValueError("remote launches require a POSIX host")
    # Preserve a virtualenv interpreter's path. Resolving its symlink to the
    # base binary can select the wrong pyvenv.cfg and a different environment.
    python = Path(args.python).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("--python must be an executable interpreter")
    code = checkout(root, args.ref)
    config = (code / args.config).resolve(strict=True)
    if not config.is_relative_to(code) or not config.is_file():
        raise ValueError("config must belong to the selected commit")
    output = run_path(root, args.outlet, args.name)
    directory(output.parent)
    command = [
        str(python),
        "-B",
        "-u",
        "-m",
        "clearvla.mainline.train",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--device",
        "cuda",
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.workers),
    ]
    command += initialization_arguments(args)
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    if getattr(args, "retain_checkpoint_epochs", None):
        command += ["--retain-checkpoint-epochs", *map(str, args.retain_checkpoint_epochs)]
    if args.smoke:
        command += ["--smoke", "--max-train-batches", "2", "--max-val-batches", "1"]
    environment = {k: v for k, v in os.environ.items() if k in ENVIRONMENT}
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1"
    )
    if not environment.get("PYTORCH_CUDA_ALLOC_CONF"):
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(output, flush=True)
    return run_logged(
        command,
        code,
        output,
        environment,
        {"kind": "managed-run", "commit": code.name, "command": command, "gpu": args.gpu},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="create one namespace and optionally its fixed Git repository"
    )
    init.add_argument("--clone", action="store_true")
    init.add_argument("--remote", default=DEFAULT_REMOTE)
    init.add_argument("--branch", help="explicit management branch for a fresh central clone")
    cp = sub.add_parser("checkout", help="create/reuse a clean detached worktree at a fetched ref")
    cp.add_argument("--ref", required=True)
    sub.add_parser("list", help="list all canonical experiments without guessing process status")
    for name in ("show", "activate", "import-run", "launch"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--outlet", required=True, type=segment)
        cmd.add_argument("--name", required=True, type=segment)
        if name == "import-run":
            for flag in ("output", "console", "code"):
                cmd.add_argument("--" + flag, required=True)
            cmd.add_argument("--pid", type=int)
        if name == "launch":
            for flag in ("ref", "config", "python", "gpu"):
                cmd.add_argument("--" + flag, required=True)
            cmd.add_argument("--batch-size", type=int, default=8)
            cmd.add_argument("--workers", type=int, default=4)
            cmd.add_argument("--epochs", type=int)
            cmd.add_argument("--retain-checkpoint-epochs", type=int, nargs="+")
            cmd.add_argument("--init-checkpoint", type=Path)
            cmd.add_argument(
                "--init-model-contract-migration",
                choices=MODEL_CONTRACT_MIGRATIONS,
            )
            cmd.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
            cmd.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "init" and args.clone and not args.branch:
        raise ValueError("init --clone requires --branch; do not silently choose a model/version")
    if args.command in ("list", "show"):
        root = args.root.expanduser().absolute()
    else:
        root = layout(args.root)
    if args.command == "init":
        if args.clone:
            if (root / "repo").exists():
                if git(root / "repo", "remote", "get-url", "origin") != args.remote:
                    raise ValueError("central repo already has a different origin")
            else:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        args.branch,
                        "--",
                        args.remote,
                        str(root / "repo"),
                    ],
                    check=True,
                )
        if (root / "repo/scripts/clearvla_workspace.py").is_file():
            link(root / "workspace", root / "repo/scripts/clearvla_workspace.py")
            link(
                root / "README.md", root / "repo/docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md"
            )
        print(root)
    elif args.command == "checkout":
        print(checkout(root, args.ref))
    elif args.command == "import-run":
        print(import_run(root, args))
    elif args.command == "activate":
        activate(root, args.outlet, args.name)
        print(root / "active" / args.outlet)
    elif args.command == "launch":
        return launch(root, args)
    elif args.command == "show":
        path = run_path(root, args.outlet, args.name)
        print(json.dumps({"path": str(path), **read_entry(path)}, indent=2))
    else:
        for path in sorted((root / "experiments").glob("*/*")):
            entry = read_entry(path)
            state = (
                "running"
                if is_running(entry)
                else (
                    "unverified" if process_unverified(entry) else entry.get("status", "inactive")
                )
            )
            if state == "running" and not is_running(entry):
                state = "inactive"
            selected = root / "active" / path.parent.name
            current = selected.is_symlink() and selected.resolve() == path.resolve()
            print(
                f"{path.parent.name}/{path.name}\t{state}\t{'active' if current else '-'}\t{path / 'console.log'}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"workspace: {error}", file=sys.stderr)
        raise SystemExit(2)
