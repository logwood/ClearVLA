from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")


def _launcher(script: str, overrides: dict[str, str]) -> list[str]:
    bash = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("launcher argument test requires Bash")
    # Never pass arbitrary account credentials into a shell argument test or
    # expose them through a subprocess failure's pytest traceback.
    inherited = {"PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "TEMP", "TMP"}
    env = {key: value for key, value in os.environ.items() if key.upper() in inherited}
    env.update(overrides)
    # Intercept the command only inside this subprocess. No model, file copy,
    # training or checkpoint read occurs; quoted paths still cross real Bash.
    command = 'exec() { builtin printf "AUDIT_ARG:%s\\n" "$@"; }; source "$1"'
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-c", command, "audit", script],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [
        row.removeprefix("AUDIT_ARG:")
        for row in result.stdout.splitlines()
        if row.startswith("AUDIT_ARG:")
    ]


@pytest.mark.parametrize(
    "script", ["scripts/train_mainline.sh", "scripts/validate_mainline_checkpoint.sh"]
)
def test_launchers_preserve_config_paths_unless_explicitly_relocated(script: str) -> None:
    env = {
        "MAINLINE_CONFIG": "configs/mainline/custom outlet.json",
        "CHECKPOINT": "runs/smoke/last.pt",
    }
    args = _launcher(script, env)
    assert args[args.index("--config") + 1] == env["MAINLINE_CONFIG"]
    for flag in ("--data-root", "--decoded-cache", "--dino-cache", "--t5-condition"):
        assert flag not in args
    relocation = {
        "DATA_ROOT": "/data/custom outlet",
        "CACHE_DIR": "/cache/raw custom",
        "DINO_CACHE_DIR": "/cache/dino custom",
        "T5_CONDITION_PATH": "/models/custom t5.pt",
    }
    explicit = _launcher(script, {**env, **relocation})
    for flag, key in (
        ("--data-root", "DATA_ROOT"),
        ("--decoded-cache", "CACHE_DIR"),
        ("--dino-cache", "DINO_CACHE_DIR"),
        ("--t5-condition", "T5_CONDITION_PATH"),
    ):
        assert explicit[explicit.index(flag) + 1] == relocation[key]
    if "validate" in script:
        assert explicit[explicit.index("--validate-checkpoint") + 1] == env["CHECKPOINT"]
