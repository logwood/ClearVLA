"""The human-facing root entry cannot silently launch an archived graph."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash is required")
@pytest.mark.parametrize("arguments", [[], ["--help"], ["--legacy-v48", "--flag", "two words"]])
def test_root_entry_requires_explicit_legacy_opt_in(tmp_path: Path, arguments: list[str]) -> None:
    script = tmp_path / "run_current_policy.sh"
    shutil.copyfile(ROOT / script.name, script)
    legacy_dir = tmp_path / "scripts"
    legacy_dir.mkdir()
    (legacy_dir / "current_v48_justok.sh").write_text(
        '#!/usr/bin/env bash\nprintf "legacy:<%s>\\n" "$@"\n', encoding="utf-8"
    )
    result = subprocess.run(
        ["bash", str(script), *arguments], capture_output=True, text=True, timeout=5
    )
    if not arguments:
        assert result.returncode == 2 and "No default policy" in result.stderr
        assert "legacy:<" not in result.stdout
    elif arguments == ["--help"]:
        assert result.returncode == 0 and "clearvla_workspace.py" in result.stdout
        assert "legacy:<" not in result.stdout
    else:
        assert result.returncode == 0
        assert result.stdout.splitlines() == ["legacy:<--flag>", "legacy:<two words>"]
