"""The review gate cannot trade an old error for a new error in another owner."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_structural_static import type_counter


def _payload(path: Path, line: int) -> dict[str, object]:
    return {
        "generalDiagnostics": [
            {
                "file": str(path),
                "severity": "error",
                "rule": "rule",
                "message": "same diagnostic",
                "range": {"start": {"line": line - 1}},
            }
        ]
    }


def test_type_gate_retains_multiplicity(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text("def a():\n    return 1\n")
    counter = type_counter(_payload(path, 2), tmp_path)
    assert sum(((counter + counter) - counter).values()) == 1


def test_type_gate_follows_definition_across_line_shift(tmp_path: Path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "module.py").write_text("def a():\n    return 1\n")
    (new / "module.py").write_text("# inserted\n\ndef a():\n    return 1\n")
    before = type_counter(_payload(old / "module.py", 2), old)
    after = type_counter(_payload(new / "module.py", 4), new)
    assert before == after


def test_type_gate_rejects_same_message_in_different_definition(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text("def a():\n    return 1\ndef b():\n    return 1\n")
    before = type_counter(_payload(path, 2), tmp_path)
    after = type_counter(_payload(path, 4), tmp_path)
    assert sum((after - before).values()) == 1
