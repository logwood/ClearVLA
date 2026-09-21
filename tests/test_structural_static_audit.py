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


def test_junit_merge_retains_each_source_test_and_records_killed_process(tmp_path: Path):
    import xml.etree.ElementTree as ET

    from scripts.run_structural_tests import merge_junit

    xml = tmp_path / "one.xml"
    xml.write_text(
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1">'
        '<testcase name="ok"/><testcase name="bad"><failure/></testcase>'
        '<testcase name="skip"><skipped/></testcase></testsuite></testsuites>'
    )
    output = tmp_path / "merged.xml"
    counts = merge_junit([xml], [("two.py", "process killed")], output)
    assert counts == {"tests": 4, "failures": 1, "errors": 1, "skipped": 1}
    root = ET.parse(output).getroot()
    assert len(root.findall("testsuite")) == 2
    assert len(root.findall(".//testcase")) == 4
    assert root.find(".//error") is not None


def test_junit_merge_rejects_empty_file(tmp_path: Path):
    import pytest

    from scripts.run_structural_tests import merge_junit

    xml = tmp_path / "empty.xml"
    xml.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(ValueError, match="empty JUnit"):
        merge_junit([xml], [], tmp_path / "out.xml")
