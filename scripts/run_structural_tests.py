#!/usr/bin/env python3
"""Run the full selected regression inventory with bounded process lifetime.

Each file runs in a fresh pytest process. This bounds accumulated PyTorch/pytest
allocator state without dropping tests, changing fixtures or accepting child
failures. Per-file output is retained; an abnormal/empty run is an explicit
JUnit error and a failed summary, not a missing or passing test group.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def merge_junit(paths: list[Path], failures: list[tuple[str, str]], output: Path) -> dict[str, int]:
    root = ET.Element("testsuites")
    counts: dict[str, int] = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    for path in paths:
        parsed = ET.parse(path).getroot()
        suites = [parsed] if parsed.tag == "testsuite" else list(parsed.findall("testsuite"))
        if not suites or sum(int(s.get("tests", "0")) for s in suites) == 0:
            raise ValueError(f"empty JUnit result: {path}")
        for suite in suites:
            root.append(suite)
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
    for filename, message in failures:
        suite = ET.SubElement(root, "testsuite", name="process_failure", tests="1", errors="1")
        case = ET.SubElement(suite, "testcase", classname="structural_runner", name=filename)
        ET.SubElement(case, "error", message=message).text = message
        counts["tests"] += 1
        counts["errors"] += 1
    for key, value in counts.items():
        root.set(key, str(value))
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return counts


def run_inventory(root: Path, files: list[str], output: Path, label: str, timeout: float) -> int:
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError("test output must be outside the source checkout")
    if not files or len(set(files)) != len(files):
        raise ValueError("test inventory must be non-empty and unique")
    if not label or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label
    ):
        raise ValueError("label must be a simple filename component")
    for filename in files:
        path = (root / filename).resolve()
        if Path(filename).is_absolute() or root not in path.parents or not path.is_file():
            raise ValueError(f"test must exist inside the checkout: {filename}")
    directory = output / f"{label}-test-files"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{label}-test-processes.json"
    report: dict[str, Any] = {
        "root": str(root),
        "inventory": files,
        "isolation": "one pytest process per file",
        "complete": False,
        "passed": False,
        "processes": [],
    }

    def save_report() -> None:
        pending = report_path.with_suffix(".tmp")
        pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.replace(report_path)

    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Do not accidentally import a sibling/current tree when checking baseline.
    environment.pop("PYTHONPATH", None)
    results: list[Path] = []
    process_failures: list[tuple[str, str]] = []
    save_report()
    for index, filename in enumerate(files):
        stem = f"{index:03d}-{Path(filename).stem}"
        xml, log = directory / f"{stem}.xml", directory / f"{stem}.txt"
        xml.unlink(missing_ok=True)  # a previous run must not become this child's result
        started = time.monotonic()
        error: str | None = None
        code: int | None = None
        print(f"[{label} {index + 1}/{len(files)}] {filename}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        filename,
                        "--tb=short",
                        f"--junitxml={xml}",
                    ],
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                error = f"pytest file exceeded {timeout:g} seconds"
        valid_xml = False
        if xml.is_file():
            try:
                parsed = ET.parse(xml).getroot()
                suites = (
                    [parsed] if parsed.tag == "testsuite" else list(parsed.findall("testsuite"))
                )
                valid_xml = bool(suites) and sum(int(s.get("tests", "0")) for s in suites) > 0
            except (ET.ParseError, ValueError):
                valid_xml = False
        if valid_xml:
            results.append(xml)
        if code != 0 or not valid_xml:
            error = error or f"pytest exit={code}, non-empty valid JUnit={valid_xml}"
            # Even an assertion failure has a separate process marker. Its
            # original test cases remain in JUnit and raw log; neither is lost.
            process_failures.append((filename, error))
        report["processes"].append(
            {
                "file": filename,
                "returncode": code,
                "error": error,
                "seconds": time.monotonic() - started,
                "log": str(log),
                "junit": str(xml),
            }
        )
        save_report()
    report["counts"] = merge_junit(results, process_failures, output / f"{label}-tests.xml")
    report["complete"] = True
    report["passed"] = (
        not process_failures and not report["counts"]["errors"] and not report["counts"]["failures"]
    )
    save_report()
    print(json.dumps({"label": label, "passed": report["passed"], **report["counts"]}), flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout-per-file", type=float, default=600.0)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.timeout_per_file <= 0:
        raise ValueError("timeout must be positive")
    return run_inventory(args.root, args.files, args.output, args.label, args.timeout_per_file)


if __name__ == "__main__":
    raise SystemExit(main())
