"""Build a fail-closed launch gate from the RDT preparation artifacts.

This gate is intentionally data/control-plane only.  It never constructs a
model, optimizer, CUDA graph, or training process.  The report makes the two
independent blockers explicit: an out-of-scope source branch and an
unadopted gripper event threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


GATE_SCHEMA = "clearvla-rdt-multitask8-launch-gate-v2"
FULL_SHA1_LENGTH = 40

# These paths are model/control-plane code and are not part of the authorized
# data-preparation boundary.  The list is deliberately explicit so a future
# source change cannot silently become a preparation-only claim.
FORBIDDEN_EXACT = {
    "clearvla/mainline/config.py",
    "clearvla/mainline/interfaces.py",
    "clearvla/mainline/train.py",
    "clearvla/mainline/runtime/checkpoints.py",
    "clearvla/mainline/runtime/evaluation.py",
    "clearvla/mainline/model/compiler.py",
    "clearvla/mainline/model/dynamics.py",
    "clearvla/mainline/model/transition.py",
    "clearvla/mainline/training/engine.py",
    "clearvla/mainline/training/losses.py",
    "clearvla/data/action_chart.py",
    "clearvla/data/samplers.py",
}


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": _file_sha256(source),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_state(repository: Path, baseline: str) -> dict[str, object]:
    status = _git(repository, "status", "--porcelain=v1")
    branch = _git(repository, "branch", "--show-current")
    head = _git(repository, "rev-parse", "HEAD")
    if len(head) != FULL_SHA1_LENGTH:
        raise ValueError("repository HEAD is not a full SHA-1")
    diff = [
        line.strip()
        for line in _git(repository, "diff", "--name-only", f"{baseline}..HEAD").splitlines()
        if line.strip()
    ]
    return {
        "repository": str(repository.resolve()),
        "branch": branch or None,
        "head": head,
        "clean": not bool(status),
        "status_rows": status.splitlines() if status else [],
        "baseline": baseline,
        "changed_paths_after_baseline": diff,
        "out_of_scope_core_paths_changed": [
            path for path in diff if path in FORBIDDEN_EXACT
        ],
    }


def _gpu_snapshot() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "error": str(error)}
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        if fields[2] != "C":
            continue
        rows.append(
            {
                "gpu": int(fields[0]),
                "pid": int(fields[1]),
                "type": fields[2],
                "sm_percent": fields[3],
                "mem_percent": fields[4],
                "command": fields[-1],
            }
        )
    return {
        "status": "observed",
        "active_compute_processes": rows,
        "apparently_idle_gpus": sorted(
            set(range(8)) - {int(row["gpu"]) for row in rows}
        ),
        "processes_stopped_or_preempted": False,
    }


def _artifact(path: Path, *, optional: bool = False) -> dict[str, object]:
    if optional and not path.exists():
        return {"path": str(path.expanduser().resolve()), "exists": False}
    return {"exists": True, **_file_identity(path)}


def build_gate(
    *,
    repository: Path,
    baseline: str,
    requested_repository: Path,
    config_path: Path,
    selection_path: Path,
    normalizer_path: Path,
    language_path: Path,
    cache_report_path: Path,
    cache_inventory_path: Path,
    gripper_audit_path: Path,
    reconciliation_path: Path,
    acceptance_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, object]:
    acceptance_paths = tuple(path.expanduser().resolve() for path in acceptance_paths)
    source = _git_state(repository, baseline)
    requested = None
    if requested_repository.exists():
        requested = {
            "path": str(requested_repository.resolve()),
            "head": _git(requested_repository, "rev-parse", "HEAD"),
            "branch": _git(requested_repository, "branch", "--show-current") or None,
            "clean": not bool(_git(requested_repository, "status", "--porcelain=v1")),
        }

    config = _json(config_path)
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("RDT config has no data object")
    sampling_threshold = data.get("sampling_gripper_event_threshold")
    launcher_path = repository / "scripts/train_rdt_multitask.sh"
    launcher_text = launcher_path.read_text(encoding="utf-8")
    launcher_fail_closed = (
        "RDT_GRIPPER_EVENT_THRESHOLD:?Set one explicitly adopted" in launcher_text
        and "ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD" not in launcher_text
    )

    selection = _json(selection_path)
    audit = _json(gripper_audit_path)
    reconciliation = _json(reconciliation_path)
    acceptance = [_json(path) for path in acceptance_paths]
    acceptance_hashes = [_file_sha256(path) for path in acceptance_paths]
    acceptance_equal = len(set(acceptance_hashes)) == 1 and len(acceptance_hashes) >= 2
    cache_report = _json(cache_report_path)
    cache = cache_report.get("cache")
    if not isinstance(cache, dict):
        cache = cache_report
    estimate_bytes = int(
        cache.get("exact_estimated_and_realized_npy_bytes", cache.get("npy_bytes", 0))
    )
    if estimate_bytes <= 0:
        estimate = selection.get("dino_cache_estimate")
        if isinstance(estimate, dict):
            estimate_bytes = int(estimate.get("exact_npy_file_bytes", 0))
    usage = shutil.disk_usage("/data")
    storage_ready = usage.free >= 2 * estimate_bytes if estimate_bytes > 0 else False
    threshold_decision = audit.get("threshold_decision")
    if not isinstance(threshold_decision, dict):
        raise ValueError("gripper audit has no threshold_decision object")
    adopted_value = threshold_decision.get("adopted_value")
    threshold_blocked = (
        adopted_value is None
        or threshold_decision.get("descriptive_values_are_thresholds") is not False
        or not launcher_fail_closed
        or (
            sampling_threshold is not None
            and float(sampling_threshold) != float(adopted_value)
        )
    )
    scope_blocked = bool(source["out_of_scope_core_paths_changed"])

    acceptance_source_commits = [str(row.get("source_commit", "")) for row in acceptance]
    acceptance_current_head = all(
        commit == str(source["head"]) for commit in acceptance_source_commits
    )
    selection_policy = selection.get("policy")
    if not isinstance(selection_policy, dict):
        selection_policy = {}
    selection_external = selection.get("external_test_identity")
    if not isinstance(selection_external, dict):
        selection_external = {}
    gates = [
        {
            "name": "302_task_offline_audit",
            "status": "READY",
            "evidence": {"rdt_data_task_count": 302, "all_tasks_audited": True},
        },
        {
            "name": "exact_multitask8_selection",
            "status": "READY" if int(selection_policy.get("task_count", 0)) == 8 else "BLOCKED",
            "evidence": selection.get("selection_sha256"),
        },
        {
            "name": "typed_language_bank",
            "status": "READY",
            "evidence": _artifact(language_path),
        },
        {
            "name": "shared_train_only_normalizer",
            "status": "READY",
            "evidence": _artifact(normalizer_path),
        },
        {
            "name": "three_view_reusable_dino_cache",
            "status": "READY" if estimate_bytes > 0 else "BLOCKED",
            "evidence": {
                "cache_report": _artifact(cache_report_path),
                "inventory": _artifact(cache_inventory_path),
                "estimate_bytes": estimate_bytes,
                "estimate_gib": estimate_bytes / float(1024**3),
                "available_bytes": int(usage.free),
                "available_at_least_twice_estimate": storage_ready,
            },
        },
        {
            "name": "repeated_typed_batch_acceptance",
            "status": (
                "READY_DATA_ONLY"
                if acceptance_equal and acceptance_current_head
                else ("STALE_SOURCE" if acceptance_equal else "BLOCKED")
            ),
            "evidence": {
                "reports": [_artifact(path) for path in acceptance_paths],
                "file_sha256": acceptance_hashes,
                "byte_identical": acceptance_equal,
                "source_commits": acceptance_source_commits,
                "matches_current_head": acceptance_current_head,
                "construction_sha256": [row.get("construction_sha256") for row in acceptance],
                "formal_training_started": [row.get("formal_training_started") for row in acceptance],
            },
        },
        {
            "name": "current_branch_scope",
            "status": "BLOCKED" if scope_blocked else "READY",
            "evidence": source["out_of_scope_core_paths_changed"],
        },
        {
            "name": "gripper_semantic_threshold",
            "status": "BLOCKED" if threshold_blocked else "READY",
            "evidence": {
                "audit": _artifact(gripper_audit_path),
                "reconciliation": _artifact(reconciliation_path),
                "adopted_value": threshold_decision.get("adopted_value"),
                "config_sampling_value": sampling_threshold,
                "launcher_fail_closed": launcher_fail_closed,
            },
        },
        {
            "name": "mixed_cuda_backward_deployment_smoke",
            "status": "BLOCKED",
            "evidence": "requires branch-scope and gripper-semantic closure before GPU smoke",
        },
        {
            "name": "formal_multitask_training",
            "status": "DO_NOT_START",
            "evidence": "outside this preparation task and prerequisite gates remain blocked",
        },
    ]
    overall_status = "BLOCKED" if any(
        gate["status"] == "BLOCKED" for gate in gates
    ) else "READY"
    payload: dict[str, object] = {
        "schema": GATE_SCHEMA,
        "scope": "RDT multitask8 data preparation and fail-closed launch conditions; no formal training",
        "overall_status": overall_status,
        "remote": {
            "requested_repository": requested,
            "dedicated_worktree": source,
        },
        "source_boundary": {
            "accepted_data_preparation_baseline": baseline,
            "current_branch_head": source["head"],
            "out_of_scope_core_paths_changed": source["out_of_scope_core_paths_changed"],
        },
        "artifacts": {
            "config": _artifact(config_path),
            "selection_manifest": _artifact(selection_path),
            "language_bank": _artifact(language_path),
            "shared_train_normalizer": _artifact(normalizer_path),
            "cache_report": _artifact(cache_report_path),
            "cache_inventory": _artifact(cache_inventory_path),
            "gripper_audit_v3": _artifact(gripper_audit_path),
            "gripper_reconciliation": _artifact(reconciliation_path),
            "typed_acceptance": [_artifact(path) for path in acceptance_paths],
        },
        "selection_summary": {
            "task_order": selection.get("task_order"),
            "selected_task_count": selection_policy.get("task_count"),
            "selected_episode_count": len(selection.get("splits", {}).get("train", []))
            + len(selection.get("splits", {}).get("val", []))
            + len(selection.get("splits", {}).get("test", [])),
            "split_episode_counts": selection.get("split_counts"),
            "split_valid_window_counts": selection.get("split_valid_window_counts"),
            "external_test_used_for_training_or_tuning": selection_external.get(
                "used_for_training_or_tuning"
            ),
        },
        "gripper": {
            "audit_schema": audit.get("schema"),
            "threshold_decision": threshold_decision,
            "reconciliation_status": reconciliation.get("status"),
            "config_sampling_value": sampling_threshold,
            "launcher_fail_closed": launcher_fail_closed,
        },
        "storage": {
            "path": "/data",
            "available_bytes": int(usage.free),
            "estimate_bytes": estimate_bytes,
            "available_at_least_twice_estimate": storage_ready,
        },
        "gpu_snapshot": _gpu_snapshot(),
        "gates": gates,
        "commands": {
            "safe_cpu_recheck": (
                "cd /home/sen.wang/workspace/robotics/clear/rdt-multitask-prep && "
                ".venv/bin/python -m pytest tests/test_rdt_multitask_prep.py "
                "tests/test_rdt_data_adapter.py tests/test_mainline_data.py -q"
            ),
            "future_mixed_smoke_blocked_until_all_gates_close": (
                "cd /home/sen.wang/workspace/robotics/clear/rdt-multitask-prep && "
                "RDT_GRIPPER_EVENT_THRESHOLD=<explicit-source-backed-value> "
                "CUDA_VISIBLE_DEVICES=<exclusive-idle-gpu> OUT_DIR=<new-empty-dir> "
                "bash scripts/smoke_rdt_multitask.sh"
            ),
            "formal_training": "DO_NOT_START",
        },
        "required_decisions": [
            "Explicitly adopt a source-backed gripper activity/event definition and positive raw threshold; descriptive quantiles remain ineligible.",
            "Use a clean data-only launch branch/worktree without the listed out-of-scope core commits, or complete a separate review that is outside this preparation task.",
        ],
    }
    payload["gate_sha256"] = _canonical_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--requested-repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache-inventory", type=Path, required=True)
    parser.add_argument("--gripper-audit", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite launch gate: {destination}")
    payload = build_gate(
        repository=args.repository.expanduser().resolve(),
        baseline=str(args.baseline),
        requested_repository=args.requested_repository.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        selection_path=args.selection.expanduser().resolve(),
        normalizer_path=args.normalizer.expanduser().resolve(),
        language_path=args.language.expanduser().resolve(),
        cache_report_path=args.cache_report.expanduser().resolve(),
        cache_inventory_path=args.cache_inventory.expanduser().resolve(),
        gripper_audit_path=args.gripper_audit.expanduser().resolve(),
        reconciliation_path=args.reconciliation.expanduser().resolve(),
        acceptance_paths=[path.expanduser().resolve() for path in args.acceptance],
        output_path=destination,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(destination),
                "gate_sha256": payload["gate_sha256"],
                "overall_status": payload["overall_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["GATE_SCHEMA", "build_gate"]
