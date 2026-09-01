"""Reconcile historical RDT gripper audits without choosing a threshold.

The first audit version mixed a command value with converted qpos at the first
window boundary.  Later audits corrected the boundary but still exposed their
quantiles under a threshold-like name.  This small producer records the
rejection/supersession chain and the file digests so stale numbers cannot be
used accidentally by a launcher or a human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RECONCILIATION_SCHEMA = "clearvla-rdt-gripper-audit-reconciliation-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"audit must be a JSON object: {path}")
    return value


def _legacy_values(value: dict[str, Any]) -> list[dict[str, object]]:
    rows = value.get("candidate_thresholds", [])
    if not isinstance(rows, list):
        return []
    result: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "source_quantile": row.get("source_quantile"),
                "numeric_value": row.get("threshold"),
                "event_window_fraction": row.get("event_window_fraction"),
            }
        )
    return result


def _descriptive_values(value: dict[str, Any]) -> list[dict[str, object]]:
    rows = value.get("descriptive_activity_quantiles", [])
    if not isinstance(rows, list):
        return []
    result: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "source_quantile": row.get("source_quantile"),
                "raw_abs_adjacent_command_delta_quantile": row.get(
                    "raw_abs_adjacent_command_delta_quantile"
                ),
                "source_units": row.get("source_units"),
                "eligible_for_threshold_adoption": row.get(
                    "eligible_for_threshold_adoption"
                ),
            }
        )
    return result


def build_reconciliation(
    *,
    legacy_path: Path,
    corrected_path: Path,
    authoritative_path: Path,
) -> dict[str, object]:
    legacy = _load(legacy_path)
    corrected = _load(corrected_path)
    authoritative = _load(authoritative_path)
    if legacy.get("schema") != "clearvla-rdt-multitask-gripper-train-audit-v1":
        raise ValueError("legacy audit is not the expected v1 mixed-boundary artifact")
    if corrected.get("schema") != "clearvla-rdt-multitask-gripper-train-audit-v2":
        raise ValueError("corrected audit is not the expected v2 artifact")
    if authoritative.get("schema") != "clearvla-rdt-multitask-gripper-train-audit-v3":
        raise ValueError("authoritative audit is not the expected v3 artifact")
    decision = authoritative.get("threshold_decision")
    if not isinstance(decision, dict) or decision.get("adopted_value") is not None:
        raise ValueError("authoritative audit must have no adopted threshold")
    if decision.get("descriptive_values_are_thresholds") is not False:
        raise ValueError("authoritative quantiles must be descriptive only")

    payload: dict[str, object] = {
        "schema": RECONCILIATION_SCHEMA,
        "status": "legacy_rejected_corrected_superseded_authoritative_descriptive_only",
        "training_threshold": None,
        "training_ready": False,
        "reason": {
            "legacy_v1": (
                "The first sampler row mixed action command with converted qpos; "
                "all v1 candidate values are non-comparable and rejected."
            ),
            "corrected_v2": (
                "The adjacent-command boundary is numerically corrected, but its "
                "threshold-like field names are superseded by v3 descriptive names."
            ),
            "authoritative_v3": (
                "Only train-split adjacent-command distributions are reported; no "
                "quantile is eligible for training until source semantics are adopted."
            ),
        },
        "artifacts": {
            "legacy_v1": {
                "path": str(legacy_path.resolve()),
                "sha256": _file_sha256(legacy_path),
                "schema": legacy.get("schema"),
                "use_for_training": False,
                "values": _legacy_values(legacy),
            },
            "corrected_v2": {
                "path": str(corrected_path.resolve()),
                "sha256": _file_sha256(corrected_path),
                "schema": corrected.get("schema"),
                "use_for_training": False,
            },
            "authoritative_v3": {
                "path": str(authoritative_path.resolve()),
                "sha256": _file_sha256(authoritative_path),
                "schema": authoritative.get("schema"),
                "use_for_training": False,
                "values": _descriptive_values(authoritative),
            },
        },
    }
    payload["reconciliation_sha256"] = _canonical_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite reconciliation: {destination}")
    payload = build_reconciliation(
        legacy_path=args.legacy.expanduser().resolve(),
        corrected_path=args.corrected.expanduser().resolve(),
        authoritative_path=args.authoritative.expanduser().resolve(),
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
                "reconciliation_sha256": payload["reconciliation_sha256"],
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["RECONCILIATION_SCHEMA", "build_reconciliation"]
