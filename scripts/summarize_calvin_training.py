#!/usr/bin/env python3
"""Extract compact epoch-level training/validation diagnostics from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KEYS = (
    "loss",
    "physical_flow",
    "physical_flow_native",
    "decoded_action",
    "full_rmse",
    "first_rmse",
    "first8_rmse",
    "tail_rmse",
    "tail_first_ratio",
    "arm_full_rmse",
    "gripper_full_rmse",
    "gripper_event_precision",
    "gripper_event_recall",
    "gripper_event_f1",
    "gripper_event_ratio",
    "gripper_pred_events",
    "gripper_target_events",
    "execution_ablation_primary_full_rmse",
    "execution_ablation_hard_full_rmse",
    "execution_ablation_neutral_full_rmse",
    "execution_ablation_full_capacity_full_rmse",
    "execution_ablation_three_basis_reduction_full_rmse",
)


def _compact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in KEYS or any(
            token in str(key)
            for token in (
                "full_rmse",
                "first8_rmse",
                "first_rmse",
                "tail_rmse",
                "tail_first_ratio",
                "arm_full_rmse",
                "gripper_full_rmse",
                "gripper_event_",
                "physical_flow",
                "decoded_action",
                "loss_total",
                "loss_action_flow",
            )
        ):
            result[str(key)] = item
    return result


def _find_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Handle both nested epoch schemas and flat metric snapshots."""
    candidates: list[dict[str, Any]] = []
    for name in ("validation", "val", "multitask_validation", "train"):
        value = row.get(name)
        if isinstance(value, dict):
            candidates.append(value)
            nested = value.get("metrics")
            if isinstance(nested, dict):
                candidates.append(nested)
    for name in ("metrics",):
        value = row.get(name)
        if isinstance(value, dict):
            candidates.append(value)
    merged: dict[str, Any] = {}
    for candidate in candidates:
        merged.update(candidate)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    epochs: list[dict[str, Any]] = []
    batch_last: dict[str, Any] | None = None
    debug_keys: dict[str, Any] | None = None
    with args.metrics.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "epoch":
                train = row.get("train") if isinstance(row.get("train"), dict) else {}
                validation = row.get("validation") or row.get("val") or row.get("multitask_validation")
                # Some independent-mainline snapshots store all public values
                # under the epoch's train/validation object with a prefix.
                train = _find_metrics({"train": train})
                validation = _find_metrics({"validation": validation})
                if debug_keys is None:
                    debug_keys = {
                        "row_keys": sorted(row.keys()),
                        "train_keys": sorted(train.keys())[:80],
                        "validation_keys": sorted(validation.keys())[:80],
                        "validation_metric_names": sorted(
                            key
                            for key in validation
                            if any(token in key for token in ("rmse", "gripper_event"))
                        ),
                    }
                epochs.append(
                    {
                        "epoch": row.get("epoch"),
                        "step": row.get("step"),
                        "train": _compact(train),
                        "validation": _compact(validation),
                    }
                )
            elif row.get("kind") == "train":
                batch_last = _find_metrics(row)
    print(json.dumps({"debug_keys": debug_keys, "epochs": epochs, "last_batch": _compact(batch_last)}, indent=2))


if __name__ == "__main__":
    main()
