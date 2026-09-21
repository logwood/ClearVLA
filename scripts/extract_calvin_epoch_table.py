#!/usr/bin/env python3
"""Print a compact epoch table from a ClearVLA independent-mainline log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (
    "loss_total",
    "loss_action_flow",
    "loss_action_flow_native",
    "loss_decoded_action",
    "validation_action_rmse_physical",
    "validation_action_rmse_normalized",
    "validation_first_rmse_physical",
    "validation_first8_rmse_physical",
    "validation_tail_rmse_physical",
    "validation_tail_first_ratio_physical",
    "validation_arm_rmse_physical",
    "validation_arm_rmse_normalized",
    "validation_gripper_rmse_physical",
    "validation_gripper_rmse_normalized",
    "validation_decoded_gripper_event_precision",
    "validation_decoded_gripper_event_recall",
    "validation_decoded_gripper_event_f1",
    "validation_decoded_gripper_event_ratio",
    "validation_decoded_gripper_events_predicted",
    "validation_decoded_gripper_events_target",
)


def _pick(values: object) -> dict[str, object]:
    if not isinstance(values, dict):
        return {}
    return {field: values[field] for field in FIELDS if field in values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    with args.metrics.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") != "epoch":
                continue
            train = row.get("train")
            validation = row.get("validation")
            print(
                json.dumps(
                    {
                        "epoch": row.get("epoch"),
                        "step": row.get("step"),
                        "train": _pick(train),
                        "validation": _pick(validation),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
