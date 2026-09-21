#!/usr/bin/env python3
"""Summarize the read-only CALVIN intent/object-memory intervention probe.

The probe JSON intentionally stores full arrays so this reducer uses only the
Python standard library.  It reports language effects at each coarse-read seam,
intervention effects relative to the native object memory, and a permutation-
invariant comparison of visual object facts across layouts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _flat(value: Any) -> Iterable[float]:
    if isinstance(value, list):
        for item in value:
            yield from _flat(item)
    else:
        yield float(value)


def _rmse(left: Any, right: Any) -> float:
    a = list(_flat(left))
    b = list(_flat(right))
    if len(a) != len(b):
        raise ValueError(f"array length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def _array(record: dict[str, Any], name: str) -> Any:
    return record[name]["array"]


def _row_permute(value: Any, permutation: tuple[int, ...]) -> Any:
    # All fact arrays have a leading batch dimension and K as axis one.
    return [[value[0][index] for index in permutation]]


def _best_fact_rmse(left: Any, right: Any) -> float:
    k = len(left[0])
    return min(_rmse(left, _row_permute(right, p)) for p in itertools.permutations(range(k)))


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    layouts = data["layouts"]
    branch_names = ("intent_read", "object_read", "history_read")
    print(f"schema={data.get('schema')} layouts={','.join(layouts)}")

    all_object_language_rmse: list[float] = []
    for layout, layout_record in layouts.items():
        instructions = layout_record["instructions"]
        reference_name = "go push the blue block right"
        if reference_name not in instructions:
            reference_name = next(iter(instructions))
        reference = instructions[reference_name]["interventions"]["baseline"]
        print(f"\nLAYOUT {layout} reference={reference_name}")
        for instruction, record in instructions.items():
            baseline = record["interventions"]["baseline"]
            branch_effects = {
                name: _rmse(
                    baseline["branches"][name]["delta"]["array"],
                    reference["branches"][name]["delta"]["array"],
                )
                for name in branch_names
            }
            coarse_effect = _rmse(
                baseline["coarse_interval_action"]["array"],
                reference["coarse_interval_action"]["array"],
            )
            w_effect = _rmse(
                baseline["w_semantic_delta"]["array"],
                reference["w_semantic_delta"]["array"],
            )
            all_object_language_rmse.append(branch_effects["object_read"])
            print(
                "LANG",
                instruction,
                "coarse=" + _fmt(coarse_effect),
                "W=" + _fmt(w_effect),
                "intent=" + _fmt(branch_effects["intent_read"]),
                "object=" + _fmt(branch_effects["object_read"]),
                "history=" + _fmt(branch_effects["history_read"]),
            )
            for mode in ("object_zero", "object_mean", "object_permute"):
                item = record["interventions"][mode]
                delta = item["delta_vs_baseline"]
                print(
                    "  INTERVENTION",
                    mode,
                    "coarse=" + _fmt(delta["coarse_interval_action_rmse"]),
                    "W=" + _fmt(delta["w_semantic_delta_rmse"]),
                    "object_branch=" + _fmt(delta["object_read_delta_rmse"]),
                    (
                        "velocity=" + _fmt(delta["physical_velocity_rmse"])
                        if "physical_velocity_rmse" in delta
                        else ""
                    ),
                    (
                        "action=" + _fmt(delta["deployed_action_normalized_rmse"])
                        if "deployed_action_normalized_rmse" in delta
                        else ""
                    ),
                    (
                        "arm=" + _fmt(delta["deployed_arm_normalized_rmse"])
                        if "deployed_arm_normalized_rmse" in delta
                        else ""
                    ),
                    (
                        "gripper_disagree="
                        + _fmt(delta["gripper_command_disagreement_rate"])
                        if "gripper_command_disagreement_rate" in delta
                        else ""
                    ),
                )

    if all_object_language_rmse:
        print(
            "\nMAX_OBJECT_BRANCH_LANGUAGE_RMSE="
            + _fmt(max(all_object_language_rmse))
        )

    print("\nFACT_CROSS_LAYOUT_BEST_PERMUTATION_RMSE")
    names = list(layouts)
    if len(names) >= 2:
        reference_instruction = "go push the blue block right"
        fields = ("semantic", "appearance", "geometry", "object_memory")
        for left_name, right_name in itertools.combinations(names, 2):
            left = layouts[left_name]["instructions"][reference_instruction]["facts"]
            right = layouts[right_name]["instructions"][reference_instruction]["facts"]
            values = {
                field: _best_fact_rmse(_array(left, field), _array(right, field))
                for field in fields
            }
            print(left_name, "vs", right_name, " ".join(f"{k}={_fmt(v)}" for k, v in values.items()))


if __name__ == "__main__":
    main()
