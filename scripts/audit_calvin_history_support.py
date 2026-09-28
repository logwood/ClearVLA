"""Read-only history-support census of actual admitted training windows.

No model construction, source mutation, sampling change or new policy inputs.
This reports availability, not whether a network actually uses its history.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from typing import Any
from collections import Counter
from pathlib import Path

from clearvla.data.history_clock import sparse_history_clock
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import (
    MainlineDataBundle, load_mainline_data, load_mainline_data_for_smoke,
)


def context_report(bundle: MainlineDataBundle) -> dict[str, Any]:
    splits = {}
    reset_clock = None
    for name, dataset in bundle.datasets.items():
        base = dataset.base
        if not base.config.emit_history_timing or base.config.state_profile != "calvin_relative_7d_v1":
            raise ValueError("history census requires explicit CALVIN source clocks")
        if reset_clock is None:
            reset_clock = sparse_history_clock(
                0, state_offsets=base.config.state_history_offsets,
                action_offsets=base.config.executed_action_offsets,
            )
        counts, starts, first_centers, cached = Counter(), [], {}, {}
        def clock_at(center):
            if center not in cached:
                cached[center] = sparse_history_clock(center,
                    state_offsets=base.config.state_history_offsets,
                    action_offsets=base.config.executed_action_offsets)
            return cached[center]
        for ref in base.refs:
            clock = clock_at(ref.center)
            key = (
                int(clock["history_state_observed"].sum()),
                int(clock["history_action_executed"].sum()),
            )
            counts[key] += 1
            previous = first_centers.get(ref.episode_idx)
            first_centers[ref.episode_idx] = ref.center if previous is None else min(previous,ref.center)
        for episode_index in base.episode_ids:
            first = first_centers.get(episode_index)
            declared = base.instruction_starts.get(episode_index)
            if declared is not None and first is not None:
                clock = clock_at(first)
                starts.append({
                    "episode_index": int(episode_index),
                    "episode_id": base.episodes[episode_index].episode_id,
                    "declared_instruction_start": int(declared),
                    "first_admitted_center": int(first),
                    "instruction_age": int(first-declared),
                    "observed_state_rows": int(clock["history_state_observed"].sum()),
                    "executed_action_rows": int(clock["history_action_executed"].sum()),
                    "previous_step_source_available": bool(first > 0),
                    "four_step_source_available": bool(first >= 4),
                })
        splits[name] = {
            "windows": len(base.refs),
            "selected_modes": {
                "window_boundary_contract": base.config.window_boundary_contract,
                "instruction_reference_mode": base.config.instruction_reference_mode,
                "annotation_goal_mode": base.config.annotation_goal_mode,
                "robot_feedback_mode": base.config.robot_feedback_mode,
                "world_feedback_mode": base.config.world_feedback_mode,
            },
            "context_counts": [
                {"observed_state_rows":s, "executed_action_rows":a, "windows":n}
                for (s,a),n in sorted(counts.items())
            ],
            "instruction_starts": starts,
        }
    if reset_clock is None:
        raise ValueError("history census requires at least one materialized dataset")
    return {
        "schema": "clearvla-history-support-census-v1",
        "scope": "actual admitted dataset centers; no trained-model causality claim",
        "reset_context": {k: v.tolist() for k,v in reset_clock.items()},
        "splits": splits,
        "limitations": [
            "Instructions are not environment resets; recorded preceding history remains valid.",
            "Action-history dropout does not remove preceding state or visual observations.",
            "This census does not measure target relative geometry or content-level coverage.",
            "Only materialized splits are reported; bounded smoke output is not full inventory coverage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--split",choices=("train","val","test"))
    parser.add_argument("--episode-limit", type=int,
                        help="Bounded smoke episodes; requires --split. Default with --split is 1.")
    args = parser.parse_args()
    if args.episode_limit is not None and (args.split is None or args.episode_limit <= 0):
        parser.error("--episode-limit must be positive and requires --split")
    dest = Path(args.output)
    if dest.exists():
        parser.error(f"output already exists; preserving previous evidence: {dest}")
    cfg = load_config(args.config)
    bundle = (load_mainline_data(cfg) if args.split is None else
              load_mainline_data_for_smoke(cfg,split=args.split,episode_limit=args.episode_limit or 1))
    result = context_report(bundle)
    result["config"] = args.config
    result["bounded"] = args.split is not None
    result["requested_episode_limit"] = (args.episode_limit or 1) if args.split else None
    # Runtime paths and normalized config accompany the census; no input bytes are rewritten.
    result["resolved_config_sha256"] = hashlib.sha256(
        json.dumps(cfg.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    # Never overwrite prior diagnostic evidence.
    with dest.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
