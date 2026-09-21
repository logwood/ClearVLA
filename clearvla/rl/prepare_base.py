"""Audit native experts and prepare a NEW BC run; never starts training."""

import argparse
import json
from pathlib import Path

import numpy as np

from clearvla.data.instructions import instruction_key
from clearvla.data.split import EPISODE_SPLIT_MANIFEST_SCHEMA
from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.gripper_contract import (
    CONTINUOUS_GRIPPER_OUTPUT_MODE,
    MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
    is_binary_gripper_mode,
)
from clearvla.simulation.admission import (
    STACKCUBE_INSTRUCTION,
    STACKCUBE_PROFILE,
    STACKCUBE_REPAIRED_PROFILE,
    audit_stackcube_experts,
)

from .checkpoint import atomic_json


def prepare(
    root: Path,
    output: Path,
    *,
    cache_root: Path,
    language_bank: Path,
    run_root: Path,
    seed: int = 0,
    profile: str = STACKCUBE_REPAIRED_PROFILE,
    gripper_output_mode: str | None = None,
) -> dict:
    resolved_gripper_mode = (
        gripper_output_mode
        if gripper_output_mode is not None
        else (
            MANISKILL_BINARY_GRIPPER_OUTPUT_MODE
            if profile == STACKCUBE_REPAIRED_PROFILE
            else CONTINUOUS_GRIPPER_OUTPUT_MODE
        )
    )
    audit = audit_stackcube_experts(
        root,
        profile=profile,
        require_binary_gripper=is_binary_gripper_mode(resolved_gripper_mode),
    )
    if audit["episodes"] < 10:
        raise ValueError(
            "base pilot requires at least 10 verified demonstrations; 2-episode conversion smoke is not training data"
        )
    if seed < 0:
        raise ValueError("split seed must be nonnegative")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preparation requires a new empty output directory")
    names = sorted(Path(name).stem for name in audit["lengths"])
    np.random.default_rng(seed).shuffle(names)
    held = max(1, int(round(0.1 * len(names))))
    splits = {"train": names[2 * held :], "val": names[:held], "test": names[held : 2 * held]}
    payload = {
        "data": {
            "raw_hdf5_root": str(root.resolve()),
            "hdf5_glob": "*.hdf5",
            "decoded_cache": str((cache_root / "decoded_336").resolve()),
            "dino_cache": str((cache_root / "dinov2_base_336").resolve()),
            "t5_condition": str(language_bank.resolve()),
            "output_dir": str(run_root.resolve()),
            "data_profile": profile,
            "state_key": "state",
            "action_state_key": "action_state",
            "split_mode": "episode-manifest",
            "split_manifest": str((output / "splits.json").resolve()),
            "train_episodes": 0,
            "val_episodes": 0,
            "test_episodes": 0,
            "sampling_gripper_event_threshold": 0.1,
            "seed": seed,
        },
        "bottom": {
            "arm_flow_mode": "relative_command_adapter",
            "gripper_output_mode": resolved_gripper_mode,
        },
        "objectives": {
            "gripper_command": 0.1
            if is_binary_gripper_mode(resolved_gripper_mode)
            else 0.0,
            "gripper_event_threshold": 0.1,
        },
        "optimizer": {"epochs": 8, "batch_size": 8},
    }
    config_from_mapping(payload).validate()
    instruction = STACKCUBE_INSTRUCTION
    summary = dict(
        schema="clearvla-stackcube-base-preparation-v1",
        dataset=audit,
        split_counts={name: len(rows) for name, rows in splits.items()},
        pretraining_required=True,
        formal_training_started=False,
        reason="same-task BC base required before bounded sparse-reward residual SAC",
        initialization="fresh mainline weights; no Pen/LIBERO or official PPO weight substitution",
        gripper_output_mode=resolved_gripper_mode,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "splits.json",
        dict(
            schema=EPISODE_SPLIT_MANIFEST_SCHEMA,
            split_unit="episode_and_reset_seed",
            split_seed=seed,
            splits=splits,
        ),
    )
    atomic_json(
        output / "instructions.json",
        {
            "schema": "clearvla-instruction-inventory-v1",
            "instructions": [{"key": instruction_key(instruction), "instruction": instruction}],
        },
    )
    atomic_json(output / "config.json", payload)
    # Readiness is a diagnostic stdout summary, not another retained JSON dump.
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--language-bank", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=(STACKCUBE_PROFILE, STACKCUBE_REPAIRED_PROFILE),
                        default=STACKCUBE_REPAIRED_PROFILE)
    parser.add_argument(
        "--gripper-output-mode",
        choices=(CONTINUOUS_GRIPPER_OUTPUT_MODE, MANISKILL_BINARY_GRIPPER_OUTPUT_MODE),
        default=None,
        help="explicitly select the native gripper owner; repaired v2 defaults to binary",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.dataset,
                args.output,
                cache_root=args.cache_root,
                language_bank=args.language_bank,
                run_root=args.run_root,
                seed=args.seed,
                profile=args.profile,
                gripper_output_mode=args.gripper_output_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
