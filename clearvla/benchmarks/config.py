"""Derive an isolated mainline config for one converted benchmark dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clearvla.data.split import load_episode_split_manifest_inventory
from clearvla.mainline.config import config_from_mapping

from .common import (
    LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA,
    LIBERO_CONVERTER_SCHEMA,
    LIBERO_CONVERTER_SCHEMAS,
    LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA,
    atomic_json,
    audit_benchmark_dataset,
)

_BENCHMARK_OUTLET_CONTRACTS: dict[str, dict[str, str]] = {
    "CALVIN": {
        "data_profile": "calvin_relative_7d_v1",
        "arm_flow_mode": "relative_command_direct",
        "gripper_output_mode": "calvin_binary_command",
    },
    "LIBERO": {
        "data_profile": "libero_relative_7d_v1",
        "arm_flow_mode": "relative_command_direct",
        "gripper_output_mode": "continuous",
    },
}


def _load_dataset_manifest(root: Path) -> dict[str, Any] | None:
    """Read the optional converted-root identity without broadening old tests.

    The split-only fixture used by compatibility callers predates benchmark
    manifests, so absence remains allowed.  A real converted root carries the
    manifest and therefore gets an early outlet/profile consistency check.
    """

    path = root / "dataset_manifest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid benchmark dataset manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid benchmark dataset manifest: {path}")
    if value.get("schema") != "clearvla-external-benchmark-dataset-v1":
        raise ValueError(f"benchmark dataset manifest schema differs: {path}")
    benchmark = str(value.get("benchmark", "")).upper()
    contract = _BENCHMARK_OUTLET_CONTRACTS.get(benchmark)
    if contract is None:
        raise ValueError(f"unknown converted benchmark identity {benchmark!r}: {path}")
    # A real LIBERO conversion must carry the complete outlet identity.  The
    # manifest is the only data-side guard before a generic Pen base config is
    # adapted, so treating these fields as optional would permit a silent
    # semantic fallback.  CALVIN manifests from the older converter remain
    # readable; their fields are checked when present for compatibility.
    required = set(contract) if benchmark == "LIBERO" else set()
    if benchmark == "LIBERO":
        required.update(
            {
                "converter_schema",
                "controller",
                "cameras",
                "split_unit",
                "valid_center_start",
                "valid_center_end",
                "evaluator_language_verified",
            }
        )
    missing = sorted(field for field in required if field not in value)
    if missing:
        raise ValueError(
            f"{path} is missing required {benchmark} manifest fields: "
            + ", ".join(missing)
        )
    converter_schema = str(value.get("converter_schema", ""))
    if benchmark == "LIBERO" and converter_schema not in LIBERO_CONVERTER_SCHEMAS:
        raise ValueError(f"{path} has unsupported LIBERO converter_schema")
    if benchmark == "LIBERO" and converter_schema != LIBERO_CONVERTER_SCHEMA:
        causal_required = {
            "window_boundary_contract",
            "strict_valid_center_start",
            "strict_valid_center_end",
            "observation_alignment",
            "state_normalizer_reference_semantics",
        }
        if converter_schema == LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA:
            causal_required.update({"terminal_padding_mode", "terminal_suffix_rows"})
        causal_missing = sorted(
            field for field in causal_required if field not in value
        )
        if causal_missing:
            raise ValueError(
                f"{path} is missing causal LIBERO manifest fields: "
                + ", ".join(causal_missing)
            )
    for field, expected in contract.items():
        declared = value.get(field)
        if declared is None or str(declared) != expected:
            raise ValueError(
                f"{path} declares {field}={declared!r}, expected {expected!r}"
            )
    if benchmark == "LIBERO":
        if value.get("evaluator_language_verified") is not True:
            raise ValueError(
                f"{path} LIBERO evaluator language must be verified from official BDDL tasks"
            )
        if value.get("controller") != "OSC_POSE":
            raise ValueError(f"{path} LIBERO controller must be OSC_POSE")
        if value.get("cameras") != ["agentview_rgb", "eye_in_hand_rgb"]:
            raise ValueError(
                f"{path} LIBERO cameras must be ['agentview_rgb', 'eye_in_hand_rgb']"
            )
        if value.get("split_unit") != "episode":
            raise ValueError(f"{path} LIBERO split_unit must be episode")
        if converter_schema == LIBERO_CONVERTER_SCHEMA:
            expected_boundary = (24, "length - 49", None)
        elif converter_schema == LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA:
            expected_boundary = (0, "length - 49", "causal_prefix_v1")
        elif converter_schema == LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA:
            expected_boundary = (
                0,
                "source_action_count - 1",
                "causal_prefix_terminal_suffix_v2",
            )
        else:
            raise AssertionError("unreachable LIBERO converter schema")
        expected_start, expected_end, expected_window_contract = expected_boundary
        if (
            value.get("valid_center_start") != expected_start
            or value.get("valid_center_end") != expected_end
        ):
            raise ValueError(f"{path} LIBERO valid-center contract is stale")
        if expected_window_contract is not None and value.get(
            "window_boundary_contract"
        ) != expected_window_contract:
            raise ValueError(f"{path} LIBERO window-boundary contract differs")
    return value


def build_benchmark_config(
    base: Path,
    output: Path,
    *,
    hdf5_root: Path,
    cache_root: Path,
    language_bank: Path,
    run_root: Path,
    num_workers: int | None = None,
    data_profile: str | None = None,
    arm_flow_mode: str | None = None,
    gripper_output_mode: str | None = None,
    gripper_command_weight: float | None = None,
    gripper_event_threshold: float | None = None,
    window_boundary_contract: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("base mainline config has no data mapping")
    data = payload["data"]
    resolved_hdf5_root = hdf5_root.resolve()
    data.update(
        {
            "raw_hdf5_root": str(resolved_hdf5_root),
            "decoded_cache": str((cache_root / "decoded_336").resolve()),
            "dino_cache": str((cache_root / "dinov2_base_336").resolve()),
            "t5_condition": str(language_bank.resolve()),
            "output_dir": str(run_root.resolve()),
            "action_key": "action",
            "action_state_key": "action_state",
            "state_key": "state",
            "top_camera_key": "observations/images/cam_high",
            "wrist_camera_key": "observations/images/cam_right_wrist",
            # CALVIN/LIBERO converters emit the compact three-way episode
            # manifest.  ``manifest`` is reserved for the stricter RDT
            # per-task four-lane manifest; selecting it here would make a
            # converted benchmark fail in ``load_rdt_split_manifest``.
            "split_mode": "episode-manifest",
            # Manifest membership owns the episode counts.  Leaving the
            # ordered-count fields at zero is part of DataConfig's contract;
            # populating them with the manifest lengths makes an otherwise
            # valid benchmark config fail closed during validation.
            "split_manifest": str((resolved_hdf5_root / "splits.json").resolve()),
            "train_episodes": 0,
            "val_episodes": 0,
            "test_episodes": 0,
        }
    )
    split_manifest = resolved_hdf5_root / "splits.json"
    if not split_manifest.is_file():
        raise FileNotFoundError(
            f"converted benchmark root has no split manifest: {split_manifest}"
        )
    # Use the same strict parser as the training loader.  The former ad-hoc
    # list check accepted a wrong schema, duplicate membership and omitted
    # episodes, allowing an outlet config to be built for a different root.
    _, split_payload, _ = load_episode_split_manifest_inventory(split_manifest)
    data["task_filter"] = str(split_payload.get("task_filter", ""))
    dataset_manifest = _load_dataset_manifest(resolved_hdf5_root)
    outlet_contract = None
    if dataset_manifest is not None:
        outlet_contract = _BENCHMARK_OUTLET_CONTRACTS[
            str(dataset_manifest["benchmark"]).upper()
        ]
        # A converted benchmark owns its outlet identity.  Do not let a
        # generic Pen base config silently leak into LIBERO (or CALVIN).
        resolved_profile = (
            str(data_profile) if data_profile is not None else outlet_contract["data_profile"]
        )
        resolved_arm_mode = (
            str(arm_flow_mode)
            if arm_flow_mode is not None
            else outlet_contract["arm_flow_mode"]
        )
        resolved_gripper_mode = (
            str(gripper_output_mode)
            if gripper_output_mode is not None
            else outlet_contract["gripper_output_mode"]
        )
        if resolved_profile != outlet_contract["data_profile"]:
            raise ValueError(
                f"{dataset_manifest['benchmark']} data profile must be "
                f"{outlet_contract['data_profile']}, got {resolved_profile!r}"
            )
        if resolved_arm_mode != outlet_contract["arm_flow_mode"]:
            raise ValueError(
                f"{dataset_manifest['benchmark']} arm flow mode must be "
                f"{outlet_contract['arm_flow_mode']}, got {resolved_arm_mode!r}"
            )
        if resolved_gripper_mode != outlet_contract["gripper_output_mode"]:
            raise ValueError(
                f"{dataset_manifest['benchmark']} gripper output mode must be "
                f"{outlet_contract['gripper_output_mode']}, got {resolved_gripper_mode!r}"
            )
        data["data_profile"] = resolved_profile
        bottom = payload.setdefault("bottom", {})
        if not isinstance(bottom, dict):
            raise ValueError("base mainline config bottom must be a mapping")
        bottom["arm_flow_mode"] = resolved_arm_mode
        bottom["gripper_output_mode"] = resolved_gripper_mode
        if str(dataset_manifest["benchmark"]).upper() == "LIBERO":
            converter_schema = str(dataset_manifest["converter_schema"])
            manifest_window_contract = (
                "strict_complete_v1"
                if converter_schema == LIBERO_CONVERTER_SCHEMA
                else str(dataset_manifest["window_boundary_contract"])
            )
            if (
                window_boundary_contract is not None
                and str(window_boundary_contract) != manifest_window_contract
            ):
                raise ValueError(
                    "LIBERO config window boundary differs from the converted-root "
                    f"contract {manifest_window_contract!r}"
                )
            # The original audited converter already matches the current
            # loader's strict in-episode window rule and needs no new config
            # field.  Prefix/terminal roots remain gated on the separate
            # opt-in window-boundary integration unit.
            if converter_schema != LIBERO_CONVERTER_SCHEMA:
                data["window_boundary_contract"] = manifest_window_contract
    elif window_boundary_contract is not None:
        data["window_boundary_contract"] = str(window_boundary_contract)
    # A fully materialized benchmark conversion carries a task inventory.  In
    # that case audit every episode before emitting a training config; a
    # split-only compatibility fixture (which predates benchmark manifests)
    # remains intentionally lightweight and is validated by the loader later.
    if dataset_manifest is not None and "tasks" in dataset_manifest:
        audit_benchmark_dataset(resolved_hdf5_root)
    if num_workers is not None:
        data["num_workers"] = int(num_workers)
    if data_profile is not None:
        data["data_profile"] = str(data_profile)
    if gripper_event_threshold is not None:
        data["sampling_gripper_event_threshold"] = float(gripper_event_threshold)
        objectives = payload.setdefault("objectives", {})
        if not isinstance(objectives, dict):
            raise ValueError("base mainline config objectives must be a mapping")
        objectives["gripper_event_threshold"] = float(gripper_event_threshold)
    if gripper_output_mode is not None:
        bottom = payload.setdefault("bottom", {})
        if not isinstance(bottom, dict):
            raise ValueError("base mainline config bottom must be a mapping")
        bottom["gripper_output_mode"] = str(gripper_output_mode)
    if arm_flow_mode is not None:
        bottom = payload.setdefault("bottom", {})
        if not isinstance(bottom, dict):
            raise ValueError("base mainline config bottom must be a mapping")
        bottom["arm_flow_mode"] = str(arm_flow_mode)
    if gripper_command_weight is not None:
        objectives = payload.setdefault("objectives", {})
        if not isinstance(objectives, dict):
            raise ValueError("base mainline config objectives must be a mapping")
        objectives["gripper_command"] = float(gripper_command_weight)
    config_from_mapping(payload).validate()
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--language-bank", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data-profile", default=None)
    parser.add_argument("--arm-flow-mode", default=None)
    parser.add_argument("--gripper-output-mode", default=None)
    parser.add_argument("--gripper-command-weight", type=float, default=None)
    parser.add_argument("--gripper-event-threshold", type=float, default=None)
    parser.add_argument("--window-boundary-contract", default=None)
    args = parser.parse_args()
    payload = build_benchmark_config(
        args.base,
        args.output,
        hdf5_root=args.hdf5_root,
        cache_root=args.cache_root,
        language_bank=args.language_bank,
        run_root=args.run_root,
        num_workers=args.num_workers,
        data_profile=args.data_profile,
        arm_flow_mode=args.arm_flow_mode,
        gripper_output_mode=args.gripper_output_mode,
        gripper_command_weight=args.gripper_command_weight,
        gripper_event_threshold=args.gripper_event_threshold,
        window_boundary_contract=args.window_boundary_contract,
    )
    print(json.dumps(payload["data"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_benchmark_config"]
