"""Fail-closed native StackCube evidence admission, independent of RL math."""

import json
from pathlib import Path

import h5py
import numpy as np

from .dataset import SIM_DATASET_SCHEMA, validate_recorded_episode
from .state_chart import CONTINUOUS_STATE_CHART, CONTINUOUS_STATE_SEMANTICS, LEGACY_STATE_SEMANTICS

STACKCUBE_PROFILE = "maniskill_pd_ee_delta_pose_7d_v1"
STACKCUBE_REPAIRED_PROFILE = "maniskill_pd_ee_delta_pose_7d_v2"
STACKCUBE_WINDOW_CONTRACT = "all_source_first_action_causal_reset_real_tail48_v2"
STACKCUBE_INSTRUCTION = "stack the red cube on top of the green cube"
EXPERT_COLLECTOR = "ManiSkillOfficialMotionPlanningReplay:pd_ee_delta_pose"


def validate_stackcube_descriptor(value: dict, *, profile: str = STACKCUBE_PROFILE) -> None:
    if profile not in {STACKCUBE_PROFILE, STACKCUBE_REPAIRED_PROFILE}:
        raise ValueError("unknown StackCube data contract")
    for key, expected in {
        "backend": "maniskill3",
        "benchmark": "ManiSkill3",
        "task": "StackCube-v1",
        "robot": "panda_wristcam",
        "kinematic_proxy": False,
        "action_state_semantics": "previous clipped native 7D action",
        "action_semantics": "ManiSkill normalized pd_ee_delta_pose: translation xyz, axis-angle rotation xyz, gripper target",
        "state_semantics": CONTINUOUS_STATE_SEMANTICS
        if profile == STACKCUBE_REPAIRED_PROFILE
        else LEGACY_STATE_SEMANTICS,
        "camera_semantics": {
            "top": "StackCube base_camera external view",
            "wrist": "panda_wristcam hand_camera",
        },
    }.items():
        if value.get(key) != expected:
            raise ValueError(f"StackCube environment {key} does not match the native contract")


def audit_stackcube_experts(
    root: str | Path,
    *,
    minimum_length: int = 73,
    profile: str = STACKCUBE_PROFILE,
    require_binary_gripper: bool = False,
) -> dict:
    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SIM_DATASET_SCHEMA:
        raise ValueError("not a simulator dataset manifest")
    descriptor = manifest["environment"]
    validate_stackcube_descriptor(descriptor, profile=profile)
    repaired = profile == STACKCUBE_REPAIRED_PROFILE
    files = sorted(root.glob("*.hdf5"))
    if not files:
        raise ValueError("no replay-verified StackCube expert HDF5 episodes")
    lengths, seeds = {}, set()
    for path in files:
        report = validate_recorded_episode(path)
        if (
            report["collector"] != EXPERT_COLLECTOR
            or not report["success"]
            or report["instruction"] != STACKCUBE_INSTRUCTION
        ):
            raise ValueError(f"{path.name}: not a verified StackCube expert")
        if report["steps"] < minimum_length:
            raise ValueError(
                f"{path.name}: {report['steps']} < typed-window minimum {minimum_length}"
            )
        with h5py.File(path, "r") as stream:
            saved = json.loads(stream.attrs["environment_descriptor_json"])
            if saved != descriptor:
                raise ValueError(f"{path.name}: environment identity differs")
            seed = int(stream.attrs["seed"])
            if seed in seeds:
                raise ValueError("duplicate expert reset seed would leak across episode splits")
            seeds.add(seed)
            action = np.asarray(stream["action"])
            if np.any(np.abs(action) > 1):
                raise ValueError(f"{path.name}: out-of-bounds expert command")
            if require_binary_gripper:
                if action.ndim != 2 or action.shape[1] <= 6:
                    raise ValueError(f"{path.name}: expert action has no native gripper column")
                if not np.isin(action[:, -1], (-1.0, 1.0)).all():
                    raise ValueError(
                        f"{path.name}: binary ManiSkill mode requires every expert gripper command to be exactly -1 or +1"
                    )
                action_state = np.asarray(stream["action_state"])
                if action_state.ndim != 2 or not np.isin(
                    action_state[:, -1], (-1.0, 1.0)
                ).all():
                    raise ValueError(
                        f"{path.name}: binary ManiSkill mode requires every action_state gripper command to be exactly -1 or +1"
                    )
            if not bool(stream["sim/success"][-1]):
                raise ValueError(f"{path.name}: final replay did not succeed")
            provenance = json.loads(stream.attrs.get("provenance_json", "{}"))
            source_steps = provenance.get("source_action_steps")
            held_steps = provenance.get("post_success_hold_steps")
            if (
                type(source_steps) is not int
                or source_steps < 24
                or type(held_steps) is not int
                or held_steps < (48 if repaired else 25)
                or source_steps + held_steps != report["steps"]
                or provenance.get("hold_semantics")
                != "real_env_steps_zero_ee_delta_hold_last_gripper"
            ):
                raise ValueError(
                    f"{path.name}: insufficient real post-success hold support for selected contract"
                )
            if repaired:
                if not np.allclose(
                    np.asarray(stream["action_state"][0]), [0, 0, 0, 0, 0, 0, 1], rtol=0, atol=1e-6
                ):
                    raise ValueError("ManiSkill v2 reset history must use zero EE / open gripper")
                if (
                    provenance.get("state_chart") != CONTINUOUS_STATE_CHART
                    or provenance.get("window_contract") != STACKCUBE_WINDOW_CONTRACT
                    or int(stream.attrs.get("valid_center_start", -1)) != 0
                    or int(stream.attrs.get("valid_center_end", -1)) != source_steps - 1
                ):
                    raise ValueError(
                        f"{path.name}: every real source action must own a first-executed-row center"
                    )
                if source_steps - 1 + 48 >= len(action):
                    raise ValueError("last source center lacks real future support")
            if not np.allclose(action[source_steps:, :6], 0, rtol=0, atol=1e-6):
                raise ValueError("post-success arm commands are not zero EE delta")
            if not np.allclose(
                action[source_steps:, -1], action[source_steps - 1, -1], rtol=0, atol=1e-6
            ):
                raise ValueError("post-success hold changed the gripper command")
        lengths[path.name] = report["steps"]
    return {
        "environment": descriptor,
        "episodes": len(files),
        "steps": sum(lengths.values()),
        "lengths": lengths,
        "reset_seeds": sorted(seeds),
        "minimum_length": minimum_length,
        "profile": profile,
        "binary_gripper_required": bool(require_binary_gripper),
        "expert_admission": "stackcube_all_source_first_action_v2"
        if repaired
        else "stackcube_real_success_hold_v1",
    }
