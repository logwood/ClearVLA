"""Replay converted ManiSkill expert actions into ClearVLA wrist-camera HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .admission import STACKCUBE_WINDOW_CONTRACT
from .contracts import ACTION_DIM, EvaluationState, SimulationEnvironment, StepResult
from .dataset import EpisodeRecorder, audit_simulation_dataset
from .state_chart import CONTINUOUS_STATE_CHART, CONTINUOUS_STATE_SEMANTICS, LEGACY_STATE_CHART


@dataclass(frozen=True)
class ConvertedEpisode:
    episode_id: int
    seed: int
    steps: int

    @property
    def group_name(self) -> str:
        return f"traj_{self.episode_id}"


def recording_step(result: StepResult, *, continue_success: bool) -> StepResult:
    """Fixed-horizon expert collection may continue past SUCCESS, not failure.

    ManiSkill's vendor replay likewise continues stepping to the end of the
    expert command sequence. Physics remains live after its success terminal.
    Keep the original task terminal in evaluator metadata; suppress it only
    for this collector's finite source+hold recording. Online RL is unchanged.
    """
    metrics = dict(result.evaluation.metrics)
    if (continue_success and result.terminated and not result.truncated
            and bool(metrics.get("success", False)) and not bool(metrics.get("fail", False))):
        metrics.update(native_terminated=True, collection_success_continuation=True)
        return replace(result, terminated=False, evaluation=EvaluationState(metrics))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_converted_index(
    trajectory_path: str | Path,
    metadata_path: str | Path,
) -> tuple[ConvertedEpisode, ...]:
    """Validate the controller/episode index without trusting filename labels."""

    trajectory = Path(trajectory_path)
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("converted ManiSkill metadata must be a mapping")
    env_info = metadata.get("env_info")
    if not isinstance(env_info, Mapping) or env_info.get("env_id") != "StackCube-v1":
        raise ValueError("converted demonstration must own StackCube-v1 env_info")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise ValueError("converted demonstration has no env_kwargs mapping")
    if env_kwargs.get("control_mode") != "pd_ee_delta_pose":
        raise ValueError("converted demonstration is not pd_ee_delta_pose")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("converted demonstration metadata has no episodes")

    refs: list[ConvertedEpisode] = []
    seen: set[int] = set()
    with h5py.File(trajectory, "r") as stream:
        for row in episodes:
            if not isinstance(row, Mapping):
                raise ValueError("converted episode metadata row must be a mapping")
            try:
                episode_id = int(row["episode_id"])
                seed = int(row["episode_seed"])
                elapsed_steps = int(row["elapsed_steps"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("converted episode identity is malformed") from error
            if episode_id < 0 or episode_id in seen or seed < 0 or elapsed_steps <= 0:
                raise ValueError("converted episode id/seed/length is invalid or duplicated")
            if row.get("control_mode") != "pd_ee_delta_pose":
                raise ValueError(f"converted episode {episode_id} changed control mode")
            if row.get("success") is not True:
                raise ValueError(f"converted episode {episode_id} is not source-successful")
            group_name = f"traj_{episode_id}"
            group = stream.get(group_name)
            if not isinstance(group, h5py.Group):
                raise ValueError(f"converted trajectory has no group {group_name}")
            actions = group.get("actions")
            if not isinstance(actions, h5py.Dataset):
                raise ValueError(f"converted trajectory {group_name} has no actions")
            if actions.shape != (elapsed_steps, ACTION_DIM) or actions.dtype != np.float32:
                raise ValueError(
                    f"converted {group_name} actions must be float32 "
                    f"[{elapsed_steps},{ACTION_DIM}], got {actions.shape} {actions.dtype}"
                )
            seen.add(episode_id)
            refs.append(ConvertedEpisode(episode_id, seed, elapsed_steps))
    return tuple(refs)


def import_converted_episodes(
    environment: SimulationEnvironment,
    *,
    trajectory_path: Path,
    metadata_path: Path,
    output_dir: Path,
    start_index: int,
    count: int,
    min_steps: int = 58,
    require_all_success: bool = False,
    settle_steps: int = 0,
) -> dict[str, Any]:
    """Replay source-successful actions and retain only wristcam-successful episodes."""

    if start_index < 0 or count <= 0 or min_steps <= 0:
        raise ValueError("start_index must be non-negative; count/min_steps must be positive")
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative")
    repaired = environment.descriptor.state_semantics == CONTINUOUS_STATE_SEMANTICS
    if repaired and settle_steps < 48:
        raise ValueError("v2 expert collection requires at least 48 real tail steps")
    refs = load_converted_index(trajectory_path, metadata_path)
    selected = refs[start_index : start_index + count]
    if len(selected) != count:
        raise ValueError(
            f"requested converted rows [{start_index}:{start_index + count}], "
            f"but only {len(refs)} exist"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    instruction = "stack the red cube on top of the green cube"
    with h5py.File(trajectory_path, "r") as stream:
        for ref in selected:
            if ref.steps + settle_steps > environment.descriptor.max_episode_steps:
                raise ValueError("episode time limit cannot cover source plus real post-success hold")
            if ref.steps < min_steps:
                rejected.append(
                    {
                        "episode_id": ref.episode_id,
                        "seed": ref.seed,
                        "reason": f"too_short_{ref.steps}_lt_{min_steps}",
                    }
                )
                continue
            destination_id = f"expert_{ref.episode_id:06d}"
            if (output_dir / f"{destination_id}.hdf5").exists():
                raise FileExistsError(f"refusing to overwrite imported episode {destination_id}")
            group = stream.get(ref.group_name)
            if not isinstance(group, h5py.Group):
                raise ValueError(f"trajectory group {ref.group_name!r} is missing or invalid")
            actions_dataset = group.get("actions")
            if not isinstance(actions_dataset, h5py.Dataset):
                raise ValueError(f"trajectory group {ref.group_name!r} has no actions dataset")
            actions = np.asarray(actions_dataset, dtype=np.float32)
            reset = environment.reset(seed=ref.seed)
            observation = reset.observation
            recorder = EpisodeRecorder(
                output_dir,
                descriptor=environment.descriptor,
                episode_id=destination_id,
                seed=ref.seed,
                instruction=instruction,
                collector="ManiSkillOfficialMotionPlanningReplay:pd_ee_delta_pose",
                provenance={"source_action_steps": ref.steps,
                            "post_success_hold_steps": settle_steps,
                            "state_chart": CONTINUOUS_STATE_CHART if repaired else LEGACY_STATE_CHART,
                            "window_contract": STACKCUBE_WINDOW_CONTRACT if repaired else "legacy_complete_history_v1",
                            "success_terminal_handling": (
                                "continue_fixed_source_and_hold_record_native_terminal" if settle_steps > 0
                                else "native_termination"),
                            "hold_semantics": "real_env_steps_zero_ee_delta_hold_last_gripper"},
                valid_center_bounds=(0, ref.steps - 1) if repaired else None,
            )
            reward_sum = 0.0
            executed = 0
            last_metrics = dict(reset.evaluation.metrics)
            terminal_early = False
            for action in actions:
                command = environment.clip_action(action)
                if not np.allclose(command, action, rtol=0.0, atol=1e-6):
                    raise ValueError(
                        f"converted {ref.group_name} contains an out-of-bounds expert action"
                    )
                result = recording_step(environment.step(command), continue_success=settle_steps > 0)
                recorder.append(observation, command, result)
                observation = result.observation
                reward_sum += float(result.reward)
                executed += 1
                last_metrics = dict(result.evaluation.metrics)
                if result.terminated or result.truncated:
                    terminal_early = executed != ref.steps
                    break
            final_success = bool(last_metrics.get("success", False))
            if terminal_early or executed != ref.steps or not final_success:
                rejected.append(
                    {
                        "episode_id": ref.episode_id,
                        "seed": ref.seed,
                        "source_steps": ref.steps,
                        "executed_steps": executed,
                        "final_success": final_success,
                        "terminal_early": terminal_early,
                        "final_evaluator_metrics": last_metrics,
                        "reason": "wristcam_replay_did_not_reproduce_full_success",
                    }
                )
                continue
            held = 0
            # Real simulator transitions, NOT synthetic absorbing RGB/state.
            # They provide 48-frame Teacher support for the last expert actions.
            if settle_steps and (result.terminated or result.truncated):
                rejected.append({"episode_id": ref.episode_id, "reason": "terminal_before_hold",
                                 "terminated": bool(result.terminated),
                                 "truncated": bool(result.truncated),
                                 "final_evaluator_metrics": last_metrics})
                continue
            for _ in range(settle_steps):
                command = environment.clip_action(environment.hold_action(observation))
                result = recording_step(environment.step(command), continue_success=True)
                recorder.append(observation, command, result)
                observation = result.observation
                held += 1
                reward_sum += float(result.reward)
                last_metrics = dict(result.evaluation.metrics)
                if result.terminated or result.truncated:
                    break
            if held != settle_steps or not bool(last_metrics.get("success", False)):
                rejected.append({"episode_id": ref.episode_id, "reason": "post_success_hold_failed",
                                 "held_steps": held, "required_steps": settle_steps,
                                 "final_evaluator_metrics": last_metrics})
                continue
            artifact = recorder.finish()
            accepted.append(
                {
                    "episode_id": ref.episode_id,
                    "seed": ref.seed,
                    "steps": artifact.steps,
                    "source_action_steps": ref.steps,
                    "post_success_hold_steps": held,
                    "reward_sum": reward_sum,
                    "final_evaluator_metrics": last_metrics,
                    "path": str(artifact.path.resolve()),
                }
            )
    # Retain failed admission evidence too. Previously both early raises lost
    # every rejection reason and made a transport/hold failure unobservable.
    audit = audit_simulation_dataset(output_dir) if accepted else None
    payload = {
        "schema": "clearvla-maniskill-expert-import-v1",
        "source": {
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_sha256": _sha256(trajectory_path),
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": _sha256(metadata_path),
            "controller": "pd_ee_delta_pose",
        },
        "selection": {
            "start_index": start_index,
            "count": count,
            "min_steps": min_steps,
            "settle_steps": settle_steps,
        },
        "accepted": accepted,
        "rejected": rejected,
        "dataset_audit": audit,
    }
    # Report to the caller/stdout only; do not retain another diagnostic JSON.
    # Immutable dataset manifest and HDF5 provenance remain required inputs.
    if require_all_success and rejected:
        raise RuntimeError(f"{len(rejected)} converted demonstrations were rejected: {json.dumps(rejected)}")
    if not accepted:
        raise RuntimeError(f"no converted demonstration reproduced success with panda_wristcam: {json.dumps(rejected)}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=58)
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="real zero-EE/hold-gripper steps after successful source replay")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=400,
        help="StackCube protocol time limit; use a smaller value only for smoke",
    )
    parser.add_argument("--state-chart", choices=(LEGACY_STATE_CHART, CONTINUOUS_STATE_CHART),
                        default=LEGACY_STATE_CHART)
    parser.add_argument(
        "--maniskill-sim-backend",
        default="physx_cpu",
        choices=("physx_cpu", "physx_cuda"),
    )
    parser.add_argument(
        "--maniskill-render-backend",
        default="gpu",
        choices=("gpu", "cpu"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = args.metadata or args.trajectory.with_suffix(".json")
    from .maniskill_adapter import ManiSkillStackCubeEnv

    environment = ManiSkillStackCubeEnv(
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.maniskill_sim_backend,
        render_backend=args.maniskill_render_backend,
        state_chart=args.state_chart,
    )
    try:
        result = import_converted_episodes(
            environment,
            trajectory_path=args.trajectory,
            metadata_path=metadata,
            output_dir=args.output_dir,
            start_index=args.start_index,
            count=args.count,
            min_steps=args.min_steps,
            require_all_success=args.require_all_success,
            settle_steps=args.settle_steps,
        )
    finally:
        environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
