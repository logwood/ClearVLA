"""Record complete native ManiSkill episodes, NPZ traces, telemetry and videos.

The 15 trajectory keys match the user's reference archive. Privileged object
facts are evaluator output only; the policy receives RGB and causal robot history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from clearvla.simulation.admission import STACKCUBE_INSTRUCTION, validate_stackcube_descriptor
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv

REFERENCE_KEYS = (
    "raw_chunks", "raw_first", "raw_executed", "executed",
    "executed_plan_index", "executed_chunk_row", "observed_history_time_index",
    "observed_after_executed_steps", "latency_seconds", "robot_obs_before_action",
    "robot_obs_trajectory", "object_positions", "object_robot_contact",
    "target_surface_contact_preserved", "oracle_success",
)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def required_vector(metrics: Mapping[str, Any], key: str, width: int) -> np.ndarray:
    if key not in metrics:
        raise ValueError(f"Required evaluator telemetry is missing: {key}")
    value = np.asarray(metrics[key], dtype=np.float32).reshape(-1)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid evaluator telemetry: {key}")
    return value.copy()


def robot_row(environment: Any, observation: Any) -> np.ndarray:
    qpos = environment._env.unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, "detach"):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.size < 9 or not np.isfinite(qpos).all():
        raise ValueError("Panda joint state must contain seven arm and two finger joints")
    # The last column is a command, not the physical finger opening.
    return np.concatenate((observation.state[:7], qpos[:7],
                           observation.action_state[-1:])).astype(np.float32)


def scene_row(metrics: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    a = required_vector(metrics, "telemetry_cube_a_pose", 7)[:3]
    b = required_vector(metrics, "telemetry_cube_b_pose", 7)[:3]
    positions = np.stack((a, b, b + np.array([0, 0, .04], np.float32))).astype(np.float64)
    contacts = []
    for label in ("cube_a", "cube_b"):
        forces = [required_vector(metrics, f"telemetry_{label}_finger{i}_contact_force", 3)
                  for i in (1, 2)]
        contacts.append(any(np.linalg.norm(f) > 1e-8 for f in forces))
    if "is_cubeA_on_cubeB" not in metrics or "success" not in metrics:
        raise ValueError("StackCube oracle fields are missing")
    return positions, np.asarray([*contacts, bool(metrics["is_cubeA_on_cubeB"])], dtype=np.bool_)


class EpisodeVideo:
    def __init__(self, path: Path, fps: float):
        import imageio.v2 as imageio
        self.path = path
        self.temporary = path.with_name("." + path.stem + ".partial.mp4")
        self.writer = imageio.get_writer(
            str(self.temporary), format="FFMPEG", mode="I", fps=fps,
            codec="libx264", macro_block_size=1, quality=7)
        self.frames = 0

    def append(self, observation: Any, *, seed: int, step: int, success: bool) -> None:
        import cv2
        frame = np.concatenate([observation.rgb["top"], observation.rgb["wrist"]], axis=1)
        bar = np.full((32, frame.shape[1], 3), 20, dtype=np.uint8)
        text = f"top | wrist   seed {seed}   step {step:03d}   success={int(success)}"
        cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        self.writer.append_data(np.concatenate((frame, bar), axis=0))
        self.frames += 1

    def close(self, *, complete: bool) -> None:
        self.writer.close()
        if complete:
            os.replace(self.temporary, self.path)


def record_episode(environment: Any, policy: Any, *, seed: int, episode_dir: Path,
                   max_steps: int, replan_steps: int, record_video: bool = True) -> dict[str, Any]:
    reset = environment.reset(seed=seed)
    reset.observation.validate()
    policy.reset()
    policy.begin_instruction(STACKCUBE_INSTRUCTION)
    history = CausalHistory(executed_world=policy.requires_executed_world_history)
    history.reset(reset.observation)
    episode_dir.mkdir(parents=True, exist_ok=False)
    video = EpisodeVideo(episode_dir / "video.mp4", environment.descriptor.control_hz) if record_video else None
    rows: dict[str, list] = {k: [] for k in REFERENCE_KEYS}
    reset_scene, _ = scene_row(reset.evaluation.metrics)
    initial_b_z = reset_scene[1, 2]
    telemetry = (episode_dir / "telemetry.jsonl").open("w", encoding="utf-8")
    plan_times = []
    started = time.perf_counter()
    reward = 0.0
    terminated = truncated = False
    chunk = None
    chunk_row = 0
    complete = False

    def observe(observation, metrics, step):
        scene, contacts = scene_row(metrics)
        rows["robot_obs_trajectory"].append(robot_row(environment, observation))
        rows["object_positions"].append(scene)
        rows["object_robot_contact"].append(contacts)
        rows["target_surface_contact_preserved"].append(abs(scene[1, 2] - initial_b_z) <= .01)
        rows["oracle_success"].append(bool(metrics["success"]))
        telemetry.write(json.dumps({"observation_index": step, "metrics": dict(metrics)},
                                   allow_nan=False) + "\n")
        if video:
            video.append(observation, seed=seed, step=step, success=bool(metrics["success"]))

    try:
        observe(reset.observation, reset.evaluation.metrics, 0)
        for step in range(max_steps):
            if chunk is None:
                before = time.perf_counter()
                chunk = np.asarray(policy.act(history.snapshot(), STACKCUBE_INSTRUCTION), dtype=np.float32)
                rows["latency_seconds"].append(time.perf_counter() - before)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < replan_steps or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid policy chunk: {chunk.shape}, replan={replan_steps}")
                rows["raw_chunks"].append(chunk.copy())
                plan_times.append(step)
                chunk_row = 0
            raw_action = chunk[chunk_row].copy()
            submitted = environment.clip_action(raw_action.copy())
            rows["robot_obs_before_action"].append(rows["robot_obs_trajectory"][-1].copy())
            result = environment.step(submitted.copy())
            result.validate()
            # The simulator's receipt is the source of truth for history.
            applied = result.command_receipt.applied.copy()
            np.testing.assert_allclose(result.observation.action_state, applied, rtol=0, atol=1e-6)
            np.testing.assert_allclose(applied, submitted, rtol=0, atol=1e-6)
            rows["raw_executed"].append(raw_action)
            rows["executed"].append(applied)
            rows["executed_plan_index"].append(len(rows["raw_chunks"]) - 1)
            rows["executed_chunk_row"].append(chunk_row)
            history.append(applied, result.observation)
            observe(result.observation, result.evaluation.metrics, step + 1)
            reward += float(result.reward)
            terminated = bool(result.terminated)
            truncated = bool(result.truncated)
            # Success alone never shortens recording. Native termination and
            # the declared time limit still own the episode boundary.
            if terminated or truncated:
                break
            chunk_row += 1
            if chunk_row == replan_steps:
                chunk = None
            else:
                rows["observed_history_time_index"].append(step + 1)
                rows["observed_after_executed_steps"].append(step + 1)
        complete = True
    finally:
        telemetry.close()
        if video:
            video.close(complete=complete)

    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays["raw_chunks"] = np.stack(rows["raw_chunks"]).astype(np.float32)
    arrays["raw_first"] = arrays["raw_chunks"][:, 0].copy()
    for key in ("executed_plan_index", "executed_chunk_row", "observed_history_time_index",
                "observed_after_executed_steps"):
        arrays[key] = arrays[key].astype(np.int64)
    for key in ("object_robot_contact", "target_surface_contact_preserved", "oracle_success"):
        arrays[key] = arrays[key].astype(np.bool_)
    arrays["latency_seconds"] = arrays["latency_seconds"].astype(np.float64)
    n = len(arrays["executed"])
    assert len(arrays["robot_obs_trajectory"]) == n + 1
    assert len(arrays["object_positions"]) == n + 1
    np.testing.assert_array_equal(
        arrays["raw_executed"],
        arrays["raw_chunks"][arrays["executed_plan_index"], arrays["executed_chunk_row"]])
    np.testing.assert_allclose(arrays["robot_obs_trajectory"][1:, -1], arrays["executed"][:, -1], rtol=0, atol=0)
    assert tuple(arrays) == REFERENCE_KEYS
    temporary = episode_dir / ".trajectory.npz.tmp"
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, episode_dir / "trajectory.npz")
    success_steps = np.flatnonzero(arrays["oracle_success"])
    report = {
        "schema": "clearvla-maniskill-trajectory-npz-v2", "seed": seed,
        "success": bool(success_steps.size), "success_at_end": bool(arrays["oracle_success"][-1]),
        "first_success_step": int(success_steps[0]) if success_steps.size else None,
        "steps": n, "terminated": terminated, "truncated": truncated or n == max_steps,
        "reward_sum": reward, "replan_steps": replan_steps,
        "policy_decisions": len(rows["raw_chunks"]), "plan_time_index": plan_times,
        "elapsed_seconds": time.perf_counter() - started,
        "trajectory": "trajectory.npz", "telemetry": "telemetry.jsonl",
        "video": "video.mp4" if video else None, "video_frames": video.frames if video else 0,
        "shapes": {k: list(a.shape) for k, a in arrays.items()},
        "scene_projection": {
            "object_positions": "[red cube A, green cube B, desired red-cube center 4cm above B]",
            "object_robot_contact": "[A finger contact, B finger contact, A on B oracle]",
            "target_surface_contact_preserved": "B center z within 1cm of reset; diagnostic only",
            "robot_obs": "[tcp_xyz, causal_rotvec, opening, panda_arm_qpos7, previous_gripper_command]",
            "observation_rows": "0=reset; row t follows t executed actions",
            "observed_history_time_index": "intermediate observations between policy replans; all observations are recorded",
        },
    }
    atomic_json(episode_dir / "result.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=18)
    p.add_argument("--eval-seed", type=int, default=5_000_000)
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--t5-condition", type=Path)
    p.add_argument("--dinov3-model", type=Path)
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    if min(a.episodes, a.max_episode_steps, a.replan_steps) < 1 or min(a.eval_seed, a.policy_seed) < 0:
        p.error("Episode counts and horizons must be positive; seeds nonnegative")
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(a.checkpoint, device=torch.device(a.device),
        t5_condition=a.t5_condition, dinov3_model=a.dinov3_model, seed=a.policy_seed)
    cfg = policy.bundle.config
    if cfg.data.data_profile != "maniskill_pd_ee_delta_pose_7d_v2":
        raise ValueError("This recorder requires the ManiSkill v2 native state/action contract")
    if a.replan_steps > cfg.dimensions.action_horizon:
        raise ValueError("Replan prefix exceeds the policy action horizon")
    environment = ManiSkillStackCubeEnv(image_size=336, max_episode_steps=a.max_episode_steps,
        sim_backend="physx_cpu", render_backend="gpu", state_chart="fixed_down_causal_rotvec_v2")
    validate_stackcube_descriptor(environment.descriptor.to_dict(), profile=cfg.data.data_profile)
    health = policy.deployment_health()
    atomic_json(a.output_dir / "policy_health.json", health)
    manifest = {
        "schema": "clearvla-maniskill-trajectory-npz-panel-v2", "complete": False,
        "checkpoint": str(a.checkpoint.resolve()), "checkpoint_identity": health["checkpoint"],
        "evaluator_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment.descriptor.to_dict(), "instruction": STACKCUBE_INSTRUCTION,
        "eval_seed": a.eval_seed, "requested_episodes": a.episodes, "policy_seed": a.policy_seed,
        "replan_steps": a.replan_steps, "max_episode_steps": a.max_episode_steps,
        "full_trajectory": True, "execution_policy": cfg.bottom.execution_eval_policy,
        "success_definition": "any native oracle success; final success also reported separately",
        "episodes": [],
    }
    atomic_json(a.output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for index in range(a.episodes):
            seed = a.eval_seed + index
            episode_dir = a.output_dir / f"episode_{index:05d}_seed_{seed}"
            report = record_episode(environment, policy, seed=seed, episode_dir=episode_dir,
                max_steps=a.max_episode_steps, replan_steps=a.replan_steps, record_video=not a.no_video)
            report["episode_index"] = index
            report["directory"] = episode_dir.name
            manifest["episodes"].append(report)
            manifest["success_count"] = sum(int(row["success"]) for row in manifest["episodes"])
            manifest["success_rate"] = manifest["success_count"] / len(manifest["episodes"])
            manifest["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(a.output_dir / "manifest.json", manifest)
            print(json.dumps({k: report[k] for k in ("episode_index", "seed", "success",
                "success_at_end", "steps", "policy_decisions", "elapsed_seconds")}), flush=True)
        manifest["complete"] = True
        atomic_json(a.output_dir / "manifest.json", manifest)
        print(json.dumps({"complete": True, "episodes": a.episodes,
                          "success_rate": manifest["success_rate"]}), flush=True)
    except Exception as error:
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(a.output_dir / "manifest.json", manifest)
        raise
    finally:
        environment.close()


if __name__ == "__main__":
    main()

