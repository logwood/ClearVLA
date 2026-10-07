"""Separate controller tracking error from the trained robot innovation.

Restores the full checkpoint strictly, then executes its real observer on
recorded one-step inputs. No vision, policy sampling, training or environment
configuration is changed. Constructed histories test interface observability,
not the existence of two pixel-identical physical rollouts.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.data.state_features import encode_state_features
from clearvla.mainline.robot_execution import ExecutedRobotStep
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.benchmarks.bridge import policy_observation
from clearvla.simulation.history import CausalHistory


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def equal_tree(a, b):
    if dataclasses.is_dataclass(a):
        return all(equal_tree(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def quantiles(value):
    return dict(zip(["min", "median", "p90", "max"], np.quantile(value, [0, .5, .9, 1]).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--training-episodes", type=Path)
    ap.add_argument("--raw-root", type=Path)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    bundle = load_deployment_checkpoint(args.checkpoint, device=device)
    observer = bundle.model.policy_compiler.plan_compiler.robot_observer
    assert observer is not None
    use_bf16 = device.type == "cuda" and bundle.config.runtime.compute_dtype == "bf16"

    raw_label_checks = []
    if args.training_episodes is not None:
        assert args.raw_root is not None
        episodes = json.loads(args.training_episodes.read_text())
        for episode in [x for x in episodes if x["split"] == "train"][::43]:
            for t in [episode["source_start"], episode["source_start"] + 16, episode["source_end"] - 8]:
                with np.load(args.raw_root / "training" / f"episode_{t:07d}.npz", allow_pickle=False) as z:
                    absolute, relative, current = z["actions"], z["rel_actions"], z["robot_obs"]
                with np.load(args.raw_root / "training" / f"episode_{t+1:07d}.npz", allow_pickle=False) as z:
                    following = z["robot_obs"]
                angular_error = (absolute[3:6] - following[3:6] + np.pi) % (2*np.pi) - np.pi
                raw_label_checks.append({"episode": episode["episode"], "frame": t,
                    "actions_vs_next_observed_tcp_xyz_max_m": float(np.max(np.abs(absolute[:3] - following[:3]))),
                    "actions_vs_next_observed_euler_max_rad": float(np.max(np.abs(angular_error))),
                    "relative_vs_clipped_next_observed_motion_max": float(np.max(np.abs(
                        relative[:3] - np.clip(following[:3] - current[:3], -.02, .02)/.02)))})
        print("raw training label origin", len(raw_label_checks), "checked", flush=True)

    def state_tensor(value):
        encoded = encode_state_features(np.asarray(value, dtype=np.float32), bundle.state_normalizer,
            mode=bundle.config.top.state_feature_mode, profile=bundle.config.data.data_profile)
        return torch.from_numpy(np.ascontiguousarray(encoded)).to(device=device, dtype=torch.float32)

    # Same finite causal interface, different discarded old commands.
    base = policy_observation(top=np.zeros((8, 8, 3), np.uint8),
        wrist=np.zeros((8, 8, 3), np.uint8), state=np.zeros(7, np.float32),
        action_state=np.zeros(7, np.float32))
    histories = [CausalHistory(executed_world=True), CausalHistory(executed_world=True)]
    totals = []
    for branch, history in enumerate(histories):
        history.reset(base)
        total = np.zeros(3, np.float64)
        for t in range(32):
            command = np.zeros(7, np.float32)
            if branch == 1 and t == 1:
                command[0] = .5
            total += command[:3] * np.float32(.02)
            history.append(command, dataclasses.replace(base, action_state=command.copy()))
        totals.append(total)
    alias = {"snapshot_equal": equal_tree(histories[0].snapshot(), histories[1].snapshot()),
        "controller_target_difference_m": (totals[1] - totals[0]).tolist(),
        "time_index": 32, "scope": "constructed interface counterexample; not a physical paired rollout"}
    assert alias["snapshot_equal"] and np.isclose(totals[1][0] - totals[0][0], .01)

    # A correctly predicted stall yields zero innovation even with a command.
    controlled = copy.deepcopy(observer)
    with torch.no_grad():
        for parameter in controlled.response.parameters():
            parameter.zero_()
        previous = torch.zeros((1, observer.state_dim), device=device)
        command = torch.zeros((1, observer.action_dim), device=device)
        command[:, 0] = 1
        step = ExecutedRobotStep(previous, command, torch.ones(1, dtype=torch.bool, device=device),
            torch.tensor([[-1, -1, 0]], dtype=torch.long, device=device))
        query = torch.ones((1, 1, 1, observer.plan_query.in_features), device=device)
        negative, loss = controlled.observe(step, previous)
        negative_read = controlled.read(negative, query)
        positive_state = previous.clone()
        positive_state[:, 0] = .1
        positive, _ = controlled.observe(step, positive_state)
        positive_read = controlled.read(positive, query)
    mechanism = {"zero_prediction_stall_loss": float(loss),
        "zero_prediction_stall_innovation_max": float(negative.innovation.abs().max()),
        "zero_prediction_stall_read_max": float(negative_read.abs().max()),
        "positive_control_innovation_max": float(positive.innovation.abs().max()),
        "positive_control_read_rms": float(positive_read.float().square().mean().sqrt()),
        "scope": "response zeroed only in a diagnostic copy; source checkpoint parameters untouched"}
    assert mechanism["zero_prediction_stall_loss"] == 0
    assert mechanism["zero_prediction_stall_read_max"] == 0
    assert mechanism["positive_control_read_rms"] > 0

    special = {2: [96, 104, 110], 9: [56, 64, 72], 10: [64, 72, 128, 136, 144],
        14: [336, 344, 352, 354], 17: [64, 72, 128, 136, 144]}
    rows, cases, projection_checks = [], [], []
    scale = np.asarray(bundle.state_normalizer.scale).reshape(-1)[:3]
    for folder in sorted(args.rollout.glob("case_*")):
        if not folder.is_dir() or not (folder / "trajectory.npz").is_file():
            continue
        case_id = int(folder.name.split("_")[1])
        result = json.loads((folder / "result.json").read_text())
        info = json.loads((folder / "environment_info.json").read_text())
        with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
            robot = z["robot_obs"]
            actions = z["executed"]
            # Confirm ignored privileged fields do not enter the public adapter.
            obs = {"rgb_obs": {k: z[k][0] for k in ["rgb_static", "rgb_gripper"]},
                "robot_obs": robot[0], "scene_obs": z["scene_obs"][0]}
        altered = dict(obs, scene_obs=np.full_like(obs["scene_obs"], 10000),
            controller_targets={"target_pos": [10000] * 3}, contacts=[{"body": 999}], success=True)
        altered["robot_obs"] = robot[0].copy()
        altered["robot_obs"][7:] = 10000
        projection_checks.append(equal_tree(calvin_policy_observation(obs, np.zeros(7, np.float32)),
            calvin_policy_observation(altered, np.zeros(7, np.float32))))
        states = state_tensor(robot[:, :7])
        commands = torch.from_numpy(bundle.action_normalizer.encode(actions.astype(np.float32))).to(device)
        observed = torch.ones(1, dtype=torch.bool, device=device)
        offsets = torch.tensor([[-1, -1, 0]], dtype=torch.long, device=device)
        innovations = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            for t in range(1, len(robot)):
                step = ExecutedRobotStep(states[t-1:t], commands[t-1:t], observed, offsets)
                feedback, _ = observer.observe(step, states[t:t+1])
                innovations.append(feedback.innovation)
        innovation = torch.cat(innovations).float().cpu().numpy()
        predicted_delta_xyz = np.diff(states.float().cpu().numpy()[:, :3], axis=0) / scale - innovation[:, :3] / scale
        desired = np.array([r["target_pos"] for r in info["controller_targets"]])
        desired_orn = np.array([r["target_orn"] for r in info["controller_targets"]])
        assert desired.shape == robot[:, :3].shape
        error = desired - robot[:, :3]
        angle_error = (desired_orn - robot[:, 3:6] + np.pi) % (2*np.pi) - np.pi
        increments = actions[:, :3] * np.float32(.02)
        recurrence = float(np.max(np.abs(desired[1:] - desired[:-1] - increments)))
        error_recurrence = float(np.max(np.abs(np.diff(error, axis=0) - increments + np.diff(robot[:, :3], axis=0))))
        case_rows = []
        for t in range(1, len(robot)):
            row = {"case_id": case_id, "state": t, "success": bool(result["success"]),
                "controller_error_xyz_m": error[t].tolist(), "controller_error_norm_m": float(np.linalg.norm(error[t])),
                "controller_angle_error_norm_rad": float(np.linalg.norm(angle_error[t])),
                "observed_tcp_delta_m": (robot[t, :3] - robot[t-1, :3]).tolist(),
                "predicted_tcp_delta_m": predicted_delta_xyz[t-1].tolist(),
                "innovation_xyz_m": (innovation[t-1, :3] / scale).tolist(),
                "innovation_xyz_norm_m": float(np.linalg.norm(innovation[t-1, :3] / scale)),
                "innovation_feature_rms": float(np.sqrt(np.mean(innovation[t-1]**2))),
                "requested_xyz_increment_m": increments[t-1].tolist(),
                "current_is_replan": t % 8 == 0 and t < len(actions)}
            case_rows.append(row)
        rows.extend(case_rows)
        cases.append({"case_id": case_id, "success": bool(result["success"]), "transitions": len(case_rows),
            "controller_recurrence_max_error_m": recurrence, "tracking_error_recurrence_max_error_m": error_recurrence,
            "tracking_error_norm_m": quantiles([x["controller_error_norm_m"] for x in case_rows]),
            "innovation_xyz_norm_m": quantiles([x["innovation_xyz_norm_m"] for x in case_rows]),
            "critical_windows": [x for x in case_rows if x["state"] in special.get(case_id, [])]})
        print(case_id, len(case_rows), "complete", flush=True)
    assert len(cases) == 18 and len(rows) == 3890 and all(projection_checks)
    assert max(x["controller_recurrence_max_error_m"] for x in cases) < 1e-12
    summary = {"checkpoint_sha256": bundle.checkpoint_sha256, "source_commit": bundle.identity.git_commit,
        "epoch": bundle.epoch, "global_step": bundle.global_step,
        "compute_dtype": "bf16 autocast batch1" if use_bf16 else "fp32",
        "transitions": len(rows), "cases": cases, "history_alias": alias,
        "observer_definition_control": mechanism, "privileged_projection_unchanged_cases": sum(projection_checks),
        "raw_training_label_origin": {"checks": len(raw_label_checks),
            "max_errors": {k: max(x[k] for x in raw_label_checks) for k in [
                "actions_vs_next_observed_tcp_xyz_max_m", "actions_vs_next_observed_euler_max_rad",
                "relative_vs_clipped_next_observed_motion_max"]} if raw_label_checks else {},
            "scope": "next achieved pose labels must not be called the demonstrator controller's held setpoint"},
        "scope": "Trained one-step observer evaluated directly on its exact recorded inputs; no full-policy action or P3 query contribution is inferred from this probe. Controller error is audit-only and is not a current policy input.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(args.output / "transitions.json", rows)
    dump(args.output / "raw_training_label_checks.json", raw_label_checks)
    dump(args.output / "summary.json", summary)
    dump(args.output / "complete.json", {"status": "complete", "checkpoint_sha256": bundle.checkpoint_sha256,
        "transitions": len(rows), "cases": len(cases), "script_sha256": summary["script_sha256"]})


if __name__ == "__main__":
    main()
