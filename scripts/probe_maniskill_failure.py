"""Reproducible ManiSkill failure probes using the unchanged deployment path.

Oracle facts stay in reports. Expert labels and image swaps are diagnostics,
never online inputs or claimed closed-loop benchmark improvements.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

from clearvla.simulation.admission import STACKCUBE_INSTRUCTION
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
import clearvla.simulation.clearvla_policy as policy_module
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from clearvla.mainline.runtime.sampling import sample_action
from record_maniskill_npz import record_episode, atomic_json


def native(result, policy):
    value = policy.bundle.action_normalizer.decode(result.action[0].float().cpu().numpy()).astype(np.float32)
    value[:, -1] = result.gripper_command[0].float().cpu().numpy()
    return value


def array(t):
    return t.detach().float().cpu().numpy()


class Capture:
    def __init__(self, policy):
        self.policy = policy
        self.sample = None
        self.tensors = {}

    def infer(self, history):
        model = self.policy.bundle.model
        encode, velocity = model.encode_online, model.velocity

        def observe_encode(*args, **kwargs):
            cache, state, metrics = encode(*args, **kwargs)
            f, s = state.top.facts, cache.top.intent
            self.tensors = {"g_" + k: array(getattr(f, k)) for k in (
                "content", "semantic", "geometry", "camera_coordinates", "camera_support",
                "object_to_chart", "validity", "existence")}
            self.tensors.update({"s_" + k: array(getattr(s, k)) for k in (
                "typed_common_value", "typed_policy_components", "public_interval_carrier")})
            self.tensors["binding_probability"] = array(s.target_binding.log_probability.exp())
            self.tensors["p1_detail"] = array(cache.factual_dock.protected_detail)
            return cache, state, metrics

        def observe_velocity(*args, **kwargs):
            result = velocity(*args, **kwargs)
            if bool((kwargs["time"] == 1).all()):
                self.tensors.update({
                    "p2_semantic": array(result.compiled.effect.semantic),
                    "p2_geometry": array(result.compiled.effect.geometry),
                    "p3_temporal": array(result.compiled.plan.temporal),
                    "p3_state_change": array(result.compiled.plan.state_change),
                })
            return result

        def observe_sample(*args, **kwargs):
            self.sample = sample_action(*args, **kwargs)
            return self.sample

        with patch.object(model, "encode_online", observe_encode), patch.object(model, "velocity", observe_velocity), patch.object(policy_module, "sample_action", observe_sample):
            action, online = self.policy.act_with_input(history, STACKCUBE_INSTRUCTION)
        self.tensors["action"] = action
        self.tensors["gripper_logits"] = array(self.sample.gripper_command_logits)
        return action, online, dict(self.tensors)


def h5_history(h, step):
    timeline = CausalHistory()
    def observation(t):
        return PolicyObservation(
            rgb={"top": h["observations/images/cam_high"][t], "wrist": h["observations/images/cam_right_wrist"][t]},
            state=h["state"][t], action_state=h["action_state"][t])
    timeline.reset(observation(0))
    for t in range(step):
        timeline.append(h["action"][t], observation(t + 1))
    return timeline.snapshot()


def contracts(args):
    env = ManiSkillStackCubeEnv()
    splits = json.loads((args.data / "prepared/splits.json").read_text())["splits"]
    rows = []
    try:
        for split, names in splits.items():
            for name in names:
                with h5py.File(args.data / "experts" / (name + ".hdf5")) as h:
                    seed = int(h.attrs["seed"])
                    reset = env.reset(seed=seed)
                    m = reset.evaluation.metrics
                    cmd = h["action"][:]
                    gr = cmd[:, -1]
                    c = np.flatnonzero((gr < 0) & (h["action_state"][:, -1] > 0))
                    row = {"episode": name, "split": split, "seed": seed,
                           "reset_state_max_error": float(np.max(np.abs(reset.observation.state - h["state"][0]))),
                           "reset_rgb_mae": {cam: float(np.mean(np.abs(reset.observation.rgb[cam].astype(float) - h[key][0].astype(float))))
                                             for cam, key in (("top", "observations/images/cam_high"), ("wrist", "observations/images/cam_right_wrist"))},
                           "red_xyz": m["telemetry_cube_a_pose"][:3], "green_xyz": m["telemetry_cube_b_pose"][:3],
                           "first_close": int(c[0]) if len(c) else None, "source_steps": int(h.attrs["valid_center_end"]) + 1}
                    if name in splits["train"][:3] + splits["val"][:3]:
                        states, red, parity = [], [], []
                        for t, command in enumerate(cmd):
                            if t < len(cmd):
                                parity.append(float(np.max(np.abs(reset.observation.state - h["state"][t]))))
                            states.append(reset.observation.state.copy())
                            red.append(reset.evaluation.metrics["telemetry_cube_a_pose"][:3])
                            reset = env.step(command)
                        row.update(replayed_steps=len(cmd), replay_success=bool(reset.evaluation.metrics["success"]),
                                   replay_state_error_max=max(parity),
                                   close_xy_error_m=float(np.linalg.norm(np.asarray(states)[c[0], :2] - np.asarray(red)[c[0], :2])) if len(c) else None)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    atomic_json(args.output / "contracts.json", {"episodes": rows})
    finally:
        env.close()


def load_policy(args):
    return ClearVLACheckpointPolicy(args.checkpoint, device=torch.device("cuda"),
        t5_condition=args.data / "language/t5_xxl.pt", dinov3_model=args.dino, seed=0)


def teacher(args):
    policy = load_policy(args)
    capture = Capture(policy)
    splits = json.loads((args.data / "prepared/splits.json").read_text())["splits"]
    rows = []
    for split in ("train", "val"):
        for name in splits[split][:args.count]:
            with h5py.File(args.data / "experts" / (name + ".hdf5")) as h:
                actions = h["action"][:]
                close = np.flatnonzero((actions[:, -1] < 0) & (h["action_state"][:, -1] > 0))
                opening = np.flatnonzero((actions[:, -1] > 0) & (h["action_state"][:, -1] < 0))
                c = int(close[0]); o = int(opening[opening > c][0])
                stages = {"reset": 0, "approach": 16, "preclose": max(0, c - 8), "close": c,
                          "lift": c + 8, "transport": (c + o)//2, "release": o}
                for stage, t in stages.items():
                    policy.reset()
                    raw, online, tensors = capture.infer(h5_history(h, t))
                    target = actions[t:t+24]
                    out = args.output / f"{split}_{name}_{stage}_{t}.npz"
                    np.savez_compressed(out, **tensors, target=target, initial_noise=array(capture.sample.initial_physical_noise))
                    content = tensors["g_content"][0]
                    unit = content / np.maximum(np.linalg.norm(content, axis=1, keepdims=True), 1e-12)
                    row = {"split": split, "episode": name, "seed": int(h.attrs["seed"]), "stage": stage, "step": t,
                           "arm_rmse_first8": float(np.sqrt(np.mean((raw[:8, :6] - target[:8, :6])**2))),
                           "translation_rmse_first8": float(np.sqrt(np.mean((raw[:8, :3] - target[:8, :3])**2))),
                           "grip_correct_first8": float(np.mean(raw[:8, -1] == target[:8, -1])),
                           "predicted_first": raw[0].tolist(), "target_first": target[0].tolist(),
                           "g_slot_pair_cosine": float((unit @ unit.T)[~np.eye(len(unit), dtype=bool)].mean()),
                           "binding_probability": tensors["binding_probability"].tolist(), "artifact": out.name}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    atomic_json(args.output / "teacher.json", {"note": "Expert-state diagnostic, no closed-loop success claim; fixed seed 0 per observation", "rows": rows})


def counterfactual(args):
    policy = load_policy(args)
    capture = Capture(policy)
    env = ManiSkillStackCubeEnv()
    rows = []
    try:
        # Physical red-cube placements with identical robot state and green cube.
        # Cube teleports are evaluator-only interventions. Current RGB is rerendered.
        import sapien
        reset = env.reset(seed=5000015)
        original = env._env.unwrapped.cubeA.pose.raw_pose[0].cpu().numpy().copy()
        for dx, dy in ((-.10, -.18), (.10, -.18), (-.10, .18), (.10, .18), (0., -.08), (0., .08)):
            pose = original.copy(); pose[:3] = [dx, dy, .02]
            env._env.unwrapped.cubeA.set_pose(sapien.Pose(p=pose[:3], q=pose[3:]))
            obs = env._observation(env._env.unwrapped.get_obs())
            np.testing.assert_allclose(obs.state, reset.observation.state, atol=1e-6, rtol=0)
            timeline = CausalHistory(); timeline.reset(obs)
            policy.reset()
            raw, online, tensors = capture.infer(timeline.snapshot())
            label = f"red_x{dx:+.2f}_y{dy:+.2f}"
            cv2.imwrite(str(args.output / (label + ".png")), cv2.cvtColor(np.concatenate(list(obs.rgb.values()), axis=1), cv2.COLOR_RGB2BGR))
            np.savez_compressed(args.output / (label + ".npz"), **tensors)
            row = {"red_xy": [dx, dy], "label": label, "first8_mean_xyz_command": raw[:8, :3].mean(0).tolist(),
                   "first24_sum_xyz_command": raw[:, :3].sum(0).tolist(), "first8_grip": raw[:8, -1].tolist(),
                   "binding_probability": tensors["binding_probability"].tolist()}
            rows.append(row); print(json.dumps(row), flush=True)
            atomic_json(args.output / "counterfactual.json", {"seed": 5000015, "fixed_policy_noise": 0, "rows": rows})
    finally:
        env.close()


def closedloop(args):
    policy = load_policy(args)
    env = ManiSkillStackCubeEnv()
    splits = json.loads((args.data / "prepared/splits.json").read_text())["splits"]
    rows = []
    try:
        for split in ("train", "val"):
            for name in splits[split][:args.count]:
                with h5py.File(args.data / "experts" / (name + ".hdf5")) as h:
                    seed = int(h.attrs["seed"])
                row = record_episode(env, policy, seed=seed, episode_dir=args.output / f"{split}_{name}_seed_{seed}", max_steps=400, replan_steps=8)
                row.update(split=split, source_episode=name)
                rows.append(row); print(json.dumps(row), flush=True)
                atomic_json(args.output / "closedloop.json", {"episodes": rows})
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("contracts", "teacher", "counterfactual", "closedloop"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dino", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with torch.no_grad():
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
