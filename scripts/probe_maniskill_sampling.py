"""Matched-observation/noise controls; no solver or deployed policy is changed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

from clearvla.simulation.admission import STACKCUBE_INSTRUCTION
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory
import clearvla.simulation.clearvla_policy as policy_module
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from probe_maniskill_failure import load_policy, h5_history, native
from record_maniskill_npz import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "dino", "output", "probe_root", "panel"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    policy = load_policy(args)
    state = np.load(args.panel / "episode_00015_seed_5000015/trajectory.npz")["robot_obs_trajectory"][0, :7]
    cases = []
    cf = json.loads((args.probe_root / "counterfactual-r1/counterfactual.json").read_text())
    for row in cf["rows"]:
        rgb = cv2.cvtColor(cv2.imread(str(args.probe_root / "counterfactual-r1" / (row["label"] + ".png"))), cv2.COLOR_BGR2RGB)
        observation = PolicyObservation(rgb={"top": rgb[:, :336].copy(), "wrist": rgb[:, 336:].copy()}, state=state.copy(), action_state=np.array([0, 0, 0, 0, 0, 0, 1], np.float32))
        history = CausalHistory(); history.reset(observation)
        cases.append((row["label"], history.snapshot(), None))
    splits = json.loads((args.data / "prepared/splits.json").read_text())["splits"]
    for split in ("train", "val"):
        for name in splits[split][:2]:
            with h5py.File(args.data / "experts" / (name + ".hdf5")) as h:
                actions = h["action"][:]
                c = int(np.flatnonzero((actions[:, -1] < 0) & (h["action_state"][:, -1] > 0))[0])
                for t in (16, max(c - 8, 0)):
                    cases.append((f"{split}_{name}_t{t}", h5_history(h, t), actions[t:t+24]))
    rows = []
    with torch.no_grad():
        for label, history, target in cases:
            captured = {}
            def observe(*a, **kw):
                captured["sample"] = sample_action(*a, **kw)
                return captured["sample"]
            policy.reset()
            with patch.object(policy_module, "sample_action", observe):
                primary, online = policy.act_with_input(history, STACKCUBE_INSTRUCTION)
            noise = captured["sample"].initial_physical_noise.clone()
            actions = {"seed0_q5": primary}
            for variant in ("seed1_q5", "seed2_q5", "seed3_q5", "seed42_q5", "zero_q5", "antithetic_q5", "seed0_e5"):
                if variant.startswith("seed") and variant != "seed0_e5":
                    seed = int(variant.split("_")[0][4:])
                    n = torch.randn(noise.shape, device=noise.device, dtype=noise.dtype, generator=torch.Generator(device=noise.device).manual_seed(seed))
                else:
                    n = torch.zeros_like(noise) if variant == "zero_q5" else -noise if variant == "antithetic_q5" else noise
                result = sample_action(policy.bundle.model, online, policy.bundle.config,
                    initial_physical_noise=n,
                    flow_schedule=DeploymentFlowSchedule.uniform_five() if variant == "seed0_e5" else None)
                actions[variant] = native(result, policy)
            ensemble = np.mean([actions["seed0_q5"], actions["seed1_q5"], actions["seed2_q5"], actions["seed3_q5"]], axis=0)
            ensemble[:, -1] = np.where(ensemble[:, -1] >= 0, 1, -1)
            actions["mean4_q5"] = ensemble
            anti = (primary + actions["antithetic_q5"]) / 2
            anti[:, -1] = np.where(anti[:, -1] >= 0, 1, -1)
            actions["antithetic_mean_q5"] = anti
            np.savez_compressed(args.output / (label + ".npz"), **actions, **({"target": target} if target is not None else {}))
            record = {"case": label, "variants": {}}
            for name, value in actions.items():
                item = {"first8_mean_xyz": value[:8, :3].mean(0).tolist(), "first24_sum_xyz": value[:, :3].sum(0).tolist(),
                        "arm_delta_from_seed0_rms": float(np.sqrt(np.mean((value[:, :6] - primary[:, :6])**2)))}
                if target is not None:
                    item.update(arm_rmse_first8=float(np.sqrt(np.mean((value[:8, :6] - target[:8, :6])**2))),
                                gripper_accuracy_first8=float(np.mean(value[:8, -1] == target[:8, -1])))
                record["variants"][name] = item
            rows.append(record)
            print(json.dumps(record), flush=True)
            atomic_json(args.output / "sampling.json", {"rows": rows})


if __name__ == "__main__":
    main()
