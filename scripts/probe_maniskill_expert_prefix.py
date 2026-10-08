"""Locate the failure stage with explicit expert-prefix / learned-policy handoff.

These are assisted diagnostic episodes, never autonomous benchmark scores.
All applied commands and both cameras are recorded for the full trajectory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import h5py
import numpy as np
import torch

from probe_maniskill_failure import load_policy
from record_maniskill_npz import record_episode, atomic_json
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv


class ExpertPrefix:
    def __init__(self, policy, actions, steps):
        self.policy, self.actions, self.steps = policy, actions, steps
        self.requires_executed_world_history = policy.requires_executed_world_history

    def reset(self):
        self.policy.reset()

    def begin_instruction(self, instruction):
        self.policy.begin_instruction(instruction)

    def act(self, history, instruction):
        if history.time_index < self.steps:
            return self.actions[history.time_index:history.time_index + 24].copy()
        return self.policy.act(history, instruction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "dino", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    policy = load_policy(args)
    env = ManiSkillStackCubeEnv()
    rows = []
    try:
        for name in ("expert_000005", "expert_000061"):
            with h5py.File(args.data / "experts" / (name + ".hdf5")) as h:
                actions = h["action"][:]
                seed = int(h.attrs["seed"])
                close = int(np.flatnonzero((actions[:, -1] < 0) & (h["action_state"][:, -1] > 0))[0])
            for steps in (16, ((close + 8 + 7) // 8) * 8):
                label = f"{name}_seed_{seed}_expert{steps}"
                wrapper = ExpertPrefix(policy, actions, steps)
                row = record_episode(env, wrapper, seed=seed, episode_dir=args.output / label, max_steps=400, replan_steps=8)
                row.update(assisted_diagnostic=True, expert_prefix_steps=steps, source_episode=name,
                           policy_rng="seed0 begins at handoff; expert prefix consumes no policy RNG", directory=label)
                rows.append(row)
                atomic_json(args.output / "handoff.json", {"not_an_autonomous_success_rate": True, "episodes": rows})
                print(json.dumps({k: row[k] for k in ("seed", "expert_prefix_steps", "success", "success_at_end", "directory")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
