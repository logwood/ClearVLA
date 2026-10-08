"""Paired action-history ablation on recorded expert observations.

Diagnostic only: remove the same executed-command condition trained with
dropout, while retaining all RGB/state history, instruction and sampling noise.
This is not a deployment modification or a closed-loop success claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from probe_maniskill_failure import Capture, array, h5_history, load_policy
from record_maniskill_npz import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'dino', 'teacher', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    policy = load_policy(args)
    capture = Capture(policy)
    stage = policy.bundle.model.conditioning
    prepare = stage.prepare

    def remove_actions(*a, **kw):
        conditioned, proposal, goal_keep, history_keep = prepare(*a, **kw)
        zero = torch.zeros_like(history_keep)
        h = conditioned.history
        h = replace(h, executed_action_history=torch.zeros_like(h.executed_action_history),
                    timing=None if h.timing is None else h.timing.without_actions(zero),
                    executed_world_window=None if h.executed_world_window is None else h.executed_world_window.without_actions(zero),
                    executed_robot_step=None if h.executed_robot_step is None else h.executed_robot_step.without_actions(zero))
        return replace(conditioned, history=h), proposal, goal_keep, zero

    rows = []
    cases = json.loads((args.teacher / 'teacher.json').read_text())['rows']
    with torch.no_grad():
        for case in cases:
            with h5py.File(args.data / 'experts' / (case['episode'] + '.hdf5')) as h:
                history = h5_history(h, case['step'])
            with np.load(args.teacher / case['artifact'], allow_pickle=False) as z:
                target, baseline, expected_noise = z['target'], z['action'], z['initial_noise']
            # Verify the saved comparison uses the identical input/noise path.
            if case['stage'] == 'reset':
                policy.reset()
                replay, _, _ = capture.infer(history)
                np.testing.assert_allclose(replay, baseline, rtol=0, atol=1e-6)
            policy.reset()
            with patch.object(stage, 'prepare', remove_actions):
                raw, _, tensors = capture.infer(history)
            np.testing.assert_array_equal(array(capture.sample.initial_physical_noise), expected_noise)
            if case['stage'] == 'reset':
                np.testing.assert_allclose(raw, baseline, rtol=0, atol=1e-6)
            name = case['artifact']
            np.savez_compressed(args.output / name, **tensors, target=target,
                                baseline_action=baseline, initial_noise=expected_noise)
            row = {k: case[k] for k in ('split', 'episode', 'seed', 'stage', 'step')}
            row.update(baseline_arm_rmse_first8=case['arm_rmse_first8'],
                       ablated_arm_rmse_first8=float(np.sqrt(np.mean((raw[:8, :6]-target[:8, :6])**2))),
                       action_change_rms_first8=float(np.sqrt(np.mean((raw[:8, :6]-baseline[:8, :6])**2))),
                       baseline_grip_correct_first8=case['grip_correct_first8'],
                       ablated_grip_correct_first8=float(np.mean(raw[:8, -1]==target[:8, -1])), artifact=name)
            rows.append(row)
            atomic_json(args.output / 'history.json', {'intervention': 'executed-action condition removed; visual/state history unchanged; exact matched initial noise', 'rows': rows})
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
