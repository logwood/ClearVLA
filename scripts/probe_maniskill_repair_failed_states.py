"""Read-only grounding comparison on identical archived autonomous failure states.

Replay every one of the original 18 complete executed command streams. The
control and repair are queried at the same four pre-existing stage anchors.
Archived failed commands are never treated as expert action targets. Renderer
labels are evaluator-only and cannot enter policy inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from clearvla.simulation.admission import STACKCUBE_INSTRUCTION
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from analyze_maniskill_spatial_repair import grounding
from probe_maniskill_failure import load_policy
from probe_maniskill_spatial_grounding import SpatialCapture, simulator_labels
from record_maniskill_npz import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'dino', 'checkpoint', 'output', 'rollouts'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--arm', choices=['control', 'candidate'], required=True)
    p.add_argument('--paired-control', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    policy = load_policy(a)
    capture = SpatialCapture(policy)
    env = ManiSkillStackCubeEnv()
    rows, episodes = [], []
    try:
        with torch.no_grad():
            for seed in range(5000000, 5000018):
                folder = next(a.rollouts.glob(f'episode_*_seed_{seed}'))
                with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
                    actions = z['executed'].copy()
                    states = z['robot_obs_trajectory'].copy()
                    objects = z['object_positions'].copy()
                assert actions.shape == (400, 7) and states.shape == (401, 15)
                closes = np.flatnonzero((actions[:, -1] < 0) & (states[:-1, -1] > 0))
                close = int(closes[0]) if len(closes) else None
                closing_plan = (close // 8) * 8 if close is not None else 48
                selected = {16: 'approach', closing_plan: 'closing_plan' if close is not None else 'no_close_48',
                            160: 'middle_failure', 392: 'late_failure'}
                assert len(selected) == 4 and 0 not in selected
                current = env.reset(seed=seed)
                history = CausalHistory()
                history.reset(current.observation)
                policy.reset()
                state_error = object_error = 0.
                for t in range(401):
                    state_error = max(state_error, float(np.max(np.abs(current.observation.state - states[t, :7]))))
                    actual = np.array([current.evaluation.metrics['telemetry_cube_a_pose'][:3],
                                       current.evaluation.metrics['telemetry_cube_b_pose'][:3]])
                    object_error = max(object_error, float(np.max(np.abs(actual - objects[t, :2]))))
                    if state_error > 1e-6 or object_error > 1e-6:
                        raise RuntimeError(f'Replay mismatch: seed={seed}, step={t}, state={state_error}, objects={object_error}')
                    if t == 0:
                        policy.act(history.snapshot(), STACKCUBE_INSTRUCTION)
                    elif t in selected:
                        masks, poses = simulator_labels(env)
                        action, _, tensors = capture.infer_spatial(history.snapshot())
                        label = f'seed_{seed}_t{t:03d}_{selected[t]}'
                        saved = {k: tensors[k] for k in ('g_density', 'binding_probability', 'coarse_action', 'initial_noise')}
                        saved.update(predicted_action=action, masks=masks, object_positions=poses,
                                     robot_state=current.observation.state, rgb=np.stack(list(current.observation.rgb.values())))
                        assert np.isfinite(action).all()
                        if a.paired_control is not None:
                            with np.load(a.paired_control / (label + '.npz'), allow_pickle=False) as reference:
                                for key in ('masks', 'object_positions', 'robot_state', 'rgb', 'initial_noise'):
                                    np.testing.assert_array_equal(saved[key], reference[key], err_msg=key)
                        path = a.output / (label + '.npz')
                        np.savez_compressed(path, **saved)
                        measured = grounding(path)
                        row = dict(arm=a.arm, label=label, episode=str(seed), seed=seed, step=t, stage=selected[t],
                                   grounding=measured, first_close=close, source_episode=str(folder),
                                   first8_mean_xyz=action[:8, :3].mean(0).tolist(), first8_gripper=action[:8, -1].tolist(),
                                   paired_inputs_exact=a.paired_control is not None, artifact=path.name)
                        rows.append(row)
                        print(json.dumps({k: row[k] for k in ('arm', 'label', 'paired_inputs_exact')}), flush=True)
                    elif t % 8 == 0 and t < 400:
                        policy.bundle.model.outlet_adapter.sample_noise(
                            1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    if t < 400:
                        current = env.step(actions[t])
                        history.append(actions[t], current.observation)
                episodes.append(dict(seed=seed, steps=400, robot_state_max_error=state_error,
                                     object_position_max_error=object_error, source=str(folder)))
                atomic_json(a.output / 'summary.json', dict(complete=False, note=__doc__, rows=rows, episodes=episodes))
    finally:
        env.close()
    assert len(rows) == 72 and len(episodes) == 18
    atomic_json(a.output / 'summary.json', dict(complete=True, note=__doc__, rows=rows, episodes=episodes,
        limitations='Fixed states from the original failed policy, not new task scores or expert action labels. Replay checks the 7-D policy state and both cube XYZ; full joint-state equality is not asserted.'))
    atomic_json(a.output / 'provenance.json', policy.deployment_health())


if __name__ == '__main__':
    main()
