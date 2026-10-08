"""Endpoint representation controls on recorded contact/withdrawal observations.

Each intervention holds the complete decoded arm plan and final W cache fixed.
No revised command is executed. Observed labels/body identities never enter it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime import sampling
from clearvla.mainline.endpoint_supervision import endpoint_condition
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from probe_endpoint_arm_representation import arm_reencode


def array(t):
    return t.detach().float().cpu().numpy()


def rms(x):
    return float(np.sqrt(np.square(x).mean()))


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--memory-cap-gib', type=float, default=12)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('set deterministic CUBLAS environment before CUDA')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device('cuda:0'),
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'), seed=0)
    model, config = policy.bundle.model, policy.bundle.config
    versions = {n: p._version for n, p in model.named_parameters()}
    captured = {}
    original = sampling.sample_refined_cached_action_with_cache

    def sampled(*a, **kw):
        result, cache = original(*a, **kw)
        captured.update(result=result, cache=cache, kwargs=kw)
        return result, cache

    sampling.sample_refined_cached_action_with_cache = sampled
    report = dict(complete=False, records=[], optimizer_updates=0,
        identity=dict(checkpoint=str(args.checkpoint), checkpoint_sha256=policy.bundle.checkpoint_sha256,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest()), scope=__doc__)
    try:
        for row in json.loads(args.plan.read_text()):
            path = Path(row['case']) / 'trajectory.npz'
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in ('rgb_static', 'rgb_gripper', 'robot_obs', 'executed', 'raw_chunks')}
            wanted = set(row['steps'])
            if not wanted or min(wanted) < 0 or max(wanted) >= len(data['executed']) or any(t % 8 for t in wanted):
                raise ValueError('invalid actual replan window')
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(wanted) + 1):
                previous = np.zeros(7, np.float32) if t == 0 else data['executed'][t-1]
                obs = calvin_policy_observation({'rgb_obs': {k: data[k][t] for k in ('rgb_static', 'rgb_gripper')}, 'robot_obs': data['robot_obs'][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                if t not in wanted:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row['instruction'])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                action, _ = policy.act_with_input(history.snapshot(), row['instruction'])
                sample, cache, kw = captured['result'], captured['cache'], captured['kwargs']
                field = sample.physical_field.float()
                outlet, state, boundary = model.outlet_adapter, cache.history.action_state, cache.history.codec_gripper_boundary
                native_arm = sample.action[..., :6].float()
                canonical = arm_reencode(outlet, field, state, boundary)
                item = dict(case_id=row['case_id'], state=t, case=row['case'], instruction=row['instruction'],
                    recorded_arm_rms=rms(action[:8, :6] - data['raw_chunks'][t//8, :8, :6]),
                    recorded_gripper_changed_rows=int((action[:8, 6] != data['raw_chunks'][t//8, :8, 6]).sum()),
                    baseline_first8=action[:8].tolist(), per_row_residual_rms=array((field[..., :12]-canonical[..., :12]).square().mean(-1).sqrt())[0].tolist(), variants=[])
                for mode, value in [('repeat', field), ('halfway', field + .5*(canonical-field)), ('canonical', canonical)]:
                    arm = outlet.decode(value, state, codec_gripper_boundary=boundary)[..., :6]
                    arm_error = float((arm-native_arm).abs().max())
                    if arm_error > 2e-5 or not torch.equal(value[..., 12:], field[..., 12:]):
                        raise AssertionError('arm command or unused gripper field changed')
                    time, context = endpoint_condition(value, context_enabled=config.runtime.deployment_flow_schedule is not None)
                    with torch.no_grad(), torch.autocast('cuda', dtype=resolve_compute_dtype(config)):
                        out = sampling._velocity_at(model, cache, noisy_action_field=value, time=time,
                            execution_mode=kw.get('execution_mode', 'learned'), deployment_fastpath=kw.get('deployment_fastpath', False),
                            collect_diagnostics=False, flow_step_context=context, accepts_flow_step_context=sampling._accepts_flow_step_context(model))
                    logits = out.bottom.gripper_command_logits.float()
                    diff = float((logits-sample.gripper_command_logits.float()).abs().max())
                    if mode == 'repeat' and diff != 0:
                        raise AssertionError(('same-input head repeat failed', diff))
                    command = logits.argmax(-1)
                    changed = command != sample.gripper_command_logits.argmax(-1)
                    item['variants'].append(dict(mode=mode, decoded_arm_max_error=arm_error, logit_rms=float((logits-sample.gripper_command_logits.float()).square().mean().sqrt()),
                        first8_changed=int(changed[:, :8].sum()), all24_changed=int(changed.sum()),
                        first8_command=(array(command)[0, :8]*2-1).tolist(),
                        first8_open_probability=array(logits.softmax(-1))[0, :8, 1].tolist()))
                report['records'].append(item)
                (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print('ENDPOINT_FACTUAL', row['case_id'], t, item['variants'][-1]['first8_changed'], flush=True)
        if versions != {n: p._version for n, p in model.named_parameters()}:
            raise AssertionError('parameter mutation')
        report.update(complete=True, parameters_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    finally:
        sampling.sample_refined_cached_action_with_cache = original


if __name__ == '__main__':
    main()
