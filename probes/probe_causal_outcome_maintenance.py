"""Fixed factual contact/withdrawal reads through observed and predicted outcomes.

Zero controls remove one declared consumer value on the same RGB/history/noise.
They are interface sensitivity tests, not physical counterfactual rollouts or
independent contribution percentages. Ground-truth object state is audit-only.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.model.annotation_goal import AnnotatedGoalValueRead
from causal_identity_node_trace import NodeTrace, compare_nodes


def array(x):
    return x.detach().float().cpu().numpy().copy()


def rms(x):
    return float(np.sqrt(np.square(np.asarray(x, dtype=np.float64)).mean()))


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('deterministic audit requires CUBLAS_WORKSPACE_CONFIG before CUDA initialization')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device('cuda:0'),
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'), seed=0)
    model = policy.bundle.model
    outcome = model.intent.organizer.observed_outcome
    if outcome is None:
        raise ValueError('checkpoint does not own the before-proposal observed outcome consumer')
    import clearvla.simulation.clearvla_policy as frontend
    trace = NodeTrace(model)
    original = dict(encode=model.encode_online, sample=frontend.sample_action, outcome=outcome.forward,
                    image=outcome.image.forward, semantic=outcome.semantic.forward)
    capture, control, readers = {}, {'name': 'baseline'}, []
    versions = {name: value._version for name, value in model.named_parameters()}

    def encode(*a, **kw):
        result = original['encode'](*a, **kw)
        capture['cache'], capture['state'] = result[:2]
        return result

    def sampled(*a, **kw):
        result = original['sample'](*a, **kw)
        capture['sample'] = result
        return result

    def observed(feedback, facts, binding):
        capture['observed_input'] = dict(
            semantic_rms=rms(array(feedback.observed_semantic)),
            image=array(feedback.observed_image)[0].tolist(),
            view_observed=array(feedback.view_observed)[0].tolist(),
            null=array(feedback.null)[0].tolist())
        mode = control['name']
        result = original['outcome'](feedback, facts, binding)
        if mode == 'observed_outcome_zero':
            result = torch.zeros_like(result)
        trace.record('s_observed_outcome', result)
        capture['observed_output_rms'] = rms(array(result))
        return result

    def image_value(*a, **kw):
        result = original['image'](*a, **kw)
        return torch.zeros_like(result) if control['name'] == 'observed_image_value_zero' else result

    def semantic_value(*a, **kw):
        result = original['semantic'](*a, **kw)
        return torch.zeros_like(result) if control['name'] == 'observed_semantic_value_zero' else result

    for name, module in model.named_modules():
        if not isinstance(module, AnnotatedGoalValueRead):
            continue
        old = module.prepare
        readers.append((module, old))

        def prepare(evidence, old=old, name=name):
            value = old(evidence)
            capture.setdefault('goal_components', {})[name] = {
                field: rms(array(getattr(value, field))) for field in ('content', 'image', 'joint', 'robot')}
            fields = {'goal_visual_zero': ('content', 'image', 'joint'),
                      'goal_robot_zero': ('robot',),
                      'goal_all_zero': ('content', 'image', 'joint', 'robot')}.get(control['name'], ())
            return replace(value, **{field: torch.zeros_like(getattr(value, field)) for field in fields}) if fields else value

        module.prepare = prepare
    if len(readers) != 2:
        raise ValueError('expected the existing S and P3 annotated-goal value readers')
    model.encode_online, frontend.sample_action, outcome.forward = encode, sampled, observed
    # Remove only prepared consumer values; retain the validated factual
    # observed/predicted/innovation contract and its source/support laws.
    outcome.image.forward, outcome.semantic.forward = image_value, semantic_value
    report = dict(identity=dict(checkpoint=str(args.checkpoint), checkpoint_sha256=policy.bundle.checkpoint_sha256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), deterministic=True,
        config=policy.bundle.config.top.__dict__, scope=__doc__), records=[], complete=False)

    def summarize(result):
        intent = capture['cache'].top.intent
        native = policy.bundle.action_normalizer.decode(array(result.action)[0])
        native[:, -1] = array(result.gripper_command)[0]
        coarse = policy.bundle.action_normalizer.decode(array(capture['state'].top.coarse_action.action_prediction)[0])
        goal = intent.annotated_goal
        prediction = goal.prediction
        # Model state has rotation6d (10 channels); the native normalizer owns
        # 7 channels. Only the unchanged first-three TCP channels are decoded.
        offset = np.asarray(policy.bundle.state_normalizer.offset).reshape(-1)[:3]
        scale = np.asarray(policy.bundle.state_normalizer.scale).reshape(-1)[:3]
        endpoint = (array(prediction.reference.state + prediction.robot_delta)[0, :3] - offset) / scale
        current = (array(goal.instruction_change.current_state)[0, :3] - offset) / scale
        return dict(first8_native=native[:8].tolist(), first8_mean=native[:8].mean(0).tolist(),
            coarse_first8=coarse[:8].tolist(), binding=array(intent.target_binding.mass)[0].tolist(),
            binding_null=array(intent.target_binding.null_mass)[0].tolist(),
            current_tcp_xyz=current.tolist(), predicted_endpoint_tcp_xyz=endpoint.tolist(), predicted_remaining_tcp_xyz=(endpoint-current).tolist(),
            observed_input=capture['observed_input'], observed_output_rms=capture['observed_output_rms'],
            goal_component_source_rms=capture['goal_components'])

    try:
        for row in json.loads(args.plan.read_text()):
            wanted = sorted(set(row['steps']))
            with np.load(Path(row['case'])/'trajectory.npz', allow_pickle=False) as z:
                data = {key: z[key] for key in ('rgb_static', 'rgb_gripper', 'robot_obs', 'executed', 'raw_chunks')}
            if not wanted or min(wanted) < 0 or max(wanted) >= len(data['executed']) or any(t % 8 for t in wanted):
                raise ValueError('probe states must be valid recorded replan states')
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(wanted)+1):
                previous = np.zeros(7, np.float32) if t == 0 else data['executed'][t-1]
                obs = calvin_policy_observation({'rgb_obs': {key: data[key][t] for key in ('rgb_static', 'rgb_gripper')},
                                                'robot_obs': data['robot_obs'][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                control['name'] = 'baseline'
                if t not in wanted:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row['instruction'])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                capture.clear()
                trace.reset()
                action, online = policy.act_with_input(history.snapshot(), row['instruction'])
                sample = capture['sample']
                baseline = summarize(sample)
                np.testing.assert_allclose(baseline['current_tcp_xyz'], data['robot_obs'][t, :3], atol=2e-6, rtol=0.)
                reference_nodes = trace.finish(capture['cache'].top.intent)
                entry = dict(case_id=row['case_id'], state=t, case=row['case'], baseline=baseline,
                    recorded_arm_rms=rms(action[:8, :6]-data['raw_chunks'][t//8, :8, :6]),
                    recorded_gripper_changed_rows=int(np.sum(action[:8, 6] != data['raw_chunks'][t//8, :8, 6])),
                    variants=[])
                for mode in ('repeat', 'observed_outcome_zero', 'observed_image_value_zero', 'observed_semantic_value_zero',
                             'goal_robot_zero', 'goal_visual_zero', 'goal_all_zero'):
                    control['name'] = mode
                    capture.clear()
                    trace.reset()
                    with torch.no_grad():
                        result = sample_action(model, online, policy.bundle.config, initial_physical_noise=sample.initial_physical_noise)
                    value = summarize(result)
                    nodes = trace.finish(capture['cache'].top.intent)
                    value.update(mode=mode, nodes=compare_nodes(reference_nodes, nodes),
                        first8_arm_delta_rms=rms(np.asarray(value['first8_native'])[:, :6]-np.asarray(baseline['first8_native'])[:, :6]),
                        coarse_arm_delta_rms=rms(np.asarray(value['coarse_first8'])[:, :6]-np.asarray(baseline['coarse_first8'])[:, :6]),
                        gripper_changed_rows=int(np.sum(np.asarray(value['first8_native'])[:, 6] != np.asarray(baseline['first8_native'])[:, 6])))
                    np.testing.assert_array_equal(value['binding'], baseline['binding'])
                    np.testing.assert_array_equal(value['binding_null'], baseline['binding_null'])
                    if mode == 'repeat':
                        np.testing.assert_array_equal(value['first8_native'], baseline['first8_native'])
                    entry['variants'].append(value)
                report['records'].append(entry)
                dump(args.output/'results.json', report)
                print('OUTCOME_MAINTENANCE', row['case_id'], t, flush=True)
        if versions != {name: value._version for name, value in model.named_parameters()}:
            raise RuntimeError('read-only probe modified a model parameter')
        report.update(complete=True, parameters_unchanged=True)
        dump(args.output/'results.json', report)
    finally:
        trace.restore()
        model.encode_online, frontend.sample_action, outcome.forward = original['encode'], original['sample'], original['outcome']
        outcome.image.forward, outcome.semantic.forward = original['image'], original['semantic']
        for module, old in readers:
            module.prepare = old


if __name__ == '__main__':
    main()
