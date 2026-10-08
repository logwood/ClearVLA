"""Read-only command-head sensitivity to redundant arm representation.

Decode/re-encode uses the actual outlet. Generated, halfway and re-encoded
fields have the same decoded arm plan, within reported FP32 roundoff. Only
the endpoint head is called again; its output never feeds back into sampling.
Expert agreement is descriptive, not proof of correct gripper for generated arm.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import sys

import torch
from clearvla.mainline import train
from clearvla.mainline.training import engine as engine_module
from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast
from clearvla.mainline.endpoint_supervision import clean_endpoint_field, endpoint_condition
from probe_endpoint_condition_distribution import counts


class ProbeComplete(Exception):
    pass


def arm_reencode(outlet, field, state, boundary):
    action = outlet.decode(field, state, codec_gripper_boundary=boundary)
    encoded = outlet.encode(action, state, codec_gripper_boundary=boundary)
    n = 2 * outlet.arm_dim
    return torch.cat((encoded[..., :n], field[..., n:]), -1)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--probe-batches', type=int, default=16)
    parser.add_argument('--memory-cap-gib', type=float)
    args, remaining = parser.parse_known_args()
    if '--validate-checkpoint' not in remaining or args.report.exists() or args.probe_batches < 1:
        raise ValueError('requires fresh report, positive batches and read-only validation')
    torch.use_deterministic_algorithms(True)
    if args.memory_cap_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.memory_cap_gib * 1024**3 / total
        if not 0 < fraction < 1:
            raise ValueError('invalid audit-only allocator cap')
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    original_eval = MainlineTrainingEngine.eval_step
    original_flow = engine_module.sample_flow_matching
    original_sample = train.sample_refined_cached_action_with_cache
    original_validate = train._validate
    pending = {}
    report = dict(complete=False, records=[], optimizer_updates=0,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Fixed generated native arm; redundant-field sensitivity at endpoint only. No policy, checkpoint or official panel change. Expert agreement is not behavior qualification.')

    def flow(*a, **kw):
        value = original_flow(*a, **kw)
        pending['flow'] = value
        return value

    def eval_step(engine, batch, **kw):
        pending.update(engine=engine, batch=batch)
        return original_eval(engine, batch, **kw)

    def validate(*a, **kw):
        pending['collect_validation'] = True
        try:
            return original_validate(*a, **kw)
        finally:
            pending['collect_validation'] = False

    @torch.no_grad()
    def sample(model, cache, config, **kw):
        result, refined = original_sample(model, cache, config, **kw)
        if not pending.get('collect_validation', False):
            return result, refined
        engine, batch = pending['engine'], pending['batch']
        if model is not engine.model or not torch.equal(cache.history.state, batch.online.history.state):
            raise ValueError('validation sample lost matched observed batch')
        outlet = model.outlet_adapter
        if config.data.data_profile != 'calvin_relative_7d_v1' or not outlet.is_binary_command:
            raise ValueError('requires admitted CALVIN binary outlet')
        if result.physical_field.shape[0] != 8:
            raise ValueError('this matched audit declares BS8')
        versions = {n: p._version for n, p in model.named_parameters()}
        state = cache.history.action_state
        boundary_field = cache.history.codec_gripper_boundary
        raw = result.physical_field.float()
        canonical = arm_reencode(outlet, raw, state, boundary_field)
        halfway = raw + 0.5 * (canonical - raw)
        labelled = clean_endpoint_field(replace(pending['flow'], source_physical_noise=result.initial_physical_noise), arm_dim=outlet.arm_dim).float()
        valid = batch.action_target.row_valid
        if valid is None:
            valid = torch.ones_like(batch.action_target.raw_units[..., -1], dtype=torch.bool)
        truth = batch.action_target.raw_units[..., -1] > 0
        boundary = batch.action_target.gripper_transition_boundary_raw_units[..., -1] > 0
        event = truth != torch.cat((boundary[:, None], truth[:, :-1]), 1)
        labelled_roundtrip = arm_reencode(outlet, labelled, state, boundary_field)
        # Unknown-tail noise remains exactly untouched in this numerical control.
        labelled_roundtrip = torch.where(valid.all(-1)[:, None, None], labelled_roundtrip, labelled)
        fields = dict(raw=raw, halfway=halfway, canonical=canonical, labelled=labelled)
        decoded = {n: outlet.decode(f, state, codec_gripper_boundary=boundary_field)[..., :outlet.arm_dim].float() for n, f in fields.items()}
        checks = {n: float((decoded[n] - decoded['raw']).abs().max()) for n in ('halfway', 'canonical')}
        checks['sampler_arm_max'] = float((decoded['raw'] - result.action[..., :outlet.arm_dim]).abs().max())
        checks['canonical_idempotence_max'] = float((arm_reencode(outlet, canonical, state, boundary_field) - canonical).abs().max())
        checks['complete_label_roundtrip_max'] = float((labelled_roundtrip - labelled).abs().max())
        if max(checks.values()) > 2e-5:
            raise AssertionError(('command-preservation/numerical control failed', checks))
        if not torch.equal(canonical[..., 12:], raw[..., 12:]):
            raise AssertionError('legacy grip noise was changed')
        row_residual = (raw[..., :12] - canonical[..., :12]).square().mean(-1).sqrt()
        item = dict(index=len(report['records']), sample_ids=batch.audit.sample_index.cpu().tolist(),
                    episode_ids=batch.audit.episode_index.cpu().tolist(), global_step=engine.global_step,
                    preservation=checks, conditions={}, pairs={},
                    field_residual_rms=float(row_residual.square().mean().sqrt()),
                    field_residual_first8_rms=float(row_residual[:, :8].square().mean().sqrt()),
                    per_row_field_residual_rms=row_residual.cpu().tolist(),
                    decoded_generated_vs_label_arm_rms=float((decoded['raw'] - decoded['labelled']).square().mean().sqrt()))
        outputs = {}

        def head(name, value_cache, field):
            time, context = endpoint_condition(field, context_enabled=config.runtime.deployment_flow_schedule is not None)
            with _autocast(engine.device, engine.dtype):
                prediction = model.velocity(value_cache, noisy_action_field=field, time=time,
                                            flow_step_context=context, require_execution_supervision=False,
                                            collect_diagnostics=False)
            logits = prediction.bottom.gripper_command_logits.float()
            if not bool(torch.isfinite(logits).all()):
                raise AssertionError('nonfinite endpoint logits')
            outputs[name] = logits
            item['conditions'][name] = counts(logits, truth, valid, event, boundary)
            item['conditions'][name]['first8_open_probability'] = logits.softmax(-1)[:, :8, 1].cpu().tolist()

        for cache_name, value_cache in [('coarse', cache), ('refined', refined)]:
            for field_name, field in fields.items():
                head(cache_name + '_' + field_name, value_cache, field)
        head('refined_raw_repeat', refined, raw)
        head('refined_label_roundtrip', refined, labelled_roundtrip)
        repeat = float((outputs['refined_raw'] - outputs['refined_raw_repeat']).abs().max())
        sampler_repeat = float((outputs['refined_raw'] - result.gripper_command_logits.float()).abs().max())
        item['repeat_max'] = repeat
        item['original_sampler_repeat_max'] = sampler_repeat
        if repeat != 0 or sampler_repeat != 0:
            raise AssertionError(('uncontrolled same-input repeat', repeat, sampler_repeat))
        pair_names = [('refined_raw', 'refined_canonical'), ('refined_raw', 'refined_halfway'),
                      ('coarse_raw', 'coarse_canonical'), ('coarse_canonical', 'refined_canonical'),
                      ('refined_labelled', 'refined_label_roundtrip'), ('refined_labelled', 'refined_raw'),
                      ('refined_labelled', 'refined_canonical')]
        for a, b in pair_names:
            pa, pb = outputs[a].argmax(-1), outputs[b].argmax(-1)
            delta = outputs[a] - outputs[b]
            stats = dict(logit_rms=float(delta.square().mean().sqrt()))
            for suffix, mask in [('all', valid), ('first8', valid & (torch.arange(24, device=valid.device)[None] < 8)),
                                 ('complete', valid & valid.all(-1, keepdim=True))]:
                stats[suffix] = dict(rows=int(mask.sum()), flips=int(((pa != pb) & mask).sum()),
                                     corrected=int(((pa != truth) & (pb == truth) & mask).sum()),
                                     regressed=int(((pa == truth) & (pb != truth) & mask).sum()))
            item['pairs'][a + '__' + b] = stats
        if any(p._version != versions[n] for n, p in model.named_parameters()):
            raise AssertionError('model parameters mutated')
        report['records'].append(item)
        report['parameters_unchanged'] = True
        report['complete'] = len(report['records']) == args.probe_batches
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(engine.device)
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(engine.device)
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(args.report)
        print('ENDPOINT_ARM_REPRESENTATION', item['index'], item['pairs']['refined_raw__refined_canonical'], checks, flush=True)
        if report['complete']:
            raise ProbeComplete()
        return result, refined

    MainlineTrainingEngine.eval_step = eval_step
    engine_module.sample_flow_matching = flow
    train.sample_refined_cached_action_with_cache = sample
    train._validate = validate
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        print('COMPLETE', args.report, flush=True)
    finally:
        MainlineTrainingEngine.eval_step = original_eval
        engine_module.sample_flow_matching = original_flow
        train.sample_refined_cached_action_with_cache = original_sample
        train._validate = original_validate


if __name__ == '__main__':
    main()
