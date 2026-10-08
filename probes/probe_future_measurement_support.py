"""Read-only real masked BS8 audit of source-mask reuse on future targets.

Keep the actual source facts, matcher, predictions and loss reductions fixed.
The counterfactual target plane admits the independently supplied future frame,
without copying a current-frame augmentation mask into its coordinates. It is
returned only to the audit; the ordinary forward receives its original target.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys

import torch
from clearvla.mainline import train
from clearvla.mainline.model import source_measurement as measurement_module
from clearvla.mainline.model.teacher import ObjectFutureTeacher
from clearvla.mainline.training import losses as loss_module
from clearvla.mainline.training.engine import MainlineTrainingEngine, TrainStepResult, _autocast
from clearvla.vision.observed_correspondence import observed_feature_correspondence


class ProbeComplete(Exception):
    pass


def scalar(value):
    return float(value.detach().float())


def independent_target_function():
    original = measurement_module.source_consistent_measurement
    source = inspect.getsource(original)
    before = 'target_mask=source_mask[:,None].expand(b,f,c,h,w) & observed[:,:,None,None,None]'
    after = 'target_mask=observed[:,:,None,None,None].expand(b,f,c,h,w)'
    if source.count(before) != 1:
        raise ValueError('actual production target-mask assignment differs')
    namespace = dict(vars(measurement_module))
    exec(compile(source.replace(before, after), '<audit-independent-target-support>', 'exec'), namespace)
    return namespace['source_consistent_measurement'], dict(before=before, after=after,
        production_function_sha256=hashlib.sha256(source.encode()).hexdigest())


def exact_match_control():
    source = torch.eye(9).reshape(1, 9, 9)
    target = source.reshape(1, 3, 3, 9).roll(1, 2).reshape(1, 9, 9)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    mask[:, :, 1] = False
    mask = mask.flatten(1)
    old = observed_feature_correspondence(source, target, mask, mask)
    independent = observed_feature_correspondence(source, target, mask, torch.ones_like(mask))
    # Source first-column descriptors move into the hidden second column.
    rows = torch.tensor([0, 3, 6])
    expected = rows + 1
    correct = independent[0, rows, expected]
    assert torch.equal(correct, torch.ones_like(correct))
    assert old[0, rows, expected].count_nonzero() == 0
    same = observed_feature_correspondence(source, source, mask, torch.ones_like(mask))
    assert torch.equal(same[0, mask[0], :-1], torch.eye(9)[mask[0]])
    return dict(scope='Known descriptor permutation, not a real-scene motion oracle',
                moved_visible_sources=3, original_correct_mass=old[0, rows, expected].tolist(),
                independent_correct_mass=correct.tolist(), same_image_exact=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--probe-batches', type=int, default=4)
    parser.add_argument('--memory-cap-gib', type=float, default=14.)
    args, remaining = parser.parse_known_args()
    if args.report.exists() or '--init-checkpoint' not in remaining or args.probe_batches < 1:
        raise ValueError('requires unused report and explicit read-only initialization')
    checkpoint = Path(remaining[remaining.index('--init-checkpoint') + 1])
    saved = torch.load(checkpoint, map_location='meta', weights_only=False)
    step = int(saved['global_step'])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 1024**3 / total, 0)
    torch.use_deterministic_algorithms(True)
    alternate, patch = independent_target_function()
    original_measure = measurement_module.source_consistent_measurement
    original_teacher = ObjectFutureTeacher.forward
    original_loss = loss_module.future_dynamics_terms
    original_step = MainlineTrainingEngine.train_step
    pending = {'active': False}
    report = dict(complete=False, records=[], checkpoint=str(checkpoint),
                  checkpoint_source=saved['identity']['git_commit'], global_step=step,
                  runtime_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  counterfactual_patch=patch, exact_match_control=exact_match_control(),
                  training_mask=True, optimizer_updates=0, production_changed=False,
                  scope=__doc__)

    def measure(module, facts, observations, offsets, observed):
        baseline = original_measure(module, facts, observations, offsets, observed)
        if not pending['active']:
            return baseline
        other = alternate(module, facts, observations, offsets, observed)
        source_mask = facts.dense_chart.cell_observed[..., 0]
        known = torch.ones(observations.shape[:2], dtype=torch.bool, device=observations.device) if observed is None else observed
        fields = ('successor_per_support', 'transport_per_support', 'covariance_per_support',
                  'null_probability', 'current_reference')
        errors = {name: scalar((getattr(baseline, name) - getattr(other, name)).abs().max()) for name in fields}
        assert errors['current_reference'] == 0
        if bool(source_mask.all()):
            assert max(errors.values()) == 0
        future_count = int(observations.shape[1])
        record = dict(future_supports=future_count, source_support_fraction=scalar(source_mask.float().mean()),
                      observed_frames=int(known.sum()), independent_target_available_values_finite=bool(torch.isfinite(observations[known]).all()),
                      excluded_target_value_rms=scalar(observations.float().masked_select(
                          ((~source_mask)[:, None] & known[:, :, None, None, None])[..., None].expand_as(observations)).square().mean().sqrt())
                          if bool(((~source_mask)[:, None] & known[:, :, None, None, None]).any()) else 0.,
                      original_null=scalar(baseline.null_probability.mean()), independent_null=scalar(other.null_probability.mean()),
                      original_transport_rms=scalar(baseline.transport_per_support.square().mean().sqrt()),
                      independent_transport_rms=scalar(other.transport_per_support.square().mean().sqrt()),
                      max_changes=errors)
        pending['record']['measurement_calls'].append(record)
        return baseline

    def teacher(module, *positional, **kw):
        baseline = original_teacher(module, *positional, **kw)
        if not pending['active']:
            return baseline
        measurement_module.source_consistent_measurement = alternate
        try:
            other = original_teacher(module, *positional, **kw)
        finally:
            measurement_module.source_consistent_measurement = measure
        pending['teacher_pair'] = (baseline[0], other[0])
        return baseline

    def dynamics(prediction, target, **kw):
        value = original_loss(prediction, target, **kw)
        if pending['active']:
            before, after = pending['teacher_pair']
            if target is not before:
                raise ValueError('world loss lost the actual measured teacher target')
            other = original_loss(prediction, after, **kw)
            pending['record']['world_loss'] = {name: {'original': scalar(v), 'independent': scalar(other[name])}
                for name, v in value.items() if v.ndim == 0 and name.startswith('future_')}
        return value

    @torch.no_grad()
    def train_step(engine, batch, **unused):
        if batch.online.batch != 8 or engine.optimizer.state:
            raise ValueError('requires BS8 and a fresh, never-updated optimizer')
        engine.global_step = step
        engine.model.set_training_step(step)
        engine.model.train()
        versions = {n: p._version for n, p in engine.model.named_parameters()}
        record = dict(sample_index=batch.audit.sample_index.cpu().tolist(),
                      episode_index=batch.audit.episode_index.cpu().tolist(), measurement_calls=[])
        pending.update(active=True, record=record)
        with _autocast(engine.device, engine.dtype):
            ledger, metrics = engine._forward(batch, training=True, collect_diagnostics=False,
                generator=engine.train_flow_generator, condition_generator=engine.train_condition_generator)
        pending['active'] = False
        if engine.optimizer.state or any(p.grad is not None or p._version != versions[n] for n, p in engine.model.named_parameters()):
            raise RuntimeError('read-only audit changed model or optimizer')
        if not any(x['future_supports'] > 1 and x['source_support_fraction'] < 1 for x in record['measurement_calls']):
            raise ValueError('did not observe the actual masked future teacher')
        record['ordinary_loss'] = scalar(ledger.total)
        report['records'].append(record)
        report['complete'] = len(report['records']) == args.probe_batches
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.report)
        print('FUTURE_SUPPORT_AUDIT', len(report['records']), flush=True)
        if report['complete']:
            raise ProbeComplete()
        return TrainStepResult(loss=ledger.total.detach(), gradient_norm=ledger.total.new_zeros(()),
            learning_rate=0., metrics=engine._tensor_metrics(ledger, metrics), gradient_norm_scalar=0.)

    measurement_module.source_consistent_measurement = measure
    ObjectFutureTeacher.forward = teacher
    loss_module.future_dynamics_terms = dynamics
    MainlineTrainingEngine.train_step = train_step
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        pass
    finally:
        measurement_module.source_consistent_measurement = original_measure
        ObjectFutureTeacher.forward = original_teacher
        loss_module.future_dynamics_terms = original_loss
        MainlineTrainingEngine.train_step = original_step


if __name__ == '__main__':
    main()
