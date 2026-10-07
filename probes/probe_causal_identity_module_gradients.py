"""One real production BS8 loss VJP, no update, for A/B's changed consumers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import torch

from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    original = MainlineTrainingEngine.train_step

    def inspect(self, batch, **unused):
        if batch.online.batch != 8:
            raise ValueError('the declared full-loss probe requires BS8')
        model = self.model
        model.train()
        model.set_training_step(self.global_step)
        versions = {n: p._version for n, p in model.named_parameters()}
        prefixes = ('intent.organizer.observed_outcome.', 'intent.organizer.shared_binder.',
                    'intent.organizer.object_identity.', 'world.dynamics.object_identity.',
                    'world.dynamics.observed_view_content.', 'world.dynamics.observed_view_role.',
                    'grounding.grounder.canonical_decoder.')
        selected = {n: p for n, p in model.named_parameters() if p.requires_grad and n.startswith(prefixes)}
        if not selected or self.optimizer.state:
            raise ValueError('expected changed consumers and an unstepped fresh optimizer')
        with _autocast(self.device, self.dtype):
            ledger, _ = self._forward(batch, training=True, collect_diagnostics=False,
                                      generator=self.train_flow_generator,
                                      condition_generator=self.train_condition_generator)
        losses = {'total': ledger.total, 'action_flow': ledger.contributions['action_flow'],
                  'world': ledger.contributions['future_dynamics'] + ledger.contributions['future_transition']}
        if 'identity_correspondence' in ledger.contributions:
            losses['identity'] = ledger.contributions['identity_correspondence'] + ledger.contributions['identity_source_prediction']
        results = {}
        for index, (name, loss) in enumerate(losses.items()):
            grads = torch.autograd.grad(loss, list(selected.values()), retain_graph=index < len(losses) - 1, allow_unused=True) if loss.requires_grad else [None] * len(selected)
            rows = {}
            for key, value in zip(selected, grads):
                rows[key] = dict(connected=value is not None, finite=value is None or bool(torch.isfinite(value).all()),
                                 l2=0. if value is None else float(value.detach().float().norm()),
                                 rms=0. if value is None else float(value.detach().float().square().mean().sqrt()))
            if not all(row['finite'] for row in rows.values()):
                raise ValueError('nonfinite changed-consumer VJP in ' + name)
            results[name] = dict(loss=float(loss.detach()), parameters=rows)
        if self.optimizer.state or any(p.grad is not None or p._version != versions[n] for n, p in model.named_parameters()):
            raise RuntimeError('VJP probe changed model or optimizer state')
        report = dict(complete=True, records=[results], global_step=self.global_step, batch_size=8,
                      runtime_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      sample_index=batch.audit.sample_index.cpu().tolist(),
                      episode_index=batch.audit.episode_index.cpu().tolist(),
                      training_mask=True, optimizer_updates=0, parameter_grad_writes=False,
                      scope='ordinary production-loss local VJPs; nonzero does not prove correct identity or task success')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('MODULE_VJP_COMPLETE', str(args.report), flush=True)
        raise ProbeComplete()

    MainlineTrainingEngine.train_step = inspect
    sys.argv = [sys.argv[0], *remaining]
    try:
        train.main()
    except ProbeComplete:
        pass
    finally:
        MainlineTrainingEngine.train_step = original


if __name__ == '__main__':
    main()
