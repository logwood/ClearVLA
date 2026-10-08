"""Two ordinary BS8 updates with a read-only first-loss parity/VJP audit.

The probe adds no optimizer operation. Original loss and RNG evolution are
preserved; extra backwards use autograd.grad without writing parameter.grad.
It is a mechanical check, not a meaningful training or behavior qualification.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import torch
from clearvla.mainline import train
from clearvla.mainline.training import identity


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--audit-report',type=Path,required=True)
    parser.add_argument('--reference-source',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    if args.audit_report.exists():raise FileExistsError(args.audit_report)
    source=args.reference_source/'clearvla/mainline/training/identity.py'
    spec=importlib.util.spec_from_file_location('clearvla.mainline.training._reference_identity_v2',source)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    original=identity.identity_terms
    inspected=False

    def inspect(model,online,facts,labels):
        nonlocal inspected
        if inspected:return original(model,online,facts,labels)
        if not model.training or not torch.is_grad_enabled():return original(model,online,facts,labels)
        if online.batch!=8 or model.config.top.identity_supervision_mode!='rgbd_temporal_regions_v3':
            raise ValueError('expected the explicit v3 BS8 graph')
        versions={n:p._version for n,p in model.named_parameters()}
        rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        values=original(model,online,facts,labels)
        after_rng=torch.random.get_rng_state();after_cuda=torch.cuda.get_rng_state_all()
        proxy=SimpleNamespace(grounding=model.grounding,observation=model.observation,
            config=SimpleNamespace(top=SimpleNamespace(identity_supervision_mode='rgbd_temporal_conditional_v2')))
        torch.random.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        with torch.no_grad():old=reference.identity_terms(proxy,online,facts,labels)
        torch.random.set_rng_state(after_rng);torch.cuda.set_rng_state_all(after_cuda)
        parity={name:float((values[name].detach().float()-value.float()).abs().max()) for name,value in old.items()}
        if max(parity.values())>3e-7:raise ValueError('original identity objectives changed: '+str(parity))
        selected={n:p for n,p in model.named_parameters() if p.requires_grad and n.startswith('grounding.grounder.')}
        records={}
        for name in ('identity_correspondence','identity_source_prediction','identity_region_separation','identity_region_prediction'):
            grads=torch.autograd.grad(values[name],list(selected.values()),retain_graph=True,allow_unused=True)
            rows={n:dict(connected=g is not None,finite=g is None or bool(torch.isfinite(g).all()),
                l2=0. if g is None else float(g.detach().float().norm())) for n,g in zip(selected,grads)}
            if not all(r['finite'] for r in rows.values()):raise ValueError('nonfinite ordinary VJP in '+name)
            records[name]=dict(raw_loss=float(values[name].detach()),parameters=rows,
                combined_l2=sum(r['l2']**2 for r in rows.values())**.5)
            del grads
        if any(p._version!=versions[n] or p.grad is not None for n,p in model.named_parameters()):
            raise RuntimeError('first-loss VJP changed model state')
        if float(values['identity_region_supported_negative_pairs'])<=0:
            raise ValueError('first real batch supplies no negative pairs; cannot qualify this branch')
        if records['identity_region_separation']['combined_l2']<=0 or records['identity_region_prediction']['combined_l2']<=0:
            raise ValueError('new losses have no ordinary gradient on this real batch')
        report=dict(complete=True,records=[records],original_objective_parity=parity,
            supported_negative_pairs=float(values['identity_region_supported_negative_pairs']),
            prediction_strata=float(values['identity_region_prediction_strata']),
            batch_size=online.batch,training_mask=True,ordinary_optimizer_unchanged=True,
            reference_source=str(args.reference_source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__)
        args.audit_report.parent.mkdir(parents=True,exist_ok=True)
        args.audit_report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('REGION_FIRST_LOSS_AUDIT_COMPLETE',args.audit_report,flush=True)
        inspected=True
        return values

    identity.identity_terms=inspect
    sys.argv=[sys.argv[0],*remaining]
    try:train.main()
    finally:identity.identity_terms=original
    if not inspected:raise RuntimeError('training entry never reached the explicit v3 loss')


if __name__=='__main__':main()
