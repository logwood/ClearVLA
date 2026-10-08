"""Read-only real-batch VJPs plus ordinary optimizer updates before full training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clearvla.mainline.model.component_contracts import legacy_named_parameters
from clearvla.mainline.training.engine import MainlineTrainingEngine


def owner(name):
    if "spatial_posterior_context" in name:
        return "W_spatial" if ".dynamics." in name else "P2_spatial"
    if ".spatial_context." in name: return "S_spatial"
    if ".effect_context_key." in name: return "P2_expected_actual"
    if "shared_binder" in name: return "shared_target"
    return None


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--memory-cap-gib",type=float,default=18.)
    args,rest=parser.parse_known_args()
    if args.receipt.exists():raise FileExistsError(args.receipt)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("Select exactly one GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    forward=MainlineTrainingEngine._forward_encoded
    train=MainlineTrainingEngine.train_step
    receipt={"status":"running","updates":[]}
    audited=0
    def save():
        temp=args.receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        temp.replace(args.receipt)
    def traced(engine,batch,*,encoded,**kwargs):
        nonlocal audited
        ledger,metrics=forward(engine,batch,encoded=encoded,**kwargs)
        if audited >= 2 or not engine.model.training:return ledger,metrics
        audited += 1
        tracked=[(n,p) for n,p in legacy_named_parameters(engine.model) if p.requires_grad and owner(n)]
        counts={key:0 for key in ("W_spatial","P2_spatial","S_spatial","P2_expected_actual","shared_target")}
        params=[p for g in engine.optimizer.param_groups for p in g["params"]]
        assert tracked and all(sum(x is p for x in params)==1 for _,p in tracked)
        rng=torch.cuda.get_rng_state().clone()
        grads=torch.autograd.grad(ledger.total,[p for _,p in tracked],retain_graph=True,allow_unused=True)
        rows=[]
        for (n,p),g in zip(tracked,grads):
            if g is None:continue
            if not torch.isfinite(g).all():raise FloatingPointError(n)
            rms=float(g.float().square().mean().sqrt())
            rows.append({"name":n,"owner":owner(n),"rms":rms})
            if rms>0:counts[owner(n)]+=1
        # Fresh W heads are exactly zero on the first forward. The next
        # ordinary update must reach every new owner without artificial gates.
        if audited == 2 and not all(counts.values()):raise RuntimeError("Missing structural VJP: "+repr(counts))
        if not torch.equal(rng,torch.cuda.get_rng_state()):raise RuntimeError("Read-only VJP changed CUDA RNG")
        law=encoded.cache.top.belief.camera_position_probability
        assert law is encoded.cache.top.predicted_dynamics.camera_position_probability
        assert law.shape[-2:]==(16,16)
        receipt.update(audited_step=engine.global_step,owner_vjps=rows,live_owner_counts=counts,
                       loss_groups={k:float(v.detach()) for k,v in ledger.groups.items()},
                       structural_terms={k:float(v.detach()) for k,v in ledger.terms.items() if k.startswith("spatial_")},
                       spatial_law_shape=list(law.shape),optimizer_ownership="exactly_once",
                       read_only_vjp_rng_unchanged=True)
        save()
        return ledger,metrics
    def stepped(engine,batch,**kwargs):
        before={n:p.detach().clone() for n,p in legacy_named_parameters(engine.model) if p.requires_grad and owner(n)}
        result=train(engine,batch,**kwargs)
        changed={key:0 for key in ("W_spatial","P2_spatial","S_spatial","P2_expected_actual","shared_target")}
        for n,p in legacy_named_parameters(engine.model):
            if n in before:
                if not torch.isfinite(p).all():raise FloatingPointError(n)
                if not torch.equal(p,before[n]):changed[owner(n)]+=1
        if receipt["updates"] and not all(changed.values()):raise RuntimeError("No optimizer change for "+repr(changed))
        receipt["updates"].append({"step":engine.global_step,"changed_owner_counts":changed})
        receipt["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/1024**3
        save()
        return result
    sys.argv=["clearvla.mainline.train",*rest]
    try:
        with patch.object(MainlineTrainingEngine,"_forward_encoded",traced),patch.object(MainlineTrainingEngine,"train_step",stepped):
            runpy.run_module("clearvla.mainline.train",run_name="__main__",alter_sys=True)
        if audited != 2 or len(receipt["updates"])!=2:raise RuntimeError("Expected exactly two real updates")
        receipt["status"]="passed";save()
        print("STRUCTURAL_PREFLIGHT_PASSED "+str(args.receipt),flush=True)
    except BaseException as error:
        receipt["status"]="failed";receipt["error"]=repr(error);save();raise

if __name__=="__main__":main()

