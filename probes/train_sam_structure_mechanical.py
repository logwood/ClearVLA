"""Ordinary BS8 training with read-only hooks on its actual backward gradients."""
from pathlib import Path
import argparse,json,sys,torch
from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine

def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument("--vjp-report",type=Path,required=True)
    args,rest=ap.parse_known_args()
    if args.vjp_report.exists():raise FileExistsError(args.vjp_report)
    original=MainlineTrainingEngine.train_step;rows=[]
    def step(self,batch,*a,**kw):
        selected={n:p for n,p in self.model.named_parameters() if n.startswith("grounding.grounder.image_feedback.")}
        if not selected or batch.online.batch!=8:raise ValueError("adapter or BS8 missing")
        owners=[id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if any(owners.count(id(p))!=1 for p in selected.values()):raise ValueError("adapter optimizer ownership mismatch")
        before={n:p.detach().clone() for n,p in selected.items()}
        gradients={};hooks=[]
        for n,p in selected.items():
            def record(g,name=n):
                gradients[name]={"connected":True,"finite":bool(torch.isfinite(g).all()),
                                 "l2":float(g.detach().float().norm())}
                return g
            hooks.append(p.register_hook(record))
        clock=self.global_step
        try:result=original(self,batch,*a,**kw)
        finally:
            for handle in hooks:handle.remove()
        if set(gradients)!=set(selected) or not all(g["finite"] for g in gradients.values()):
            raise ValueError("missing/nonfinite ordinary adapter gradient")
        if gradients["grounding.grounder.image_feedback.out.weight"]["l2"]<=0:
            raise ValueError("zero ordinary output gradient")
        deltas={n:float((p.detach()-before[n]).float().norm()) for n,p in selected.items()}
        rows.append({"step_before_update":clock,"step_after_update":self.global_step,
            "batch_size":8,"training_mask":True,"sample_index":batch.audit.sample_index.cpu().tolist(),
            "total_loss_gradients":gradients,"parameter_update_l2":deltas})
        args.vjp_report.write_text(json.dumps({"complete":False,"rows":rows},indent=2)+"\n")
        return result
    MainlineTrainingEngine.train_step=step;sys.argv=[sys.argv[0],*rest]
    try:train.main()
    finally:MainlineTrainingEngine.train_step=original
    if len(rows)!=2:raise ValueError("expected two BS8 updates")
    args.vjp_report.write_text(json.dumps({"complete":True,"rows":rows,
        "scope":"hooks on ordinary production total-loss backward, no extra VJP/backward or optimizer modification"},indent=2)+"\n")
if __name__=="__main__":main()
