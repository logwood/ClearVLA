"""Ordinary two-step BS8 training with read-only actual-loss adapter VJPs."""
from pathlib import Path
import argparse,json,sys,torch
from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine

def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument("--vjp-report",type=Path,required=True)
    args,rest=ap.parse_known_args()
    if args.vjp_report.exists():raise FileExistsError(args.vjp_report)
    original=MainlineTrainingEngine._forward;rows=[]
    def forward(self,batch,*a,**kw):
        result=original(self,batch,*a,**kw)
        if kw.get("training") and torch.is_grad_enabled():
            selected={n:p for n,p in self.model.named_parameters() if n.startswith("grounding.grounder.image_feedback.")}
            if not selected or batch.online.batch!=8:raise ValueError("adapter or real BS8 missing")
            owners=[id(p) for group in self.optimizer.param_groups for p in group["params"]]
            if any(owners.count(id(p))!=1 for p in selected.values()):raise ValueError("adapter optimizer ownership mismatch")
            ledger=result[0];losses={"total":ledger.total,"action":ledger.contributions["action_flow"],
                "world":ledger.contributions["future_dynamics"]+ledger.contributions["future_transition"],
                "identity":ledger.contributions["identity_correspondence"]+ledger.contributions["identity_source_prediction"]}
            metrics={}
            for name,loss in losses.items():
                grads=torch.autograd.grad(loss,list(selected.values()),retain_graph=True,allow_unused=True)
                metrics[name]={n:{"connected":g is not None,"finite":g is None or bool(torch.isfinite(g).all()),
                    "l2":0. if g is None else float(g.detach().float().norm())} for n,g in zip(selected,grads)}
                if not all(x["finite"] for x in metrics[name].values()):raise ValueError("nonfinite adapter loss VJP")
            if not all(x["connected"] for x in metrics["total"].values()):raise ValueError("adapter disconnected from total loss")
            if metrics["total"]["grounding.grounder.image_feedback.out.weight"]["l2"]<=0:raise ValueError("adapter output has no actual-loss gradient")
            rows.append({"step_before_update":self.global_step,"batch_size":8,"training_mask":True,
                         "sample_index":batch.audit.sample_index.cpu().tolist(),"gradients":metrics})
            args.vjp_report.write_text(json.dumps({"complete":False,"rows":rows},indent=2)+"\n")
        return result
    MainlineTrainingEngine._forward=forward;sys.argv=[sys.argv[0],*rest]
    try:train.main()
    finally:MainlineTrainingEngine._forward=original
    if len(rows)!=2:raise ValueError("expected two actual BS8 forward/backward steps")
    args.vjp_report.write_text(json.dumps({"complete":True,"rows":rows,"scope":"actual production loss VJPs; optimizer/backward unchanged"},indent=2)+"\n")
if __name__=="__main__":main()
