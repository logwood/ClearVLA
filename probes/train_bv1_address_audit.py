"""Read-only hooks on each actual BS8 optimizer update; no auxiliary backward."""
from pathlib import Path
import argparse,json,sys,torch
from clearvla.mainline import train
from clearvla.mainline.training.engine import MainlineTrainingEngine
KEY="grounding.grounder.address_memory_gain"

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--audit-report",type=Path,required=True)
    p.add_argument("--expected-updates",type=int,required=True)
    p.add_argument("--address-mode",choices=("none","conditional_logits_v1"),required=True)
    a,rest=p.parse_known_args();a.audit_report.mkdir(exist_ok=False)
    original=MainlineTrainingEngine.train_step;count=0
    with (a.audit_report/"updates.jsonl").open("x") as log:
        def step(self,batch,*args,**kwargs):
            nonlocal count
            params={n:q for n,q in self.model.named_parameters() if n==KEY}
            if bool(params)!=(a.address_mode!="none") or batch.online.batch!=8:
                raise ValueError("actual BS8/parameter selector differs")
            owners=[id(q) for group in self.optimizer.param_groups for q in group["params"]]
            if any(owners.count(id(q))!=1 for q in params.values()):raise ValueError("gain optimizer owner differs")
            before={n:q.detach().clone() for n,q in params.items()};grads={}
            handles=[]
            for n,q in params.items():
                def record(g,name=n):
                    grads[name]=g.detach().float().cpu().tolist()
                    return g
                handles.append(q.register_hook(record))
            clock=self.global_step
            try:result=original(self,batch,*args,**kwargs)
            finally:
                for h in handles:h.remove()
            if set(grads)!=set(params) or any(not torch.isfinite(torch.tensor(g)).all() for g in grads.values()):
                raise ValueError("disconnected/nonfinite ordinary gain gradient")
            row=dict(step_before=clock,step_after=self.global_step,
                sample_index=batch.audit.sample_index.cpu().tolist(),batch_size=8,
                actual_total_loss_gradients=grads,
                gain_after={n:q.detach().float().cpu().tolist() for n,q in params.items()},
                update_l2={n:float((q.detach()-before[n]).float().norm()) for n,q in params.items()})
            log.write(json.dumps(row,allow_nan=False)+"\n");log.flush();count+=1
            return result
        MainlineTrainingEngine.train_step=step;sys.argv=[sys.argv[0],*rest]
        try:train.main()
        finally:MainlineTrainingEngine.train_step=original
    if count!=a.expected_updates:raise ValueError("declared optimizer updates incomplete")
    (a.audit_report/"summary.json").write_text(json.dumps(dict(complete=True,updates=count,batch_size=8,
        scope="hooks on actual total-loss backward; no added loss, backward, clipping or optimizer"),indent=2)+"\n")
if __name__=="__main__":main()
