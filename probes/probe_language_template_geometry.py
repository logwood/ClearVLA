"""Matched natural language template audit; no image, action or optimizer inputs."""
import argparse,hashlib,itertools,json
from pathlib import Path
import torch
from clearvla.simulation.checkpoint import load_deployment_checkpoint

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    bundle=load_deployment_checkpoint(a.checkpoint,device=torch.device("cpu"),
       t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"))
    model=bundle.model.eval();organizer=model.intent.organizer;bank=bundle.language
    groups={}
    for i,text in enumerate(bank.instructions):
        words=text.split()
        colors=[c for c in ("red","blue","pink") if c in words]
        directions=[c for c in ("left","right") if c in words]
        if len(colors)!=1 or len(directions)!=1 or "block" not in words:continue
        color,direction=colors[0],directions[0]
        template=" ".join("{color}" if w==color else "{direction}" if w==direction else w for w in words)
        groups.setdefault(template,{})[(color,direction)]=i
    groups={k:v for k,v in groups.items() if len(v)==6}
    if not groups:raise RuntimeError("no matched six-task templates")
    rows=[]
    def rms(x):return float(x.float().square().mean().sqrt())
    with torch.no_grad():
        for template,indices in sorted(groups.items()):
            conditions=list(indices);idx=[indices[c] for c in conditions]
            tokens=bank.tokens[idx];mask=bank.mask[idx]
            memory=organizer.goal_input(torch.where(mask[...,None],tokens,0.0))
            protected,_,_=organizer.goal_read(organizer.goal_queries.expand(len(idx),-1,-1),
                memory,padding_mask=~mask,diagnostics=False)
            final=organizer.goal_self(protected)
            values={"t5_mean":(tokens*mask[...,None]).sum(1)/mask.sum(1)[:,None],
                    "goal_input_mean":(memory*mask[...,None]).sum(1)/mask.sum(1)[:,None],
                    "goal_read":protected,"goal_self":final}
            pairs=[]
            for i,j in itertools.combinations(range(6),2):
                ca,da=conditions[i];cb,db=conditions[j]
                if (ca==cb)==(da==db):continue
                pairs.append({"a":conditions[i],"b":conditions[j],"kind":"color" if da==db else "direction",
                    "rms_delta":{key:rms(value[i]-value[j]) for key,value in values.items()}})
            rows.append({"template":template,"pairs":pairs})
    summary={}
    for kind in ("color","direction"):
        pairs=[p for r in rows for p in r["pairs"] if p["kind"]==kind]
        summary[kind]={"pairs":len(pairs),**{key:sum(p["rms_delta"][key] for p in pairs)/len(pairs)
            for key in pairs[0]["rms_delta"]}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"checkpoint_sha256":bundle.checkpoint_sha256,
        "global_step":bundle.global_step,"templates":len(rows),"summary":summary,"rows":rows,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"pure trained language stages, FP32 CPU, exact natural templates; no action/identity efficacy claim"},indent=2)+"\n")
    print(json.dumps({"templates":len(rows),"summary":summary}),flush=True)
if __name__=="__main__":main()
