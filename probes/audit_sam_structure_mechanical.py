"""Validate actual two-update training, per-parameter backward and cold deployment."""
from pathlib import Path
import argparse,hashlib,json,math
import torch

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    for name in ("run","vjp","cold","output"):ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--source",required=True);ap.add_argument("--mode",required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);ck=a.run/"checkpoints/latest.pt"
    data=torch.load(ck,map_location="cpu",weights_only=False)
    if data["global_step"]!=12038 or data["epoch"]!=1 or data["identity"]["git_commit"]!=a.source:
        raise ValueError("mechanical checkpoint source/clock differs")
    from clearvla.mainline.config import config_from_mapping
    config=config_from_mapping(data["config"])
    if config.digest()!=data["identity"]["config_digest"] or config.top.entity_image_feedback_mode!=a.mode:
        raise ValueError("checkpoint config identity differs")
    tensors=[x for x in data["model"].values() if isinstance(x,torch.Tensor) and x.is_floating_point()]
    if not all(bool(torch.isfinite(x).all()) for x in tensors):raise ValueError("nonfinite saved model")
    grads=json.loads(a.vjp.read_text())
    if not grads["complete"] or len(grads["rows"])!=2:raise ValueError("ordinary backward report incomplete")
    for row in grads["rows"]:
        if row["batch_size"]!=8 or not all(v["finite"] and v["connected"] for v in row["total_loss_gradients"].values()):
            raise ValueError("adapter backward not connected/finite")
    # The upstream projection is zero-gradient only at exact-zero output init.
    if not all(v["l2"]>0 for v in grads["rows"][-1]["total_loss_gradients"].values()):
        raise ValueError("adapter did not open its ordinary gradient after the first update")
    rows=[json.loads(x) for x in (a.run/"metrics.jsonl").read_text().splitlines()]
    train=[x for x in rows if x.get("kind")=="train"];epochs=[x for x in rows if x.get("kind")=="epoch"]
    if len(train)!=2 or [x["step"] for x in train]!=[12037,12038] or any(x["window_samples"]!=8 for x in train) or len(epochs)!=1:
        raise ValueError("two BS8 update exposure differs")
    for x in rows:
        for name in ("metrics","train","validation"):
            if any(isinstance(v,float) and not math.isfinite(v) for v in x.get(name,{}).values()):
                raise ValueError("nonfinite ordinary metrics")
    cold=json.loads((a.cold/"results.json").read_text())
    if not cold["complete"] or len(cold["records"])!=4:raise ValueError("cold four-window evaluation incomplete")
    for row in cold["records"]:
        if row["natural"][1]["native_arm_difference_rms"]!=0 or row["natural"][1]["native_gripper_changed_rows"]!=0:
            raise ValueError("cold deterministic same-instruction repeat differs")
        if any(row["static_contract"][k]!=0 for k in ("semantic_max","image_max","covariance_max")):
            raise ValueError("same-image measurement changed")
    report=dict(mechanically_admitted=True,source=a.source,mode=a.mode,checkpoint=str(ck),checkpoint_sha256=sha(ck),
        global_step=12038,epoch=1,batch_size=8,updates=2,cold_windows=4,
        parameters={n:int(v.numel()) for n,v in data["model"].items() if n.startswith("grounding.grounder.image_feedback.")},
        gradients=grads["rows"],train_metrics=[{k:v for k,v in x["metrics"].items() if k.startswith(("gradient_window","runtime_window")) or k in ("loss_total","loss_contribution_gap","runtime_cuda_peak_allocated_gib")} for x in train],
        exploration_authorized=True,behavior_improvement_claimed=False)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("gradients","train_metrics")}),flush=True)
if __name__=="__main__":main()
