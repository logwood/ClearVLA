"""Admitted fresh full ManiSkill training followed by the common complete panel."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def atomic(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--memory-cap-gib",type=float,default=18.)
    args=p.parse_args();r=args.root
    from clearvla.mainline.checkpoint import active_source_snapshot
    admission=json.loads((r/"admission.json").read_text())
    if admission["status"]!="passed":raise RuntimeError("Structural admission did not pass")
    if admission["source_digest"]!=active_source_snapshot(ROOT).digest:raise RuntimeError("Qualified source changed")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Training requires clean committed source")
    if (r/"pipeline-status.json").exists():raise FileExistsError("Use a new pipeline attempt")
    state={"status":"running","source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
           "source_digest":admission["source_digest"],"initialization":"fresh; frozen DINO/T5; all policy owners trainable",
           "gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"stages":[]}
    def save():atomic(r/"pipeline-status.json",state)
    def run(name,command):
        state["active_stage"]=name
        row={"name":name,"command":command,"started_unix":time.time(),"log":str(r/(name+".log"))}
        state["stages"].append(row);save()
        with (r/(name+".log")).open("x") as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            row["pid"]=proc.pid;save()
            row["returncode"]=proc.wait()
        row["finished_unix"]=time.time();save()
        if row["returncode"]:raise RuntimeError(name+" failed; inspect "+row["log"])
    py=sys.executable
    cfg=ROOT/"configs/mainline/maniskill_spatial_policy_full_20261008.json"
    data="/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2"
    dino="/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"
    try:
        run("full-train",[py,"-B","-u","scripts/train_mainline_memory_capped.py",
            "--memory-cap-gib",str(args.memory_cap_gib),"--config",str(cfg),
            "--output-dir",str(r/"train"),"--device","cuda:0"])
        rows=[json.loads(x) for x in (r/"train/metrics.jsonl").read_text().splitlines()]
        epochs=[x for x in rows if x.get("kind")=="epoch"]
        if len(epochs)!=20:raise RuntimeError("Full 20-epoch training is incomplete")
        context=json.loads((r/"train/run_context.json").read_text())
        if context["config"]["runtime"]["max_train_batches"] or context["config"]["runtime"]["max_val_batches"]:
            raise RuntimeError("Full training/validation was capped")
        if context["identity"]["source"]["digest"]!=admission["source_digest"]:raise RuntimeError("Training source differs")
        if any(x.get("kind")=="gradient_failure" for x in rows):raise RuntimeError("Nonfinite gradient")
        checkpoint=r/"train/checkpoints/best.pt"
        state["selected_checkpoint"]={"path":str(checkpoint),"sha256":sha256(checkpoint),
                                      "selection":"lowest full-validation normalized action RMSE"}
        save()
        run("common-eval18",[py,"-B","-u","scripts/record_maniskill_npz.py",
            "--checkpoint",str(checkpoint),"--output-dir",str(r/"common-eval18"),
            "--episodes","18","--eval-seed","5000000","--policy-seed","0",
            "--max-episode-steps","400","--replan-steps","8",
            "--t5-condition",data+"/language/t5_xxl.pt","--dinov3-model",dino])
        panel=json.loads((r/"common-eval18/manifest.json").read_text())
        if not panel["complete"] or len(panel["episodes"])!=18:raise RuntimeError("Incomplete common panel")
        state["status"]="complete";state["active_stage"]=None;save()
    except BaseException as error:
        state["status"]="failed";state["error"]=repr(error);save();raise

if __name__=="__main__":main()

