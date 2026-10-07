"""Finish fixed-observation identity/persistence probes after the repair checkpoint.

This watcher owns no training/standard-evaluation process. It checks the final
checkpoint and production source, then uses the pinned training code. Stored
original trajectories are deliberate fixed observations, not new policy rollouts.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args();receipt=json.loads(a.receipt.read_text())
    out=Path(receipt["output"]);out.mkdir(exist_ok=False)
    status=out/"status.json"
    def write(state,**details):
        payload={"state":state,"unix_time":time.time(),**details}
        tmp=status.with_suffix(".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(status)
        print(json.dumps(payload),flush=True)
    try:
        write("waiting_final_checkpoint")
        cp=Path(receipt["checkpoint"]);deadline=time.time()+4*3600
        while not cp.exists():
            if time.time()>deadline:raise TimeoutError("repair checkpoint wait expired")
            try:command=Path("/proc",str(receipt["train_pid"]),"cmdline").read_bytes()
            except FileNotFoundError:raise RuntimeError("training ended without the final checkpoint")
            if receipt["train_config"].encode() not in command:
                raise RuntimeError("training PID no longer owns the declared run")
            time.sleep(30)
        size=cp.stat().st_size;time.sleep(3)
        if cp.stat().st_size!=size:raise RuntimeError("checkpoint size changed during admission")
        import torch
        payload=torch.load(cp,map_location="meta",weights_only=False)
        if (payload.get("epoch"),payload.get("global_step"))!=(1,11524):
            raise RuntimeError("not the 512-update repair checkpoint")
        if payload["identity"]["git_commit"]!=receipt["train_commit"]:
            raise RuntimeError("checkpoint source differs from pinned repair")
        checkpoint_sha=sha(cp);del payload
        source=Path(receipt["production_source"])
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()!=receipt["train_commit"]:
            raise RuntimeError("pinned source moved")
        if subprocess.check_output(["git","status","--porcelain","--","clearvla"],cwd=source,text=True).strip():
            raise RuntimeError("pinned production source is dirty")
        for path,digest in receipt["script_sha256"].items():
            if sha(Path(path))!=digest:raise RuntimeError("probe changed after registration: "+path)
        identity={"checkpoint_sha256":checkpoint_sha,"epoch":1,"global_step":11524,
                  "train_commit":receipt["train_commit"],"production_source":str(source),
                  "scope":"original trajectories with new weights; standard fresh closed loop runs separately"}
        (out/"checkpoint_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
        env=os.environ.copy();env.update(PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=str(receipt["gpu"]),
          HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
        for stage in receipt["stages"]:
            write("running_probe",stage=stage["name"],**identity)
            with (out/(stage["name"]+".log")).open("w") as log:
                subprocess.run(stage["command"],cwd=source,env=env,stdin=subprocess.DEVNULL,
                  stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((Path(stage["output"])/"complete.json").read_text())
            if result["checkpoint_sha256"]!=checkpoint_sha:
                raise RuntimeError("probe checkpoint digest differs: "+stage["name"])
        write("complete",**identity,stages=[s["name"] for s in receipt["stages"]])
    except BaseException as error:
        write("failed",error=repr(error))
        raise

if __name__=="__main__":main()
