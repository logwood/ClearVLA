"""Actual B-v1 address trial: mechanical, cold, BS8 short, standard18, factual.

The runner never starts a formal long or changes existing jobs. A no-memory
continuation uses the identical exposure/checkpoint to separate more training
from the structural candidate. Cold checks are mechanical, not effectiveness.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,math,os,subprocess,time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
def dump(path,value):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True);a=p.parse_args()
    r=json.loads(a.receipt.read_text());E=Path(r["root"]);T=Path(r["source"]);out=Path(r["output"])
    out.mkdir(exist_ok=False);held=None
    def state(s,**kw):dump(out/"status.json",dict(state=s,unix_time=time.time(),formal_promoted=False,**kw))
    try:
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=T,text=True).strip()!=r["commit"] or subprocess.check_output(["git","status","--porcelain"],cwd=T,text=True).strip():
            raise ValueError("immutable production source changed")
        for name,digest in r["sha256"].items():
            if sha(name)!=digest:raise ValueError("pinned input differs: "+name)
        init=json.loads(Path(r["initialization_admission"]).read_text())
        if not init["complete"] or len(init["records"])!=2:raise ValueError("full-model B-v1 initialization proof missing")
        fact=json.loads(Path(r["zero_admission"]).read_text())
        if not fact["complete"] or not fact["all_zero_init_exact"]:raise ValueError("actual zero initialization not exact")
        import torch
        from clearvla.mainline.config import load_config
        from clearvla.mainline.runtime.checkpoints import _initialization_config_view
        from clearvla.mainline.runtime.address_memory_migration import config_view
        mechanical=load_config(r["mechanical_config"]);short=load_config(r["short_config"])
        if config_view(_initialization_config_view(mechanical)) != config_view(_initialization_config_view(short)):
            raise ValueError("short model/objective/data changed from mechanical")
        for cfg,steps,val in [(mechanical,2,2),(short,1024,256)]:
            if cfg.optimizer.batch_size!=8 or cfg.optimizer.update_origin!=12036 or cfg.optimizer.epochs!=1 or cfg.runtime.max_train_batches!=steps or cfg.runtime.max_val_batches!=val:
                raise ValueError("declared clock/BS8/exposure changed")
            if cfg.top.identity_supervision_mode!="rgbd_temporal_v1" or cfg.top.entity_address_memory_mode!=r["mode"]:
                raise ValueError("not the declared B-v1 objective/structure")
            if Path(cfg.data.output_dir).exists():raise FileExistsError(cfg.data.output_dir)
        # Shared lock with both existing SAM explorations, plus fresh whole-GPU check.
        state("waiting_idle_gpu");deadline=time.monotonic()+72*3600
        locks=Path(r["gpu_locks"]);locks.mkdir(exist_ok=True)
        while time.monotonic()<deadline:
            rows=[list(map(str.strip,line.split(","))) for line in subprocess.check_output(
                ["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total","--format=csv,noheader,nounits"],text=True).splitlines()]
            for index,uuid,used,total in rows:
                if uuid not in r["allowed_gpus"] or int(used)>=512 or int(total)<24000:continue
                handle=(locks/(uuid+".lock")).open("a")
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                fresh=int(subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True))
                if fresh>=512:handle.close();continue
                held=handle;gpu=uuid;egl=index;break
            if held is not None:break
            time.sleep(30)
        if held is None:raise TimeoutError("no idle GPU within72h")
        env=dict(os.environ,PYTHONPATH=str(T),CUDA_VISIBLE_DEVICES=gpu,EGL_VISIBLE_DEVICES=egl,
            OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",PYTORCH_ALLOC_CONF="expandable_segments:True")
        env.pop("CUBLAS_WORKSPACE_CONFIG",None)
        def execute(stage,cmd,deterministic=False):
            runenv=dict(env)
            if deterministic:runenv["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            with (out/(stage+".log")).open("x") as f:
                child=subprocess.Popen(cmd,cwd=T,env=runenv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                state("running",stage=stage,pid=child.pid,gpu=gpu,egl=egl,command=cmd)
                if child.wait():raise RuntimeError(stage+" failed; original log retained")
        def train(config,mechanical_run):
            cmd=[r["python"],"-B","-u"]
            if mechanical_run and r["mode"]!="none":
                cmd += [str(T/"probes/train_bv1_address_mechanical.py"),"--vjp-report",str(out/"ordinary-backward.json")]
            else:cmd += [r["training_audit"],"--audit-report",str(out/("mechanical-updates" if mechanical_run else "short-updates")),
                "--expected-updates","2" if mechanical_run else "1024","--address-mode",r["mode"]]
            return cmd+["--config",config,"--init-checkpoint",r["checkpoint"],
                "--init-model-contract-migration","b_v1_address_memory_v1","--init-training-clock","checkpoint","--device","cuda:0"]
        def audit(cfg,steps,label):
            run=Path(cfg.data.output_dir);ck=run/"checkpoints/latest.pt"
            payload=torch.load(ck,map_location="cpu",weights_only=False,mmap=True)
            if payload["epoch"]!=1 or payload["global_step"]!=12036+steps or payload["identity"]["git_commit"]!=r["commit"] or payload["identity"]["config_digest"]!=cfg.digest():
                raise ValueError(label+" checkpoint identity differs")
            if not all(bool(torch.isfinite(v).all()) for v in payload["model"].values() if isinstance(v,torch.Tensor) and v.is_floating_point()):
                raise ValueError("saved weights nonfinite")
            rows=[json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
            trainrows=[x for x in rows if x.get("kind")=="train"];epochs=[x for x in rows if x.get("kind")=="epoch"]
            if sum(x["window_batches"] for x in trainrows)!=steps or sum(x["window_samples"] for x in trainrows)!=steps*8 or len(epochs)!=1:
                raise ValueError(label+" exposure incomplete")
            for row in rows:
                for key in ("metrics","train","validation"):
                    if any(isinstance(v,float) and not math.isfinite(v) for v in row.get(key,{}).values()):raise ValueError("nonfinite metrics")
            key="grounding.grounder.address_memory_gain"
            report=dict(complete=True,source=r["commit"],checkpoint=str(ck),sha256=sha(ck),
                epoch=1,global_step=12036+steps,batch_size=8,updates=steps,
                address_gain=payload["model"][key].tolist() if key in payload["model"] else None,
                validation=epochs[0]["validation"],scope="health/identity only; not behavior admission")
            dump(out/(label+"-audit.json"),report)
            return ck
        execute("mechanical",train(r["mechanical_config"],True))
        ck=audit(mechanical,2,"mechanical")
        if r["mode"]!="none":
            v=json.loads((out/"ordinary-backward.json").read_text())
            if not v["complete"] or len(v["rows"])!=2:raise ValueError("ordinary backward incomplete")
            for row in v["rows"]:
                g=row["total_loss_gradients"]["grounding.grounder.address_memory_gain"]
                if not g["finite"] or not g["connected"] or g["l2"]<=0:raise ValueError("address gradient not live")
        cold=out/"cold4"
        cmd=[r["python"],"-B","-u",r["factual_probe"],"--checkpoint",str(ck),"--plan",r["factual_plan"],
             "--masks",r["factual_masks"],"--output",str(cold),"--deterministic"]
        execute("cold4",cmd,True)
        c=json.loads((cold/"results.json").read_text())
        if not c["complete"] or len(c["records"])!=4:raise ValueError("cold4 incomplete")
        for row in c["records"]:
            if row["natural"][1]["native_arm_difference_rms"]!=0 or row["natural"][1]["native_gripper_changed_rows"]!=0:
                raise ValueError("same-runtime repeat differs")
            if any(row["static_contract"][k]!=0 for k in ("semantic_max","image_max","covariance_max")):
                raise ValueError("same-image measurement differs")
        dump(out/"mechanical-admission.json",dict(mechanically_admitted=True,source=r["commit"],mode=r["mode"],
            scope="permits meaningful short only; no formal promotion or effectiveness claim"))
        # Always initialize the actual short from original completed B-v1,
        # never continue from the two mechanical updates.
        execute("short1024",train(r["short_config"],False))
        ck=audit(short,1024,"short")
        status=out/"training-complete.json"
        dump(status,dict(status="training_and_offline_finished",returncode=0))
        panel=E/(r["name"]+"-closed-loop-replan8")
        execute("standard18",[r["python"],"-B","-u",r["panel_runner"],"--repo",str(T),
            "--run",short.data.output_dir,"--manifest",r["panel_manifest"],"--output",str(panel),
            "--training-status",str(status),"--source",r["commit"],"--step","13060","--epoch","1",
            "--checkpoint-file","latest.pt","--gpu",gpu,"--egl",egl,"--port",str(r["port"])])
        q=json.loads((panel/"summary.json").read_text())
        if not q["complete"] or q["trials"]!=18 or q["errors"]:raise ValueError("standard18 incomplete")
        inventory=[]
        for row in q["records"]:
            z=Path(row["result_path"]).with_name("trajectory.npz")
            inventory.append(dict(index=row["index"],path=str(z),sha256=sha(z),bytes=z.stat().st_size))
        dump(out/"trajectory-inventory.json",dict(complete=True,records=inventory))
        final=out/"final-factual4"
        cmd=[r["python"],"-B","-u",r["factual_probe"],"--checkpoint",str(ck),"--plan",r["factual_plan"],
             "--masks",r["factual_masks"],"--output",str(final),"--deterministic"]
        execute("final-factual4",cmd,True)
        for stage in r.get("post_panel_stages",[]):
            execute(stage["name"],stage["command"],stage.get("deterministic",False))
            value=json.loads(Path(stage["result"]).read_text())
            kind=stage["contract"]
            passed=(isinstance(value,list) and len(value)==18) if kind=="plan18" else (
                value.get("status")=="complete" if kind=="status_complete" else value.get("complete") is True and bool(value.get("records")))
            if not passed:raise ValueError(stage["name"]+" output incomplete")
        state("complete",successes=q["successes"],trials=18,checkpoint=str(ck),
            scope="short/standard18/source/object/own-trajectory evidence complete; review required; no formal promotion")
    except BaseException as exc:state("failed",error=repr(exc));raise
    finally:
        if held is not None:held.close()
if __name__=="__main__":main()
