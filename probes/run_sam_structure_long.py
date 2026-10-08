"""Pinned B-v2 exploratory long run followed by its standard18 R8 panel."""
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
    r=json.loads(a.receipt.read_text());root=Path(r["root"]);source=Path(r["source"]);out=Path(r["output"])
    out.mkdir(exist_ok=False)
    def state(name,**kw):dump(out/"status.json",{"state":name,"unix_time":time.time(),"exploratory":True,**kw})
    def check():
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()!=r["commit"] or subprocess.check_output(["git","status","--porcelain"],cwd=source,text=True).strip():
            raise ValueError("pinned production checkout changed")
        for name,digest in r["file_sha256"].items():
            if sha(name)!=digest:raise ValueError("pinned input changed: "+name)
        for path in r["admissions"]:
            if json.loads(Path(path).read_text()).get("mechanically_admitted") is not True:
                raise ValueError("actual backward/cold admission missing")
        cfg=json.loads(Path(r["config"]).read_text())
        if cfg["optimizer"]["batch_size"]!=8 or cfg["optimizer"]["epochs"]!=1 or cfg["optimizer"]["update_origin"]!=12036 or cfg["runtime"]["max_train_batches"]!=11012 or cfg["runtime"]["max_val_batches"]!=256:
            raise ValueError("declared BS8/11012+256 exposure changed")
        if cfg["top"]["identity_supervision_mode"]!="rgbd_temporal_conditional_v2" or cfg["top"]["observation_measurement_mode"]!="source_consistent_v1" or cfg["top"]["entity_image_feedback_mode"]!=r["mode"]:
            raise ValueError("not the declared B-v2-only structure trial")
        if Path(cfg["data"]["output_dir"]).exists():raise FileExistsError("training output already exists")
        return cfg
    held=None
    try:
        cfg=check();deadline=time.monotonic()+72*3600;state("waiting_idle_gpu")
        locks=root/"gpu-locks";locks.mkdir(exist_ok=True)
        while time.monotonic()<deadline:
            protected=set()
            for item in r.get("mainline_reservations",[]):
                status=Path(item["status"])
                if not status.exists() or json.loads(status.read_text()).get("state") not in {"complete","failed"}:
                    protected.add(item["gpu"])
            lines=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total","--format=csv,noheader,nounits"],text=True).splitlines()
            rows=[list(map(str.strip,line.split(","))) for line in lines]
            rows.sort(key=lambda row:r["allowed_gpus"].index(row[1]) if row[1] in r["allowed_gpus"] else 100)
            for index,uuid,used,total in rows:
                if uuid not in r["allowed_gpus"] or uuid in protected or int(used)>=512 or int(total)<24000:continue
                handle=(locks/(uuid+".lock")).open("a")
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                fresh=int(subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
                if fresh>=512:handle.close();continue
                held=handle;gpu=uuid;egl=index;break
            if held is not None:break
            time.sleep(30)
        if held is None:raise TimeoutError("no idle GPU before 72h scheduling deadline")
        env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=gpu,EGL_VISIBLE_DEVICES=egl,
            OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",PYTORCH_ALLOC_CONF="expandable_segments:True")
        def execute(stage,cmd,log):
            with Path(log).open("x") as f:
                child=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                info=dict(pid=child.pid,command=cmd,gpu=gpu,egl=egl,source=r["commit"],unix_time=time.time())
                dump(out/(stage+"-job.json"),info);state("running",stage=stage,**info)
                code=child.wait()
            if code:raise RuntimeError(stage+" failed; log retained: "+str(log))
        cmd=[r["python"],"-B","-u","-m","clearvla.mainline.train","--config",r["config"],
            "--init-checkpoint",r["init_checkpoint"],"--init-model-contract-migration","sam_structure_v1",
            "--init-training-clock","checkpoint","--device","cuda:0"]
        execute("training",cmd,root/(r["name"]+".log"))
        # Audit the latest checkpoint, never substitute an earlier best epoch.
        import torch
        run=Path(cfg["data"]["output_dir"]);ck=run/"checkpoints/latest.pt"
        payload=torch.load(ck,map_location="meta",weights_only=False)
        from clearvla.mainline.config import load_config
        if payload["epoch"]!=1 or payload["global_step"]!=23048 or payload["identity"]["git_commit"]!=r["commit"] or payload["identity"]["config_digest"]!=load_config(r["config"]).digest():
            raise ValueError("long checkpoint identity/position differs")
        rows=[json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
        training=[x for x in rows if x.get("kind")=="train"];epochs=[x for x in rows if x.get("kind")=="epoch"]
        if sum(x["window_batches"] for x in training)!=11012 or sum(x["window_samples"] for x in training)!=88096 or len(epochs)!=1:
            raise ValueError("long logged exposure differs")
        for row in rows:
            for label in ("metrics","train","validation"):
                for value in row.get(label,{}).values():
                    if isinstance(value,float) and not math.isfinite(value):raise ValueError("nonfinite completed metric")
        ck_sha=sha(ck);audit=dict(complete=True,source=r["commit"],checkpoint=str(ck),checkpoint_sha256=ck_sha,
            epoch=1,global_step=23048,updates=11012,batch_size=8,training_samples=88096,
            validation=epochs[0]["validation"],scope="exploratory structural training; behavior evaluated separately")
        dump(root/(r["name"]+"-final-audit.json"),audit)
        status_path=root/(r["name"]+"-status.json")
        dump(status_path,dict(status="training_and_offline_finished",returncode=0,unix_time=time.time()))
        panel=root/(r["name"]+"-closed-loop-replan8")
        cmd=[r["python"],"-B","-u",str(source/"probes/run_sam_structure_panel.py"),
            "--repo",str(source),"--run",str(run),"--manifest",r["panel_manifest"],"--output",str(panel),
            "--training-status",str(status_path),"--source",r["commit"],"--step","23048","--epoch","1",
            "--checkpoint-file","latest.pt","--gpu",gpu,"--egl",egl,"--port",str(r["port"])]
        execute("standard18",cmd,out/"standard18.log")
        panel_data=json.loads((panel/"summary.json").read_text())
        if not panel_data["complete"] or panel_data["trials"]!=18 or panel_data["errors"]:raise ValueError("standard18 incomplete")
        trajectories=[]
        for row in panel_data["records"]:
            npz=Path(row["result_path"]).with_name("trajectory.npz")
            if not npz.is_file():raise FileNotFoundError(npz)
            trajectories.append(dict(index=row["index"],path=str(npz),sha256=sha(npz),bytes=npz.stat().st_size))
        dump(out/"trajectory-inventory.json",dict(complete=True,records=trajectories))
        state("complete",successes=panel_data["successes"],trials=18,checkpoint_sha256=ck_sha,
            scope="long and standard18 complete; deeper identity/trajectory audit still required")
    except BaseException as exc:
        state("failed",error=repr(exc));raise
    finally:
        if held is not None:held.close()
if __name__=="__main__":main()
