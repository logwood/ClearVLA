"""Finish already-started mechanical runs; cold-load each; queue authorized longs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
def dump(path,value):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True);a=p.parse_args()
    r=json.loads(a.receipt.read_text());X=Path(r["root"]);T=Path(r["source"]);P=Path(r["probe"])
    out=X/"mechanical-cold-continuation-r1";out.mkdir(exist_ok=False)
    def status(state,**kw):dump(out/"status.json",dict(state=state,unix_time=time.time(),**kw))
    env=dict(os.environ,PYTHONPATH=str(T),CUDA_VISIBLE_DEVICES="3",EGL_VISIBLE_DEVICES="3",
        OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",PYTORCH_ALLOC_CONF="expandable_segments:True",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    try:
        status("waiting_mechanical_finish");deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            proc=Path("/proc")/str(r["region_pid"])
            if not proc.exists():break
            stat=proc.joinpath("stat").read_text().split()[2]
            if stat=="Z":break
            time.sleep(5)
        else:raise TimeoutError("existing region mechanical still running")
        if not json.loads(Path(r["region_vjp"]).read_text())["complete"]:
            raise ValueError("region ordinary training/offline failed")
        while int(subprocess.check_output(["nvidia-smi","-i","3","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True))>=512:
            if time.monotonic()>deadline:raise TimeoutError("GPU3 not idle for cold load")
            time.sleep(5)
        for item in r["variants"]:
            ck=Path(item["mechanical"])/"checkpoints/latest.pt"
            cold=X/(item["name"]+"-cold4-r1")
            cmd=[r["python"],"-B","-u",str(T/"probes/probe_causal_identity_admission.py"),
                "--checkpoint",str(ck),"--plan",r["plan"],"--masks",r["masks"],"--output",str(cold),"--deterministic"]
            with (out/(item["name"]+"-cold.log")).open("x") as f:
                child=subprocess.Popen(cmd,cwd=T,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                status("running",stage=item["name"]+"-cold4",pid=child.pid)
                if child.wait():raise RuntimeError("cold deployment failed: "+item["name"])
            cmd=[r["python"],"-B","-u",str(P/"probes/audit_sam_structure_mechanical.py"),
                "--run",item["mechanical"],"--vjp",item["vjp"],"--cold",str(cold),"--output",item["admission"],
                "--source",r["commit"],"--mode",item["mode"]]
            with (out/(item["name"]+"-audit.log")).open("x") as f:
                subprocess.run(cmd,cwd=T,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
        # Both are now mechanically concrete; authorization to explore comes
        # from the user, not from claiming these checks prove behavioral gain.
        status("admitted_launching")
        for item in r["variants"]:
            receipt=json.loads(Path(item["long_receipt"]).read_text())
            if sha(receipt["config"])!=receipt["file_sha256"][receipt["config"]]:
                raise ValueError("long config changed")
            log=X/(item["name"]+"-long-watcher.log")
            launch_env=dict(env);launch_env.pop("CUBLAS_WORKSPACE_CONFIG",None)
            cmd=[r["python"],"-B","-u",str(P/"probes/run_sam_structure_long.py"),"--receipt",item["long_receipt"]]
            with log.open("x") as f:
                child=subprocess.Popen(cmd,cwd=T,env=launch_env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            dump(X/(item["name"]+"-long-queue-job.json"),dict(pid=child.pid,command=cmd,receipt=item["long_receipt"],log=str(log)))
        status("complete",scope="both mechanical+cold admissions passed and user-authorized long continuations dispatched")
    except BaseException as e:status("failed",error=repr(e));raise
if __name__=="__main__":main()
