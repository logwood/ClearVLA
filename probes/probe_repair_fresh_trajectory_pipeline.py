"""Audit the complete fresh repair rollout, admitting masks by exact RGB replay."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--receipt",type=Path,required=True);a=ap.parse_args()
    r=json.loads(a.receipt.read_text());out=Path(r["output"]);out.mkdir(exist_ok=False)
    def status(state,**items):
        value={"state":state,"unix_time":time.time(),**items}
        tmp=out/"status.tmp";tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(out/"status.json")
        print(json.dumps(value),flush=True)
    try:
        root=Path(r["rollout"]);deadline=time.time()+4*3600
        status("waiting_standard_18")
        while not (root/"summary.json").exists():
            if time.time()>deadline:raise TimeoutError("standard 18-case panel did not finish")
            time.sleep(30)
        panel=json.loads((root/"summary.json").read_text())
        if panel.get("execute_rows")!=8 or panel.get("max_steps")!=360 or panel.get("controller_anchor")!="stored_target":
            raise RuntimeError("fresh audit requires the declared standard controller/8-row/360-step setup")
        if not panel["complete"] or panel["trials"]!=18 or panel["errors"]:
            raise RuntimeError("standard panel incomplete; retain failed cases for recovery")
        if panel["source_commit"]!=r["train_commit"] or panel["checkpoint"]!=r["checkpoint"]:
            raise RuntimeError("fresh rollout identity differs")
        health=json.loads((root/"bridge_health.json").read_text())["deployment"]["checkpoint"]
        expected_step=r.get("global_step",11524)
        if not isinstance(expected_step,int) or expected_step<1:
            raise ValueError("receipt global_step must be an explicit positive completed-update count")
        if health["global_step"]!=expected_step or health["git_commit"]!=r["train_commit"]:
            raise RuntimeError("bridge did not serve the declared repair checkpoint")
        if "checkpoint_sha256" in r and health["sha256"]!=r["checkpoint_sha256"]:
            raise RuntimeError("served weights differ from the admitted checkpoint hash")
        for path,sha in r["script_sha256"].items():
            if digest(path)!=sha:raise RuntimeError("probe changed: "+path)
        source=Path(r["production_source"])
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()!=r["train_commit"]:
            raise RuntimeError("pinned source moved")
        if subprocess.check_output(["git","status","--porcelain","--","clearvla"],cwd=source,text=True).strip():
            raise RuntimeError("pinned production source dirty")
        status("waiting_gpu",gpu=r["gpu"],successes=panel["successes"])
        while True:
            memory=int(subprocess.check_output(["nvidia-smi","-i",str(r["gpu"]),
              "--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
            if memory<512:break
            if time.time()>deadline:raise TimeoutError("diagnostic GPU did not become free")
            time.sleep(30)
        env=os.environ.copy();env.update(PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=str(r["gpu"]),
          EGL_VISIBLE_DEVICES=str(r["gpu"]),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
          OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
        for stage in r["stages"]:
            status("running",stage=stage["name"],successes=panel["successes"])
            with (out/(stage["name"]+".log")).open("w") as log:
                subprocess.run(stage["command"],cwd=source,env=env,stdin=subprocess.DEVNULL,
                   stdout=log,stderr=subprocess.STDOUT,check=True)
            if stage["name"]=="phase-plan":
                plan=json.loads((out/"phase-plan/probe_plan.json").read_text())
                if len(plan)!=18:raise RuntimeError("phase plan is not all 18 cases")
            else:
                result=json.loads((Path(stage["output"])/"status.json").read_text())
                if result["status"]!="complete":raise RuntimeError("stage incomplete: "+stage["name"])
        identity=json.loads((out/"physical-identity/identity.json").read_text())
        if identity["checkpoint_sha256"]!=health["sha256"]:
            raise RuntimeError("identity probe checkpoint differs from fresh rollout")
        status("complete",successes=panel["successes"],checkpoint_sha256=health["sha256"],
               scope="fresh 18-case rollout; exact two-camera mask replay; matched replan RNG")
    except BaseException as error:
        status("failed",error=repr(error));raise
if __name__=="__main__":main()
