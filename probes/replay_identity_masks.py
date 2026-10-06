"""Admit physical object masks only after exact two-camera recorded RGB replay."""
from pathlib import Path
import argparse,json,hashlib,time,subprocess,sys
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment,_official_task_assets
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 q=argparse.ArgumentParser();q.add_argument("--plan",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
 q.add_argument("--case-index",type=int,default=None,help="Internal child process: one fresh simulator per case")
 a=q.parse_args();a.output.mkdir(exist_ok=False)
 plan=json.loads(a.plan.read_text())
 if a.case_index is None:
  results=[];started=time.time()
  for index,row in enumerate(plan):
   # A reused simulator left contact state that broke exact replay in case 02.
   # A new interpreter/process per case is the reproducibility boundary.
   case=Path(row["case"]);child=a.output/("worker_%02d"%index)
   with (a.output/(case.name+".log")).open("w") as log:
    done=subprocess.run([sys.executable,"-B","-u",str(Path(__file__).resolve()),"--plan",str(a.plan.resolve()),
       "--output",str(child.resolve()),"--case-index",str(index)],stdout=log,stderr=subprocess.STDOUT)
   item={"case":case.name,"returncode":done.returncode,"states":row["steps"]};results.append(item)
   if done.returncode:
    (a.output/"status.json").write_text(json.dumps({"status":"failed","cases":results},indent=2)+"\n")
    raise RuntimeError("isolated mask replay failed: "+case.name)
   (child/case.name).rename(a.output/case.name)
   (a.output/"status.json").write_text(json.dumps({"status":"running","cases":results,"elapsed_seconds":time.time()-started},indent=2)+"\n")
  (a.output/"status.json").write_text(json.dumps({"status":"complete","cases":results,"elapsed_seconds":time.time()-started},indent=2)+"\n")
  return
 plan=[plan[a.case_index]];official,*_=_official_task_assets()
 env=_environment(Path("/data/senwang/data/calvin/raw/task_ABC_D"),show_gui=False)
 results=[];started=time.time()
 try:
  for row in plan:
   case=Path(row["case"]);dest=a.output/case.name;dest.mkdir()
   result=json.loads((case/"result.json").read_text())
   with np.load(case/"trajectory.npz",allow_pickle=False) as z:d={k:z[k] for k in z.files}
   robot,scene=official.get_env_state_for_initial_condition(result["initial_state"])
   env.reset(robot_obs=robot,scene_obs=scene);obs=env.get_obs()
   objects={o.name:int(o.uid) for o in env.scene.movable_objects};names=list(objects);checks=[]
   for t in range(max(row["steps"])+1):
    if t:obs,*_=env.step(d["executed"][t-1].copy())
    if t not in row["steps"]:continue
    images=[d["rgb_static"][t],d["rgb_gripper"][t]]
    actual=[obs["rgb_obs"]["rgb_static"],obs["rgb_obs"]["rgb_gripper"]]
    equality=[bool(np.array_equal(x,y)) for x,y in zip(images,actual)]
    if not all(equality):raise RuntimeError("RGB replay mismatch "+case.name+" state "+str(t))
    payload={"object_names":np.array(names)}
    for camera,image,key in zip(env.cameras,images,["top","wrist"]):
     view=camera.viewMatrix if hasattr(camera,"viewMatrix") else camera.view_matrix
     projection=camera.projectionMatrix if hasattr(camera,"projectionMatrix") else camera.projection_matrix
     frame=p.getCameraImage(width=camera.width,height=camera.height,viewMatrix=view,projectionMatrix=projection,
        physicsClientId=camera.cid,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
     rendered=np.asarray(frame[2]).reshape(camera.height,camera.width,4)[...,:3]
     if not np.array_equal(rendered,image):raise RuntimeError("mask-render RGB mismatch")
     label=np.asarray(frame[4]).reshape(camera.height,camera.width)
     body=np.where(label<0,-1,label&((1<<24)-1))
     payload[key+"_masks"]=np.stack([body==objects[n] for n in names])
    meta={"case":str(case),"time":t,"object_uids":objects,"top_rgb_sha256":digest(images[0]),
          "wrist_rgb_sha256":digest(images[1]),"admission":"exact trajectory RGB replay in both cameras; simulator body masks"}
    payload["metadata_json"]=np.array(json.dumps(meta,sort_keys=True))
    np.savez_compressed(dest/("state_%03d.npz"%t),**payload)
    checks.append({"step":t,"exact_rgb_match":equality,"robot_max_abs_error":float(np.max(np.abs(obs["robot_obs"]-d["robot_obs"][t])))})
   (dest/"summary.json").write_text(json.dumps(checks,indent=2)+"\n")
   results.append({"case":case.name,"complete":True,"states":row["steps"]})
   (a.output/"status.json").write_text(json.dumps({"status":"running","cases":results,"elapsed_seconds":time.time()-started},indent=2)+"\n")
   print(case.name,"accepted",len(checks),flush=True)
 finally:env.close()
 (a.output/"status.json").write_text(json.dumps({"status":"complete","cases":results,"elapsed_seconds":time.time()-started},indent=2)+"\n")
if __name__=="__main__":main()
