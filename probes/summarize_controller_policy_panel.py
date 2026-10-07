"""Audit complete or partial CALVIN controller-anchor diagnostic panels.

Environment object/contact truth is used only here for evaluation, never as a
policy input. Each panel retains its own changed/unchanged controller contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def summarize_case(case):
    result=json.loads((case/"result.json").read_text())
    info=json.loads((case/"environment_info.json").read_text())
    with np.load(case/"trajectory.npz") as z:
        state=z["robot_obs"].copy()
        scene=z["scene_obs"].copy()
        action=z["executed"].copy()
        raw=z["raw_executed"].copy()
        chunks=z["executed_chunk_row"].copy()
        goals=z["controller_applied_goal"].copy() if "controller_applied_goal" in z else None
    n=len(action)
    assert state.shape[0]==scene.shape[0]==len(info["info"])==n+1
    if goals is None:
        goals=np.asarray([row["target_pos"]+row["target_orn"] for row in info["controller_targets"][1:]])
    assert goals.shape==(n,6)
    color=result["task"].split("_")[1]
    idx={"red":6,"blue":12,"pink":18}[color]
    uid={"red":2,"blue":3,"pink":4}[color]
    sign=-1 if result["task"].endswith("left") else 1
    progress=sign*(scene[:,idx]-scene[0,idx])
    delta=state[1:,:3]-state[:-1,:3]
    requested=(action[:,:3].astype(np.float32)*np.float32(.02)).astype(float)
    anchor=result.get("controller_anchor","stored_target")
    previous=np.vstack((state[0,:6],goals[:-1]))
    selected=previous[:,:3].copy()
    if anchor=="measured_tcp": selected=state[:-1,:3].copy()
    elif anchor=="replan_tcp": selected[chunks==0]=state[:-1,:3][chunks==0]
    recurrence=goals[:,:3]-selected-requested
    error=goals[:,:3]-state[1:,:3]
    contacts=[];force=[];target_frames=[];free=[]
    for t,row in enumerate(info["info"]):
        cs=row["robot_info"].get("contacts",[])
        others=[int(c[2] if int(c[1])==0 else c[1]) for c in cs]
        objects=sorted(set(others)&{2,3,4})
        if objects: contacts.append({"state":t,"objects":objects})
        if uid in objects: target_frames.append(t)
        force.append(sum(float(c[9]) for c in cs))
        if t<n: free.append(not cs and state[t,2]>.51)
    free=np.asarray(free,dtype=bool)
    denom=np.square(requested[free]).sum()
    gain=float((requested[free]*delta[free]).sum()/denom) if denom>0 else None
    phases=[]
    for lo,hi in [(0,24),(24,64),(64,128),(128,160),(320,360)]:
        hi=min(hi,n)
        if lo>=hi:continue
        phases.append({"steps":[lo,hi],"tcp_net_mm":(1000*(state[hi,:3]-state[lo,:3])).tolist(),
          "requested_sum_mm":(1000*requested[lo:hi].sum(0)).tolist(),
          "target_progress_delta_mm":float(1000*(progress[hi]-progress[lo])),
          "target_contact_observations":sum(lo<=t<=hi for t in target_frames),
          "open_command_fraction":float((action[lo:hi,6]>0).mean()),
          "max_goal_tcp_error_mm":float(1000*np.linalg.norm(error[lo:hi],axis=-1).max())})
    return {"case_id":int(case.name[5:7]),"task":result["task"],"success":bool(result["success"]),
      "steps":n,"anchor":anchor,"initial_rgb_sha256":initial_hash(case),
      "target_progress_final_mm":float(progress[-1]*1000),"target_progress_max_mm":float(progress.max()*1000),
      "first_object_contact":contacts[0] if contacts else None,
      "first_target_contact":target_frames[0] if target_frames else None,
      "max_goal_tcp_error_mm":float(1000*np.linalg.norm(error,axis=-1).max()),
      "goal_tcp_error_median_mm":float(1000*np.median(np.linalg.norm(error,axis=-1))),
      "controller_recurrence_max_abs_m":float(np.abs(recurrence).max()),
      "max_robot_contact_force_N":float(max(force)),
      "free_motion_observations":int(free.sum()),"free_motion_request_to_actual_ls_gain":gain,
      "arm_clip_fraction":float((np.abs(raw[:,:6])>1).mean()),
      "gripper_switches":int(np.count_nonzero(action[1:,6]!=action[:-1,6])),"phases":phases}

def initial_hash(case):
    with np.load(case/"trajectory.npz") as z:
        return {key:hashlib.sha256(z[key][0].tobytes()).hexdigest() for key in ["rgb_static","rgb_gripper"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",type=Path,action="append",required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();rows=[]
    for p in a.panel:
        cases=[summarize_case(c) for c in sorted(p.glob("case_*")) if c.is_dir() and (c/"result.json").exists()]
        rows.append({"path":str(p),"complete":len(cases)==18,"cases":cases,
          "successes":sum(c["success"] for c in cases),"finished_cases":len(cases)})
    if rows:
        initial={c["case_id"]:c["initial_rgb_sha256"] for c in rows[0]["cases"]}
        for panel in rows[1:]:
            for c in panel["cases"]:c["initial_rgb_matches_baseline"]=initial.get(c["case_id"])==c["initial_rgb_sha256"]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"panels":rows,"scope":"controller contracts are distinct; no pooled score"},indent=2)+"\n")
    for row in rows:
        print(row["path"],row["successes"],row["finished_cases"],[(c["case_id"],c["success"],round(c["target_progress_max_mm"],1),round(c["max_goal_tcp_error_mm"],1)) for c in row["cases"]])
if __name__=="__main__":main()
