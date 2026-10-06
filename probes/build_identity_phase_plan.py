"""Build physical identity, phase and targeted-probe ledgers from original rollouts."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def spans(a):
 d=np.diff(np.r_[False,a,False].astype(int))
 return [[int(x),int(y-1)] for x,y in zip(np.where(d==1)[0],np.where(d==-1)[0])]

def main():
 q=argparse.ArgumentParser();q.add_argument("--root",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
 a=q.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 rows=[];plan=[]
 extra={1:[160,168,176,184,192],2:[80,88,96,208,216,224,320],3:[320,328,336,344],9:[64,72,256,264],
        14:[320,328,336,344,352],15:[320,328,336,344],17:[112,120,128,136,144,160]}
 for case in sorted(p for p in a.root.glob("case_*") if p.is_dir()):
  r=json.loads((case/"result.json").read_text());tele=json.loads((case/"environment_info.json").read_text())
  infos=tele["info"]
  with np.load(case/"trajectory.npz",allow_pickle=False) as z:d={k:z[k] for k in z.files}
  t=len(d["executed"]);idx=int(case.name.split("_")[1]);assert len(infos)==t+1
  names=list(infos[0]["scene_info"]["movable_objects"]);target="block_"+r["task"].split("_")[1];sign=-1 if r["task"].endswith("left") else 1
  positions={n:np.asarray([f["scene_info"]["movable_objects"][n]["current_pos"] for f in infos]) for n in names}
  object_rows={};touch={}
  for n in names:
   uids={f["scene_info"]["movable_objects"][n]["uid"] for f in infos};assert len(uids)==1
   matches=[j for j in range(22) if np.max(np.abs(d["scene_obs"][:,j:j+3]-positions[n]))<2e-7]
   assert len(matches)==1,(n,matches)
   touch[n]=np.array([any(c[9]>0 and f["robot_info"]["uid"] in (c[1],c[2]) for c in f["scene_info"]["movable_objects"][n]["contacts"]) for f in infos])
   object_rows[n]={"uid":list(uids)[0],"scene_obs_xyz_start":matches[0],"initial_position":positions[n][0].tolist(),
    "final_position":positions[n][-1].tolist(),"robot_contact_spans":spans(touch[n])}
  tcp=d["robot_obs"][:,:3];ctrl=np.array([v["target_pos"] for v in tele["controller_targets"]]);gap=np.linalg.norm(ctrl-tcp,axis=1)
  progress=sign*(positions[target][:,0]-positions[target][0,0])
  last=(t-1)//8*8
  wanted={0,24,min(40,last),last,*extra.get(idx,[])}
  if touch[target].any():
   first=int(np.flatnonzero(touch[target])[0])//8*8
   wanted.update((max(0,first-8),first,min(first+8,last)))
  peak=int(np.argmax(progress))//8*8;wanted.add(min(peak,last))
  wanted=sorted(x for x in wanted if x<=last and x%8==0)
  events=[]
  for s in np.arange(0,t,8):
   e=min(s+8,t);action=d["executed"][s:e];touching=[n for n in names if touch[n][s+1:e+1].any()]
   events.append({"state":int(s),"tcp":tcp[s].tolist(),"target_position":positions[target][s].tolist(),
     "remaining_dx_to_0_1m":float(.1-progress[s]),"signed_target_dx":float(progress[s]),"next8_signed_target_dx":float(progress[e]-progress[s]),
     "next8_contact_objects":touching,"native_xyz_mean":action[:,:3].mean(0).tolist(),"native_rotation_mean":action[:,3:6].mean(0).tolist(),
     "gripper":action[:,6].tolist(),"tcp_delta":(tcp[e]-tcp[s]).tolist(),"controller_target_delta":(ctrl[e]-ctrl[s]).tolist(),
     "controller_gap":float(gap[s]),"max_gap_next8":float(gap[s:e+1].max())})
  firsts={n:next((i for i,x in enumerate(touch[n]) if x),None) for n in names}
  row={"case":case.name,"case_id":idx,"target":target,"direction":sign,"success":r["success"],"steps":t,"instruction":r["instruction"],
      "initial_state":r["initial_state"],"objects":object_rows,"first_contacts":firsts,
      "target_signed_dx_max":float(progress.max()),"target_signed_dx_final":float(progress[-1]),
      "controller_gap_max":float(gap.max()),"controller_gap_max_at":int(gap.argmax()),
      "initial_target_xy_distance":float(np.linalg.norm(tcp[0,:2]-positions[target][0,:2])),
      "probe_steps":wanted,"replans":events}
  rows.append(row);plan.append({"case":str(case),"case_id":idx,"steps":wanted,"success":r["success"],"target":target,"instruction":r["instruction"]})
 (a.output/"physical_identity_and_phases.json").write_text(json.dumps(rows,indent=2)+"\n")
 (a.output/"probe_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 print(json.dumps({"cases":len(rows),"states":sum(len(x["steps"]) for x in plan),"failures":[r["case_id"] for r in rows if not r["success"]]}))
 for r in rows:
  if not r["success"]:print(r["case_id"],r["target"],"first",r["first_contacts"],"max_dx",r["target_signed_dx_max"],"probe",r["probe_steps"])

if __name__=="__main__":main()
