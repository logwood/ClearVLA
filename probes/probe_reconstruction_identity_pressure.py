"""Audit whether the existing reconstruction rewards physically distinct K owners.

Actual production tensors are captured first. All alternatives/gradients are
detached diagnostic copies, never policy inputs or parameter updates. Physical
masks only partition reported errors; they never define an objective or owner.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from probes.probe_source_delta_consumer import observe_return

def cpu(x): return x.detach().float().cpu().numpy().copy()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def audit(source, masks, names):
    owner=source["reconstruction_owner"].detach().float().clone()
    content=source["content"].detach().float().clone()
    position=source["decoded_position"].detach().float().clone()
    target=source["target_content"].detach().float().clone()
    observed=source["observed"].detach().float().clone()[...,0]
    support=owner.sum(1)
    prediction=torch.einsum("bkcyx,bkd->bcyxd",owner,content)+support[...,None]*position
    per_owner=content[:,:,None,None,None,:]+position[:,None]
    per_owner_error=(per_owner-target[:,None]).square().mean(-1)
    # Best existing K value at each observed destination, holding its source
    # content and shared position decoder fixed. No true-object mask is used.
    best=per_owner_error.argmin(1)
    error=(prediction-target).square().mean(-1)
    legal=support>0
    logits=torch.where(owner>0,owner.clamp_min(1e-30).log(),-1e9).requires_grad_()
    law=logits.softmax(1)*support[:,None]
    local_prediction=torch.einsum("bkcyx,bkd->bcyxd",law,content)+support[...,None]*position
    loss=((local_prediction-target).square().mean(-1)*observed).sum()/observed.sum().clamp_min(1)
    gradient=torch.autograd.grad(loss,logits)[0]
    grad_cell=gradient.square().sum(1).sqrt()
    uniform=support[:,None].expand_as(owner)/owner.shape[1]
    uniform_prediction=torch.einsum("bkcyx,bkd->bcyxd",uniform,content)+support[...,None]*position
    all_k1=support[...,None]*(content[:,0,None,None,None,:]+position)
    camera_content=source["camera_content"].detach().float().clone()
    residual=source["content_residual"].detach().float().clone()
    camera_values=camera_content+residual[:,:,None]
    per_camera_prediction=torch.einsum("bkcyx,bkcd->bcyxd",owner,camera_values)+support[...,None]*position
    no_residual_prediction=torch.einsum("bkcyx,bkcd->bcyxd",owner,camera_content)+support[...,None]*position
    # Analytical per-image constants are an optimistic no-identity control,
    # not a trained decoder or an deployable value path.
    centered_target=target-position
    camera_constant=(centered_target*observed[...,None]).sum((2,3))/observed.sum((2,3)).clamp_min(1)[...,None]
    camera_constant_prediction=camera_constant[:,:,None,None]+position
    global_constant=(centered_target*observed[...,None]).sum((1,2,3))/observed.sum((1,2,3)).clamp_min(1)[:,None]
    global_constant_prediction=global_constant[:,None,None,None]+position
    alternatives={"production_fp32":error,"uniform_owner":(uniform_prediction-target).square().mean(-1),
      "all_K1_owner":(all_k1-target).square().mean(-1),
      "best_existing_K_per_cell":per_owner_error.min(1).values,
      "existing_per_camera_values_plus_slot_residual":(per_camera_prediction-target).square().mean(-1),
      "existing_per_camera_values":(no_residual_prediction-target).square().mean(-1),
      "fit_one_value_per_camera_no_identity":(camera_constant_prediction-target).square().mean(-1),
      "fit_one_global_value_no_identity":(global_constant_prediction-target).square().mean(-1)}
    region=[]
    for camera,raw_masks in enumerate(masks):
        weights=F.adaptive_avg_pool2d(torch.as_tensor(raw_masks,device=owner.device,dtype=torch.float32)[None],owner.shape[-2:])[0]
        weights=torch.cat((weights,(1-weights.sum(0)).clamp_min(0)[None]),0)
        for j,name in enumerate(names+["background"]):
            w=weights[j]*observed[0,camera];den=w.sum().clamp_min(1e-12)
            region.append({"camera":["top","wrist"][camera],"object":name,"area_cells":float(w.sum()),
              "area_fraction":float(w.sum()/observed.sum()),
              "mse":{k:float((v[0,camera]*w).sum()/den) for k,v in alternatives.items()},
              "production_loss_fraction":float((error[0,camera]*w).sum()/(error*observed).sum().clamp_min(1e-20)),
              "ownership_gradient_l1_fraction":float((grad_cell[0,camera]*w).sum()/(grad_cell*observed).sum().clamp_min(1e-20)),
              "mean_reconstruction_owner":cpu((owner[0,:,camera]*w).sum((-1,-2))/den).tolist(),
              "best_existing_K_histogram":cpu(torch.stack([((best[0,camera]==k)*w).sum()/den for k in range(owner.shape[1])])).tolist(),
              "per_K_value_mse":cpu((per_owner_error[0,:,camera]*w).sum((-1,-2))/den).tolist()})
    return {"source_reconstruction_error":float(source["reconstruction_error"]),
       "source_vs_fp32_reconstruction_rms":float((source["reconstructed"].float()-prediction).square().mean().sqrt()),
       "global_mse":{k:float((v*observed).sum()/observed.sum().clamp_min(1)) for k,v in alternatives.items()},
       "gradient_contract":"analytic detached destination-owner logit gradient; no network VJP or parameter update",
       "owner_logit_grad_l2":float(gradient.norm()),"regions":region}

def main():
    ap=argparse.ArgumentParser()
    for name in ["checkpoint","plan","masks","output"]:ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device("cuda:0"),
      t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
      dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"),seed=0)
    model=policy.bundle.model;grounder=model.grounding.grounder;old=grounder.forward;sources=[]
    def capture(*args,**kw):
        out,local=observe_return(old,args,kw)
        source={k:local[k] for k in ["reconstruction_owner","content","decoded_position","target_content","observed","reconstructed","reconstruction_error"]}
        source["camera_content"]=out[0].camera_content
        source["content_residual"]=local["content"]-local["aggregated_content"]
        if source["camera_content"] is None:raise ValueError("requires declared per-camera G values")
        sources.append(source)
        return out
    grounder.forward=capture;rows=[]
    try:
      plan=json.loads(a.plan.read_text())
      for cid,state in [(1,24),(5,24),(11,24),(17,136)]:
        item=next(r for r in plan if r["case_id"]==cid);case=Path(item["case"])
        with np.load(case/"trajectory.npz",allow_pickle=False) as z:
            data={k:z[k] for k in ["rgb_static","rgb_gripper","robot_obs","executed","raw_chunks"]}
        policy.reset();history=CausalHistory(executed_world=True)
        for t in range(state+1):
            command=np.zeros(7,np.float32) if t==0 else data["executed"][t-1]
            observation=calvin_policy_observation({"rgb_obs":{k:data[k][t] for k in ["rgb_static","rgb_gripper"]},"robot_obs":data["robot_obs"][t]},command)
            if t==0:history.reset(observation,reset_action=command)
            else:history.append(command,observation)
            if t not in (0,state):
                if t%8==0:model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                continue
            sources.clear();action,_=policy.act_with_input(history.snapshot(),item["instruction"])
            if t==state:
                with np.load(a.masks/case.name/f"state_{t:03d}.npz",allow_pickle=False) as z:
                    masks=[z[k+"_masks"].copy() for k in ["top","wrist"]];names=z["object_names"].tolist()
                with torch.inference_mode(False),torch.enable_grad():
                    result=audit(sources[0],masks,names)
                rows.append({"case_id":cid,"state":t,"instruction":item["instruction"],
                  "recorded_arm_rmse":float(np.sqrt(np.mean((action[:8,:6]-data["raw_chunks"][t//8,:8,:6])**2))),**result})
                dump(a.output/"summary.json",rows);print(cid,t,"complete",flush=True)
      dump(a.output/"complete.json",{"checkpoint_sha256":policy.bundle.checkpoint_sha256,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"masks are report-only, alternatives use existing predicted K values and raw target, no policy or loss change, no optimizer updates",
        "windows":len(rows)})
    finally:grounder.forward=old
if __name__=="__main__":main()
