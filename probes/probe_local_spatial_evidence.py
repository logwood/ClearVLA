"""Trace real-object coverage BEFORE global K; coherent source-only interventions.

All masks are read-only audit truth. Interventions remove one producer score
term and rerun its actual value/materialization path. No oracle enters policy.
"""
from pathlib import Path
from dataclasses import replace
import argparse, json, importlib.util, hashlib, math, time
import numpy as np
import torch
import torch.nn.functional as F
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action
import clearvla.mainline.v120_core.flow_dino_evidence as flow
from probes.probe_source_delta_consumer import observe_return

def arr(x): return x.detach().float().cpu().numpy().copy()
def rms(x): return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))
def dump(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
def load_coverage(path):
    spec=importlib.util.spec_from_file_location("mask_coverage",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.object_coverage

def mask_measure(prob, xy, masks):
    """Report equal-query conditional coverage; do not call it global K mass."""
    prob=prob.detach().float()
    xy=xy.detach().float()
    if xy.ndim==4:
        xy=xy[:,:,None,None,None].expand(*prob.shape,2)
    rows=[]
    for c,mask in enumerate(masks):
        m=torch.as_tensor(mask,device=xy.device,dtype=torch.float32)[None]
        grid=xy[0,c].reshape(1,-1,1,2)
        cell=F.grid_sample(m,grid,align_corners=True,mode="bilinear",padding_mode="border")[0,:,:,0]
        cell=cell.reshape(mask.shape[0],*prob[0,c].shape)
        radius=math.ceil(mask.shape[-1]/32)
        dilated=F.max_pool2d(m,2*radius+1,stride=1,padding=radius)
        expanded=F.grid_sample(dilated,grid,align_corners=True,mode="bilinear",padding_mode="border")[0,:,:,0].reshape_as(cell)
        local=(cell*prob[0,c][None]).sum(-1).flatten(1)
        wide=(expanded*prob[0,c][None]).sum(-1).flatten(1)
        entropy=-(prob[0,c]*prob[0,c].clamp_min(1e-30).log()).sum(-1)
        rows.append({"camera":("top","wrist")[c],"equal_query_object_mass":arr(local.mean(-1)).tolist(),
          "max_query_object_mass":arr(local.max(-1).values).tolist(),
          "expanded_mask_equal_query_mass":arr(wide.mean(-1)).tolist(),
          "mask_expansion_pixels":radius,"effective_points_mean":float(entropy.exp().mean()),
          "max_probability_mean":float(prob[0,c].max(-1).values.mean())})
    return rows

def main():
    ap=argparse.ArgumentParser()
    for name in ["checkpoint","plan","masks","coverage-script","output"]:ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    coverage=load_coverage(args.coverage_script)
    policy=ClearVLACheckpointPolicy(args.checkpoint,device=torch.device("cuda:0"),
      t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
      dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"),seed=0)
    model=policy.bundle.model;grounder=model.grounding.grounder;enc=model.observation.compiler.encoder
    original_update=enc.update_progressive_grounding_address;original_law=flow.couple_local_observation
    original_g=grounder.forward;original_encode=model.encode_online
    import clearvla.simulation.clearvla_policy as frontend
    original_sample=frontend.sample_action
    captured={};gs=[];caches=[];samples=[];variant="baseline";group=0
    def update(state,rollout,**kw):
        nonlocal group
        if kw["stage"]==1:group+=1
        out=original_update(state,rollout,**kw)
        if group==1 and kw["stage"]==1:
            captured["coarse_base"]=state.bank.coarse_base_logits
            captured["coarse_logits"]=out.coarse_logits
            captured["coarse_probability"]=out.coarse_probability
            captured["coarse_xy"]=state.bank.coarse_candidate_coordinates
        return out
    def law(typed,parent,valid):
        active=variant if group==1 else "baseline"
        changed=dict(typed);prior=parent
        if active=="neutral_parent":prior=torch.log_softmax(torch.zeros_like(parent).masked_fill(~valid,-1e9),-1)
        if active=="parent_half":prior=torch.log_softmax(.5*parent.masked_fill(~valid,-1e9),-1)
        if active.startswith("neutral_") and active[8:] in changed:changed[active[8:]]=torch.zeros_like(changed[active[8:]])
        if active=="parent_only":changed={k:torch.zeros_like(v) for k,v in changed.items()}
        out=original_law(changed,prior,valid)
        if group==1:captured.update(typed=dict(typed),original_parent=parent,used_parent=prior,law=out,valid=valid)
        return out
    def gh(*a,**kw):
        out,local=observe_return(original_g,a,kw)
        gs.append((out[0],{k:local[k] for k in ["chart","camera_read","structured_read","candidate_prior","validity","candidate_shape"]}))
        return out
    def encode(*a,**kw):
        out=original_encode(*a,**kw);caches[:]=[out[0]];return out
    def sample(*a,**kw):
        out=original_sample(*a,**kw);samples[:]=[out];return out
    def clear():
        nonlocal group
        group=0;captured.clear();gs.clear();caches.clear();samples.clear()
    def native(sample):
        x=policy.bundle.action_normalizer.decode(arr(sample.action)[0]);x[:,-1]=arr(sample.gripper_command)[0];return x
    enc.update_progressive_grounding_address=update;flow.couple_local_observation=law
    grounder.forward=gh;model.encode_online=encode;frontend.sample_action=sample
    rows=[];plan=json.loads(args.plan.read_text())
    try:
        for cid,steps in [(1,[24]),(5,[24]),(11,[24]),(17,[136])]:
            row=next(r for r in plan if r["case_id"]==cid);case=Path(row["case"])
            with np.load(case/"trajectory.npz") as z:data={k:z[k] for k in z.files}
            policy.reset();history=CausalHistory(executed_world=True);variant="baseline"
            for t in range(max(steps)+1):
                prev=np.zeros(7,np.float32) if t==0 else data["executed"][t-1]
                obs=calvin_policy_observation({"rgb_obs":{k:data[k][t] for k in ["rgb_static","rgb_gripper"]},"robot_obs":data["robot_obs"][t]},prev)
                if t==0:history.reset(obs,reset_action=prev)
                else:history.append(prev,obs)
                if t not in steps:
                    if t==0:
                        clear();policy.act_with_input(history.snapshot(),row["instruction"])
                    elif t%8==0:model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    continue
                clear();baseline,online=policy.act_with_input(history.snapshot(),row["instruction"]);seed=samples[0].initial_physical_noise
                maskpath=args.masks/case.name/f"state_{t:03d}.npz"
                with np.load(maskpath) as z:masks=[z[k+"_masks"].copy() for k in ["top","wrist"]];names=z["object_names"].tolist()
                entry={"case_id":cid,"state":t,"object_names":names,"recorded_arm_rmse":rms(baseline[:8,:6]-data["raw_chunks"][t//8,:8,:6]),"variants":[]}
                for variant in ["baseline","neutral_parent","parent_half","parent_only","neutral_semantic","neutral_appearance","neutral_geometry"]:
                    clear()
                    with torch.no_grad():out=sample_action(model,online,policy.bundle.config,initial_physical_noise=seed)
                    cache=caches[0];match=[(i,f,s) for i,(f,s) in enumerate(gs) if f.camera_coordinates is cache.top.belief.camera_coordinates]
                    assert len(match)==1 and match[0][0]==0
                    _,facts,source=match[0];spatial=source["chart"].current_image_support
                    cover=coverage({"spatial":spatial,"camera_read":source["camera_read"]},
                        [data["rgb_static"][t],data["rgb_gripper"][t]],maskpath,arr(facts.camera_coordinates)[0])
                    stages={}
                    if variant=="baseline":
                        xy=captured["coarse_xy"]
                        stages["uniform_coarse"]=mask_measure(torch.full_like(captured["coarse_probability"],1/captured["coarse_probability"].shape[-1]),xy,masks)
                        stages["pre_G1"]=mask_measure(captured["coarse_base"].float().softmax(-1),xy,masks)
                        stages["G1"]=mask_measure(captured["coarse_probability"],xy,masks)
                        stages["G2_parent"]=mask_measure(captured["original_parent"].float().softmax(-1),spatial.coordinates,masks)
                        typed=sum(captured["typed"].values())/math.sqrt(3)
                        stages["G2_typed_only"]=mask_measure(typed.float().softmax(-1),spatial.coordinates,masks)
                    stages["G2_output"]=mask_measure(spatial.probability,spatial.coordinates,masks)
                    act=native(out)
                    record={"variant":variant,"stages":stages,"G_coverage":cover,
                       "global_K_camera_mass":arr(source["structured_read"].flatten(3).sum(-1))[0].tolist(),
                       "binding":arr(cache.top.intent.target_binding.mass)[0].tolist(),
                       "null":float(cache.top.intent.target_binding.null_mass[0]),
                       "native_first8":act[:8].tolist(),"native_vs_baseline_arm_rmse":rms(act[:8,:6]-baseline[:8,:6]),
                       "g_reconstruction_error":float(facts.reconstruction_error)}
                    if variant=="baseline":
                        record["within_candidate_score_rms"]={k:rms(arr(v-v.mean(-1,keepdim=True))) for k,v in {**captured["typed"],"parent":captured["original_parent"],"G1_base":captured["coarse_base"],"G1_delta":captured["coarse_logits"]-captured["coarse_base"]}.items()}
                    if variant in ["baseline","neutral_parent","parent_half"]:
                        record["natural_colors"]=[]
                        for color in ["red","blue","pink"]:
                            instruction="go push the "+color+" block "+row["instruction"].split()[-1]
                            tokens,mask=policy._goal(instruction)
                            other=replace(online,goal=replace(online.goal,tokens=tokens.to(policy.device),mask=mask.to(policy.device)))
                            clear()
                            with torch.no_grad():v=sample_action(model,other,policy.bundle.config,initial_physical_noise=seed)
                            record["natural_colors"].append({"color":color,"binding":arr(caches[0].top.intent.target_binding.mass)[0].tolist(),
                                "native_first8":native(v)[:8].tolist()})
                    entry["variants"].append(record)
                    dump(args.output/"results.json",rows+[entry])
                    print(json.dumps({"case":cid,"state":t,"variant":variant,"arm_delta":record["native_vs_baseline_arm_rmse"]}),flush=True)
                rows.append(entry);variant="baseline"
        dump(args.output/"complete.json",{"complete":True,"windows":len(rows),"checkpoint_sha256":policy.bundle.checkpoint_sha256,
          "scope":"current online local producer only; coherent recomputation; masks never enter model; no closed-loop efficacy claim",
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    finally:
        enc.update_progressive_grounding_address=original_update;flow.couple_local_observation=original_law
        grounder.forward=original_g;model.encode_online=original_encode;frontend.sample_action=original_sample
if __name__=="__main__":main()
