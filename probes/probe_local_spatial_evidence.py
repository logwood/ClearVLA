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
SCRIPT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    ap.add_argument("--bank-source", action="store_true")
    ap.add_argument("--frozen-bank", action="store_true")
    ap.add_argument("--language-chain", action="store_true")
    ap.add_argument("--first-window-only", action="store_true")
    ap.add_argument("--reconstruction-chain", action="store_true")
    ap.add_argument("--native-value", action="store_true")
    args=ap.parse_args();args.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    coverage=load_coverage(args.coverage_script)
    policy=ClearVLACheckpointPolicy(args.checkpoint,device=torch.device("cuda:0"),
      t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
      dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"),seed=0)
    model=policy.bundle.model;grounder=model.grounding.grounder;enc=model.observation.compiler.encoder
    original_update=enc.update_progressive_grounding_address;original_law=flow.couple_local_observation
    original_g=grounder.forward;original_encode=model.encode_online
    banks=[m for m in enc.modules() if isinstance(m,flow._SoftMultiResolutionAddressCompiler)]
    assert len(banks)==1
    bank_module=banks[0];original_bank=bank_module.forward;bank_records={}
    import clearvla.simulation.clearvla_policy as frontend
    original_sample=frontend.sample_action
    from clearvla.mainline.model.intent import StatelessObjectIntentOrganizer, _diagnostic_attention_weights
    organizers=[m for m in model.modules() if isinstance(m,StatelessObjectIntentOrganizer)]
    assert len(organizers)==1
    organizer=organizers[0];binder=organizer.shared_binder
    original_bind=binder.forward;language_trace={};language_hooks=[]
    def keep(name):
        def save(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            language_trace[name]=arr(value)
        return save
    def bind(*a,**kw):
        out,local=observe_return(original_bind,a,kw)
        language_trace.update({name:arr(local[name]) for name in ["obj","task_value","read","object_score","task_object_score","score"]})
        return out
    if args.language_chain:
        binder.forward=bind
        for name in ["goal_input","goal_read","goal_self"]:
            language_hooks.append(getattr(organizer,name).register_forward_hook(keep(name)))
    captured={};gs=[];caches=[];samples=[];variant="baseline";group=0
    def bank_forward(*a,**kw):
        hook=None;key_hook=None;floor=bank_module.flow_prior_floor;prior=bank_module.flow_prior_log_scale.detach().clone()
        if group==0 and (args.bank_source or args.frozen_bank or args.native_value):
            if variant in ["bank_content_zero","bank_content_half"]:
                factor=0.0 if variant=="bank_content_zero" else .5
                hook=bank_module.query_norm.register_forward_hook(lambda m,i,o:o*factor)
            elif variant in ["bank_frozen_dino","bank_frozen_and_native"]:
                source=kw["source_dino"].float()
                target=kw["target_dino"].float()
                b,c,h,w,d=source.shape
                source_grid=flow.resize_endpoint_chart(
                    source.permute(0,1,4,2,3).reshape(b*c,d,h,w),
                    (bank_module.grid,bank_module.grid),bank_module.visual_chart_mode
                ).reshape(b,c,d,bank_module.grid,bank_module.grid).permute(0,1,3,4,2)
                # Fixed observed DINO cosine uses the same sqrt(D) metric as
                # the endpoint measurement. This intervention changes only
                # current producer scores; all values are rematerialized.
                scale=bank_module.content_log_scale.float().exp().clamp(1.,16.)
                q=F.normalize(source_grid,dim=-1,eps=1e-6)[:,:,:,:,None].expand(-1,-1,-1,-1,bank_module.slots,-1)
                q=q*(math.sqrt(d*bank_module.route_dim)/scale)
                k=F.normalize(target,dim=-1,eps=1e-6).reshape(b,c,h*w,d)
                hook=bank_module.query_norm.register_forward_hook(lambda m,i,o:q)
                key_hook=bank_module.key_norm.register_forward_hook(lambda m,i,o:k)
            elif variant=="bank_spatial_zero":
                bank_module.flow_prior_floor=0.0
                with torch.no_grad():bank_module.flow_prior_log_scale.fill_(-100)
        try:
            out,local=observe_return(original_bank,a,kw)
            bank_records[id(local["bank"])]={k:local[k] for k in [
                "content_logits","floor_flow_bias","adaptive_flow_bias","sigma",
                "content_scale","confidence_grid","uncertainty_dino","occlusion_grid",
                "flow_center_dino","source_high"]}
            bank_records[id(local["bank"])]["native_current_dino"]=kw["target_dino"]
            return out
        finally:
            if hook is not None:hook.remove()
            if key_hook is not None:key_hook.remove()
            bank_module.flow_prior_floor=floor
            with torch.no_grad():bank_module.flow_prior_log_scale.copy_(prior)
    def update(state,rollout,**kw):
        nonlocal group
        if kw["stage"]==1:group+=1
        old_content=state.bank.dense_current_dino_content
        if group==1 and kw["stage"]==3 and variant in ["native_value_only","bank_frozen_and_native"]:
            with torch.autocast(device_type=rollout.device.type,enabled=False):
                native=bank_records[id(state.bank)]["native_current_dino"]
                state.bank.dense_current_dino_content=enc.teacher_norm(native.float()).to(old_content.dtype)
        try:out=original_update(state,rollout,**kw)
        finally:state.bank.dense_current_dino_content=old_content
        if group==1 and kw["stage"]==1:
            captured["bank_parts"]=bank_records[id(state.bank)]
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
        source={k:local[k] for k in ["chart","camera_read","structured_read","candidate_prior","validity","candidate_shape"]}
        if args.reconstruction_chain:
            owner=local["reconstruction_owner"]
            value=local["aggregated_content"]
            residual=grounder.decode_content_residual(local["slots"])
            source["reconstruction"]={
                "observed":local["observed"],"target":local["target_content"],
                "full":local["reconstructed"],"owner":owner,
                "without_slot_residual":torch.einsum("bkcyx,bkd->bcyxd",owner.to(value.dtype),value)+local["shared_support"].to(local["decoded_position"].dtype)[...,None]*local["decoded_position"],
                "without_position":torch.einsum("bkcyx,bkd->bcyxd",owner.to(local["content"].dtype),local["content"]),
            }
        gs.append((out[0],source))
        return out
    def encode(*a,**kw):
        out=original_encode(*a,**kw);caches[:]=[out[0]];return out
    def sample(*a,**kw):
        out=original_sample(*a,**kw);samples[:]=[out];return out
    def clear():
        nonlocal group
        group=0;captured.clear();gs.clear();caches.clear();samples.clear();bank_records.clear();language_trace.clear()
    def native(sample):
        x=policy.bundle.action_normalizer.decode(arr(sample.action)[0]);x[:,-1]=arr(sample.gripper_command)[0];return x
    bank_module.forward=bank_forward
    enc.update_progressive_grounding_address=update;flow.couple_local_observation=law
    grounder.forward=gh;model.encode_online=encode;frontend.sample_action=sample
    rows=[];plan=json.loads(args.plan.read_text())
    try:
        windows=[(1,[24])] if args.first_window_only else [(1,[24]),(5,[24]),(11,[24]),(17,[136])]
        for cid,steps in windows:
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
                variants=(["baseline","bank_content_zero","bank_content_half","bank_spatial_zero"] if args.bank_source else ["baseline","neutral_parent","parent_half","parent_only","neutral_semantic","neutral_appearance","neutral_geometry"])
                if args.language_chain or args.reconstruction_chain: variants=["baseline"]
                if args.frozen_bank: variants=["baseline","bank_frozen_dino"]
                if args.native_value: variants=["baseline","native_value_only","bank_frozen_and_native"]
                for variant in variants:
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
                        for name in ["content_logits","floor_flow_bias","adaptive_flow_bias"]:
                            stages["bank_"+name]=mask_measure(captured["bank_parts"][name].float().softmax(-1),xy,masks)
                        parts=captured["bank_parts"]
                        stages["bank_spatial_only"]=mask_measure((parts["floor_flow_bias"]+parts["adaptive_flow_bias"]).float().softmax(-1),xy,masks)
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
                    if args.reconstruction_chain:
                        recon=source["reconstruction"];diagnostic=[]
                        for camera,mask in enumerate(masks):
                            # Area occupancy is diagnostic support on DINO cells,
                            # not an oracle mask entering model or its objective.
                            size=recon["full"].shape[2:4]
                            weights=F.adaptive_avg_pool2d(torch.as_tensor(mask,device=policy.device,dtype=torch.float32)[None],size)[0]
                            background=(1-weights.sum(0)).clamp_min(0)
                            weights=torch.cat([weights,background[None]],0)
                            target=recon["target"][0,camera].float()
                            for oi,name in enumerate(names+["background"]):
                                w=weights[oi]*recon["observed"][0,camera,...,0].float()
                                den=w.sum()
                                error={}
                                for key in ["full","without_slot_residual","without_position"]:
                                    cell=(recon[key][0,camera].float()-target).square().mean(-1)
                                    error[key]=float((cell*w).sum()/den.clamp_min(1e-12))
                                owners=(recon["owner"][0,:,camera]*w[None]).sum((1,2))/den.clamp_min(1e-12)
                                diagnostic.append({"camera":["top","wrist"][camera],"object":name,"cell_area":float(den),
                                    "reconstruction_mse":error,"conditional_reconstruction_K":arr(owners).tolist()})
                        record["reconstruction_object_audit"]=diagnostic
                    if variant=="baseline":
                        record["within_candidate_score_rms"]={k:rms(arr(v-v.mean(-1,keepdim=True))) for k,v in {**captured["typed"],"parent":captured["original_parent"],"G1_base":captured["coarse_base"],"G1_delta":captured["coarse_logits"]-captured["coarse_base"]}.items()}
                    parts=captured["bank_parts"]
                    record["bank_parameters"]={k:{"min":float(parts[k].min()),"mean":float(parts[k].mean()),"max":float(parts[k].max())} for k in ["sigma","content_scale","confidence_grid","uncertainty_dino","occlusion_grid"]}
                    record["bank_centered_score_rms"]={k:rms(arr(parts[k]-parts[k].mean(-1,keepdim=True))) for k in ["content_logits","floor_flow_bias","adaptive_flow_bias"]}
                    if not args.reconstruction_chain and variant in ["baseline","neutral_parent","parent_half","bank_content_zero","bank_content_half","bank_spatial_zero","bank_frozen_dino","native_value_only","bank_frozen_and_native"]:
                        record["natural_colors"]=[]
                        natural_conditions=([(color,direction) for color in ["red","blue","pink"] for direction in ["left","right"]]
                            if args.language_chain else [(color,row["instruction"].split()[-1]) for color in ["red","blue","pink"]])
                        traces={}
                        for color,direction in natural_conditions:
                            instruction="go push the "+color+" block "+direction
                            tokens,mask=policy._goal(instruction)
                            other=replace(online,goal=replace(online.goal,tokens=tokens.to(policy.device),mask=mask.to(policy.device)))
                            clear()
                            with torch.no_grad():v=sample_action(model,other,policy.bundle.config,initial_physical_noise=seed)
                            natural={"color":color,"direction":direction,"binding":arr(caches[0].top.intent.target_binding.mass)[0].tolist(),
                                "native_first8":native(v)[:8].tolist()}
                            if args.language_chain:
                                trace={name:value.copy() for name,value in language_trace.items()}
                                valid=arr(mask).astype(bool)
                                trace["t5_mean"]=arr(tokens)[0,valid[0]].mean(0)
                                trace["projected_goal_mean"]=trace["goal_input"][0,valid[0]].mean(0)
                                trace["native_arm"]=np.asarray(natural["native_first8"])[:,:6]
                                trace["binding"]=np.asarray(natural["binding"])
                                natural["valid_tokens"]=int(valid.sum())
                                natural["trace_rms"]={name:rms(value) for name,value in trace.items()}
                                memory=torch.from_numpy(trace["goal_input"]).to(policy.device)
                                with torch.no_grad():
                                    weights=_diagnostic_attention_weights(organizer.goal_read.attention,
                                      organizer.goal_read.query_norm(organizer.goal_queries.to(memory)),
                                      organizer.goal_read.memory_norm(memory),~mask.to(policy.device).bool())
                                natural["goal_read_attention"]=arr(weights)[0,:,:int(valid.sum())].tolist()
                                natural["language_token_variation"]={name:rms(trace[name]-trace[name].mean(1,keepdims=True))
                                    for name in ["goal_read","goal_self","task_value"]}
                                natural["t5_per_token_rms"]=np.sqrt((arr(tokens)[0,valid[0]].astype(float)**2).mean(-1)).tolist()
                                trace["t5_valid_tokens"]=arr(tokens)[0,valid[0]]
                                natural["scores"]={name:trace[name].tolist() for name in ["object_score","task_object_score","score"]}
                                traces[(color,direction)]=trace
                            record["natural_colors"].append(natural)
                        if args.language_chain:
                            pairs=[]
                            conditions=list(traces)
                            for ia,key_a in enumerate(conditions):
                                for key_b in conditions[ia+1:]:
                                    if key_a[0]!=key_b[0] and key_a[1]!=key_b[1]:continue
                                    ta,tb=traces[key_a],traces[key_b]
                                    keys=set(ta)&set(tb)
                                    pairs.append({"a":key_a,"b":key_b,"kind":"color" if key_a[0]!=key_b[0] else "direction",
                                      "absolute_rms_delta":{name:rms(ta[name]-tb[name]) for name in keys if ta[name].shape==tb[name].shape},
                                      "K_log_odds_rms_delta":rms((ta["score"]-ta["score"].mean(-1,keepdims=True))-(tb["score"]-tb["score"].mean(-1,keepdims=True))),
                                      "per_token_t5_delta":np.sqrt(((ta["t5_valid_tokens"]-tb["t5_valid_tokens"]).astype(float)**2).mean(-1)).tolist() if ta["t5_valid_tokens"].shape==tb["t5_valid_tokens"].shape else None})
                            record["language_chain_pairs"]=pairs
                            record["binder_weight_rms"]={name:rms(arr(value)) for name,value in binder.named_parameters()}

                    entry["variants"].append(record)
                    dump(args.output/"results.json",rows+[entry])
                    print(json.dumps({"case":cid,"state":t,"variant":variant,"arm_delta":record["native_vs_baseline_arm_rmse"]}),flush=True)
                rows.append(entry);variant="baseline"
        dump(args.output/"complete.json",{"complete":True,"windows":len(rows),"checkpoint_sha256":policy.bundle.checkpoint_sha256,
          "scope":"current online local producer only; coherent recomputation; masks never enter model; no closed-loop efficacy claim",
          "script_sha256":SCRIPT_SHA256})
    finally:
        for hook in language_hooks:hook.remove()
        binder.forward=original_bind
        bank_module.forward=original_bank
        enc.update_progressive_grounding_address=original_update;flow.couple_local_observation=original_law
        grounder.forward=original_g;model.encode_online=original_encode;frontend.sample_action=original_sample
if __name__=="__main__":main()
