"""Read-only VJPs through the actual training entry and current loss ledger.

The usual initialization, identity proof, mature clock, dataset, online DINO,
condition mask, flow schedule and formal engine forward are retained. Exit
before backward/optimizer/scheduler/checkpoint writing. No synthetic task loss,
oracle object mask, architecture intervention or deployment reinterpretation.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from clearvla.mainline import train
from clearvla.mainline.data.loading import MainlineDataBundle
from clearvla.data.instructions import normalize_instruction
from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast

class ProbeComplete(Exception):
    pass

def tensor_stats(x):
    if x is None:
        return {"connected":False,"l2":0.0,"rms":0.0}
    x=x.detach().float()
    if x.numel()==0:
        return {"connected":True,"l2":0.0,"rms":0.0,"elements":0,"finite":True}
    return {"connected":True,"l2":float(x.norm()),"rms":float(x.square().mean().sqrt()),
            "finite":bool(torch.isfinite(x).all())}

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--focus-windows",type=Path)
    args,remaining=ap.parse_known_args()
    if args.report.exists():raise FileExistsError(args.report)
    old_train_step=MainlineTrainingEngine.train_step
    old_loader=MainlineDataBundle.loader
    old_load_data=train.load_mainline_data
    source={}
    def load_data(*a,**kw):
        bundle=old_load_data(*a,**kw);source["bundle"]=bundle
        return bundle
    train.load_mainline_data=load_data
    focus=None if args.focus_windows is None else json.loads(args.focus_windows.read_text())
    def focused_loader(self,split,**kw):
        if split!="train" or focus is None:return old_loader(self,split,**kw)
        refs=self.datasets["train"].base.refs
        lookup={(self.episodes[r.episode_idx].episode_id,int(r.center)):i for i,r in enumerate(refs)}
        episodes={e.episode_id:e for e in self.episodes}
        indices=[]
        for row in focus:
            episode=episodes[row["episode"]]
            if episode.source_start is None or episode.context_start is None:
                raise ValueError("focus age requires verified annotation/context source origin")
            prefix=int(episode.source_start)-int(episode.context_start)
            if prefix!=episode.valid_center_start:
                raise ValueError("focus mapping disagrees with physical task boundary")
            center=prefix+int(row["age"])
            indices.append(lookup[(row["episode"],center)])
        if len(indices)!=kw["batch_size"]:raise ValueError("focus must form exactly one diagnostic batch")
        source["focus_indices"]=indices
        return torch.utils.data.DataLoader(self.datasets["train"],batch_sampler=[indices],
          num_workers=0,pin_memory=kw["device"].type=="cuda",generator=kw["generator"])
    MainlineDataBundle.loader=focused_loader
    def inspect(self,batch,**unused):
        model=self.model
        model.train();model.set_training_step(self.global_step)
        before_clock=(self.global_step,self.schedule.step_index)
        versions={name:p._version for name,p in model.named_parameters()}
        if self.optimizer.state:raise RuntimeError("probe requires fresh optimizer initialization")
        captured={};handles=[];organizer=model.intent.organizer
        def keep(name):
            def hook(module,inputs,output):
                captured[name]=output[0] if isinstance(output,tuple) else output
            return hook
        for name in ("goal_input","goal_read","goal_self"):
            handles.append(getattr(organizer,name).register_forward_hook(keep(name)))
        original_condition=model.conditioning.prepare
        def conditioning(*a,**kw):
            result=original_condition(*a,**kw)
            captured["goal_keep"]=result[2].detach().clone()
            return result
        model.conditioning.prepare=conditioning
        def goal_support(module,inputs,kwargs):
            captured["goal_valid"]=((~kwargs["padding_mask"]) & captured["goal_keep"][:,None].bool()).detach().clone()
        handles.append(organizer.goal_read.register_forward_pre_hook(goal_support,with_kwargs=True))
        original=self._forward_encoded
        def encoded(batch,**kw):
            captured["encoded"]=kw["encoded"]
            return original(batch,**kw)
        self._forward_encoded=encoded
        try:
            with _autocast(self.device,self.dtype):
                ledger,metrics=self._forward(batch,training=True,collect_diagnostics=False,
                    generator=self.train_flow_generator,condition_generator=self.train_condition_generator)
        finally:
            self._forward_encoded=original
            model.conditioning.prepare=original_condition
            for h in handles:h.remove()
        enc=captured["encoded"];facts=enc.training_state.top.facts;intent=enc.training_state.top.intent
        seams={name:captured[name] for name in ("goal_input","goal_read","goal_self")}
        seams.update({"g_content":facts.content,"g_semantic":facts.semantic,
          "g_reconstruction":facts.reconstructed_dino,
          "g_current_read_log_probability":facts.current_image_source.log_measure,
          "binding_mass":intent.target_binding.mass,
          "binding_log_probability":intent.target_binding.log_probability,
          "s_public_interval":intent.public_interval_carrier,
          "s_typed_value":intent.typed_relevance_value,
          "endpoint_scene":intent.annotated_goal.prediction.scene,
          "endpoint_correspondence":intent.annotated_goal.prediction.log_probability})
        groups={
          "g_all":model.grounding.grounder,
          "g_content_residual":model.grounding.grounder.decode_content_residual,
          "g_shared_position":model.grounding.grounder.decode_position,
          "s_goal_input":organizer.goal_input,
          "s_goal_read":organizer.goal_read,
          "s_goal_self":organizer.goal_self,
          "s_binder":organizer.shared_binder,
          "s_task_object_score":organizer.shared_binder.task_object_score,
          "s_endpoint_predictor":organizer.endpoint_goal}
        owned={name:tuple(p for p in module.parameters() if p.requires_grad) for name,module in groups.items()}
        parameters=list({id(p):p for values in owned.values() for p in values}.values())
        differentiable={k:v for k,v in seams.items() if v.requires_grad}
        values=list(differentiable.values())+parameters
        losses={"total":ledger.total,**{key:ledger.contributions[key] for key in
          ("action_flow","coarse_action","intent_online","object_reconstruction","annotated_goal",
           "future_dynamics","future_transition","gripper_command_transition")}}
        bundle=source["bundle"]
        bank={}
        for i,episode in enumerate(bundle.episodes):
            bank[normalize_instruction(episode.instruction)]=int(bundle.goal.episode_condition_indices[i])
        directions=[]
        for b,episode_id in enumerate(batch.audit.episode_index.tolist()):
            if not bool(captured["goal_valid"][b].any()):continue
            instruction=normalize_instruction(bundle.episodes[episode_id].instruction)
            color=next(c for c in ("red","blue","pink") if c in instruction.split())
            alternatives=[("color",instruction.replace(color,c)) for c in ("red","blue","pink") if c!=color]
            direction="left" if "left" in instruction.split() else "right"
            alternatives.append(("direction",instruction.replace(direction,"right" if direction=="left" else "left")))
            for kind,text in alternatives:
                if text not in bank:continue
                j=bank[text]
                if not torch.equal(bundle.goal.mask[j].to(batch.online.goal.mask),batch.online.goal.mask[b]):continue
                with torch.no_grad(),_autocast(self.device,self.dtype):
                    change=organizer.goal_input(bundle.goal.tokens[j:j+1].to(self.device))-organizer.goal_input(batch.online.goal.tokens[b:b+1])
                change=torch.where(captured["goal_valid"][b][None,:,None],change,0.0)
                directions.append({"sample":b,"kind":kind,"correct":instruction,"wrong":text,
                  "delta":change[0].detach().float()})
        report={"global_step":self.global_step,"source":str(Path.cwd()),
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "batch":{name:getattr(batch.audit,name).detach().cpu().tolist() for name in
                   ("sample_index","episode_index","frame_progress")},
          "loss_values":{k:float(v.detach()) for k,v in losses.items()},
          "goal_terms":{k:float(v.detach()) for k,v in ledger.terms.items() if "annotated_goal" in k and v.numel()==1},
          "focus_windows":focus,"focus_indices":source.get("focus_indices"),
          "instruction_directional_derivatives":{},
          "goal_source_kept":captured["goal_valid"].any(-1).cpu().tolist(),
          "supported_language_gradients":{},
          "seam_gradients":{},"owner_gradients":{}}
        for name,loss in losses.items():
            grads=torch.autograd.grad(loss,values,allow_unused=True,retain_graph=True) if loss.requires_grad else (None,)*len(values)
            report["seam_gradients"][name]={k:{"requires_grad":v.requires_grad} for k,v in seams.items()}
            report["seam_gradients"][name].update({k:tensor_stats(g) for k,g in zip(differentiable,grads)})
            report["supported_language_gradients"][name]={}
            for key in ("goal_input","goal_read","goal_self"):
                gradient=grads[list(differentiable).index(key)]
                if gradient is None:
                    report["supported_language_gradients"][name][key]=tensor_stats(None)
                else:
                    support=captured["goal_valid"] if key=="goal_input" else captured["goal_valid"].any(-1)[:,None].expand(gradient.shape[:2])
                    report["supported_language_gradients"][name][key]=tensor_stats(gradient[support])
            language_gradient=grads[list(differentiable).index("goal_input")]
            report["instruction_directional_derivatives"][name]=[
                {**{k:v for k,v in row.items() if k!="delta"},
                 "first_order_loss_increase_toward_wrong_instruction":
                    None if language_gradient is None else float((language_gradient[row["sample"]].float()*row["delta"]).sum())}
                for row in directions]
            pg={id(p):g for p,g in zip(parameters,grads[len(differentiable):])}
            report["owner_gradients"][name]={}
            for owner,params in owned.items():
                connected=[pg[id(p)].detach().float() for p in params if pg[id(p)] is not None]
                square=sum(float(g.square().sum()) for g in connected)
                report["owner_gradients"][name][owner]={"connected_tensors":len(connected),
                  "parameter_tensors":len(params),"l2":square**.5,
                  "rms":(square/max(1,sum(p.numel() for p in params)))**.5,
                  "finite":all(bool(torch.isfinite(g).all()) for g in connected)}
            print(json.dumps({"loss":name,"value":report["loss_values"][name],
                 "binder":report["owner_gradients"][name]["s_binder"],
                 "g":report["owner_gradients"][name]["g_all"]}),flush=True)
        assert before_clock==(self.global_step,self.schedule.step_index)
        assert not self.optimizer.state
        assert all(p.grad is None and p._version==versions[name] for name,p in model.named_parameters())
        report["scope"]={"optimizer_step":False,"checkpoint_write":False,
          "parameter_grad_write":False,"parameter_versions_unchanged":True,
          "formal_forward":True,"losses":"weighted production contributions",
          "claim":"local gradient ownership only; no physical success or semantic direction guarantee"}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
        raise ProbeComplete()
    MainlineTrainingEngine.train_step=inspect
    sys.argv=[sys.argv[0],*remaining]
    try:train.main()
    except ProbeComplete:print(json.dumps({"complete":True,"report":str(args.report)}),flush=True)
    finally:
        MainlineTrainingEngine.train_step=old_train_step
        MainlineDataBundle.loader=old_loader
        train.load_mainline_data=old_load_data

if __name__=="__main__":main()
