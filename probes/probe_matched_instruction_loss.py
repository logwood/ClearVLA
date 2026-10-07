"""Matched complete production-loss forwards on recorded non-nearest target windows.

Only natural language changes. Future labels are fixed diagnostic observations,
not synthetic demonstrations for the wrong instruction. No optimizer update.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,math,itertools
from pathlib import Path
from dataclasses import replace
import torch
from clearvla.mainline import train
from clearvla.mainline.data.loading import MainlineDataBundle
from clearvla.data.instructions import normalize_instruction
from clearvla.mainline.training.engine import MainlineTrainingEngine,_autocast
import clearvla.mainline.training.engine as engine_module
class ProbeComplete(Exception):pass

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--focus-windows",type=Path)
    ap.add_argument("--deterministic",action="store_true")
    ap.add_argument("--fp32-diagnostic",action="store_true")
    ap.add_argument("--drop-history-diagnostic",action="store_true")
    ap.add_argument("--flow-times",type=float,nargs="+",default=[])
    ap.add_argument("--flow-seeds",type=int,nargs="+",default=[10001,10002,10003])
    args,remaining=ap.parse_known_args()
    torch.use_deterministic_algorithms(args.deterministic)
    if any(not 0<t<0.999 for t in args.flow_times):raise ValueError("diagnostic times must lie inside observed training support")
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
        model=self.model;model.eval();model.set_training_step(self.global_step)
        if args.fp32_diagnostic:
            # Change only the no-update diagnostic execution after strict restore.
            # The already encoded online DINO observations remain identical.
            self.dtype=torch.float32
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            if any(p.dtype!=torch.float32 for p in model.parameters()):
                raise TypeError("FP32 diagnostic expects existing FP32 model weights")
        clock=(self.global_step,self.schedule.step_index)
        versions={name:p._version for name,p in model.named_parameters()}
        if self.optimizer.state:raise RuntimeError("requires untouched initialized optimizer")
        bundle=source["bundle"];bank={}
        for i,e in enumerate(bundle.episodes):
            bank[normalize_instruction(e.instruction)]=int(bundle.goal.episode_condition_indices[i])
        alternatives=[]
        for row,eid in enumerate(batch.audit.episode_index.tolist()):
            text=normalize_instruction(bundle.episodes[eid].instruction)
            color=next(c for c in ("red","blue","pink") if c in text.split())
            direction=next(d for d in ("left","right") if d in text.split())
            candidates=[("color",text.replace(color,c)) for c in ("red","blue","pink") if c!=color]
            candidates.append(("direction",text.replace(direction,"right" if direction=="left" else "left")))
            for kind,wrong in candidates:
                if wrong not in bank:raise ValueError("missing exact natural template: "+wrong)
                idx=bank[wrong]
                if not torch.equal(bundle.goal.mask[idx].to(batch.online.goal.mask),batch.online.goal.mask[row]):
                    raise ValueError("matched task instructions differ in source support")
                alternatives.append({"row":row,"kind":kind,"correct":text,"wrong":wrong,"bank_index":idx})
        captures={};original=self._forward_encoded
        original_conditioning=model.conditioning.prepare
        if args.drop_history_diagnostic:
            def conditioned(value,**kw):
                config=kw["config"]
                config=replace(config,top=replace(config.top,goal_condition_dropout=0.0,action_history_condition_dropout=1.0))
                result=original_conditioning(value,**{**kw,"config":config,"training":True,"training_mask":True})
                online,proposal,goal_keep,history_keep=result
                if not bool((goal_keep==1).all()) or not bool((history_keep==0).all()):
                    raise AssertionError("diagnostic source masks did not apply")
                if bool(online.history.timing.action_executed.any()):
                    raise AssertionError("dropped command support leaked")
                return result
            model.conditioning.prepare=conditioned
        def encoded(value,**kw):
            enc=kw["encoded"];intent=enc.training_state.top.intent;facts=enc.training_state.top.facts
            captures["g_content"]=facts.content.detach().float().cpu()
            captures["binding"]=intent.target_binding.mass.detach().float().cpu()
            captures["s_public"]=intent.public_interval_carrier.detach().float().cpu()
            return original(value,**kw)
        self._forward_encoded=encoded
        original_flow=engine_module.sample_flow_matching
        override_time=None
        def matched_flow(*a,**kw):
            state=original_flow(*a,**kw)
            captures["sampled_flow_times"]=state.time.detach().cpu().clone()
            if override_time is None:return state
            time=torch.full_like(state.time,override_time)
            alpha=time.to(state.source_physical_noise.dtype)[:,None,None]
            noisy=(1-alpha)*state.source_physical_noise+alpha*state.target_physical
            if state.row_valid is not None:
                noisy=torch.where(state.row_valid[...,None],noisy,state.source_physical_noise)
            context=state.flow_step_context
            if context is not None:
                schedule=kw["schedule"]
                boundaries=torch.tensor(schedule.boundaries,device=time.device,dtype=time.dtype)
                index=torch.bucketize(time,boundaries[1:-1],right=True)
                context=replace(context,time=time,step_size=(boundaries[1:]-boundaries[:-1]).index_select(0,index),
                    normalized_index=index.to(time.dtype)/schedule.interval_count,endpoint=torch.zeros_like(time))
            return replace(state,time=time,noisy_physical=noisy,flow_step_context=context)
        engine_module.sample_flow_matching=matched_flow
        def run(value,seed):
            captures.clear()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            global_before=torch.cuda.get_rng_state(self.device).clone()
            generator=torch.Generator(device=self.device).manual_seed(seed)
            with torch.no_grad(),_autocast(self.device,self.dtype):
                ledger,metrics=self._forward(value,training=False,collect_diagnostics=False,generator=generator)
            captures["global_rng_consumed"]=torch.tensor(not torch.equal(global_before,torch.cuda.get_rng_state(self.device)))
            losses={"total":float(ledger.total),**{key:float(value) for key,value in ledger.contributions.items()}}
            if not all(math.isfinite(v) for v in losses.values()):raise FloatingPointError("nonfinite probe loss")
            return losses,{k:v.clone() for k,v in captures.items()}
        rows=[]
        try:
            for override_time,seed in itertools.product(args.flow_times or [None],args.flow_seeds):
                baseline,base_tensors=run(batch,seed)
                rows.append({"seed":seed,"flow_time_override":override_time,"variant":"correct","loss":baseline,"global_rng_consumed":bool(base_tensors["global_rng_consumed"]),"sampled_flow_times":base_tensors["sampled_flow_times"].tolist()})
                for alternative in alternatives:
                    tokens=batch.online.goal.tokens.clone();mask=batch.online.goal.mask.clone()
                    row=alternative["row"];j=alternative["bank_index"]
                    tokens[row]=bundle.goal.tokens[j].to(tokens)
                    mask[row]=bundle.goal.mask[j].to(mask)
                    changed=replace(batch,online=replace(batch.online,goal=replace(batch.online.goal,tokens=tokens,mask=mask)))
                    if changed.action_target is not batch.action_target or changed.future is not batch.future:
                        raise AssertionError("fixed measured labels changed")
                    loss,tensors=run(changed,seed)
                    if not torch.equal(tensors["g_content"],base_tensors["g_content"]):
                        raise AssertionError("G changed under a pure language intervention")
                    item={"seed":seed,"flow_time_override":override_time,"variant":"wrong_instruction",**alternative,
                      "loss":loss,"loss_increase":{k:v-baseline[k] for k,v in loss.items()},
                      "changed_row_rms":{"binding":float((tensors["binding"][row]-base_tensors["binding"][row]).square().mean().sqrt()),
                        "s_public":float((tensors["s_public"][row]-base_tensors["s_public"][row]).square().mean().sqrt())}}
                    rows.append(item)
                    print(json.dumps({"seed":seed,"row":row,"kind":alternative["kind"],
                       "action_flow_increase":item["loss_increase"].get("action_flow"),"total_increase":item["loss_increase"]["total"]}),flush=True)
                repeat,_=run(batch,seed)
                rows.append({"seed":seed,"flow_time_override":override_time,"variant":"correct_repeat","loss":repeat,
                    "loss_increase":{k:v-baseline[k] for k,v in repeat.items()}})
        finally:
            self._forward_encoded=original
            model.conditioning.prepare=original_conditioning
            engine_module.sample_flow_matching=original_flow
        assert clock==(self.global_step,self.schedule.step_index)
        assert not self.optimizer.state
        assert all(p.grad is None and p._version==versions[name] for name,p in model.named_parameters())
        report={"global_step":self.global_step,"source":str(Path.cwd()),
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "batch":{name:getattr(batch.audit,name).detach().cpu().tolist() for name in ("sample_index","episode_index","frame_progress")},
          "focus_windows":focus,"focus_indices":source.get("focus_indices"),"rows":rows,
          "scope":{"optimizer_updates":0,"parameter_grad_write":False,"parameter_versions_unchanged":True,
            "formal_loss_forward":True,"history_source_dropout_diagnostic":args.drop_history_diagnostic,"diagnostic_flow_times":args.flow_times,"deterministic_algorithms":args.deterministic,"compute_dtype":str(self.dtype),"fp32_policy_diagnostic_override":args.fp32_diagnostic,"model_mode":"eval, conditioning retained, matched owned flow and global CPU/CUDA RNG seeds",
            "units":"batch-mean weighted production losses; exactly one row language changed",
            "claim":"sensitivity to wrong instructions under fixed recorded future; NOT a physical counterfactual ground truth"}}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
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
