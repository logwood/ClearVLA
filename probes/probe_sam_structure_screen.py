"""Read-only screening on real G inputs; adapter outputs never enter the policy.

CPU runtime is an explicit structural control, not reproduction of CUDA actions.
Body masks are used only AFTER all candidate computations for descriptive scoring.
No feature tensors are written; no optimizer exists in this probe.
"""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sam_structure_adapters import SlotToImageFeedback,RegionImageFusion,mechanical_contracts


def dump(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def fingerprint(module):
    h=hashlib.sha256()
    for n,p in module.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
class Capture(Exception):pass


def contrast(z,legal):
    z=z.float();w=legal.float();count=w.sum(1,keepdim=True).clamp_min(1)
    mean=(z*w[...,None]).sum(1)/count
    total=(z.square()*w[...,None]).sum()/(w.sum()*z.shape[-1]).clamp_min(1)
    centered=((z-mean[:,None]).square()*w[...,None]).sum()/(w.sum()*z.shape[-1]).clamp_min(1)
    return dict(rms=float(total.sqrt()),spatial_centered_rms=float(centered.sqrt()),
                centered_energy_fraction=float(centered/total.clamp_min(1e-30)))


def body_scores(values,legal,shape,masks,names):
    b,n,d=values.shape;c,h,w=shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(len(names),-1,-1,-1)
    out=[]
    for cam,mask in enumerate(masks):
        weights=F.grid_sample(torch.from_numpy(mask.astype(np.float32))[:,None],grid,align_corners=True)[:,0]
        weights=weights*legal.reshape(b,c,h,w)[0,cam].cpu()
        count=weights.sum((1,2));means=torch.einsum('ohw,hwd->od',weights,values.reshape(b,c,h,w,d)[0,cam].float().cpu())/count[:,None].clamp_min(1e-12)
        pairs=[]
        for i in range(len(names)):
            for j in range(i):
                if count[i]>0 and count[j]>0:
                    diff=(means[i]-means[j]).square().mean().sqrt()
                    scale=((means[i].square().mean()+means[j].square().mean())*.5).sqrt()
                    pairs.append(dict(objects=[names[j],names[i]],difference_rms=float(diff),relative_rms=float(diff/scale.clamp_min(1e-12))))
        out.append(dict(camera=cam,supported_atom_weights=count.tolist(),pairs=pairs))
    return out


def latency(call,device,repeat=12):
    for _ in range(3):call()
    if device.type=='cuda':torch.cuda.synchronize()
    times=[]
    for _ in range(repeat):
        t=time.perf_counter();call()
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
    return statistics.median(times)


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','plan','masks','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--limit',type=int,default=4)
    args=ap.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.manual_seed(1729);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    contracts={cls.__name__:mechanical_contracts(cls) for cls in (SlotToImageFeedback,RegionImageFusion)}
    dump(args.output/'mechanical.json',contracts)
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    from clearvla.simulation.history import CausalHistory
    from clearvla.benchmarks.calvin_eval import calvin_policy_observation
    from clearvla.mainline.model import canonical_grounding as canonical
    import clearvla.simulation.clearvla_policy as frontend
    policy=ClearVLACheckpointPolicy(args.checkpoint,device=device,
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
    model=policy.bundle.model;g=model.grounding.grounder;before=fingerprint(g)
    holder={};original_sample=frontend.sample_action;original_owner=canonical.encode_owners
    original_canonical=canonical.canonical_grounding
    def capture_online(_model,online,*a,**kw):holder['online']=online;raise Capture()
    def capture_owner(module,candidates,mass,legal):
        holder.update(candidates=candidates.detach(),mass=mass.detach(),legal=legal.detach());raise Capture()
    def capture_shape(module,local,*a,**kw):
        holder['shape']=tuple(local.current_observed_content.shape[1:4])
        return original_canonical(module,local,*a,**kw)
    frontend.sample_action=capture_online;canonical.encode_owners=capture_owner;canonical.canonical_grounding=capture_shape
    ident=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=policy.bundle.checkpoint_sha256,
        device=str(device),torch_version=torch.__version__,capture_dtype=policy.bundle.config.runtime.compute_dtype,
        adapter_math='FP32 fixed real candidate inputs',rank=32,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        masks='scoring only; never adapter/producer input',optimizer_updates=0,
        scope='isolated untrained structural screen, no policy actions, no behavioral improvement claim',production_admitted=False)
    records=[];dump(args.output/'identity.json',ident)
    try:
        for row in json.loads(args.plan.read_text())[:args.limit]:
            with np.load(Path(row['case'])/'trajectory.npz',allow_pickle=False) as z:
                data={k:z[k] for k in ('rgb_static','rgb_gripper','robot_obs','executed')}
            policy.reset();history=CausalHistory(executed_world=True)
            for t in range(max(row['steps'])+1):
                prev=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
                obs=calvin_policy_observation({'rgb_obs':{k:data[k][t] for k in ('rgb_static','rgb_gripper')},'robot_obs':data['robot_obs'][t]},prev)
                if t==0:history.reset(obs,reset_action=prev)
                else:history.append(prev,obs)
                if t!=0 and t not in row['steps']:continue
                try:policy.act_with_input(history.snapshot(),row['instruction'])
                except Capture:pass
                if t not in row['steps']:continue
                with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=policy.bundle.config.runtime.compute_dtype=='bf16'):
                    try:model.encode_online(holder['online'],training_mask=False,geometry_supervision=False)
                    except Capture:pass
                x=holder['candidates'].float();mass=holder['mass'];legal=holder['legal'];shape=holder['shape']
                with torch.no_grad():slots,log=original_owner(g,x,mass,legal)
                q=log.exp()[...,:g.objects]
                with np.load(args.masks/Path(row['case']).name/('state_%03d.npz'%t),allow_pickle=False) as z:
                    masks=[z['top_masks'],z['wrist_masks']];names=z['object_names'].tolist()
                record=dict(case_id=row['case_id'],step=t,shape=list(x.shape),camera_grid=list(shape),
                    input=contrast(x,legal),input_body=body_scores(x,legal,shape,masks,names),variants={})
                for cls in (SlotToImageFeedback,RegionImageFusion):
                    torch.manual_seed(1729);adapter=cls(x.shape[-1],32).to(device)
                    with torch.no_grad():
                        out=adapter(x,slots,q,legal,shape);latent=adapter.latent(x,slots,q,legal,shape)
                        assert torch.equal(out,x),'zero-init changed real address features'
                        sl,lp=original_owner(g,out,mass,legal)
                        assert torch.equal(sl,slots) and torch.equal(lp,log),'zero-init changed actual ownership'
                        # Collapse only for an information-transport control; not a model input.
                        qc=(q*legal[...,None]).sum(1,keepdim=True)/legal.sum(1,keepdim=True)[...,None].clamp_min(1)
                        qc=qc.expand_as(q)
                        collapsed=adapter.latent(x,slots,qc,legal,shape)
                        item=dict(parameters=sum(p.numel() for p in adapter.parameters()),
                            zero_init_features_and_ownership_exact=True,latent=contrast(latent,legal),
                            latent_body=body_scores(latent,legal,shape,masks,names),
                            spatially_constant_ownership_control=contrast(collapsed,legal))
                        # Same real input replicated to BS8: operator microbenchmark only.
                        if not records:
                            xb=x.expand(8,-1,-1).contiguous();sb=slots.expand(8,-1,-1).contiguous();qb=q.expand(8,-1,-1).contiguous();lb=legal.expand(8,-1).contiguous();mb=mass.expand(8,-1).contiguous()
                            item['bs8_replicated_forward_seconds']=latency(lambda:adapter(xb,sb,qb,lb,shape),device)
                            item['bs8_baseline_g_forward_seconds']=latency(lambda:original_owner(g,xb,mb,lb),device,repeat=4)
                            item['timing_scope']='CPU/selected-device isolated FP32 forward, repeated one observation; not full training or 8 independent samples'
                    xx=x.detach().requires_grad_();ss=slots.detach().requires_grad_();qq=q.detach().requires_grad_()
                    cot=torch.randn_like(x)
                    result=adapter(xx,ss,qq,legal,shape)
                    grads=torch.autograd.grad((result*cot).sum(),tuple(adapter.parameters()),allow_unused=True)
                    item['zero_init_parameter_vjp']={n:None if v is None else float(v.norm()) for (n,_),v in zip(adapter.named_parameters(),grads)}
                    assert item['zero_init_parameter_vjp']['out.weight']>0
                    item['vjp_scope']='random diagnostic cotangent, not training loss'
                    record['variants'][cls.__name__]=item
                records.append(record);dump(args.output/'results.json',dict(identity=ident,records=records,complete=False))
                print('SCREEN',row['case_id'],t,flush=True)
        assert fingerprint(g)==before
        dump(args.output/'results.json',dict(identity=ident,records=records,complete=True,grounder_parameters_unchanged=True))
    finally:
        frontend.sample_action=original_sample;canonical.encode_owners=original_owner;canonical.canonical_grounding=original_canonical


if __name__=='__main__':main()
