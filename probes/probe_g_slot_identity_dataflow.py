from pathlib import Path
from dataclasses import replace
import json, numpy as np, torch
import torch.nn.functional as F
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from scripts.probe_calvin_internal_layers_npz import _history

def cos_k(v):
    x=v.detach().float()
    a=x[:,1].reshape(x.shape[0],-1); b=x[:,3].reshape(x.shape[0],-1)
    return float(F.cosine_similarity(a,b,dim=-1).mean())
def finite(v): return bool(torch.isfinite(v).all())
def shape(v): return list(v.shape)
def sums(v, dims):
    return v.detach().float().sum(dim=dims).reshape(-1).tolist()
def main():
    device=torch.device('cuda')
    ckpt=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/short-bs8-gpu0-r9/checkpoints/best.pt')
    obs=Path('/data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003/standard.npz')
    t5=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt')
    policy=ClearVLACheckpointPolicy(ckpt,device=device,t5_condition=t5,seed=0)
    base=_history(obs)
    world=ExecutedWorldSnapshot(anchor_index=0,rgb_history=base.rgb_history,state=base.state.copy(),action_state=base.action_state.copy(),commands=base.executed_action_history[:4].copy(),visual_offsets=np.zeros(3,dtype=np.int64),observed=True)
    history=replace(base,time_index=4,previous_state=base.state.copy(),executed_world=world)
    _,online=policy.act_with_input(history,'go push the blue block right')
    calls=[]
    g_facts=[]
    original=DenseObjectGrounder._competition
    original_forward=DenseObjectGrounder.forward
    def wrapped(self,*args,**kwargs):
        out=original(self,*args,**kwargs)
        owner,mass,null,read,owner_log,read_log=out
        calls.append({'owner_shape':shape(owner),'read_shape':shape(read),'owner_finite':finite(owner),'read_finite':finite(read),'mass_finite':finite(mass),'owner_real_mass_k2k4':[float(x) for x in mass[:,:,1:4:2].detach().float().mean(dim=(0,1)).tolist()]})
        return out
    def forward_wrapped(self,*args,**kwargs):
        out=original_forward(self,*args,**kwargs)
        g_facts.append(out[0])
        return out
    DenseObjectGrounder._competition=wrapped
    DenseObjectGrounder.forward=forward_wrapped
    try:
        with torch.no_grad():
            cache,static=deployment_cache(policy.bundle.model,online,policy.bundle.config,collect_diagnostics=True)
            noise=torch.randn(1,policy.bundle.config.dimensions.action_horizon,policy.bundle.model.outlet_adapter.physical_dim,device=device,generator=torch.Generator(device=device).manual_seed(12345))
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                out=policy.bundle.model.velocity(cache,noisy_action_field=noise,time=torch.full((1,),0.4,device=device),collect_diagnostics=True)
        b=g_facts[-1]; intent=cache.top.intent; dyn=cache.top.predicted_dynamics
        ca=b.candidate_assignment
        null=b.null_assignment
        result={
            'schema':'clearvla-g-slot-identity-dataflow-v1',
            'checkpoint':str(ckpt),
            'competition':calls,
            'g':{
                'content':{'shape':shape(b.content),'k2_k4_cosine':cos_k(b.content),'finite':finite(b.content)},
                'semantic':{'shape':shape(b.semantic),'k2_k4_cosine':cos_k(b.semantic),'finite':finite(b.semantic)},
                'geometry':{'shape':shape(b.geometry),'k2_k4_cosine':cos_k(b.geometry),'finite':finite(b.geometry)},
                'camera_coordinates':{'shape':shape(b.camera_coordinates),'k2_k4_cosine':cos_k(b.camera_coordinates),'finite':finite(b.camera_coordinates)},
                'object_to_chart':{'shape':shape(b.object_to_chart),'k2_k4_cosine':cos_k(b.object_to_chart),'finite':finite(b.object_to_chart),'per_k_sum':[float(x) for x in b.object_to_chart.detach().float().sum(dim=(2,3,4)).reshape(-1).tolist()]},
                'candidate_assignment':{'shape':shape(ca),'k2_k4_cosine':cos_k(ca),'finite':finite(ca),'per_k_sum':[float(x) for x in ca.detach().float().sum(dim=(2,3,4,5)).reshape(-1).tolist()]},
                'null_assignment':{'shape':shape(null),'finite':finite(null),'sum':float(null.detach().float().sum())},
                'validity':{'shape':shape(b.validity),'values':b.validity.detach().float().reshape(-1).tolist(),'finite':finite(b.validity)},
                'reconstructed_dino':{'shape':shape(b.reconstructed_dino),'finite':finite(b.reconstructed_dino)},
                'reconstruction_error':float(b.reconstruction_error.detach().float()),
            },
            's':{
                'public_interval':{'shape':shape(intent.public_interval_carrier),'variation':float((intent.public_interval_carrier-intent.public_interval_carrier.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),'finite':finite(intent.public_interval_carrier)},
                'object_attention':{'shape':shape(intent.interval_object_attention),'variation':float((intent.interval_object_attention-intent.interval_object_attention.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),'finite':finite(intent.interval_object_attention)},
                'typed_value':{'shape':shape(intent.typed_relevance_value),'finite':finite(intent.typed_relevance_value)},
            },
            'p2':{
                'semantic_delta':{'shape':shape(dyn.semantic_delta),'variation':float((dyn.semantic_delta-dyn.semantic_delta.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),'finite':finite(dyn.semantic_delta)},
                'transport_mean':{'shape':shape(dyn.transport_mean),'variation':float((dyn.transport_mean-dyn.transport_mean.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),'finite':finite(dyn.transport_mean)},
                'physical_velocity':{'shape':shape(out.bottom.physical_velocity),'rms':float(out.bottom.physical_velocity.detach().float().square().mean().sqrt()),'finite':finite(out.bottom.physical_velocity)},
            },
            'metrics':{k:float(v.detach().float()) for k,v in out.metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1 and (str(k).startswith(('grounding_g','object_p2','object_consequence')))},
        }
    finally:
        DenseObjectGrounder._competition=original
        DenseObjectGrounder.forward=original_forward
    out_path=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/probes/g-slot-identity-dataflow-20261005.json')
    out_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n')
    print(json.dumps({'output':str(out_path),'competition_calls':len(calls),'g_shapes':{k:v['shape'] for k,v in result['g'].items() if isinstance(v,dict) and 'shape' in v}},indent=2))
if __name__=='__main__': main()
