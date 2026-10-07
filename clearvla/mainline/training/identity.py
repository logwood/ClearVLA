"""Geometry/temporal identity pressure and genuinely held-source prediction.

All auxiliary inputs stay on the training label plane. A source-only encoding
is rebuilt from that source's raw observed descriptors before any G1/G2 fusion;
target-view/context/history values cannot leak through the full online slots.
"""
import math
import torch
import torch.nn.functional as F
from ..model.canonical_grounding import encode_owners
from ..model.grounding import _coordinate_basis


def sample(field,xy):
    """B,D,Y,X -> B,N,D, same endpoint chart as actual online image reads."""
    return F.grid_sample(field.float(),xy.float()[:,None],align_corners=True,padding_mode='border').squeeze(2).transpose(1,2)


def js_divergence(left,right):
    with torch.autocast(device_type=left.device.type,enabled=False):
        left=left.float();right=right.float();middle=(left+right)/2
        a=left*(left.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())
        b=right*(right.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())
        return .5*(a+b).sum(-1)


def identity_terms(model,online,facts,labels):
    module=model.grounding.grounder
    if module.canonical_decoder is None or facts.image_ownership is None:
        raise ValueError('identity training requires the integrated canonical graph')
    labels.validate(batch=online.batch,device=online.device)
    raw=online.observation.dino_history[:,-2:].detach()
    b,t,c,n,d=raw.shape;side=math.isqrt(n);k=module.objects
    if side*side!=n or c!=2:raise ValueError('identity sources lost their full camera chart')
    with torch.autocast(device_type=raw.device.type,enabled=False):
        observed=model.observation.compiler.encoder.teacher_norm(raw.float())
    grid=torch.stack(torch.meshgrid(torch.linspace(-1,1,side,device=raw.device),torch.linspace(-1,1,side,device=raw.device),indexing='ij'),-1).flip(-1).reshape(n,2)
    # B*T*C independent encodings: every source excludes all target values,
    # producer priors, context fusion and history learned from other sources.
    source=observed.reshape(b*t*c,n,d)
    key=module.content_key(source)+module.coordinate_key(_coordinate_basis(grid.to(source),16))[None]
    key=module.candidate_norm(key)
    mass=torch.ones(b*t*c,n,device=source.device)
    state,log_owner=encode_owners(module,key,mass,mass.bool())
    owner=log_owner.exp().reshape(b,t,c,side,side,k+1).permute(0,1,2,5,3,4)
    conditional=torch.softmax(log_owner[...,:k].float(),-1).reshape(b,t,c,side,side,k).permute(0,1,2,5,3,4)
    state=state.reshape(b,t,c,k,module.hidden)
    target=observed[:,-1].reshape(b,c,side,side,d).permute(0,1,4,2,3)
    full=facts.image_ownership
    if full.shape!=(b,k+1,c,side,side):raise ValueError('online ownership differs from admitted identity chart')
    pair_losses=[];prediction_losses=[];counts=[];without=[]
    for kind,time in (('cross',1),('temporal',0)):
        for camera in range(c):
            destination=1-camera if kind=='cross' else camera
            valid=getattr(labels,kind+'_valid')[:,camera]
            xy_source=torch.where(valid[...,None],getattr(labels,kind+'_source')[:,camera],0.)
            xy_target=torch.where(valid[...,None],getattr(labels,kind+'_target')[:,camera],0.)
            source_law=sample(owner[:,time,camera],xy_source)
            target_law=sample(full[:,:,destination],xy_target)
            # Also align actual online ownership across cameras. Slot index
            # equality alone never supplies a positive without a sensor match.
            if kind=='cross':
                current_law=sample(full[:,:,camera],xy_source)
                divergence=(js_divergence(source_law,target_law)+js_divergence(current_law,target_law))/2
            else:divergence=js_divergence(source_law,target_law)
            count=valid.float().sum();counts.append(count)
            pair_losses.append(torch.where(valid,divergence,0.).sum()/count.clamp_min(1))
            prototype,background=module.canonical_decoder(state[:,time,camera])
            q=sample(conditional[:,time,camera],xy_source)
            predicted=torch.einsum('bnk,bkd->bnd',q.to(prototype),prototype[:,:,destination])
            position=module.decode_position(_coordinate_basis(xy_target.to(prototype),16))
            predicted=predicted+position+background[destination][None,None]
            truth=sample(target[:,destination],xy_target).detach()
            error=(predicted.float()-truth).square().mean(-1)
            prediction_losses.append(torch.where(valid,error,0.).sum()/count.clamp_min(1))
            with torch.no_grad():
                zero_value,zero_background=module.canonical_decoder(torch.zeros_like(state[:,time,camera]))
                zero_prediction=torch.einsum('bnk,bkd->bnd',q.to(zero_value),zero_value[:,:,destination])+position+zero_background[destination][None,None]
                zero_error=(zero_prediction.float()-truth).square().mean(-1)
                without.append(torch.where(valid,zero_error,0.).sum()/count.clamp_min(1))
    return dict(identity_correspondence=torch.stack(pair_losses).mean(),identity_source_prediction=torch.stack(prediction_losses).mean(),
        identity_cross_admitted_pairs=counts[0]+counts[1],identity_temporal_admitted_pairs=counts[2]+counts[3],
        identity_source_removed_prediction_mse=torch.stack(without).mean())
