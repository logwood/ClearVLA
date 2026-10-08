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


def conditional_real_law(log_measure,supported,*,dim=1):
    """Identity given a real entity; abstention is not an identity label.

    Normalize BEFORE spatial interpolation so a varying real/null fraction
    cannot reweight the correspondence stencil. Producer support alone owns
    missing cells. The old joint K+null objective remains a separate version.
    """
    with torch.autocast(device_type=log_measure.device.type,enabled=False):
        legal=supported.any(dim,keepdim=True)
        safe=torch.where(supported,log_measure.float(),-torch.inf)
        safe=torch.where(legal,safe,0.)
        conditional=torch.softmax(safe,dim=dim)
        return torch.where(legal,conditional,0.),legal


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
    region_mode=model.config.top.identity_supervision_mode=='rgbd_temporal_regions_v3'
    conditional_mode=model.config.top.identity_supervision_mode in {'rgbd_temporal_conditional_v2','rgbd_temporal_regions_v3'}
    if conditional_mode:
        image_source=facts.current_image_source
        full_real,full_supported=conditional_real_law(image_source.log_measure,image_source.supported)
    pair_losses=[];prediction_losses=[];counts=[];without=[];identity_counts=[];null_source=[];null_target=[]
    balanced=[];balanced_counts=[]
    for kind,time in (('cross',1),('temporal',0)):
        for camera in range(c):
            destination=1-camera if kind=='cross' else camera
            valid=getattr(labels,kind+'_valid')[:,camera]
            xy_source=torch.where(valid[...,None],getattr(labels,kind+'_source')[:,camera],0.)
            xy_target=torch.where(valid[...,None],getattr(labels,kind+'_target')[:,camera],0.)
            source_law=sample(owner[:,time,camera],xy_source)
            target_law=sample(full[:,:,destination],xy_target)
            count=valid.float().sum();counts.append(count)
            null_source.append(torch.where(valid,source_law[...,-1],0.).sum()/count.clamp_min(1))
            null_target.append(torch.where(valid,target_law[...,-1],0.).sum()/count.clamp_min(1))
            identity_valid=valid
            if conditional_mode:
                source_law=sample(conditional[:,time,camera],xy_source)
                target_law=sample(full_real[:,:,destination],xy_target)
                identity_valid=valid & (sample(full_supported[:,:,destination].float(),xy_target)[...,0]>1-1e-6)
            # Also align actual online ownership across cameras. Slot index
            # equality alone never supplies a positive without a sensor match.
            if kind=='cross':
                current_law=sample(full_real[:,:,camera] if conditional_mode else full[:,:,camera],xy_source)
                if conditional_mode:
                    identity_valid=identity_valid & (sample(full_supported[:,:,camera].float(),xy_source)[...,0]>1-1e-6)
                divergence=(js_divergence(source_law,target_law)+js_divergence(current_law,target_law))/2
            else:divergence=js_divergence(source_law,target_law)
            identity_count=identity_valid.float().sum();identity_counts.append(identity_count)
            pair_losses.append(torch.where(identity_valid,divergence,0.).sum()/identity_count.clamp_min(1))
            prototype,background=module.canonical_decoder(state[:,time,camera])
            q=sample(conditional[:,time,camera],xy_source)
            predicted=torch.einsum('bnk,bkd->bnd',q.to(prototype),prototype[:,:,destination])
            position=module.decode_position(_coordinate_basis(xy_target.to(prototype),16))
            predicted=predicted+position+background[destination][None,None]
            truth=sample(target[:,destination],xy_target).detach()
            error=(predicted.float()-truth).square().mean(-1)
            prediction_losses.append(torch.where(valid,error,0.).sum()/count.clamp_min(1))
            if region_mode:
                from .identity_regions import region_balanced_prediction
                extra=region_balanced_prediction(error,labels.prediction_region[:,0 if kind=='cross' else 1,camera],valid)
                balanced.append(extra['loss']);balanced_counts.append(extra['supported_regions'])
            with torch.no_grad():
                zero_value,zero_background=module.canonical_decoder(torch.zeros_like(state[:,time,camera]))
                zero_prediction=torch.einsum('bnk,bkd->bnd',q.to(zero_value),zero_value[:,:,destination])+position+zero_background[destination][None,None]
                zero_error=(zero_prediction.float()-truth).square().mean(-1)
                without.append(torch.where(valid,zero_error,0.).sum()/count.clamp_min(1))
    result=dict(identity_correspondence=torch.stack(pair_losses).mean(),identity_source_prediction=torch.stack(prediction_losses).mean(),
        identity_cross_admitted_pairs=counts[0]+counts[1],identity_temporal_admitted_pairs=counts[2]+counts[3],
        identity_source_removed_prediction_mse=torch.stack(without).mean())
    if conditional_mode:
        result.update(identity_conditional_supported_pairs=torch.stack(identity_counts).sum().detach(),
            identity_conditional_support_fraction=(torch.stack(identity_counts).sum()/torch.stack(counts).sum().clamp_min(1)).detach(),
            identity_source_null_mass=torch.stack(null_source).mean().detach(),
            identity_target_null_mass=torch.stack(null_target).mean().detach())
    if region_mode:
        from .identity_regions import region_separation
        separation=[];negative_counts=[]
        for camera in range(c):
            # The full original 32x32 source grid is retained independently of
            # cross-view correspondence acceptance. Only actual producer
            # support may remove an online identity operand.
            xy=labels.cross_source[:,camera]
            source_law=sample(conditional[:,1,camera],xy)
            current_law=sample(full_real[:,:,camera],xy)
            online_support=sample(full_supported[:,:,camera].float(),xy)[...,0]>1-1e-6
            for law,groups in ((source_law,labels.region_group[:,camera]),
                (current_law,torch.where(online_support,labels.region_group[:,camera],-1))):
                extra=region_separation(law,groups,labels.region_different[:,camera],
                    normalized_js_margin=model.config.top.identity_region_js_margin)
                separation.append(extra['loss']);negative_counts.append(extra['supported_pairs'])
        result.update(identity_region_separation=torch.stack(separation).mean(),
            identity_region_prediction=torch.stack(balanced).mean(),
            identity_region_supported_negative_pairs=torch.stack(negative_counts).sum().detach(),
            identity_region_prediction_strata=torch.stack(balanced_counts).sum().detach())
    return result
