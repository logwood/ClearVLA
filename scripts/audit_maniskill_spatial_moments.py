"""Short CPU audit of current-position reduction using the trained checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from clearvla.vision.entity_chart import ImageLogMeasure, current_image_grid
from clearvla.mainline.model.view_geometry import ViewConditionedTransport

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    weights=ck['model']
    prefix='intent.organizer.task_relation_encoder.coordinates.'
    w0=weights[prefix+'0.weight'].float()
    w2=weights[prefix+'2.weight'].float()
    grid=current_image_grid(9,9,device=torch.device('cpu'))
    horizontal=torch.zeros(1,1,2,9,9)
    vertical=torch.zeros_like(horizontal)
    horizontal[...,4,1]=.5;horizontal[...,4,7]=.5
    vertical[...,1,4]=.5;vertical[...,7,4]=.5
    def moments(a):
        law=ImageLogMeasure(torch.where(a>0,a.clamp_min(1e-30).log(),0.),a>0)
        return law.camera_centers()
    first,second=moments(horizontal),moments(vertical)
    geometry=ViewConditionedTransport(hidden=w2.shape[0],camera_names=('top','wrist'))
    prefix_g='policy_compiler.effect_reader.view_geometry.'
    geometry.load_state_dict({k[len(prefix_g):]:v for k,v in weights.items() if k.startswith(prefix_g)},strict=True)
    support=torch.ones(1,4,1,2,dtype=torch.bool)
    covariance=torch.zeros(1,4,1,2,3);covariance[...,0]=.02;covariance[...,2]=.02
    with torch.no_grad():
        ga=geometry.context_features(first,covariance,support)
        gb=geometry.context_features(second,covariance,support)
        features=F.linear(F.silu(F.linear(grid.reshape(-1,2),w0)),w2)
        sa=torch.einsum('bkcn,nh->bkch',horizontal.flatten(-2),features)
        sb=torch.einsum('bkcn,nh->bkch',vertical.flatten(-2),features)
        design=torch.cat((grid.reshape(-1,2),torch.ones(81,1)),1)
        fitted=design@torch.linalg.lstsq(design,features).solution
        affine_r2=1-float((features-fitted).square().sum()/(features-features.mean(0)).square().sum())
    assert torch.equal(first,second)
    assert torch.equal(ga,gb)
    assert not torch.equal(sa,sb), 'S full-law spatial features unexpectedly collide'
    keys=[prefix+'0.weight',prefix+'2.weight']+sorted(k for k in weights if k.startswith(prefix_g))
    digest=hashlib.sha256()
    for k in keys:
        digest.update(k.encode());digest.update(weights[k].detach().cpu().contiguous().numpy().tobytes())
    result=dict(status='passed',checkpoint=str(args.checkpoint),global_step=ck['global_step'],
        selected_weight_sha256=digest.hexdigest(),
        distributions='equal mass at left/right versus up/down; disjoint supports, identical mean',
        current_coordinate_max_difference=float((first-second).abs().max()),
        p2_current_context_max_difference=float((ga-gb).abs().max()),
        s_spatial_feature_difference_rms=float((sa-sb).square().mean().sqrt()),
        s_coordinate_feature_affine_r2=affine_r2,
        meaning='P2 current-position coordinate branch cannot distinguish these distributions at fixed W transport covariance. S separately keeps nonlinear expected coordinate features and does distinguish them. This is not a claim that the entire policy has identical output.',
        gpu_used=False,new_forward_rollouts=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
