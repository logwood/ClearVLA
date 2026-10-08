"""Training-only StackCube region targets on the existing G image measure.

No label enters G/S/W/P or the deployment cache. Red/green masks are weak
labels from current RGB, not simulator poses. One permutation of K is matched
jointly across visible cameras; unmatched slots and null retain their owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from clearvla.vision.entity_chart import ImageLogMeasure, current_image_grid
from ..model.types import ObjectFactSet


@dataclass(frozen=True)
class StackCubeRegionLabels:
    masks: Tensor  # [B,2,C,H,W], exact current RGB color masks
    visible: Tensor  # [B,2,C], at least 20 color pixels


@torch.no_grad()
def stackcube_region_labels(current_rgb: Tensor) -> StackCubeRegionLabels:
    """The already audited narrow HSV/channel-dominance mask; no gradients."""
    import cv2

    if current_rgb.ndim != 5 or current_rgb.shape[2] != 3:
        raise ValueError("current RGB must be [B,C,3,H,W]")
    if not torch.isfinite(current_rgb).all() or current_rgb.min() < 0 or current_rgb.max() > 1:
        raise ValueError("region labels require finite current RGB in [0,1]")
    rgb = (current_rgb.detach().float() * 255).round().to(torch.uint8)
    rgb = rgb.permute(0,1,3,4,2).cpu().numpy()
    labels = []
    for sample in rgb:
        views = []
        for view in sample:
            h,s,v = cv2.split(cv2.cvtColor(view,cv2.COLOR_RGB2HSV))
            red,green,blue = view.astype(np.float32).transpose(2,0,1)
            views.append(np.stack([
                ((h<4)|(h>176)) & (red>2*green) & (red>2*blue) & (s>110) & (v>45),
                (h>40)&(h<82)&(green>1.8*red)&(green>1.8*blue)&(s>110)&(v>45),
            ]))
        labels.append(np.stack(views,axis=1))
    masks = torch.as_tensor(np.stack(labels),device=current_rgb.device)
    return StackCubeRegionLabels(masks=masks,visible=masks.sum((-2,-1))>=20)


def matched_region_terms(measure: ImageLogMeasure, labels: StackCubeRegionLabels) -> dict[str,Tensor]:
    """Balanced joint-view regional NLL plus conditional centroid distance.

The 12-pixel region tolerance matches the existing audit's 16x16 source grid.
Stable log reductions retain gradients for tiny supported mass. Empty labels
own zero loss; an admitted target with no spatial support is a source error.
The matching is a training assignment only, with one slot per visible object.
"""
    log = measure.log_mass.float()
    support = measure.supported
    batch,slots,cameras,height,width = log.shape
    if labels.masks.shape != (batch,2,cameras,height,width) or labels.visible.shape != (batch,2,cameras):
        raise ValueError("region labels and G image chart disagree")
    if slots < 2 or slots > 8:
        raise ValueError("region matching requires 2..8 G slots")
    if labels.masks.requires_grad or labels.visible.requires_grad:
        raise ValueError("spatial labels must be detached")
    masks = labels.masks.float()
    region = F.max_pool2d(masks.reshape(-1,1,height,width),25,stride=1,padding=12).reshape_as(masks)>0
    # Safe source normalization; no fake support is created in an empty view.
    flat = log.masked_fill(~support,-torch.inf).flatten(2)
    if not support.flatten(2).any(-1).all():
        raise ValueError("region training requires supported G slots")
    joint_log = log-torch.logsumexp(flat,-1)[...,None,None,None]
    regional_support = support[:, :, None] & region[:, None]
    has_region = regional_support.flatten(-2).any(-1)
    admitted = labels.visible[:,None].expand(-1,slots,-1,-1)
    if bool((admitted & ~has_region).any()):
        raise ValueError("visible color target has no source-grid support")
    masked = joint_log[:,:,None].expand(-1,-1,2,-1,-1,-1).masked_fill(~regional_support,-torch.inf)
    safe = torch.where(has_region[...,None,None],masked,torch.zeros_like(masked))
    region_log = torch.logsumexp(safe.flatten(-2),-1)
    views = labels.visible.sum(-1).clamp_min(1)
    nll_view = -(region_log + views[:,None,:,None].float().log())
    nll = torch.where(admitted,nll_view,0.).sum(-1)/views[:,None]

    probability,_ = measure.normalized((-2,-1))
    grid = current_image_grid(height,width,device=log.device)
    predicted_center = torch.einsum('bkcyx,yxd->bkcd',probability,grid)
    target_center = torch.einsum('bocyx,yxd->bocd',masks,grid)/masks.sum((-2,-1)).clamp_min(1)[...,None]
    distance = (predicted_center[:,:,None]-target_center[:,None]).square().sum(-1)
    centroid = torch.where(admitted,distance,0.).sum(-1)/views[:,None]
    cost = nll+centroid
    present = labels.visible.any(-1)
    cost = torch.where(present[:,None],cost,0.)
    choices = torch.tensor(list(permutations(range(slots),2)),device=log.device)
    candidate_cost = cost[:,choices[:,0],0]+cost[:,choices[:,1],1]
    selected = choices[candidate_cost.detach().argmin(-1)]
    b = torch.arange(batch,device=log.device)[:,None]
    obj = torch.arange(2,device=log.device)[None]
    count = present.sum().clamp_min(1)
    def reduce(value):
        return torch.where(present,value[b,selected,obj],0.).sum()/count
    mass = region_log.exp()
    conditional_mass = mass / joint_log.masked_fill(~support,-torch.inf).exp().sum((-2,-1))[:,:,None].clamp_min(1e-30)
    mean_mass = torch.where(admitted,conditional_mass,0.).sum(-1)/views[:,None]
    return {
        "spatial_grounding": reduce(cost),
        "spatial_region_nll": reduce(nll),
        "spatial_centroid_error_sq": reduce(centroid),
        "spatial_matched_region_mass": reduce(mean_mass).detach(),
        "spatial_visible_objects": present.sum().float(),
        "spatial_visible_views": labels.visible.sum().float(),
    }


def stackcube_grounding_terms(facts: ObjectFactSet,current_rgb: Tensor) -> dict[str,Tensor]:
    if facts.current_image_source is None:
        raise ValueError("spatial supervision requires the retained G image measure")
    with torch.autocast(device_type=current_rgb.device.type,enabled=False):
        labels = stackcube_region_labels(current_rgb)
        measure = facts.current_image_source.on_image(rows=current_rgb.shape[-2],columns=current_rgb.shape[-1])
        return matched_region_terms(measure,labels)
