"""Optional region objectives; no selector, slot target, or policy value path.

These numerical leaves are not enabled by a production configuration. A label
producer must independently qualify every different-region edge. In particular,
neither an allocation score nor two different masks alone supplies that edge.
"""
import math
import torch
from .identity import js_divergence


def region_separation(law, group, different, *, normalized_js_margin=.1):
    """Separate only witnessed region pairs, without one-hot or uniform K targets.

    law: B,N,K, conditional real-K BEFORE interpolation; group: B,N (-1 unknown);
    different: B,G,G detached Boolean symmetric adjacency. The online producer
    may set unsupported point groups to -1; inactive edges then contribute zero.
    A region mean is an operand for a negative edge, never a within-region
    positive label. A perfectly identical law is a stationary symmetry point;
    this loss alone is not a mathematical guarantee against every collapse.
    """
    if law.ndim!=3 or group.shape!=law.shape[:2] or different.ndim!=3:
        raise ValueError('region axes differ')
    if different.shape[0]!=law.shape[0] or different.shape[1]!=different.shape[2]:
        raise ValueError('region adjacency must preserve its batch and group axes')
    if group.dtype!=torch.long or different.dtype!=torch.bool or group.requires_grad or different.requires_grad:
        raise ValueError('region labels must be detached integer/Boolean evidence')
    if group.device!=law.device or different.device!=law.device:
        raise ValueError('region labels must be on the loss device')
    if not 0<normalized_js_margin<1:
        raise ValueError('normalized JS margin must lie strictly between zero and one')
    if not torch.equal(different,different.transpose(1,2)) or bool(different.diagonal(dim1=1,dim2=2).any()):
        raise ValueError('negative adjacency must be symmetric and exclude self edges')
    if bool(((group < -1)|(group >= different.shape[1])).any()):
        raise ValueError('region point ID is outside the declared adjacency')
    losses=[];pair_counts=[];group_counts=[]
    with torch.autocast(device_type=law.device.type,enabled=False):
        law=law.float()
        for p,ids,adj in zip(law,group,different):
            good=ids>=0;alive,inverse=torch.unique(ids[good],sorted=True,return_inverse=True)
            zero=p.sum()*0.
            if not len(alive):
                losses.append(zero);pair_counts.append(0);group_counts.append(0);continue
            values=p.new_zeros((len(alive),p.shape[-1])).index_add(0,inverse,p[good])
            count=torch.bincount(inverse,minlength=len(alive)).to(p)
            values=values/count[:,None]
            edges=torch.triu(adj[alive[:,None],alive[None,:]],diagonal=1).nonzero()
            if len(edges):
                divergence=js_divergence(values[edges[:,0]],values[edges[:,1]])
                losses.append(torch.relu(1-divergence/(math.log(2.)*normalized_js_margin)).mean())
            else:losses.append(zero)
            pair_counts.append(len(edges));group_counts.append(len(alive))
    return dict(loss=torch.stack(losses).mean(),
                supported_pairs=law.new_tensor(pair_counts).sum(),
                supported_groups=law.new_tensor(group_counts).sum())


def region_balanced_prediction(error, group, valid):
    """Additional equal-region MSE; the caller retains the full original MSE.

    Regions are strata for weighting, not asserted object identities. Unknown
    samples remain in the original objective and contribute no added stratum.
    Never use a model prediction or null probability to construct these labels.
    """
    if error.ndim!=2 or group.shape!=error.shape or valid.shape!=error.shape:
        raise ValueError('prediction region axes differ')
    if group.dtype!=torch.long or valid.dtype!=torch.bool or group.requires_grad or valid.requires_grad:
        raise ValueError('prediction strata must be detached independent labels')
    if group.device!=error.device or valid.device!=error.device:
        raise ValueError('prediction strata must be on the loss device')
    if bool((group < -1).any()):raise ValueError('unknown region must be -1')
    losses=[];counts=[]
    for row,ids,support in zip(error.float(),group,valid):
        good=support&(ids>=0);alive,inverse=torch.unique(ids[good],sorted=True,return_inverse=True)
        if len(alive):
            total=row.new_zeros(len(alive)).index_add(0,inverse,row[good])
            count=torch.bincount(inverse,minlength=len(alive)).to(row)
            losses.append((total/count).mean())
        else:losses.append(torch.where(support,row,0.).sum()*0.)
        counts.append(len(alive))
    return dict(loss=torch.stack(losses).mean(),supported_regions=error.new_tensor(counts).sum())
