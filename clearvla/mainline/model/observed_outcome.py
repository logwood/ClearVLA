"""S-owned observed outcome, before coarse proposal and candidate W.

The existing measured spatial distribution associates past evidence to current
K; S's single binding then selects the target. Prediction and innovation are
deliberately not inputs. Unknown correspondence remains a separate value.
"""
import math
import torch
from torch import nn


class ObservedOutcomeRead(nn.Module):
    def __init__(self, hidden, content_dim, camera_names):
        super().__init__()
        self.camera_names=tuple(camera_names)
        self.semantic=nn.Linear(content_dim,hidden,bias=False)
        self.image=nn.Linear(2,hidden,bias=False)
        self.status=nn.Linear(3,hidden,bias=False)
        self.view=nn.Linear(len(camera_names),hidden,bias=False)
        names=sorted(camera_names)
        self.register_buffer('view_basis',torch.eye(len(names))[[names.index(n) for n in camera_names]],persistent=False)
        self.output=nn.Linear(3*hidden,hidden,bias=False)
        # New consumer starts neutral, without rewriting inherited S values.
        nn.init.zeros_(self.output.weight)

    def forward(self,feedback,facts,binding):
        feedback.validate(strict=True)
        if feedback.observed_semantic is None or feedback.observed_image is None:
            raise ValueError('S requires observed outcome separately from forecast innovation')
        if feedback.camera_names!=self.camera_names or facts.current_image_source is None:
            raise ValueError('outcome source chart differs from current physical facts')
        image=facts.current_image_source.on_image(rows=feedback.measurement_shape[0],columns=feedback.measurement_shape[1])
        current,_=image.restrict(binding.supported[:,:,None,None,None]).normalized((-2,-1))
        current=current.flatten(-2)
        current_view=(facts.camera_validity[...,0]>0)&binding.supported[...,None]
        current=torch.where(current_view[...,None],current,0.)
        past=torch.where(feedback.view_observed[...,None],feedback.posterior,0.)
        # Density ratio to a uniform image read. No learned K query, hard
        # object ID, extra language selector or renormalized unknown mass.
        overlap=torch.einsum('bkcn,bjcn->bkj',current.float(),past.float())*math.prod(feedback.measurement_shape)
        views=(current_view[:,:,None]&feedback.view_observed[:,None]).sum(-1)
        overlap=overlap/views.clamp_min(1)
        match=overlap/(overlap.sum(-1,keepdim=True)+1.)
        dtype=self.output.weight.dtype
        supported=feedback.view_observed.any(-1)
        semantic=self.semantic(torch.where(supported[...,None],feedback.observed_semantic,0.).to(dtype))
        spatial=self.image(torch.where(feedback.view_observed[...,None],feedback.observed_image,0.).to(dtype))
        role=1+torch.tanh(self.view(self.view_basis.to(dtype)))
        spatial=torch.where(feedback.view_observed[...,None],spatial*role[None,None],0.).sum(2)/feedback.view_observed.sum(-1,keepdim=True).clamp_min(1)
        law=torch.cat((past.flatten(2),feedback.null),-1)
        entropy=-(law*law.clamp_min(1e-8).log()).sum(-1)/math.log(law.shape[-1])
        unknown=feedback.null[...,0]+(1-feedback.null[...,0])*entropy
        sem=torch.einsum('bkj,bjh->bkh',match,semantic.float())
        geo=torch.einsum('bkj,bjh->bkh',match,spatial.float())
        status=torch.stack((1-match.sum(-1),torch.einsum('bkj,bj->bk',match,unknown),match.sum(-1)),-1)
        observed=current_view.any(-1)&feedback.window.observed[:,None]
        status=self.status(torch.where(observed[...,None],status,0.).to(dtype))
        value=self.output(torch.cat((sem.to(dtype),geo.to(dtype),status),-1))
        value=torch.where(observed[...,None],value,0.)
        return (value*binding.mass[...,None]).sum(1)
