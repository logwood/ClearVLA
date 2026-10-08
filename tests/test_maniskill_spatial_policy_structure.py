"""Structural anti-averaging gates, not a claim of robot task success."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch

from clearvla.mainline.config import load_config, config_from_mapping
from clearvla.mainline.model.compiler import ObjectFutureEffectReader
from clearvla.mainline.model.spatial_posterior import SpatialPosteriorContext, spatial_compatibility
from clearvla.mainline.model.view_geometry import ViewConditionedTransport
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.p2_geometry import POSTERIOR_VIEW_TRANSPORT, p2_geometry_metadata
from clearvla.mainline.training.spatial_supervision import StackCubeRegionLabels, matched_identity_binding_terms, native_region_density
from clearvla.vision.entity_chart import ImageLogMeasure, current_image_grid


def collision():
    a = torch.zeros(1, 2, 2, 16, 16)
    a[..., 7:9, 2] = .25
    a[..., 7:9, 13] = .25
    b = a.transpose(-2, -1).contiguous()
    return a, b


def test_equal_centroids_do_not_alias_in_new_W_or_P2_features():
    torch.manual_seed(139)
    a, b = collision()
    grid = current_image_grid(16, 16, device=a.device)
    torch.testing.assert_close((a[..., None]*grid).sum((-3,-2)), (b[..., None]*grid).sum((-3,-2)))
    support = torch.ones(1, 2, 2, dtype=torch.bool)
    w = SpatialPosteriorContext(32)
    assert not torch.allclose(w(a,support), w(b,support))
    p2 = ViewConditionedTransport(hidden=32,camera_names=("top","wrist"),posterior=True)
    xy = torch.zeros(1,2,2,2); cov=torch.zeros(1,4,2,2,3)
    views=support[:,None].expand(1,4,2,2)
    ca=p2.context_features(xy,cov,views,a)
    cb=p2.context_features(xy,cov,views,b)
    assert (ca-cb).abs().max() > .01
    zero=torch.zeros_like(ca)
    assert torch.count_nonzero(p2.condition(zero,ca,views,value=True)) == 0


def test_full_law_compatibility_matches_direct_integral_and_has_gradient():
    torch.manual_seed(140)
    p=torch.randn(2,3,2,16,16).flatten(-2).softmax(-1).reshape(2,3,2,16,16).requires_grad_()
    q=torch.rand(2,3,2,1,1,2,2).requires_grad_()
    d=torch.rand(2,4,3,2,2).mul(.1).requires_grad_()
    cov=torch.zeros(2,4,3,2,3)
    metric=ObjectFutureEffectReader._covariance_aware_distance
    score=spatial_compatibility(p,q,d,cov,metric)
    grid=current_image_grid(16,16,device=p.device).flatten(0,1)
    delta=q[...,None,:]-(grid+d[...,None,:]).clamp(-1,1)[:,None,None]
    direct=((-.25*delta.square().sum(-1)).clamp(-1,0).exp()*p.flatten(-2)[:,None,None,None]).sum(-1).log()
    torch.testing.assert_close(score,direct)
    grads=torch.autograd.grad(score.sum(),(p,q,d))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(spatial_compatibility(p[:,perm],q,d[:,:,perm],cov[:,:,perm],metric),score[...,perm,:])


def test_missing_view_quarantine_and_gradient():
    p,_=collision();p[...,1,:,:]=torch.nan;p.requires_grad_()
    support=torch.ones(1,2,2,dtype=torch.bool);support[...,1]=False
    m=SpatialPosteriorContext(16);out=m(p,support)
    assert torch.isfinite(out).all() and torch.count_nonzero(out[...,1,:])==0
    out.square().sum().backward()
    assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad[...,1,:,:])==0
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters())


def label_fixture():
    masks=torch.zeros(1,2,2,16,16,dtype=torch.bool)
    masks[0,0,0,3:5,3:5]=True;masks[0,0,1,10:12,10:12]=True
    masks[0,1,0,10:12,10:12]=True;masks[0,1,1,3:5,3:5]=True
    labels=StackCubeRegionLabels(masks,torch.ones(1,2,2,dtype=torch.bool))
    log=torch.zeros(1,4,2,16,16)
    log[:,:2]=masks.float()*6
    binding=TargetBinding.from_logits(torch.tensor([[4.,-2.,-2.,-2.]],requires_grad=True),
                                    torch.zeros(1,1,requires_grad=True),torch.ones(1,4,dtype=torch.bool))
    return log.requires_grad_(),labels,binding


def test_label_projection_preserves_image_coordinates_at_edges_and_inside():
    mask=torch.zeros(1,2,2,63,81)
    mask[...,0,0]=1;mask[...,9:13,59:64]=1;mask[...,-1,-1]=1
    density=native_region_density(mask,16,16)
    original=mask/mask.sum((-2,-1),keepdim=True)
    fine=current_image_grid(63,81,device=mask.device)
    native=current_image_grid(16,16,device=mask.device)
    torch.testing.assert_close((original[...,None]*fine).sum((-3,-2)),
                               (density[...,None]*native).sum((-3,-2)),atol=2e-7,rtol=0)
    torch.testing.assert_close(density.sum((-2,-1)),torch.ones(1,2,2))


def terms(log,labels,binding):
    return matched_identity_binding_terms(ImageLogMeasure(log,torch.ones_like(log,dtype=torch.bool)),labels,binding)


def test_cross_view_conflict_and_distant_second_mode_are_penalized():
    log,labels,b=label_fixture()
    good=terms(log,labels,b)["spatial_identity"]
    swapped=log.detach().clone();swapped[:,:,1]=swapped[:,[1,0,2,3],1]
    assert terms(swapped,labels,b)["spatial_identity"] > good+1
    diffuse=log.detach().clone();diffuse[:,:2]+=labels.masks[:,[1,0]].float()*6
    assert terms(diffuse,labels,b)["spatial_identity"] > good+.4


def test_assignment_and_target_label_are_equivariant_and_have_live_gradients():
    log,labels,b=label_fixture();out=terms(log,labels,b)
    perm=torch.tensor([2,0,3,1]);other=terms(log[:,perm],labels,b.permute(perm))
    for name in ("spatial_identity","spatial_target_binding"):
        torch.testing.assert_close(out[name],other[name])
    (out["spatial_identity"]+out["spatial_target_binding"]).backward()
    assert torch.isfinite(log.grad).all() and log.grad.abs().sum()>0


def test_no_visible_target_is_not_a_null_or_forced_binding_label():
    log,labels,b=label_fixture()
    visible=labels.visible.clone();visible[:,0]=False
    t=terms(log,replace(labels,visible=visible),b)
    assert t["spatial_target_binding"]==0
    empty=replace(labels,visible=torch.zeros_like(visible))
    loss=terms(log,empty,b)
    assert loss["spatial_identity"]==0 and loss["spatial_target_binding"]==0
    (loss["spatial_identity"]+loss["spatial_target_binding"]).backward()
    assert torch.count_nonzero(log.grad)==0


def test_selected_config_is_explicit_and_legacy_identity_unchanged():
    cfg=load_config("configs/mainline/maniskill_spatial_policy_full_20261008.json")
    assert config_from_mapping(cfg.as_dict())==cfg
    assert cfg.optimizer.epochs==20 and cfg.runtime.max_train_batches==0 and cfg.runtime.max_val_batches==0
    old=load_config("configs/mainline/maniskill_spatial_candidate_20261008.json")
    assert "maniskill_target_binding" not in old.as_dict()["objectives"]
    assert p2_geometry_metadata(("top","wrist"))["schema"]=="p2-view-conditioned-transport-v1"
    meta=p2_geometry_metadata(("top","wrist"),POSTERIOR_VIEW_TRANSPORT)
    assert meta["schema"]=="p2-posterior-view-transport-v2"
    with pytest.raises(ValueError):
        replace(cfg,top=replace(cfg.top,task_execution_mode="joint_object_scene_v1")).validate()

