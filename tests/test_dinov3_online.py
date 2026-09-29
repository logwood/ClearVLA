"""Source contracts only. Injected model below is NOT pretrained DINOv3."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from clearvla.vision.dinov3_online import (
    DinoV3SpatialSpec, FrozenDinoV3Encoder, extract_native_patches,
    full_image_to_patch_grid, patch_centers_in_full_image,
    preprocess_full_fov, sample_native_patch_chart,
)

class MarkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(model_type='dinov3_vit',patch_size=16,hidden_size=768,num_register_tokens=4)
        self.calls = []
    def forward(self, pixel_values):
        self.calls.append(tuple(pixel_values.shape))
        # Test-only source marker, not any approximation to a visual backbone.
        x = pixel_values.mean((1,2,3))[:,None,None] * self.weight
        return SimpleNamespace(last_hidden_state=x.expand(-1,261,768))


def test_register_tokens_are_not_spatial():
    hidden = torch.arange(261).reshape(1,261,1).expand(1,261,768).float()
    result = extract_native_patches(hidden)
    assert result.shape == (1,256,768)
    assert result[0,0,0] == 5 and result[0,-1,0] == 260

@pytest.mark.parametrize('n',[256,257,260,262])
def test_bad_token_count_fails(n):
    with pytest.raises(ValueError): extract_native_patches(torch.zeros(1,n,768))

@pytest.mark.parametrize('hw',[(336,336),(200,320),(84,84)])
def test_patch_center_transform(hw):
    centers = patch_centers_in_full_image(image_hw=hw)
    grid = full_image_to_patch_grid(centers, image_hw=hw)
    assert torch.allclose(grid[0,:,0], torch.linspace(-1,1,16),atol=2e-7)
    assert torch.allclose(grid[:,0,1], torch.linspace(-1,1,16),atol=2e-7)


def test_coordinate_field_reads_original_position_and_backpropagates():
    centers = patch_centers_in_full_image(image_hw=(336,336))
    chart = centers[None].clone().requires_grad_()
    q = torch.tensor([[[-.8,-.3],[0.,0.],[.8,.3]]],requires_grad=True)
    sampled = sample_native_patch_chart(chart,q,image_hw=(336,336))
    torch.testing.assert_close(sampled,q,atol=2e-7,rtol=0)
    sampled.sum().backward()
    assert torch.isfinite(chart.grad).all() and torch.isfinite(q.grad).all()
    # Missing conversion is observably not equivalent to a correct read.
    old = torch.nn.functional.grid_sample(chart.detach().permute(0,3,1,2),q.detach()[:,:,None],align_corners=True)
    assert (old[:,:,:,0].transpose(1,2)-q.detach()).abs().max() > .04


def test_explicit_patch_edge_behavior():
    chart=patch_centers_in_full_image(image_hw=(336,336))[None]
    q=torch.tensor([[[-1.,-1.],[1.,1.]]])
    r=sample_native_patch_chart(chart,q,image_hw=(336,336))
    torch.testing.assert_close(r[0,0],chart[0,0,0])
    torch.testing.assert_close(r[0,1],chart[0,-1,-1])


def test_full_fov_not_center_crop():
    image=torch.zeros(1,3,336,336); image[:,:,:,:20]=1; image[:,:,:,-20:]=.5
    out=preprocess_full_fov(image)
    mean=out.new_tensor([.485,.456,.406])[None,:,None,None]
    std=out.new_tensor([.229,.224,.225])[None,:,None,None]
    unnormalized=out*std+mean
    assert unnormalized[...,0].mean()>.99
    assert abs(unnormalized[...,-1].mean().item()-.5)<1e-5
    assert unnormalized[...,128].abs().max()<1e-6

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,255.])
def test_preprocessing_rejects_wrong_value_domain(bad):
    x=torch.zeros(1,3,8,8);x[0,0,0,0]=bad
    with pytest.raises(ValueError):preprocess_full_fov(x)


def test_multicamera_multitime_axis_and_fixed_batches():
    model=MarkerModel(); enc=FrozenDinoV3Encoder(model,microbatch=4)
    x=torch.linspace(0,1,6).reshape(1,3,2,1,1,1).expand(1,3,2,3,336,336)
    y=enc(x)
    assert y.shape==(1,3,2,256,768)
    assert model.calls==[(4,3,256,256),(4,3,256,256)]
    assert torch.all(y.flatten(0,2)[1:,0,0]>y.flatten(0,2)[:-1,0,0])
    assert not y.requires_grad and not y.is_inference()
    head=nn.Linear(768,1);head(y).sum().backward()
    assert head.weight.grad is not None and model.weight.grad is None
    enc.train(True)
    assert not enc.training and not model.training


def test_repeat_no_change_and_does_not_touch_rng():
    e=FrozenDinoV3Encoder(MarkerModel(),microbatch=2)
    x=torch.rand(3,2,3,32,32)
    state=torch.get_rng_state().clone(); a=e(x);b=e(x)
    assert torch.equal(a,b) and torch.equal(torch.get_rng_state(),state)

@pytest.mark.parametrize('field,value',[('model_type','dinov2'),('patch_size',14),('hidden_size',384),('num_register_tokens',0)])
def test_backbone_mismatch_fails(field,value):
    m=MarkerModel();setattr(m.config,field,value)
    with pytest.raises(ValueError):FrozenDinoV3Encoder(m)


def test_no_legacy_production_switch_is_implied():
    from clearvla.mainline.config import ExperimentConfig
    # This unit alone must not make the existing production path claim DINOv3.
    config=ExperimentConfig()
    assert 'dinov2' in config.data.dinov2_model
