"""Exact visual-ingress layout checks, not pretrained/robot acceptance."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.interfaces import CurrentObservation
from clearvla.mainline.model.restored_observation import RestoredV120ObservationCompiler
from clearvla.mainline.v120_core.flow_dino_evidence import _RawImagePyramid


class RecordingEncoder(nn.Module):
    """Record real prepare() ingress without allocating the full policy."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.last = None

    def forward(self, dino, *, raw_visual, source_offsets):
        self.last = (dino, raw_visual)
        return SimpleNamespace(value=dino.sum() + raw_visual.sum())


def compiler(config):
    obj = object.__new__(RestoredV120ObservationCompiler)
    nn.Module.__init__(obj)
    obj.config = config
    obj.encoder = RecordingEncoder()
    return obj


def observation():
    rgb = torch.rand(1, 3, 2, 3, 336, 336)
    # Same HWC transport layout used by the deployment adapter.
    rgb = rgb.permute(0,1,2,4,5,3).contiguous().permute(0,1,2,5,3,4)
    rgb.requires_grad_()
    dino = torch.rand(1,3,2,768,256).transpose(-1,-2).requires_grad_()
    return CurrentObservation(dino_history=dino, raw_rgb=rgb)


@pytest.mark.parametrize('with_clock', [False, True])
def test_new_chart_canonicalizes_ingress_without_changing_values_or_gradients(with_clock):
    path = Path(__file__).resolve().parents[1]/'configs/mainline/dinov3_online_cumulative_calvin.json'
    c = compiler(load_config(path))
    x = observation()
    rgb_stride, dino_stride = x.raw_rgb.stride(), x.dino_history.stride()
    offsets = torch.tensor([[-8,-4,0]]) if with_clock else None
    prepared = c.prepare(x, source_offsets=offsets, training_mask=False)
    dino, rgb = c.encoder.last
    assert dino.is_contiguous() and rgb.is_contiguous()
    assert torch.equal(dino,x.dino_history) and torch.equal(rgb,x.raw_rgb)
    assert x.raw_rgb.stride()==rgb_stride and x.dino_history.stride()==dino_stride
    prepared.pack.value.backward()
    assert torch.equal(x.raw_rgb.grad,torch.ones_like(x.raw_rgb))
    assert torch.equal(x.dino_history.grad,torch.ones_like(x.dino_history))


def test_legacy_ingress_layout_is_unchanged():
    c=compiler(ExperimentConfig())
    x=observation()
    c.prepare(x,training_mask=False)
    dino,rgb=c.encoder.last
    assert dino.stride()==x.dino_history.stride()
    assert rgb.stride()==x.raw_rgb.stride()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_canonical_raw_pyramid_is_exact_for_hwc_and_nchw_sources(dtype):
    torch.manual_seed(44)
    m=_RawImagePyramid(8,activation_checkpoint=False,visual_chart_mode='full_rgb_endpoint_v1').to(dtype).eval()
    a=torch.rand(1,3,64,64).to(dtype)
    b=a.contiguous(memory_format=torch.channels_last)
    assert torch.equal(a,b) and a.stride()!=b.stride()
    with torch.no_grad():
        first=m(a.contiguous());second=m(b.contiguous())
    for x,y in zip(first,second):
        torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_online_raster_layout_matches_worker_scatter():
    from test_dinov3_production_integration import pipeline
    from clearvla.vision.online_pipeline import rasterize_full_rgb
    p=pipeline()
    rgb=torch.rand(1,3,2,3,336,336)
    actual=p(rgb)
    expected=rasterize_full_rgb(p.encoder(rgb),(336,336))
    assert actual.is_contiguous()
    assert p.identity()['raster']['tensor_layout']=='contiguous_row_major_visual_ingress'
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
