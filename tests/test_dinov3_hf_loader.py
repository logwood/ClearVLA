"""Real HF architecture + safetensors I/O, explicitly RANDOM weights.

Opt-in because Transformers is optional and the full B/16 model is ~86M params.
No test in this file asserts pretrained semantics or robot success.
"""
import gc
import os

import pytest
import torch

from clearvla.vision.dinov3_online import FrozenDinoV3Encoder


@pytest.mark.skipif(os.environ.get('CLEARVLA_TEST_HF_DINOV3')!='1',reason='opt-in real HF interface; not a pretrained acceptance')
def test_hf_vitb16_strict_safetensors_load_and_actual_forward(tmp_path):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    torch.manual_seed(993)
    model=DINOv3ViTModel(DINOv3ViTConfig(hidden_size=768,intermediate_size=3072,
        num_hidden_layers=12,num_attention_heads=12,num_register_tokens=4,patch_size=16))
    model.save_pretrained(tmp_path,safe_serialization=True)
    del model
    gc.collect()
    encoder=FrozenDinoV3Encoder.from_pretrained(tmp_path,device='cpu',microbatch=1)
    pixels=torch.linspace(0,1,336).reshape(1,1,1,336).expand(1,3,336,336)
    features=encoder(pixels)
    assert features.shape==(1,256,768) and torch.isfinite(features).all()
    assert not features.is_inference() and not features.requires_grad
    head=torch.nn.Linear(768,1)
    head(features).square().mean().backward()
    assert torch.isfinite(head.weight.grad).all()
    assert all(p.grad is None for p in encoder.parameters())
    assert encoder.identity()['weights']['kind']=='local_safetensors_sha256'
    del encoder,features,head
    gc.collect()
