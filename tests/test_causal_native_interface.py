"""336px / 256x768 interface checks, NOT pretrained DINO or CUDA quality tests."""
from dataclasses import replace
import pytest
import test_dinov3_global_task_integration as reference
from clearvla.mainline.p2_values import CONTEXTUAL_EFFECT_VALUES


def selected_fixture(monkeypatch):
    original = reference.global_fixture
    def build(path, amp=False):
        config, bundle = original(path, amp)
        config = replace(config, top=replace(config.top, p2_effect_value_mode=CONTEXTUAL_EFFECT_VALUES))
        config.validate()
        return config, bundle
    monkeypatch.setattr(reference, 'global_fixture', build)


@pytest.mark.parametrize('amp', [False, True])
def test_native_causal_chain_forward(tmp_path, monkeypatch, amp):
    selected_fixture(monkeypatch)
    reference.test_native_chart_forward_without_update(tmp_path, amp)


def test_native_causal_chain_fresh_checkpoint_adapter(tmp_path, monkeypatch):
    selected_fixture(monkeypatch)
    reference.test_native_fresh_checkpoint_real_adapter(tmp_path, monkeypatch)
