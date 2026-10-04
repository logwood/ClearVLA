"""Explicit marker backbone, native 336/256x768 interface; not pretrained tests."""
from dataclasses import replace
import pytest
import test_dinov3_global_task_integration as reference


def selected(monkeypatch):
    original=reference.global_fixture
    def fixture(path,amp=False):
        c,b=original(path,amp)
        c=replace(c,top=replace(c.top,p2_effect_value_mode='s_conditioned_values_v1',task_role_value_mode='source_conditioned_values_v1',world_feedback_value_mode='innovation_and_status_v1'))
        c.validate();return c,b
    monkeypatch.setattr(reference,'global_fixture',fixture)


@pytest.mark.parametrize('amp',[False,True])
def test_native_stage2_forward(tmp_path,monkeypatch,amp):
    selected(monkeypatch);reference.test_native_chart_forward_without_update(tmp_path,amp)


def test_native_stage2_fresh_adapter(tmp_path,monkeypatch):
    selected(monkeypatch);reference.test_native_fresh_checkpoint_real_adapter(tmp_path,monkeypatch)
