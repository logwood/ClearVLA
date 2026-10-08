"""Ordinary S value gradients, with explicit replay of the old surrogate.

Tests use artificial observations; no physical-object or training convergence claim.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.causal_identity import causal_identity_metadata
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.intent import _interval_common_residual
from clearvla.mainline.runtime.causal_identity_migration import (
    CAUSAL_IDENTITY_AB_V1,
    CAUSAL_UNIFIED_SOURCE_V1,
    SOURCE_DIGEST,
    config_view,
    validate_selection,
)

LEGACY = "legacy_common_surrogate_v1"
ORDINARY = "ordinary_v1"


@pytest.mark.parametrize("common_direction", [False, True])
def test_ordinary_recombination_has_exact_identity_vjp_and_jvp(common_direction):
    torch.manual_seed(451)
    value = torch.randn(2, 4, 3, 3, 5, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(value)
    if common_direction:
        direction = direction.mean(1, keepdim=True).expand_as(value)
    else:
        direction = direction - direction.mean(1, keepdim=True)

    def joined(x):
        common, residual = _interval_common_residual(x)
        return common[:, None] + residual

    output = joined(value)
    torch.testing.assert_close(output, value, atol=1e-15, rtol=1e-15)
    gradient = torch.autograd.grad((output * direction).sum(), value)[0]
    torch.testing.assert_close(gradient, direction, atol=1e-15, rtol=1e-15)
    _, tangent = torch.autograd.functional.jvp(joined, value, direction)
    torch.testing.assert_close(tangent, direction, atol=1e-15, rtol=1e-15)
    assert torch.autograd.gradcheck(joined, (value,), fast_mode=True)


def test_legacy_surrogate_is_retained_but_not_called_an_ordinary_derivative():
    value = torch.arange(16, dtype=torch.float64).view(1, 4, 4).requires_grad_()
    common, residual = _interval_common_residual(value, preserve_common_grad=True)
    torch.testing.assert_close(common[:, None] + residual, value, atol=0, rtol=0)
    gradient = torch.autograd.grad((common[:, None] + residual)[0, 0, 0], value)[0]
    torch.testing.assert_close(
        gradient[0, :, 0], torch.tensor([1.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    )
    assert gradient[0, :, 1:].count_nonzero() == 0


def test_legacy_config_payload_is_unchanged_and_new_contract_is_explicit():
    old = ExperimentConfig()
    old_top = old.as_dict()["top"]
    assert isinstance(old_top, dict)
    assert "typed_interval_gradient_mode" not in old_top
    assert causal_identity_metadata(old.top) is None
    new = replace(old, top=replace(old.top, typed_interval_gradient_mode=ORDINARY))
    payload = new.as_dict()
    new_top = payload["top"]
    assert isinstance(new_top, dict) and new_top["typed_interval_gradient_mode"] == ORDINARY
    restored = config_from_mapping(payload)
    assert restored.top.typed_interval_gradient_mode == ORDINARY
    assert restored.digest() == new.digest() != old.digest()
    meta = causal_identity_metadata(restored.top)
    assert meta is not None
    assert meta["typed_interval_gradient"] == "ordinary-common-residual-identity-v1"
    with pytest.raises(ValueError, match="gradient contract"):
        replace(old.top, typed_interval_gradient_mode="straight_through").validate()
    with pytest.raises(ValueError, match="gradient contract"):
        causal_identity_metadata({"typed_interval_gradient_mode": "unrecorded"})


def test_ordinary_gradient_change_requires_named_unified_migration():
    old = load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    new = load_config("configs/mainline/unified_source_a_calvin_check.json")
    validate_selection(old, new, SOURCE_DIGEST, mode=CAUSAL_UNIFIED_SOURCE_V1)
    with pytest.raises(ValueError, match="explicit unified"):
        validate_selection(old, new, SOURCE_DIGEST, mode=CAUSAL_IDENTITY_AB_V1)
    before = {"top": {"typed_interval_gradient_mode": LEGACY}, "objectives": {}}
    after = {"top": {"typed_interval_gradient_mode": ORDINARY}, "objectives": {}}
    assert config_view(before) != config_view(after)
    assert config_view(before, mode=CAUSAL_UNIFIED_SOURCE_V1) == config_view(
        after, mode=CAUSAL_UNIFIED_SOURCE_V1
    )


def test_actual_online_s_uses_selected_vjp_without_changing_forward_or_parameters(monkeypatch):
    from test_causal_chain_integration import _batch, _model_engine, config

    import clearvla.mainline.model.intent as module

    original = module._interval_common_residual
    captures = []

    def record(value, **kwargs):
        if "preserve_common_grad" in kwargs:
            captures.append((value, kwargs["preserve_common_grad"]))
        return original(value, **kwargs)

    monkeypatch.setattr(module, "_interval_common_residual", record)
    outputs, weights = [], []
    for mode in (LEGACY, ORDINARY):
        torch.manual_seed(1251)
        c = config()
        c = replace(c, top=replace(c.top, typed_interval_gradient_mode=mode))
        model, _ = _model_engine(c)
        model.eval()
        assert model.intent.organizer.typed_interval_gradient_mode == mode
        weights.append({name: v.detach().clone() for name, v in model.state_dict().items()})
        _, state, _ = model.encode_online(_batch().online, geometry_supervision=False)
        value, preserve = captures[-1]
        assert preserve == (mode == LEGACY)
        intent = state.top.intent
        joined = intent.typed_common_value[:, None] + intent.typed_interval_residual_value
        outputs.append(joined.detach())
        gradient = torch.autograd.grad(joined[0, 0].sum(), value)[0]
        expected = torch.zeros_like(value)
        expected[0, 0] = 1
        if mode == LEGACY:
            expected[0] += 0.25
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    assert weights[0].keys() == weights[1].keys()
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name], weights[1][name], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_deployment_abi_cannot_drop_or_relabel_interval_gradient(tmp_path):
    from test_causal_chain_integration import config
    from test_mainline_global_task import abi_for

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    c = config()
    c = replace(c, top=replace(c.top, typed_interval_gradient_mode=ORDINARY))
    _, abi, _ = abi_for(c, tmp_path)
    validate_deployment_abi(abi)
    corrupt = deepcopy(abi)
    identity = corrupt["causal_identity"]
    assert isinstance(identity, dict)
    identity.pop("typed_interval_gradient")
    with pytest.raises(ValueError):
        validate_deployment_abi(corrupt)
