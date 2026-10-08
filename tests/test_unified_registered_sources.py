"""Typed source transfer and optional feedback fields, without numeric changes."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from clearvla.mainline.model.policy import _detach_registered


def test_typed_transfer_checks_before_mutating_and_preserves_object_identity():
    owner = nn.Module()
    child = nn.Linear(3, 4)
    parameter = nn.Parameter(torch.ones(2))
    buffer = torch.zeros(3)
    owner.add_module("child", child)
    owner.register_parameter("coefficient", parameter)
    owner.register_buffer("source", buffer)
    owner.register_module("absent", None)
    before = dict(owner.named_parameters())
    with pytest.raises(TypeError, match="child"):
        _detach_registered(owner, "child", nn.Parameter)
    assert owner._modules["child"] is child
    assert before.keys() == dict(owner.named_parameters()).keys()
    assert all(before[name] is p for name, p in owner.named_parameters())
    with pytest.raises(TypeError, match="absent"):
        _detach_registered(owner, "absent", nn.Module)
    assert "absent" in owner._modules
    assert _detach_registered(owner, "child", nn.Linear) is child
    assert _detach_registered(owner, "coefficient", nn.Parameter) is parameter
    assert _detach_registered(owner, "source", torch.Tensor) is buffer
    assert not list(owner.parameters()) and not list(owner.buffers())
    with pytest.raises(KeyError):
        _detach_registered(owner, "child", nn.Module)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_constructor_matches_unchecked_transfer_law(monkeypatch, variant):
    from clearvla.mainline.model import policy
    from scripts.check_unified_model_flow import configuration

    config = configuration(
        "small", variant, "conditional_object_v1", outcome_mode="robot_world_before_proposal_v2"
    )
    torch.manual_seed(1744)
    checked = policy.ClearVLAMainlinePolicy(config)
    checked_rng = torch.random.get_rng_state().clone()

    def unchecked(source, name, expected):
        # The former implementation moved the same object with no type check.
        for registry in (source._modules, source._parameters, source._buffers):
            if name in registry:
                return registry.pop(name)
        raise KeyError(name)

    monkeypatch.setattr(policy, "_detach_registered", unchecked)
    torch.manual_seed(1744)
    original = policy.ClearVLAMainlinePolicy(config)
    assert torch.equal(checked_rng, torch.random.get_rng_state())
    assert list(checked.state_dict()) == list(original.state_dict())
    assert [n for n, _ in checked.named_parameters()] == [n for n, _ in original.named_parameters()]
    for name, value in checked.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)


def test_optional_world_outcomes_are_all_absent_or_all_typed():
    from clearvla.mainline.executed_world import ExecutedWorldFeedback
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from scripts.check_unified_model_flow import configuration

    config = configuration("small", "A")
    batch, _ = synthetic_batch(config, count=1, raw_side=32, device=torch.device("cpu"))
    window = batch.online.history.executed_world_window
    assert window is not None
    semantic = torch.zeros(1, 1, config.dimensions.visual_token_dim)
    image = torch.zeros(1, 1, 2, 2)
    feedback = ExecutedWorldFeedback(
        semantic=semantic,
        image=image,
        covariance=torch.zeros(1, 1, 2, 3),
        null=torch.zeros(1, 1, 1),
        posterior=torch.full((1, 1, 2, 4), 1 / 8),
        past_content=semantic.clone(),
        view_observed=torch.ones(1, 1, 2, dtype=torch.bool),
        window=window,
        current_dino=batch.online.observation.dino_history,
        camera_names=("top", "wrist"),
        measurement_shape=(2, 2),
    )
    feedback.validate(strict=True)
    full = replace(
        feedback,
        observed_semantic=semantic.clone(),
        predicted_semantic=semantic.clone(),
        observed_image=image.clone(),
        predicted_image=image.clone(),
    )
    full.validate(strict=True)
    for name in ("observed_semantic", "predicted_semantic", "observed_image", "predicted_image"):
        with pytest.raises(ValueError, match="must be complete"):
            replace(full, **{name: None}).validate(strict=True)
    with pytest.raises(ValueError, match="observed minus predicted"):
        replace(full, observed_semantic=torch.ones_like(semantic)).validate(strict=True)
