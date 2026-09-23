"""Same object identity must not disguise changed online tensors or weights."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from test_mainline_endpoint_supervision import _config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import EncodedTrainingBatch, MainlineTrainingEngine

Production = tuple[ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch]


@pytest.fixture(scope="module")
def production() -> Production:
    torch.manual_seed(7408)
    model, engine = _model_engine(_config())
    return model, engine, _batch()


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("stale graph reached Teacher computation")


def _reject(
    production: Production,
    encoded: EncodedTrainingBatch,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    model, engine, batch = production
    monkeypatch.setattr(model, "build_training_targets", forbidden)
    generator = torch.Generator().manual_seed(83)
    local_rng = generator.get_state().clone()
    global_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match=message):
        engine.eval_step(batch, encoded=encoded, generator=generator)
    assert torch.equal(generator.get_state(), local_rng)
    assert torch.equal(torch.random.get_rng_state(), global_rng)


@pytest.mark.parametrize("source", ["dino", "state", "goal", "mask", "reference", "executed"])
def test_mutated_online_tensor_rejected_before_teacher_or_rng(
    production: Production,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    _, engine, b = production
    encoded = engine.encode_eval(b, collect_diagnostics=False)
    reference = b.online.instruction_reference
    step = b.online.history.executed_robot_step
    assert reference is not None and step is not None
    tensor = {
        "dino": b.online.observation.dino_history,
        "state": b.online.history.state,
        "goal": b.online.goal.tokens,
        "mask": b.online.goal.mask,
        "reference": reference.dino,
        "executed": step.command,
    }[source]
    old = tensor.clone()
    try:
        with torch.no_grad():
            if tensor.dtype == torch.bool:
                tensor.logical_not_()
            else:
                tensor.add_(0.125)
        _reject(production, encoded, monkeypatch, "online tensor.*changed")
    finally:
        with torch.no_grad():
            tensor.copy_(old)


@pytest.mark.parametrize("update", ["inplace", "load_state", "replace_parameter"])
def test_weight_update_outside_engine_invalidates_encoded_source(
    production: Production,
    monkeypatch: pytest.MonkeyPatch,
    update: str,
) -> None:
    model, engine, b = production
    encoded = engine.encode_eval(b, collect_diagnostics=False)
    name, parameter = next(iter(model.named_parameters()))
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    original = parameter.detach().clone()
    try:
        if update == "load_state":
            # Even loading exactly equal values invalidates an old autograd
            # graph: that operation changed the actual parameter storage.
            model.load_state_dict(model.state_dict())
        elif update == "replace_parameter":
            setattr(parent, attr, torch.nn.Parameter(original.clone(), parameter.requires_grad))
        else:
            with torch.no_grad():
                parameter.add_(0.01)
        _reject(production, encoded, monkeypatch, "model parameter.*changed")
    finally:
        setattr(parent, attr, parameter)
        with torch.no_grad():
            parameter.copy_(original)


def test_fresh_encode_after_mutation_is_usable(production: Production) -> None:
    model, engine, b = production
    with torch.no_grad():
        b.online.observation.dino_history.add_(0.01)
    encoded = engine.encode_eval(b, collect_diagnostics=False)
    a = engine.eval_step(
        b, encoded=encoded, collect_diagnostics=False, generator=torch.Generator().manual_seed(15)
    )
    z = engine.eval_step(
        b, encoded=encoded, collect_diagnostics=False, generator=torch.Generator().manual_seed(15)
    )
    torch.testing.assert_close(a.loss, z.loss, rtol=0, atol=0)
    assert torch.isfinite(a.loss)
    assert not model.training


def test_future_labels_not_owned_by_online_version_guard(production: Production) -> None:
    _, engine, b = production
    encoded = engine.encode_eval(b, collect_diagnostics=False)
    future = replace(b.future, state_sequence=b.future.state_sequence.clone())
    future.state_sequence.add_(0.1)
    result = engine.eval_step(
        replace(b, future=future),
        encoded=encoded,
        collect_diagnostics=False,
        generator=torch.Generator().manual_seed(15),
    )
    assert torch.isfinite(result.loss)


def test_version_snapshot_is_reference_only_and_does_not_consume_rng() -> None:
    from clearvla.mainline.training.tensor_versions import TrainingSourceVersions

    x = torch.randn(2, 3)
    module = torch.nn.Linear(3, 2)
    rng = torch.random.get_rng_state().clone()
    snapshot = TrainingSourceVersions.capture({"x": x}, module)
    assert snapshot.online[0].tensor is x
    assert snapshot.parameters[0].tensor is module.weight
    snapshot.validate({"x": x}, module)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_storage_alias_mutation_invalidates_online_snapshot() -> None:
    from clearvla.mainline.training.tensor_versions import TrainingSourceVersions

    x = torch.zeros(2, 3)
    module = torch.nn.Linear(3, 2)
    snapshot = TrainingSourceVersions.capture({"x": x}, module)
    alias = x.view(-1)
    alias.add_(0.1)
    with pytest.raises(ValueError, match="online tensor.*changed"):
        snapshot.validate({"x": x}, module)


def test_grad_accumulation_is_not_a_parameter_value_update() -> None:
    from clearvla.mainline.training.tensor_versions import TrainingSourceVersions

    x = torch.randn(2, 3)
    module = torch.nn.Linear(3, 2)
    snapshot = TrainingSourceVersions.capture({"x": x}, module)
    module(x).square().mean().backward()
    snapshot.validate({"x": x}, module)
    module.zero_grad(set_to_none=True)
    snapshot.validate({"x": x}, module)
    module.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="model parameter.*changed"):
        snapshot.validate({"x": x}, module)


def test_inference_tensor_limit_is_explicit_and_does_not_break_reading() -> None:
    from clearvla.mainline.training.tensor_versions import TrainingSourceVersions

    with torch.inference_mode():
        x = torch.ones(2, 3)
    module = torch.nn.Linear(3, 2)
    snapshot = TrainingSourceVersions.capture({"x": x}, module)
    assert snapshot.unversioned_names == ("online[x]",)
    snapshot.validate({"x": x}, module)
    # Reference replacement still fails even though no version counter exists.
    with torch.inference_mode():
        other = x.clone()
    with pytest.raises(ValueError, match="online tensor.*changed"):
        snapshot.validate({"x": other}, module)
