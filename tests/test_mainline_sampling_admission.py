"""Runtime admission rejects wrong graph/noise before encoding or RNG use."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from test_mainline_endpoint_supervision import _config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from clearvla.mainline.runtime.sampling import (
    deployment_cache,
    refine_cached_world,
    sample_action,
    sample_cached_action,
    sample_refined_cached_action,
)

Production = tuple[ExperimentConfig, ClearVLAMainlinePolicy, TrainingBatch, OnlinePolicyCache]


@pytest.fixture(scope="module")
def production() -> Production:
    torch.manual_seed(7201)
    c = _config()
    model, _ = _model_engine(c)
    batch = _batch()
    cache, _ = deployment_cache(model, batch.online, c)
    return c, model, batch, cache


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("invalid runtime input reached model computation")


@pytest.mark.parametrize("entry", ["online", "cache", "single", "refined", "world"])
def test_graph_config_mismatch_rejected_before_computation(
    production: Production,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    c, m, b, cache = production
    other = replace(
        c, bottom=replace(c.bottom, normalization_floor=c.bottom.normalization_floor * 2)
    )
    other.validate()
    monkeypatch.setattr(m, "encode_online", _forbidden)
    monkeypatch.setattr(m, "velocity", _forbidden)
    monkeypatch.setattr(m.world, "refine_deployment_world", _forbidden)
    m.train()
    before = torch.random.get_rng_state().clone()
    g = torch.Generator().manual_seed(712)
    rng = g.get_state().clone()
    try:
        with pytest.raises(ValueError, match="model.*config|config.*model"):
            if entry == "online":
                sample_action(m, b.online, other, generator=g)
            elif entry == "cache":
                deployment_cache(m, b.online, other)
            elif entry == "single":
                sample_cached_action(m, cache, other, generator=g)
            elif entry == "refined":
                sample_refined_cached_action(m, cache, other, generator=g)
            else:
                refine_cached_world(m, cache, b.action_target.normalized, other)
        assert m.training  # failed admission must not toggle the caller's mode
        assert torch.equal(g.get_state(), rng)
        assert torch.equal(torch.random.get_rng_state(), before)
    finally:
        m.eval()


@pytest.mark.parametrize("kind", ["integer", "nan", "inf", "shape"])
@pytest.mark.parametrize("entry", ["online", "single", "refined"])
def test_invalid_initial_field_rejected_before_computation(
    production: Production,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    entry: str,
) -> None:
    c, m, b, cache = production
    shape = (b.online.batch, c.dimensions.action_horizon, m.outlet_adapter.physical_dim)
    noise = torch.zeros(shape)
    if kind == "integer":
        noise = noise.long()
    elif kind == "nan":
        noise[0, 0, 0] = float("nan")
    elif kind == "inf":
        noise[0, 0, 0] = float("inf")
    else:
        noise = noise[:, :-1]
    monkeypatch.setattr(m, "encode_online", _forbidden)
    monkeypatch.setattr(m, "velocity", _forbidden)
    m.train()
    try:
        with pytest.raises((TypeError, ValueError), match="initial.*noise|initial.*field"):
            if entry == "online":
                sample_action(m, b.online, c, initial_physical_noise=noise)
            elif entry == "single":
                sample_cached_action(m, cache, c, initial_physical_noise=noise)
            else:
                sample_refined_cached_action(m, cache, c, initial_physical_noise=noise)
        assert m.training
    finally:
        m.eval()


def test_invalid_explicit_schedule_rejected_before_online_encode(
    production: Production,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c, m, b, _ = production
    monkeypatch.setattr(m, "encode_online", _forbidden)
    m.train()
    try:
        with pytest.raises((TypeError, ValueError)):
            sample_action(m, b.online, c, flow_schedule={"unknown_schedule": 1})
        assert m.training
    finally:
        m.eval()


def test_valid_runtime_and_data_overrides_preserve_sample_and_noise(production: Production) -> None:
    c, m, b, cache = production
    changed = replace(
        c,
        data=replace(c.data, output_dir="runs/runtime-relocated"),
        optimizer=replace(c.optimizer, learning_rate=c.optimizer.learning_rate * 2),
    )
    noise = torch.randn(b.online.batch, c.dimensions.action_horizon, m.outlet_adapter.physical_dim)
    original = noise.clone()
    schedule = DeploymentFlowSchedule.uniform_five()
    a = sample_cached_action(m, cache, c, initial_physical_noise=noise, flow_schedule=schedule)
    z = sample_cached_action(
        m, cache, changed, initial_physical_noise=noise, flow_schedule=schedule
    )
    torch.testing.assert_close(a.action, z.action, rtol=0, atol=0)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)
