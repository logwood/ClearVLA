"""Five-step deployment runtime with an explicit static evidence cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import OnlinePolicyInput
from ..model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from .numerics import resolve_compute_dtype


@dataclass(frozen=True)
class SamplingResult:
    action: Tensor
    physical_field: Tensor
    initial_physical_noise: Tensor
    step_times: Tensor
    metrics: dict[str, Tensor]


def _integrate_cache(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None,
    initial_physical_noise: Tensor | None,
    collect_diagnostics: bool,
    dtype: torch.dtype,
    static_metrics: dict[str, Tensor] | None = None,
) -> SamplingResult:
    """Integrate one already materialized static cache."""

    cache.validate(config)
    batch = cache.history.batch
    device = cache.history.state.device
    autocast_enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    dims = config.dimensions
    if initial_physical_noise is None:
        value = model.action_codec.sample_noise(
            batch,
            device=device,
            dtype=cache.history.action_state.dtype,
            generator=generator,
        )
    else:
        expected = (batch, dims.action_horizon, model.action_codec.physical_dim)
        if tuple(initial_physical_noise.shape) != expected:
            raise ValueError(f"initial physical action noise must be {expected}")
        value = initial_physical_noise.to(device=device)
    noise = value.clone()
    steps = config.runtime.inference_steps
    dt = 1.0 / float(steps)
    times = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) * dt
    dynamic_metrics: dict[str, Tensor] = {}
    for index in range(steps):
        time = times[index].expand(batch)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            output = model.velocity(
                cache,
                noisy_action_field=value,
                time=time,
                collect_diagnostics=collect_diagnostics and index == steps - 1,
            )
        value = value + dt * output.bottom.physical_velocity.to(dtype=value.dtype)
        if collect_diagnostics and index == steps - 1:
            dynamic_metrics = output.metrics
    return SamplingResult(
        action=model.action_codec.decode(value, cache.history.action_state).float(),
        physical_field=value,
        initial_physical_noise=noise,
        step_times=times,
        metrics={**(static_metrics or {}), **dynamic_metrics},
    )


@torch.no_grad()
def sample_action(
    model: ClearVLAMainlinePolicy,
    policy_input: OnlinePolicyInput,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None = None,
    initial_physical_noise: Tensor | None = None,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> SamplingResult:
    """Integrate the 18-D physical action flow with exactly five velocity calls."""

    config.validate()
    dtype = resolve_compute_dtype(config, dtype)
    model.eval()
    autocast_enabled = policy_input.device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.autocast(
        device_type=policy_input.device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        cache, training_state, static_metrics = model.encode_online(
            policy_input,
            training_mask=False,
            geometry_supervision=False,
            collect_diagnostics=collect_diagnostics,
        )
    # High-resolution source charts are required only while G and P1 are
    # materialized.  They are deliberately not retained across ODE steps.
    del training_state
    return _integrate_cache(
        model,
        cache,
        config,
        generator=generator,
        initial_physical_noise=initial_physical_noise,
        collect_diagnostics=collect_diagnostics,
        dtype=dtype,
        static_metrics=static_metrics,
    )


@torch.no_grad()
def sample_cached_action(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None = None,
    initial_physical_noise: Tensor | None = None,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> SamplingResult:
    """Deploy from a cache already built for another read-only consumer."""

    config.validate()
    dtype = resolve_compute_dtype(config, dtype)
    model.eval()
    return _integrate_cache(
        model,
        cache,
        config,
        generator=generator,
        initial_physical_noise=initial_physical_noise,
        collect_diagnostics=collect_diagnostics,
        dtype=dtype,
    )


def deployment_cache(
    model: ClearVLAMainlinePolicy,
    policy_input: OnlinePolicyInput,
    config: ExperimentConfig,
    *,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[OnlinePolicyCache, dict[str, Tensor]]:
    """Expose cache construction for deployment integration and profiling."""

    config.validate()
    dtype = resolve_compute_dtype(config, dtype)
    model.eval()
    autocast_enabled = policy_input.device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.no_grad():
        with torch.autocast(
            device_type=policy_input.device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            cache, training_state, metrics = model.encode_online(
                policy_input,
                training_mask=False,
                geometry_supervision=False,
                collect_diagnostics=collect_diagnostics,
            )
        del training_state
        return cache, metrics


__all__ = [
    "SamplingResult",
    "deployment_cache",
    "sample_action",
    "sample_cached_action",
]
