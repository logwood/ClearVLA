"""Five-step deployment runtime with an explicit static evidence cache."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import OnlinePolicyInput
from ..model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from ..model.types import PhysicalActionCondition
from .numerics import resolve_compute_dtype


@dataclass(frozen=True)
class SamplingResult:
    action: Tensor
    physical_field: Tensor
    motion_logits: Tensor
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
    execution_mode: str = "learned",
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
    # V120 evaluates the vector field at [1,.8,.6,.4,.2] on its
    # noise-to-clean chart.  The mainline chart is reversed, so the exact
    # corresponding update nodes are [0,.2,.4,.6,.8], not midpoints.
    times = torch.arange(steps, device=device, dtype=torch.float32) * dt
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
                execution_mode=execution_mode,
                collect_diagnostics=collect_diagnostics and index == steps - 1,
            )
        value = value + dt * output.bottom.physical_velocity.to(dtype=value.dtype)
        if collect_diagnostics and index == steps - 1:
            dynamic_metrics = output.metrics
    # V120 evaluates the retained motion head once more at the clean endpoint.
    # This is a head-producing dynamic forward, not a sixth integration step:
    # the resulting physical field is deliberately left unchanged.
    endpoint_time = torch.ones(batch, device=device, dtype=torch.float32)
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        endpoint_output = model.velocity(
            cache,
            noisy_action_field=value,
            time=endpoint_time,
            execution_mode=execution_mode,
            collect_diagnostics=False,
        )
    return SamplingResult(
        action=model.action_codec.decode(value, cache.history.action_state).float(),
        physical_field=value,
        motion_logits=endpoint_output.bottom.motion_logits.float(),
        initial_physical_noise=noise,
        step_times=times,
        metrics={
            **(static_metrics or {}),
            **dynamic_metrics,
            "sampling_update_time_first": times[0].detach().float(),
            "sampling_update_time_last": times[-1].detach().float(),
            "sampling_endpoint_head_time": endpoint_time[0].detach().float(),
            "sampling_velocity_update_calls": times.new_tensor(float(steps)),
            "sampling_endpoint_head_calls": times.new_ones(()),
        },
    )


def refine_cached_world(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    proposal_action: Tensor,
    config: ExperimentConfig,
    *,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[OnlinePolicyCache, dict[str, Tensor]]:
    """Re-materialize W once for the decoded outer proposal.

    The first deployment pass produces a concrete 24-row native proposal.
    This helper converts that proposal to the canonical four-interval action
    ABI and rebuilds only W from the cached compact belief.  G, S, P1 and all
    dense source charts remain untouched.  Callers can then run a second
    integration with the same initial noise, making the refinement explicit
    without rerunning W at every ODE node.
    """

    config.validate()
    cache.validate(config)
    if proposal_action.ndim != 3 or tuple(proposal_action.shape[1:]) != (
        config.dimensions.action_horizon,
        config.dimensions.action_dim,
    ):
        raise ValueError("outer proposal action must be [B,24,7]")
    if int(proposal_action.shape[0]) != cache.history.batch:
        raise ValueError("outer proposal action batch does not align with cache")
    runtime_dtype = resolve_compute_dtype(config, dtype)
    action_condition = PhysicalActionCondition.from_horizon_action(
        proposal_action.to(device=cache.history.action_state.device),
        cache.history.action_state,
    )
    device = cache.history.state.device
    autocast_enabled = device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.autocast(
        device_type=device.type,
        dtype=runtime_dtype,
        enabled=autocast_enabled,
    ):
        refined_top, metrics = model.top.refine_deployment_world(
            cache.top,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
    refined_cache = replace(cache, top=refined_top)
    refined_cache.validate(config)
    return refined_cache, metrics


@torch.no_grad()
def sample_refined_cached_action_with_cache(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None = None,
    initial_physical_noise: Tensor | None = None,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[SamplingResult, OnlinePolicyCache]:
    """Run one proposal pass, one W rerun, and one refined pass.

    This remains the Schema28-compatible deployment surface used by Schema29/30.
    ``sample_cached_action`` remains
    the single-pass primitive so matched ablations can hold the world cache
    fixed; normal deployment and validation use this explicit outer closure.
    """

    config.validate()
    runtime_dtype = resolve_compute_dtype(config, dtype)
    model.eval()
    proposal = _integrate_cache(
        model,
        cache,
        config,
        generator=generator,
        initial_physical_noise=initial_physical_noise,
        collect_diagnostics=False,
        dtype=runtime_dtype,
    )
    refined_cache, refinement_metrics = refine_cached_world(
        model,
        cache,
        proposal.action,
        config,
        collect_diagnostics=collect_diagnostics,
        dtype=runtime_dtype,
    )
    refined = _integrate_cache(
        model,
        refined_cache,
        config,
        generator=None,
        initial_physical_noise=proposal.initial_physical_noise,
        collect_diagnostics=collect_diagnostics,
        dtype=runtime_dtype,
    )
    proposal_action = proposal.action.detach().float()
    refined_action = refined.action.detach().float()
    # The second ODE pass is intentionally bounded to one outer refinement.
    # Its final decoded action can therefore move away from the action that
    # conditioned the cached W.  Project the final 24-row action through the
    # same deterministic four-interval ABI and expose that residual instead
    # of silently claiming fixed-point closure.
    final_world_condition = PhysicalActionCondition.from_horizon_action(
        refined_action,
        cache.history.action_state,
    )
    refined_world_condition = refined_cache.top.action_condition
    final_interval = final_world_condition.interval_action.detach().float()
    final_delta = final_world_condition.interval_delta.detach().float()
    refined_interval = refined_world_condition.interval_action.detach().float()
    refined_delta = refined_world_condition.interval_delta.detach().float()
    outer_metrics = {
        **refinement_metrics,
        "sampling_outer_world_refinement": refined.action.new_ones(
            (), dtype=torch.float32
        ),
        "sampling_outer_proposal_action_rms": proposal_action.square().mean().sqrt(),
        "sampling_outer_refined_action_rms": refined_action.square().mean().sqrt(),
        "sampling_outer_refined_action_delta_rms": (
            refined_action - proposal_action
        ).square().mean().sqrt(),
        "sampling_outer_final_world_action_interval_rms": final_interval.square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_delta_rms": final_delta.square().mean().sqrt(),
        "sampling_outer_final_world_action_interval_mismatch_rms": (
            final_interval - refined_interval
        ).square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_delta_mismatch_rms": (
            final_delta - refined_delta
        ).square()
        .mean()
        .sqrt(),
    }
    return replace(refined, metrics={**refined.metrics, **outer_metrics}), refined_cache


@torch.no_grad()
def sample_refined_cached_action(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None = None,
    initial_physical_noise: Tensor | None = None,
    collect_diagnostics: bool = False,
    dtype: torch.dtype | None = None,
) -> SamplingResult:
    """Run the bounded outer closure and return only its final sample."""

    result, _ = sample_refined_cached_action_with_cache(
        model,
        cache,
        config,
        generator=generator,
        initial_physical_noise=initial_physical_noise,
        collect_diagnostics=collect_diagnostics,
        dtype=dtype,
    )
    return result


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
    """Integrate five updates, then evaluate heads at the clean endpoint."""

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
    # Keep the static encoding metrics on the final result while performing
    # the bounded outer proposal -> W -> correction closure.
    result = sample_refined_cached_action(
        model,
        cache,
        config,
        generator=generator,
        initial_physical_noise=initial_physical_noise,
        collect_diagnostics=collect_diagnostics,
        dtype=dtype,
    )
    return replace(result, metrics={**static_metrics, **result.metrics})


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
    execution_mode: str = "learned",
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
        execution_mode=execution_mode,
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
    "refine_cached_world",
    "sample_action",
    "sample_cached_action",
    "sample_refined_cached_action",
    "sample_refined_cached_action_with_cache",
]
