"""Shared differentiable E5/H5 lifecycle for hybrid training and deployment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from clearvla.action_representations.composite import EndpointSpec, build_hybrid_v1_contract

from ..model.policy import PolicyStepOutput

if TYPE_CHECKING:
    from ..config import ExperimentConfig
    from ..model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
    from .sampling import SamplingResult


@contextmanager
def deployment_field_mode(model):
    """Disable dropout, not gradients, also during checkpoint recomputation."""
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def role_contract(model):
    """One host-side boundary object; the learned B-spine owner is unchanged."""
    contract = getattr(model, "_hybrid_role_contract", None)
    if contract is None:
        contract = build_hybrid_v1_contract(
            codec_id="physical_action_field_value_adjacent_difference_v1",
            normalizer_id="checkpoint_owned_pen_action_normalizer",
            causal_boundary_id="pen_current_action_state",
            endpoint_specs=(
                EndpointSpec(
                    role_id="motion_logits",
                    decode_group_id=None,
                    semantic_kind="arm_motion",
                    payload_kind="logits",
                    payload_shape=(24,),
                    axis_names=("action_time",),
                    temporal_alignment="action_horizon",
                    distribution_kind="independent_binary",
                    vocabulary_id="arm_motion_boolean_v1",
                    usage="auxiliary",
                    producer_id="terminal_action_controller.motion_head",
                    action_mapping="none",
                    boundary_policy="selected_clean_state",
                ),
            ),
        )
        # A dataclass, not an nn.Module: fixed boundary matrices are not a
        # second checkpoint/optimizer owner. Their fingerprints are serialized.
        model._hybrid_role_contract = contract
    return contract


def _velocity(
    model, cache, config, value, time, *, dtype, execution_mode, diagnostics
) -> PolicyStepOutput:
    def evaluate(state, clock):
        enabled = state.device.type in {"cpu", "cuda"} and dtype in {
            torch.bfloat16,
            torch.float16,
        }
        with (
            deployment_field_mode(model),
            torch.autocast(
                device_type=state.device.type,
                dtype=dtype,
                enabled=enabled,
                cache_enabled=False,
            ),
        ):
            return model.velocity(
                cache,
                noisy_action_field=state,
                time=clock,
                execution_mode=execution_mode,
                deployment_fastpath=False,
                require_execution_supervision=False,
                collect_diagnostics=diagnostics,
            )

    if torch.is_grad_enabled() and config.hybrid.checkpoint_rollout:
        return cast(
            PolicyStepOutput,
            checkpoint(evaluate, value, time, use_reentrant=False, preserve_rng_state=True),
        )
    return cast(PolicyStepOutput, evaluate(value, time))


def integrate_hybrid_pass(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    initial: Tensor,
    *,
    method: str,
    dtype: torch.dtype,
    collect_diagnostics: bool = False,
    execution_mode: str = "learned",
) -> SamplingResult:
    """Integrate the entire state and read fresh endpoints on its final value."""
    from .sampling import SamplingResult

    if method not in {"euler", "heun"}:
        raise ValueError("hybrid pass requires euler or heun")
    cache.validate(config)
    expected = (
        cache.history.batch,
        config.dimensions.action_horizon,
        model.outlet_adapter.physical_dim,
    )
    if tuple(initial.shape) != expected:
        raise ValueError(f"hybrid initial noise must be {expected}")
    value = initial.clone()
    steps = config.runtime.inference_steps
    dt = 1.0 / float(steps)
    times = torch.arange(steps, device=value.device, dtype=torch.float32) * dt
    dynamic_metrics = {}
    for index in range(steps):
        output = _velocity(
            model,
            cache,
            config,
            value,
            times[index].expand(expected[0]),
            dtype=dtype,
            execution_mode=execution_mode,
            diagnostics=collect_diagnostics and index == steps - 1,
        )
        v0 = output.bottom.physical_velocity.to(dtype=value.dtype)
        if method == "euler":
            value = value + dt * v0
        else:
            predicted = value + dt * v0
            end_time = (times[index + 1] if index + 1 < steps else times.new_ones(())).expand(
                expected[0]
            )
            corrected = _velocity(
                model,
                cache,
                config,
                predicted,
                end_time,
                dtype=dtype,
                execution_mode=execution_mode,
                diagnostics=False,
            )
            v1 = corrected.bottom.physical_velocity.to(dtype=value.dtype)
            value = value + (0.5 * dt) * (v0 + v1)
        if collect_diagnostics and index == steps - 1:
            dynamic_metrics = output.metrics
    endpoint = _velocity(
        model,
        cache,
        config,
        value,
        times.new_ones((expected[0],)),
        dtype=dtype,
        execution_mode=execution_mode,
        diagnostics=False,
    )
    # Exercise the latest composite on actual pass outputs, outside the ODE
    # loop. The retained view is exact and keeps the full autograd path.
    representation = role_contract(model).representation
    packed = representation.encode(
        value,
        endpoints={"motion_logits": endpoint.bottom.motion_logits.float()},
    )
    selected = representation.decode(packed, view="retained")
    if selected.requires_endpoint_refresh:
        raise RuntimeError("hybrid selected field invalidated its endpoint producer")
    selected_field = selected.continuous_state
    outlet = model.outlet_adapter.finalize(
        selected_field,
        cache.history.action_state,
        codec_gripper_boundary=cache.history.codec_gripper_boundary,
        command_logits=endpoint.bottom.gripper_command_logits,
    )
    nfe = steps * (2 if method == "heun" else 1)
    metrics = {
        **dynamic_metrics,
        **model.outlet_adapter.sampling_metrics(outlet),
        "sampling_update_time_first": times[0],
        "sampling_update_time_last": times[-1],
        "sampling_endpoint_head_time": times.new_ones(()),
        "sampling_velocity_update_calls": times.new_tensor(float(nfe)),
        "sampling_endpoint_head_calls": times.new_ones(()),
        "hybrid_role_retained_identity_max_abs": (selected_field - value).detach().abs().max(),
        "hybrid_role_endpoint_refresh_required": times.new_zeros(()),
        "hybrid_solver_heun": times.new_tensor(float(method == "heun")),
    }
    return SamplingResult(
        action=outlet.deployed_action,
        physical_field=selected_field,
        motion_logits=selected.endpoints["motion_logits"],
        initial_physical_noise=initial.clone(),
        step_times=times,
        metrics=metrics,
        gripper_command_logits=outlet.command_logits,
        gripper_command=outlet.command,
        continuous_action=outlet.continuous_action,
    )


@dataclass(frozen=True)
class HybridRolloutResult:
    proposal: SamplingResult
    refined: SamplingResult
    refined_cache: OnlinePolicyCache
    metrics: dict[str, Tensor]

    @property
    def refined_action(self):
        return self.refined.action


def differentiable_hybrid_rollout(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    initial_physical_noise: Tensor,
    *,
    dtype: torch.dtype,
    collect_diagnostics: bool = False,
    execution_mode: str = "learned",
) -> HybridRolloutResult:
    """E5 -> one differentiable W rebuild -> H5 from the identical noise."""
    from ..model.types import PhysicalActionCondition
    from .sampling import refine_cached_world

    if not config.hybrid.enabled:
        raise ValueError("hybrid rollout requires an enabled hybrid configuration")
    proposal = integrate_hybrid_pass(
        model,
        cache,
        config,
        initial_physical_noise,
        method=config.hybrid.proposal_method,
        dtype=dtype,
        execution_mode=execution_mode,
    )
    with deployment_field_mode(model):
        refined_cache, rebuild_metrics = refine_cached_world(
            model,
            cache,
            proposal.action,
            config,
            collect_diagnostics=collect_diagnostics,
            dtype=dtype,
        )
    refined = integrate_hybrid_pass(
        model,
        refined_cache,
        config,
        initial_physical_noise,
        method=config.hybrid.refined_method,
        dtype=dtype,
        execution_mode=execution_mode,
        collect_diagnostics=collect_diagnostics,
    )
    final_condition = PhysicalActionCondition.from_horizon_action(
        refined.action,
        cache.history.action_state,
    )
    world_condition = refined_cache.top.action_condition
    metric = initial_physical_noise.new_tensor
    metrics = {
        **rebuild_metrics,
        "sampling_outer_world_refinement": metric(1.0),
        "sampling_outer_proposal_action_rms": proposal.action.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "sampling_outer_refined_action_rms": refined.action.detach().float().square().mean().sqrt(),
        "sampling_outer_refined_action_delta_rms": (refined.action - proposal.action)
        .detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_interval_rms": final_condition.interval_action.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_delta_rms": final_condition.interval_delta.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_interval_mismatch_rms": (
            final_condition.interval_action - world_condition.interval_action
        )
        .detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "sampling_outer_final_world_action_delta_mismatch_rms": (
            final_condition.interval_delta - world_condition.interval_delta
        )
        .detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "hybrid_solver_proposal_physical_nfe": metric(5.0),
        "hybrid_solver_refined_physical_nfe": metric(10.0),
        "hybrid_solver_physical_nfe_total": metric(15.0),
        "hybrid_solver_endpoint_calls_total": metric(2.0),
        "hybrid_solver_total_dynamic_calls": metric(17.0),
        "hybrid_solver_w_rebuild_count": metric(1.0),
        "hybrid_solver_activation_checkpointing": metric(
            float(torch.is_grad_enabled() and config.hybrid.checkpoint_rollout)
        ),
    }
    return HybridRolloutResult(proposal, refined, refined_cache, metrics)


__all__ = [
    "HybridRolloutResult",
    "differentiable_hybrid_rollout",
    "integrate_hybrid_pass",
    "role_contract",
]
