"""Frozen-checkpoint probe for training-bridge/deploy-sampling mismatch.

The probe is deliberately outside the training path.  It reuses one visual
context, one target action and one source noise per sample, then separates:

1. the teacher-conditioned training call contract from the deploy call
   contract at the same exact flow-matching bridge state;
2. the exact bridge state from the state reached by recursive deployment; and
3. the five-step solver from finer 10/20-step solvers.

No parameter, buffer, optimizer state or checkpoint is mutated.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.policy.system import V39PolicySystem

from .policy_runtime_v36_3 import decode
from .policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    _action_path_probe_batch_selection,
    _is_contract_stage,
    _uses_layer_adapter_contract,
    _validate_complete_v104_model_contract,
    prepare_v39_policy_sample,
)
from .world_runtime import autocast_context

_PRIMARY_PATH_STEPS = 5


def _sample_mse(lhs: Tensor, rhs: Tensor) -> Tensor:
    if tuple(lhs.shape) != tuple(rhs.shape):
        raise ValueError(
            "paired probe tensors must have identical shapes; "
            f"got {tuple(lhs.shape)} and {tuple(rhs.shape)}"
        )
    return (lhs.float() - rhs.detach().float()).square().flatten(1).mean(dim=1)


def _action_time_bands(horizon: int) -> tuple[tuple[str, int, int], ...]:
    """Return disjoint policy-action bands without assuming a 24-step chunk."""

    boundaries = (
        ("early_1_4", 0, min(4, horizon)),
        ("middle_5_12", min(4, horizon), min(12, horizon)),
        ("late_13_end", min(12, horizon), horizon),
    )
    return tuple((name, start, end) for name, start, end in boundaries if end > start)


def _future_error_by_horizon(
    output: dict[str, Tensor],
    target_pack: dict[str, Tensor],
    *,
    anchors: int,
    predictive_change: bool,
) -> Tensor | None:
    """Per-sample/per-horizon masked MSE in the frozen teacher chart."""

    prediction = output.get("flow_jepa_future_pred")
    future = target_pack.get("flow_jepa_future_target")
    mask = target_pack.get("flow_jepa_future_target_mask")
    if not all(torch.is_tensor(value) for value in (prediction, future, mask)):
        return None
    assert isinstance(prediction, Tensor)
    assert isinstance(future, Tensor)
    assert isinstance(mask, Tensor)
    if prediction.ndim != 3 or tuple(prediction.shape) != tuple(future.shape):
        raise ValueError("future probe expects aligned [B,N,D] prediction/target tensors")
    batch, tokens, hidden = prediction.shape
    if anchors <= 0 or tokens % anchors:
        raise ValueError("future probe cannot divide teacher tokens into configured horizons")
    positions = tokens // anchors
    pred_h = prediction.float().reshape(batch, anchors, positions, hidden)
    target_h = future.detach().float().reshape(batch, anchors, positions, hidden)
    if predictive_change:
        current = target_pack.get("flow_jepa_current_target")
        if not torch.is_tensor(current) or tuple(current.shape) != (
            batch,
            positions,
            hidden,
        ):
            raise ValueError(
                "predictive-change future probe requires [B,P,D] current teacher targets"
            )
        target_h = target_h - current.detach().float()[:, None]
    selected = mask.to(device=prediction.device, dtype=torch.float32).reshape(
        batch, anchors, positions
    )
    rows = (pred_h - target_h).square().mean(dim=-1)
    return (rows * selected).sum(dim=-1) / selected.sum(dim=-1).clamp_min(1.0)


@torch.no_grad()
def _probe_sampling_path_batch(
    *,
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    physical_noise: Tensor,
    solver_steps: Sequence[int],
    stop_at_midcut: bool,
) -> dict[str, Any]:
    """Run one paired batch while holding every non-probed input fixed."""

    if system.training:
        raise RuntimeError("sampling-path probe requires system.eval()")
    if getattr(system, "_condition_eval_intervention", None) is not None:
        raise RuntimeError("sampling-path probe cannot run under a condition intervention")
    if _PRIMARY_PATH_STEPS not in solver_steps:
        raise ValueError("sampling-path probe solver_steps must include 5")

    visual = sample["visual"]
    state_history = sample["history_state"]
    executed_history = sample["executed_action_history"]
    state = sample["state"]
    action_state = sample["action_state"]
    target_action = sample["policy_action"]
    raw_visual = sample.get("raw_visual")
    batch = int(target_action.shape[0])
    codec_state = action_state.to(device=visual.device, dtype=visual.dtype)
    target_physical = system.codec.encode(
        target_action.to(device=visual.device, dtype=visual.dtype),
        codec_state,
    )
    noise_physical = physical_noise.to(
        device=target_physical.device,
        dtype=target_physical.dtype,
    )
    if tuple(noise_physical.shape) != tuple(target_physical.shape):
        raise ValueError("physical_noise must match the encoded target action")
    target_flow = system._flow_encode(target_physical)
    noise_flow = system._flow_encode(noise_physical)
    target_flow_velocity = noise_flow - target_flow
    target_physical_velocity = system._flow_decode(target_flow_velocity)

    proposal = system.proposal(executed_history)
    proposal_tokens = system._policy_proposal_tokens(proposal["tokens"])
    goal_tokens, goal_mask = system._goal_language_batch(
        batch,
        device=executed_history.device,
        dtype=executed_history.dtype,
        tokens=None,
        mask=None,
    )
    proposal_keep = torch.ones(batch, device=visual.device, dtype=visual.dtype)
    condition_keep = torch.ones_like(proposal_keep)
    visual_context = system.planner.encode_visual_context(
        visual,
        raw_visual=raw_visual,
    )
    if visual_context is None:
        raise RuntimeError("sampling-path probe failed to compile the visual context")
    target_visual = sample.get("target_visual")
    target_pack = (
        system.build_rollout_target_pack(
            visual,
            target_visual,
            visual_context=visual_context,
        )
        if torch.is_tensor(target_visual)
        else {}
    )
    use_self_condition = bool(
        int(getattr(system.policy_config, "action_consequence_self_condition", 0))
        and int(getattr(system.policy_config, "layer_recurrent_consequence", 0))
        and int(getattr(system.policy_config, "layer_contract_adapters", 0))
    )

    def policy_velocity(
        flow_state: Tensor,
        time: Tensor,
        *,
        training_call_contract: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        physical = system._flow_decode(flow_state)
        consequence = physical
        if use_self_condition:
            preview = system._policy_forward(
                physical,
                time,
                visual,
                state_history,
                state,
                executed_history,
                proposal_tokens,
                proposal_keep,
                executed_memory=proposal["history_tokens"],
                goal_language_tokens=goal_tokens,
                goal_language_mask=goal_mask,
                goal_condition_keep=condition_keep,
                action_history_condition_keep=condition_keep,
                stop_at_midcut=stop_at_midcut,
                consequence_physical=physical,
                cvae_target_physical=None,
                enable_layer_contracts=False,
                collect_diagnostics=training_call_contract,
                visual_context=visual_context,
            )
            consequence = (
                physical - time[:, None, None] * preview["pred_physical_velocity"]
            ).detach()
            del preview
        output = system._policy_forward(
            physical,
            time,
            visual,
            state_history,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            executed_memory=proposal["history_tokens"],
            goal_language_tokens=goal_tokens,
            goal_language_mask=goal_mask,
            goal_condition_keep=condition_keep,
            action_history_condition_keep=condition_keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence,
            cvae_target_physical=(
                target_physical if training_call_contract else None
            ),
            enable_layer_contracts=training_call_contract,
            # ``collect_diagnostics=False`` returns the deploy-minimal action
            # dictionary before Flow-JEPA appends ``future_pred``.  The V104
            # Evidence decoder ignores this flag when computing action, so
            # request the observation-only tail for both call contracts while
            # preserving the deploy action inputs and layer settings.
            collect_diagnostics=True,
            visual_context=visual_context,
        )
        flow_velocity = system._flow_velocity_from_output(output)
        future_error = (
            _future_error_by_horizon(
                output,
                target_pack,
                anchors=anchors,
                predictive_change=predictive_change,
            )
            if target_pack and future_offsets
            else None
        )
        physical_velocity = system._flow_decode(flow_velocity)
        del output
        return flow_velocity, physical_velocity, future_error

    metrics: dict[str, Tensor] = {}
    recursive_flow = system._flow_project_state(noise_flow, codec_state)
    future_offsets = tuple(
        int(value)
        for value in getattr(system.policy_config, "flow_jepa_window_offsets", ())
    )
    anchors = int(getattr(system.policy_config, "future_anchors", 0))
    if future_offsets and len(future_offsets) != anchors:
        raise ValueError("future offsets do not match the configured anchor count")
    predictive_change = bool(
        int(getattr(system.policy_config, "flow_jepa_predictive_change_contract", 0))
    )
    action_bands = _action_time_bands(int(target_physical.shape[1]))

    for index in range(_PRIMARY_PATH_STEPS, 0, -1):
        time_value = float(index) / float(_PRIMARY_PATH_STEPS)
        time = torch.full(
            (batch,),
            time_value,
            device=visual.device,
            dtype=visual.dtype,
        )
        time_label = f"{time_value:.2f}"
        bridge_flow = (
            (1.0 - time[:, None, None]) * target_flow
            + time[:, None, None] * noise_flow
        )
        teacher_flow_v, teacher_physical_v, teacher_future_error = policy_velocity(
            bridge_flow,
            time,
            training_call_contract=True,
        )
        deploy_bridge_flow_v, deploy_bridge_physical_v, deploy_bridge_future_error = (
            policy_velocity(
                bridge_flow,
                time,
                training_call_contract=False,
            )
        )
        recursive_flow_v, recursive_physical_v, recursive_future_error = (
            policy_velocity(
                recursive_flow,
                time,
                training_call_contract=False,
            )
        )

        prefix = f"time/{time_label}"
        metrics[f"{prefix}/state/recursive_bridge_flow"] = _sample_mse(
            recursive_flow,
            bridge_flow,
        )
        metrics[f"{prefix}/state/recursive_bridge_physical"] = _sample_mse(
            system._flow_decode(recursive_flow),
            system._flow_decode(bridge_flow),
        )
        metrics[f"{prefix}/velocity/teacher_target"] = _sample_mse(
            teacher_physical_v,
            target_physical_velocity,
        )
        metrics[f"{prefix}/velocity/deploy_bridge_target"] = _sample_mse(
            deploy_bridge_physical_v,
            target_physical_velocity,
        )
        metrics[f"{prefix}/velocity/recursive_target"] = _sample_mse(
            recursive_physical_v,
            target_physical_velocity,
        )
        metrics[f"{prefix}/velocity/teacher_deploy_delta"] = _sample_mse(
            teacher_physical_v,
            deploy_bridge_physical_v,
        )
        metrics[f"{prefix}/velocity/bridge_recursive_delta"] = _sample_mse(
            deploy_bridge_physical_v,
            recursive_physical_v,
        )
        for band_name, start, end in action_bands:
            band_prefix = f"{prefix}/action_band/{band_name}"
            metrics[f"{band_prefix}/teacher_target"] = _sample_mse(
                teacher_physical_v[:, start:end],
                target_physical_velocity[:, start:end],
            )
            metrics[f"{band_prefix}/deploy_bridge_target"] = _sample_mse(
                deploy_bridge_physical_v[:, start:end],
                target_physical_velocity[:, start:end],
            )
            metrics[f"{band_prefix}/recursive_target"] = _sample_mse(
                recursive_physical_v[:, start:end],
                target_physical_velocity[:, start:end],
            )

        if target_pack and future_offsets:
            future_modes = (
                ("teacher", teacher_future_error),
                ("deploy_bridge", deploy_bridge_future_error),
                ("recursive", recursive_future_error),
            )
            missing_future_modes = [
                mode for mode, horizon_error in future_modes if horizon_error is None
            ]
            if missing_future_modes:
                raise RuntimeError(
                    "sampling-path future diagnostics are incomplete at "
                    f"time={time_label}; missing={missing_future_modes}. "
                    "All paired modes must expose flow_jepa_future_pred."
                )
            for mode, horizon_error in future_modes:
                assert horizon_error is not None
                for horizon_index, offset in enumerate(future_offsets):
                    metrics[
                        f"{prefix}/future_h{offset}/{mode}_target"
                    ] = horizon_error[:, horizon_index]

        recursive_flow = system._flow_project_state(
            recursive_flow - recursive_flow_v / float(_PRIMARY_PATH_STEPS),
            codec_state,
        )

    internal_primary_action = system.codec.decode(
        system._flow_decode(recursive_flow),
        codec_state,
    )
    solver_actions: dict[int, Tensor] = {}
    for steps in solver_steps:
        action = system.sample(
            visual,
            state_history,
            executed_history,
            state,
            raw_visual=raw_visual,
            action_state=action_state,
            steps=int(steps),
            noise=noise_physical,
            use_proposal=True,
            stop_at_midcut=stop_at_midcut,
            collect_diagnostics=False,
        )
        if not torch.is_tensor(action):
            raise TypeError("sampling-path solver did not return an action tensor")
        solver_actions[int(steps)] = action

    return {
        "sample_mse": metrics,
        "solver_actions": solver_actions,
        "internal_primary_action": internal_primary_action,
        "future_offsets": future_offsets,
        "action_bands": tuple(name for name, _, _ in action_bands),
    }


def _mean_metric(rows: dict[str, list[np.ndarray]], key: str) -> float:
    values = rows.get(key)
    if not values:
        raise KeyError(f"sampling-path probe did not collect {key!r}")
    return float(np.concatenate(values).mean())


def _rmse_metric(rows: dict[str, list[np.ndarray]], key: str) -> float:
    return math.sqrt(max(_mean_metric(rows, key), 0.0))


@torch.no_grad()
def evaluate_sampling_path_probe(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    probe_batches: int,
    solver_steps: Sequence[int] = (5, 10, 20),
    max_batches: int = 0,
    require_complete_v104_contract: bool = False,
) -> dict[str, Any]:
    """Evaluate bridge/call-contract/off-path/solver errors on frozen V104."""

    if probe_batches <= 0:
        raise ValueError("probe_batches must be positive")
    normalized_steps = tuple(sorted({int(value) for value in solver_steps}))
    if not normalized_steps or any(value <= 0 for value in normalized_steps):
        raise ValueError("solver_steps must contain positive integers")
    if _PRIMARY_PATH_STEPS not in normalized_steps:
        raise ValueError("solver_steps must include the deployed five-step baseline")
    if require_complete_v104_contract:
        _validate_complete_v104_model_contract(system.policy_config, trainer)

    planned_batches = len(loader)
    if max_batches:
        planned_batches = min(planned_batches, int(max_batches))
    if planned_batches <= 0:
        raise ValueError("validation loader is empty")
    budget = min(int(probe_batches), planned_batches)
    selected_indices, selection_metadata = _action_path_probe_batch_selection(
        loader=loader,
        planned_batches=planned_batches,
        budget=budget,
        gripper_index=system.policy_config.gripper_index,
        event_threshold=trainer.gripper_event_threshold,
    )

    system.eval()
    metric_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    solver_prediction_rows: dict[int, list[np.ndarray]] = {
        steps: [] for steps in normalized_steps
    }
    target_normalized_rows: list[np.ndarray] = []
    target_raw_rows: list[np.ndarray] = []
    internal_identity_max_abs = 0.0
    future_offsets: tuple[int, ...] = ()
    action_bands: tuple[str, ...] = ()
    finished_batches = 0
    sample_count = 0
    stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(
        trainer
    )

    print(
        "[v104-sampling-path-probe]"
        f" selected={sorted(selected_indices)} solver_steps={normalized_steps}"
        " target_semantics=point_anchors_from_checkpoint",
        flush=True,
    )
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > planned_batches:
            break
        if batch_index not in selected_indices:
            continue
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=True,
        )
        current_batch = int(sample["policy_action"].shape[0])
        generator = torch.Generator(device=device)
        generator.manual_seed(104729 + batch_index)
        physical_noise = system.codec.sample_noise(
            current_batch,
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
            action_state=sample["action_state"],
        )
        with autocast_context(device, dtype):
            batch_result = _probe_sampling_path_batch(
                system=system,
                sample=sample,
                physical_noise=physical_noise,
                solver_steps=normalized_steps,
                stop_at_midcut=stop_midcut_eval,
            )

        for key, value in batch_result["sample_mse"].items():
            metric_rows[key].append(value.detach().float().cpu().numpy())
        for steps, action in batch_result["solver_actions"].items():
            solver_prediction_rows[int(steps)].append(
                action.detach().float().cpu().numpy()
            )
        internal_identity_max_abs = max(
            internal_identity_max_abs,
            float(
                (
                    batch_result["internal_primary_action"]
                    - batch_result["solver_actions"][_PRIMARY_PATH_STEPS]
                )
                .detach()
                .float()
                .abs()
                .max()
                .cpu()
            ),
        )
        target_normalized_rows.append(
            sample["policy_action"].detach().float().cpu().numpy()
        )
        target_raw_rows.append(sample["policy_action_raw"].detach().cpu().numpy())
        future_offsets = tuple(int(value) for value in batch_result["future_offsets"])
        action_bands = tuple(str(value) for value in batch_result["action_bands"])
        finished_batches += 1
        sample_count += current_batch

        bridge_key = "time/0.20/state/recursive_bridge_physical"
        teacher_key = "time/0.20/velocity/teacher_target"
        deploy_key = "time/0.20/velocity/deploy_bridge_target"
        recursive_key = "time/0.20/velocity/recursive_target"
        print(
            "[v104-sampling-path-probe]"
            f" batch={batch_index}/{planned_batches}"
            f" selected={finished_batches}/{len(selected_indices)}"
            f" end_bridge_rmse={math.sqrt(float(batch_result['sample_mse'][bridge_key].mean())):.6f}"
            f" contract_excess_mse={float((batch_result['sample_mse'][deploy_key] - batch_result['sample_mse'][teacher_key]).mean()):+.3e}"
            f" offpath_excess_mse={float((batch_result['sample_mse'][recursive_key] - batch_result['sample_mse'][deploy_key]).mean()):+.3e}",
            flush=True,
        )

    if finished_batches != len(selected_indices):
        raise RuntimeError(
            "sampling-path probe finished "
            f"{finished_batches}/{len(selected_indices)} selected batches"
        )

    matched_times: dict[str, Any] = {}
    time_values = tuple(
        f"{float(index) / float(_PRIMARY_PATH_STEPS):.2f}"
        for index in range(_PRIMARY_PATH_STEPS, 0, -1)
    )
    for time_label in time_values:
        prefix = f"time/{time_label}"
        teacher_mse = _mean_metric(metric_rows, f"{prefix}/velocity/teacher_target")
        deploy_mse = _mean_metric(
            metric_rows,
            f"{prefix}/velocity/deploy_bridge_target",
        )
        recursive_mse = _mean_metric(
            metric_rows,
            f"{prefix}/velocity/recursive_target",
        )
        row: dict[str, Any] = {
            "state": {
                "recursive_vs_exact_bridge_flow_rmse": _rmse_metric(
                    metric_rows,
                    f"{prefix}/state/recursive_bridge_flow",
                ),
                "recursive_vs_exact_bridge_physical_rmse": _rmse_metric(
                    metric_rows,
                    f"{prefix}/state/recursive_bridge_physical",
                ),
            },
            "velocity": {
                "teacher_contract_target_mse": teacher_mse,
                "teacher_contract_target_rmse": math.sqrt(max(teacher_mse, 0.0)),
                "deploy_bridge_target_mse": deploy_mse,
                "deploy_bridge_target_rmse": math.sqrt(max(deploy_mse, 0.0)),
                "recursive_target_mse": recursive_mse,
                "recursive_target_rmse": math.sqrt(max(recursive_mse, 0.0)),
                "call_contract_excess_mse": deploy_mse - teacher_mse,
                "off_path_excess_mse": recursive_mse - deploy_mse,
                "teacher_vs_deploy_prediction_rmse": _rmse_metric(
                    metric_rows,
                    f"{prefix}/velocity/teacher_deploy_delta",
                ),
                "bridge_vs_recursive_prediction_rmse": _rmse_metric(
                    metric_rows,
                    f"{prefix}/velocity/bridge_recursive_delta",
                ),
            },
            "action_bands": {},
            "future_point_anchors": {},
        }
        for band_name in action_bands:
            band_prefix = f"{prefix}/action_band/{band_name}"
            band_teacher = _mean_metric(
                metric_rows,
                f"{band_prefix}/teacher_target",
            )
            band_deploy = _mean_metric(
                metric_rows,
                f"{band_prefix}/deploy_bridge_target",
            )
            band_recursive = _mean_metric(
                metric_rows,
                f"{band_prefix}/recursive_target",
            )
            row["action_bands"][band_name] = {
                "teacher_contract_target_rmse": math.sqrt(max(band_teacher, 0.0)),
                "deploy_bridge_target_rmse": math.sqrt(max(band_deploy, 0.0)),
                "recursive_target_rmse": math.sqrt(max(band_recursive, 0.0)),
                "call_contract_excess_mse": band_deploy - band_teacher,
                "off_path_excess_mse": band_recursive - band_deploy,
            }
        for offset in future_offsets:
            teacher_key = f"{prefix}/future_h{offset}/teacher_target"
            if teacher_key not in metric_rows:
                continue
            future_teacher = _mean_metric(metric_rows, teacher_key)
            future_deploy = _mean_metric(
                metric_rows,
                f"{prefix}/future_h{offset}/deploy_bridge_target",
            )
            future_recursive = _mean_metric(
                metric_rows,
                f"{prefix}/future_h{offset}/recursive_target",
            )
            row["future_point_anchors"][f"h{offset}"] = {
                "teacher_contract_target_rmse": math.sqrt(
                    max(future_teacher, 0.0)
                ),
                "deploy_bridge_target_rmse": math.sqrt(
                    max(future_deploy, 0.0)
                ),
                "recursive_target_rmse": math.sqrt(
                    max(future_recursive, 0.0)
                ),
                "call_contract_excess_mse": future_deploy - future_teacher,
                "off_path_excess_mse": future_recursive - future_deploy,
            }
        matched_times[time_label] = row

    teacher_keys = [
        f"time/{time_label}/velocity/teacher_target" for time_label in time_values
    ]
    deploy_keys = [
        f"time/{time_label}/velocity/deploy_bridge_target"
        for time_label in time_values
    ]
    recursive_keys = [
        f"time/{time_label}/velocity/recursive_target" for time_label in time_values
    ]

    def pooled_mean(keys: Sequence[str]) -> float:
        return float(
            np.concatenate(
                [np.concatenate(metric_rows[key]) for key in keys]
            ).mean()
        )

    pooled_teacher = pooled_mean(teacher_keys)
    pooled_deploy = pooled_mean(deploy_keys)
    pooled_recursive = pooled_mean(recursive_keys)
    target_normalized = np.concatenate(target_normalized_rows)
    target_raw = np.concatenate(target_raw_rows)
    normalized_predictions = {
        steps: np.concatenate(rows)
        for steps, rows in solver_prediction_rows.items()
    }
    raw_predictions = {
        steps: decode(
            action_normalizer,
            torch.from_numpy(prediction),
        )
        for steps, prediction in normalized_predictions.items()
    }
    solver_metrics: dict[str, Any] = {}
    baseline_raw_mse = float(
        ((raw_predictions[_PRIMARY_PATH_STEPS] - target_raw) ** 2).mean()
    )
    for steps in normalized_steps:
        normalized_mse = float(
            ((normalized_predictions[steps] - target_normalized) ** 2).mean()
        )
        raw_mse = float(((raw_predictions[steps] - target_raw) ** 2).mean())
        baseline_delta = float(
            np.sqrt(
                (
                    (
                        normalized_predictions[steps]
                        - normalized_predictions[_PRIMARY_PATH_STEPS]
                    )
                    ** 2
                ).mean()
            )
        )
        solver_metrics[str(steps)] = {
            "normalized_action_rmse": math.sqrt(max(normalized_mse, 0.0)),
            "raw_action_rmse": math.sqrt(max(raw_mse, 0.0)),
            "raw_action_mse_delta_vs_5": raw_mse - baseline_raw_mse,
            "prediction_delta_rmse_vs_5": baseline_delta,
        }

    return {
        "schema": "clearvla-v104-sampling-path-probe-v1",
        "complete_v103_contract_verified": bool(require_complete_v104_contract),
        "complete_v104_contract_verified": bool(require_complete_v104_contract),
        "primary_path_steps": _PRIMARY_PATH_STEPS,
        "solver_steps": list(normalized_steps),
        "target_semantics": {
            "checkpoint_future_targets": "point_anchors",
            "future_offsets": list(future_offsets),
            "proposed_interval_targets_not_applied": {
                "h4": [4, 8],
                "h12": [8, 16],
                "h24": [16, 32],
                "h48": [32, 48],
            },
        },
        "summary": {
            "teacher_contract_velocity_rmse": math.sqrt(
                max(pooled_teacher, 0.0)
            ),
            "deploy_bridge_velocity_rmse": math.sqrt(
                max(pooled_deploy, 0.0)
            ),
            "recursive_velocity_rmse": math.sqrt(
                max(pooled_recursive, 0.0)
            ),
            "call_contract_excess_mse": pooled_deploy - pooled_teacher,
            "off_path_excess_mse": pooled_recursive - pooled_deploy,
            "internal_vs_public_5_step_max_abs": internal_identity_max_abs,
        },
        "matched_times": matched_times,
        "solvers": solver_metrics,
        "planned_batches": int(planned_batches),
        "selected_batch_indices": sorted(selected_indices),
        "finished_intervention_batches": int(finished_batches),
        "intervention_samples": int(sample_count),
        "intervention_coverage": float(finished_batches / planned_batches),
        **selection_metadata,
        "interpretation": (
            "positive call_contract_excess_mse localizes error to the "
            "teacher-conditioned training call versus deployment call at the "
            "same bridge state; positive off_path_excess_mse localizes "
            "additional error to recursive states; lower 10/20-step final MSE "
            "localizes a solver-discretization component. Values may be "
            "negative when a component compensates another, so inspect all "
            "three paired measurements rather than summing absolute metrics."
        ),
    }


__all__ = ["evaluate_sampling_path_probe"]
