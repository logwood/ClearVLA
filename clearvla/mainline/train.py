"""Single training entry point for the capability-named ClearVLA mainline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .checkpoint import (
    CheckpointIdentity,
    build_checkpoint_identity,
    checkpoint_identity_from_mapping,
    compare_checkpoint_identity,
)
from .config import ExperimentConfig, load_config
from .data.loading import MainlineDataBundle, load_mainline_data, to_training_batch
from .model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from .model.types import PhysicalActionCondition
from .runtime.checkpoints import (
    load_checkpoint_exact,
    load_checkpoint_for_validation,
    migrate_bottom_only,
    save_checkpoint,
)
from .runtime.evaluation import (
    MatchedCoreAttributionAccumulator,
    MatchedP2InterventionAccumulator,
    ValidationAccumulator,
)
from .runtime.identity import (
    dataset_identity,
    language_identity,
    v120_normalizer_fingerprint,
)
from .runtime.logging import (
    DeviceMetricAccumulator,
    JsonlRunLogger,
    archival_metrics,
    validate_resume_metric_boundary,
)
from .runtime.numerics import resolve_compute_dtype
from .runtime.sampling import (
    sample_cached_action,
    sample_refined_cached_action,
    sample_refined_cached_action_with_cache,
)
from .training.engine import (
    MainlineTrainingEngine,
    NonFiniteGradientError,
    validate_finite_training_batch,
)
from .training.gradient_audit import (
    FiniteGradientSpikeReport,
    GradientPreclipWindowAccumulator,
)
from .training.optimizer import WarmupCosineSchedule, build_optimizer, role_lr_scale


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the independent object_intent_dynamics_323 mainline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mainline/object_intent_dynamics_323.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-checkpoint", type=Path)
    parser.add_argument("--migrate-bottom", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-null-goal", action="store_true")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data = config.data
    optimizer = config.optimizer
    runtime = config.runtime
    if args.output_dir is not None:
        data = replace(data, output_dir=str(args.output_dir))
    if args.data_root is not None:
        data = replace(data, raw_hdf5_root=str(args.data_root))
    if args.decoded_cache is not None:
        data = replace(data, decoded_cache=str(args.decoded_cache))
    if args.dino_cache is not None:
        data = replace(data, dino_cache=str(args.dino_cache))
    if args.t5_condition is not None:
        data = replace(data, t5_condition=str(args.t5_condition))
    if args.num_workers is not None:
        data = replace(data, num_workers=int(args.num_workers))
    if args.epochs is not None:
        optimizer = replace(optimizer, epochs=int(args.epochs))
    if args.batch_size is not None:
        optimizer = replace(optimizer, batch_size=int(args.batch_size))
    if args.max_train_batches is not None:
        runtime = replace(runtime, max_train_batches=int(args.max_train_batches))
    if args.max_val_batches is not None:
        runtime = replace(runtime, max_val_batches=int(args.max_val_batches))
    if args.dtype is not None:
        runtime = replace(runtime, compute_dtype=str(args.dtype))
    if args.smoke:
        data = replace(data, num_workers=0 if args.num_workers is None else data.num_workers)
        optimizer = replace(
            optimizer,
            epochs=1 if args.epochs is None else optimizer.epochs,
            batch_size=1 if args.batch_size is None else optimizer.batch_size,
        )
        runtime = replace(
            runtime,
            log_every=1,
            max_train_batches=(2 if args.max_train_batches is None else runtime.max_train_batches),
            max_val_batches=1 if args.max_val_batches is None else runtime.max_val_batches,
        )
    result = replace(config, data=data, optimizer=optimizer, runtime=runtime)
    result.validate()
    if args.allow_null_goal and not args.smoke:
        raise ValueError("--allow-null-goal is restricted to explicit smoke runs")
    checkpoint_modes = (
        args.resume is not None,
        args.validate_checkpoint is not None,
        args.migrate_bottom is not None,
    )
    if sum(checkpoint_modes) > 1:
        raise ValueError(
            "exact resume, read-only validation and bottom-only migration are mutually exclusive"
        )
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit(loader_length: int, maximum: int) -> int:
    return loader_length if maximum <= 0 else min(loader_length, maximum)


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    gib = float(1024**3)
    allocated = torch.cuda.memory_allocated(index) / gib
    reserved = torch.cuda.memory_reserved(index) / gib
    peak_allocated = torch.cuda.max_memory_allocated(index) / gib
    peak_reserved = torch.cuda.max_memory_reserved(index) / gib
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    device_used = (total_bytes - free_bytes) / gib
    # On the dedicated experiment GPU this is a conservative process peak:
    # PyTorch's exact peak reservation plus the currently visible CUDA context
    # overhead.  If another process occupies the device it deliberately grows
    # and the production-memory gate refuses to claim a controlled result.
    non_pytorch_context = max(device_used - reserved, 0.0)
    return {
        "runtime_cuda_allocated_gib": allocated,
        "runtime_cuda_reserved_gib": reserved,
        "runtime_cuda_peak_allocated_gib": peak_allocated,
        "runtime_cuda_peak_reserved_gib": peak_reserved,
        "runtime_cuda_device_used_gib": device_used,
        "runtime_cuda_non_pytorch_context_estimate_gib": non_pytorch_context,
        "runtime_cuda_peak_process_estimate_gib": peak_reserved + non_pytorch_context,
    }


def _write_gradient_spike(
    logger: JsonlRunLogger,
    report: FiniteGradientSpikeReport,
    *,
    epoch: int,
    batch: int,
    step: int,
) -> None:
    """Persist a finite raw-gradient spike before any clipping mutation."""

    payload = report.as_dict()
    logger.write(
        "gradient_spike",
        epoch=int(epoch),
        batch=int(batch),
        step=int(step),
        **payload,
    )
    print(
        "[mainline-gradient-spike] "
        f"epoch={int(epoch):03d} batch={int(batch):04d} step={int(step)} "
        f"global_preclip={report.gradient_global_preclip_l2:.6g} "
        f"threshold={report.gradient_spike_audit_threshold:.6g} "
        f"max_l2_parameter={report.max_l2.parameter_name} "
        f"max_l2={report.max_l2.l2:.6g} "
        f"max_abs_parameter={report.max_abs.parameter_name} "
        f"max_abs={report.max_abs.max_abs:.6g}",
        flush=True,
    )


def _emit_training_window(
    *,
    logger: JsonlRunLogger,
    config: ExperimentConfig,
    window_metrics: DeviceMetricAccumulator,
    gradient_window: GradientPreclipWindowAccumulator,
    epoch: int,
    batch: int,
    step: int,
    window_seconds: float,
    window_samples: int,
    window_batches: int,
    learning_rate: float,
    boundary: str,
) -> dict[str, float]:
    """Persist one complete periodic or epoch-tail training window."""

    if boundary not in {"periodic", "epoch_tail"}:
        raise ValueError("training-window boundary is invalid")
    if int(window_batches) <= 0 or int(window_samples) <= 0:
        raise ValueError("training-window counts must be positive")
    if not math.isfinite(float(window_seconds)) or float(window_seconds) < 0.0:
        raise ValueError(
            "training-window duration must be finite and non-negative"
        )
    values = archival_metrics(window_metrics.materialize())
    values.update(gradient_window.materialize())
    values["runtime_window_seconds_per_batch"] = float(window_seconds) / float(
        window_batches
    )
    values["runtime_window_samples_per_second"] = float(window_samples) / max(
        float(window_seconds), 1e-8
    )
    values["learning_rate"] = float(learning_rate)
    values["learning_rate_history_proposal"] = float(
        learning_rate
    ) * role_lr_scale("history_proposal", config)
    values["learning_rate_bottom_decoder"] = float(
        learning_rate
    ) * role_lr_scale("bottom_mmdit", config)
    values["learning_rate_bottom_capacity"] = float(
        learning_rate
    ) * role_lr_scale("bottom_capacity", config)
    logger.write(
        "train",
        epoch=int(epoch),
        batch=int(batch),
        step=int(step),
        window_boundary=boundary,
        window_batches=int(window_batches),
        window_samples=int(window_samples),
        metrics=values,
    )
    first_window = int(batch) <= int(config.runtime.log_every)
    epoch_tail = boundary == "epoch_tail"
    health_stride = int(config.runtime.log_every) * 5
    detail_stride = int(config.runtime.log_every) * 10
    health_due = first_window or epoch_tail or int(batch) % health_stride == 0
    detail_due = first_window or epoch_tail or int(batch) % detail_stride == 0
    # The JSONL cadence remains unchanged.  Console is a decision surface: a
    # small task/gradient/runtime row every five archival windows and a source
    # localization panel every ten.  Non-finite gradients and finite spikes
    # still print synchronously at the exact offending batch.
    if health_due:
        print(
            logger.compact_line(
                "train",
                epoch=int(epoch),
                batch=int(batch),
                step=int(step),
                metrics=values,
            )
            + f" window_boundary={boundary}",
            flush=True,
        )
    if detail_due:
        for detail_line in logger.diagnostic_lines(
            "train",
            epoch=int(epoch),
            batch=int(batch),
            step=int(step),
            metrics=values,
        ):
            print(detail_line, flush=True)
    return values


def _data_state(bundle: MainlineDataBundle) -> dict[str, object]:
    return {
        "splits": {name: list(ids) for name, ids in bundle.splits.items()},
        "action_normalizer": bundle.action_normalizer.to_dict(),
        "state_normalizer": bundle.state_normalizer.to_dict(),
        "goal": bundle.goal.metadata,
    }


def _optimizer_group_context(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> dict[str, dict[str, float | int]]:
    """Serialize optimizer geometry without ambiguous LR/count names."""

    base_learning_rate = float(config.optimizer.learning_rate)
    result: dict[str, dict[str, float | int]] = {}
    for group in optimizer.param_groups:
        learning_rate = float(group["lr"])
        parameters = list(group["params"])
        result[str(group["name"])] = {
            "base_learning_rate": base_learning_rate,
            "initial_learning_rate": learning_rate,
            "role_learning_rate_scale": learning_rate / base_learning_rate,
            "weight_decay": float(group["weight_decay"]),
            "parameter_tensor_count": len(parameters),
            "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        }
    return result


def _module_parameter_context(
    model: ClearVLAMainlinePolicy,
) -> dict[str, dict[str, int]]:
    """Serialize visible module counts so graph changes cannot be silent."""

    modules = {
        "complete_model": model,
        "observation": model.observation,
        "grounding_g1_g2_g3": model.top.grounding_blocks,
        "global_object_grounder": model.top.grounder,
        "stateless_intent": model.top.intent,
        "future_dynamics_w1_w2": model.top.dynamics,
        "factual_precision_p1": model.factual_reader,
        "future_effect_p2": model.top.effect_reader,
        "policy_compiler_p3": model.top.plan_compiler,
        "controlled_transition": model.transition,
        "retained_bottom": model.bottom,
        "retained_bottom_decoder": model.bottom.decoder,
    }
    result: dict[str, dict[str, int]] = {}
    for name, module in modules.items():
        parameters = tuple(module.parameters())
        result[name] = {
            "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
            "trainable_parameter_count": sum(
                int(parameter.numel())
                for parameter in parameters
                if parameter.requires_grad
            ),
            "parameter_tensor_count": len(parameters),
        }
    return result


def _prepare_output_directory(
    output_dir: Path,
    *,
    exact_resume: bool,
) -> None:
    """Reserve one output directory for one semantic experiment.

    Fresh and bottom-migration runs must never append to an earlier metrics
    stream or overwrite its checkpoints.  Exact resume may reuse an existing
    directory only when it contains the serialized run context that is
    checked after the current data/source identity has been built.
    """

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"mainline output path is not a directory: {output_dir}")
    entries = tuple(output_dir.iterdir()) if output_dir.is_dir() else ()
    if entries and not exact_resume:
        raise ValueError(
            "fresh mainline and bottom-migration runs require an empty output "
            f"directory: {output_dir}"
        )
    if entries and exact_resume and not (output_dir / "run_context.json").is_file():
        raise ValueError("exact resume found a non-empty output directory without run_context.json")
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_resume_output_identity(
    output_dir: Path,
    identity: CheckpointIdentity,
) -> None:
    """Reject appending a valid resume to an unrelated metrics stream."""

    context_path = output_dir / "run_context.json"
    if not context_path.exists():
        return
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("exact resume output context is unreadable") from error
    if not isinstance(context, dict) or not isinstance(context.get("identity"), dict):
        raise ValueError("exact resume output context has no checkpoint identity")
    try:
        existing = checkpoint_identity_from_mapping(context["identity"])
    except (TypeError, ValueError) as error:
        raise ValueError("exact resume output context identity is invalid") from error
    comparison = compare_checkpoint_identity(existing, identity)
    if not comparison.exact_resume:
        raise ValueError(
            "exact resume output directory belongs to another experiment: "
            + "; ".join(comparison.reasons)
        )


def _preflight(
    *,
    engine: MainlineTrainingEngine,
    raw_batch: dict[str, torch.Tensor],
    bundle: MainlineDataBundle,
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    flow_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> None:
    batch = to_training_batch(
        raw_batch,
        goal=bundle.goal,
        config=config,
        device=device,
    )
    validate_finite_training_batch(batch)
    encoded = engine.encode_eval(batch, collect_diagnostics=True)
    result = engine.eval_step(
        batch,
        collect_diagnostics=True,
        generator=flow_generator,
        encoded=encoded,
    )
    cache = encoded.cache
    del encoded
    sample = sample_refined_cached_action(
        engine.model,
        cache,
        config,
        collect_diagnostics=True,
        dtype=dtype,
        generator=sampling_generator,
    )
    if not bool(torch.isfinite(result.loss)) or not bool(torch.isfinite(sample.action).all()):
        raise FloatingPointError("mainline preflight produced non-finite values")
    print(
        "[mainline-preflight] "
        f"capability=object_intent_dynamics_323 dtype={str(dtype).removeprefix('torch.')} "
        f"steps={config.runtime.inference_steps} teacher=isolated optimizer=complete",
        flush=True,
    )


def _diagnostic_batch_indices(*, planned_batches: int, budget: int) -> set[int]:
    """Spread a bounded diagnostic budget over the complete validation run."""

    if planned_batches < 0 or budget < 0:
        raise ValueError("validation batch counts and diagnostic budgets must be non-negative")
    if planned_batches == 0:
        return set()
    if budget == 0 or budget >= planned_batches:
        return set(range(1, planned_batches + 1))
    if budget == 1:
        return {1 + (planned_batches - 1) // 2}
    return {
        1 + round(index * (planned_batches - 1) / float(budget - 1))
        for index in range(budget)
    }


CORE_ATTRIBUTION_MODES = (
    "explicit_none",
    "world_dynamic_neutral",
    "consequence_effect_neutral",
    "controlled_transition_delta_neutral",
    "world_and_controlled_transition_neutral",
    "wrong_action_world",
)


def _maximum_identity_error(*pairs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    errors = [
        (left.detach().float() - right.detach().float()).abs().amax()
        for left, right in pairs
    ]
    if not errors:
        raise ValueError("identity accounting requires at least one tensor pair")
    return torch.stack(errors).amax()


def _world_dynamic_neutral_cache(
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
) -> tuple[OnlinePolicyCache, dict[str, torch.Tensor]]:
    """Neutralize only W-predicted dynamics while retaining current facts."""

    cache.validate(config)
    world = cache.top.candidate_world
    dynamics = world.dynamics
    neutral_dynamics = replace(
        dynamics,
        successor_content=dynamics.current_reference[:, None].expand_as(
            dynamics.successor_content
        ),
        semantic_delta=torch.zeros_like(dynamics.semantic_delta),
        transport_mean=torch.zeros_like(dynamics.transport_mean),
        transport_covariance=torch.zeros_like(dynamics.transport_covariance),
    )
    neutral_world = replace(world, dynamics=neutral_dynamics)
    neutral_cache = replace(
        cache,
        top=replace(cache.top, candidate_world=neutral_world),
    )
    neutral_cache.validate(config)
    metrics = {
        "first_boundary_successor_delta_rms": (
            dynamics.successor_content.detach().float()
            - neutral_dynamics.successor_content.detach().float()
        )
        .square()
        .mean()
        .sqrt(),
        "first_boundary_semantic_delta_rms": dynamics.semantic_delta.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "first_boundary_transport_delta_rms": dynamics.transport_mean.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "first_boundary_covariance_delta_rms": dynamics.transport_covariance.detach()
        .float()
        .square()
        .mean()
        .sqrt(),
        "retained_current_reference_identity_max_abs": _maximum_identity_error(
            (neutral_dynamics.current_reference, dynamics.current_reference),
        ),
        "retained_support_identity_max_abs": _maximum_identity_error(
            (neutral_dynamics.chart_availability, dynamics.chart_availability),
            (
                neutral_dynamics.log_chart_availability,
                dynamics.log_chart_availability,
            ),
            (
                neutral_dynamics.camera_chart_availability,
                dynamics.camera_chart_availability,
            ),
            (
                neutral_dynamics.log_camera_chart_availability,
                dynamics.log_camera_chart_availability,
            ),
        ),
        "retained_camera_identity_max_abs": _maximum_identity_error(
            (neutral_dynamics.camera_coordinates, dynamics.camera_coordinates),
        ),
        "retained_action_condition_identity_max_abs": _maximum_identity_error(
            (
                neutral_world.action_condition.fingerprint,
                world.action_condition.fingerprint,
            ),
        ),
    }
    return neutral_cache, metrics


def _wrong_action_world_cache(
    model: ClearVLAMainlinePolicy,
    cache: OnlinePolicyCache,
    config: ExperimentConfig,
    *,
    dtype: torch.dtype,
) -> tuple[OnlinePolicyCache, dict[str, torch.Tensor]]:
    """Rebuild W for a deterministic in-batch donor action only."""

    cache.validate(config)
    primary_world = cache.top.candidate_world
    primary_condition = primary_world.action_condition
    batch = primary_condition.batch
    shift = batch // 2 if batch >= 2 else 0
    donor_interval_action = primary_condition.interval_action.roll(
        shifts=shift,
        dims=0,
    )
    donor_delta = (
        donor_interval_action.detach().float()
        - primary_condition.interval_action.detach().float()
    )
    donor_valid = donor_delta.abs().amax(dim=(1, 2)) > 0.0
    # The current action anchor belongs to the receiving sample.  Only the four
    # proposed interval actions are donated; delta is reconstructed from that
    # retained anchor by the canonical physical-action ABI.
    wrong_condition = PhysicalActionCondition.from_interval_action(
        donor_interval_action,
        primary_condition.current_action,
    )
    device = cache.history.state.device
    autocast_enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        wrong_world, _ = model.top.build_candidate_world(
            belief=cache.top.belief,
            action_condition=wrong_condition,
            collect_diagnostics=False,
        )
    wrong_cache = replace(
        cache,
        top=replace(cache.top, candidate_world=wrong_world),
    )
    wrong_cache.validate(config)
    primary_dynamics = primary_world.dynamics
    wrong_dynamics = wrong_world.dynamics
    metrics = {
        "donor_valid_rows": donor_valid.detach().float().sum(),
        "donor_total_rows": donor_valid.new_tensor(float(batch), dtype=torch.float32),
        "donor_valid_fraction": donor_valid.detach().float().mean(),
        "donor_valid_batches": donor_valid.new_tensor(
            float(bool(donor_valid.any())), dtype=torch.float32
        ),
        "first_boundary_action_condition_delta_rms": (
            wrong_condition.fingerprint.detach().float()
            - primary_condition.fingerprint.detach().float()
        )
        .square()
        .mean()
        .sqrt(),
        "first_boundary_semantic_delta_rms": (
            wrong_dynamics.semantic_delta.detach().float()
            - primary_dynamics.semantic_delta.detach().float()
        )
        .square()
        .mean()
        .sqrt(),
        "first_boundary_transport_delta_rms": (
            wrong_dynamics.transport_mean.detach().float()
            - primary_dynamics.transport_mean.detach().float()
        )
        .square()
        .mean()
        .sqrt(),
        "retained_current_action_identity_max_abs": _maximum_identity_error(
            (wrong_condition.current_action, primary_condition.current_action),
        ),
        "retained_current_reference_identity_max_abs": _maximum_identity_error(
            (wrong_dynamics.current_reference, primary_dynamics.current_reference),
        ),
        "retained_support_identity_max_abs": _maximum_identity_error(
            (wrong_dynamics.chart_availability, primary_dynamics.chart_availability),
            (
                wrong_dynamics.log_chart_availability,
                primary_dynamics.log_chart_availability,
            ),
            (
                wrong_dynamics.camera_chart_availability,
                primary_dynamics.camera_chart_availability,
            ),
            (
                wrong_dynamics.log_camera_chart_availability,
                primary_dynamics.log_camera_chart_availability,
            ),
        ),
        "retained_camera_identity_max_abs": _maximum_identity_error(
            (wrong_dynamics.camera_coordinates, primary_dynamics.camera_coordinates),
        ),
    }
    return wrong_cache, metrics


@torch.no_grad()
def _validate(
    *,
    engine: MainlineTrainingEngine,
    loader,
    bundle: MainlineDataBundle,
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    deployment = ValidationAccumulator.from_action_normalizer(
        bundle.action_normalizer,
        device=device,
        gripper_event_threshold=config.objectives.gripper_event_threshold,
        arm_motion_threshold=config.objectives.arm_motion_threshold,
    )
    losses = DeviceMetricAccumulator()
    p2_interventions = MatchedP2InterventionAccumulator.from_action_normalizer(
        bundle.action_normalizer,
        device=device,
        gripper_event_threshold=config.objectives.gripper_event_threshold,
        arm_motion_threshold=config.objectives.arm_motion_threshold,
    )
    core_attribution = MatchedCoreAttributionAccumulator.from_action_normalizer(
        bundle.action_normalizer,
        device=device,
        gripper_event_threshold=config.objectives.gripper_event_threshold,
        arm_motion_threshold=config.objectives.arm_motion_threshold,
    )
    proposal_ablations = DeviceMetricAccumulator()
    execution_ablations = DeviceMetricAccumulator()
    maximum = config.runtime.max_val_batches
    completed_batches = 0
    sampling_diagnostic_batches = 0
    p2_intervention_batches = 0
    core_attribution_batches = 0
    proposal_ablation_batches = 0
    execution_ablation_batches = 0
    action_scale = torch.as_tensor(
        bundle.action_normalizer.scale,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    if bool((action_scale <= 0.0).any()):
        raise ValueError("validation action normalizer scale must be positive")
    flow_generator = _owned_generator(device, config.data.seed + 10_001)
    planned_batches = len(loader)
    if maximum > 0:
        planned_batches = min(planned_batches, int(maximum))
    sampling_diagnostic_indices = _diagnostic_batch_indices(
        planned_batches=planned_batches,
        budget=config.runtime.eval_sampling_diagnostic_batches,
    )
    proposal_ablation_indices = _diagnostic_batch_indices(
        planned_batches=planned_batches,
        budget=config.runtime.eval_proposal_ablation_batches,
    )
    execution_ablation_indices = _diagnostic_batch_indices(
        planned_batches=planned_batches,
        budget=config.runtime.eval_execution_ablation_batches,
    )
    for batch_index, raw_batch in enumerate(loader, start=1):
        if maximum > 0 and batch_index > maximum:
            break
        completed_batches += 1
        batch = to_training_batch(
            raw_batch,
            goal=bundle.goal,
            config=config,
            device=device,
        )
        diagnostics = batch_index in sampling_diagnostic_indices
        run_proposal_ablation = batch_index in proposal_ablation_indices
        run_execution_ablation = batch_index in execution_ablation_indices
        encoded = engine.encode_eval(batch, collect_diagnostics=diagnostics)
        loss_result = engine.eval_step(
            batch,
            collect_diagnostics=diagnostics,
            generator=flow_generator,
            encoded=encoded,
        )
        cache = encoded.cache
        proposal_ablation_cache = None
        if run_proposal_ablation:
            autocast_enabled = device.type in {"cuda", "cpu"} and dtype in {
                torch.bfloat16,
                torch.float16,
            }
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                proposal_ablation_cache = engine.model.proposal_ablation_cache(
                    cache,
                    encoded.training_state,
                )
        del encoded
        # P2 band/type matrices are defined at the last real deployment ODE
        # update.  The validation-loss forward uses a sampled flow time, so do
        # not archive that second, non-comparable posterior under an unprefixed
        # name.  The matched deployment result is collected below.
        loss_metrics = {
            name: value
            for name, value in loss_result.metrics.items()
            if not (
                name.startswith(("object_p2_semantic_", "object_p2_geometry_"))
                and (
                    "_band_" in name
                    or name.endswith("_temporal_support_fraction")
                )
            )
        }
        losses.update(
            {
                "loss_total": loss_result.loss,
                **loss_metrics,
            },
            weight=batch.online.batch,
        )
        prediction, refined_cache = sample_refined_cached_action_with_cache(
            engine.model,
            cache,
            config,
            collect_diagnostics=diagnostics,
            dtype=dtype,
            # V120 owns one validation-noise stream per batch.  This makes a
            # batch's deployment sample independent of diagnostic budgets and
            # preserves matched cross-version/action-ablation comparisons.
            generator=_owned_generator(device, 37_237 + batch_index),
        )
        target_physical = engine.model.action_codec.encode(
            batch.action_target.normalized,
            batch.online.history.action_state,
        )
        motion_target = (
            engine.model.action_codec.split(target_physical).arm_delta.float().norm(dim=-1)
            >= float(config.objectives.arm_motion_threshold)
        )
        deployment.update(
            prediction.action,
            batch,
            motion_logits=prediction.motion_logits,
            motion_target=motion_target,
            physical_field=prediction.physical_field,
            gripper_decode_delta_blend=(
                engine.model.action_codec.decode_delta_blend
            ),
        )
        if diagnostics:
            sampling_diagnostic_batches += 1
            losses.update(
                {
                    f"validation_deploy_{name}": value
                    for name, value in prediction.metrics.items()
                },
                weight=batch.online.batch,
            )
            p2_intervention_batches += 1
            reader = engine.model.top.effect_reader
            for mode in reader.INTERVENTION_MODES:
                reader.set_eval_intervention(mode)
                try:
                    counterfactual = sample_cached_action(
                        engine.model,
                        refined_cache,
                        config,
                        initial_physical_noise=prediction.initial_physical_noise,
                        collect_diagnostics=False,
                        dtype=dtype,
                    )
                finally:
                    reader.clear_eval_intervention()
                p2_interventions.update(
                    mode,
                    primary_action=prediction.action,
                    counterfactual_action=counterfactual.action,
                    batch=batch,
                )
                del counterfactual
            core_attribution_batches += 1
            core_attribution.update_primary(prediction.action, batch)
            neutral_world_cache, neutral_world_boundary = (
                _world_dynamic_neutral_cache(refined_cache, config)
            )
            wrong_world_cache, wrong_world_boundary = _wrong_action_world_cache(
                engine.model,
                refined_cache,
                config,
                dtype=dtype,
            )
            consequence_module = engine.model.top.consequence
            transition_module = engine.model.transition
            identity_actions: dict[str, torch.Tensor] = {}
            for mode in CORE_ATTRIBUTION_MODES:
                counterfactual_cache = refined_cache
                boundary_metrics: dict[str, torch.Tensor] = {}
                use_consequence_neutral = mode == "consequence_effect_neutral"
                use_transition_neutral = mode in {
                    "controlled_transition_delta_neutral",
                    "world_and_controlled_transition_neutral",
                }
                if mode in {
                    "world_dynamic_neutral",
                    "world_and_controlled_transition_neutral",
                }:
                    counterfactual_cache = neutral_world_cache
                    boundary_metrics.update(
                        {
                            f"world_{name}": value
                            for name, value in neutral_world_boundary.items()
                        }
                    )
                elif mode == "wrong_action_world":
                    counterfactual_cache = wrong_world_cache
                    boundary_metrics.update(wrong_world_boundary)
                elif mode == "explicit_none":
                    boundary_metrics["intervention_active"] = prediction.action.new_zeros(
                        (), dtype=torch.float32
                    )
                try:
                    if use_consequence_neutral:
                        consequence_module.set_eval_intervention("effect_neutral")
                    if use_transition_neutral:
                        transition_module.set_eval_intervention("delta_neutral")
                    counterfactual = sample_cached_action(
                        engine.model,
                        counterfactual_cache,
                        config,
                        initial_physical_noise=prediction.initial_physical_noise,
                        # Boundary diagnostics execute only on the final real
                        # ODE update and prove that each named intervention
                        # changed its intended first consumer.
                        collect_diagnostics=True,
                        dtype=dtype,
                    )
                finally:
                    # Clear both modules even when setting the second mode or
                    # sampling raises; no intervention may leak into the next
                    # mode, validation batch or ordinary deployment call.
                    consequence_module.clear_eval_intervention()
                    transition_module.clear_eval_intervention()
                if use_consequence_neutral:
                    boundary_metrics.update(
                        {
                            "first_boundary_delta_rms": counterfactual.metrics[
                                "object_consequence_intervention_first_boundary_delta_rms"
                            ],
                            "effect_delta_rms": counterfactual.metrics[
                                "object_consequence_intervention_effect_delta_rms"
                            ],
                            "interaction_delta_rms": counterfactual.metrics[
                                "object_consequence_intervention_interaction_delta_rms"
                            ],
                            "retained_factual_identity_max_abs": counterfactual.metrics[
                                "object_consequence_intervention_factual_identity_max_abs"
                            ],
                            "intervention_active": counterfactual.metrics[
                                "object_consequence_intervention_active"
                            ],
                        }
                    )
                if use_transition_neutral:
                    boundary_metrics.update(
                        {
                            "controlled_transition_first_boundary_delta_rms": (
                                counterfactual.metrics[
                                    "controlled_transition_intervention_first_boundary_delta_rms"
                                ]
                            ),
                            "controlled_transition_action_neutral_identity_max_abs": (
                                counterfactual.metrics[
                                    "controlled_transition_intervention_action_neutral_identity_max_abs"
                                ]
                            ),
                            "controlled_transition_selector_identity_max_abs": (
                                counterfactual.metrics[
                                    "controlled_transition_intervention_selector_identity_max_abs"
                                ]
                            ),
                            "controlled_transition_network_executed": (
                                counterfactual.metrics[
                                    "controlled_transition_intervention_network_executed"
                                ]
                            ),
                            "controlled_transition_intervention_active": (
                                counterfactual.metrics[
                                    "controlled_transition_intervention_active"
                                ]
                            ),
                        }
                    )
                core_attribution.update(
                    mode,
                    primary_action=prediction.action,
                    counterfactual_action=counterfactual.action,
                    batch=batch,
                    boundary_metrics=boundary_metrics,
                )
                if mode in {
                    "explicit_none",
                    "world_dynamic_neutral",
                    "consequence_effect_neutral",
                }:
                    identity_actions[mode] = counterfactual.action
                del counterfactual
            core_attribution.update_identity(
                "primary_vs_explicit_none",
                prediction.action,
                identity_actions["explicit_none"],
            )
            core_attribution.update_identity(
                "world_vs_consequence_neutral",
                identity_actions["world_dynamic_neutral"],
                identity_actions["consequence_effect_neutral"],
            )
            del neutral_world_cache, wrong_world_cache, identity_actions
        if run_proposal_ablation or run_execution_ablation:
            common_sampling = {
                "initial_physical_noise": prediction.initial_physical_noise,
                "collect_diagnostics": False,
                "dtype": dtype,
            }
            target = batch.action_target.normalized.float()
            primary = prediction.action.float()
            primary_error = primary - target
        if run_proposal_ablation:
            if proposal_ablation_cache is None:
                raise RuntimeError("proposal ablation cache was not constructed")
            proposal_ablation_batches += 1
            # The recovered proposal ablation is an intentional no-op, but it
            # must still execute the same proposal -> W -> refined-action
            # schedule as primary.  A single-pass counterfactual here would
            # measure removal of Schema28's outer refinement instead of the
            # proposal boundary named by this diagnostic.
            proposal_zero = sample_refined_cached_action(
                engine.model,
                proposal_ablation_cache,
                config,
                **common_sampling,
            )
            proposal_error = proposal_zero.action.float() - target
            proposal_delta = proposal_zero.action.float() - primary
            proposal_ablations.update(
                {
                    "proposal_primary_mse_normalized": primary_error.square().mean(),
                    "proposal_primary_mse_physical": (
                        primary_error / action_scale
                    ).square().mean(),
                    "proposal_zero_mse_normalized": proposal_error.square().mean(),
                    "proposal_zero_mse_physical": (
                        proposal_error / action_scale
                    ).square().mean(),
                    "proposal_zero_action_delta_mse_normalized": (
                        proposal_delta.square().mean()
                    ),
                    "proposal_zero_action_delta_mse_physical": (
                        proposal_delta / action_scale
                    ).square().mean(),
                },
                weight=batch.online.batch,
            )
            del proposal_ablation_cache, proposal_zero
        if run_execution_ablation:
            execution_ablation_batches += 1
            execution_rows: dict[str, torch.Tensor] = {
                "execution_primary_mse_normalized": primary_error.square().mean(),
                "execution_primary_mse_physical": (
                    primary_error / action_scale
                ).square().mean(),
            }
            for mode in (
                "hard",
                "neutral",
                "full_capacity",
                "three_basis_reduction",
            ):
                execution = sample_cached_action(
                    engine.model,
                    refined_cache,
                    config,
                    execution_mode=mode,
                    **common_sampling,
                )
                error = execution.action.float() - target
                delta = execution.action.float() - primary
                stem = f"execution_{mode}"
                execution_rows[f"{stem}_mse_normalized"] = error.square().mean()
                execution_rows[f"{stem}_mse_physical"] = (
                    error / action_scale
                ).square().mean()
                execution_rows[f"{stem}_action_delta_mse_normalized"] = (
                    delta.square().mean()
                )
                execution_rows[f"{stem}_action_delta_mse_physical"] = (
                    delta / action_scale
                ).square().mean()
                del execution
            execution_ablations.update(
                execution_rows,
                weight=batch.online.batch,
            )
    result = {**losses.materialize(), **deployment.means()}
    if p2_intervention_batches:
        result.update(p2_interventions.means())
    if core_attribution_batches:
        result.update(core_attribution.means())
    result["validation_sampling_diagnostic_batches"] = float(
        sampling_diagnostic_batches
    )
    result["validation_sampling_diagnostic_coverage"] = float(
        sampling_diagnostic_batches / max(completed_batches, 1)
    )
    result["validation_p2_intervention_batches"] = float(
        p2_intervention_batches
    )
    result["validation_p2_intervention_coverage"] = float(
        p2_intervention_batches / max(completed_batches, 1)
    )
    result["validation_core_attribution_batches"] = float(core_attribution_batches)
    result["validation_core_attribution_coverage"] = float(
        core_attribution_batches / max(completed_batches, 1)
    )
    result["validation_proposal_ablation_batches"] = float(
        proposal_ablation_batches
    )
    result["validation_proposal_ablation_coverage"] = float(
        proposal_ablation_batches / max(completed_batches, 1)
    )
    result["validation_execution_ablation_batches"] = float(
        execution_ablation_batches
    )
    result["validation_execution_ablation_coverage"] = float(
        execution_ablation_batches / max(completed_batches, 1)
    )
    if proposal_ablation_batches:
        rows = proposal_ablations.materialize()
        primary_normalized = rows["proposal_primary_mse_normalized"]
        primary_physical = rows["proposal_primary_mse_physical"]
        result["validation_proposal_primary_rmse_normalized"] = float(
            primary_normalized**0.5
        )
        result["validation_proposal_primary_rmse_physical"] = float(
            primary_physical**0.5
        )
        normalized = rows["proposal_zero_mse_normalized"]
        physical = rows["proposal_zero_mse_physical"]
        result["validation_proposal_zero_rmse_normalized"] = float(normalized**0.5)
        result["validation_proposal_zero_rmse_physical"] = float(physical**0.5)
        result["validation_proposal_zero_mse_gain_vs_primary_normalized"] = float(
            primary_normalized - normalized
        )
        result["validation_proposal_zero_mse_gain_vs_primary_physical"] = float(
            primary_physical - physical
        )
        result["validation_proposal_zero_action_delta_rmse_normalized"] = float(
            rows["proposal_zero_action_delta_mse_normalized"] ** 0.5
        )
        result["validation_proposal_zero_action_delta_rmse_physical"] = float(
            rows["proposal_zero_action_delta_mse_physical"] ** 0.5
        )
    if execution_ablation_batches:
        rows = execution_ablations.materialize()
        primary_normalized = rows["execution_primary_mse_normalized"]
        primary_physical = rows["execution_primary_mse_physical"]
        result["validation_execution_primary_rmse_normalized"] = float(
            primary_normalized**0.5
        )
        result["validation_execution_primary_rmse_physical"] = float(
            primary_physical**0.5
        )
        for mode in (
            "hard",
            "neutral",
            "full_capacity",
            "three_basis_reduction",
        ):
            name = f"execution_{mode}"
            normalized = rows[f"{name}_mse_normalized"]
            physical = rows[f"{name}_mse_physical"]
            result[f"validation_{name}_rmse_normalized"] = float(normalized**0.5)
            result[f"validation_{name}_rmse_physical"] = float(physical**0.5)
            result[f"validation_{name}_mse_gain_vs_primary_normalized"] = float(
                primary_normalized - normalized
            )
            result[f"validation_{name}_mse_gain_vs_primary_physical"] = float(
                primary_physical - physical
            )
            result[f"validation_{name}_action_delta_rmse_normalized"] = float(
                rows[f"{name}_action_delta_mse_normalized"] ** 0.5
            )
            result[f"validation_{name}_action_delta_rmse_physical"] = float(
                rows[f"{name}_action_delta_mse_physical"] ** 0.5
            )
    return result


def main() -> None:
    args = _parser().parse_args()
    validation_checkpoint = args.validate_checkpoint
    validation_checkpoint_resolved: str | None = None
    config = _overrides(load_config(args.config), args)
    _seed(config.data.seed)
    device = _device(args.device)
    dtype = resolve_compute_dtype(config)
    output_dir = Path(config.data.output_dir)
    _prepare_output_directory(output_dir, exact_resume=args.resume is not None)
    bundle = load_mainline_data(config, allow_null_goal=bool(args.allow_null_goal))
    train_loader_generator = torch.Generator().manual_seed(config.data.seed + 101)
    train_flow_generator = _owned_generator(device, config.data.seed + 102)
    train_condition_generator = _owned_generator(device, config.data.seed + 103)
    train_loader = bundle.loader(
        "train",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        generator=train_loader_generator,
    )
    val_loader = bundle.loader(
        "val",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        shuffle=False,
    )
    model = ClearVLAMainlinePolicy(config).to(device)
    optimizer, ownership = build_optimizer(model, config)
    steps_per_epoch = _limit(len(train_loader), config.runtime.max_train_batches)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=max(config.optimizer.epochs * steps_per_epoch, 1),
        minimum_ratio=config.optimizer.min_lr_ratio,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        dtype=dtype,
        train_flow_generator=train_flow_generator,
        train_condition_generator=train_condition_generator,
    )
    identity = build_checkpoint_identity(
        config,
        repo_root=Path.cwd(),
        dataset=dataset_identity(bundle, config),
        language=language_identity(bundle, config),
    )
    if args.resume is not None:
        _validate_resume_output_identity(output_dir, identity)
    start_epoch = 1
    best_metric: float | None = None
    validation_state = None
    if args.resume is not None:
        restored = load_checkpoint_exact(
            args.resume,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            identity=identity,
            generators={
                "train_loader": train_loader_generator,
                "train_flow": train_flow_generator,
                "train_condition": train_condition_generator,
            },
        )
        start_epoch = restored.epoch + 1
        engine.global_step = restored.global_step
        best_metric = restored.best_metric
        validate_resume_metric_boundary(
            output_dir,
            checkpoint_epoch=restored.epoch,
            checkpoint_step=restored.global_step,
        )
    elif validation_checkpoint is not None:
        validation_state = load_checkpoint_for_validation(
            validation_checkpoint,
            model=model,
            config=config,
            identity=identity,
        )
        validation_checkpoint_resolved = str(Path(validation_checkpoint).resolve())
        engine.global_step = validation_state.global_step
    elif args.migrate_bottom is not None:
        report = migrate_bottom_only(args.migrate_bottom, model, identity=identity)
        print(
            "[mainline-migration] "
            f"loaded={len(report.loaded)} missing={len(report.missing)} "
            f"shape_mismatch={len(report.shape_mismatch)} rejected={len(report.rejected)}",
            flush=True,
        )
    context = {
        "config": config.as_dict(),
        "identity": identity.as_dict(),
        "optimizer_roles": ownership.role_counts,
        "optimizer_groups": _optimizer_group_context(optimizer, config),
        "module_parameters": _module_parameter_context(model),
        "dataset_sizes": {name: len(value) for name, value in bundle.datasets.items()},
        "splits": {name: list(value) for name, value in bundle.splits.items()},
        "skipped": list(bundle.skipped),
        "information_sampling": getattr(
            getattr(train_loader, "batch_sampler", None), "summary", None
        ),
        "normalizer_fingerprints": {
            "action_v120": v120_normalizer_fingerprint(bundle.action_normalizer),
            "state_v120": v120_normalizer_fingerprint(bundle.state_normalizer),
            "action_sha256": identity.dataset.action_normalizer_sha256,
            "state_sha256": identity.dataset.state_normalizer_sha256,
        },
        "gradient_audit": {
            "finite_spike_preclip_l2_threshold": (
                engine.gradient_spike_audit_threshold
            ),
            "parameter_scan_policy": (
                "only_after_finite_global_threshold_crossing"
            ),
        },
        "execution_mode": (
            "validation_only" if validation_state is not None else "training"
        ),
    }
    if validation_state is not None:
        if validation_checkpoint_resolved is None:
            raise RuntimeError("validation checkpoint provenance was not resolved")
        context["validation_checkpoint"] = {
            "path": validation_checkpoint_resolved,
            "epoch": validation_state.epoch,
            "global_step": validation_state.global_step,
            "best_metric": validation_state.best_metric,
            "saved_source_digest": validation_state.saved_source_digest,
            "current_source_digest": validation_state.current_source_digest,
            "changed_source_files": list(validation_state.changed_source_files),
            "optimizer_loaded": False,
            "schedule_loaded": False,
            "rng_loaded": False,
            "checkpoint_writes_enabled": False,
        }
    # Preflight uses the deterministic validation sampler and separate RNGs;
    # it must not consume formal training shuffle, condition-dropout or flow
    # randomness.
    # A fresh directory stays empty until this boundary succeeds, so a failed
    # source/data/numerics preflight can be rerun without either appending to a
    # partial metrics stream or manually deleting an experiment-shaped stub.
    first_raw = next(iter(val_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _preflight(
        engine=engine,
        raw_batch=first_raw,
        bundle=bundle,
        config=config,
        device=device,
        dtype=dtype,
        flow_generator=_owned_generator(device, config.data.seed + 20_001),
        sampling_generator=_owned_generator(device, config.data.seed + 20_002),
    )
    preflight_memory = _cuda_memory_metrics(device)
    if preflight_memory:
        print(
            "[mainline-memory] "
            + " ".join(f"{name}={value:.3f}" for name, value in preflight_memory.items()),
            flush=True,
        )
    (output_dir / "run_context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger = JsonlRunLogger(output_dir)
    if validation_state is not None:
        if validation_checkpoint_resolved is None:
            raise RuntimeError("validation checkpoint provenance was not resolved")
        validation_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation = archival_metrics(
            _validate(
                engine=engine,
                loader=val_loader,
                bundle=bundle,
                config=config,
                device=device,
                dtype=dtype,
            )
        )
        runtime = archival_metrics(
            {
                "runtime_validation_seconds": time.perf_counter() - validation_started,
                **_cuda_memory_metrics(device),
            }
        )
        logger.write(
            "epoch",
            epoch=validation_state.epoch,
            step=validation_state.global_step,
            mode="validation_only",
            source_checkpoint=validation_checkpoint_resolved,
            train={},
            validation=validation,
            runtime=runtime,
        )
        planned_batches = _limit(len(val_loader), config.runtime.max_val_batches)
        print(
            "[mainline-validation-only] "
            f"checkpoint_epoch={validation_state.epoch:03d} "
            f"step={validation_state.global_step} batches={planned_batches} "
            f"source_delta_files={len(validation_state.changed_source_files)} "
            f"runtime_validation_seconds={runtime['runtime_validation_seconds']:.6g} "
            "optimizer_load=disabled schedule_load=disabled rng_load=disabled "
            "checkpoint_write=disabled",
            flush=True,
        )
        print(
            logger.compact_line(
                "val",
                epoch=validation_state.epoch,
                batch=None,
                step=validation_state.global_step,
                metrics=validation,
            ),
            flush=True,
        )
        for detail_line in logger.diagnostic_lines(
            "val",
            epoch=validation_state.epoch,
            batch=None,
            step=validation_state.global_step,
            metrics=validation,
        ):
            print(detail_line, flush=True)
        return
    data_state = _data_state(bundle)
    for epoch in range(start_epoch, config.optimizer.epochs + 1):
        train_batch_sampler = getattr(train_loader, "batch_sampler", None)
        set_epoch = getattr(train_batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        epoch_started = time.perf_counter()
        window_started = epoch_started
        epoch_samples = 0
        epoch_batches = 0
        window_samples = 0
        window_batches = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_metrics = DeviceMetricAccumulator()
        window_metrics = DeviceMetricAccumulator()
        gradient_window = GradientPreclipWindowAccumulator()
        last_learning_rate: float | None = None
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            if (
                config.runtime.max_train_batches > 0
                and batch_index > config.runtime.max_train_batches
            ):
                break
            batch = to_training_batch(
                raw_batch,
                goal=bundle.goal,
                config=config,
                device=device,
            )
            emit = batch_index % config.runtime.log_every == 0
            try:
                result = engine.train_step(
                    batch,
                    collect_diagnostics=emit,
                    gradient_spike_handler=lambda report: _write_gradient_spike(
                        logger,
                        report,
                        epoch=epoch,
                        batch=batch_index,
                        step=engine.global_step,
                    ),
                )
            except NonFiniteGradientError as error:
                report = error.report.as_dict()
                logger.write(
                    "gradient_failure",
                    epoch=epoch,
                    batch=batch_index,
                    step=engine.global_step,
                    **report,
                )
                print(
                    "[mainline-gradient-failure] "
                    f"epoch={epoch:03d} batch={batch_index:04d} "
                    f"step={engine.global_step} "
                    f"parameter={report['parameter_name']} "
                    f"role={report['parameter_role']} "
                    f"optimizer_group={report['optimizer_group']} "
                    f"finite_fraction={report['finite_fraction']:.6f} "
                    f"finite_max_abs={report['finite_max_abs']:.6g} "
                    f"global_norm={report['global_norm']}",
                    flush=True,
                )
                raise
            epoch_samples += batch.online.batch
            epoch_batches += 1
            window_samples += batch.online.batch
            window_batches += 1
            last_learning_rate = result.learning_rate
            row = {
                "loss_total": result.loss,
                **result.metrics,
            }
            epoch_metrics.update(
                {
                    **row,
                    "gradient_epoch_preclip_l2_mean": result.gradient_norm,
                },
                weight=batch.online.batch,
            )
            window_metrics.update(row, weight=batch.online.batch)
            if result.gradient_norm_scalar is None:
                raise RuntimeError("training step lost its pre-clip gradient scalar")
            gradient_window.update(
                result.gradient_norm_scalar,
                weight=batch.online.batch,
                batch_offset=window_batches,
                global_step=engine.global_step,
            )
            if emit:
                window_seconds = time.perf_counter() - window_started
                _emit_training_window(
                    logger=logger,
                    config=config,
                    window_metrics=window_metrics,
                    gradient_window=gradient_window,
                    epoch=epoch,
                    batch=batch_index,
                    step=engine.global_step,
                    window_seconds=window_seconds,
                    window_samples=window_samples,
                    window_batches=window_batches,
                    learning_rate=result.learning_rate,
                    boundary="periodic",
                )
                window_metrics = DeviceMetricAccumulator()
                gradient_window = GradientPreclipWindowAccumulator()
                window_started = time.perf_counter()
                window_samples = 0
                window_batches = 0
        if window_batches > 0:
            if last_learning_rate is None:
                raise RuntimeError("non-empty training window has no learning rate")
            _emit_training_window(
                logger=logger,
                config=config,
                window_metrics=window_metrics,
                gradient_window=gradient_window,
                epoch=epoch,
                batch=epoch_batches,
                step=engine.global_step,
                window_seconds=time.perf_counter() - window_started,
                window_samples=window_samples,
                window_batches=window_batches,
                learning_rate=last_learning_rate,
                boundary="epoch_tail",
            )
        train_values = archival_metrics(epoch_metrics.materialize())
        epoch_seconds = time.perf_counter() - epoch_started
        train_values["runtime_epoch_seconds"] = epoch_seconds
        train_values["runtime_seconds_per_batch"] = epoch_seconds / max(epoch_batches, 1)
        train_values["runtime_samples_per_second"] = epoch_samples / max(epoch_seconds, 1e-8)
        validation = archival_metrics(
            _validate(
                engine=engine,
                loader=val_loader,
                bundle=bundle,
                config=config,
                device=device,
                dtype=dtype,
            )
        )
        # Capture the peak after validation as well: a production memory claim
        # covers the complete train/eval epoch, not only the backward path.
        train_values.update(_cuda_memory_metrics(device))
        logger.write(
            "epoch",
            epoch=epoch,
            step=engine.global_step,
            train=train_values,
            validation=validation,
        )
        runtime_names = (
            "runtime_seconds_per_batch",
            "runtime_samples_per_second",
            "runtime_cuda_peak_reserved_gib",
            "runtime_cuda_peak_process_estimate_gib",
        )
        runtime_values = {
            name: train_values[name] for name in runtime_names if name in train_values
        }
        print(
            f"[mainline-runtime] epoch={epoch:03d} step={engine.global_step} "
            + " ".join(f"{name}={value:.6g}" for name, value in runtime_values.items()),
            flush=True,
        )
        print(
            logger.compact_line(
                "val",
                epoch=epoch,
                batch=None,
                step=engine.global_step,
                metrics=validation,
            ),
            flush=True,
        )
        for detail_line in logger.diagnostic_lines(
            "val",
            epoch=epoch,
            batch=None,
            step=engine.global_step,
            metrics=validation,
        ):
            print(detail_line, flush=True)
        metric = validation["validation_action_rmse_normalized"]
        improved = best_metric is None or metric < best_metric
        if improved:
            best_metric = metric
        checkpoint_dir = output_dir / "checkpoints"
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            identity=identity,
            epoch=epoch,
            global_step=engine.global_step,
            best_metric=best_metric,
            data_state=data_state,
            generators={
                "train_loader": train_loader_generator,
                "train_flow": train_flow_generator,
                "train_condition": train_condition_generator,
            },
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                schedule=schedule,
                config=config,
                identity=identity,
                epoch=epoch,
                global_step=engine.global_step,
                best_metric=best_metric,
                data_state=data_state,
                generators={
                    "train_loader": train_loader_generator,
                    "train_flow": train_flow_generator,
                    "train_condition": train_condition_generator,
                },
            )


if __name__ == "__main__":
    main()
