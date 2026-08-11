"""Single training entry point for the capability-named ClearVLA mainline."""

from __future__ import annotations

import argparse
import json
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
from .model.policy import ClearVLAMainlinePolicy
from .runtime.checkpoints import load_checkpoint_exact, migrate_bottom_only, save_checkpoint
from .runtime.evaluation import ValidationAccumulator
from .runtime.identity import (
    dataset_identity,
    language_identity,
    v120_normalizer_fingerprint,
)
from .runtime.logging import (
    DeviceMetricAccumulator,
    JsonlRunLogger,
    active_metrics,
    archival_metrics,
)
from .runtime.numerics import resolve_compute_dtype
from .runtime.sampling import sample_cached_action
from .training.engine import MainlineTrainingEngine, validate_finite_training_batch
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
    if args.resume is not None and args.migrate_bottom is not None:
        raise ValueError("exact resume and bottom-only migration are mutually exclusive")
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
    return {
        "runtime_cuda_allocated_gib": torch.cuda.memory_allocated(index) / gib,
        "runtime_cuda_reserved_gib": torch.cuda.memory_reserved(index) / gib,
        "runtime_cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(index) / gib,
        "runtime_cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(index) / gib,
    }


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
    sample = sample_cached_action(
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
    ablations = DeviceMetricAccumulator()
    maximum = config.runtime.max_val_batches
    completed_batches = 0
    diagnostic_batches = 0
    action_scale = torch.as_tensor(
        bundle.action_normalizer.scale,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    if bool((action_scale <= 0.0).any()):
        raise ValueError("validation action normalizer scale must be positive")
    flow_generator = _owned_generator(device, config.data.seed + 10_001)
    sampling_generator = _owned_generator(device, config.data.seed + 10_002)
    for batch_index, raw_batch in enumerate(loader):
        if maximum > 0 and batch_index >= maximum:
            break
        completed_batches += 1
        batch = to_training_batch(
            raw_batch,
            goal=bundle.goal,
            config=config,
            device=device,
        )
        diagnostics = batch_index < config.runtime.eval_diagnostic_batches
        encoded = engine.encode_eval(batch, collect_diagnostics=diagnostics)
        loss_result = engine.eval_step(
            batch,
            collect_diagnostics=diagnostics,
            generator=flow_generator,
            encoded=encoded,
        )
        cache = encoded.cache
        proposal_ablation_cache = None
        if diagnostics:
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
        losses.update(
            {
                "loss_total": loss_result.loss,
                **loss_result.metrics,
            },
            weight=batch.online.batch,
        )
        prediction = sample_cached_action(
            engine.model,
            cache,
            config,
            collect_diagnostics=diagnostics,
            dtype=dtype,
            generator=sampling_generator,
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
            event_logits=prediction.event_logits,
            motion_logits=prediction.motion_logits,
            motion_target=motion_target,
        )
        if diagnostics:
            if proposal_ablation_cache is None:
                raise RuntimeError("proposal ablation cache was not constructed")
            diagnostic_batches += 1
            common_sampling = {
                "initial_physical_noise": prediction.initial_physical_noise,
                "collect_diagnostics": False,
                "dtype": dtype,
            }
            proposal_zero = sample_cached_action(
                engine.model,
                proposal_ablation_cache,
                config,
                **common_sampling,
            )
            execution_no_updates = sample_cached_action(
                engine.model,
                cache,
                config,
                execution_mode="no_updates",
                **common_sampling,
            )
            execution_full_updates = sample_cached_action(
                engine.model,
                cache,
                config,
                execution_mode="full_updates",
                **common_sampling,
            )
            target = batch.action_target.normalized.float()
            primary = prediction.action.float()
            primary_error = primary - target
            ablation_rows: dict[str, torch.Tensor] = {
                "diagnostic_primary_mse_normalized": primary_error.square().mean(),
                "diagnostic_primary_mse_physical": (
                    primary_error / action_scale
                ).square().mean(),
            }
            for name, value in (
                ("proposal_zero", proposal_zero.action),
                ("execution_no_updates", execution_no_updates.action),
                ("execution_full_updates", execution_full_updates.action),
            ):
                error = value.float() - target
                delta = value.float() - primary
                ablation_rows[f"{name}_mse_normalized"] = error.square().mean()
                ablation_rows[f"{name}_mse_physical"] = (
                    error / action_scale
                ).square().mean()
                ablation_rows[f"{name}_action_delta_mse_normalized"] = (
                    delta.square().mean()
                )
                ablation_rows[f"{name}_action_delta_mse_physical"] = (
                    delta / action_scale
                ).square().mean()
            ablations.update(ablation_rows, weight=batch.online.batch)
            # The loss forward samples one random flow time, while this row is
            # the final step of a five-step deployment solve.  Their dynamic
            # P2/P3/bottom keys overlap but do not have the same semantics;
            # merging them under one name silently averaged unlike states.
            losses.update(
                {f"validation_deploy_{name}": value for name, value in prediction.metrics.items()},
                weight=batch.online.batch,
            )
            del proposal_ablation_cache
            del proposal_zero, execution_no_updates, execution_full_updates
    result = {**losses.materialize(), **deployment.means()}
    result["validation_ablation_batches"] = float(diagnostic_batches)
    result["validation_ablation_coverage"] = float(
        diagnostic_batches / max(completed_batches, 1)
    )
    if diagnostic_batches:
        rows = ablations.materialize()
        primary_normalized = rows["diagnostic_primary_mse_normalized"]
        primary_physical = rows["diagnostic_primary_mse_physical"]
        result["validation_diagnostic_primary_rmse_normalized"] = float(
            primary_normalized**0.5
        )
        result["validation_diagnostic_primary_rmse_physical"] = float(
            primary_physical**0.5
        )
        for name in (
            "proposal_zero",
            "execution_no_updates",
            "execution_full_updates",
        ):
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
            result = engine.train_step(batch, collect_diagnostics=emit)
            epoch_samples += batch.online.batch
            epoch_batches += 1
            window_samples += batch.online.batch
            window_batches += 1
            row = {
                "loss_total": result.loss,
                "gradient_global_preclip_l2": result.gradient_norm,
                **result.metrics,
            }
            epoch_metrics.update(row, weight=batch.online.batch)
            window_metrics.update(row, weight=batch.online.batch)
            if emit:
                values = archival_metrics(window_metrics.materialize())
                window_seconds = time.perf_counter() - window_started
                values["runtime_window_seconds_per_batch"] = window_seconds / max(window_batches, 1)
                values["runtime_window_samples_per_second"] = window_samples / max(
                    window_seconds, 1e-8
                )
                values["learning_rate"] = result.learning_rate
                values["learning_rate_history_proposal"] = (
                    result.learning_rate * role_lr_scale("history_proposal", config)
                )
                values["learning_rate_bottom_decoder"] = (
                    result.learning_rate * role_lr_scale("bottom_mmdit", config)
                )
                values["learning_rate_bottom_capacity"] = (
                    result.learning_rate * role_lr_scale("bottom_capacity", config)
                )
                logger.write(
                    "train",
                    epoch=epoch,
                    batch=batch_index,
                    step=engine.global_step,
                    metrics=values,
                )
                display_values = active_metrics(values)
                print(
                    logger.compact_line(
                        "train",
                        epoch=epoch,
                        batch=batch_index,
                        step=engine.global_step,
                        metrics=display_values,
                    ),
                    flush=True,
                )
                for detail_line in logger.diagnostic_lines(
                    "train",
                    epoch=epoch,
                    batch=batch_index,
                    step=engine.global_step,
                    metrics=display_values,
                ):
                    print(detail_line, flush=True)
                window_metrics = DeviceMetricAccumulator()
                window_started = time.perf_counter()
                window_samples = 0
                window_batches = 0
        train_values = archival_metrics(epoch_metrics.materialize())
        epoch_seconds = time.perf_counter() - epoch_started
        train_values["runtime_epoch_seconds"] = epoch_seconds
        train_values["runtime_seconds_per_batch"] = epoch_seconds / max(epoch_batches, 1)
        train_values["runtime_samples_per_second"] = epoch_samples / max(epoch_seconds, 1e-8)
        train_values.update(_cuda_memory_metrics(device))
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
        logger.write(
            "epoch",
            epoch=epoch,
            step=engine.global_step,
            train=train_values,
            validation=validation,
        )
        print(
            logger.compact_line(
                "val",
                epoch=epoch,
                batch=None,
                step=engine.global_step,
                metrics=active_metrics(validation),
            ),
            flush=True,
        )
        for detail_line in logger.diagnostic_lines(
            "val",
            epoch=epoch,
            batch=None,
            step=engine.global_step,
            metrics=active_metrics(validation),
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
