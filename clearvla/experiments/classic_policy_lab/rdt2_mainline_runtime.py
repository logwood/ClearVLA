from __future__ import annotations

"""Training/evaluation runtime for the v29 chunk policy and clean latent dynamics."""

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.evaluation.metrics import compute_metrics
from .normalizer import ArrayNormalizer
from .trainer import RDTTrainerConfig, _rdt_scheduler


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _objective_grad_norm(objective: Tensor, parameters: list[Tensor]) -> float:
    if not objective.requires_grad or not parameters:
        return 0.0
    grads = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _objective_grad_pair_stats(
    first: Tensor,
    second: Tensor,
    parameters: list[Tensor],
) -> tuple[float, float, float, float]:
    """Return first norm, second norm, second/first ratio, and cosine.

    Component gradients are evaluated only at logging intervals.  The cosine
    exposes whether conservative world-model transfer reinforces or conflicts
    with the native policy objective instead of hiding that interaction inside
    a single total gradient norm.
    """

    if not parameters:
        return 0.0, 0.0, 0.0, 0.0
    first_grads = (
        torch.autograd.grad(first, parameters, retain_graph=True, allow_unused=True)
        if first.requires_grad
        else [None] * len(parameters)
    )
    second_grads = (
        torch.autograd.grad(second, parameters, retain_graph=True, allow_unused=True)
        if second.requires_grad
        else [None] * len(parameters)
    )
    first_sq = 0.0
    second_sq = 0.0
    dot = 0.0
    for first_grad, second_grad in zip(first_grads, second_grads):
        if first_grad is not None:
            first_float = first_grad.detach().float()
            first_sq += float(first_float.pow(2).sum().cpu())
        else:
            first_float = None
        if second_grad is not None:
            second_float = second_grad.detach().float()
            second_sq += float(second_float.pow(2).sum().cpu())
        else:
            second_float = None
        if first_float is not None and second_float is not None:
            dot += float((first_float * second_float).sum().cpu())
    first_norm = math.sqrt(first_sq)
    second_norm = math.sqrt(second_sq)
    ratio = second_norm / max(first_norm, 1e-12)
    cosine = dot / max(first_norm * second_norm, 1e-12)
    return first_norm, second_norm, ratio, cosine


def _concat(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("cannot concatenate an empty metric list")
    return np.concatenate(rows, axis=0)


def _condition_kwargs(condition) -> dict[str, Any]:
    return {
        "dense_tokens": condition.dense_tokens,
        "kv_cache": condition.kv_cache,
        "attention_mask": condition.attention_mask,
    }


def _timer_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _deterministic_stats_loader(train_loader: DataLoader) -> DataLoader:
    """Build a sequential calibration loader without consuming train-sampler RNG.

    Iterating the shuffled training loader before epoch 1 changes its sampler
    state and invalidates seed-matched policy comparisons.  Calibration uses
    the same dataset/collate contract but an independent deterministic loader.
    """

    batch_size = train_loader.batch_size
    if batch_size is None:
        raise ValueError("future latent calibration requires a fixed batch_size loader")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2_026_061_701)
    return DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=train_loader.collate_fn,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def _future_sample_keys(batch: dict[str, Tensor]) -> Tensor:
    if "future_image_indices" not in batch:
        raise KeyError("future_image_indices missing from latent-dynamics batch")
    frame_indices = batch["future_image_indices"].to(dtype=torch.long)
    if frame_indices.ndim != 2:
        raise ValueError(f"future_image_indices must be [B,T], got {tuple(frame_indices.shape)}")
    episode = batch["episode_idx"].to(dtype=torch.long).reshape(-1, 1).expand_as(frame_indices)
    return torch.stack([episode, frame_indices], dim=-1)


@torch.no_grad()
def _encode_future_latents(
    model,
    conditioner,
    batch: dict[str, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
    camera_names: tuple[str, ...],
) -> Tensor | None:
    """Return frozen DINO targets as [B,T,C,Patch,D]."""

    if not bool(getattr(model, "future_latent_enabled", False)):
        return None
    keys = _future_sample_keys(batch)
    batch_size, future_steps = keys.shape[:2]
    flat_keys = keys.reshape(batch_size * future_steps, 2)
    if "future_obs_image" in batch:
        future_images = batch["future_obs_image"]
        if future_images.ndim != 6:
            raise ValueError(
                "future_obs_image must be [B,T,Cam,RGB,H,W], "
                f"got {tuple(future_images.shape)}"
            )
        flat_images = future_images.reshape(
            batch_size * future_steps,
            *future_images.shape[2:],
        ).to(device=device, non_blocking=True)
    else:
        # CachedDinoV2DenseConditioner ignores image values and uses sample_keys.
        # Repeating the current image keeps the conditioner protocol uniform.
        current = batch["obs_image"]
        flat_images = current[:, None].expand(
            batch_size, future_steps, *current.shape[1:]
        ).reshape(batch_size * future_steps, *current.shape[1:]).to(
            device=device, non_blocking=True
        )
    encoded = conditioner.encode(
        flat_images,
        [instruction] * (batch_size * future_steps),
        sample_keys=flat_keys,
        image_ablation="normal",
        camera_names=camera_names,
    ).to(device=device, dtype=dtype)
    if encoded.dense_tokens is None or encoded.kv_cache is not None:
        raise ValueError("future latent dynamics requires a dense-token DINO conditioner")
    tokens = encoded.dense_tokens
    cameras = len(camera_names)
    if tokens.shape[1] % cameras:
        raise ValueError(
            f"dense future token length {tokens.shape[1]} is not divisible by {cameras} cameras"
        )
    patches = tokens.shape[1] // cameras
    return tokens.reshape(
        batch_size,
        future_steps,
        cameras,
        patches,
        tokens.shape[-1],
    )


@torch.no_grad()
def calibrate_future_latent_stats(
    model,
    conditioner,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    instruction: str,
) -> dict[str, Any]:
    """Estimate fixed [future-time,camera,channel] DINO residual statistics."""

    if not bool(getattr(model, "future_latent_enabled", False)):
        return {"enabled": False}
    model.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()
    dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(loader.dataset, "camera_names", ("top", "wrist")))
    residual_sum: Tensor | None = None
    residual_sq_sum: Tensor | None = None
    count = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["obs_image"].to(device=device, non_blocking=True)
        sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        current_condition = conditioner.encode(
            images,
            [instruction] * images.shape[0],
            sample_keys=sample_keys,
            image_ablation="normal",
            camera_names=camera_names,
        ).to(device=device, dtype=dtype)
        if current_condition.dense_tokens is None or current_condition.kv_cache is not None:
            raise ValueError("clean future dynamics requires dense current DINO tokens")
        future = _encode_future_latents(
            model,
            conditioner,
            batch,
            device=device,
            dtype=dtype,
            instruction=instruction,
            camera_names=camera_names,
        )
        if future is None:
            raise AssertionError("future latent encoding unexpectedly disabled")
        current_compressed = model.compress_current_latents(current_condition.dense_tokens).float()
        future_compressed = model.compress_future_latents(future).float()
        residual = model.future_residual_target(current_compressed, future_compressed)
        # Sum over batch and spatial positions while preserving T,C,D.
        row_sum = residual.sum(dim=(0, 3))
        row_sq_sum = residual.square().sum(dim=(0, 3))
        residual_sum = row_sum if residual_sum is None else residual_sum + row_sum
        residual_sq_sum = row_sq_sum if residual_sq_sum is None else residual_sq_sum + row_sq_sum
        count += int(residual.shape[0] * residual.shape[3])
        batches += 1
    if count <= 0 or residual_sum is None or residual_sq_sum is None:
        raise RuntimeError("future residual statistics calibration saw no samples")
    mean = residual_sum / float(count)
    variance = residual_sq_sum / float(count) - mean.square()
    std = variance.clamp_min(1e-10).sqrt()
    model.set_future_latent_stats(mean, std)
    return {
        "enabled": True,
        "target": "future_minus_current_dino",
        "batches": batches,
        "vectors_per_time_camera": count,
        "time_count": int(mean.shape[0]),
        "camera_count": int(mean.shape[1]),
        "channel_count": int(mean.shape[2]),
        "mean_rms": float(mean.square().mean().sqrt().cpu()),
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
        "per_time_camera_std_mean": std.mean(dim=-1).cpu().tolist(),
    }


def _future_metric_suffixes(model) -> tuple[list[int], int]:
    config = model.config
    return list(config.future_latent_offsets), int(config.future_latent_num_cameras)


def _format_future_metrics(prefix: str, metrics: dict[str, float], model) -> str:
    if not bool(getattr(model, "future_latent_enabled", False)):
        return f"{prefix} disabled"
    offsets, cameras = _future_metric_suffixes(model)
    fields = [
        f"world={metrics.get('future_world_objective', float('nan')):.6f}",
        f"flow={metrics.get('future_latent_flow_mse', float('nan')):.6f}",
        f"flow_u={metrics.get('future_latent_flow_mse_unweighted', float('nan')):.6f}",
        f"abs_ep={metrics.get('future_latent_absolute_endpoint_rmse', float('nan')):.6f}",
        f"r2={metrics.get('future_latent_residual_endpoint_r2', float('nan')):.4f}",
        f"cos={metrics.get('future_latent_velocity_cosine', float('nan')):.4f}",
        f"vrms={metrics.get('future_latent_velocity_pred_rms', float('nan')):.4f}/"
        f"{metrics.get('future_latent_velocity_target_rms', float('nan')):.4f}",
        f"dep={metrics.get('future_action_dependency_gap', float('nan')):.5f}/"
        f"{metrics.get('future_action_dependency_relative_gap', float('nan')):.4f}/"
        f"{metrics.get('future_action_dependency_loss', float('nan')):.5f}",
        f"skill={metrics.get('future_world_relative_skill', float('nan')):.4f}",
        f"align={metrics.get('future_align_positive_cosine', float('nan')):.3f}/"
        f"{metrics.get('future_align_negative_cosine', float('nan')):.3f}/"
        f"{metrics.get('future_align_margin', float('nan')):.3f}",
        f"nce={metrics.get('future_align_symmetric_nce_loss', float('nan')):.3f}/"
        f"{metrics.get('future_align_structured_nce_loss', float('nan')):.3f} "
        f"top1={metrics.get('future_align_action_to_future_top1', float('nan')):.2f}/"
        f"{metrics.get('future_align_future_to_action_top1', float('nan')):.2f}",
        f"std={metrics.get('future_action_embedding_std', float('nan')):.3f}/"
        f"{metrics.get('future_change_embedding_std', float('nan')):.3f} "
        f"areg={metrics.get('future_embedding_variance_loss', float('nan')):.3f}/"
        f"{metrics.get('future_embedding_covariance_loss', float('nan')):.3f}",
        f"arecon={metrics.get('future_action_reconstruction_arm_rmse', float('nan')):.4f}/"
        f"{metrics.get('future_action_reconstruction_gripper_f1', float('nan')):.3f}",
        f"inv={metrics.get('future_inverse_arm_rmse', float('nan')):.4f}/"
        f"{metrics.get('future_inverse_gripper_f1', float('nan')):.3f}/"
        f"{metrics.get('future_inverse_transition_timing_mae', float('nan')):.3f}",
        f"cycle={metrics.get('future_pred_action_cycle_arm_rmse', float('nan')):.4f}/"
        f"{metrics.get('future_pred_action_cycle_gripper_f1', float('nan')):.3f}@"
        f"{metrics.get('future_semantic_scale', 0.0):.3f}",
        f"again={metrics.get('future_inverse_future_gain', float('nan')):.4f} "
        f"jac={metrics.get('future_action_jacobian_rms', float('nan')):.4f}",
        f"ftime={metrics.get('future_action_time_weight_raw_mean', float('nan')):.3f}/"
        f"{metrics.get('future_action_time_weight_raw_max', float('nan')):.3f}",
        f"abridge={metrics.get('future_policy_bridge_time_weight_raw_mean', 0.0):.3f}/"
        f"{metrics.get('future_policy_bridge_time_weight_raw_max', 0.0):.3f}",
        f"conf={metrics.get('future_policy_dependency_confidence_mean', 0.0):.3f}/"
        f"{metrics.get('future_policy_world_skill_confidence_mean', 0.0):.3f}/"
        f"{metrics.get('future_policy_semantic_confidence_mean', 0.0):.3f}/"
        f"{metrics.get('future_policy_joint_confidence_mean', 0.0):.3f}",
        f"relative={metrics.get('future_policy_relative_regret', 0.0):.5f}/"
        f"{metrics.get('future_policy_relative_hinge', 0.0):.5f}@"
        f"{metrics.get('future_policy_consistency_scale', 0.0):.3f}",
        f"teacher={metrics.get('future_policy_teacher_consistency_mse', 0.0):.6f}/"
        f"{metrics.get('future_policy_teacher_consistency_relative', 0.0):.6f}",
        f"active={metrics.get('future_policy_relative_hinge_active_fraction', 0.0):.3f}",
        f"motion={metrics.get('future_latent_dynamic_patch_mse', float('nan')):.5f}/"
        f"{metrics.get('future_latent_static_patch_mse', float('nan')):.5f}",
        f"gates={metrics.get('future_gate_self', float('nan')):.3f}/"
        f"{metrics.get('future_gate_current', float('nan')):.3f}/"
        f"{metrics.get('future_gate_action', float('nan')):.3f}",
    ]
    per_target = []
    for time_idx, offset in enumerate(offsets):
        camera_values = [
            metrics.get(f"future_latent_t{time_idx}_c{camera_idx}_mse", float("nan"))
            for camera_idx in range(cameras)
        ]
        per_target.append(
            f"t+{offset}=" + "/".join(f"{value:.5f}" for value in camera_values)
        )
    fields.append("targets[" + ",".join(per_target) + "]")
    return " ".join(fields)


@torch.no_grad()
def evaluate_mainline_rdt2_fm(
    model,
    conditioner,
    loader: DataLoader,
    *,
    device: torch.device,
    action_normalizer: ArrayNormalizer,
    inference_steps: int,
    max_batches: int = 0,
    instruction: str = "",
    image_ablation: str = "normal",
) -> dict[str, Any]:
    allowed = {"normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"}
    if image_ablation not in allowed:
        raise ValueError(f"unsupported image_ablation={image_ablation!r}")
    model.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()
    model_dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(loader.dataset, "camera_names", ("top", "wrist")))
    full_rows: list[np.ndarray] = []
    fast_rows: list[np.ndarray] = []
    prefix_rows: list[np.ndarray] = []
    learned_prior_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    hold_rows: list[np.ndarray] = []
    past_rows: list[np.ndarray] = []
    losses: list[dict[str, float]] = []
    latency = {"fast_ms": [], "prefix_ms": [], "full_ms": []}
    future_generator = torch.Generator(device=device)
    future_generator.manual_seed(20260617)
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        state = batch["state"].to(device=device, dtype=model_dtype, non_blocking=True)
        past = batch["past"].to(device=device, dtype=model_dtype, non_blocking=True)
        hold = batch["prior"].to(device=device, dtype=model_dtype, non_blocking=True)
        action = batch["action"].to(device=device, dtype=model_dtype, non_blocking=True)
        images = batch["obs_image"].to(device=device, non_blocking=True)
        sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        conditioner_ablation = image_ablation
        if image_ablation == "shuffle-episode":
            sample_keys = loader.dataset.cross_episode_keys(sample_keys, seed=batch_index)
            images = loader.dataset.load_images_for_keys(sample_keys).to(device=device, non_blocking=True)
            conditioner_ablation = "normal"
        condition = conditioner.encode(
            images,
            [instruction] * state.shape[0],
            sample_keys=sample_keys,
            image_ablation=conditioner_ablation,
            camera_names=camera_names,
        ).to(device=device, dtype=model_dtype)
        future_latents = _encode_future_latents(
            model,
            conditioner,
            batch,
            device=device,
            dtype=model_dtype,
            instruction=instruction,
            camera_names=camera_names,
        )
        kwargs = _condition_kwargs(condition)
        loss = model.compute_loss(
            state_tokens=state,
            past_actions=past,
            physical_prior=hold,
            action_gt=action,
            future_latent_tokens=future_latents,
            global_step=10**9,
            future_flow_generator=future_generator,
            **kwargs,
        )
        losses.append({key: float(value.detach().float().cpu()) for key, value in loss.items()})
        _timer_sync(device)
        start = time.perf_counter()
        fast = model.predict_first_action(
            state_tokens=state,
            past_actions=past,
            physical_prior=hold,
            steps=inference_steps,
            **kwargs,
        )
        _timer_sync(device)
        latency["fast_ms"].append((time.perf_counter() - start) * 1000)
        _timer_sync(device)
        start = time.perf_counter()
        prefix = model.predict_prefix_action(
            state_tokens=state,
            past_actions=past,
            physical_prior=hold,
            steps=inference_steps,
            **kwargs,
        )
        _timer_sync(device)
        latency["prefix_ms"].append((time.perf_counter() - start) * 1000)
        _timer_sync(device)
        start = time.perf_counter()
        full = model.predict_action(
            state_tokens=state,
            past_actions=past,
            physical_prior=hold,
            steps=inference_steps,
            **kwargs,
        )
        _timer_sync(device)
        latency["full_ms"].append((time.perf_counter() - start) * 1000)
        learned_prior, _ = model.predict_prior(state_tokens=state, past_actions=past, physical_prior=hold)
        fast_chunk = learned_prior.clone()
        fast_chunk[:, 0] = fast
        prefix_chunk = learned_prior.clone()
        prefix_chunk[:, : prefix.shape[1]] = prefix
        full_rows.append(full.float().cpu().numpy())
        fast_rows.append(fast_chunk.float().cpu().numpy())
        prefix_rows.append(prefix_chunk.float().cpu().numpy())
        learned_prior_rows.append(learned_prior.float().cpu().numpy())
        target_rows.append(action.float().cpu().numpy())
        hold_rows.append(action_normalizer.encode(batch["prior_raw"].numpy()))
        past_rows.append(action_normalizer.encode(batch["past_raw"].numpy()))
    pred = _concat(full_rows)
    target = _concat(target_rows)
    prior = _concat(hold_rows)
    past = _concat(past_rows)
    metrics = compute_metrics(pred_norm=pred, target_norm=target, prior_norm=prior, past_norm=past, normalizer=action_normalizer)
    learned_metrics = compute_metrics(pred_norm=_concat(learned_prior_rows), target_norm=target, prior_norm=prior, past_norm=past, normalizer=action_normalizer)
    fast_metrics = compute_metrics(pred_norm=_concat(fast_rows), target_norm=target, prior_norm=prior, past_norm=past, normalizer=action_normalizer)
    prefix_metrics = compute_metrics(pred_norm=_concat(prefix_rows), target_norm=target, prior_norm=prior, past_norm=past, normalizer=action_normalizer)
    metrics.update({
        "inference_steps": int(inference_steps),
        "image_ablation": str(image_ablation),
        "learned_prior_full_mse": learned_metrics["full_mse"],
        "learned_prior_arm_first_rmse": learned_metrics.get("arm_first_rmse", learned_metrics["first_rmse"]),
        "fast_exit_arm_first_rmse": fast_metrics.get("arm_first_rmse", fast_metrics["first_rmse"]),
        "fast_exit_first_rmse": fast_metrics["first_rmse"],
        "prefix_exit_first4_rmse": prefix_metrics["first4_rmse"],
        "prefix_exit_first8_rmse": prefix_metrics["first8_rmse"],
        "latency_fast_ms": float(np.mean(latency["fast_ms"])),
        "latency_prefix_ms": float(np.mean(latency["prefix_ms"])),
        "latency_full_ms": float(np.mean(latency["full_ms"])),
    })
    for key in losses[0]:
        metrics[f"val_{key}"] = float(np.mean([row[key] for row in losses]))
    return metrics


def train_mainline_rdt2_fm(
    *,
    model,
    conditioner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    trainer: RDTTrainerConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
    inference_steps: int,
    instruction: str = "",
    future_latent_stat_batches: int = 128,
    log_component_grad_norms: bool = True,
    future_world_lr: float | None = None,
    future_world_weight_decay: float | None = None,
    future_world_grad_clip: float = 1.0,
) -> dict[str, Any]:
    if future_world_lr is not None and float(future_world_lr) <= 0.0:
        raise ValueError("future_world_lr must be positive when provided")
    if future_world_weight_decay is not None and float(future_world_weight_decay) < 0.0:
        raise ValueError("future_world_weight_decay must be non-negative when provided")
    if float(future_world_grad_clip) <= 0.0:
        raise ValueError("future_world_grad_clip must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = out_dir / "rdt2_mainline_steps.jsonl"
    epoch_log_path = out_dir / "rdt2_mainline_epochs.jsonl"
    # A fresh run should not silently append to a stale experiment log.
    step_log_path.unlink(missing_ok=True)
    epoch_log_path.unlink(missing_ok=True)

    stats_loader = _deterministic_stats_loader(train_loader)
    latent_stats = calibrate_future_latent_stats(
        model,
        conditioner,
        stats_loader,
        device=device,
        max_batches=future_latent_stat_batches,
        instruction=instruction,
    )
    context = dict(context)
    context["future_latent_stats"] = latent_stats
    context["future_randomness"] = {
        "stats_loader": "independent-sequential",
        "train_flow_generator_seed": int(context.get("args", {}).get("seed", 0)) + 2_026_061_700,
        "policy_global_rng_consumed_by_future_flow": False,
    }
    (out_dir / "future_latent_stats.json").write_text(
        json.dumps(_jsonable(latent_stats), indent=2), encoding="utf-8"
    )
    if latent_stats.get("enabled"):
        print(
            "[rdt2-latent-stats] "
            f"batches={latent_stats['batches']} vectors_per_tc={latent_stats['vectors_per_time_camera']} "
            f"shape={latent_stats['time_count']}x{latent_stats['camera_count']}x{latent_stats['channel_count']} mean_rms={latent_stats['mean_rms']:.6f} "
            f"std={latent_stats['std_mean']:.6f} "
            f"range=[{latent_stats['std_min']:.6f},{latent_stats['std_max']:.6f}]",
            flush=True,
        )

    policy_parameters = model.policy_parameters()
    world_parameters = model.future_latent_parameters()
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": policy_parameters,
            "lr": trainer.lr,
            "weight_decay": trainer.weight_decay,
            "group_name": "policy",
        }
    ]
    if world_parameters:
        parameter_groups.append({
            "params": world_parameters,
            "lr": trainer.lr if future_world_lr is None else float(future_world_lr),
            "weight_decay": (
                trainer.weight_decay
                if future_world_weight_decay is None
                else float(future_world_weight_decay)
            ),
            "group_name": "future_world",
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=trainer.lr,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    scheduler = _rdt_scheduler(
        optimizer,
        scheduler=trainer.scheduler,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    best_full = float("inf")
    best_fast = float("inf")
    best_future = float("inf")
    best_joint = float("inf")
    history = []
    global_step = 0
    dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(train_loader.dataset, "camera_names", ("top", "wrist")))
    future_train_generator = torch.Generator(device=device)
    run_seed = int(context.get("args", {}).get("seed", 0))
    future_train_generator.manual_seed(run_seed + 2_026_061_700)
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        if hasattr(conditioner, "eval"):
            conditioner.eval()
        started = time.perf_counter()
        rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            state = batch["state"].to(device=device, dtype=dtype, non_blocking=True)
            past = batch["past"].to(device=device, dtype=dtype, non_blocking=True)
            hold = batch["prior"].to(device=device, dtype=dtype, non_blocking=True)
            action = batch["action"].to(device=device, dtype=dtype, non_blocking=True)
            images = batch["obs_image"].to(device=device, non_blocking=True)
            sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
            with torch.no_grad():
                condition = conditioner.encode(
                    images,
                    [instruction] * state.shape[0],
                    sample_keys=sample_keys,
                    image_ablation="normal",
                    camera_names=camera_names,
                ).to(device=device, dtype=dtype)
                future_latents = _encode_future_latents(
                    model,
                    conditioner,
                    batch,
                    device=device,
                    dtype=dtype,
                    instruction=instruction,
                    camera_names=camera_names,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                state_tokens=state,
                past_actions=past,
                physical_prior=hold,
                action_gt=action,
                future_latent_tokens=future_latents,
                global_step=global_step,
                future_flow_generator=future_train_generator,
                **_condition_kwargs(condition),
            )
            should_log = batch_index % trainer.log_every == 0
            policy_component_grad = 0.0
            world_to_policy_grad = 0.0
            consistency_to_policy_grad = 0.0
            consistency_to_world_grad = 0.0
            consistency_to_policy_ratio = 0.0
            policy_consistency_grad_cosine = 0.0
            semantic_flow_grad = 0.0
            align_to_flow_grad_ratio = 0.0
            inverse_to_flow_grad_ratio = 0.0
            pred_align_to_flow_grad_ratio = 0.0
            cycle_to_flow_grad_ratio = 0.0
            align_flow_grad_cosine = 0.0
            inverse_flow_grad_cosine = 0.0
            pred_align_flow_grad_cosine = 0.0
            cycle_flow_grad_cosine = 0.0
            if should_log and log_component_grad_norms:
                policy_parameters = model.policy_parameters()
                world_parameters = model.future_latent_parameters()
                world_to_policy_grad = _objective_grad_norm(
                    model.config.future_world_loss_weight * loss["future_world_objective"],
                    policy_parameters,
                )
                consistency_weighted = (
                    model.config.future_consistency_loss_weight
                    * loss["future_policy_consistency_scale"]
                    * loss["future_policy_consistency_objective"]
                )
                (
                    policy_component_grad,
                    consistency_to_policy_grad,
                    consistency_to_policy_ratio,
                    policy_consistency_grad_cosine,
                ) = _objective_grad_pair_stats(
                    loss["policy_objective"],
                    consistency_weighted,
                    policy_parameters,
                )
                consistency_to_world_grad = _objective_grad_norm(
                    consistency_weighted, world_parameters
                )
                if model.future_latent_enabled:
                    flow_component = (
                        model.config.future_world_loss_weight
                        * loss["future_flow_train_objective"]
                    )
                    align_component = (
                        model.config.future_world_loss_weight
                        * model.config.future_align_loss_weight
                        * loss["future_align_train_objective"]
                    )
                    inverse_component = (
                        model.config.future_world_loss_weight
                        * model.config.future_inverse_loss_weight
                        * loss["future_inverse_train_objective"]
                    )
                    pred_align_component = (
                        model.config.future_world_loss_weight
                        * loss["future_semantic_scale"]
                        * model.config.future_pred_align_loss_weight
                        * loss["future_pred_align_train_objective"]
                    )
                    cycle_component = (
                        model.config.future_world_loss_weight
                        * loss["future_semantic_scale"]
                        * model.config.future_cycle_loss_weight
                        * loss["future_cycle_train_objective"]
                    )
                    (
                        semantic_flow_grad,
                        _align_grad,
                        align_to_flow_grad_ratio,
                        align_flow_grad_cosine,
                    ) = _objective_grad_pair_stats(flow_component, align_component, world_parameters)
                    (
                        _flow_again,
                        _inverse_grad,
                        inverse_to_flow_grad_ratio,
                        inverse_flow_grad_cosine,
                    ) = _objective_grad_pair_stats(flow_component, inverse_component, world_parameters)
                    (
                        _flow_again,
                        _pred_align_grad,
                        pred_align_to_flow_grad_ratio,
                        pred_align_flow_grad_cosine,
                    ) = _objective_grad_pair_stats(flow_component, pred_align_component, world_parameters)
                    (
                        _flow_again,
                        _cycle_grad,
                        cycle_to_flow_grad_ratio,
                        cycle_flow_grad_cosine,
                    ) = _objective_grad_pair_stats(flow_component, cycle_component, world_parameters)
            loss["loss"].backward()
            grad_policy = _grad_norm(policy_parameters)
            grad_world = _grad_norm(world_parameters)
            grad_total = math.sqrt(grad_policy * grad_policy + grad_world * grad_world)
            # Independent clipping is essential: otherwise a large auxiliary
            # world-model gradient rescales policy gradients despite zero shared
            # parameters, reintroducing task interference through the optimizer.
            policy_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(policy_parameters, trainer.grad_clip)
            )
            world_clip_norm = 0.0
            if world_parameters:
                world_clip_norm = float(
                    torch.nn.utils.clip_grad_norm_(world_parameters, future_world_grad_clip)
                )
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in loss.items()}
            row.update({
                "grad_total": grad_total,
                "grad_policy": grad_policy,
                "grad_world": grad_world,
                "grad_shared": grad_policy,
                "grad_future": grad_world,
                "grad_policy_clip_input": policy_clip_norm,
                "grad_world_clip_input": world_clip_norm,
                "grad_policy_component_shared": policy_component_grad,
                "grad_world_to_policy": world_to_policy_grad,
                "grad_consistency_to_policy": consistency_to_policy_grad,
                "grad_consistency_to_world": consistency_to_world_grad,
                "grad_consistency_to_policy_ratio": consistency_to_policy_ratio,
                "grad_policy_consistency_cosine": policy_consistency_grad_cosine,
                "grad_semantic_flow": semantic_flow_grad,
                "grad_align_to_flow_ratio": align_to_flow_grad_ratio,
                "grad_inverse_to_flow_ratio": inverse_to_flow_grad_ratio,
                "grad_pred_align_to_flow_ratio": pred_align_to_flow_grad_ratio,
                "grad_cycle_to_flow_ratio": cycle_to_flow_grad_ratio,
                "grad_align_flow_cosine": align_flow_grad_cosine,
                "grad_inverse_flow_cosine": inverse_flow_grad_cosine,
                "grad_pred_align_flow_cosine": pred_align_flow_grad_cosine,
                "grad_cycle_flow_cosine": cycle_flow_grad_cosine,
            })
            rows.append(row)
            if should_log:
                latest = rows[-trainer.log_every:]
                avg = {key: float(np.mean([item[key] for item in latest])) for key in latest[0]}
                step_payload = {
                    "schema": "clearvla-rdt2-mainline-step-v9-contrastive-action-anchor",
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "world_lr": (
                        optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
                    ),
                    "averaging_window": len(latest),
                    "metrics": avg,
                }
                _append_jsonl(step_log_path, step_payload)
                print(
                    f"[rdt2-mainline] epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={avg['loss']:.6f} policy={avg['policy_objective']:.6f} "
                    f"arm_flow={avg['full_arm_flow_mse']:.6f} prior_arm={avg['prior_arm_mse']:.6f} "
                    f"first_arm={avg['fast_first_arm_flow_mse']:.6f} prefix_arm={avg['prefix_arm_flow_mse']:.6f} "
                    f"gripper_state={avg['full_gripper_state_loss']:.6f} transition={avg['gripper_transition_loss']:.6f} "
                    f"gripper_delta={avg['gripper_delta_loss']:.6f} arm_delta={avg.get('arm_delta_loss', 0.0):.6f} "
                    f"align={avg.get('align_phase_arm_loss', 0.0):.6f}/{avg.get('align_phase_fraction', 0.0):.3f} "
                    f"visual_gate={avg['fast_visual_top_gate_mean']:.3f}/{avg['fast_visual_wrist_gate_mean']:.3f} "
                    f"grad={avg['grad_total']:.3e} shared={avg['grad_shared']:.3e} future={avg['grad_future']:.3e} "
                    f"component=policy:{avg['grad_policy_component_shared']:.3e} "
                    f"world->policy:{avg['grad_world_to_policy']:.1e} "
                    f"cons->policy/world:{avg['grad_consistency_to_policy']:.3e}/{avg['grad_consistency_to_world']:.1e} "
                    f"cons_ratio/cos:{avg['grad_consistency_to_policy_ratio']:.3e}/"
                    f"{avg['grad_policy_consistency_cosine']:.3f} "
                    f"sem_grad={avg['grad_align_to_flow_ratio']:.2e}/"
                    f"{avg['grad_inverse_to_flow_ratio']:.2e}/"
                    f"{avg['grad_pred_align_to_flow_ratio']:.2e}/"
                    f"{avg['grad_cycle_to_flow_ratio']:.2e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"world_lr={(optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0):.3e}",
                    flush=True,
                )
                if model.future_latent_enabled:
                    print(
                        "[rdt2-latent] " + _format_future_metrics("latent", avg, model),
                        flush=True,
                    )
        metrics: dict[str, Any] = {}
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_mainline_rdt2_fm(
                model,
                conditioner,
                val_loader,
                device=device,
                action_normalizer=action_normalizer,
                inference_steps=inference_steps,
                max_batches=trainer.max_val_batches,
                instruction=instruction,
                image_ablation="normal",
            )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - started,
            "train": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "val": metrics,
        }
        history.append(record)
        _append_jsonl(epoch_log_path, {
            "schema": "clearvla-rdt2-mainline-epoch-v9-contrastive-action-anchor",
            **record,
        })
        if metrics:
            print(
                f"[rdt2-mainline] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} "
                f"val_full={metrics['full_mse']:.6f} arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} "
                f"step8={metrics.get('step_8_rmse', float('nan')):.6f} step12={metrics.get('step_12_rmse', float('nan')):.6f} "
                f"step24={metrics.get('step_24_rmse', float('nan')):.6f} "
                f"endpoint={metrics.get('chunk_endpoint_rmse', float('nan')):.6f} "
                f"fast_first={metrics['fast_exit_arm_first_rmse']:.6f} prefix4={metrics['prefix_exit_first4_rmse']:.6f} "
                f"learned_prior={metrics['learned_prior_full_mse']:.6f} hold={metrics['hold_last_full_mse']:.6f} "
                f"latency_ms={metrics['latency_fast_ms']:.2f}/{metrics['latency_prefix_ms']:.2f}/{metrics['latency_full_ms']:.2f}",
                flush=True,
            )
            if model.future_latent_enabled:
                val_latent = {
                    key.removeprefix("val_"): value
                    for key, value in metrics.items()
                    if key.startswith("val_future_")
                }
                print(
                    "[rdt2-latent-val] " + _format_future_metrics("latent", val_latent, model),
                    flush=True,
                )
        payload = {
            "schema": "clearvla-rdt2-mainline-checkpoint-v7-contrastive-action-anchor",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_groups": [group.get("group_name", f"group_{idx}") for idx, group in enumerate(optimizer.param_groups)],
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "context": context,
            "action_normalizer": action_normalizer.to_dict(),
            "state_normalizer": state_normalizer.to_dict(),
            "history": history,
        }
        _save(out_dir / "checkpoints/latest.pt", payload)
        if metrics:
            if metrics["full_mse"] < best_full:
                best_full = metrics["full_mse"]
                _save(out_dir / "checkpoints/best_full.pt", payload)
            if metrics["fast_exit_arm_first_rmse"] < best_fast:
                best_fast = metrics["fast_exit_arm_first_rmse"]
                _save(out_dir / "checkpoints/best_fast_first.pt", payload)
            if model.future_latent_enabled:
                future_score = metrics["val_future_latent_flow_mse"]
                joint_score = metrics["full_mse"] + model.config.future_world_loss_weight * future_score
                if future_score < best_future:
                    best_future = future_score
                    _save(out_dir / "checkpoints/best_future_latent.pt", payload)
                if joint_score < best_joint:
                    best_joint = joint_score
                    _save(out_dir / "checkpoints/best_joint.pt", payload)
    summary = {
        "schema": "clearvla-rdt2-mainline-summary-v8-contrastive-action-anchor",
        "parameter_count": model.parameter_count(),
        "future_latent_parameter_count": sum(parameter.numel() for parameter in model.future_latent_parameters()),
        "best_full_mse": best_full,
        "best_fast_first_arm_rmse": best_fast,
        "best_future_latent_flow_mse": best_future,
        "best_joint_score": best_joint,
        "future_latent_stats": latent_stats,
        "history": history,
        "context": context,
    }
    (out_dir / "rdt2_mainline_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "calibrate_future_latent_stats",
    "evaluate_mainline_rdt2_fm",
    "train_mainline_rdt2_fm",
]
