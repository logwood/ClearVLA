from __future__ import annotations

"""Action-independent pretraining for the dynamic world representation.

This stage deliberately receives no action trajectory and contains no world
predictor or policy.  Its only purpose is to define a fixed, locally physical
dynamics space shared by every downstream predictor baseline.

The representation is anchored by three observable contracts:

* a fixed non-learned descriptor of temporal DINO changes;
* the low-dimensional state change over the same visual history;
* the state at the end of the history window.

Temporal increment consistency is imposed across current and future histories,
without instance-level InfoNCE or episode-identity classification.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner

from .model import DynamicPredictiveWorld
from .runtime import encode_sample_tokens


@dataclass(frozen=True)
class DynamicRepresentationLossConfig:
    descriptor_weight: float = 1.0
    local_motion_weight: float = 0.50
    context_state_weight: float = 0.25
    temporal_increment_weight: float = 0.50
    variance_weight: float = 0.02
    covariance_weight: float = 0.005
    token_diversity_weight: float = 0.01
    embedding_std_target: float = 0.05
    gripper_transition_boost: float = 3.0
    gripper_transition_threshold: float = 0.10

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name.endswith("_weight") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.embedding_std_target < 0:
            raise ValueError("embedding_std_target must be non-negative")
        if self.gripper_transition_threshold < 0 or self.gripper_transition_boost < 0:
            raise ValueError("invalid gripper transition configuration")


@dataclass(frozen=True)
class DynamicRepresentationTrainerConfig:
    epochs: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0


@dataclass(frozen=True)
class PreparedRepresentationBatch:
    current_tokens: Tensor
    target_tokens: Tensor
    history_state: Tensor
    history_state_raw: Tensor
    target_history_state: Tensor
    target_history_state_raw: Tensor


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")


def _grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-4)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def prepare_representation_batch(
    batch: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model: DynamicPredictiveWorld,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedRepresentationBatch:
    current, target = encode_sample_tokens(
        batch,
        conditioner=conditioner,
        model_config=model.config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    return PreparedRepresentationBatch(
        current_tokens=current,
        target_tokens=target,
        history_state=batch["history_state"].to(device=device, dtype=dtype, non_blocking=True),
        history_state_raw=batch["history_state_raw"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        target_history_state=batch["target_history_state"].to(
            device=device, dtype=dtype, non_blocking=True
        ),
        target_history_state_raw=batch["target_history_state_raw"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    )


def _stack_windows(batch: PreparedRepresentationBatch) -> tuple[Tensor, Tensor, Tensor]:
    tokens = torch.cat([batch.current_tokens[:, None], batch.target_tokens], dim=1)
    states = torch.cat([batch.history_state[:, None], batch.target_history_state], dim=1)
    states_raw = torch.cat(
        [batch.history_state_raw[:, None], batch.target_history_state_raw], dim=1
    )
    return tokens, states, states_raw


def _variance_loss(embedding: Tensor, target_std: float) -> tuple[Tensor, Tensor]:
    flat = embedding.float().reshape(-1, embedding.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=flat.shape[0] > 1) + 1e-4)
    return F.relu(float(target_std) - std).mean(), std.mean()


def _covariance_loss(embedding: Tensor) -> Tensor:
    flat = embedding.float().reshape(-1, embedding.shape[-1])
    if flat.shape[0] <= 1:
        return flat.new_zeros(())
    flat = flat - flat.mean(dim=0, keepdim=True)
    covariance = flat.T @ flat / float(flat.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    return off_diagonal.square().sum() / max(embedding.shape[-1], 1)


def _token_diversity_loss(dynamic: Tensor) -> Tensor:
    # Penalize identical query tokens, but do not force arbitrary instance IDs.
    normalized = F.normalize(dynamic.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    count = similarity.shape[-1]
    if count <= 1:
        return similarity.new_zeros(())
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    return similarity[..., mask].square().mean()


def _weighted_local_motion_loss(
    pred: Tensor,
    target: Tensor,
    target_raw: Tensor,
    *,
    gripper_index: int,
    threshold: float,
    boost: float,
) -> tuple[Tensor, Tensor]:
    per_dim = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    event = target_raw[..., gripper_index].abs() >= float(threshold)
    weight = torch.ones_like(per_dim)
    weight[..., gripper_index] = 1.0 + float(boost) * event.float()
    loss = (per_dim * weight).sum() / weight.sum().clamp_min(1.0)
    return loss, event.float().mean()


def compute_representation_losses(
    model: DynamicPredictiveWorld,
    batch: PreparedRepresentationBatch,
    *,
    config: DynamicRepresentationLossConfig,
) -> dict[str, Tensor]:
    config.validate()
    tokens, states, states_raw = _stack_windows(batch)
    batch_size, window_count = tokens.shape[:2]
    flat_tokens = tokens.reshape(batch_size * window_count, *tokens.shape[2:])
    flat_states = states.reshape(batch_size * window_count, *states.shape[2:])
    flat_states_raw = states_raw.reshape(batch_size * window_count, *states_raw.shape[2:])

    output = model.representation_outputs(flat_tokens)
    target_descriptor = model.fixed_dynamic_descriptor(flat_tokens).to(
        device=flat_tokens.device, dtype=output["descriptor"].dtype
    )
    descriptor = F.smooth_l1_loss(output["descriptor"].float(), target_descriptor.float())

    local_motion_target = flat_states[:, -1] - flat_states[:, 0]
    local_motion_target_raw = flat_states_raw[:, -1] - flat_states_raw[:, 0]
    local_motion, gripper_event_fraction = _weighted_local_motion_loss(
        output["local_motion"],
        local_motion_target,
        local_motion_target_raw,
        gripper_index=model.config.gripper_index,
        threshold=config.gripper_transition_threshold,
        boost=config.gripper_transition_boost,
    )
    context_state_target = flat_states[:, -1]
    context_state = F.smooth_l1_loss(output["context_state"].float(), context_state_target.float())

    pred_descriptor_sequence = output["descriptor"].reshape(batch_size, window_count, -1)
    target_descriptor_sequence = target_descriptor.reshape(batch_size, window_count, -1)
    pred_increment = pred_descriptor_sequence[:, 1:] - pred_descriptor_sequence[:, :-1]
    target_increment = target_descriptor_sequence[:, 1:] - target_descriptor_sequence[:, :-1]
    temporal_increment = F.smooth_l1_loss(pred_increment.float(), target_increment.float())
    temporal_increment_cosine = F.cosine_similarity(
        pred_increment.float(), target_increment.float(), dim=-1
    ).mean()

    dynamic_pooled = output["dynamic"].mean(dim=1)
    context_pooled = output["context"].mean(dim=1)
    dynamic_variance, dynamic_std = _variance_loss(dynamic_pooled, config.embedding_std_target)
    context_variance, context_std = _variance_loss(context_pooled, config.embedding_std_target)
    variance = 0.5 * (dynamic_variance + context_variance)
    covariance = 0.5 * (_covariance_loss(dynamic_pooled) + _covariance_loss(context_pooled))
    token_diversity = _token_diversity_loss(output["dynamic"])

    total = (
        config.descriptor_weight * descriptor
        + config.local_motion_weight * local_motion
        + config.context_state_weight * context_state
        + config.temporal_increment_weight * temporal_increment
        + config.variance_weight * variance
        + config.covariance_weight * covariance
        + config.token_diversity_weight * token_diversity
    )
    descriptor_cosine = F.cosine_similarity(
        output["descriptor"].float(), target_descriptor.float(), dim=-1
    ).mean()
    local_motion_rmse = F.mse_loss(
        output["local_motion"].float(), local_motion_target.float()
    ).sqrt()
    context_state_rmse = F.mse_loss(
        output["context_state"].float(), context_state_target.float()
    ).sqrt()
    return {
        "loss": total,
        "descriptor": descriptor,
        "descriptor_cosine": descriptor_cosine,
        "local_motion": local_motion,
        "local_motion_rmse": local_motion_rmse,
        "context_state": context_state,
        "context_state_rmse": context_state_rmse,
        "temporal_increment": temporal_increment,
        "temporal_increment_cosine": temporal_increment_cosine,
        "variance": variance,
        "dynamic_std": dynamic_std,
        "context_std": context_std,
        "covariance": covariance,
        "token_diversity": token_diversity,
        "gripper_event_fraction": gripper_event_fraction,
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def evaluate_dynamic_representation(
    *,
    model: DynamicPredictiveWorld,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    loss_config: DynamicRepresentationLossConfig,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch_index, raw_batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        batch = prepare_representation_batch(
            raw_batch,
            conditioner=conditioner,
            model=model,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        losses = compute_representation_losses(model, batch, config=loss_config)
        rows.append({key: float(value.detach().float().cpu()) for key, value in losses.items()})
    if not rows:
        raise RuntimeError("representation evaluation produced no batches")
    return {f"val_{key}": value for key, value in _mean_rows(rows).items()}


def _checkpoint_payload(
    *,
    model: DynamicPredictiveWorld,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    context: dict[str, Any],
    history: list[dict[str, Any]],
    trainer: DynamicRepresentationTrainerConfig,
    loss_config: DynamicRepresentationLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
) -> dict[str, Any]:
    return {
        "schema": "clearvla-v33.4-dynamic-representation-checkpoint-v1",
        "representation": model.representation_state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "context": context,
        "trainer": asdict(trainer),
        "loss_config": asdict(loss_config),
        "action_normalizer": action_normalizer.to_dict(),
        "state_normalizer": state_normalizer.to_dict(),
        "history": history,
    }


def train_dynamic_representation(
    *,
    model: DynamicPredictiveWorld,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    out_dir: Path,
    trainer: DynamicRepresentationTrainerConfig,
    loss_config: DynamicRepresentationLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
) -> dict[str, Any]:
    if model.representation_frozen:
        raise ValueError("representation pretraining requires an unfrozen representation")
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters = [
        parameter
        for module in (
            model.online_encoder,
            model.descriptor_head,
            model.local_motion_head,
            model.context_state_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=trainer.lr,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
        weight_decay=trainer.weight_decay,
    )
    steps_per_epoch = (
        min(len(train_loader), trainer.max_train_batches)
        if trainer.max_train_batches
        else len(train_loader)
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    history: list[dict[str, Any]] = []
    best_descriptor = float("inf")
    best_physical = float("inf")
    best_representation = float("inf")
    global_step = 0
    epoch_path = out_dir / "dynamic_representation_epochs.jsonl"
    if epoch_path.exists():
        epoch_path.unlink()

    for epoch in range(1, trainer.epochs + 1):
        model.train()
        start = time.perf_counter()
        rows: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            batch = prepare_representation_batch(
                raw_batch,
                conditioner=conditioner,
                model=model,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            losses = compute_representation_losses(model, batch, config=loss_config)
            losses["loss"].backward()
            raw_grad = _grad_norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, trainer.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = raw_grad
            rows.append(row)
            if batch_index % trainer.log_every == 0:
                mean = _mean_rows(rows[-trainer.log_every :])
                print(
                    "[dynamic-representation] "
                    f"epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={mean['loss']:.6f} desc={mean['descriptor']:.6f}/"
                    f"{mean['descriptor_cosine']:.3f} motion={mean['local_motion_rmse']:.5f} "
                    f"state={mean['context_state_rmse']:.5f} "
                    f"inc={mean['temporal_increment']:.6f}/"
                    f"{mean['temporal_increment_cosine']:.3f} "
                    f"std={mean['dynamic_std']:.3f}/{mean['context_std']:.3f} "
                    f"div={mean['token_diversity']:.4f} grad={mean['grad']:.3e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

        train_metrics = _mean_rows(rows)
        val_metrics = evaluate_dynamic_representation(
            model=model,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            loss_config=loss_config,
            max_batches=trainer.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        _append_jsonl(epoch_path, record)
        print(
            "[dynamic-representation] "
            f"epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.1f} "
            f"val_desc={val_metrics['val_descriptor']:.6f}/"
            f"{val_metrics['val_descriptor_cosine']:.3f} "
            f"motion={val_metrics['val_local_motion_rmse']:.5f} "
            f"state={val_metrics['val_context_state_rmse']:.5f} "
            f"inc={val_metrics['val_temporal_increment']:.6f}/"
            f"{val_metrics['val_temporal_increment_cosine']:.3f} "
            f"std={val_metrics['val_dynamic_std']:.3f}/"
            f"{val_metrics['val_context_std']:.3f}",
            flush=True,
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            context=context,
            history=history,
            trainer=trainer,
            loss_config=loss_config,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        _save(out_dir / "checkpoints/latest.pt", payload)

        descriptor_score = (
            val_metrics["val_descriptor"] + 0.5 * val_metrics["val_temporal_increment"]
        )
        physical_score = (
            val_metrics["val_local_motion_rmse"] + 0.5 * val_metrics["val_context_state_rmse"]
        )
        std_penalty = max(0.0, loss_config.embedding_std_target - val_metrics["val_dynamic_std"])
        representation_score = (
            descriptor_score
            + 0.5 * physical_score
            + 0.25 * max(0.0, 1.0 - val_metrics["val_descriptor_cosine"])
            + 0.25 * max(0.0, 1.0 - val_metrics["val_temporal_increment_cosine"])
            + std_penalty
        )
        if descriptor_score < best_descriptor:
            best_descriptor = descriptor_score
            _save(out_dir / "checkpoints/best_descriptor.pt", payload)
        if physical_score < best_physical:
            best_physical = physical_score
            _save(out_dir / "checkpoints/best_physical.pt", payload)
        if representation_score < best_representation:
            best_representation = representation_score
            _save(out_dir / "checkpoints/best_representation.pt", payload)

    summary = {
        "schema": "clearvla-v33.4-dynamic-representation-summary-v1",
        "representation_parameter_count": sum(parameter.numel() for parameter in parameters),
        "best_descriptor_score": best_descriptor,
        "best_physical_score": best_physical,
        "best_representation_score": best_representation,
        "history": history,
        "context": context,
    }
    (out_dir / "dynamic_representation_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "DynamicRepresentationLossConfig",
    "DynamicRepresentationTrainerConfig",
    "PreparedRepresentationBatch",
    "prepare_representation_batch",
    "compute_representation_losses",
    "evaluate_dynamic_representation",
    "train_dynamic_representation",
]
