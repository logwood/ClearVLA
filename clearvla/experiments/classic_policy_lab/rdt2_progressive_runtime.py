from __future__ import annotations

"""Training and evaluation runtime for the v19 progressive RDT2-FM model."""

import json
import math
import time
from pathlib import Path
from typing import Any

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
    return value


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


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


@torch.no_grad()
def evaluate_progressive_rdt2_fm(
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
    allowed = {
        "normal",
        "zero",
        "mean",
        "shuffle-batch",
        "shuffle-episode",
        "top-only",
        "wrist-only",
    }
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
            images = loader.dataset.load_images_for_keys(sample_keys).to(
                device=device, non_blocking=True
            )
            conditioner_ablation = "normal"
        condition = conditioner.encode(
            images,
            [instruction] * state.shape[0],
            sample_keys=sample_keys,
            image_ablation=conditioner_ablation,
            camera_names=camera_names,
        ).to(device=device, dtype=model_dtype)
        kwargs = _condition_kwargs(condition)
        loss = model.compute_loss(
            state_tokens=state,
            past_actions=past,
            physical_prior=hold,
            action_gt=action,
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
        learned_prior, _ = model.predict_prior(
            state_tokens=state, past_actions=past, physical_prior=hold
        )
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
    metrics = compute_metrics(
        pred_norm=pred,
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    learned_metrics = compute_metrics(
        pred_norm=_concat(learned_prior_rows),
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    fast_metrics = compute_metrics(
        pred_norm=_concat(fast_rows),
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    prefix_metrics = compute_metrics(
        pred_norm=_concat(prefix_rows),
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    metrics.update(
        {
            "inference_steps": int(inference_steps),
            "image_ablation": str(image_ablation),
            "learned_prior_full_mse": learned_metrics["full_mse"],
            "learned_prior_arm_first_rmse": learned_metrics.get(
                "arm_first_rmse", learned_metrics["first_rmse"]
            ),
            "fast_exit_arm_first_rmse": fast_metrics.get(
                "arm_first_rmse", fast_metrics["first_rmse"]
            ),
            "fast_exit_first_rmse": fast_metrics["first_rmse"],
            "prefix_exit_first4_rmse": prefix_metrics["first4_rmse"],
            "prefix_exit_first8_rmse": prefix_metrics["first8_rmse"],
            "latency_fast_ms": float(np.mean(latency["fast_ms"])),
            "latency_prefix_ms": float(np.mean(latency["prefix_ms"])),
            "latency_full_ms": float(np.mean(latency["full_ms"])),
        }
    )
    for key in losses[0]:
        metrics[f"val_{key}"] = float(np.mean([row[key] for row in losses]))
    return metrics


def train_progressive_rdt2_fm(
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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    scheduler = _rdt_scheduler(
        optimizer,
        scheduler=trainer.scheduler,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    best_full = float("inf")
    best_fast = float("inf")
    history = []
    global_step = 0
    dtype = next(model.parameters()).dtype
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
                    camera_names=tuple(
                        getattr(train_loader.dataset, "camera_names", ("top", "wrist"))
                    ),
                ).to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                state_tokens=state,
                past_actions=past,
                physical_prior=hold,
                action_gt=action,
                **_condition_kwargs(condition),
            )
            loss["loss"].backward()
            grad = _grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in loss.items()}
            row["grad"] = grad
            rows.append(row)
            if batch_index % trainer.log_every == 0:
                latest = rows[-trainer.log_every :]
                avg = {key: float(np.mean([item[key] for item in latest])) for key in latest[0]}
                print(
                    f"[rdt2-progressive] epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={avg['loss']:.6f} flow={avg['full_flow_mse']:.6f} prior={avg['prior_mse']:.6f} "
                    f"first={avg['fast_first_flow_mse']:.6f} prefix={avg['prefix_flow_mse']:.6f} "
                    f"visual={avg['visual_correction_rms']:.4e} grad={avg['grad']:.3e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        metrics: dict[str, Any] = {}
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_progressive_rdt2_fm(
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
        if metrics:
            print(
                f"[rdt2-progressive] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} "
                f"val_full={metrics['full_mse']:.6f} arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} "
                f"fast_first={metrics['fast_exit_arm_first_rmse']:.6f} prefix4={metrics['prefix_exit_first4_rmse']:.6f} "
                f"learned_prior={metrics['learned_prior_full_mse']:.6f} hold={metrics['hold_last_full_mse']:.6f} "
                f"latency_ms={metrics['latency_fast_ms']:.2f}/{metrics['latency_prefix_ms']:.2f}/{metrics['latency_full_ms']:.2f}",
                flush=True,
            )
        payload = {
            "schema": "clearvla-rdt2-progressive-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
    summary = {
        "schema": "clearvla-rdt2-progressive-summary-v1",
        "parameter_count": model.parameter_count(),
        "best_full_mse": best_full,
        "best_fast_first_arm_rmse": best_fast,
        "history": history,
        "context": context,
    }
    (out_dir / "rdt2_progressive_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = ["evaluate_progressive_rdt2_fm", "train_progressive_rdt2_fm"]
