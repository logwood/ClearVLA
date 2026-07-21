from __future__ import annotations

"""Training and evaluation runtime for the v20 grounded shallow motor policy."""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def _timer_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _condition_dense(condition) -> tuple[torch.Tensor, torch.Tensor]:
    if condition.dense_tokens is None:
        raise ValueError(
            "v20 grounded-motor requires dense condition tokens. "
            "Use debug-dense, dinov2, or dinov2-cache; raw per-layer KV caches "
            "would reintroduce the repeated-conditioning path this branch removes."
        )
    return condition.dense_tokens, condition.attention_mask


def _concat(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("cannot concatenate empty metric rows")
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def evaluate_grounded_motor_rdt2_fm(
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
    eval_seed: int = 0,
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
    dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(loader.dataset, "camera_names", ("top", "wrist")))
    full_rows: list[np.ndarray] = []
    first_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    prior_rows: list[np.ndarray] = []
    past_rows: list[np.ndarray] = []
    losses: list[dict[str, float]] = []
    fast_ms: list[float] = []
    full_ms: list[float] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_seed))
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        state = batch["state"].to(device=device, dtype=dtype, non_blocking=True)
        past = batch["past"].to(device=device, dtype=dtype, non_blocking=True)
        action = batch["action"].to(device=device, dtype=dtype, non_blocking=True)
        images = batch["obs_image"].to(device=device, non_blocking=True)
        sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        conditioner_ablation = image_ablation
        if image_ablation == "shuffle-episode":
            sample_keys = loader.dataset.cross_episode_keys(
                sample_keys, seed=batch_index + eval_seed
            )
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
        ).to(device=device, dtype=dtype)
        dense, mask = _condition_dense(condition)
        loss = model.compute_loss(
            state_tokens=state,
            past_actions=past,
            action_gt=action,
            dense_tokens=dense,
            attention_mask=mask,
        )
        losses.append({key: float(value.detach().float().cpu()) for key, value in loss.items()})
        _timer_sync(device)
        started = time.perf_counter()
        first = model.predict_first_action(
            state_tokens=state,
            past_actions=past,
            dense_tokens=dense,
            attention_mask=mask,
            generator=generator,
            inference_steps=inference_steps,
        )
        _timer_sync(device)
        fast_ms.append((time.perf_counter() - started) * 1000)
        _timer_sync(device)
        started = time.perf_counter()
        full = model.predict_action(
            state_tokens=state,
            past_actions=past,
            dense_tokens=dense,
            attention_mask=mask,
            generator=generator,
            inference_steps=inference_steps,
        )
        _timer_sync(device)
        full_ms.append((time.perf_counter() - started) * 1000)
        prior = action_normalizer.encode(batch["prior_raw"].numpy())
        past_norm = action_normalizer.encode(batch["past_raw"].numpy())
        first_chunk = torch.from_numpy(prior).to(device=device, dtype=dtype)
        first_chunk[:, 0] = first
        full_rows.append(full.float().cpu().numpy())
        first_rows.append(first_chunk.float().cpu().numpy())
        target_rows.append(action.float().cpu().numpy())
        prior_rows.append(prior)
        past_rows.append(past_norm)
    pred = _concat(full_rows)
    target = _concat(target_rows)
    prior = _concat(prior_rows)
    past = _concat(past_rows)
    metrics = compute_metrics(
        pred_norm=pred,
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    fast_metrics = compute_metrics(
        pred_norm=_concat(first_rows),
        target_norm=target,
        prior_norm=prior,
        past_norm=past,
        normalizer=action_normalizer,
    )
    metrics.update(
        {
            "inference_steps": int(inference_steps),
            "image_ablation": str(image_ablation),
            "native_first_rmse": fast_metrics["first_rmse"],
            "native_first_arm_rmse": fast_metrics.get("arm_first_rmse", fast_metrics["first_rmse"]),
            "latency_native_first_ms": float(np.mean(fast_ms)),
            "latency_full_chunk_ms": float(np.mean(full_ms)),
        }
    )
    if losses:
        for key in losses[0]:
            metrics[f"val_{key}"] = float(np.mean([row[key] for row in losses]))
    return metrics


def train_grounded_motor_rdt2_fm(
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
    eval_seed: int = 0,
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
    dtype = next(model.parameters()).dtype
    best_full = float("inf")
    best_first = float("inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    cameras = tuple(getattr(train_loader.dataset, "camera_names", ("top", "wrist")))
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        if hasattr(conditioner, "eval"):
            conditioner.eval()
        started = time.perf_counter()
        batch_rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            state = batch["state"].to(device=device, dtype=dtype, non_blocking=True)
            past = batch["past"].to(device=device, dtype=dtype, non_blocking=True)
            action = batch["action"].to(device=device, dtype=dtype, non_blocking=True)
            images = batch["obs_image"].to(device=device, non_blocking=True)
            sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
            with torch.no_grad():
                condition = conditioner.encode(
                    images,
                    [instruction] * state.shape[0],
                    sample_keys=sample_keys,
                    image_ablation="normal",
                    camera_names=cameras,
                ).to(device=device, dtype=dtype)
            dense, mask = _condition_dense(condition)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss(
                state_tokens=state,
                past_actions=past,
                action_gt=action,
                dense_tokens=dense,
                attention_mask=mask,
            )
            losses["loss"].backward()
            if trainer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            grad = _grad_norm(model.parameters())
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = grad
            row["lr"] = float(optimizer.param_groups[0]["lr"])
            batch_rows.append(row)
            if batch_index % trainer.log_every == 0:
                print(
                    f"[rdt2-grounded] epoch={epoch:03d}/{trainer.epochs:03d} "
                    f"batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"full={row['full_flow_mse']:.6f} first={row['first_flow_mse']:.6f} "
                    f"anchor={row['first_anchor_rms']:.4e} grounding={row['grounding_rms']:.4e} "
                    f"grad={row['grad']:.3e} lr={row['lr']:.3e}",
                    flush=True,
                )
        record: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": {
                key: float(np.mean([row[key] for row in batch_rows])) for key in batch_rows[0]
            },
        }
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_grounded_motor_rdt2_fm(
                model,
                conditioner,
                val_loader,
                device=device,
                action_normalizer=action_normalizer,
                inference_steps=inference_steps,
                max_batches=trainer.max_val_batches,
                instruction=instruction,
                eval_seed=eval_seed,
            )
            record["val"] = metrics
            print(
                f"[rdt2-grounded] epoch={epoch:03d}/{trainer.epochs:03d} "
                f"sec={record['seconds']:.2f} val_full={metrics['full_mse']:.6f} "
                f"arm_first={metrics.get('arm_first_rmse', metrics['first_rmse']):.6f} "
                f"native_first={metrics['native_first_arm_rmse']:.6f} "
                f"first4={metrics['first4_rmse']:.6f} first8={metrics['first8_rmse']:.6f} "
                f"prior={metrics['prior_full_mse']:.6f} "
                f"fast_ms={metrics['latency_native_first_ms']:.2f} full_ms={metrics['latency_full_chunk_ms']:.2f}",
                flush=True,
            )
            payload = {
                "model": model.state_dict(),
                "context": context,
                "action_normalizer": action_normalizer.to_dict(),
                "state_normalizer": state_normalizer.to_dict(),
                "history": history + [record],
            }
            _save(out_dir / "last.pt", payload)
            if metrics["full_mse"] < best_full:
                best_full = float(metrics["full_mse"])
                _save(out_dir / "best_full.pt", payload)
            if metrics["native_first_arm_rmse"] < best_first:
                best_first = float(metrics["native_first_arm_rmse"])
                _save(out_dir / "best_first.pt", payload)
        history.append(record)
        (out_dir / "history.json").write_text(
            json.dumps(_jsonable(history), indent=2), encoding="utf-8"
        )
    return {
        "history": history,
        "best_full_mse": best_full,
        "best_native_first_arm_rmse": best_first,
    }


__all__ = ["evaluate_grounded_motor_rdt2_fm", "train_grounded_motor_rdt2_fm"]
