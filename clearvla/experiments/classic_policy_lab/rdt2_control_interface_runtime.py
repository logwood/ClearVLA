from __future__ import annotations

"""Training and evaluation runtime for the v21 control-interface experiment."""

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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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
            "v21 control-interface requires dense condition tokens. "
            "Use debug-dense, dinov2, or dinov2-cache; per-layer KV caches bypass "
            "the scene compiler and are intentionally rejected."
        )
    return condition.dense_tokens, condition.attention_mask


def _concat(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("cannot concatenate empty metric rows")
    return np.concatenate(rows, axis=0)


def _tensor_mean(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().mean(axis=0)


def _merge_interface_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    names = tuple(rows[0]["source_group_names"])
    flow_count = len(rows[0]["flow_steps"])
    merged_steps = []
    for step in range(flow_count):
        keys = set()
        for row in rows:
            keys.update(row["flow_steps"][step].keys())
        merged: dict[str, Any] = {"step": step}
        for key in sorted(keys):
            values = [row["flow_steps"][step][key] for row in rows if key in row["flow_steps"][step]]
            if not values:
                continue
            arrays = [_tensor_mean(value) if torch.is_tensor(value) else np.asarray(value) for value in values]
            merged[key] = np.stack(arrays, axis=0).mean(axis=0).tolist()
        merged_steps.append(merged)
    return {
        "source_group_names": list(names),
        "flow_steps": merged_steps,
        "interpretation": {
            "effective_source_mass": "mean dynamic control-readout attention projected back to task/camera source groups",
            "control_attention_entropy": "mean entropy over scene-memory attention for control queries",
            "action_summary_entropy": "mean entropy over noisy-action positions for action-summary queries",
            "scene_source_mass": "mean scene-compiler attention allocated to task/camera source groups",
        },
    }


@torch.no_grad()
def evaluate_control_interface_rdt2_fm(
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
    collect_diagnostics: bool = False,
    diagnostic_batches: int = 8,
) -> dict[str, Any]:
    allowed = {"normal", "zero", "mean", "shuffle-batch", "shuffle-episode", "top-only", "wrist-only"}
    if image_ablation not in allowed:
        raise ValueError(f"unsupported image_ablation={image_ablation!r}")
    model.eval()
    if hasattr(conditioner, "eval"):
        conditioner.eval()
    dtype = next(model.parameters()).dtype
    camera_names = tuple(getattr(loader.dataset, "camera_names", ("top", "wrist")))
    full_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    prior_rows: list[np.ndarray] = []
    past_rows: list[np.ndarray] = []
    losses: list[dict[str, float]] = []
    latencies: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_seed))
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        state = batch["state"].to(device=device, dtype=dtype, non_blocking=True)
        action = batch["action"].to(device=device, dtype=dtype, non_blocking=True)
        images = batch["obs_image"].to(device=device, non_blocking=True)
        sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        conditioner_ablation = image_ablation
        if image_ablation == "shuffle-episode":
            sample_keys = loader.dataset.cross_episode_keys(sample_keys, seed=batch_index + eval_seed)
            images = loader.dataset.load_images_for_keys(sample_keys).to(device=device, non_blocking=True)
            conditioner_ablation = "normal"
        condition = conditioner.encode(
            images,
            [instruction] * state.shape[0],
            sample_keys=sample_keys,
            image_ablation=conditioner_ablation,
            camera_names=camera_names,
        ).to(device=device, dtype=dtype)
        dense, mask = _condition_dense(condition)
        loss = model.compute_loss(state_tokens=state, action_gt=action, dense_tokens=dense, attention_mask=mask)
        losses.append({key: float(value.detach().float().cpu()) for key, value in loss.items()})
        need_diagnostics = collect_diagnostics and len(diagnostic_rows) < max(int(diagnostic_batches), 0)
        _timer_sync(device)
        started = time.perf_counter()
        result = model.predict_action(
            state_tokens=state,
            dense_tokens=dense,
            attention_mask=mask,
            generator=generator,
            inference_steps=inference_steps,
            return_diagnostics=need_diagnostics,
        )
        _timer_sync(device)
        latencies.append((time.perf_counter() - started) * 1000)
        if need_diagnostics:
            full, diagnostics = result
            diagnostic_rows.append(diagnostics)
        else:
            full = result
        prior = action_normalizer.encode(batch["prior_raw"].numpy())
        past_norm = action_normalizer.encode(batch["past_raw"].numpy())
        full_rows.append(full.float().cpu().numpy())
        target_rows.append(action.float().cpu().numpy())
        prior_rows.append(prior)
        past_rows.append(past_norm)
    pred = _concat(full_rows)
    target = _concat(target_rows)
    prior = _concat(prior_rows)
    past = _concat(past_rows)
    metrics = compute_metrics(pred_norm=pred, target_norm=target, prior_norm=prior, past_norm=past, normalizer=action_normalizer)
    metrics.update({
        "inference_steps": int(inference_steps),
        "image_ablation": str(image_ablation),
        "latency_full_chunk_ms": float(np.mean(latencies)),
    })
    if losses:
        for key in losses[0]:
            metrics[f"val_{key}"] = float(np.mean([row[key] for row in losses]))
    diagnostics = _merge_interface_diagnostics(diagnostic_rows)
    if diagnostics is not None:
        metrics["condition_interface_diagnostics"] = diagnostics
    return metrics


def train_control_interface_rdt2_fm(
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
    collect_diagnostics: bool = False,
    diagnostic_batches: int = 8,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.lr,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
        weight_decay=trainer.weight_decay,
    )
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    scheduler = _rdt_scheduler(
        optimizer,
        scheduler=trainer.scheduler,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    dtype = next(model.parameters()).dtype
    best_full = float("inf")
    history: list[dict[str, Any]] = []
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
            losses = model.compute_loss(state_tokens=state, action_gt=action, dense_tokens=dense, attention_mask=mask)
            losses["loss"].backward()
            if trainer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            grad = _grad_norm(model.parameters())
            optimizer.step()
            scheduler.step()
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = grad
            row["lr"] = float(optimizer.param_groups[0]["lr"])
            batch_rows.append(row)
            if batch_index % trainer.log_every == 0:
                print(
                    f"[rdt2-control] mode={model.config.interface_mode} epoch={epoch:03d}/{trainer.epochs:03d} "
                    f"batch={batch_index:04d} flow={row['flow_mse']:.6f} scene={row['scene_rms']:.4e} "
                    f"control={row['control_rms']:.4e} grad={row['grad']:.3e} lr={row['lr']:.3e}",
                    flush=True,
                )
        record: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": {key: float(np.mean([row[key] for row in batch_rows])) for key in batch_rows[0]},
        }
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_control_interface_rdt2_fm(
                model,
                conditioner,
                val_loader,
                device=device,
                action_normalizer=action_normalizer,
                inference_steps=inference_steps,
                max_batches=trainer.max_val_batches,
                instruction=instruction,
                eval_seed=eval_seed,
                collect_diagnostics=collect_diagnostics,
                diagnostic_batches=diagnostic_batches,
            )
            record["val"] = metrics
            print(
                f"[rdt2-control] mode={model.config.interface_mode} epoch={epoch:03d}/{trainer.epochs:03d} "
                f"sec={record['seconds']:.2f} val_full={metrics['full_mse']:.6f} "
                f"arm_first={metrics.get('arm_first_rmse', metrics['first_rmse']):.6f} "
                f"first4={metrics['first4_rmse']:.6f} first8={metrics['first8_rmse']:.6f} "
                f"prior={metrics['prior_full_mse']:.6f} full_ms={metrics['latency_full_chunk_ms']:.2f}",
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
        history.append(record)
        (out_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
    return {"history": history, "best_full_mse": best_full}


__all__ = ["evaluate_control_interface_rdt2_fm", "train_control_interface_rdt2_fm"]
