from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .act_reference import ACTReference
from .dp_reference import DPReference, EMAModel
from .evaluation import evaluate_act, evaluate_dp, evaluate_rdt_small, evaluate_rdt2_fm
from .normalizer import ArrayNormalizer


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-4)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    log_every: int = 50
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_every: int = 1
    ema_decay: float = 0.999


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_act(
    *,
    model: ACTReference,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    trainer: TrainerConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay)
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    lr_sched = _scheduler(optimizer, total_steps=trainer.epochs * steps_per_epoch, warmup_steps=trainer.warmup_steps, min_lr_ratio=trainer.min_lr_ratio)
    best_full = float("inf")
    best_arm_first = float("inf")
    history = []
    global_step = 0
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        start = time.perf_counter()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            qpos = batch["qpos"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            is_pad = batch["is_pad"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(qpos, image, actions, is_pad)
            loss["loss"].backward()
            raw_grad = _grad_norm(model.parameters())
            clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip))
            optimizer.step(); lr_sched.step(); global_step += 1
            rows.append({key: float(value.detach().cpu()) for key, value in loss.items()} | {"grad": raw_grad})
            if batch_index % trainer.log_every == 0:
                mean = {key: float(np.mean([row[key] for row in rows[-trainer.log_every:]])) for key in rows[-1]}
                print(f"[act] epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} loss={mean['loss']:.6f} l1={mean['l1']:.6f} kl={mean['kl']:.6f} grad={mean['grad']:.3e} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        metrics = evaluate_act(model, val_loader, device=device, action_normalizer=action_normalizer, max_batches=trainer.max_val_batches)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "val": metrics,
        }
        history.append(record)
        print(f"[act] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} val_full={metrics['full_mse']:.6f} arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} first4={metrics['first4_rmse']:.6f} prior={metrics['prior_full_mse']:.6f} replay={metrics['history_replay_full_mse']:.6f} l1={metrics['val_l1']:.6f} kl={metrics['val_kl']:.6f}", flush=True)
        payload = {"schema": "clearvla-act-reference-checkpoint-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": lr_sched.state_dict(), "epoch": epoch, "global_step": global_step, "context": context, "action_normalizer": action_normalizer.to_dict(), "state_normalizer": state_normalizer.to_dict(), "history": history}
        _save(out_dir / "checkpoints/latest.pt", payload)
        if metrics["full_mse"] < best_full:
            best_full = metrics["full_mse"]; _save(out_dir / "checkpoints/best_full.pt", payload)
        arm_first = metrics.get("arm_first_rmse", metrics["first_rmse"])
        if arm_first < best_arm_first:
            best_arm_first = arm_first; _save(out_dir / "checkpoints/best_arm_first.pt", payload)
    summary = {"schema": "clearvla-act-reference-summary-v1", "parameter_count": model.parameter_count(), "best_full_mse": best_full, "best_arm_first_rmse": best_arm_first, "history": history, "context": context}
    (out_dir / "act_reference_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def train_dp(
    *,
    model: DPReference,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    trainer: TrainerConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
    inference_steps: int,
    use_ema: bool,
    deterministic_sampling: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=trainer.weight_decay)
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    lr_sched = _scheduler(optimizer, total_steps=trainer.epochs * steps_per_epoch, warmup_steps=trainer.warmup_steps, min_lr_ratio=trainer.min_lr_ratio)
    ema = EMAModel(model, decay=trainer.ema_decay) if use_ema else None
    best_full = float("inf")
    best_arm_first = float("inf")
    history = []
    global_step = 0
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        start = time.perf_counter()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            obs_image = batch["obs_image"].to(device, non_blocking=True)
            obs_state = batch["obs_state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(obs_image, obs_state, action)
            loss.backward()
            raw_grad = _grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            optimizer.step(); lr_sched.step(); global_step += 1
            if ema is not None:
                ema.step(model)
            rows.append({"loss": float(loss.detach().cpu()), "grad": raw_grad})
            if batch_index % trainer.log_every == 0:
                mean_loss = float(np.mean([row["loss"] for row in rows[-trainer.log_every:]]))
                mean_grad = float(np.mean([row["grad"] for row in rows[-trainer.log_every:]]))
                print(f"[dp] epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} noise_mse={mean_loss:.6f} grad={mean_grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        eval_model = ema.averaged_model if ema is not None else model
        metrics = evaluate_dp(eval_model, val_loader, device=device, action_normalizer=action_normalizer, inference_steps=inference_steps, max_batches=trainer.max_val_batches, deterministic=deterministic_sampling)
        record = {"epoch": epoch, "global_step": global_step, "seconds": time.perf_counter() - start, "train": {"loss": float(np.mean([row["loss"] for row in rows])), "grad": float(np.mean([row["grad"] for row in rows]))}, "val": metrics}
        history.append(record)
        print(f"[dp] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} val_full={metrics['full_mse']:.6f} arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} first4={metrics['first4_rmse']:.6f} prior={metrics['prior_full_mse']:.6f} replay={metrics['history_replay_full_mse']:.6f} denoise={metrics['val_denoise_loss']:.6f} sample_steps={inference_steps}", flush=True)
        payload = {"schema": "clearvla-dp-reference-checkpoint-v1", "model": model.state_dict(), "ema_model": None if ema is None else ema.averaged_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": lr_sched.state_dict(), "epoch": epoch, "global_step": global_step, "context": context, "action_normalizer": action_normalizer.to_dict(), "state_normalizer": state_normalizer.to_dict(), "history": history}
        _save(out_dir / "checkpoints/latest.pt", payload)
        if metrics["full_mse"] < best_full:
            best_full = metrics["full_mse"]; _save(out_dir / "checkpoints/best_full.pt", payload)
        arm_first = metrics.get("arm_first_rmse", metrics["first_rmse"])
        if arm_first < best_arm_first:
            best_arm_first = arm_first; _save(out_dir / "checkpoints/best_arm_first.pt", payload)
    summary = {"schema": "clearvla-dp-reference-summary-v1", "parameter_count": model.parameter_count(), "best_full_mse": best_full, "best_arm_first_rmse": best_arm_first, "history": history, "context": context}
    (out_dir / "dp_reference_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class RDTTrainerConfig:
    """Single-GPU reference trainer matching the released RDT fine-tune recipe."""

    epochs: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_every: int = 1


def _rdt_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    if scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler == "constant_with_warmup":
        def scale(step: int) -> float:
            return min((step + 1) / max(warmup_steps, 1), 1.0)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    if scheduler == "cosine":
        return _scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    raise ValueError(f"unsupported RDT scheduler: {scheduler}")


def train_rdt_small(
    *,
    model,
    vision_encoder,
    language_conditioner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    trainer: RDTTrainerConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
    inference_steps: int,
    sampler: str,
    deterministic_sampling: bool = True,
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
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    lr_sched = _rdt_scheduler(
        optimizer,
        scheduler=trainer.scheduler,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    best_full = float("inf")
    best_arm_first = float("inf")
    history = []
    global_step = 0
    model_dtype = next(model.parameters()).dtype
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        vision_encoder.eval()
        start = time.perf_counter()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            state = batch["state"].to(device, dtype=model_dtype, non_blocking=True)
            images = batch["obs_image"].to(device, non_blocking=True)
            actions = batch["action"].to(device, dtype=model_dtype, non_blocking=True)
            ctrl_freqs = batch["ctrl_freq"].to(device, dtype=model_dtype, non_blocking=True)
            with torch.no_grad():
                img_tokens = vision_encoder(images).to(device=device, dtype=model_dtype)
                lang_tokens, lang_mask = language_conditioner.batch(
                    state.shape[0], device=device, dtype=model_dtype
                )
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                state=state,
                actions=actions,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                img_tokens=img_tokens,
                ctrl_freqs=ctrl_freqs,
            )
            loss.backward()
            raw_grad = _grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            optimizer.step(); lr_sched.step(); global_step += 1
            rows.append({"loss": float(loss.detach().cpu()), "grad": raw_grad})
            if batch_index % trainer.log_every == 0:
                mean_loss = float(np.mean([row["loss"] for row in rows[-trainer.log_every:]]))
                mean_grad = float(np.mean([row["grad"] for row in rows[-trainer.log_every:]]))
                print(
                    f"[rdt-small] epoch={epoch:03d}/{trainer.epochs:03d} "
                    f"batch={batch_index:04d} sample_mse={mean_loss:.6f} "
                    f"grad={mean_grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_rdt_small(
                model,
                vision_encoder,
                language_conditioner,
                val_loader,
                device=device,
                action_normalizer=action_normalizer,
                inference_steps=inference_steps,
                sampler=sampler,
                max_batches=trainer.max_val_batches,
                deterministic=deterministic_sampling,
                eval_seed=eval_seed,
            )
        else:
            metrics = {}
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": {
                "loss": float(np.mean([row["loss"] for row in rows])),
                "grad": float(np.mean([row["grad"] for row in rows])),
            },
            "val": metrics,
        }
        history.append(record)
        if metrics:
            print(
                f"[rdt-small] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} "
                f"val_full={metrics['full_mse']:.6f} "
                f"arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} "
                f"first4={metrics['first4_rmse']:.6f} "
                f"first8={metrics['first8_rmse']:.6f} "
                f"prior={metrics['prior_full_mse']:.6f} "
                f"denoise={metrics['val_denoise_loss']:.6f} "
                f"sample_steps={inference_steps} sampler={sampler}",
                flush=True,
            )
        payload = {
            "schema": "clearvla-rdt-small-reference-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_sched.state_dict(),
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
            arm_first = metrics.get("arm_first_rmse", metrics["first_rmse"])
            if arm_first < best_arm_first:
                best_arm_first = arm_first
                _save(out_dir / "checkpoints/best_arm_first.pt", payload)
    summary = {
        "schema": "clearvla-rdt-small-reference-summary-v1",
        "parameter_count": model.parameter_count(),
        "best_full_mse": best_full,
        "best_arm_first_rmse": best_arm_first,
        "history": history,
        "context": context,
    }
    (out_dir / "rdt_small_reference_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


def train_rdt2_fm(
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
    eval_seed: int = 0,
    instruction: str = "",
) -> dict[str, Any]:
    """Single-GPU ClearVLA trainer for the RDT2-FM action expert."""
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.lr,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
        weight_decay=trainer.weight_decay,
    )
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    lr_sched = _rdt_scheduler(
        optimizer,
        scheduler=trainer.scheduler,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    best_full = float("inf")
    best_arm_first = float("inf")
    history = []
    global_step = 0
    model_dtype = next(model.parameters()).dtype
    for epoch in range(1, trainer.epochs + 1):
        model.train()
        if hasattr(conditioner, "eval"):
            conditioner.eval()
        start = time.perf_counter()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            state = batch["state"].to(device, dtype=model_dtype, non_blocking=True)
            images = batch["obs_image"].to(device, non_blocking=True)
            actions = batch["action"].to(device, dtype=model_dtype, non_blocking=True)
            sample_keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
            with torch.no_grad():
                condition = conditioner.encode(
                    images,
                    [instruction] * state.shape[0],
                    sample_keys=sample_keys,
                    image_ablation="normal",
                    camera_names=tuple(getattr(train_loader.dataset, "camera_names", ("top", "wrist"))),
                )
                condition = condition.to(device=device, dtype=model_dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                state_tokens=state,
                action_gt=actions,
                lang_tokens=condition.dense_tokens,
                lang_kv_cache=condition.kv_cache,
                lang_attn_mask=condition.attention_mask,
            )
            loss.backward()
            raw_grad = _grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            optimizer.step(); lr_sched.step(); global_step += 1
            rows.append({"loss": float(loss.detach().cpu()), "grad": raw_grad})
            if batch_index % trainer.log_every == 0:
                mean_loss = float(np.mean([row["loss"] for row in rows[-trainer.log_every:]]))
                mean_grad = float(np.mean([row["grad"] for row in rows[-trainer.log_every:]]))
                print(
                    f"[rdt2-fm] epoch={epoch:03d}/{trainer.epochs:03d} "
                    f"batch={batch_index:04d} flow_mse={mean_loss:.6f} "
                    f"grad={mean_grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        metrics = {}
        if epoch % trainer.eval_every == 0:
            metrics = evaluate_rdt2_fm(
                model,
                conditioner,
                val_loader,
                device=device,
                action_normalizer=action_normalizer,
                inference_steps=inference_steps,
                max_batches=trainer.max_val_batches,
                eval_seed=eval_seed,
                instruction=instruction,
                image_ablation="normal",
            )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": {
                "loss": float(np.mean([row["loss"] for row in rows])),
                "grad": float(np.mean([row["grad"] for row in rows])),
            },
            "val": metrics,
        }
        history.append(record)
        if metrics:
            print(
                f"[rdt2-fm] epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.2f} "
                f"val_full={metrics['full_mse']:.6f} "
                f"arm_first={metrics.get('arm_first_rmse', float('nan')):.6f} "
                f"first4={metrics['first4_rmse']:.6f} "
                f"first8={metrics['first8_rmse']:.6f} "
                f"prior={metrics['prior_full_mse']:.6f} "
                f"replay={metrics['history_replay_full_mse']:.6f} "
                f"flow={metrics['val_flow_mse']:.6f} "
                f"sample_steps={inference_steps}",
                flush=True,
            )
        payload = {
            "schema": "clearvla-rdt2-fm-reference-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_sched.state_dict(),
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
            arm_first = metrics.get("arm_first_rmse", metrics["first_rmse"])
            if arm_first < best_arm_first:
                best_arm_first = arm_first
                _save(out_dir / "checkpoints/best_arm_first.pt", payload)
    summary = {
        "schema": "clearvla-rdt2-fm-reference-summary-v1",
        "parameter_count": model.parameter_count(),
        "best_full_mse": best_full,
        "best_arm_first_rmse": best_arm_first,
        "history": history,
        "context": context,
    }
    (out_dir / "rdt2_fm_reference_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary
