from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.training.numerics import (
    assert_finite_batch,
    assert_finite_gradients,
    assert_finite_optimizer_state,
    assert_finite_parameters,
    assert_finite_tensor,
    save_nan_debug_bundle,
)
from .codec import RDTLiteCodecs
from .evaluation import evaluate_rdt_lite_model, visual_dependency_report
from .losses import RDTLiteLossConfig, compute_rdt_lite_loss
from .model import RDTLiteModel
from .schedule import CosineDiffusionSchedule, DiffusionScheduleConfig


@dataclass(frozen=True)
class RDTLiteTrainerConfig:
    epochs: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.10
    log_every: int = 50
    rolling_window: int = 100
    sampling_steps: int = 5
    parameter_finite_check_every: int = 50
    optimizer_state_finite_check_every: int = 50
    max_train_batches: int = 0

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("invalid optimizer settings")
        if self.warmup_steps < 0 or not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("invalid scheduler settings")
        if (
            min(
                self.log_every,
                self.rolling_window,
                self.sampling_steps,
                self.parameter_finite_check_every,
                self.optimizer_state_finite_check_every,
            )
            <= 0
        ):
            raise ValueError("logging, sampling, and finite-check intervals must be positive")
        if self.max_train_batches < 0:
            raise ValueError("max_train_batches must be non-negative")


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total**0.5


def _module_grad_norm(module: torch.nn.Module) -> float:
    return _grad_norm([parameter for parameter in module.parameters() if parameter.requires_grad])


def _gradient_group_norms(model: RDTLiteModel) -> dict[str, float]:
    return {
        "visual": _module_grad_norm(model.visual_adaptor),
        "state": _module_grad_norm(model.state_adaptor),
        "action": _module_grad_norm(model.action_adaptor),
        "workspace": _module_grad_norm(model.blocks),
        "decoder": _module_grad_norm(model.decoder),
        "time": _module_grad_norm(model.time_embedder),
        "frequency": _module_grad_norm(model.frequency_embedder),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _scheduler_lambda(
    step: int, *, total_steps: int, warmup_steps: int, min_lr_ratio: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), 1e-6)
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(max((step - warmup_steps) / float(total_steps - warmup_steps), 0.0), 1.0)
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _save_checkpoint(
    path: Path,
    *,
    model: RDTLiteModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    codecs: RDTLiteCodecs,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    context: dict[str, Any],
    loss_config: RDTLiteLossConfig,
    trainer: RDTLiteTrainerConfig,
    schedule_config: DiffusionScheduleConfig,
) -> None:
    _atomic_torch_save(
        {
            "schema": "clearvla-rdt-lite-lab-v13.1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "codecs": codecs.to_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "history": history,
            "model_config": model.config.to_dict(),
            "loss_config": asdict(loss_config),
            "trainer_config": asdict(trainer),
            "diffusion_schedule_config": asdict(schedule_config),
            "context": context,
        },
        path,
    )


def train_rdt_lite_lab(
    *,
    model: RDTLiteModel,
    train_loader: DataLoader,
    val_loaders_by_mode: dict[str, DataLoader],
    device: torch.device,
    codecs: RDTLiteCodecs,
    out_dir: Path,
    trainer: RDTLiteTrainerConfig = RDTLiteTrainerConfig(),
    loss_config: RDTLiteLossConfig = RDTLiteLossConfig(),
    schedule_config: DiffusionScheduleConfig = DiffusionScheduleConfig(),
    context: dict[str, Any] | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trainer.validate()
    loss_config.validate()
    schedule_config.validate()
    codecs.validate()
    out_dir = Path(out_dir)
    checkpoints = out_dir / "checkpoints"
    debug = checkpoints / "debug"
    checkpoints.mkdir(parents=True, exist_ok=True)
    context = dict(context or {})
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay
    )
    batches_per_epoch = trainer.max_train_batches or len(train_loader)
    total_steps = max(int(trainer.epochs * batches_per_epoch), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_lambda(
            step,
            total_steps=total_steps,
            warmup_steps=trainer.warmup_steps,
            min_lr_ratio=trainer.min_lr_ratio,
        ),
    )
    diffusion_schedule = CosineDiffusionSchedule(schedule_config)
    sampler = getattr(train_loader, "batch_sampler", None)
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    if resume_payload:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        history = list(resume_payload.get("history", []))
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload.get("epoch", 0)) + 1

    best_full_mse = float("inf")
    best_arm_first = float("inf")
    best_paths: dict[str, Path | None] = {"full": None, "arm_first": None}
    for record in history:
        correct = record.get("val", {}).get("correct", {})
        if "full_mse" in correct and float(correct["full_mse"]) < best_full_mse:
            best_full_mse = float(correct["full_mse"])
        if "arm_first_rmse" in correct and float(correct["arm_first_rmse"]) < best_arm_first:
            best_arm_first = float(correct["arm_first_rmse"])

    for epoch in range(start_epoch, trainer.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        values: dict[str, list[float]] = defaultdict(list)
        rolling: deque[float] = deque(maxlen=trainer.rolling_window)
        started = time.perf_counter()
        for batch_index, raw in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            global_step += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
            marker = "input"
            try:
                assert_finite_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                marker = "forward"
                result = compute_rdt_lite_loss(
                    model,
                    state_history=batch["state_history"],
                    target_actions=batch["target_actions"],
                    visual_tokens=batch["visual_tokens"],
                    config=loss_config,
                    diffusion_schedule=diffusion_schedule,
                )
                assert_finite_tensor(result.total, name="loss")
                marker = "backward"
                result.total.backward()
                named = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                parameters = [parameter for _, parameter in named]
                assert_finite_gradients(named)
                groups = _gradient_group_norms(model)
                grad_raw = _grad_norm(parameters)
                if trainer.grad_clip > 0:
                    grad_raw = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, trainer.grad_clip, error_if_nonfinite=True
                        )
                        .detach()
                        .cpu()
                    )
                grad_clipped = _grad_norm(parameters)
                marker = "optimizer"
                optimizer.step()
                scheduler.step()
                if global_step % trainer.parameter_finite_check_every == 0:
                    assert_finite_parameters(named)
                if global_step % trainer.optimizer_state_finite_check_every == 0:
                    assert_finite_optimizer_state(optimizer)
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                save_nan_debug_bundle(
                    debug / "nan_debug_batch.pt",
                    batch=batch,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                    phase=f"rdt_lite:{marker}",
                    error=exc,
                )
                _save_checkpoint(
                    checkpoints / "failure.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    codecs=codecs,
                    epoch=epoch,
                    global_step=global_step,
                    history=history,
                    context=context,
                    loss_config=loss_config,
                    trainer=trainer,
                    schedule_config=schedule_config,
                )
                raise
            total = float(result.total.detach().cpu())
            rolling.append(total)
            values["loss"].append(total)
            values["grad_raw"].append(grad_raw)
            values["grad_clipped"].append(grad_clipped)
            values["clip_ratio"].append(grad_clipped / max(grad_raw, 1e-12))
            values["lr"].append(float(optimizer.param_groups[0]["lr"]))
            for key, value in groups.items():
                values[f"grad_{key}_raw"].append(value)
            for key, value in result.components.items():
                values[f"component_{key}"].append(float(value.detach().cpu()))
            for key, value in result.diagnostics.items():
                values[f"diag_{key}"].append(float(value.detach().cpu()))
            if batch_index % trainer.log_every == 0:
                primary = (
                    "clean_action" if loss_config.objective == "rdt_denoise" else "flow_velocity"
                )
                print(
                    f"\r[rdt-lite] objective={loss_config.objective} epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={total:.6f} rolling{trainer.rolling_window}={np.mean(rolling):.6f} {primary}={float(result.components[primary].detach().cpu()):.6f} "
                    f"endpoint_code={float(result.components['endpoint_full'].detach().cpu()):.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"grad={grad_raw:.3e}->{grad_clipped:.3e} clip={values['clip_ratio'][-1]:.3f} ",
                    end="",
                    flush=True,
                )
        print(flush=True)
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - started,
        }
        for key, items in values.items():
            record[f"train_{key}"] = float(np.mean(items)) if items else 0.0
        val_modes = {
            mode: evaluate_rdt_lite_model(
                model,
                loader,
                objective=loss_config.objective,
                device=device,
                codecs=codecs,
                sampling_steps=trainer.sampling_steps,
                diffusion_schedule=diffusion_schedule,
            )
            for mode, loader in val_loaders_by_mode.items()
        }
        dependency = visual_dependency_report(val_modes)
        record["val"] = val_modes
        record["dependency"] = dependency
        correct = val_modes["correct"]
        mse = float(correct["full_mse"])
        arm_first = float(correct.get("arm_first_rmse", float("inf")))
        print(
            f"[rdt-lite] objective={loss_config.objective} repr={codecs.action_representation} epoch={epoch:03d}/{trainer.epochs:03d} "
            f"sec={record['seconds']:.2f} train={record['train_loss']:.6f} final={mse:.6f} first_rmse={float(correct['first_rmse']):.6f} "
            f"first4_rmse={float(correct['first4_rmse']):.6f} arm_first_rmse={arm_first:.6f} gripper_rmse={float(correct['gripper_full_rmse']):.6f} "
            f"boundary={float(correct['pred_boundary_jump_norm']):.6f}/{float(correct['target_boundary_jump_norm']):.6f} nmae={float(correct['normalized_mae']):.6f} "
            f"zero_gap={dependency.get('zero_gap', float('nan')):+.6f} shift_gap={dependency.get('same_episode_shift_gap', float('nan')):+.6f} "
            f"cross_gap={dependency.get('cross_episode_gap', float('nan')):+.6f} event_cross_gap={dependency.get('cross_episode_event_gap', float('nan')):+.6f}",
            flush=True,
        )
        print(
            f"[rdt-lite-metrics] per_dim_rmse={np.asarray(correct['per_dim_rmse']).round(6).tolist()} "
            f"per_dim_nrmse={np.asarray(correct['per_dim_nrmse']).round(4).tolist()} "
            f"per_horizon_rmse={np.asarray(correct['per_horizon_rmse']).round(6).tolist()} "
            f"arm_first_deg_if_rad={float(correct.get('arm_first_rmse_deg_if_rad', float('nan'))):.3f}",
            flush=True,
        )
        history.append(record)
        for path in (checkpoints / f"checkpoint_{epoch:04d}.pt", checkpoints / "latest.pt"):
            _save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                codecs=codecs,
                epoch=epoch,
                global_step=global_step,
                history=history,
                context=context,
                loss_config=loss_config,
                trainer=trainer,
                schedule_config=schedule_config,
            )
        if mse < best_full_mse:
            best_full_mse = mse
            best_paths["full"] = checkpoints / "best_full.pt"
            _save_checkpoint(
                best_paths["full"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                codecs=codecs,
                epoch=epoch,
                global_step=global_step,
                history=history,
                context=context,
                loss_config=loss_config,
                trainer=trainer,
                schedule_config=schedule_config,
            )
        if arm_first < best_arm_first:
            best_arm_first = arm_first
            best_paths["arm_first"] = checkpoints / "best_arm_first.pt"
            _save_checkpoint(
                best_paths["arm_first"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                codecs=codecs,
                epoch=epoch,
                global_step=global_step,
                history=history,
                context=context,
                loss_config=loss_config,
                trainer=trainer,
                schedule_config=schedule_config,
            )

    summary = {
        "schema": "clearvla-rdt-lite-lab-summary-v13.1",
        "objective": loss_config.objective,
        "action_representation": codecs.action_representation,
        "best_full_mse": None if best_paths["full"] is None else best_full_mse,
        "best_arm_first_rmse": None if best_paths["arm_first"] is None else best_arm_first,
        "best_checkpoints": {
            key: None if path is None else str(path) for key, path in best_paths.items()
        },
        "parameter_count": model.parameter_count(),
        "history": history,
        "model": model.config.to_dict(),
        "trainer": asdict(trainer),
        "loss": asdict(loss_config),
        "diffusion_schedule": asdict(schedule_config),
        "codecs": codecs.to_dict(),
        "context": context,
    }
    (out_dir / "rdt_lite_lab_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    export_path = best_paths["full"] or checkpoints / "latest.pt"
    export_state = torch.load(export_path, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    _atomic_torch_save(
        {
            "schema": "clearvla-rdt-lite-lab-export-v13.1",
            "model_state_dict": export_state,
            "model_config": model.config.to_dict(),
            "loss_config": asdict(loss_config),
            "trainer_config": asdict(trainer),
            "diffusion_schedule_config": asdict(schedule_config),
            "codecs": codecs.to_dict(),
            "summary": summary,
            "context": context,
        },
        out_dir / "rdt_lite_lab.pt",
    )
    return summary
