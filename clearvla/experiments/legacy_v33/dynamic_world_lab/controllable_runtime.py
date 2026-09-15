from __future__ import annotations

"""Training and evaluation runtime for V33.6 controllable world modelling."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner

from .controllable_model import ControllableDynamicWorld
from .controllable_objectives import (
    ControllableWorldLossConfig,
    compute_controllable_world_losses,
)
from .runtime import prepare_sample, gripper_transition_metrics


@dataclass(frozen=True)
class ControllableWorldTrainerConfig:
    epochs: int = 12
    prior_warmup_epochs: int = 2
    effect_warmup_epochs: int = 4
    prior_lr: float = 1e-4
    effect_lr: float = 1e-4
    adapter_lr: float = 1e-5
    encoder_lr: float = 2e-6
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 300
    min_lr_ratio: float = 0.1
    ema_decay: float = 0.995
    unfreeze_dynamic_blocks: int = 1
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_ablation_batches: int = 64

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.prior_warmup_epochs < 0 or self.effect_warmup_epochs < 0:
            raise ValueError("phase epoch counts must be non-negative")
        if self.prior_warmup_epochs + self.effect_warmup_epochs > self.epochs:
            raise ValueError("prior+effect warmup cannot exceed total epochs")
        if min(self.prior_lr, self.effect_lr, self.adapter_lr, self.encoder_lr) <= 0:
            raise ValueError("all learning rates must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.unfreeze_dynamic_blocks < 0:
            raise ValueError("unfreeze_dynamic_blocks must be non-negative")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average empty rows")
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _scheduler(optimizer, *, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _phase_for_epoch(
    model: ControllableDynamicWorld, trainer: ControllableWorldTrainerConfig, epoch: int
) -> str:
    if model.config.input_mode == "current-only":
        return "prior"
    if epoch <= trainer.prior_warmup_epochs:
        return "prior"
    if epoch <= trainer.prior_warmup_epochs + trainer.effect_warmup_epochs:
        return "effect"
    return "align"


def _phase_end_epoch(trainer: ControllableWorldTrainerConfig, phase: str) -> int:
    if phase == "prior":
        return max(trainer.prior_warmup_epochs, 1)
    if phase == "effect":
        return trainer.prior_warmup_epochs + trainer.effect_warmup_epochs
    return trainer.epochs


def _make_optimizer(
    model: ControllableDynamicWorld,
    trainer: ControllableWorldTrainerConfig,
    phase: str,
    steps_per_epoch: int,
    current_epoch: int,
):
    model.set_training_phase(phase, unfreeze_dynamic_blocks=trainer.unfreeze_dynamic_blocks)
    named = model.trainable_named_parameters()
    if not named:
        raise ValueError(f"phase {phase} has no trainable parameters")

    encoder_ids = {id(p) for p in model.online_encoder.parameters() if p.requires_grad}
    adapter_ids = {id(p) for p in model.online_adapter.parameters() if p.requires_grad}
    groups: list[dict[str, Any]] = []
    used: set[int] = set()

    def add_group(ids: set[int], lr: float, name: str) -> None:
        params = [p for _, p in named if id(p) in ids and id(p) not in used]
        if params:
            groups.append({"params": params, "lr": lr, "name": name})
            used.update(id(p) for p in params)

    if phase == "prior":
        base_lr = trainer.prior_lr
    else:
        base_lr = trainer.effect_lr
    add_group(encoder_ids, trainer.encoder_lr, "representation_encoder")
    add_group(adapter_ids, trainer.adapter_lr, "world_adapter")
    remaining = [p for _, p in named if id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr, "name": f"{phase}_world"})

    optimizer = torch.optim.AdamW(
        groups,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
        weight_decay=trainer.weight_decay,
    )
    phase_epochs = max(_phase_end_epoch(trainer, phase) - current_epoch + 1, 1)
    scheduler = _scheduler(
        optimizer,
        total_steps=phase_epochs * steps_per_epoch,
        warmup_steps=min(trainer.warmup_steps, phase_epochs * steps_per_epoch // 2),
        min_lr_ratio=trainer.min_lr_ratio,
    )
    return optimizer, scheduler, named


def _forward_prepared(
    model: ControllableDynamicWorld,
    sample: dict[str, Tensor],
    *,
    mode: str | None = None,
):
    return model(
        sample["current_tokens"],
        sample["target_tokens"],
        sample["state"],
        sample["action"],
        action_state=sample.get("action_state", sample["state"]),
        mode_override=mode,
    )


def _decode_state(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


@torch.no_grad()
def evaluate_controllable_world(
    *,
    model: ControllableDynamicWorld,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    loss_config: ControllableWorldLossConfig,
    state_normalizer: ArrayNormalizer,
    max_batches: int = 0,
    ablation_batches: int = 64,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    pred_state_rows: list[np.ndarray] = []
    prior_state_rows: list[np.ndarray] = []
    target_state_rows: list[np.ndarray] = []
    current_state_rows: list[np.ndarray] = []
    support_rows: list[np.ndarray] = []
    sample_error_rows: list[np.ndarray] = []
    sample_prior_error_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    ablation_action_rows: list[np.ndarray] = []
    ablation_full_rows: list[np.ndarray] = []
    knn_rows: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        primary = prepare_sample(
            batch["primary"],
            conditioner=conditioner,
            model_config=model.config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        pair = prepare_sample(
            batch["pair"],
            conditioner=conditioner,
            model_config=model.config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        valid = batch["pair_valid"].to(device=device)
        output = _forward_prepared(model, primary)
        pair_output = model.forward_local_pair(
            pair["current_tokens"],
            pair["target_tokens"],
            pair["state"],
            pair["action"],
            action_state=pair.get("action_state", pair["state"]),
        )
        swapped = model.swapped_action_rollout(
            primary["current_tokens"],
            primary["state"],
            pair["action"],
            action_state=primary.get("action_state", primary["state"]),
        )
        losses = compute_controllable_world_losses(
            model,
            primary,
            output,
            config=loss_config,
            phase="eval",
            pair_output=pair_output,
            pair_valid=valid,
            swapped_output=swapped,
        )
        rows.append({key: float(value.detach().float().cpu()) for key, value in losses.items()})

        pred_state_rows.append(_decode_state(state_normalizer, output["pred_state_path"]))
        prior_state_rows.append(_decode_state(state_normalizer, output["prior_state_path"]))
        target_state_rows.append(primary["future_state_raw"].cpu().numpy())
        current_state_rows.append(primary["state_raw"].cpu().numpy())

        target_dynamic = output["target_dynamic"]
        target_scene = output["target_scene"]
        full_dynamic_error = F.smooth_l1_loss(
            output["pred_dynamic"].float(), target_dynamic.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        full_scene_error = F.smooth_l1_loss(
            output["pred_scene"].float(), target_scene.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        prior_dynamic_error = F.smooth_l1_loss(
            output["prior_pred_dynamic"].float(), target_dynamic.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        prior_scene_error = F.smooth_l1_loss(
            output["prior_pred_scene"].float(), target_scene.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        full_per = full_dynamic_error + loss_config.scene_predictive_weight * full_scene_error
        prior_per = prior_dynamic_error + loss_config.scene_predictive_weight * prior_scene_error
        sample_error_rows.append(full_per.cpu().numpy())
        sample_prior_error_rows.append(prior_per.cpu().numpy())
        support_rows.append(batch["support_distance"].cpu().numpy())

        raw_g = primary["future_state_raw"][..., model.config.gripper_index]
        boundary = torch.cat(
            [primary["state_raw"][:, None, model.config.gripper_index], raw_g[:, :-1]], dim=1
        )
        event_rows.append(
            ((raw_g - boundary).abs() >= loss_config.gripper_transition_threshold)
            .any(dim=1)
            .cpu()
            .numpy()
        )

        if "support" in batch:
            support = prepare_sample(
                batch["support"],
                conditioner=conditioner,
                model_config=model.config,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            support_target = model.encode_target_world(
                support["current_tokens"], support["target_tokens"]
            )
            knn_dynamic = F.smooth_l1_loss(
                support_target["target_dynamic"].float(),
                target_dynamic.float(),
                reduction="none",
            ).mean(dim=(-1, -2, -3))
            knn_scene = F.smooth_l1_loss(
                support_target["target_scene"].float(),
                target_scene.float(),
                reduction="none",
            ).mean(dim=(-1, -2, -3))
            knn_rows.append(
                (knn_dynamic + loss_config.scene_predictive_weight * knn_scene).cpu().numpy()
            )

        if batch_index <= ablation_batches:
            action_only = _forward_prepared(model, primary, mode="action-only")
            action_dynamic_error = F.smooth_l1_loss(
                action_only["pred_dynamic"].float(),
                target_dynamic.float(),
                reduction="none",
            ).mean(dim=(-1, -2, -3))
            action_scene_error = F.smooth_l1_loss(
                action_only["pred_scene"].float(),
                target_scene.float(),
                reduction="none",
            ).mean(dim=(-1, -2, -3))
            action_per = action_dynamic_error + (
                loss_config.scene_predictive_weight * action_scene_error
            )
            ablation_action_rows.append(action_per.cpu().numpy())
            ablation_full_rows.append(full_per.cpu().numpy())

    if not rows:
        raise ValueError("evaluation loader produced no batches")
    metrics = {f"val_{key}": value for key, value in _mean_rows(rows).items()}
    pred_raw = np.concatenate(pred_state_rows)
    prior_raw = np.concatenate(prior_state_rows)
    target_raw = np.concatenate(target_state_rows)
    current_raw = np.concatenate(current_state_rows)
    error = pred_raw - target_raw
    prior_error = prior_raw - target_raw
    metrics.update(
        {
            "state_path_rmse": float(np.sqrt(np.mean(error**2))),
            "prior_state_path_rmse": float(np.sqrt(np.mean(prior_error**2))),
            "state_path_gain": float(np.sqrt(np.mean(prior_error**2)) - np.sqrt(np.mean(error**2))),
            "state_endpoint_rmse": float(np.sqrt(np.mean(error[:, -1] ** 2))),
            "arm_state_path_rmse": float(
                np.sqrt(np.mean(np.delete(error, model.config.gripper_index, axis=-1) ** 2))
            ),
            "gripper_state_path_rmse": float(
                np.sqrt(np.mean(error[..., model.config.gripper_index] ** 2))
            ),
        }
    )
    metrics.update(
        gripper_transition_metrics(
            pred_raw,
            target_raw,
            current_raw,
            gripper_index=model.config.gripper_index,
            threshold=loss_config.gripper_transition_threshold,
            tolerance=loss_config.gripper_transition_radius,
        )
    )
    prior_gripper = gripper_transition_metrics(
        prior_raw,
        target_raw,
        current_raw,
        gripper_index=model.config.gripper_index,
        threshold=loss_config.gripper_transition_threshold,
        tolerance=loss_config.gripper_transition_radius,
    )
    metrics.update({f"prior_{key}": value for key, value in prior_gripper.items()})

    full_error = np.concatenate(sample_error_rows)
    prior_world_error = np.concatenate(sample_prior_error_rows)
    event = np.concatenate(event_rows).astype(bool)
    metrics["internal_prior_predictive"] = float(prior_world_error.mean())
    metrics["full_vs_current_gain"] = float(prior_world_error.mean() - full_error.mean())
    metrics["full_vs_current_relative_gain"] = float(
        (prior_world_error.mean() - full_error.mean()) / max(prior_world_error.mean(), 1e-8)
    )
    for name, mask in (("event", event), ("non_event", ~event)):
        metrics[f"{name}_count"] = float(mask.sum())
        metrics[f"{name}_full_predictive"] = (
            float(full_error[mask].mean()) if mask.any() else float("nan")
        )
        metrics[f"{name}_prior_predictive"] = (
            float(prior_world_error[mask].mean()) if mask.any() else float("nan")
        )
        metrics[f"{name}_full_vs_current_gain"] = (
            float((prior_world_error[mask] - full_error[mask]).mean())
            if mask.any()
            else float("nan")
        )
    if ablation_action_rows:
        action_ablation = np.concatenate(ablation_action_rows)
        full_ablation = np.concatenate(ablation_full_rows)
        metrics["ablation_full_predictive"] = float(full_ablation.mean())
        metrics["ablation_action_only_predictive"] = float(action_ablation.mean())
        metrics["full_vs_action_gain"] = float((action_ablation - full_ablation).mean())
    else:
        metrics["ablation_full_predictive"] = float("nan")
        metrics["ablation_action_only_predictive"] = float("nan")
        metrics["full_vs_action_gain"] = float("nan")
    if knn_rows:
        knn_error = np.concatenate(knn_rows)
        metrics["knn_predictive"] = float(knn_error.mean())
        metrics["full_vs_knn_gain"] = float((knn_error - full_error).mean())
    else:
        metrics["knn_predictive"] = float("nan")
        metrics["full_vs_knn_gain"] = float("nan")

    support = np.concatenate(support_rows)
    quantiles = np.quantile(support, [0.25, 0.5, 0.75])
    bins = np.digitize(support, quantiles, right=True)
    for index in range(4):
        mask = bins == index
        metrics[f"support_q{index + 1}_count"] = float(mask.sum())
        metrics[f"support_q{index + 1}_predictive"] = (
            float(full_error[mask].mean()) if mask.any() else float("nan")
        )
    model.train()
    return metrics


def _checkpoint_payload(
    *,
    model: ControllableDynamicWorld,
    optimizer,
    scheduler,
    phase: str,
    epoch: int,
    global_step: int,
    context: dict[str, Any],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    history: list[dict[str, Any]],
    trainer: ControllableWorldTrainerConfig,
    loss_config: ControllableWorldLossConfig,
) -> dict[str, Any]:
    return {
        "schema": "clearvla-v33.6-controllable-world-checkpoint-v1",
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        "context": context,
        "action_normalizer": action_normalizer.to_dict(),
        "state_normalizer": state_normalizer.to_dict(),
        "trainer": asdict(trainer),
        "loss_config": asdict(loss_config),
        "history": history,
    }


def train_controllable_world(
    *,
    model: ControllableDynamicWorld,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    out_dir: Path,
    trainer: ControllableWorldTrainerConfig,
    loss_config: ControllableWorldLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
) -> dict[str, Any]:
    trainer.validate()
    loss_config.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = (
        min(len(train_loader), trainer.max_train_batches)
        if trainer.max_train_batches
        else len(train_loader)
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    epoch_path = out_dir / "controllable_world_epochs.jsonl"
    if epoch_path.exists():
        epoch_path.unlink()

    best_predictive = float("inf")
    best_action = -float("inf")
    best_balanced = float("inf")
    optimizer = scheduler = None
    trainable_named: list[tuple[str, Tensor]] = []
    active_phase = ""

    for epoch in range(1, trainer.epochs + 1):
        phase = _phase_for_epoch(model, trainer, epoch)
        if phase != active_phase:
            active_phase = phase
            optimizer, scheduler, trainable_named = _make_optimizer(
                model, trainer, phase, steps_per_epoch, epoch
            )
            print(
                f"[controllable-world] phase={phase} epoch={epoch} "
                f"trainable={sum(p.numel() for _, p in trainable_named):,}",
                flush=True,
            )
        assert optimizer is not None and scheduler is not None
        model.train()
        start = time.perf_counter()
        rows: list[dict[str, float]] = []
        trainable_parameters = [parameter for _, parameter in trainable_named]

        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            primary = prepare_sample(
                batch["primary"],
                conditioner=conditioner,
                model_config=model.config,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            if phase == "prior":
                output = _forward_prepared(model, primary, mode="current-only")
                pair_output = swapped = None
                valid = None
            else:
                pair = prepare_sample(
                    batch["pair"],
                    conditioner=conditioner,
                    model_config=model.config,
                    camera_names=camera_names,
                    device=device,
                    dtype=dtype,
                )
                valid = batch["pair_valid"].to(device=device)
                output = _forward_prepared(model, primary)
                pair_output = model.forward_local_pair(
                    pair["current_tokens"],
                    pair["target_tokens"],
                    pair["state"],
                    pair["action"],
                    action_state=pair.get("action_state", pair["state"]),
                )
                swapped = model.swapped_action_rollout(
                    primary["current_tokens"],
                    primary["state"],
                    pair["action"],
                    action_state=primary.get("action_state", primary["state"]),
                )
            losses = compute_controllable_world_losses(
                model,
                primary,
                output,
                config=loss_config,
                phase=phase,
                pair_output=pair_output,
                pair_valid=valid,
                swapped_output=swapped,
            )
            losses["loss"].backward()
            raw_grad = _grad_norm(trainable_parameters)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, trainer.grad_clip)
            optimizer.step()
            scheduler.step()
            if phase == "align":
                model.update_ema_targets(trainer.ema_decay)
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = raw_grad
            rows.append(row)
            if batch_index % trainer.log_every == 0:
                mean = _mean_rows(rows[-trainer.log_every :])
                lrs = "/".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
                print(
                    "[controllable-world] "
                    f"phase={phase} epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={mean['loss']:.6f} full={mean['world_predictive']:.6f} "
                    f"prior={mean['prior_world_predictive']:.6f} gain={mean['full_vs_prior_gain']:.6f} "
                    f"res={mean['residual']:.6f}/{mean['residual_cosine']:.3f} "
                    f"need={mean['necessity']:.6f} inv={mean['inverse_action']:.5f}/"
                    f"{mean['inverse_gripper_accuracy']:.3f} "
                    f"local={mean['local_effect']:.5f}/{mean['local_effect_cosine']:.3f} "
                    f"swap={mean['swap_regret']:.5f}/{mean['swap_correct_fraction']:.3f} "
                    f"effect={mean['action_effect_dynamic_rms']:.4f} "
                    f"adaln={mean['adaln_gate_abs_mean']:.4f}/"
                    f"{mean['action_world_joint_rms']:.4f} "
                    f"read={mean['action_read_gate_abs_mean']:.4f} "
                    f"anchor={mean['representation_anchor']:.5f} grad={mean['grad']:.3e} lr={lrs}",
                    flush=True,
                )

        train_metrics = _mean_rows(rows)
        val_metrics = evaluate_controllable_world(
            model=model,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            loss_config=loss_config,
            state_normalizer=state_normalizer,
            max_batches=trainer.max_val_batches,
            ablation_batches=trainer.eval_ablation_batches,
        )
        record = {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        _append_jsonl(epoch_path, record)
        far_offset = model.config.future_offsets[-1]
        print(
            "[controllable-world] "
            f"phase={phase} epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.1f} "
            f"val_full={val_metrics['val_world_predictive']:.6f} "
            f"prior={val_metrics['internal_prior_predictive']:.6f} "
            f"gain={val_metrics['full_vs_current_gain']:.6f}/"
            f"{100.0 * val_metrics['full_vs_current_relative_gain']:.2f}% "
            f"event_gain={val_metrics['event_full_vs_current_gain']:.6f} "
            f"res_cos={val_metrics['val_residual_cosine']:.3f} "
            f"local_cos={val_metrics['val_local_effect_cosine']:.3f} "
            f"swap={val_metrics['val_swap_regret']:.5f}/"
            f"{val_metrics['val_swap_correct_fraction']:.3f} "
            f"gap_t{far_offset}={val_metrics[f'val_closed_loop_gap_t{far_offset}']:.6f} "
            f"state={val_metrics['state_path_rmse']:.5f} "
            f"gripper_f1={val_metrics['gripper_f1']:.3f}",
            flush=True,
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase,
            epoch=epoch,
            global_step=global_step,
            context=context,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
            history=history,
            trainer=trainer,
            loss_config=loss_config,
        )
        _save(out_dir / "checkpoints/latest.pt", payload)

        predictive = val_metrics["val_world_predictive"]
        action_score = (
            val_metrics["full_vs_current_gain"]
            + val_metrics["event_full_vs_current_gain"]
            + 0.1 * val_metrics["val_residual_cosine"]
            + 0.1 * val_metrics["val_local_effect_cosine"]
        )
        balanced = (
            predictive
            + max(0.0, -val_metrics["full_vs_current_gain"])
            + 0.5 * max(0.0, -val_metrics["event_full_vs_current_gain"])
            + 0.05 * max(0.0, 0.15 - val_metrics["val_residual_cosine"])
            + 0.05 * max(0.0, 0.10 - val_metrics["val_local_effect_cosine"])
            + 0.02 * val_metrics["state_path_rmse"]
            + 0.02 * val_metrics["val_representation_anchor"]
        )
        if predictive < best_predictive:
            best_predictive = predictive
            _save(out_dir / "checkpoints/best_predictive.pt", payload)
        if action_score > best_action:
            best_action = action_score
            _save(out_dir / "checkpoints/best_action_conditioned.pt", payload)
        if balanced < best_balanced:
            best_balanced = balanced
            _save(out_dir / "checkpoints/best_balanced.pt", payload)

    summary = {
        "schema": "clearvla-v33.6-controllable-world-summary-v1",
        "parameter_count": model.parameter_count(),
        "best_predictive": best_predictive,
        "best_action_conditioned_score": best_action,
        "best_balanced_score": best_balanced,
        "history": history,
        "context": context,
    }
    (out_dir / "controllable_world_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "ControllableWorldTrainerConfig",
    "evaluate_controllable_world",
    "train_controllable_world",
]
