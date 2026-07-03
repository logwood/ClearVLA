from __future__ import annotations

"""Training/evaluation runtime for V34.1.

Parameters, optimizer states and EMA targets remain FP32.  ``dtype`` controls
forward autocast only.  Checkpoints are fully resumable, including optimizer,
scheduler, EMA, epoch/global step, history and RNG state.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import random
import time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner

from .latent_world_model import LatentWorldModel
from .latent_world_objectives import LatentWorldLossConfig, _legacy_error, compute_latent_world_losses
from .shared_runtime import encode_sample_tokens, gripper_transition_metrics


@dataclass(frozen=True)
class LatentWorldTrainerConfig:
    epochs: int = 16
    perceiver_lr: float = 3e-5
    dynamics_lr: float = 1e-4
    auxiliary_lr: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    action_warmup_steps: int = 1200
    stability_warmup_steps: int = 300
    min_lr_ratio: float = 0.1
    ema_decay_start: float = 0.99
    ema_decay_end: float = 0.999
    camera_drop_prob: float = 0.25
    state_mask_prob: float = 0.15
    patch_mask_prob: float = 0.10
    checkpoint_predictive_slack: float = 0.08
    checkpoint_hold_ratio_max: float = 2.0
    checkpoint_min_embedding_std: float = 0.02
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_ablation_batches: int = 64

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if min(self.perceiver_lr, self.dynamics_lr, self.auxiliary_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if min(self.warmup_steps, self.action_warmup_steps, self.stability_warmup_steps) < 0:
            raise ValueError("warmups must be non-negative")
        if not 0 <= self.ema_decay_start <= self.ema_decay_end < 1:
            raise ValueError("EMA decays must satisfy 0 <= start <= end < 1")
        for name in ("camera_drop_prob", "state_mask_prob", "patch_mask_prob"):
            value = float(getattr(self, name))
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if self.checkpoint_predictive_slack < 0 or self.checkpoint_hold_ratio_max <= 1:
            raise ValueError("invalid checkpoint gates")


def _autocast(device: torch.device, dtype: torch.dtype):
    enabled = dtype != torch.float32 and device.type in {"cuda", "cpu"}
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def prepare_latent_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model: LatentWorldModel,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    # DINO/cache material may use bf16 for bandwidth, but all low-dimensional
    # physical values and model parameters stay FP32.
    current, target = encode_sample_tokens(
        sample,
        conditioner=conditioner,
        model_config=model.config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    return {
        "current_tokens": current,
        "target_tokens": target,
        "state": sample["state"].to(device=device, dtype=torch.float32, non_blocking=True),
        "action_state": sample["action_state"].to(device=device, dtype=torch.float32, non_blocking=True),
        "history_state": sample["history_state"].to(device=device, dtype=torch.float32, non_blocking=True),
        "target_history_state": sample["target_history_state"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "state_raw": sample["state_raw"].to(device=device, dtype=torch.float32, non_blocking=True),
        "action": sample["action"].to(device=device, dtype=torch.float32, non_blocking=True),
        "action_raw": sample["action_raw"].to(device=device, dtype=torch.float32, non_blocking=True),
        "future_state": sample["future_state"].to(device=device, dtype=torch.float32, non_blocking=True),
        "future_state_raw": sample["future_state_raw"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "episode_idx": sample["episode_idx"].to(device=device),
        "sample_index": sample["sample_index"].to(device=device),
    }


def _forward(model: LatentWorldModel, sample: dict[str, Tensor]) -> dict[str, Tensor]:
    return model(
        sample["current_tokens"],
        sample["target_tokens"],
        sample["history_state"],
        sample["target_history_state"],
        sample["action"],
        sample["action_state"],
    )


def _masked_perception_inputs(
    sample: dict[str, Tensor], trainer: LatentWorldTrainerConfig
) -> tuple[Tensor, Tensor]:
    tokens = sample["current_tokens"].clone()
    states = sample["history_state"].clone()
    batch, _, cameras, patches, _ = tokens.shape
    device = tokens.device

    if cameras > 1 and trainer.camera_drop_prob > 0:
        apply = torch.rand(batch, device=device) < trainer.camera_drop_prob
        camera = torch.randint(cameras, (batch,), device=device)
        for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            tokens[index, :, int(camera[index])] = 0
    if trainer.patch_mask_prob > 0:
        mask = torch.rand(batch, tokens.shape[1], cameras, patches, device=device)
        tokens = tokens.masked_fill((mask < trainer.patch_mask_prob)[..., None], 0)
    if trainer.state_mask_prob > 0:
        state_mask = torch.rand(batch, device=device) < trainer.state_mask_prob
        states[state_mask] = 0
    return tokens, states


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _finite_metric(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    value = float(metrics.get(key, default))
    return value if math.isfinite(value) else float(default)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average empty rows")
    keys = set.intersection(*(set(row) for row in rows))
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


def _grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _scheduler(optimizer, *, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), max(total_steps - 1, 0))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr_ratio) + 0.5 * (1.0 - float(min_lr_ratio)) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _ramp(step: int, warmup: int) -> float:
    return 1.0 if warmup <= 0 else min(max(float(step) / float(warmup), 0.0), 1.0)


def _ema_decay(step: int, total_steps: int, trainer: LatentWorldTrainerConfig) -> float:
    progress = min(max(float(step) / float(max(total_steps, 1)), 0.0), 1.0)
    return trainer.ema_decay_start + progress * (trainer.ema_decay_end - trainer.ema_decay_start)


def _decode_state(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


@torch.no_grad()
def evaluate_latent_world(
    *,
    model: LatentWorldModel,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    loss_config: LatentWorldLossConfig,
    state_normalizer: ArrayNormalizer,
    max_batches: int = 0,
    ablation_batches: int = 64,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    pred_state_rows: list[np.ndarray] = []
    hold_state_rows: list[np.ndarray] = []
    target_state_rows: list[np.ndarray] = []
    current_state_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    hold_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    zero_world_effect_rows: list[np.ndarray] = []
    no_perception_rows: list[np.ndarray] = []
    visual_only_rows: list[np.ndarray] = []
    state_only_rows: list[np.ndarray] = []
    top_only_rows: list[np.ndarray] = []
    wrist_only_rows: list[np.ndarray] = []
    knn_rows: list[np.ndarray] = []
    support_distance_rows: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        raw_primary = batch["primary"] if "primary" in batch else batch
        primary = prepare_latent_sample(
            raw_primary,
            conditioner=conditioner,
            model=model,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        with _autocast(device, dtype):
            output = _forward(model, primary)
            pair_output = None
            pair_valid = None
            swapped = None
            if "pair" in batch:
                pair = prepare_latent_sample(
                    batch["pair"],
                    conditioner=conditioner,
                    model=model,
                    camera_names=camera_names,
                    device=device,
                    dtype=dtype,
                )
                pair_output = model.forward_local_pair(
                    pair["current_tokens"],
                    pair["target_tokens"],
                    pair["history_state"],
                    pair["target_history_state"],
                    pair["action"],
                    pair["action_state"],
                )
                pair_valid = batch["pair_valid"].to(device=device)
                swapped = model.swapped_action_rollout(
                    output["initial_world"], pair["action"], primary["action_state"]
                )
            losses = compute_latent_world_losses(
                model,
                primary,
                output,
                config=loss_config,
                action_scale=1.0,
                stability_scale=1.0,
                pair_output=pair_output,
                pair_valid=pair_valid,
                swapped_output=swapped,
            )
        rows.append({key: float(value.detach().float().cpu()) for key, value in losses.items()})

        pred_state_rows.append(_decode_state(state_normalizer, output["pred_state_path"]))
        hold_state_rows.append(_decode_state(state_normalizer, output["hold_state_path"]))
        target_state_rows.append(primary["future_state_raw"].cpu().numpy())
        current_state_rows.append(primary["state_raw"].cpu().numpy())
        full_per = _legacy_error(
            model, output["pred_world"], output["target_world"],
            scene_weight=loss_config.scene_predictive_weight, reduction="none"
        ).mean(dim=1)
        hold_per = _legacy_error(
            model, output["hold_world"], output["target_world"],
            scene_weight=loss_config.scene_predictive_weight, reduction="none"
        ).mean(dim=1)
        full_rows.append(full_per.float().cpu().numpy())
        hold_rows.append(hold_per.float().cpu().numpy())
        gripper = primary["future_state_raw"][..., model.config.gripper_index]
        boundary = torch.cat(
            [primary["state_raw"][:, None, model.config.gripper_index], gripper[:, :-1]], dim=1
        )
        event_rows.append(
            ((gripper - boundary).abs() >= loss_config.gripper_transition_threshold)
            .any(dim=1).cpu().numpy()
        )

        if "support_distance" in batch:
            support_distance_rows.append(batch["support_distance"].cpu().numpy())
        if "support" in batch:
            support = prepare_latent_sample(
                batch["support"], conditioner=conditioner, model=model,
                camera_names=camera_names, device=device, dtype=dtype
            )
            with _autocast(device, dtype):
                _, support_future = model.encode_targets(
                    support["current_tokens"], support["target_tokens"],
                    support["history_state"], support["target_history_state"]
                )
                knn_error = _legacy_error(
                    model, support_future, output["target_world"],
                    scene_weight=loss_config.scene_predictive_weight, reduction="none"
                ).mean(dim=1)
            knn_rows.append(knn_error.float().cpu().numpy())

        if batch_index <= ablation_batches:
            with _autocast(device, dtype):
                permutation = torch.arange(primary["action"].shape[0] - 1, -1, -1, device=device)
                shuffled = model.swapped_action_rollout(
                    output["initial_world"], primary["action"][permutation], primary["action_state"]
                )
                shuffled_error = _legacy_error(
                    model, shuffled["pred_world"], output["target_world"],
                    scene_weight=loss_config.scene_predictive_weight, reduction="none"
                ).mean(dim=1)
                zero_world = torch.zeros_like(output["initial_world"])
                action_tokens = model.action_tokenizer(primary["action"], primary["action_state"])
                zero_rollout = model.dynamics.rollout_pair(zero_world, action_tokens["effect_steps"])
                zero_effect = (
                    zero_rollout["pred_world"] - zero_rollout["hold_world"]
                ).float().square().mean(dim=(-1, -2, -3)).sqrt()

                def ablated_error(tokens: Tensor, states: Tensor) -> Tensor:
                    result = model(
                        tokens, primary["target_tokens"], states,
                        primary["target_history_state"], primary["action"], primary["action_state"]
                    )
                    return _legacy_error(
                        model, result["pred_world"], output["target_world"],
                        scene_weight=loss_config.scene_predictive_weight, reduction="none"
                    ).mean(dim=1)

                no_perception = ablated_error(
                    torch.zeros_like(primary["current_tokens"]),
                    torch.zeros_like(primary["history_state"]),
                )
                visual_only = ablated_error(
                    primary["current_tokens"], torch.zeros_like(primary["history_state"])
                )
                state_only = ablated_error(
                    torch.zeros_like(primary["current_tokens"]), primary["history_state"]
                )
                if model.config.num_cameras == 2:
                    top_tokens = primary["current_tokens"].clone(); top_tokens[:, :, 1] = 0
                    wrist_tokens = primary["current_tokens"].clone(); wrist_tokens[:, :, 0] = 0
                    top_error = ablated_error(top_tokens, primary["history_state"])
                    wrist_error = ablated_error(wrist_tokens, primary["history_state"])
                else:
                    top_error = wrist_error = None
            shuffled_rows.append(shuffled_error.float().cpu().numpy())
            zero_world_effect_rows.append(zero_effect.cpu().numpy())
            no_perception_rows.append(no_perception.float().cpu().numpy())
            visual_only_rows.append(visual_only.float().cpu().numpy())
            state_only_rows.append(state_only.float().cpu().numpy())
            if top_error is not None:
                top_only_rows.append(top_error.float().cpu().numpy())
                wrist_only_rows.append(wrist_error.float().cpu().numpy())

    if not rows:
        raise ValueError("evaluation loader produced no batches")
    metrics = {f"val_{key}": value for key, value in _mean(rows).items()}
    full = np.concatenate(full_rows)
    hold = np.concatenate(hold_rows)
    event = np.concatenate(event_rows).astype(bool)
    metrics.update({
        "val_full": float(full.mean()),
        "val_hold": float(hold.mean()),
        "full_vs_hold_gain": float((hold - full).mean()),
        "full_vs_hold_relative_gain": float((hold.mean() - full.mean()) / max(hold.mean(), 1e-8)),
    })
    for name, mask in (("event", event), ("non_event", ~event)):
        metrics[f"{name}_count"] = float(mask.sum())
        metrics[f"{name}_full"] = float(full[mask].mean()) if mask.any() else float("nan")
        metrics[f"{name}_hold"] = float(hold[mask].mean()) if mask.any() else float("nan")
        metrics[f"{name}_gain"] = float((hold[mask] - full[mask]).mean()) if mask.any() else float("nan")

    pred_raw = np.concatenate(pred_state_rows)
    hold_raw = np.concatenate(hold_state_rows)
    target_raw = np.concatenate(target_state_rows)
    current_raw = np.concatenate(current_state_rows)
    error = pred_raw - target_raw
    hold_error = hold_raw - target_raw
    metrics.update({
        "state_path_rmse": float(np.sqrt(np.mean(error ** 2))),
        "hold_state_path_rmse": float(np.sqrt(np.mean(hold_error ** 2))),
        "state_path_gain": float(np.sqrt(np.mean(hold_error ** 2)) - np.sqrt(np.mean(error ** 2))),
        "state_endpoint_rmse": float(np.sqrt(np.mean(error[:, -1] ** 2))),
        "arm_state_path_rmse": float(
            np.sqrt(np.mean(np.delete(error, model.config.gripper_index, axis=-1) ** 2))
        ),
        "gripper_state_path_rmse": float(
            np.sqrt(np.mean(error[..., model.config.gripper_index] ** 2))
        ),
    })
    metrics.update(gripper_transition_metrics(
        pred_raw, target_raw, current_raw,
        gripper_index=model.config.gripper_index,
        threshold=loss_config.gripper_transition_threshold,
        tolerance=loss_config.gripper_transition_radius,
    ))
    hold_gripper = gripper_transition_metrics(
        hold_raw, target_raw, current_raw,
        gripper_index=model.config.gripper_index,
        threshold=loss_config.gripper_transition_threshold,
        tolerance=loss_config.gripper_transition_radius,
    )
    metrics.update({f"hold_{key}": value for key, value in hold_gripper.items()})

    def add(name: str, values: list[np.ndarray]) -> None:
        metrics[name] = float(np.concatenate(values).mean()) if values else float("nan")

    add("shuffled_action_predictive", shuffled_rows)
    add("zero_world_action_effect_rms", zero_world_effect_rows)
    add("no_perception_predictive", no_perception_rows)
    add("visual_only_predictive", visual_only_rows)
    add("state_only_predictive", state_only_rows)
    add("top_only_predictive", top_only_rows)
    add("wrist_only_predictive", wrist_only_rows)
    add("knn_predictive", knn_rows)
    if knn_rows:
        knn = np.concatenate(knn_rows)
        metrics["full_vs_knn_gain"] = float((knn - full[: len(knn)]).mean())
    if support_distance_rows:
        support_distance = np.concatenate(support_distance_rows)
        if len(support_distance) != len(full):
            raise ValueError("support distances do not align with validation predictions")
        metrics["support_distance_mean"] = float(support_distance.mean())
        order = np.argsort(support_distance, kind="stable")
        for quartile, indices in enumerate(np.array_split(order, 4), start=1):
            metrics[f"support_q{quartile}_count"] = float(len(indices))
            metrics[f"support_q{quartile}_predictive"] = float(full[indices].mean()) if len(indices) else float("nan")
            metrics[f"support_q{quartile}_gain"] = float((hold[indices] - full[indices]).mean()) if len(indices) else float("nan")
    if shuffled_rows:
        shuffled_value = np.concatenate(shuffled_rows)
        subset = full[: len(shuffled_value)]
        metrics["full_vs_shuffled_gain"] = float((shuffled_value - subset).mean())
        metrics["shuffle_correct_fraction"] = float((subset < shuffled_value).mean())
    model.train()
    return metrics


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *, model: LatentWorldModel, optimizer, scheduler, epoch: int, global_step: int,
    context: dict[str, Any], action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer, history: list[dict[str, Any]],
    trainer: LatentWorldTrainerConfig, loss_config: LatentWorldLossConfig,
    best: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema": "clearvla-v34.1-latent-world-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "context": context,
        "model_config": asdict(model.config),
        "trainer_config": asdict(trainer),
        "loss_config": asdict(loss_config),
        "action_normalizer": action_normalizer.to_dict(),
        "state_normalizer": state_normalizer.to_dict(),
        "history": history,
        "best": best,
        "rng": _rng_state(),
    }


def _checkpoint_eligible(
    val: dict[str, float], *, best_predictive: float, trainer: LatentWorldTrainerConfig
) -> tuple[bool, dict[str, bool]]:
    full = _finite_metric(val, "val_full", float("inf"))
    hold = _finite_metric(val, "val_hold", float("inf"))
    std = _finite_metric(val, "val_embedding_std", 0.0)
    gap = abs(_finite_metric(val, "val_closed_loop_gap", float("inf")))
    zero_raw = val.get("zero_world_action_effect_rms", float("nan"))
    zero = abs(float(zero_raw)) if math.isfinite(float(zero_raw)) else float("inf")
    no_perception = _finite_metric(val, "no_perception_predictive", float("inf"))
    state_only = _finite_metric(val, "state_only_predictive", float("inf"))
    reference = best_predictive if math.isfinite(best_predictive) else full
    ablations_enabled = trainer.eval_ablation_batches > 0
    gates = {
        "predictive": full <= reference * (1.0 + trainer.checkpoint_predictive_slack) + 1e-8,
        "hold_bounded": hold <= trainer.checkpoint_hold_ratio_max * max(full, 1e-8),
        "embedding": std >= trainer.checkpoint_min_embedding_std,
        "closed_loop": gap <= max(0.05, 0.75 * max(full, 1e-8)),
        "zero_world": (not ablations_enabled) or zero <= 1e-6,
        "perception_required": (not ablations_enabled) or full <= no_perception + 1e-8,
        "vision_not_harmful": (not ablations_enabled) or full <= (1.0 + trainer.checkpoint_predictive_slack) * state_only + 1e-8,
    }
    return all(gates.values()), gates


def train_latent_world(
    *,
    model: LatentWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    out_dir: Path,
    trainer: LatentWorldTrainerConfig,
    loss_config: LatentWorldLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    trainer.validate(); loss_config.validate()
    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = out_dir / "latent_world_epochs.jsonl"

    # FP32 master parameters and FP32 EMA; dtype is autocast only.
    model.to(device=device, dtype=torch.float32)
    perceiver_ids = {id(p) for p in model.online_perceiver.parameters()}
    dynamics_ids = {id(p) for p in list(model.action_tokenizer.parameters()) + list(model.dynamics.parameters())}
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in perceiver_ids], "lr": trainer.perceiver_lr, "name": "perceiver"},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in dynamics_ids], "lr": trainer.dynamics_lr, "name": "dynamics"},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in perceiver_ids and id(p) not in dynamics_ids], "lr": trainer.auxiliary_lr, "name": "auxiliary"},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(
        groups, betas=(trainer.beta1, trainer.beta2), eps=trainer.eps,
        weight_decay=trainer.weight_decay
    )
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    total_steps = max(steps_per_epoch * trainer.epochs, 1)
    scheduler = _scheduler(
        optimizer, total_steps=total_steps, warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    best = {"predictive": float("inf"), "controllable": -float("inf"), "balanced": float("inf")}
    if resume is not None:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        schema = str(payload.get("schema", ""))
        if not schema.startswith("clearvla-v34.1"):
            raise ValueError(f"resume checkpoint schema is not V34.1: {schema}")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
        best.update(payload.get("best", {}))
        _restore_rng(payload.get("rng"))
        print(f"[latent-world] resumed {resume} at epoch={start_epoch} step={global_step}", flush=True)

    for epoch in range(start_epoch, trainer.epochs + 1):
        model.train()
        started = time.time()
        train_rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            raw_primary = batch["primary"] if "primary" in batch else batch
            primary = prepare_latent_sample(
                raw_primary, conditioner=conditioner, model=model, camera_names=camera_names,
                device=device, dtype=dtype
            )
            with _autocast(device, dtype):
                output = _forward(model, primary)
                masked_tokens, masked_states = _masked_perception_inputs(primary, trainer)
                output["masked_initial_world"] = model.encode_online(masked_tokens, masked_states)
                pair_output = None; pair_valid = None; swapped = None
                if "pair" in batch:
                    pair = prepare_latent_sample(
                        batch["pair"], conditioner=conditioner, model=model,
                        camera_names=camera_names, device=device, dtype=dtype
                    )
                    pair_output = model.forward_local_pair(
                        pair["current_tokens"], pair["target_tokens"], pair["history_state"],
                        pair["target_history_state"], pair["action"], pair["action_state"]
                    )
                    pair_valid = batch["pair_valid"].to(device=device)
                    swapped = model.swapped_action_rollout(
                        output["initial_world"], pair["action"], primary["action_state"]
                    )
                action_scale = _ramp(global_step + 1, trainer.action_warmup_steps)
                stability_scale = _ramp(global_step + 1, trainer.stability_warmup_steps)
                losses = compute_latent_world_losses(
                    model, primary, output, config=loss_config,
                    action_scale=action_scale, stability_scale=stability_scale,
                    pair_output=pair_output, pair_valid=pair_valid, swapped_output=swapped
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            grad = _grad_norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, trainer.grad_clip)
            optimizer.step(); scheduler.step(); global_step += 1
            decay = _ema_decay(global_step, total_steps, trainer)
            model.update_ema(decay)

            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row.update({"grad": grad, "ema_decay": decay, "lr": float(max(g["lr"] for g in optimizer.param_groups))})
            train_rows.append(row)
            if trainer.log_every and batch_index % trainer.log_every == 0:
                print(
                    "[latent-world-v34.1] "
                    f"epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={row['loss']:.6f} full={row['legacy_full']:.6f} hold={row['legacy_hold']:.6f} "
                    f"gain={row['full_vs_hold_gain']:.6f} inc_cos={row['increment_cosine']:.3f} "
                    f"local={row['local_effect_cosine']:.3f} state={row['state_path']:.5f} "
                    f"grip_inv={row['pred_inverse_gripper_accuracy']:.3f} view={row['view_descriptor']:.5f} "
                    f"adaln={row.get('adaln_gate_abs_mean', 0.0):.4f} grad={grad:.3e} lr={row['lr']:.3e}",
                    flush=True,
                )

        val = evaluate_latent_world(
            model=model, loader=val_loader, conditioner=conditioner, device=device, dtype=dtype,
            camera_names=camera_names, loss_config=loss_config,
            state_normalizer=state_normalizer, max_batches=trainer.max_val_batches,
            ablation_batches=trainer.eval_ablation_batches
        )
        eligible, gates = _checkpoint_eligible(val, best_predictive=best["predictive"], trainer=trainer)
        val.update({f"checkpoint_gate_{name}": float(value) for name, value in gates.items()})
        val["checkpoint_eligible"] = float(eligible)
        epoch_row = {
            "epoch": epoch, "global_step": global_step, "seconds": time.time() - started,
            "train": _mean(train_rows), "val": val,
        }
        history.append(epoch_row); _append_jsonl(epoch_path, epoch_row)

        predictive = _finite_metric(val, "val_full", float("inf"))
        improved_predictive = predictive < best["predictive"]
        if improved_predictive:
            best["predictive"] = predictive
        # Gains are clipped so checkpoint selection cannot be dominated by a
        # deliberately degraded hold trajectory.  Hard gates above enforce
        # bounded hold error and genuine perception use before these scores are
        # considered at all.
        relative_gain = min(max(_finite_metric(val, "full_vs_hold_relative_gain"), 0.0), 0.25)
        event_gain = min(max(_finite_metric(val, "event_gain"), 0.0), 0.10)
        controllable = (
            relative_gain
            + 0.5 * event_gain
            + 0.25 * _finite_metric(val, "val_residual_cosine")
            + 0.25 * _finite_metric(val, "val_local_effect_cosine")
            + 0.10 * _finite_metric(val, "shuffle_correct_fraction")
        )
        balanced = (
            predictive
            - 0.25 * min(max(_finite_metric(val, "full_vs_hold_gain"), 0.0), predictive)
            - 0.10 * event_gain
            + 0.10 * _finite_metric(val, "val_representation_anchor")
            + 0.10 * _finite_metric(val, "state_path_rmse")
            - 0.02 * _finite_metric(val, "gripper_f1")
        )
        improved_controllable = eligible and controllable > best["controllable"]
        improved_balanced = eligible and balanced < best["balanced"]
        if improved_controllable:
            best["controllable"] = controllable
        if improved_balanced:
            best["balanced"] = balanced

        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            global_step=global_step, context=context, action_normalizer=action_normalizer,
            state_normalizer=state_normalizer, history=history, trainer=trainer,
            loss_config=loss_config, best=best
        )
        if improved_predictive:
            torch.save(payload, checkpoint_dir / "best_predictive.pt")
        if improved_controllable:
            torch.save(payload, checkpoint_dir / "best_controllable.pt")
        if improved_balanced:
            torch.save(payload, checkpoint_dir / "best_balanced.pt")
        # Save latest last so resume always sees the fully updated best-state metadata.
        torch.save(payload, checkpoint_dir / "latest.pt")

        print(
            "[latent-world-v34.1] "
            f"epoch={epoch:03d} val_full={val['val_full']:.6f} val_hold={val['val_hold']:.6f} "
            f"gain={val['full_vs_hold_relative_gain']:.3%} event_gain={val.get('event_gain', float('nan')):.6f} "
            f"inc_cos={val.get('val_increment_cosine', float('nan')):.3f} "
            f"local={val.get('val_local_effect_cosine', float('nan')):.3f} "
            f"state={val['state_path_rmse']:.5f} gripper_f1={val.get('gripper_f1', 0.0):.3f} "
            f"eligible={eligible} gates={gates}",
            flush=True,
        )

    summary = {
        "schema": "clearvla-v34.1-latent-world-summary-v1",
        "best": best,
        "epochs": history,
        "parameter_report": model.parameter_report(),
        "context": context,
    }
    (out_dir / "latent_world_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "LatentWorldTrainerConfig",
    "prepare_latent_sample",
    "evaluate_latent_world",
    "train_latent_world",
]
