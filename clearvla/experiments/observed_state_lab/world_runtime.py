from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.dynamic_world_lab.shared_runtime import (
    encode_current_tokens,
    encode_target_tokens,
    gripper_transition_metrics,
)

from .world_model import V35ObservedStateWorldModel
from .world_objectives import V35WorldLossConfig, compute_v35_world_losses, legacy_error


@dataclass(frozen=True)
class V35WorldTrainerConfig:
    epochs: int = 16
    encoder_lr: float = 3e-5
    dynamics_lr: float = 1e-4
    auxiliary_lr: float = 1e-4
    weight_decay: float = 0.01
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
    executed_action_mask_prob: float = 0.10
    patch_mask_prob: float = 0.10
    checkpoint_predictive_slack: float = 0.08
    checkpoint_hold_ratio_max: float = 2.0
    checkpoint_min_embedding_std: float = 0.02
    checkpoint_zero_world_max: float = 1e-7
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_ablation_batches: int = 64


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type=device.type, enabled=False)


@torch.no_grad()
def prepare_v35_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model: V35ObservedStateWorldModel,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    current = encode_current_tokens(
        sample,
        conditioner=conditioner,
        model_config=model.config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    target = encode_target_tokens(
        sample,
        conditioner=conditioner,
        model_config=model.config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    keys = (
        "state", "state_raw", "history_state", "executed_action_history",
        "target_history_state", "target_executed_action_history", "action", "action_raw",
        "action_state", "future_state", "future_state_raw", "segment_state", "segment_state_raw",
        "sample_index", "episode_idx",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in (
        "state", "history_state", "executed_action_history", "target_history_state",
        "target_executed_action_history", "action", "action_state", "future_state", "segment_state",
    ):
        out[key] = out[key].to(torch.float32)
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["current_visual"] = current.to(dtype=compute_dtype)
    out["target_visual"] = target.to(dtype=compute_dtype)
    return out


def forward_v35(model: V35ObservedStateWorldModel, sample: dict[str, Tensor]) -> dict[str, Tensor]:
    return model(
        sample["current_visual"],
        sample["target_visual"],
        sample["history_state"],
        sample["target_history_state"],
        sample["executed_action_history"],
        sample["target_executed_action_history"],
        sample["action"],
        sample["action_state"],
    )


def forward_v35_pair_minimal(model: V35ObservedStateWorldModel, sample: dict[str, Tensor]) -> dict[str, Tensor]:
    return model.forward_pair(
        sample["current_visual"],
        sample["target_visual"],
        sample["history_state"],
        sample["target_history_state"],
        sample["executed_action_history"],
        sample["target_executed_action_history"],
        sample["action"],
        sample["action_state"],
    )


def masked_evidence(sample: dict[str, Tensor], trainer: V35WorldTrainerConfig) -> tuple[Tensor, Tensor, Tensor]:
    visual = sample["current_visual"].clone()
    state = sample["history_state"].clone()
    executed = sample["executed_action_history"].clone()
    batch, _, cameras, patches, _ = visual.shape
    device = visual.device
    if cameras > 1 and trainer.camera_drop_prob > 0:
        apply = torch.rand(batch, device=device) < trainer.camera_drop_prob
        camera = torch.randint(cameras, (batch,), device=device)
        for row in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            visual[row, :, int(camera[row])] = 0
    if trainer.patch_mask_prob > 0:
        mask = torch.rand(batch, visual.shape[1], cameras, patches, device=device)
        visual = visual.masked_fill((mask < trainer.patch_mask_prob)[..., None], 0)
    if trainer.state_mask_prob > 0:
        state[torch.rand(batch, device=device) < trainer.state_mask_prob] = 0
    if trainer.executed_action_mask_prob > 0:
        executed[torch.rand(batch, device=device) < trainer.executed_action_mask_prob] = 0
    return visual, state, executed


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), separators=(",", ":")) + "\n")


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average empty metrics")
    keys = set.intersection(*(set(row) for row in rows))
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}



def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
        "cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }

def grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), max(total_steps - 1, 0))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        p = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        p = min(max(p, 0.0), 1.0)
        return float(min_lr_ratio) + 0.5 * (1 - float(min_lr_ratio)) * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def ramp(step: int, warmup: int) -> float:
    return 1.0 if warmup <= 0 else min(max(float(step) / float(warmup), 0.0), 1.0)


def ema_decay(step: int, total_steps: int, trainer: V35WorldTrainerConfig) -> float:
    p = min(max(float(step) / float(max(total_steps, 1)), 0.0), 1.0)
    return trainer.ema_decay_start + p * (trainer.ema_decay_end - trainer.ema_decay_start)


def decode(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


def interpolate_segment_states(current: np.ndarray, endpoints: np.ndarray, segment_length: int) -> np.ndarray:
    rows = []
    previous = current
    for index in range(endpoints.shape[1]):
        target = endpoints[:, index]
        for step in range(1, segment_length + 1):
            alpha = float(step) / float(segment_length)
            rows.append((1 - alpha) * previous + alpha * target)
        previous = target
    return np.stack(rows, axis=1)


@torch.no_grad()
def evaluate_v35_world(
    *,
    model: V35ObservedStateWorldModel,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    loss_config: V35WorldLossConfig,
    state_normalizer: ArrayNormalizer,
    action_normalizer: ArrayNormalizer,
    max_batches: int = 0,
    ablation_batches: int = 64,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    pred_endpoint, target_endpoint, current_state = [], [], []
    pred_path, target_path = [], []
    inverse_action, action_target = [], []
    full_error, hold_error, event_mask = [], [], []
    ablations: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "no_perception", "visual_only", "proprio_only", "top_only", "wrist_only",
            "shuffled_action", "zero_world_effect"
        )
    }

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        primary_raw = batch["primary"] if "primary" in batch else batch
        sample = prepare_v35_sample(
            primary_raw, conditioner=conditioner, model=model, camera_names=camera_names,
            device=device, dtype=dtype,
        )
        with autocast_context(device, dtype):
            output = forward_v35(model, sample)
            pair_output = None
            pair_valid = None
            swapped = None
            if "pair" in batch:
                pair = prepare_v35_sample(
                    batch["pair"], conditioner=conditioner, model=model, camera_names=camera_names,
                    device=device, dtype=dtype,
                )
                pair_output = forward_v35_pair_minimal(model, pair)
                pair_valid = batch["pair_valid"].to(device=device)
                pair_tokens = model.action_tokenizer(pair["action"], sample["action_state"])
                swapped = model.dynamics.rollout_pair(
                    output["initial_world"], pair_tokens["actual_tokens"], pair_tokens["hold_tokens"]
                )
            losses = compute_v35_world_losses(
                model, sample, output, config=loss_config,
                pair_output=pair_output, pair_valid=pair_valid, swapped_output=swapped,
            )
        rows.append({key: float(value.detach().float().cpu()) for key, value in losses.items()})

        pred_endpoint_raw = decode(state_normalizer, output["pred_segment_state"])
        target_endpoint_raw = sample["segment_state_raw"].cpu().numpy()
        current_raw = sample["state_raw"].cpu().numpy()
        pred_endpoint.append(pred_endpoint_raw)
        target_endpoint.append(target_endpoint_raw)
        current_state.append(current_raw)
        pred_path.append(interpolate_segment_states(current_raw, pred_endpoint_raw, model.config.segment_length))
        target_path.append(sample["future_state_raw"].cpu().numpy())
        inverse_action.append(decode(action_normalizer, output["pred_inverse_action"]).reshape(-1, model.config.world_horizon, model.config.action_dim))
        action_target.append(sample["action_raw"].cpu().numpy())

        full = legacy_error(
            model, output["pred_world"], output["target_world"],
            global_weight=loss_config.legacy_global_weight, reduction="none"
        ).mean(dim=1)
        hold = legacy_error(
            model, output["hold_world"], output["target_world"],
            global_weight=loss_config.legacy_global_weight, reduction="none"
        ).mean(dim=1)
        full_error.append(full.float().cpu().numpy())
        hold_error.append(hold.float().cpu().numpy())
        gripper = sample["action_raw"][..., model.config.gripper_index]
        boundary = torch.cat([sample["state_raw"][:, None, model.config.gripper_index], gripper[:, :-1]], dim=1)
        event_mask.append(((gripper - boundary).abs() >= loss_config.gripper_transition_threshold).any(dim=1).cpu().numpy())

        if batch_index <= ablation_batches:
            def world_from(visual: Tensor, state: Tensor, executed: Tensor) -> Tensor:
                return model.encode_online(visual, state, executed)

            actual_tokens = output["actual_tokens"]
            hold_tokens = output["hold_tokens"]
            zero_visual = torch.zeros_like(sample["current_visual"])
            zero_state = torch.zeros_like(sample["history_state"])
            zero_executed = torch.zeros_like(sample["executed_action_history"])
            worlds = {
                "no_perception": world_from(zero_visual, zero_state, zero_executed),
                "visual_only": world_from(sample["current_visual"], zero_state, zero_executed),
                "proprio_only": world_from(zero_visual, sample["history_state"], sample["executed_action_history"]),
            }
            if model.config.num_cameras >= 2:
                top = sample["current_visual"].clone(); top[:, :, 1:] = 0
                wrist = sample["current_visual"].clone(); wrist[:, :, :1] = 0
                worlds["top_only"] = world_from(top, sample["history_state"], sample["executed_action_history"])
                worlds["wrist_only"] = world_from(wrist, sample["history_state"], sample["executed_action_history"])
            else:
                worlds["top_only"] = worlds["visual_only"]
                worlds["wrist_only"] = worlds["visual_only"]
            for name, world in worlds.items():
                roll = model.dynamics.rollout_pair(world, actual_tokens, hold_tokens)
                err = legacy_error(
                    model, roll["pred_world"], output["target_world"],
                    global_weight=loss_config.legacy_global_weight, reduction="none"
                ).mean(dim=1)
                ablations[name].append(err.float().cpu().numpy())
            permutation = torch.arange(sample["action"].shape[0] - 1, -1, -1, device=device)
            shuffled_tokens = model.action_tokenizer(
                sample["action"][permutation], sample["action_state"]
            )
            shuffled = model.dynamics.rollout_pair(
                output["initial_world"], shuffled_tokens["actual_tokens"], shuffled_tokens["hold_tokens"]
            )
            shuffled_err = legacy_error(
                model, shuffled["pred_world"], output["target_world"],
                global_weight=loss_config.legacy_global_weight, reduction="none"
            ).mean(dim=1)
            ablations["shuffled_action"].append(shuffled_err.float().cpu().numpy())
            zero_world = torch.zeros_like(output["initial_world"])
            zero_roll = model.dynamics.rollout_pair(zero_world, actual_tokens, hold_tokens)
            zero_effect = zero_roll["action_world_effect"].float().square().mean(dim=(1, 2, 3)).sqrt()
            ablations["zero_world_effect"].append(zero_effect.cpu().numpy())

    metrics = {f"val_{key}": value for key, value in mean_rows(rows).items()}
    pred_endpoint_np = np.concatenate(pred_endpoint)
    target_endpoint_np = np.concatenate(target_endpoint)
    current_np = np.concatenate(current_state)
    pred_path_np = np.concatenate(pred_path)
    target_path_np = np.concatenate(target_path)
    inverse_np = np.concatenate(inverse_action)
    action_np = np.concatenate(action_target)
    full_np = np.concatenate(full_error)
    hold_np = np.concatenate(hold_error)
    event_np = np.concatenate(event_mask).astype(bool)

    metrics.update({
        "segment_state_rmse": float(np.sqrt(np.mean((pred_endpoint_np - target_endpoint_np) ** 2))),
        "segment_arm_rmse": float(np.sqrt(np.mean((pred_endpoint_np[..., :-1] - target_endpoint_np[..., :-1]) ** 2))),
        "segment_gripper_rmse": float(np.sqrt(np.mean((pred_endpoint_np[..., -1] - target_endpoint_np[..., -1]) ** 2))),
        "interpolated_state_path_rmse": float(np.sqrt(np.mean((pred_path_np - target_path_np) ** 2))),
        "inverse_action_rmse": float(np.sqrt(np.mean((inverse_np - action_np) ** 2))),
        "full_predictive": float(full_np.mean()),
        "hold_predictive": float(hold_np.mean()),
        "full_vs_hold_gain": float((hold_np - full_np).mean()),
        "full_vs_hold_relative_gain": float(((hold_np - full_np) / np.maximum(hold_np, 1e-8)).mean()),
        "event_full_vs_hold_gain": float((hold_np[event_np] - full_np[event_np]).mean()) if event_np.any() else float("nan"),
        "non_event_full_vs_hold_gain": float((hold_np[~event_np] - full_np[~event_np]).mean()) if (~event_np).any() else float("nan"),
    })
    metrics.update(gripper_transition_metrics(
        inverse_np, action_np, current_np,
        gripper_index=model.config.gripper_index,
        threshold=loss_config.gripper_transition_threshold,
        tolerance=2,
    ))
    for name, values in ablations.items():
        metrics[f"{name}_predictive" if name != "zero_world_effect" else "zero_world_effect_rms"] = (
            float(np.concatenate(values).mean()) if values else float("nan")
        )
    if math.isfinite(metrics.get("shuffled_action_predictive", float("nan"))):
        metrics["shuffled_action_gap"] = float(metrics["shuffled_action_predictive"] - metrics["full_predictive"])
    else:
        metrics["shuffled_action_gap"] = float("nan")
    metrics["hold_action_gap"] = float(metrics["hold_predictive"] - metrics["full_predictive"])
    metrics["teacher_self_gap"] = float(metrics.get("val_teacher_self_gap", float("nan")))
    metrics["overshoot_active"] = float(loss_config.overshoot_weight > 0 and loss_config.overshoot_depth > 0)
    return metrics


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def eligible(metrics: dict[str, float], best_predictive: float, trainer: V35WorldTrainerConfig) -> bool:
    full = float(metrics.get("full_predictive", float("inf")))
    hold = float(metrics.get("hold_predictive", float("inf")))
    std = float(metrics.get("val_embedding_std", 0.0))
    zero = float(metrics.get("zero_world_effect_rms", float("inf")))
    no_perception = float(metrics.get("no_perception_predictive", 0.0))
    proprio = float(metrics.get("proprio_only_predictive", 0.0))
    return (
        math.isfinite(full)
        and full <= best_predictive * (1 + trainer.checkpoint_predictive_slack)
        and hold <= full * trainer.checkpoint_hold_ratio_max
        and std >= trainer.checkpoint_min_embedding_std
        and zero <= trainer.checkpoint_zero_world_max
        and full < no_perception
        and full <= proprio * (1 + trainer.checkpoint_predictive_slack)
    )


def train_v35_world(
    *,
    model: V35ObservedStateWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    out_dir: Path,
    trainer: V35WorldTrainerConfig,
    loss_config: V35WorldLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model.to(device=device, dtype=torch.float32)
    groups = [
        {"params": model.online_encoder.parameters(), "lr": trainer.encoder_lr},
        {"params": list(model.action_tokenizer.parameters()) + list(model.dynamics.parameters()) + list(model.consequence_anchorer.parameters()), "lr": trainer.dynamics_lr},
        {"params": list(model.state_decoder.parameters()) + list(model.inverse_decoder.parameters()) + list(model.region_decoder.parameters()) + list(model.action_only_probe.parameters()), "lr": trainer.auxiliary_lr},
    ]
    optimizer = torch.optim.AdamW(
        groups, weight_decay=trainer.weight_decay, betas=(trainer.beta1, trainer.beta2), eps=trainer.eps
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    total_steps = max(steps_per_epoch * trainer.epochs, 1)
    schedule = scheduler(optimizer, total_steps, trainer.warmup_steps, trainer.min_lr_ratio)
    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []
    best = {"predictive": float("inf"), "controllable": -float("inf"), "balanced": -float("inf")}

    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
            raise ValueError("resume checkpoint is not V35")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        history = list(payload.get("history", []))
        best.update(payload.get("best", {}))
        restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        model.train()
        train_rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            raw = batch["primary"] if "primary" in batch else batch
            sample = prepare_v35_sample(
                raw, conditioner=conditioner, model=model, camera_names=camera_names,
                device=device, dtype=dtype,
            )
            pair = None
            pair_valid = None
            need_pair_branch = (
                float(loss_config.pair_direction_weight) > 0.0
                or float(loss_config.swap_rank_weight) > 0.0
            )
            if need_pair_branch and "pair" in batch:
                pair = prepare_v35_sample(
                    batch["pair"], conditioner=conditioner, model=model, camera_names=camera_names,
                    device=device, dtype=dtype,
                )
                pair_valid = batch["pair_valid"].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            action_scale = ramp(global_step, trainer.action_warmup_steps)
            stability_scale = ramp(global_step, trainer.stability_warmup_steps)
            with autocast_context(device, dtype):
                output = forward_v35(model, sample)
                masked_visual, masked_state, masked_executed = masked_evidence(sample, trainer)
                masked_world = model.encode_online(masked_visual, masked_state, masked_executed)
                pair_output = forward_v35_pair_minimal(model, pair) if pair is not None else None
                swapped = None
                if pair is not None and float(loss_config.swap_rank_weight) > 0.0:
                    # Counterfactual swapped-action rollout is only a ranking baseline.
                    # It should not build or retain an auxiliary backward graph.
                    with torch.no_grad():
                        pair_action = model.action_tokenizer(pair["action"], sample["action_state"])
                        swapped = model.dynamics.rollout_pair(
                            output["initial_world"].detach(),
                            pair_action["actual_tokens"],
                            pair_action["hold_tokens"],
                        )
                losses = compute_v35_world_losses(
                    model, sample, output, config=loss_config,
                    masked_world=masked_world, pair_output=pair_output, pair_valid=pair_valid,
                    swapped_output=swapped, stability_scale=stability_scale, action_scale=action_scale,
                )
            losses["loss"].float().backward()
            grad = grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            optimizer.step()
            schedule.step()
            global_step += 1
            decay = ema_decay(global_step, total_steps, trainer)
            model.update_ema(decay)
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row.update({"grad": grad, "ema_decay": decay, "action_scale": action_scale})
            row.update(cuda_memory_metrics(device))
            train_rows.append(row)
            if trainer.log_every and batch_index % trainer.log_every == 0:
                print(
                    f"[v35-world] epoch={epoch:03d} batch={batch_index:04d} "
                    f"loss={row['loss']:.6f} full={row['val_full']:.6f} "
                    f"state={row['state_endpoint']:.6f} inc_cos={row['increment_cosine']:.3f} "
                    f"local={row['local_effect_cosine']:.3f} gate={row['adaln_gate_abs_mean']:.3f} "
                    f"grad={grad:.3e} lr={optimizer.param_groups[1]['lr']:.3e}",
                    flush=True,
                )

        train_metrics = mean_rows(train_rows)
        for memory_key in ("cuda_peak_allocated_mb", "cuda_peak_reserved_mb", "cuda_allocated_mb", "cuda_reserved_mb"):
            if train_rows and memory_key in train_rows[0]:
                train_metrics["max_" + memory_key] = max(float(row[memory_key]) for row in train_rows)
        val_metrics = evaluate_v35_world(
            model=model, loader=val_loader, conditioner=conditioner, device=device, dtype=dtype,
            camera_names=camera_names, loss_config=loss_config,
            state_normalizer=state_normalizer, action_normalizer=action_normalizer,
            max_batches=trainer.max_val_batches, ablation_batches=trainer.eval_ablation_batches,
        )
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics}
        history.append(record)
        append_jsonl(out_dir / "v35_world_epochs.jsonl", record)

        full = float(val_metrics["full_predictive"])
        if full < best["predictive"]:
            best["predictive"] = full
            save_names = ["best_predictive.pt"]
        else:
            save_names = []
        control = (
            max(float(val_metrics.get("full_vs_hold_relative_gain", 0.0)), -1.0)
            + max(float(val_metrics.get("event_full_vs_hold_gain", 0.0)), -1.0)
            + max(float(val_metrics.get("val_local_effect_cosine", 0.0)), -1.0)
            + 0.25 * max(float(val_metrics.get("val_increment_cosine", 0.0)), -1.0)
        )
        is_eligible = eligible(val_metrics, best["predictive"], trainer)
        if is_eligible and control > best["controllable"]:
            best["controllable"] = control
            save_names.append("best_controllable.pt")
        balanced = -full + 0.1 * control - 0.05 * float(val_metrics.get("segment_state_rmse", 0.0))
        if is_eligible and balanced > best["balanced"]:
            best["balanced"] = balanced
            save_names.append("best_balanced.pt")

        def payload() -> dict[str, Any]:
            return {
                "schema": "clearvla-v35-world-checkpoint-v1",
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": schedule.state_dict(),
                "model_config": asdict(model.config),
                "loss_config": asdict(loss_config),
                "trainer_config": asdict(trainer),
                "action_normalizer": action_normalizer.to_dict(),
                "state_normalizer": state_normalizer.to_dict(),
                "context": context,
                "history": history,
                "best": best,
                "rng": rng_state(),
            }

        state = payload()
        for name in save_names:
            torch.save(state, checkpoint_dir / name)
        torch.save(state, checkpoint_dir / "latest.pt")
        summary = {"schema": "clearvla-v35-world-summary-v1", "best": best, "latest": record}
        (out_dir / "v35_world_summary.json").write_text(
            json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V35WorldTrainerConfig", "prepare_v35_sample", "forward_v35", "evaluate_v35_world",
    "train_v35_world", "jsonable",
]
