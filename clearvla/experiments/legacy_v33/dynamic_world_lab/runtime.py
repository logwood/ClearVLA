from __future__ import annotations

"""Training and evaluation runtime for the standalone dynamic world model."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner

from .model import DynamicPredictiveWorld, DynamicPredictiveWorldConfig
from .objectives import DynamicWorldLossConfig, compute_dynamic_world_losses


@dataclass(frozen=True)
class DynamicWorldTrainerConfig:
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
    eval_ablation_batches: int = 64


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


def _reshape_dense_tokens(
    dense: Tensor,
    *,
    batch: int,
    history: int,
    config: DynamicPredictiveWorldConfig,
) -> Tensor:
    expected_tokens = config.num_cameras * config.patches_per_camera
    if dense.ndim != 3 or dense.shape[0] != batch * history:
        raise ValueError(f"conditioner dense tokens have invalid shape {tuple(dense.shape)}")
    if dense.shape[1] != expected_tokens or dense.shape[2] != config.latent_dim:
        raise ValueError(
            "DINO token geometry mismatch: "
            f"got {tuple(dense.shape[1:])}, expected ({expected_tokens},{config.latent_dim})"
        )
    return dense.reshape(
        batch,
        history,
        config.num_cameras,
        config.patches_per_camera,
        config.latent_dim,
    )


@torch.no_grad()
def encode_current_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DynamicPredictiveWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    batch = int(sample["state"].shape[0])
    history = model_config.history_length
    if "history_obs_image" in sample:
        images = sample["history_obs_image"]
        flat_images = images.reshape(batch * history, *images.shape[2:])
        condition = conditioner.encode(flat_images, camera_names=camera_names)
    else:
        keys = sample["history_keys"].reshape(batch * history, 2)
        dummy = torch.zeros(
            batch * history, model_config.num_cameras, 3, 1, 1, dtype=torch.float32
        )
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("dynamic world requires dense DINO tokens, not KV conditions")
    current = _reshape_dense_tokens(
        condition.dense_tokens, batch=batch, history=history, config=model_config
    )
    return current.to(device=device, dtype=dtype)


@torch.no_grad()
def encode_target_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DynamicPredictiveWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    batch = int(sample["state"].shape[0])
    history = model_config.history_length
    future = model_config.num_future
    if "target_history_obs_image" in sample:
        target_images = sample["target_history_obs_image"]
        flat_target = target_images.reshape(batch * future * history, *target_images.shape[3:])
        condition = conditioner.encode(flat_target, camera_names=camera_names)
    else:
        target_keys = sample["target_history_keys"].reshape(batch * future * history, 2)
        dummy = torch.zeros(
            batch * future * history, model_config.num_cameras, 3, 1, 1, dtype=torch.float32
        )
        condition = conditioner.encode(dummy, sample_keys=target_keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("dynamic world requires dense DINO tokens, not KV conditions")
    return _reshape_dense_tokens(
        condition.dense_tokens,
        batch=batch * future, history=history, config=model_config,
    ).reshape(
        batch, future, history, model_config.num_cameras,
        model_config.patches_per_camera, model_config.latent_dim,
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_sample_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DynamicPredictiveWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    batch = int(sample["state"].shape[0])
    history = model_config.history_length
    future = model_config.num_future
    current = encode_current_tokens(
        sample, conditioner=conditioner, model_config=model_config,
        camera_names=camera_names, device=device, dtype=dtype,
    )

    target = encode_target_tokens(
        sample, conditioner=conditioner, model_config=model_config,
        camera_names=camera_names, device=device, dtype=dtype,
    )
    return current, target


def prepare_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DynamicPredictiveWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    current, target = encode_sample_tokens(
        sample,
        conditioner=conditioner,
        model_config=model_config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    return {
        "current_tokens": current,
        "target_tokens": target,
        "state": sample["state"].to(device=device, dtype=dtype, non_blocking=True),
        "action_state": sample.get("action_state", sample["state"]).to(
            device=device, dtype=dtype, non_blocking=True
        ),
        "state_raw": sample["state_raw"].to(device=device, dtype=torch.float32, non_blocking=True),
        "action": sample["action"].to(device=device, dtype=dtype, non_blocking=True),
        "action_raw": sample["action_raw"].to(device=device, dtype=torch.float32, non_blocking=True),
        "future_state": sample["future_state"].to(device=device, dtype=dtype, non_blocking=True),
        "future_state_raw": sample["future_state_raw"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "episode_idx": sample["episode_idx"].to(device=device),
        "sample_index": sample["sample_index"].to(device=device),
    }


def _forward_prepared(model: DynamicPredictiveWorld, sample: dict[str, Tensor], *, mode: str | None = None):
    return model(
        sample["current_tokens"],
        sample["target_tokens"],
        sample["state"],
        sample["action"],
        mode_override=mode,
    )


def _decode_state(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


def _dilate_event(event: np.ndarray, radius: int) -> np.ndarray:
    event = np.asarray(event, dtype=np.bool_)
    if radius <= 0:
        return event
    out = event.copy()
    for shift in range(1, radius + 1):
        out[:, shift:] |= event[:, :-shift]
        out[:, :-shift] |= event[:, shift:]
    return out


def gripper_transition_metrics(
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    current_raw: np.ndarray,
    *,
    gripper_index: int,
    threshold: float,
    tolerance: int,
) -> dict[str, float]:
    """Global one-to-one event metrics in raw physical units.

    Events are matched greedily by minimum timing error, within the requested
    tolerance and with the same transition direction.  This avoids the
    duplicate-credit problem of dilated point-wise masks while retaining the
    delay tolerance needed for gradual gripper trajectories.
    """
    pred_g = np.asarray(pred_raw[..., gripper_index], dtype=np.float64)
    target_g = np.asarray(target_raw[..., gripper_index], dtype=np.float64)
    current_g = np.asarray(current_raw[..., gripper_index], dtype=np.float64).reshape(-1, 1)
    pred_boundary = np.concatenate([current_g, pred_g[:, :-1]], axis=1)
    target_boundary = np.concatenate([current_g, target_g[:, :-1]], axis=1)
    pred_delta = pred_g - pred_boundary
    target_delta = target_g - target_boundary

    def match_direction(direction: int) -> tuple[int, int, int, list[float]]:
        tp = fp = fn = 0
        timing: list[float] = []
        for row in range(len(pred_delta)):
            pred_idx = np.flatnonzero(
                (np.abs(pred_delta[row]) >= threshold) & (np.sign(pred_delta[row]) == direction)
            ).tolist()
            target_idx = np.flatnonzero(
                (np.abs(target_delta[row]) >= threshold) & (np.sign(target_delta[row]) == direction)
            ).tolist()
            # Sequence-alignment DP maximizes one-to-one matches first and
            # minimizes total timing error second.  A nearest-first greedy rule
            # can lose valid matches for shifted event runs.
            dp: list[list[tuple[int, float, tuple[float, ...]]]] = [
                [(0, 0.0, ()) for _ in range(len(target_idx) + 1)]
                for _ in range(len(pred_idx) + 1)
            ]

            def better(
                left: tuple[int, float, tuple[float, ...]],
                right: tuple[int, float, tuple[float, ...]],
            ) -> tuple[int, float, tuple[float, ...]]:
                if left[0] != right[0]:
                    return left if left[0] > right[0] else right
                return left if left[1] <= right[1] else right

            for pred_pos in range(1, len(pred_idx) + 1):
                for target_pos in range(1, len(target_idx) + 1):
                    best = better(dp[pred_pos - 1][target_pos], dp[pred_pos][target_pos - 1])
                    distance = abs(pred_idx[pred_pos - 1] - target_idx[target_pos - 1])
                    if distance <= tolerance:
                        previous = dp[pred_pos - 1][target_pos - 1]
                        matched_option = (
                            previous[0] + 1,
                            previous[1] + float(distance),
                            previous[2] + (float(distance),),
                        )
                        best = better(best, matched_option)
                    dp[pred_pos][target_pos] = best
            matched, _, matched_timing = dp[-1][-1]
            timing.extend(matched_timing)
            tp += matched
            fp += len(pred_idx) - matched
            fn += len(target_idx) - matched
        return tp, fp, fn, timing

    def summarize(tp: int, fp: int, fn: int, prefix: str) -> dict[str, float]:
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
        return {
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1": f1,
            f"{prefix}tp": float(tp),
            f"{prefix}fp": float(fp),
            f"{prefix}fn": float(fn),
        }

    close_tp, close_fp, close_fn, close_timing = match_direction(+1)
    open_tp, open_fp, open_fn, open_timing = match_direction(-1)
    tp = close_tp + open_tp
    fp = close_fp + open_fp
    fn = close_fn + open_fn
    timing = close_timing + open_timing
    metrics = summarize(tp, fp, fn, "gripper_")
    metrics.update(summarize(close_tp, close_fp, close_fn, "gripper_close_"))
    metrics.update(summarize(open_tp, open_fp, open_fn, "gripper_open_"))
    metrics.update(
        {
            "gripper_pred_events": float(tp + fp),
            "gripper_target_events": float(tp + fn),
            "gripper_timing_mae_steps": float(np.mean(timing)) if timing else float("nan"),
            "gripper_close_timing_mae_steps": (
                float(np.mean(close_timing)) if close_timing else float("nan")
            ),
            "gripper_open_timing_mae_steps": (
                float(np.mean(open_timing)) if open_timing else float("nan")
            ),
        }
    )
    return metrics


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@torch.no_grad()
def evaluate_dynamic_world(
    *,
    model: DynamicPredictiveWorld,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    loss_config: DynamicWorldLossConfig,
    state_normalizer: ArrayNormalizer,
    max_batches: int = 0,
    ablation_batches: int = 64,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    pred_state_rows, target_state_rows, current_state_rows = [], [], []
    support_rows, error_rows = [], []
    ablation_current, ablation_action, knn_rows = [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        primary = prepare_sample(
            batch["primary"], conditioner=conditioner, model_config=model.config,
            camera_names=camera_names, device=device, dtype=dtype,
        )
        pair = prepare_sample(
            batch["pair"], conditioner=conditioner, model_config=model.config,
            camera_names=camera_names, device=device, dtype=dtype,
        )
        valid = batch["pair_valid"].to(device=device)
        output = _forward_prepared(model, primary)
        pair_output = model.forward_local_pair(
            pair["current_tokens"], pair["target_tokens"], pair["state"], pair["action"]
        )
        swapped = model.rollout_from_encoded(
            output["context"],
            output["initial_scene"],
            output["initial_dynamic"],
            pair["action"],
            primary["state"],
        )
        losses = compute_dynamic_world_losses(
            model, primary, output, config=loss_config,
            pair=pair, pair_output=pair_output, pair_valid=valid, swapped_output=swapped,
        )
        rows.append({key: float(value.detach().float().cpu()) for key, value in losses.items()})
        pred_raw = _decode_state(state_normalizer, output["pred_state_path"])
        target_raw = primary["future_state_raw"].detach().cpu().numpy()
        pred_state_rows.append(pred_raw)
        target_state_rows.append(target_raw)
        current_state_rows.append(primary["state_raw"].detach().cpu().numpy())
        dynamic_error = F.smooth_l1_loss(
            output["pred_dynamic"].float(),
            output["target_dynamic"].float(),
            reduction="none",
        ).mean(dim=(-1, -2, -3))
        pred_scene_delta = output["pred_scene"] - output["initial_scene"][:, None]
        target_scene_delta = (
            output["target_scene"] - output["target_initial_scene"][:, None]
        )
        scene_error = F.smooth_l1_loss(
            pred_scene_delta.float(), target_scene_delta.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        per_sample_error = (
            dynamic_error + loss_config.scene_predictive_weight * scene_error
        ).cpu().numpy()
        error_rows.append(per_sample_error)
        support_rows.append(batch["support_distance"].cpu().numpy())
        if "support" in batch:
            support = prepare_sample(
                batch["support"], conditioner=conditioner, model_config=model.config,
                camera_names=camera_names, device=device, dtype=dtype,
            )
            (
                support_initial_scene,
                _,
                support_target_scene,
                support_target_dynamic,
            ) = model.encode_targets(support["current_tokens"], support["target_tokens"])
            support_scene_delta = support_target_scene - support_initial_scene[:, None]
            knn_dynamic = F.smooth_l1_loss(
                support_target_dynamic.float(), output["target_dynamic"].float()
            )
            knn_scene = F.smooth_l1_loss(
                support_scene_delta.float(), target_scene_delta.float()
            )
            knn_rows.append(float(
                (knn_dynamic + loss_config.scene_predictive_weight * knn_scene).cpu()
            ))

        if batch_index <= ablation_batches:
            current_only = _forward_prepared(model, primary, mode="current-only")
            action_only = _forward_prepared(model, primary, mode="action-only")

            def world_error(candidate):
                candidate_dynamic = F.smooth_l1_loss(
                    candidate["pred_dynamic"].float(), output["target_dynamic"].float()
                )
                candidate_scene_delta = (
                    candidate["pred_scene"] - candidate["initial_scene"][:, None]
                )
                candidate_scene = F.smooth_l1_loss(
                    candidate_scene_delta.float(), target_scene_delta.float()
                )
                return candidate_dynamic + loss_config.scene_predictive_weight * candidate_scene

            ablation_current.append(float(world_error(current_only).cpu()))
            ablation_action.append(float(world_error(action_only).cpu()))

    if not rows:
        raise ValueError("evaluation loader produced no batches")
    metrics = {f"val_{key}": value for key, value in _mean_rows(rows).items()}
    pred_raw = np.concatenate(pred_state_rows, axis=0)
    target_raw = np.concatenate(target_state_rows, axis=0)
    current_raw = np.concatenate(current_state_rows, axis=0)
    error = pred_raw - target_raw
    metrics.update(
        {
            "state_path_rmse": float(np.sqrt(np.mean(error**2))),
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
    metrics["ablation_current_only_predictive"] = float(np.mean(ablation_current))
    metrics["ablation_action_only_predictive"] = float(np.mean(ablation_action))
    metrics["full_vs_current_gain"] = (
        metrics["ablation_current_only_predictive"] - metrics["val_world_predictive"]
    )
    metrics["full_vs_action_gain"] = (
        metrics["ablation_action_only_predictive"] - metrics["val_world_predictive"]
    )
    metrics["knn_predictive"] = float(np.mean(knn_rows)) if knn_rows else float("nan")
    metrics["full_vs_knn_gain"] = (
        metrics["knn_predictive"] - metrics["val_world_predictive"]
        if knn_rows else float("nan")
    )

    support = np.concatenate(support_rows)
    sample_error = np.concatenate(error_rows)
    quantiles = np.quantile(support, [0.25, 0.5, 0.75])
    bins = np.digitize(support, quantiles, right=True)
    for bin_idx in range(4):
        mask = bins == bin_idx
        metrics[f"support_q{bin_idx + 1}_count"] = float(mask.sum())
        metrics[f"support_q{bin_idx + 1}_predictive"] = (
            float(sample_error[mask].mean()) if mask.any() else float("nan")
        )
    model.train()
    return metrics


def _checkpoint_payload(
    *,
    model: DynamicPredictiveWorld,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    context: dict[str, Any],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    history: list[dict[str, Any]],
    trainer: DynamicWorldTrainerConfig,
    loss_config: DynamicWorldLossConfig,
) -> dict[str, Any]:
    return {
        "schema": "clearvla-v33.4-dynamic-predictive-world-checkpoint-v3",
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "context": context,
        "action_normalizer": action_normalizer.to_dict(),
        "state_normalizer": state_normalizer.to_dict(),
        "trainer": asdict(trainer),
        "loss_config": asdict(loss_config),
        "history": history,
    }


def train_dynamic_world(
    *,
    model: DynamicPredictiveWorld,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    out_dir: Path,
    trainer: DynamicWorldTrainerConfig,
    loss_config: DynamicWorldLossConfig,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    context: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=trainer.lr,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
        weight_decay=trainer.weight_decay,
    )
    steps_per_epoch = min(len(train_loader), trainer.max_train_batches) if trainer.max_train_batches else len(train_loader)
    scheduler = _scheduler(
        optimizer,
        total_steps=trainer.epochs * steps_per_epoch,
        warmup_steps=trainer.warmup_steps,
        min_lr_ratio=trainer.min_lr_ratio,
    )
    history: list[dict[str, Any]] = []
    best_predictive = float("inf")
    best_closed_loop = float("inf")
    best_action_score = -float("inf")
    best_balanced = float("inf")
    global_step = 0
    epoch_path = out_dir / "dynamic_world_epochs.jsonl"
    if epoch_path.exists():
        epoch_path.unlink()

    for epoch in range(1, trainer.epochs + 1):
        model.train()
        start = time.perf_counter()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            primary = prepare_sample(
                batch["primary"], conditioner=conditioner, model_config=model.config,
                camera_names=camera_names, device=device, dtype=dtype,
            )
            pair = prepare_sample(
                batch["pair"], conditioner=conditioner, model_config=model.config,
                camera_names=camera_names, device=device, dtype=dtype,
            )
            valid = batch["pair_valid"].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward_prepared(model, primary)
            pair_output = model.forward_local_pair(
                pair["current_tokens"], pair["target_tokens"], pair["state"], pair["action"]
            )
            swapped = model.rollout_from_encoded(
                output["context"],
                output["initial_scene"],
                output["initial_dynamic"],
                pair["action"],
                primary["state"],
            )
            losses = compute_dynamic_world_losses(
                model, primary, output, config=loss_config,
                pair=pair, pair_output=pair_output, pair_valid=valid, swapped_output=swapped,
            )
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
                    "[dynamic-world] "
                    f"epoch={epoch:03d}/{trainer.epochs:03d} batch={batch_index:04d} "
                    f"loss={mean['loss']:.6f} world={mean['world_predictive']:.6f} "
                    f"dyn={mean['predictive']:.6f} scene={mean['scene_predictive']:.6f} "
                    f"one={mean['world_teacher_forced']:.6f} gap={mean['closed_loop_gap']:.6f} "
                    f"desc={mean['descriptor']:.6f} state={mean['state_path']:.6f} "
                    f"local={mean['local_effect']:.6f}/{mean['local_effect_cosine']:.3f} "
                    f"swap={mean['swap_regret']:.6f}/{mean['swap_correct_fraction']:.3f} "
                    f"amp={mean['pred_dynamic_rms']:.3f}/{mean['target_dynamic_rms']:.3f} "
                    f"gate={mean['effect_gate_mean']:.3f}/{mean['scene_effect_gate_mean']:.3f} "
                    f"grad={mean['grad']:.3e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

        train_metrics = _mean_rows(rows)
        val_metrics = evaluate_dynamic_world(
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
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        _append_jsonl(epoch_path, record)
        far_gap = val_metrics[
            f"val_closed_loop_gap_t{model.config.future_offsets[-1]}"
        ]
        print(
            "[dynamic-world] "
            f"epoch={epoch:03d}/{trainer.epochs:03d} sec={record['seconds']:.1f} "
            f"val_world={val_metrics['val_world_predictive']:.6f} "
            f"dyn={val_metrics['val_predictive']:.6f} "
            f"scene={val_metrics['val_scene_predictive']:.6f} "
            f"one={val_metrics['val_world_teacher_forced']:.6f} "
            f"gap={val_metrics['val_closed_loop_gap']:.6f}/{far_gap:.6f} "
            f"swap={val_metrics['val_swap_regret']:.6f}/{val_metrics['val_swap_correct_fraction']:.3f} "
            f"local_cos={val_metrics['val_local_effect_cosine']:.3f} "
            f"full-current={val_metrics['full_vs_current_gain']:.6f} "
            f"full-action={val_metrics['full_vs_action_gain']:.6f} "
            f"knn={val_metrics['knn_predictive']:.6f} "
            f"full-knn={val_metrics['full_vs_knn_gain']:.6f} "
            f"state={val_metrics['state_path_rmse']:.5f} "
            f"gripper_f1={val_metrics['gripper_f1']:.3f}",
            flush=True,
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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
        closed_loop_score = predictive + max(
            0.0, val_metrics["val_max_closed_loop_gap"]
        )
        action_score = val_metrics["val_swap_regret"] + val_metrics["val_local_effect_cosine"]
        balanced = (
            predictive
            + max(0.0, -val_metrics["val_swap_regret"])
            + 0.1 * max(0.0, 1.0 - val_metrics["val_local_effect_cosine"])
            + 0.1 * max(0.0, -val_metrics["full_vs_current_gain"])
            + 0.1 * max(0.0, -val_metrics["full_vs_knn_gain"])
            + 0.05 * val_metrics["state_path_rmse"]
        )
        if predictive < best_predictive:
            best_predictive = predictive
            _save(out_dir / "checkpoints/best_predictive.pt", payload)
        if closed_loop_score < best_closed_loop:
            best_closed_loop = closed_loop_score
            _save(out_dir / "checkpoints/best_closed_loop.pt", payload)
        if action_score > best_action_score:
            best_action_score = action_score
            _save(out_dir / "checkpoints/best_action_conditioned.pt", payload)
        if balanced < best_balanced:
            best_balanced = balanced
            _save(out_dir / "checkpoints/best_balanced.pt", payload)

    summary = {
        "schema": "clearvla-v33.4-dynamic-predictive-world-summary-v3",
        "parameter_count": model.parameter_count(),
        "best_predictive": best_predictive,
        "best_closed_loop_score": best_closed_loop,
        "best_action_conditioned_score": best_action_score,
        "best_balanced_score": best_balanced,
        "history": history,
        "context": context,
    }
    (out_dir / "dynamic_world_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "DynamicWorldTrainerConfig",
    "encode_current_tokens",
    "encode_target_tokens",
    "encode_sample_tokens",
    "prepare_sample",
    "evaluate_dynamic_world",
    "train_dynamic_world",
]
