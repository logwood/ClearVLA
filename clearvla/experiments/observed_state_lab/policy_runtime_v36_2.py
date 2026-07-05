from __future__ import annotations

"""Training/evaluation runtime for V36.2 physical-action-flow policy."""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.dynamic_world_lab.shared_runtime import encode_current_tokens, gripper_transition_metrics

from .policy_v36_2 import V362PolicySystem
from .world_runtime import autocast_context, grad_norm, jsonable, scheduler


@dataclass(frozen=True)
class V362PolicyTrainerConfig:
    epochs: int = 12
    lr: float = 1e-4
    proposal_lr: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    proposal_loss_weight: float = 0.05
    first_weight: float = 1.5
    first4_weight: float = 1.3
    first8_weight: float = 1.15
    tail_weight: float = 1.10
    event_loss_weight: float = 0.08
    event_positive_weight: float = 6.0
    event_focal_gamma: float = 1.0
    gripper_transition_l1_weight: float = 0.04
    smooth_delta_weight: float = 0.02
    decoded_action_loss_weight: float = 0.04
    physical_delta_consistency_weight: float = 0.03
    arm_motion_loss_weight: float = 0.03
    arm_motion_threshold: float = 0.02
    gripper_event_threshold: float = 0.10
    deploy_min_recall: float = 0.40
    deploy_min_event_ratio: float = 0.70
    deploy_max_event_ratio: float = 1.80
    deploy_max_tail_first_ratio: float = 2.60
    eval_inference_steps: int = 5
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0


@torch.no_grad()
def prepare_v362_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V362PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    visual = encode_current_tokens(
        sample, conditioner=conditioner, model_config=system.world_config,
        camera_names=camera_names, device=device, dtype=dtype,
    )
    keys = (
        "state", "state_raw", "action_state", "history_state", "executed_action_history",
        "executed_action_history_raw", "policy_action", "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in ("state", "action_state", "history_state", "executed_action_history", "policy_action"):
        out[key] = out[key].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    return out


def position_weights(config, trainer: V362PolicyTrainerConfig, device: torch.device) -> Tensor:
    weight = torch.full((config.action_horizon,), float(trainer.tail_weight), device=device)
    weight[:8] = float(trainer.first8_weight)
    weight[:4] = float(trainer.first4_weight)
    weight[0] = float(trainer.first_weight)
    return weight / weight.mean()


def gripper_event_labels(*, target_raw: Tensor, current_raw: Tensor, gripper_index: int, threshold: float) -> Tensor:
    target_g = target_raw[..., gripper_index].float()
    current_g = current_raw[..., gripper_index].float().reshape(-1, 1)
    boundary = torch.cat([current_g, target_g[:, :-1]], dim=1)
    delta = target_g - boundary
    labels = torch.zeros_like(delta, dtype=torch.long)
    labels = torch.where(delta <= -float(threshold), torch.ones_like(labels), labels)
    labels = torch.where(delta >= float(threshold), torch.full_like(labels, 2), labels)
    return labels


def _focal_cross_entropy(logits: Tensor, labels: Tensor, weights: Tensor, gamma: float) -> Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce.detach()).clamp(min=1e-6, max=1.0)
        ce = ((1.0 - pt) ** float(gamma)) * ce
    return (ce * weights).mean()


def event_head_metrics(logits_rows: list[np.ndarray], target_rows: list[np.ndarray]) -> dict[str, float]:
    if not logits_rows:
        return {}
    logits = np.concatenate(logits_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    pred = logits.argmax(axis=-1)
    out: dict[str, float] = {"event_head_accuracy": float((pred == target).mean())}
    pos_pred = pred != 0
    pos_target = target != 0
    tp = float(np.logical_and(pos_pred, pos_target).sum())
    fp = float(np.logical_and(pos_pred, ~pos_target).sum())
    fn = float(np.logical_and(~pos_pred, pos_target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    out.update({
        "event_head_precision": float(precision),
        "event_head_recall": float(recall),
        "event_head_f1": float(f1),
        "event_head_pred_events": float(pos_pred.sum()),
        "event_head_target_events": float(pos_target.sum()),
    })
    for label, name in ((1, "open"), (2, "close")):
        p = pred == label
        t = target == label
        ltp = float(np.logical_and(p, t).sum())
        lfp = float(np.logical_and(p, ~t).sum())
        lfn = float(np.logical_and(~p, t).sum())
        lp = ltp / max(ltp + lfp, 1.0)
        lr = ltp / max(ltp + lfn, 1.0)
        lf1 = 2.0 * lp * lr / max(lp + lr, 1e-8)
        out[f"event_head_{name}_precision"] = float(lp)
        out[f"event_head_{name}_recall"] = float(lr)
        out[f"event_head_{name}_f1"] = float(lf1)
    return out


def arm_motion_labels(system: V362PolicySystem, target_action: Tensor, action_state: Tensor, threshold: float) -> Tensor:
    physical = system.codec.encode(target_action, action_state)
    parts = system.codec.split_physical(physical)
    norm = parts["arm_delta"].float().norm(dim=-1)
    return (norm >= float(threshold)).to(target_action.dtype)


def flow_losses(
    system: V362PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V362PolicyTrainerConfig,
) -> dict[str, Tensor]:
    cfg = system.policy_config
    device = output["pred_physical_velocity"].device
    weight = position_weights(cfg, trainer, device)
    physical_error = (output["pred_physical_velocity"] - output["target_physical_velocity"]).square().mean(dim=-1)
    flow = (physical_error * weight[None]).mean()
    proposal = F.smooth_l1_loss(output["proposal_action"], sample["policy_action"])

    labels = gripper_event_labels(
        target_raw=sample["policy_action_raw"].to(device=device),
        current_raw=sample["state_raw"].to(device=device),
        gripper_index=cfg.gripper_index,
        threshold=trainer.gripper_event_threshold,
    )
    flat_labels = labels.reshape(-1)
    flat_logits = output["event_logits"].reshape(-1, 3)
    event_weights = torch.ones_like(flat_labels, dtype=flat_logits.dtype)
    event_weights = event_weights + (flat_labels != 0).to(flat_logits.dtype) * float(trainer.event_positive_weight)
    event = _focal_cross_entropy(flat_logits, flat_labels, event_weights, trainer.event_focal_gamma)

    motion_target = arm_motion_labels(
        system,
        sample["policy_action"].to(device=device),
        sample["action_state"].to(device=device),
        trainer.arm_motion_threshold,
    )
    motion = F.binary_cross_entropy_with_logits(output["motion_logits"].float(), motion_target.float())

    transition_mask = (labels != 0).to(output["pred_action_estimate"].dtype)
    grip_idx = cfg.gripper_index
    pred_g = output["pred_action_estimate"][..., grip_idx]
    target_g = sample["policy_action"].to(device=device)[..., grip_idx]
    transition_l1 = (F.smooth_l1_loss(pred_g, target_g, reduction="none") * (1.0 + transition_mask * 8.0)).mean()

    pred_boundary = torch.cat([sample["action_state"].to(device=device)[:, None], output["pred_action_estimate"][:, :-1]], dim=1)
    target_boundary = torch.cat([sample["action_state"].to(device=device)[:, None], sample["policy_action"].to(device=device)[:, :-1]], dim=1)
    pred_delta = output["pred_action_estimate"] - pred_boundary
    target_delta = sample["policy_action"].to(device=device) - target_boundary
    smooth_delta = F.smooth_l1_loss(pred_delta, target_delta)
    decoded_action = F.smooth_l1_loss(output["pred_action_estimate"], sample["policy_action"].to(device=device))
    physical_delta_consistency = system.codec.delta_consistency_loss(
        output["clean_physical_estimate"], sample["action_state"].to(device=device), output["pred_action_estimate"]
    )

    total = (
        flow
        + trainer.proposal_loss_weight * proposal
        + trainer.event_loss_weight * event
        + trainer.arm_motion_loss_weight * motion
        + trainer.gripper_transition_l1_weight * transition_l1
        + trainer.smooth_delta_weight * smooth_delta
        + trainer.decoded_action_loss_weight * decoded_action
        + trainer.physical_delta_consistency_weight * physical_delta_consistency
    )
    pred_event = output["event_logits"].argmax(dim=-1)
    pos_target = labels != 0
    pos_pred = pred_event != 0
    tp = (pos_pred & pos_target).sum().to(torch.float32)
    fp = (pos_pred & ~pos_target).sum().to(torch.float32)
    fn = (~pos_pred & pos_target).sum().to(torch.float32)
    event_precision = tp / torch.clamp(tp + fp, min=1.0)
    event_recall = tp / torch.clamp(tp + fn, min=1.0)
    motion_pred = torch.sigmoid(output["motion_logits"]) >= 0.5
    motion_target_bool = motion_target >= 0.5
    mtp = (motion_pred & motion_target_bool).sum().to(torch.float32)
    mfp = (motion_pred & ~motion_target_bool).sum().to(torch.float32)
    mfn = (~motion_pred & motion_target_bool).sum().to(torch.float32)
    motion_precision = mtp / torch.clamp(mtp + mfp, min=1.0)
    motion_recall = mtp / torch.clamp(mtp + mfn, min=1.0)
    return {
        "loss": total,
        "physical_flow": flow,
        "proposal": proposal,
        "event": event,
        "motion": motion,
        "transition_l1": transition_l1,
        "smooth_delta": smooth_delta,
        "decoded_action": decoded_action,
        "physical_delta_consistency": physical_delta_consistency,
        "first_physical_flow": physical_error[:, 0].mean(),
        "first4_physical_flow": physical_error[:, :4].mean(),
        "first8_physical_flow": physical_error[:, :8].mean(),
        "tail_physical_flow": physical_error[:, 8:].mean(),
        "event_head_precision": event_precision,
        "event_head_recall": event_recall,
        "motion_head_precision": motion_precision,
        "motion_head_recall": motion_recall,
    }


def decode(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(row) for row in rows)) if rows else set()
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


@torch.no_grad()
def evaluate_v362_policy(
    *,
    system: V362PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V362PolicyTrainerConfig,
    max_batches: int = 0,
) -> dict[str, float]:
    system.eval()
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    event_logits_rows: list[np.ndarray] = []
    event_target_rows: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        sample = prepare_v362_policy_sample(batch, conditioner=conditioner, system=system, camera_names=camera_names, device=device, dtype=dtype)
        generator = torch.Generator(device=device)
        generator.manual_seed(36236 + batch_index)
        noise = system.codec.sample_noise(
            sample["policy_action"].shape[0],
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
        )
        with autocast_context(device, dtype):
            pred_pack = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=True, return_event_logits=True,
            )
            assert isinstance(pred_pack, dict)
            no_proposal = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=False,
            )
        pred_rows.append(decode(action_normalizer, pred_pack["action"]))
        no_proposal_rows.append(decode(action_normalizer, no_proposal))
        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"], current_raw=sample["state_raw"],
            gripper_index=system.policy_config.gripper_index, threshold=trainer.gripper_event_threshold,
        )
        event_logits_rows.append(pred_pack["event_logits"].detach().float().cpu().numpy())
        event_target_rows.append(labels.cpu().numpy())
    pred = np.concatenate(pred_rows)
    no_proposal = np.concatenate(no_proposal_rows)
    target = np.concatenate(target_rows)
    current = np.concatenate(current_rows)
    squared = (pred - target) ** 2
    metrics = {
        "full_mse": float(squared.mean()),
        "full_rmse": float(np.sqrt(squared.mean())),
        "first_rmse": float(np.sqrt(squared[:, 0].mean())),
        "first4_rmse": float(np.sqrt(squared[:, :4].mean())),
        "first8_rmse": float(np.sqrt(squared[:, :8].mean())),
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean())) if squared.shape[1] > 8 else float("nan"),
        "arm_full_rmse": float(np.sqrt(squared[..., :-1].mean())),
        "gripper_full_rmse": float(np.sqrt(squared[..., -1].mean())),
        "proposal_utility_mse_gain": float(((no_proposal - target) ** 2).mean() - squared.mean()),
    }
    metrics.update(gripper_transition_metrics(
        pred, target, current, gripper_index=system.policy_config.gripper_index,
        threshold=trainer.gripper_event_threshold, tolerance=2,
    ))
    metrics.update(event_head_metrics(event_logits_rows, event_target_rows))
    metrics["tail_first_ratio"] = float(metrics["tail_rmse"] / max(metrics["first_rmse"], 1e-8))
    metrics["gripper_event_ratio"] = float(metrics.get("gripper_pred_events", 0.0) / max(metrics.get("gripper_target_events", 0.0), 1.0))
    return metrics


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def balanced_score(metrics: dict[str, float], trainer: V362PolicyTrainerConfig) -> float:
    full = float(metrics["full_rmse"])
    f1 = float(metrics.get("gripper_f1", 0.0))
    recall = float(metrics.get("gripper_recall", 0.0))
    ratio = float(metrics.get("gripper_event_ratio", 0.0))
    tail_first = float(metrics.get("tail_first_ratio", 999.0))
    ratio_penalty = 0.0 if ratio > 0 else 1.0
    if ratio > 0:
        low = float(trainer.deploy_min_event_ratio)
        high = float(trainer.deploy_max_event_ratio)
        ratio_penalty = max(0.0, math.log(low / ratio)) + max(0.0, math.log(ratio / high))
    return float(
        full
        + 0.03 * (1.0 - f1)
        + 0.05 * max(0.0, float(trainer.deploy_min_recall) - recall)
        + 0.02 * ratio_penalty
        + 0.01 * max(0.0, tail_first - float(trainer.deploy_max_tail_first_ratio))
    )


def is_deploy_eligible(metrics: dict[str, float], trainer: V362PolicyTrainerConfig) -> bool:
    ratio = float(metrics.get("gripper_event_ratio", 0.0))
    return (
        float(metrics.get("gripper_recall", 0.0)) >= float(trainer.deploy_min_recall)
        and float(trainer.deploy_min_event_ratio) <= ratio <= float(trainer.deploy_max_event_ratio)
        and float(metrics.get("tail_first_ratio", 999.0)) <= float(trainer.deploy_max_tail_first_ratio)
    )


def train_v362_policy(
    *,
    system: V362PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V362PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    system.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [
            {"params": system.planner.parameters(), "lr": trainer.lr},
            {"params": system.decoder.parameters(), "lr": trainer.lr},
            {"params": system.proposal.parameters(), "lr": trainer.proposal_lr},
        ],
        weight_decay=trainer.weight_decay, betas=(trainer.beta1, trainer.beta2), eps=trainer.eps,
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    schedule = scheduler(optimizer, steps_per_epoch * trainer.epochs, trainer.warmup_steps, trainer.min_lr_ratio)
    start_epoch, global_step = 1, 0
    history: list[dict[str, Any]] = []
    best = {"full_mse": float("inf"), "gripper_f1": -float("inf"), "gripper_recall": -float("inf"), "balanced": float("inf"), "deploy_full_rmse": float("inf")}
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clearvla-v36-2-policy-checkpoint-v1":
            raise ValueError("resume checkpoint is not V36.2 policy")
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
        history = list(payload.get("history", [])); best.update(payload.get("best", {})); restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train(); rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            sample = prepare_v362_policy_sample(batch, conditioner=conditioner, system=system, camera_names=camera_names, device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                output = system.flow_training_forward(sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"], sample["policy_action"])
                losses = flow_losses(system, sample, output, trainer)
            losses["loss"].float().backward()
            grad = grad_norm(system.parameters())
            torch.nn.utils.clip_grad_norm_(system.parameters(), trainer.grad_clip)
            optimizer.step(); schedule.step(); global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = grad; rows.append(row)
            if trainer.log_every and batch_index % trainer.log_every == 0:
                print(
                    f"[v36.2-physical-action-flow] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} decode={row['decoded_action']:.6f} cons={row['physical_delta_consistency']:.6f} "
                    f"event={row['event']:.6f} motion={row['motion']:.6f} first={row['first_physical_flow']:.6f} "
                    f"evtR={row['event_head_recall']:.3f} motR={row['motion_head_recall']:.3f} grad={grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = mean_rows(rows)
        val_metrics = evaluate_v362_policy(system=system, loader=val_loader, conditioner=conditioner, device=device, dtype=dtype, camera_names=camera_names, action_normalizer=action_normalizer, trainer=trainer, max_batches=trainer.max_val_batches)
        score = balanced_score(val_metrics, trainer)
        deploy_eligible = is_deploy_eligible(val_metrics, trainer)
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with (out_dir / "v36_2_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics["full_mse"])
        f1 = float(val_metrics.get("gripper_f1", 0.0))
        recall = float(val_metrics.get("gripper_recall", 0.0))
        save = []
        if full < best["full_mse"]:
            best["full_mse"] = full; save.append("best_full.pt")
        if f1 > best["gripper_f1"]:
            best["gripper_f1"] = f1; save.append("best_gripper_f1.pt")
        if recall > best["gripper_recall"]:
            best["gripper_recall"] = recall; save.append("best_gripper_recall.pt")
        if score < best["balanced"]:
            best["balanced"] = score; save.append("best_balanced.pt")
        if deploy_eligible and float(val_metrics["full_rmse"]) < best["deploy_full_rmse"]:
            best["deploy_full_rmse"] = float(val_metrics["full_rmse"]); save.append("best_deploy.pt")
        payload = {
            "schema": "clearvla-v36-2-policy-checkpoint-v1", "epoch": epoch, "global_step": global_step,
            "model": system.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
            "world_config": asdict(system.world_config), "policy_config": asdict(system.policy_config),
            "trainer_config": asdict(trainer), "action_normalizer": action_normalizer.to_dict(), "state_normalizer": state_normalizer.to_dict(),
            "context": context, "history": history, "best": best, "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v36_2_policy_summary.json").write_text(json.dumps(jsonable({"schema": "clearvla-v36-2-policy-summary-v1", "best": best, "latest": record}), indent=2), encoding="utf-8")
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V362PolicyTrainerConfig",
    "prepare_v362_policy_sample",
    "gripper_event_labels",
    "flow_losses",
    "evaluate_v362_policy",
    "train_v362_policy",
    "balanced_score",
    "is_deploy_eligible",
]
