from __future__ import annotations

"""Training/evaluation runtime for V37 full-latent world-shaped policy."""

import json
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
from clearvla.experiments.dynamic_world_lab.shared_runtime import (
    encode_current_tokens,
    encode_target_tokens,
    gripper_transition_metrics,
)

from .policy_v37 import V37PolicySystem
from .policy_runtime_v36_3 import (
    V363PolicyTrainerConfig,
    arm_motion_labels,
    balanced_score,
    decode,
    event_head_metrics,
    flow_losses as v363_flow_losses,
    gripper_event_labels,
    is_deploy_eligible,
    mean_rows,
)
from .world_runtime import autocast_context, grad_norm, jsonable, scheduler


@dataclass(frozen=True)
class V37PolicyTrainerConfig(V363PolicyTrainerConfig):
    future_latent_loss_weight: float = 0.03
    future_latent_loss_start_epoch: int = 2
    future_latent_max_batches: int = 0


@torch.no_grad()
def prepare_v37_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V37PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    include_target_visual: bool = False,
) -> dict[str, Tensor]:
    visual = encode_current_tokens(
        sample,
        conditioner=conditioner,
        model_config=system.policy_config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    keys = (
        "state",
        "state_raw",
        "action_state",
        "history_state",
        "executed_action_history",
        "executed_action_history_raw",
        "policy_action",
        "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in (
        "state",
        "action_state",
        "history_state",
        "executed_action_history",
        "policy_action",
    ):
        out[key] = out[key].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    if include_target_visual:
        target_visual = encode_target_tokens(
            sample,
            conditioner=conditioner,
            model_config=system.policy_config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        out["target_visual"] = target_visual.to(dtype=compute_dtype)
    return out


def future_latent_loss(output: dict[str, Tensor]) -> Tensor:
    if "future_latent_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["future_latent_pred"].float()
    target = output["future_latent_target"].float().detach()
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def flow_losses(
    system: V37PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V37PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    losses = v363_flow_losses(system, sample, output, trainer)  # type: ignore[arg-type]
    fl = future_latent_loss(output)
    losses["future_latent"] = fl
    if enable_future_loss and float(trainer.future_latent_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.future_latent_loss_weight) * fl
    return losses


@torch.no_grad()
def evaluate_v37_policy(
    *,
    system: V37PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V37PolicyTrainerConfig,
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
        sample = prepare_v37_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(37237 + batch_index)
        noise = system.codec.sample_noise(
            sample["policy_action"].shape[0],
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
        )
        with autocast_context(device, dtype):
            pred_pack = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=True,
                return_event_logits=True,
            )
            assert isinstance(pred_pack, dict)
            no_proposal = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=False,
            )
        pred_rows.append(decode(action_normalizer, pred_pack["action"]))
        no_proposal_rows.append(decode(action_normalizer, no_proposal))
        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"],
            current_raw=sample["state_raw"],
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
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
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean()))
        if squared.shape[1] > 8
        else float("nan"),
        "arm_full_rmse": float(np.sqrt(squared[..., :-1].mean())),
        "gripper_full_rmse": float(np.sqrt(squared[..., -1].mean())),
        "proposal_utility_mse_gain": float(((no_proposal - target) ** 2).mean() - squared.mean()),
    }
    metrics.update(
        gripper_transition_metrics(
            pred,
            target,
            current,
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
            tolerance=2,
        )
    )
    metrics.update(event_head_metrics(event_logits_rows, event_target_rows))
    metrics["tail_first_ratio"] = float(metrics["tail_rmse"] / max(metrics["first_rmse"], 1e-8))
    metrics["gripper_event_ratio"] = float(
        metrics.get("gripper_pred_events", 0.0)
        / max(metrics.get("gripper_target_events", 0.0), 1.0)
    )
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


def train_v37_policy(
    *,
    system: V37PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V37PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    system.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [
            {"params": system.planner.parameters(), "lr": trainer.lr},
            {"params": system.proposal.parameters(), "lr": trainer.proposal_lr},
        ],
        weight_decay=trainer.weight_decay,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    schedule = scheduler(
        optimizer, steps_per_epoch * trainer.epochs, trainer.warmup_steps, trainer.min_lr_ratio
    )
    start_epoch, global_step = 1, 0
    history: list[dict[str, Any]] = []
    best = {
        "full_mse": float("inf"),
        "gripper_f1": -float("inf"),
        "gripper_recall": -float("inf"),
        "balanced": float("inf"),
        "deploy_full_rmse": float("inf"),
    }
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clearvla-v37-policy-checkpoint-v1":
            raise ValueError("resume checkpoint is not V37 policy")
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        history = list(payload.get("history", []))
        best.update(payload.get("best", {}))
        restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train()
        rows = []
        include_future = float(trainer.future_latent_loss_weight) > 0 and epoch >= int(
            trainer.future_latent_loss_start_epoch
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            use_future = include_future and (
                not trainer.future_latent_max_batches
                or batch_index <= trainer.future_latent_max_batches
            )
            sample = prepare_v37_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
                include_target_visual=use_future,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                output = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                    target_visual=sample.get("target_visual"),
                )
                losses = flow_losses(system, sample, output, trainer, enable_future_loss=use_future)
            losses["loss"].float().backward()
            grad = grad_norm(system.parameters())
            torch.nn.utils.clip_grad_norm_(system.parameters(), trainer.grad_clip)
            optimizer.step()
            schedule.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = grad
            rows.append(row)
            if trainer.log_every and batch_index % trainer.log_every == 0:
                print(
                    f"[v37-full-latent] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} decode={row['decoded_action']:.6f} future={row['future_latent']:.6f} "
                    f"event={row['event']:.6f} grad={grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = mean_rows(rows)
        val_metrics = evaluate_v37_policy(
            system=system,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            action_normalizer=action_normalizer,
            trainer=trainer,
            max_batches=trainer.max_val_batches,
        )
        score = balanced_score(val_metrics, trainer)  # type: ignore[arg-type]
        deploy_eligible = is_deploy_eligible(val_metrics, trainer)  # type: ignore[arg-type]
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        with (out_dir / "v37_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics["full_mse"])
        f1 = float(val_metrics.get("gripper_f1", 0.0))
        recall = float(val_metrics.get("gripper_recall", 0.0))
        save = []
        if full < best["full_mse"]:
            best["full_mse"] = full
            save.append("best_full.pt")
        if f1 > best["gripper_f1"]:
            best["gripper_f1"] = f1
            save.append("best_gripper_f1.pt")
        if recall > best["gripper_recall"]:
            best["gripper_recall"] = recall
            save.append("best_gripper_recall.pt")
        if score < best["balanced"]:
            best["balanced"] = score
            save.append("best_balanced.pt")
        if deploy_eligible and float(val_metrics["full_rmse"]) < best["deploy_full_rmse"]:
            best["deploy_full_rmse"] = float(val_metrics["full_rmse"])
            save.append("best_deploy.pt")
        payload = {
            "schema": "clearvla-v37-policy-checkpoint-v1",
            "epoch": epoch,
            "global_step": global_step,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "policy_config": asdict(system.policy_config),
            "trainer_config": asdict(trainer),
            "action_normalizer": action_normalizer.to_dict(),
            "state_normalizer": state_normalizer.to_dict(),
            "context": context,
            "history": history,
            "best": best,
            "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v37_policy_summary.json").write_text(
            json.dumps(
                jsonable(
                    {"schema": "clearvla-v37-policy-summary-v1", "best": best, "latest": record}
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V37PolicyTrainerConfig",
    "prepare_v37_policy_sample",
    "future_latent_loss",
    "flow_losses",
    "evaluate_v37_policy",
    "train_v37_policy",
]
