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
    gripper_transition_metrics,
)

from .policy import V35PolicySystem
from .world_runtime import autocast_context, grad_norm, jsonable, scheduler


@dataclass(frozen=True)
class V35PolicyTrainerConfig:
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
    proposal_loss_weight: float = 0.10
    first_weight: float = 4.0
    first4_weight: float = 2.0
    first8_weight: float = 1.5
    tail_weight: float = 1.0
    eval_inference_steps: int = 5
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0


@torch.no_grad()
def prepare_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V35PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    visual = encode_current_tokens(
        sample,
        conditioner=conditioner,
        model_config=system.world_config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    keys = (
        "state",
        "state_raw",
        "history_state",
        "executed_action_history",
        "executed_action_history_raw",
        "policy_action",
        "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in ("state", "history_state", "executed_action_history", "policy_action"):
        out[key] = out[key].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    return out


def position_weights(config, trainer: V35PolicyTrainerConfig, device: torch.device) -> Tensor:
    weight = torch.full((config.action_horizon,), float(trainer.tail_weight), device=device)
    weight[:8] = float(trainer.first8_weight)
    weight[:4] = float(trainer.first4_weight)
    weight[0] = float(trainer.first_weight)
    return weight / weight.mean()


def flow_losses(
    system: V35PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V35PolicyTrainerConfig,
) -> dict[str, Tensor]:
    weight = position_weights(system.policy_config, trainer, output["pred_velocity"].device)
    error = (output["pred_velocity"] - output["target_velocity"]).square().mean(dim=-1)
    flow = (error * weight[None]).mean()
    proposal = torch.nn.functional.smooth_l1_loss(
        output["proposal_action"], sample["policy_action"]
    )
    total = flow + trainer.proposal_loss_weight * proposal
    return {
        "loss": total,
        "flow": flow,
        "proposal": proposal,
        "first_flow": error[:, 0].mean(),
        "first4_flow": error[:, :4].mean(),
        "first8_flow": error[:, :8].mean(),
        "tail_flow": error[:, 8:].mean(),
    }


def decode(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(row) for row in rows))
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


@torch.no_grad()
def evaluate_v35_policy(
    *,
    system: V35PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V35PolicyTrainerConfig,
    max_batches: int = 0,
) -> dict[str, float]:
    system.eval()
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        sample = prepare_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(35035 + batch_index)
        noise = torch.randn(
            sample["policy_action"].shape,
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
        )
        with autocast_context(device, dtype):
            pred = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=True,
            )
            no_proposal = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=False,
            )
        pred_rows.append(decode(action_normalizer, pred))
        no_proposal_rows.append(decode(action_normalizer, no_proposal))
        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
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
            threshold=0.10,
            tolerance=2,
        )
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


def train_v35_policy(
    *,
    system: V35PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V35PolicyTrainerConfig,
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
            {"params": system.expert.parameters(), "lr": trainer.lr},
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
    best = {"full_mse": float("inf"), "balanced": float("inf")}
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clearvla-v35-policy-checkpoint-v1":
            raise ValueError("resume checkpoint is not V35 policy")
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
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            sample = prepare_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                output = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                )
                losses = flow_losses(system, sample, output, trainer)
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
                    f"[v35-policy] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"flow={row['flow']:.6f} first={row['first_flow']:.6f} proposal={row['proposal']:.6f} "
                    f"grad={grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = mean_rows(rows)
        val_metrics = evaluate_v35_policy(
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
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        with (out_dir / "v35_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics["full_mse"])
        balanced = (
            full
            + 0.25 * float(val_metrics["first8_rmse"])
            + 0.10 * (1 - float(val_metrics.get("gripper_f1", 0.0)))
        )
        save = []
        if full < best["full_mse"]:
            best["full_mse"] = full
            save.append("best_full.pt")
        if balanced < best["balanced"]:
            best["balanced"] = balanced
            save.append("best_balanced.pt")
        payload = {
            "schema": "clearvla-v35-policy-checkpoint-v1",
            "epoch": epoch,
            "global_step": global_step,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "world_config": asdict(system.world_config),
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
        (out_dir / "v35_policy_summary.json").write_text(
            json.dumps(
                jsonable(
                    {"schema": "clearvla-v35-policy-summary-v1", "best": best, "latest": record}
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V35PolicyTrainerConfig",
    "prepare_policy_sample",
    "evaluate_v35_policy",
    "train_v35_policy",
]
