from __future__ import annotations

"""Training/evaluation runtime for V38.6 controlled-residual latent dynamics policy."""

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
from clearvla.experiments.dynamic_world_lab.shared_runtime import encode_current_tokens, encode_target_tokens, gripper_transition_metrics

from .policy_v38 import V38PolicySystem
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
class V38PolicyTrainerConfig(V363PolicyTrainerConfig):
    # V38.6.2 primary action-centered controlled-residual latent-dynamics objectives. Future
    # target tokens are targets only; no future-noisy latent is fed to the model.
    rollout_dynamics_loss_weight: float = 0.03
    rollout_delta_loss_weight: float = 0.01
    rollout_contrast_loss_weight: float = 0.06
    rollout_contrast_margin: float = 0.02
    # Kept for CLI/checkpoint compatibility; defaults disabled to avoid the
    # V38.3 future self-denoise shortcut.
    future_latent_loss_weight: float = 0.0
    action_effect_loss_weight: float = 0.0
    future_latent_loss_start_epoch: int = 1
    future_latent_max_batches: int = 0

    # Lightweight CUDA memory accounting. Disabled by default. Set
    # --memory-report-every N to emit a [cuda-mem] line and append JSONL rows
    # every N batches. Set --memory-report-detail 1 to also log stage-level
    # points inside the selected batch. This is monitoring only; it does not
    # change the V38.3 throughput-optimized training/eval semantics.
    memory_report_every: int = 0
    memory_report_detail: int = 0
    memory_report_sync: int = 0


def _validate_current_token_tensor(tokens: Tensor, *, system: V38PolicySystem) -> None:
    cfg = system.policy_config
    expected = (
        cfg.visual_history_length,
        cfg.num_cameras,
        cfg.patches_per_camera,
        cfg.visual_token_dim,
    )
    if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"history_dinov2_tokens must be [B,{expected}], got {tuple(tokens.shape)}")


def _validate_target_anchor_token_tensor(tokens: Tensor, *, system: V38PolicySystem) -> None:
    cfg = system.policy_config
    expected_tail = (cfg.num_cameras, cfg.patches_per_camera, cfg.visual_token_dim)
    if tokens.ndim != 5 or tuple(tokens.shape[2:]) != expected_tail:
        raise ValueError(f"target_future_dinov2_tokens must be [B,F,{expected_tail}], got {tuple(tokens.shape)}")
    if int(tokens.shape[1]) < int(cfg.future_anchors):
        raise ValueError(f"target_future_dinov2_tokens has only {tokens.shape[1]} anchors; need {cfg.future_anchors}")


@torch.no_grad()
def encode_target_anchor_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode only V38's required future-anchor target tokens.

    V38's residual future-flow target only consumes the last target-history
    frame for the first ``future_anchors`` offsets.  The legacy helper encoded
    all ``num_future × history`` targets; with the default 12 future offsets and
    3 history frames that read 36 frames while V38 needed only 4.  This helper
    returns [B,F,1,C,P,D] so it stays compatible with
    ``TemporalWorldActionDiT.target_future_latent`` without over-reading.
    """
    batch = int(sample["state"].shape[0])
    anchors = int(getattr(model_config, "future_anchors", getattr(model_config, "num_future", 1)))
    if "target_future_dinov2_tokens" in sample:
        tokens = sample["target_future_dinov2_tokens"][:, :anchors]
        _validate_target_anchor_token_tensor(tokens, system=model_config_owner(model_config))
        return tokens.to(device=device, dtype=dtype, non_blocking=True)[:, :, None]
    if "target_history_obs_image" in sample:
        images = sample["target_history_obs_image"][:, :anchors, -1]
        flat = images.reshape(batch * anchors, *images.shape[2:])
        condition = conditioner.encode(flat, camera_names=camera_names)
    else:
        keys = sample["target_history_keys"][:, :anchors, -1, :].reshape(batch * anchors, 2)
        dummy = torch.zeros(batch * anchors, model_config.num_cameras, 3, 1, 1, dtype=torch.float32)
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("V38 future target requires dense DINO tokens")
    dense = condition.dense_tokens
    expected_tokens = model_config.num_cameras * model_config.patches_per_camera
    if dense.ndim != 3 or dense.shape[0] != batch * anchors or dense.shape[1] != expected_tokens or dense.shape[2] != model_config.latent_dim:
        raise ValueError(
            "DINO target-anchor geometry mismatch: "
            f"got {tuple(dense.shape)}, expected ({batch * anchors},{expected_tokens},{model_config.latent_dim})"
        )
    return dense.reshape(
        batch,
        anchors,
        model_config.num_cameras,
        model_config.patches_per_camera,
        model_config.latent_dim,
    ).to(device=device, dtype=dtype)[:, :, None]


def model_config_owner(model_config):
    # Small adapter so token-prefetch validation can reuse the full system-style
    # shape checker.  It intentionally exposes only policy_config.
    class _Owner:
        policy_config = model_config
    return _Owner()


@torch.no_grad()
def prepare_v38_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V38PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    include_target_visual: bool = False,
) -> dict[str, Tensor]:
    if "history_dinov2_tokens" in sample:
        visual = sample["history_dinov2_tokens"]
        _validate_current_token_tensor(visual, system=system)
        visual = visual.to(device=device, dtype=dtype, non_blocking=True)
    else:
        visual = encode_current_tokens(
            sample, conditioner=conditioner, model_config=system.policy_config,
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
    if include_target_visual:
        target_visual = encode_target_anchor_tokens(
            sample, conditioner=conditioner, model_config=system.policy_config,
            camera_names=camera_names, device=device, dtype=dtype,
        )
        out["target_visual"] = target_visual.to(dtype=compute_dtype)
    return out


def _effect_distance(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.float() - target.float().detach()
    mse = diff.square().mean(dim=(1, 2))
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target.float().detach(), dim=-1)
    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean(dim=1)
    return mse + 0.10 * cosine



def _rollout_residual_target(output: dict[str, Tensor]) -> Tensor:
    target = output["rollout_effect_target"].float().detach()
    if "rollout_base_effect_pred" not in output:
        return target
    base = output["rollout_base_effect_pred"].float().detach()
    return target - base


def rollout_delta_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise the action-centered controlled delta after removing visual baseline.

    This is the anti-average-future objective: the weak base may explain common
    visual/phase trends, while the delta must explain the residual that should
    change under action counterfactuals.
    """
    if "rollout_effect_target" not in output or "rollout_delta_pred" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_delta_pred"].float()
    target = _rollout_residual_target(output)
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def rollout_dynamics_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise action-conditioned rollout latent against future residual.

    The target is stop-gradient DINO future-current residual. It is never fed
    as an input to the model, so this cannot become future self-denoising.
    """
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def rollout_contrast_loss(output: dict[str, Tensor], *, margin: float = 0.02) -> Tensor:
    """Force real action-controlled delta to beat hold/shuffled deltas.

    V38.6.2 applies contrast to the action-centered ``controlled_delta`` rather
    than to the full ``base + delta`` effect.  The fixed observation baseline is
    therefore shared by real/hold/shuffle and cannot dilute the counterfactual
    signal; coefficient centering additionally removes action-independent bias
    from the delta branch itself.
    """
    if "rollout_effect_target" not in output or "rollout_delta_pred_hold_action" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    target = _rollout_residual_target(output)
    real = _effect_distance(output["rollout_delta_pred"], target)
    hold = _effect_distance(output["rollout_delta_pred_hold_action"], target)
    shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target)
    m = torch.as_tensor(float(margin), device=real.device, dtype=real.dtype)
    return (F.relu(m + real - hold) + F.relu(m + real - shuf)).mean()

def rollout_diagnostics(output: dict[str, Tensor]) -> dict[str, Tensor]:
    rows: dict[str, Tensor] = {}
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        z = torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
        for key in (
            "rollout_distance_real", "rollout_distance_hold", "rollout_distance_shuffle",
            "rollout_delta_hold", "rollout_delta_shuffle", "rollout_full_distance_real",
        ):
            rows[key] = z
        return rows
    target_full = output["rollout_effect_target"].float().detach()
    full_real = _effect_distance(output["rollout_effect_pred"], target_full).mean()
    rows["rollout_full_distance_real"] = full_real.detach()
    rows["rollout_distance_real"] = full_real.detach()
    rows["rollout_base_mse"] = (
        (output.get("rollout_base_effect_pred", output["rollout_effect_pred"]).float() - target_full).square().mean().detach()
    )
    rows["rollout_delta_target_norm"] = _rollout_residual_target(output).detach().float().norm(dim=-1).mean()
    if "rollout_delta_pred" in output:
        target_delta = _rollout_residual_target(output)
        delta_real = _effect_distance(output["rollout_delta_pred"], target_delta).mean()
        rows["rollout_delta_distance_real"] = delta_real.detach()
        # In V38.6 the legacy rollout_distance/delta names refer to the
        # controlled-delta contrast path because that is the causal path being
        # tested by train logs and offline diagnostics.
        rows["rollout_distance_real"] = delta_real.detach()
    if "rollout_delta_pred_hold_action" in output:
        target_delta = _rollout_residual_target(output)
        hold = _effect_distance(output["rollout_delta_pred_hold_action"], target_delta).mean()
        shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target_delta).mean()
        real = rows["rollout_distance_real"]
        rows["rollout_distance_hold"] = hold.detach()
        rows["rollout_distance_shuffle"] = shuf.detach()
        rows["rollout_delta_hold"] = (hold - real).detach()
        rows["rollout_delta_shuffle"] = (shuf - real).detach()
        rows["rollout_effect_change_hold"] = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_hold_action"].float()).square().mean().detach()
        rows["rollout_effect_change_shuffle"] = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_shuffle_action"].float()).square().mean().detach()
        rows["rollout_full_effect_change_hold"] = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_hold_action"].float()).square().mean().detach()
        rows["rollout_full_effect_change_shuffle"] = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_shuffle_action"].float()).square().mean().detach()
    for key in (
        "rollout_coeff_abs_mean",
        "rollout_neutral_coeff_abs_mean",
        "rollout_centered_coeff_abs_mean",
        "rollout_basis_norm",
        "rollout_delta_norm",
        "rollout_base_norm",
        "rollout_delta_gain",
    ):
        if key in output:
            rows[key] = output[key].detach().float()
    return rows

def future_latent_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this is the full base+delta rollout loss, not a
    # future-noisy denoising loss.
    return rollout_dynamics_loss(output)


def action_effect_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this now tracks the controlled action delta path.
    return rollout_delta_loss(output)

def flow_losses(
    system: V38PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V38PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    losses = v363_flow_losses(system, sample, output, trainer)  # type: ignore[arg-type]
    dyn = rollout_dynamics_loss(output)
    delta = rollout_delta_loss(output)
    con = rollout_contrast_loss(output, margin=float(trainer.rollout_contrast_margin))
    losses["rollout_dynamics"] = dyn
    losses["rollout_delta"] = delta
    losses["rollout_contrast"] = con
    # Compatibility log names: these no longer correspond to self-denoise.
    losses["future_latent"] = dyn.detach()
    losses["action_effect"] = delta.detach()
    if enable_future_loss and float(trainer.rollout_dynamics_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_dynamics_loss_weight) * dyn
    if enable_future_loss and float(trainer.rollout_delta_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_delta_loss_weight) * delta
    if enable_future_loss and float(trainer.rollout_contrast_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_contrast_loss_weight) * con
    # Disabled by default. Kept only as compatibility knobs; they map to the
    # same dynamics-bound target rather than future-noisy denoise.
    if enable_future_loss and float(trainer.future_latent_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.future_latent_loss_weight) * dyn
    if enable_future_loss and float(trainer.action_effect_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.action_effect_loss_weight) * dyn
    losses.update(rollout_diagnostics(output))
    if "gate_self" in output:
        losses["gate_self"] = output["gate_self"].detach()
        losses["gate_visual"] = output["gate_visual"].detach()
        losses["gate_rollout"] = output.get("gate_rollout", torch.zeros_like(output["gate_self"])).detach()
        losses["gate_ffn"] = output["gate_ffn"].detach()
    for key in (
        "mod_content_norm", "mod_time_norm", "mod_content_to_time",
        "future_conditioned_action_loss",
    ):
        if key in output:
            losses[key] = output[key].detach()
    if "rollout_alpha" in output:
        losses["rollout_alpha_mean"] = output["rollout_alpha"].detach().float().mean()
    return losses


@torch.no_grad()
def evaluate_v38_policy(
    *,
    system: V38PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V38PolicyTrainerConfig,
    max_batches: int = 0,
    memory_reporter: CudaMemoryReporter | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, float]:
    system.eval()
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    event_logits_rows: list[np.ndarray] = []
    event_target_rows: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        report_mem = memory_reporter is not None and memory_reporter.should_report(batch_index)
        if report_mem:
            memory_reporter.reset_peak()
            if memory_reporter.detail:
                memory_reporter.snapshot(tag="eval_batch_start", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
        sample = prepare_v38_policy_sample(batch, conditioner=conditioner, system=system, camera_names=camera_names, device=device, dtype=dtype)
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(tag="eval_after_prepare", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
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
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=True, return_event_logits=True,
            )
            assert isinstance(pred_pack, dict)
            no_proposal = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=False,
            )
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(tag="eval_after_sample", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
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
        if report_mem:
            memory_reporter.snapshot(tag="eval_batch_end", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step, print_line=True)
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






def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


class CudaMemoryReporter:
    """Small built-in CUDA memory profiler for V38.5 training/eval.

    It records PyTorch allocator state, not an operator-level trace. It is
    cheap enough to keep disabled by default and turn on for short smoke or
    formal runs when diagnosing memory pressure.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        out_dir: Path,
        every: int = 0,
        detail: int = 0,
        sync: int = 0,
    ) -> None:
        self.device = device
        self.enabled = bool(device.type == "cuda" and int(every) > 0 and torch.cuda.is_available())
        self.every = max(int(every), 0)
        self.detail = bool(int(detail))
        self.sync = bool(int(sync))
        self.trace_path = out_dir / "v38_cuda_memory.jsonl"
        if self.enabled:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema": "clearvla-v38-cuda-memory-trace-v1", "event": "start", "variant": "v38_5_latent_dynamics_bound"}, separators=(",", ":")) + "\n")

    def should_report(self, batch_index: int) -> bool:
        return self.enabled and self.every > 0 and int(batch_index) % self.every == 0

    def reset_peak(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(
        self,
        *,
        tag: str,
        epoch: int | None = None,
        batch: int | None = None,
        global_step: int | None = None,
        phase: str = "train",
        print_line: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self.sync:
            torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        row: dict[str, Any] = {
            "schema": "clearvla-v38-cuda-memory-trace-v1",
            "variant": "v38_5_latent_dynamics_bound",
            "phase": phase,
            "tag": tag,
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "allocated_gib": _bytes_to_gib(torch.cuda.memory_allocated(self.device)),
            "reserved_gib": _bytes_to_gib(torch.cuda.memory_reserved(self.device)),
            "max_allocated_gib": _bytes_to_gib(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_gib": _bytes_to_gib(torch.cuda.max_memory_reserved(self.device)),
            "free_gib": _bytes_to_gib(free_bytes),
            "total_gib": _bytes_to_gib(total_bytes),
        }
        row["used_by_context_gib"] = row["total_gib"] - row["free_gib"]
        if extra:
            row.update(extra)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")
        if print_line:
            print(
                f"[cuda-mem] phase={phase} tag={tag} epoch={epoch} batch={batch} step={global_step} "
                f"alloc={row['allocated_gib']:.3f}GiB reserved={row['reserved_gib']:.3f}GiB "
                f"peak_alloc={row['max_allocated_gib']:.3f}GiB peak_reserved={row['max_reserved_gib']:.3f}GiB "
                f"ctx_used={row['used_by_context_gib']:.3f}/{row['total_gib']:.3f}GiB",
                flush=True,
            )
        return row


def _accumulate_metric_tensors(acc: dict[str, Tensor], losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> None:
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        detached = value.detach().float()
        acc[key] = acc.get(key, torch.zeros((), device=detached.device, dtype=torch.float32)) + detached
    if grad is not None:
        g = grad.detach().float() if torch.is_tensor(grad) else torch.tensor(float(grad))
        acc["grad"] = acc.get("grad", torch.zeros((), device=g.device, dtype=torch.float32)) + g


def _finalize_metric_tensors(acc: dict[str, Tensor], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: float((value / float(count)).detach().cpu()) for key, value in acc.items()}


def _sync_loss_row(losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> dict[str, float]:
    row = {key: float(value.detach().float().cpu()) for key, value in losses.items() if torch.is_tensor(value)}
    if grad is not None:
        row["grad"] = float(grad.detach().float().cpu()) if torch.is_tensor(grad) else float(grad)
    return row

def train_v38_policy(
    *,
    system: V38PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V38PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    memory_reporter = CudaMemoryReporter(
        device=device,
        out_dir=out_dir,
        every=int(getattr(trainer, "memory_report_every", 0)),
        detail=int(getattr(trainer, "memory_report_detail", 0)),
        sync=int(getattr(trainer, "memory_report_sync", 0)),
    )
    system.to(device=device, dtype=torch.float32)
    if memory_reporter.enabled:
        memory_reporter.snapshot(tag="after_model_to_device", phase="setup", print_line=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": system.planner.parameters(), "lr": trainer.lr},
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
        if payload.get("schema") != "clearvla-v38-policy-checkpoint-v1":
            raise ValueError("resume checkpoint is not V38 policy")
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
        history = list(payload.get("history", [])); best.update(payload.get("best", {})); restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train(); metric_sums: dict[str, Tensor] = {}; metric_count = 0
        include_future = (
            (float(trainer.rollout_dynamics_loss_weight) > 0 or float(trainer.rollout_contrast_loss_weight) > 0
             or float(trainer.future_latent_loss_weight) > 0 or float(trainer.action_effect_loss_weight) > 0)
            and epoch >= int(trainer.future_latent_loss_start_epoch)
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            use_future = include_future and (not trainer.future_latent_max_batches or batch_index <= trainer.future_latent_max_batches)
            report_mem = memory_reporter.should_report(batch_index)
            if report_mem:
                memory_reporter.reset_peak()
                if memory_reporter.detail:
                    memory_reporter.snapshot(tag="train_batch_start", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            sample = prepare_v38_policy_sample(
                batch, conditioner=conditioner, system=system, camera_names=camera_names, device=device, dtype=dtype,
                include_target_visual=use_future,
            )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_prepare", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            optimizer.zero_grad(set_to_none=True)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_zero_grad", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            with autocast_context(device, dtype):
                output = system.flow_training_forward(
                    sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"], sample["policy_action"],
                    action_state=sample["action_state"], target_visual=sample.get("target_visual"), make_counterfactuals=use_future,
                )
                losses = flow_losses(system, sample, output, trainer, enable_future_loss=use_future)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_forward_loss", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            losses["loss"].float().backward()
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_backward", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            grad = torch.nn.utils.clip_grad_norm_(system.parameters(), trainer.grad_clip)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_clip", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            optimizer.step(); schedule.step(); global_step += 1
            if report_mem:
                memory_reporter.snapshot(tag="train_after_step", epoch=epoch, batch=batch_index, global_step=global_step, print_line=True, extra={"use_future": bool(use_future)})
            _accumulate_metric_tensors(metric_sums, losses, grad=grad)
            metric_count += 1
            if trainer.log_every and batch_index % trainer.log_every == 0:
                row = _sync_loss_row(losses, grad=grad)
                print(
                    f"[v38-full-latent] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} decode={row['decoded_action']:.6f} rollout={row.get('rollout_dynamics', 0.0):.6f} "
                    f"delta={row.get('rollout_delta', 0.0):.6f} contrast={row.get('rollout_contrast', 0.0):.6f} "
                    f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} event={row['event']:.6f} grad={row['grad']:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = _finalize_metric_tensors(metric_sums, metric_count)
        val_metrics = evaluate_v38_policy(
            system=system, loader=val_loader, conditioner=conditioner, device=device, dtype=dtype,
            camera_names=camera_names, action_normalizer=action_normalizer, trainer=trainer,
            max_batches=trainer.max_val_batches, memory_reporter=memory_reporter, epoch=epoch, global_step=global_step,
        )
        score = balanced_score(val_metrics, trainer)  # type: ignore[arg-type]
        deploy_eligible = is_deploy_eligible(val_metrics, trainer)  # type: ignore[arg-type]
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with (out_dir / "v38_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
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
            "schema": "clearvla-v38-policy-checkpoint-v1", "epoch": epoch, "global_step": global_step,
            "model": system.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
            "policy_config": asdict(system.policy_config), "trainer_config": asdict(trainer),
            "action_normalizer": action_normalizer.to_dict(), "state_normalizer": state_normalizer.to_dict(),
            "context": context, "history": history, "best": best, "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v38_policy_summary.json").write_text(json.dumps(jsonable({"schema": "clearvla-v38-policy-summary-v1", "best": best, "latest": record}), indent=2), encoding="utf-8")
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V38PolicyTrainerConfig",
    "prepare_v38_policy_sample",
    "encode_target_anchor_tokens",
    "future_latent_loss",
    "action_effect_loss",
    "rollout_dynamics_loss",
    "rollout_delta_loss",
    "rollout_contrast_loss",
    "flow_losses",
    "evaluate_v38_policy",
    "CudaMemoryReporter",
    "train_v38_policy",
]
