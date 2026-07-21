from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.data.normalizer import ZScoreNormalizer
from clearvla.evaluation.metrics import compute_metrics
from clearvla.training.numerics import (
    assert_finite_batch,
    assert_finite_gradients,
    assert_finite_optimizer_state,
    assert_finite_parameters,
    assert_finite_tensor,
    save_nan_debug_bundle,
)
from .evaluation import evaluate_residual_flow_model, visual_dependency_report
from .flow import ResidualBridgeConfig, sample_residual_bridge
from .losses import ResidualFlowLossConfig, residual_flow_loss, source_pretrain_loss
from .model import ResidualFlowLabModel, ResidualFlowLabOutput


LAB_PHASES = ("source_pretrain", "residual_flow")


@dataclass(frozen=True)
class LabPhaseEpochs:
    source_pretrain: int = 4
    residual_flow: int = 12

    def ordered(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, int(getattr(self, name))) for name in LAB_PHASES)

    def validate(self) -> None:
        if any(value < 0 for _, value in self.ordered()):
            raise ValueError("phase epochs must be non-negative")
        if sum(value for _, value in self.ordered()) <= 0:
            raise ValueError("at least one lab phase epoch is required")


@dataclass(frozen=True)
class ResidualFlowTrainerConfig:
    phase_epochs: LabPhaseEpochs = LabPhaseEpochs()
    source_lr: float = 1e-4
    flow_lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 50
    rolling_window: int = 100
    integration_steps: int = 4
    parameter_finite_check_every: int = 50
    optimizer_state_finite_check_every: int = 50

    def validate(self) -> None:
        self.phase_epochs.validate()
        if self.source_lr <= 0 or self.flow_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if (
            min(
                self.log_every,
                self.rolling_window,
                self.integration_steps,
                self.parameter_finite_check_every,
                self.optimizer_state_finite_check_every,
            )
            <= 0
        ):
            raise ValueError("logging, integration and finite-check intervals must be positive")


def _set_phase(model: ResidualFlowLabModel, phase: str) -> None:
    """Freeze the history source during residual-flow training by construction."""
    if phase not in LAB_PHASES:
        raise ValueError(phase)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if phase == "source_pretrain":
        enabled = (model.history_source,)
    else:
        enabled = (
            model.visual_adaptor,
            model.scene_encoder,
            model.history_encoder,
            model.source_proj,
            model.condition,
            model.residual_in,
            model.blocks,
            model.decoder,
        )
    for module in enabled:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if phase == "residual_flow":
        model.source_pos.requires_grad_(True)
        model.residual_pos.requires_grad_(True)


def _optimizer(
    model: ResidualFlowLabModel, config: ResidualFlowTrainerConfig, phase: str
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"no trainable parameters for phase={phase}")
    lr = config.source_lr if phase == "source_pretrain" else config.flow_lr
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=config.weight_decay)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total**0.5


def _module_grad_norm(module: torch.nn.Module) -> float:
    return _grad_norm([parameter for parameter in module.parameters() if parameter.requires_grad])


def _gradient_group_norms(model: ResidualFlowLabModel) -> dict[str, float]:
    return {
        "source": _module_grad_norm(model.history_source),
        "visual": _module_grad_norm(model.visual_adaptor),
        "scene": _module_grad_norm(model.scene_encoder),
        "history": _module_grad_norm(model.history_encoder),
        "workspace": _module_grad_norm(model.blocks),
        "decoder": _module_grad_norm(model.decoder),
    }


def _assert_finite_output(output: ResidualFlowLabOutput, *, prefix: str) -> None:
    for name in (
        "residual_velocity",
        "endpoint_residual",
        "endpoint_actions",
        "learned_source",
        "trajectory_tokens",
    ):
        assert_finite_tensor(getattr(output, name), name=f"{prefix}.{name}")
    for name, value in output.diagnostics.items():
        assert_finite_tensor(value, name=f"{prefix}.diagnostics.{name}")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: ResidualFlowLabModel,
    optimizer: torch.optim.Optimizer,
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    _atomic_torch_save(
        {
            "schema": "clearvla-residual-flow-lab-v1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "phase": phase,
            "phase_epoch": int(phase_epoch),
            "global_epoch": int(global_epoch),
            "global_step": int(global_step),
            "history": history,
            "context": context,
        },
        path,
    )


@torch.no_grad()
def _evaluate_source(
    model: ResidualFlowLabModel,
    loader: DataLoader,
    *,
    device: torch.device,
    normalizer: ZScoreNormalizer,
) -> dict[str, Any]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        source, _ = model.predict_source(batch["past"], batch["prior"])
        rows["source"].append(source.cpu().numpy())
        rows["future"].append(batch["future"].cpu().numpy())
        rows["prior"].append(batch["prior"].cpu().numpy())
        rows["past"].append(batch["past"].cpu().numpy())
    joined = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    return compute_metrics(
        pred_norm=joined["source"],
        target_norm=joined["future"],
        prior_norm=joined["prior"],
        past_norm=joined["past"],
        normalizer=normalizer,
    )


def train_residual_flow_lab(
    *,
    model: ResidualFlowLabModel,
    train_loader: DataLoader,
    val_loaders_by_mode: dict[str, DataLoader],
    device: torch.device,
    normalizer: ZScoreNormalizer,
    out_dir: Path,
    trainer: ResidualFlowTrainerConfig = ResidualFlowTrainerConfig(),
    bridge: ResidualBridgeConfig = ResidualBridgeConfig(),
    loss_config: ResidualFlowLossConfig = ResidualFlowLossConfig(),
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trainer.validate()
    bridge.validate()
    loss_config.validate()
    out_dir = Path(out_dir)
    checkpoints = out_dir / "checkpoints"
    debug = checkpoints / "debug"
    checkpoints.mkdir(parents=True, exist_ok=True)
    context = dict(context or {})
    history: list[dict[str, Any]] = []
    global_epoch = 0
    global_step = 0
    best_mse = float("inf")
    best_path: Path | None = None
    model.to(device)
    sampler = getattr(train_loader, "batch_sampler", None)

    for phase, epochs in trainer.phase_epochs.ordered():
        if epochs == 0:
            continue
        _set_phase(model, phase)
        optimizer = _optimizer(model, trainer, phase)
        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(global_epoch)
            model.train()
            # The corrective phase treats the learned history source as a fixed
            # anchor.  ``eval()`` also disables its training-only history noise;
            # freezing parameters alone would still produce a moving source.
            if phase == "residual_flow":
                model.history_source.eval()
            values: dict[str, list[float]] = defaultdict(list)
            rolling: deque[float] = deque(maxlen=trainer.rolling_window)
            started = time.perf_counter()
            for batch_index, raw in enumerate(train_loader, start=1):
                global_step += 1
                batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
                marker = "input"
                try:
                    assert_finite_batch(batch)
                    optimizer.zero_grad(set_to_none=True)
                    if phase == "source_pretrain":
                        marker = "source_forward"
                        source, _ = model.predict_source(batch["past"], batch["prior"])
                        loss = source_pretrain_loss(
                            source, batch["future"], huber_beta=loss_config.huber_beta
                        )
                        detached = {"source": float(loss.detach().cpu())}
                        noise_rms = 0.0
                    else:
                        marker = "residual_forward"
                        with torch.no_grad():
                            source, _ = model.predict_source(batch["past"], batch["prior"])
                        bridge_batch = sample_residual_bridge(source, batch["future"], bridge)
                        prepared = model.prepare_visual(batch["visual_tokens"])
                        flow_memory = model.prepare_flow_memory(prepared)
                        correct = model.predict_residual_velocity_prepared(
                            past=batch["past"],
                            learned_source=source,
                            prepared_visual=prepared,
                            residual_state=bridge_batch.residual_state,
                            bridge_time=bridge_batch.time,
                            step_size=bridge_batch.step_size_hint,
                            noise_level=bridge_batch.noise_level,
                            prepared_flow=flow_memory,
                        )
                        _assert_finite_output(correct, prefix="correct")
                        wrong = None
                        if loss_config.ranking_weight > 0:
                            if "negative_visual_tokens" not in batch:
                                raise ValueError("ranking requires negative_visual_tokens")
                            wrong_prepared = model.prepare_visual(batch["negative_visual_tokens"])
                            wrong = model.predict_residual_velocity_prepared(
                                past=batch["past"],
                                learned_source=source,
                                prepared_visual=wrong_prepared,
                                residual_state=bridge_batch.residual_state,
                                bridge_time=bridge_batch.time,
                                step_size=bridge_batch.step_size_hint,
                                noise_level=bridge_batch.noise_level,
                            )
                            _assert_finite_output(wrong, prefix="wrong")
                        marker = "loss"
                        result = residual_flow_loss(
                            correct=correct,
                            wrong=wrong,
                            target_actions=batch["future"],
                            target_velocity=bridge_batch.target_velocity,
                            config=loss_config,
                        )
                        loss = result.total
                        detached = result.detached_floats()
                        noise_rms = float(bridge_batch.noise_level.mean().detach().cpu())
                        for key, value in correct.diagnostics.items():
                            values[f"diag_{key}"].append(float(value.detach().cpu()))
                    assert_finite_tensor(loss, name="loss")
                    marker = "backward"
                    loss.backward()
                    named = [
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ]
                    parameters = [parameter for _, parameter in named]
                    assert_finite_gradients(named)
                    grad_groups = _gradient_group_norms(model)
                    grad_raw = _grad_norm(parameters)
                    if trainer.grad_clip > 0:
                        returned = torch.nn.utils.clip_grad_norm_(
                            parameters, trainer.grad_clip, error_if_nonfinite=True
                        )
                        grad_raw = float(returned.detach().cpu())
                    grad_clipped = _grad_norm(parameters)
                    marker = "optimizer"
                    optimizer.step()
                    if global_step % trainer.parameter_finite_check_every == 0:
                        assert_finite_parameters(named)
                    if global_step % trainer.optimizer_state_finite_check_every == 0:
                        assert_finite_optimizer_state(optimizer)
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    save_nan_debug_bundle(
                        debug / "nan_debug_batch.pt",
                        batch=batch,
                        epoch=global_epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        phase=f"{phase}:{marker}",
                        error=exc,
                    )
                    _save_checkpoint(
                        checkpoints / "failure.pt",
                        model=model,
                        optimizer=optimizer,
                        phase=phase,
                        phase_epoch=phase_epoch,
                        global_epoch=global_epoch,
                        global_step=global_step,
                        history=history,
                        context=context,
                    )
                    raise
                total = float(loss.detach().cpu())
                rolling.append(total)
                values["loss"].append(total)
                values["noise_rms"].append(noise_rms)
                values["grad_raw"].append(grad_raw)
                values["grad_clipped"].append(grad_clipped)
                values["clip_ratio"].append(grad_clipped / max(grad_raw, 1e-12))
                for key, value in grad_groups.items():
                    values[f"grad_{key}_raw"].append(value)
                for key, value in detached.items():
                    values[f"component_{key}"].append(value)
                if batch_index % trainer.log_every == 0:
                    if phase == "source_pretrain":
                        detail = f"source={detached['source']:.6f}"
                    else:
                        detail = (
                            f"flow={detached['flow']:.6f} action={detached['action']:.6f} "
                            f"rank={detached['ranking']:.6f} wrong_action={detached['wrong_action']:.6f} "
                            f"noise={noise_rms:.5f}"
                        )
                    print(
                        f"\r[residual-flow] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} "
                        f"batch={batch_index:04d} loss={total:.6f} rolling{trainer.rolling_window}={np.mean(rolling):.6f} "
                        f"{detail} grad={grad_raw:.3e}->{grad_clipped:.3e} clip={values['clip_ratio'][-1]:.3f} ",
                        end="",
                        flush=True,
                    )
            print(flush=True)
            record: dict[str, Any] = {
                "phase": phase,
                "phase_epoch": phase_epoch,
                "global_epoch": global_epoch,
                "global_step": global_step,
                "seconds": time.perf_counter() - started,
            }
            for key, items in values.items():
                record[f"train_{key}"] = float(np.mean(items)) if items else 0.0
            if phase == "source_pretrain":
                val = _evaluate_source(
                    model, val_loaders_by_mode["correct"], device=device, normalizer=normalizer
                )
                record["val"] = {"correct": val}
                print(
                    f"[residual-flow] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} sec={record['seconds']:.2f} "
                    f"train={record['train_loss']:.6f} val_source_mse={float(val['full_mse']):.6f} "
                    f"val_source_nmae={float(val['normalized_mae']):.6f}",
                    flush=True,
                )
            else:
                val_modes = {
                    mode: evaluate_residual_flow_model(
                        model,
                        loader,
                        device=device,
                        normalizer=normalizer,
                        integration_steps=trainer.integration_steps,
                    )
                    for mode, loader in val_loaders_by_mode.items()
                }
                dependency = visual_dependency_report(val_modes)
                record["val"] = val_modes
                record["dependency"] = dependency
                correct = val_modes["correct"]
                correct_mse = float(correct["full_mse"])
                print(
                    f"[residual-flow] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} sec={record['seconds']:.2f} "
                    f"train={record['train_loss']:.6f} source={float(correct['source_full_mse']):.6f} "
                    f"final={correct_mse:.6f} gain={float(correct['relative_gain_vs_source']):+.3f} "
                    f"first_rmse={float(correct['first_rmse']):.6f} first4_rmse={float(correct['first4_rmse']):.6f} "
                    f"arm_first_rmse={float(correct.get('arm_first_rmse', float('nan'))):.6f} "
                    f"arm_first4_rmse={float(correct.get('arm_first4_rmse', float('nan'))):.6f} "
                    f"gripper_rmse={float(correct['gripper_full_rmse']):.6f} nmae={float(correct['normalized_mae']):.6f} "
                    f"zero_gap={dependency.get('zero_gap', float('nan')):+.6f} "
                    f"shift_gap={dependency.get('same_episode_shift_gap', float('nan')):+.6f} "
                    f"cross_gap={dependency.get('cross_episode_gap', float('nan')):+.6f} "
                    f"event_cross_gap={dependency.get('cross_episode_event_gap', float('nan')):+.6f}",
                    flush=True,
                )
                print(
                    f"[residual-flow-metrics] per_dim_rmse={np.asarray(correct['per_dim_rmse']).round(6).tolist()} "
                    f"per_dim_nrmse={np.asarray(correct['per_dim_nrmse']).round(4).tolist()} "
                    f"per_horizon_rmse={np.asarray(correct['per_horizon_rmse']).round(6).tolist()} "
                    f"arm_first_deg_if_rad={float(correct.get('arm_first_rmse_deg_if_rad', float('nan'))):.3f} "
                    f"arm_first4_deg_if_rad={float(correct.get('arm_first4_rmse_deg_if_rad', float('nan'))):.3f}",
                    flush=True,
                )
                if correct_mse < best_mse:
                    best_mse = correct_mse
                    best_path = checkpoints / "best.pt"
                    _save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        phase=phase,
                        phase_epoch=phase_epoch,
                        global_epoch=global_epoch,
                        global_step=global_step,
                        history=history + [record],
                        context=context,
                    )
            history.append(record)
            _save_checkpoint(
                checkpoints / f"checkpoint_{global_epoch:04d}_{phase}.pt",
                model=model,
                optimizer=optimizer,
                phase=phase,
                phase_epoch=phase_epoch,
                global_epoch=global_epoch,
                global_step=global_step,
                history=history,
                context=context,
            )
            _save_checkpoint(
                checkpoints / "latest.pt",
                model=model,
                optimizer=optimizer,
                phase=phase,
                phase_epoch=phase_epoch,
                global_epoch=global_epoch,
                global_step=global_step,
                history=history,
                context=context,
            )

    summary = {
        "schema": "clearvla-residual-flow-lab-summary-v1",
        "best_full_mse": None if best_path is None else best_mse,
        "best_checkpoint": None if best_path is None else str(best_path),
        "history": history,
        "trainer": asdict(trainer),
        "bridge": bridge.to_dict(),
        "loss": asdict(loss_config),
        "context": context,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "residual_flow_lab_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    export_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_path is not None:
        export_state = torch.load(best_path, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
    _atomic_torch_save(
        {
            "schema": "clearvla-residual-flow-lab-export-v1",
            "model_state_dict": export_state,
            "model_config": model.config.to_dict(),
            "summary": summary,
        },
        out_dir / "residual_flow_lab.pt",
    )
    return summary
