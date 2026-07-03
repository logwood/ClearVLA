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
from clearvla.training.numerics import (
    assert_finite_batch,
    assert_finite_gradients,
    assert_finite_optimizer_state,
    assert_finite_parameters,
    assert_finite_tensor,
    save_nan_debug_bundle,
)
from .evaluation import evaluate_vision_usage_model, visual_dependency_report
from .flow import ActionBridgeConfig, sample_action_bridge
from .losses import VisionUsageLabLossConfig, vision_usage_lab_loss
from .model import AdaptiveSolverConfig, VisionUsageLabModel, VisionUsageLabOutput


LAB_PHASES = ("representation_pretrain", "action_flow")


@dataclass(frozen=True)
class LabPhaseEpochs:
    representation_pretrain: int = 4
    action_flow: int = 12

    def ordered(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, int(getattr(self, name))) for name in LAB_PHASES)

    def validate(self) -> None:
        if any(value < 0 for _, value in self.ordered()):
            raise ValueError("phase epochs must be non-negative")
        if sum(value for _, value in self.ordered()) <= 0:
            raise ValueError("at least one lab phase epoch is required")


@dataclass(frozen=True)
class VisionUsageLabTrainerConfig:
    phase_epochs: LabPhaseEpochs = LabPhaseEpochs()
    lr: float = 1e-4
    representation_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 50
    rolling_window: int = 100
    integration_steps: int = 4
    adaptive_eval: bool = True
    adaptive_solver: AdaptiveSolverConfig = AdaptiveSolverConfig()
    parameter_finite_check_every: int = 50
    optimizer_state_finite_check_every: int = 50

    def validate(self) -> None:
        self.phase_epochs.validate()
        if self.lr <= 0 or self.representation_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if min(self.log_every, self.rolling_window, self.integration_steps, self.parameter_finite_check_every,
               self.optimizer_state_finite_check_every) <= 0:
            raise ValueError("logging, integration and finite-check intervals must be positive")
        self.adaptive_solver.validate()


def _assert_finite_lab_output(output: VisionUsageLabOutput, *, prefix: str) -> None:
    for name in ("velocity", "endpoint", "visual_delta_tokens", "event_logit", "demand_logit", "demand_score", "action_tokens", "learned_source", "fast_prefix", "streaming_actions", "streaming_teacher_forced_actions", "base_velocity", "visual_velocity", "visual_gate", "scene_tokens", "dense_visual_tokens"):
        value = getattr(output, name)
        if value is not None:
            assert_finite_tensor(value, name=f"{prefix}.{name}")
    for name, value in output.diagnostics.items():
        assert_finite_tensor(value, name=f"{prefix}.diagnostics.{name}")


def _set_phase(model: VisionUsageLabModel, phase: str) -> None:
    if phase not in LAB_PHASES:
        raise ValueError(phase)
    modules = {
        "representation": (
            model.visual_adaptor,
            model.scene_encoder,
            model.history_source,
            model.dynamics_prior,
            model.dynamics_head,
            model.event_head,
            model.demand_head,
            model.fast_prefix_head,
            model.streaming_tail_head,
        ),
        "action": (
            model.action_anchor,
            model.history_field,
            model.action_in,
            model.condition,
            model.fusion_blocks,
            model.action_out_norm,
            model.visual_velocity_head,
            model.visual_gate_head,
        ),
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    enabled = ("representation",) if phase == "representation_pretrain" else ("representation", "action")
    for group in enabled:
        for module in modules[group]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if phase == "action_flow":
        model.action_pos.requires_grad_(True)


def _optimizer(model: VisionUsageLabModel, config: VisionUsageLabTrainerConfig, phase: str) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"no trainable parameters for phase={phase}")
    lr = config.representation_lr if phase == "representation_pretrain" else config.lr
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=config.weight_decay)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total ** 0.5


def _module_grad_norm(module: torch.nn.Module) -> float:
    return _grad_norm([parameter for parameter in module.parameters() if parameter.requires_grad])


def _gradient_group_norms(model: VisionUsageLabModel) -> dict[str, float]:
    return {
        "visual": _module_grad_norm(model.visual_adaptor),
        "scene": _module_grad_norm(model.scene_encoder),
        "source": _module_grad_norm(model.history_source),
        "dynamics": _module_grad_norm(model.dynamics_head),
        "demand": _module_grad_norm(model.demand_head),
        "prefix": _module_grad_norm(model.fast_prefix_head),
        "streaming": _module_grad_norm(model.streaming_tail_head),
        "base_field": _module_grad_norm(model.history_field),
        "workspace": _module_grad_norm(model.fusion_blocks),
        "visual_velocity": _module_grad_norm(model.visual_velocity_head),
        "visual_gate": _module_grad_norm(model.visual_gate_head),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: VisionUsageLabModel,
    optimizer: torch.optim.Optimizer,
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    _atomic_torch_save({
        "schema": "clearvla-vision-usage-lab-v5",
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "phase": phase,
        "phase_epoch": int(phase_epoch),
        "global_epoch": int(global_epoch),
        "global_step": int(global_step),
        "history": history,
        "context": context,
    }, path)


def _eval_modes(
    model: VisionUsageLabModel,
    loaders: dict[str, DataLoader],
    *,
    device: torch.device,
    normalizer: ZScoreNormalizer,
    integration_steps: int,
    adaptive_solver: AdaptiveSolverConfig | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    metrics = {
        mode: evaluate_vision_usage_model(
            model, loader, device=device, normalizer=normalizer, integration_steps=integration_steps,
        )
        for mode, loader in loaders.items()
    }
    adaptive_correct = None
    if adaptive_solver is not None:
        adaptive_correct = evaluate_vision_usage_model(
            model, loaders["correct"], device=device, normalizer=normalizer,
            integration_steps=integration_steps, adaptive_solver=adaptive_solver,
        )
    return metrics, visual_dependency_report(metrics), adaptive_correct


def train_vision_usage_lab(
    *,
    model: VisionUsageLabModel,
    train_loaders_by_phase: dict[str, DataLoader],
    val_loaders_by_mode: dict[str, DataLoader],
    device: torch.device,
    normalizer: ZScoreNormalizer,
    out_dir: Path,
    trainer: VisionUsageLabTrainerConfig = VisionUsageLabTrainerConfig(),
    bridge: ActionBridgeConfig = ActionBridgeConfig(),
    loss_config: VisionUsageLabLossConfig = VisionUsageLabLossConfig(),
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
    best_selection_mse = float("inf")
    best_path: Path | None = None
    model.to(device)

    for phase, epochs in trainer.phase_epochs.ordered():
        if epochs == 0:
            continue
        _set_phase(model, phase)
        optimizer = _optimizer(model, trainer, phase)
        loader = train_loaders_by_phase[phase]
        sampler = getattr(loader, "batch_sampler", None)
        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(global_epoch)
            model.train()
            values: dict[str, list[float]] = defaultdict(list)
            rolling: deque[float] = deque(maxlen=trainer.rolling_window)
            t0 = time.perf_counter()
            for batch_index, raw in enumerate(loader, start=1):
                global_step += 1
                batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
                marker = "input"
                try:
                    assert_finite_batch(batch)
                    optimizer.zero_grad(set_to_none=True)
                    marker = "forward"
                    consistency_output = None
                    if phase == "representation_pretrain":
                        bridge_batch = None
                        prepared = model.prepare_visual(batch["visual_tokens"])
                        correct = model.forward_prepared(
                            past=batch["past"], prior=batch["prior"], prepared_visual=prepared,
                            future_actions=batch["future"], compute_action=False, compute_auxiliary=True,
                        )
                        wrong = None
                        target_velocity = None
                        noise_mean = 0.0
                    else:
                        learned_source, learned_source_tokens = model.predict_source(batch["past"], batch["prior"])
                        bridge_batch = sample_action_bridge(learned_source.detach(), batch["future"], bridge)
                        prepared = model.prepare_visual(batch["visual_tokens"])
                        flow_memory = model.prepare_flow_memory(past=batch["past"], prepared_visual=prepared)
                        correct = model.forward_prepared(
                            past=batch["past"], prior=batch["prior"], prepared_visual=prepared,
                            action_state=bridge_batch.state, bridge_time=bridge_batch.time,
                            noise_level=bridge_batch.noise_level, future_actions=batch["future"],
                            source_trajectory=learned_source, source_tokens=learned_source_tokens,
                            prepared_flow=flow_memory, compute_action=True, compute_auxiliary=True,
                        )
                        if loss_config.consistency_weight > 0:
                            assert correct.velocity is not None
                            consistency_time = 0.5 * (bridge_batch.time + 1.0)
                            consistency_state = bridge_batch.state + (
                                consistency_time - bridge_batch.time
                            )[:, None, None] * correct.velocity.detach()
                            consistency_output = model.forward_prepared(
                                past=batch["past"], prior=batch["prior"], prepared_visual=prepared,
                                action_state=consistency_state, bridge_time=consistency_time,
                                noise_level=bridge_batch.noise_level, source_trajectory=learned_source,
                                source_tokens=learned_source_tokens, prepared_flow=flow_memory,
                                compute_action=True, compute_auxiliary=False,
                                compute_fast_paths=False,
                            )
                        wrong = None
                        if loss_config.ranking_weight > 0:
                            if "negative_visual_tokens" not in batch:
                                raise ValueError("action_flow requires explicit negative_visual_tokens")
                            wrong_prepared = model.prepare_visual(batch["negative_visual_tokens"])
                            wrong_flow_memory = model.prepare_flow_memory(past=batch["past"], prepared_visual=wrong_prepared)
                            wrong = model.forward_prepared(
                                past=batch["past"], prior=batch["prior"], prepared_visual=wrong_prepared,
                                action_state=bridge_batch.state, bridge_time=bridge_batch.time,
                                noise_level=bridge_batch.noise_level, source_trajectory=learned_source,
                                source_tokens=learned_source_tokens, prepared_flow=wrong_flow_memory,
                                compute_action=True, compute_auxiliary=False,
                                compute_fast_paths=False,
                            )
                        target_velocity = bridge_batch.target_velocity
                        noise_mean = float(bridge_batch.noise_level.mean().detach().cpu())
                    _assert_finite_lab_output(correct, prefix="correct")
                    if wrong is not None:
                        _assert_finite_lab_output(wrong, prefix="wrong")
                    if consistency_output is not None:
                        _assert_finite_lab_output(consistency_output, prefix="consistency")
                    marker = "loss"
                    result = vision_usage_lab_loss(
                        correct=correct,
                        wrong=wrong,
                        consistency_output=consistency_output,
                        target_actions=batch["future"],
                        target_velocity=target_velocity,
                        target_visual_delta_tokens=batch["future_visual_delta_tokens"],
                        event_flag=batch.get("event_flag"),
                        demand_target=batch.get("demand_target"),
                        config=loss_config,
                        phase=phase,
                    )
                    loss = result.total
                    assert_finite_tensor(loss, name="loss")
                    marker = "backward"
                    loss.backward()
                    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                    parameters = [parameter for _, parameter in named]
                    assert_finite_gradients(named)
                    grad_groups_raw = _gradient_group_norms(model)
                    grad_raw = _grad_norm(parameters)
                    if trainer.grad_clip > 0:
                        returned_raw = torch.nn.utils.clip_grad_norm_(
                            parameters, trainer.grad_clip, error_if_nonfinite=True,
                        )
                        grad_raw = float(returned_raw.detach().cpu())
                    grad_clipped = _grad_norm(parameters)
                    clip_ratio = grad_clipped / max(grad_raw, 1e-12)
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
                        checkpoints / "failure.pt", model=model, optimizer=optimizer, phase=phase,
                        phase_epoch=phase_epoch, global_epoch=global_epoch, global_step=global_step,
                        history=history, context=context,
                    )
                    raise
                detached = result.detached_floats()
                total = float(loss.detach().cpu())
                rolling.append(total)
                values["loss"].append(total)
                values["noise_rms"].append(noise_mean)
                values["grad_raw"].append(grad_raw)
                values["grad_clipped"].append(grad_clipped)
                values["clip_ratio"].append(clip_ratio)
                for key, value in grad_groups_raw.items():
                    values[f"grad_{key}_raw"].append(value)
                for key, value in detached.items():
                    values[f"component_{key}"].append(value)
                for key, value in correct.diagnostics.items():
                    values[f"diag_{key}"].append(float(value.detach().cpu()))
                if batch_index % trainer.log_every == 0:
                    if phase == "representation_pretrain":
                        detail = (
                            f"dyn={detached['dynamics_huber']:.6f} "
                            f"cos={detached['dynamics_cosine']:.6f} source={detached['source']:.6f} "
                            f"prefix={detached['prefix']:.6f} stream={detached['streaming']:.6f} "
                            f"event={detached['event']:.6f} demand={detached['demand']:.6f}"
                        )
                    else:
                        detail = (
                            f"flow={detached['flow']:.6f} action={detached['action']:.6f} "
                            f"source={detached['source']:.6f} prefix={detached['prefix']:.6f} stream={detached['streaming']:.6f} "
                            f"stream_tf={detached['streaming_teacher_forced']:.6f} stream_teacher={detached['streaming_teacher']:.6f} cons={detached['consistency']:.6f} dyn={detached['dynamics_huber']:.6f} demand={detached['demand']:.6f} "
                            f"rank={detached['ranking']:.6f} wrong_action={detached['wrong_action']:.6f} "
                            f"noise={noise_mean:.5f}"
                        )
                    print(
                        f"\r[vision-lab] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} "
                        f"batch={batch_index:04d} loss={total:.6f} "
                        f"rolling{trainer.rolling_window}={np.mean(rolling):.6f} "
                        f"{detail} grad={values['grad_raw'][-1]:.3e}->{values['grad_clipped'][-1]:.3e} "
                        f"clip={values['clip_ratio'][-1]:.3f} ",
                        end="", flush=True,
                    )
            print(flush=True)
            if phase == "representation_pretrain":
                val_metrics = {
                    "correct": evaluate_vision_usage_model(
                        model, val_loaders_by_mode["correct"], device=device, normalizer=normalizer,
                        integration_steps=trainer.integration_steps,
                    )
                }
                dependency: dict[str, Any] = {}
                adaptive_correct = None
            else:
                val_metrics, dependency, adaptive_correct = _eval_modes(
                    model, val_loaders_by_mode, device=device, normalizer=normalizer,
                    integration_steps=trainer.integration_steps,
                    adaptive_solver=trainer.adaptive_solver if trainer.adaptive_eval else None,
                )
            record: dict[str, Any] = {
                "phase": phase,
                "phase_epoch": phase_epoch,
                "global_epoch": global_epoch,
                "global_step": global_step,
                "seconds": time.perf_counter() - t0,
                "dependency": dependency,
                "val": val_metrics,
                "val_adaptive_correct": adaptive_correct,
            }
            for key, items in values.items():
                record[f"train_{key}"] = float(np.mean(items)) if items else 0.0
            history.append(record)
            correct_metrics = val_metrics["correct"]
            correct_mse = float(correct_metrics["full_mse"])
            if phase == "representation_pretrain":
                print(
                    f"[vision-lab] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} "
                    f"sec={record['seconds']:.2f} train={record['train_loss']:.6f} "
                    f"train_dyn={record['train_component_dynamics_huber']:.6f} "
                    f"train_cos={record['train_component_dynamics_cosine']:.6f} "
                    f"val_dyn={float(correct_metrics['latent_dynamics_huber']):.6f} "
                    f"val_demand={float(correct_metrics.get('demand_mae', float('nan'))):.6f} "
                    f"demand={float(correct_metrics.get('demand_score_mean', float('nan'))):.3f} "
                    f"val_prior={correct_mse:.6f}",
                    flush=True,
                )
            else:
                adaptive_mse = float("nan") if adaptive_correct is None else float(adaptive_correct["full_mse"])
                adaptive_steps = float("nan") if adaptive_correct is None else float(adaptive_correct["mean_solver_steps"])
                print(
                    f"[vision-lab] phase={phase} epoch={phase_epoch:03d}/{epochs:03d} "
                    f"sec={record['seconds']:.2f} train={record['train_loss']:.6f} "
                    f"val_correct={correct_mse:.6f} val_adaptive={adaptive_mse:.6f} steps={adaptive_steps:.3f} "
                    f"val_dyn={float(correct_metrics['latent_dynamics_huber']):.6f} "
                    f"demand={float(correct_metrics.get('demand_score_mean', float('nan'))):.3f} "
                    f"demand_corr={float(correct_metrics.get('demand_target_corr', float('nan'))):+.3f} "
                    f"zero_gap={dependency.get('zero_gap', float('nan')):+.6f} "
                    f"shift_gap={dependency.get('same_episode_shift_gap', float('nan')):+.6f} "
                    f"cross_gap={dependency.get('cross_episode_gap', float('nan')):+.6f} "
                    f"event_cross_gap={dependency.get('cross_episode_event_gap', float('nan')):+.6f}",
                    flush=True,
                )
            epoch_path = checkpoints / f"checkpoint_{global_epoch:04d}_{phase}.pt"
            _save_checkpoint(
                epoch_path, model=model, optimizer=optimizer, phase=phase, phase_epoch=phase_epoch,
                global_epoch=global_epoch, global_step=global_step, history=history, context=context,
            )
            _save_checkpoint(
                checkpoints / "latest.pt", model=model, optimizer=optimizer, phase=phase, phase_epoch=phase_epoch,
                global_epoch=global_epoch, global_step=global_step, history=history, context=context,
            )
            selection_mse = correct_mse if adaptive_correct is None else float(adaptive_correct["full_mse"])
            if phase == "action_flow" and selection_mse < best_selection_mse:
                best_selection_mse = selection_mse
                best_path = checkpoints / "best.pt"
                _save_checkpoint(
                    best_path, model=model, optimizer=optimizer, phase=phase, phase_epoch=phase_epoch,
                    global_epoch=global_epoch, global_step=global_step, history=history, context=context,
                )

    summary = {
        "schema": "clearvla-vision-usage-lab-summary-v5",
        "best_selection_full_mse": None if best_path is None else best_selection_mse,
        "best_selection_mode": "adaptive" if trainer.adaptive_eval else "fixed",
        "best_checkpoint": None if best_path is None else str(best_path),
        "history": history,
        "trainer": asdict(trainer),
        "bridge": asdict(bridge),
        "loss": asdict(loss_config),
        "context": context,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vision_usage_lab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_path is not None:
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        export_state_dict = best_payload["model_state_dict"]
        export_source = str(best_path)
    else:
        export_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        export_source = "final_model_without_action_flow_best"
    _atomic_torch_save({
        "schema": "clearvla-vision-usage-lab-export-v5",
        "model_state_dict": export_state_dict,
        "model_config": model.config.to_dict(),
        "export_source": export_source,
        "summary": summary,
    }, out_dir / "vision_usage_lab.pt")
    return summary
