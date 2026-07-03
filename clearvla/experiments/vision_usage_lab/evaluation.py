from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.data.normalizer import ZScoreNormalizer
from clearvla.evaluation.metrics import compute_metrics
from .model import AdaptiveSolverConfig, VisionUsageLabModel


def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return float("nan")
    if float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _subset_metrics(
    *,
    pred: np.ndarray,
    future: np.ndarray,
    prior: np.ndarray,
    past: np.ndarray,
    mask: np.ndarray,
    normalizer: ZScoreNormalizer,
) -> dict[str, float]:
    subset = compute_metrics(
        pred_norm=pred[mask],
        target_norm=future[mask],
        prior_norm=prior[mask],
        past_norm=past[mask],
        normalizer=normalizer,
    )
    return {
        "full_mse": float(subset["full_mse"]),
        "full_rmse": float(subset["full_rmse"]),
        "full_mae": float(subset["full_mae"]),
        "normalized_mae": float(subset["normalized_mae"]),
        "first_mse": float(subset["first_mse"]),
        "first_mae": float(subset["first_mae"]),
        "first4_mse": float(subset["first4_mse"]),
        "first4_mae": float(subset["first4_mae"]),
        "delta_mse": float(subset["delta_mse"]),
        "delta_mae": float(subset["delta_mae"]),
    }


@torch.no_grad()
def evaluate_vision_usage_model(
    model: VisionUsageLabModel,
    loader: DataLoader,
    *,
    device: torch.device,
    normalizer: ZScoreNormalizer,
    integration_steps: int = 4,
    adaptive_solver: AdaptiveSolverConfig | None = None,
    include_timeline: bool = False,
) -> dict[str, Any]:
    """Evaluate fixed-depth or correction-demand-routed inference.

    Counterfactual loaders can reuse this function.  Under adaptive routing the
    reported solver-step distribution reflects actual grouped workspace calls.
    """
    model.eval()
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    event_pred: list[np.ndarray] = []
    event_target: list[np.ndarray] = []
    demand_pred: list[np.ndarray] = []
    demand_target: list[np.ndarray] = []
    solver_steps: list[np.ndarray] = []
    dyn_huber: list[float] = []
    field_gate: list[np.ndarray] = []
    field_base_norm: list[np.ndarray] = []
    field_visual_norm: list[np.ndarray] = []
    field_correction_norm: list[np.ndarray] = []
    visual_delta_magnitude: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    ref_offset = 0
    dataset_refs = getattr(loader.dataset, "refs", None)
    if include_timeline and dataset_refs is None:
        raise ValueError("timeline export requires a dataset with ordered refs")
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        prepared = model.prepare_visual(batch["visual_tokens"])
        source, source_tokens = model.predict_source(batch["past"], batch["prior"])
        flow_memory = model.prepare_flow_memory(past=batch["past"], prepared_visual=prepared)
        if adaptive_solver is None:
            pred = model.integrate_prepared(
                past=batch["past"],
                prior=batch["prior"],
                prepared_visual=prepared,
                steps=integration_steps, source_trajectory=source, prepared_flow=flow_memory,
            )
        else:
            adaptive = model.integrate_adaptive_prepared(
                past=batch["past"],
                prior=batch["prior"],
                prepared_visual=prepared,
                solver=adaptive_solver, source_trajectory=source, prepared_flow=flow_memory,
            )
            pred = adaptive.prediction
            solver_steps.append(adaptive.solver_steps.cpu().numpy())
        prefix = model.fast_prefix_head(prepared.scene_tokens, source_tokens, source)
        streaming_pred = model.streaming_tail_head.rollout(batch["past"], source, prepared.scene_tokens, prefix)
        # Auxiliary dynamics and correction demand reuse the same prepared
        # visual memory.  Validation must not encode the same observation twice.
        visual_delta, _, batch_demand_score, _ = model.predict_auxiliary_prepared(
            past=batch["past"], prior=source, prepared_visual=prepared,
        )
        dyn_huber.append(float(torch.nn.functional.smooth_l1_loss(
            visual_delta,
            batch["future_visual_delta_tokens"],
            beta=0.03,
        ).cpu()))
        zeros = torch.zeros((batch["past"].shape[0],), device=device, dtype=batch["past"].dtype)
        _, _, _, probe_base, probe_visual, probe_gate, _ = model.predict_velocity_prepared(
            past=batch["past"], source=source, prepared_visual=prepared, action_state=source,
            bridge_time=zeros, noise_level=zeros, prepared_flow=flow_memory,
        )
        probe_correction = probe_gate[:, :, None] * probe_visual
        field_gate.append(probe_gate.mean(dim=1).cpu().numpy())
        field_base_norm.append(torch.linalg.vector_norm(probe_base, dim=-1).mean(dim=1).cpu().numpy())
        field_visual_norm.append(torch.linalg.vector_norm(probe_visual, dim=-1).mean(dim=1).cpu().numpy())
        field_correction_norm.append(torch.linalg.vector_norm(probe_correction, dim=-1).mean(dim=1).cpu().numpy())
        visual_delta_magnitude.append(prepared.delta_magnitude_by_camera.cpu().numpy())
        rows["pred"].append(pred.cpu().numpy())
        rows["source_pred"].append(source.cpu().numpy())
        rows["streaming_pred"].append(streaming_pred.cpu().numpy())
        rows["prefix_pred"].append(prefix.cpu().numpy())
        rows["future"].append(batch["future"].cpu().numpy())
        rows["prior"].append(batch["prior"].cpu().numpy())
        rows["past"].append(batch["past"].cpu().numpy())
        demand_pred.append(batch_demand_score.cpu().numpy())
        if "demand_target" in batch:
            demand_target.append(batch["demand_target"].cpu().numpy())
        if "event_flag" in batch:
            event_pred.append(torch.sigmoid(prepared.event_logit).cpu().numpy())
            event_target.append(batch["event_flag"].cpu().numpy())
        if include_timeline:
            batch_size = int(batch["past"].shape[0])
            refs = dataset_refs[ref_offset:ref_offset + batch_size]
            if len(refs) != batch_size:
                raise RuntimeError("timeline refs are not aligned with evaluation batches")
            score_np = batch_demand_score.cpu().numpy()
            target_np = None if "demand_target" not in batch else batch["demand_target"].cpu().numpy()
            event_np = None if "event_flag" not in batch else batch["event_flag"].cpu().numpy()
            step_np = None
            if adaptive_solver is not None:
                step_np = adaptive.solver_steps.cpu().numpy()
            gate_np = field_gate[-1]
            base_norm_np = field_base_norm[-1]
            visual_norm_np = field_visual_norm[-1]
            correction_norm_np = field_correction_norm[-1]
            delta_magnitude_np = visual_delta_magnitude[-1]
            for local_index, ref in enumerate(refs):
                timeline.append({
                    "episode_idx": int(ref.episode_idx),
                    "center": int(ref.center),
                    "demand_score": float(score_np[local_index]),
                    "demand_target": None if target_np is None else float(target_np[local_index]),
                    "event_flag": None if event_np is None else bool(event_np[local_index] >= 0.5),
                    "solver_steps": int(integration_steps) if step_np is None else int(step_np[local_index]),
                    "fast_prefix_first_action_l2_norm": float(np.linalg.norm(prefix.cpu().numpy()[local_index, 0])),
                    "visual_gate_mean": float(gate_np[local_index]),
                    "history_base_velocity_norm": float(base_norm_np[local_index]),
                    "visual_velocity_norm": float(visual_norm_np[local_index]),
                    "applied_visual_correction_norm": float(correction_norm_np[local_index]),
                    "visual_delta_magnitude_by_camera": [float(x) for x in delta_magnitude_np[local_index]],
                })
            ref_offset += batch_size
    joined = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    metrics = compute_metrics(
        pred_norm=joined["pred"],
        target_norm=joined["future"],
        prior_norm=joined["prior"],
        past_norm=joined["past"],
        normalizer=normalizer,
    )
    metrics["inference_mode"] = "adaptive" if adaptive_solver is not None else "fixed"
    metrics["fixed_integration_steps"] = None if adaptive_solver is not None else int(integration_steps)
    metrics["latent_dynamics_huber"] = float(np.mean(dyn_huber)) if dyn_huber else float("nan")
    if field_gate:
        gate = np.concatenate(field_gate)
        base_norm = np.concatenate(field_base_norm)
        visual_norm = np.concatenate(field_visual_norm)
        correction_norm = np.concatenate(field_correction_norm)
        metrics["visual_gate_mean"] = float(gate.mean())
        metrics["visual_gate_std"] = float(gate.std())
        metrics["history_base_velocity_norm_mean"] = float(base_norm.mean())
        metrics["visual_velocity_norm_mean"] = float(visual_norm.mean())
        metrics["applied_visual_correction_norm_mean"] = float(correction_norm.mean())
    if visual_delta_magnitude:
        delta_magnitude = np.concatenate(visual_delta_magnitude, axis=0)
        metrics["visual_delta_magnitude_mean"] = float(delta_magnitude.mean())
        metrics["visual_delta_magnitude_by_camera_mean"] = [float(x) for x in delta_magnitude.mean(axis=0)]
    source_metrics = compute_metrics(
        pred_norm=joined["source_pred"], target_norm=joined["future"], prior_norm=joined["prior"],
        past_norm=joined["past"], normalizer=normalizer,
    )
    streaming_metrics = compute_metrics(
        pred_norm=joined["streaming_pred"], target_norm=joined["future"], prior_norm=joined["prior"],
        past_norm=joined["past"], normalizer=normalizer,
    )
    stage_keys = (
        "full_mse", "full_rmse", "full_mae", "normalized_mae",
        "first_mse", "first_rmse", "first_mae",
        "first4_mse", "first4_rmse", "first4_mae",
        "delta_mse", "delta_rmse", "delta_mae",
        "relative_mse_improvement_vs_prior", "per_dim_rmse", "per_dim_mae",
    )
    for key in stage_keys:
        if key in source_metrics:
            metrics[f"source_{key}"] = source_metrics[key]
        if key in streaming_metrics:
            metrics[f"streaming_{key}"] = streaming_metrics[key]
    prefix_len = joined["prefix_pred"].shape[1]
    prefix_target = joined["future"][:, :prefix_len]
    prefix_metrics = compute_metrics(
        pred_norm=joined["prefix_pred"],
        target_norm=prefix_target,
        prior_norm=joined["prior"][:, :prefix_len],
        past_norm=joined["past"],
        normalizer=normalizer,
    )
    metrics["fast_prefix_mse_norm"] = float(np.mean(np.square(joined["prefix_pred"] - prefix_target)))
    metrics["fast_prefix_mse"] = float(prefix_metrics["full_mse"])
    metrics["fast_prefix_rmse"] = float(prefix_metrics["full_rmse"])
    metrics["fast_prefix_mae"] = float(prefix_metrics["full_mae"])
    metrics["fast_prefix_normalized_mae"] = float(prefix_metrics["normalized_mae"])
    metrics["fast_prefix_per_dim_rmse"] = prefix_metrics["per_dim_rmse"]
    metrics["fast_prefix_per_dim_mae"] = prefix_metrics["per_dim_mae"]

    demand = np.concatenate(demand_pred) if demand_pred else np.empty((0,), dtype=np.float32)
    if demand.size:
        metrics["demand_score_mean"] = float(demand.mean())
        metrics["demand_score_std"] = float(demand.std())
        metrics["demand_score_min"] = float(demand.min())
        metrics["demand_score_max"] = float(demand.max())
    if demand_target:
        demand_y = np.concatenate(demand_target)
        metrics["demand_target_mean"] = float(demand_y.mean())
        metrics["demand_mae"] = float(np.mean(np.abs(demand - demand_y)))
        metrics["demand_target_corr"] = _safe_corrcoef(demand, demand_y)

    if solver_steps:
        steps = np.concatenate(solver_steps).astype(np.int64)
        metrics["mean_solver_steps"] = float(steps.mean())
        metrics["solver_steps_min"] = int(steps.min())
        metrics["solver_steps_max"] = int(steps.max())
        for value in sorted(set(int(x) for x in steps.tolist())):
            metrics[f"solver_steps_{value}_ratio"] = float(np.mean(steps == value))

    if event_target:
        y = np.concatenate(event_target)
        score = np.concatenate(event_pred)
        metrics["event_score_mean"] = float(score.mean())
        metrics["event_target_rate"] = float(y.mean())
        for label, mask in (("event", y >= 0.5), ("regular", y < 0.5)):
            if bool(mask.any()):
                subset = _subset_metrics(
                    pred=joined["pred"], future=joined["future"], prior=joined["prior"], past=joined["past"],
                    mask=mask, normalizer=normalizer,
                )
                for key, value in subset.items():
                    metrics[f"{label}_{key}"] = value
                if demand.size:
                    metrics[f"{label}_demand_score_mean"] = float(demand[mask].mean())
                if demand_target:
                    metrics[f"{label}_demand_target_mean"] = float(demand_y[mask].mean())
                if solver_steps:
                    metrics[f"{label}_mean_solver_steps"] = float(steps[mask].mean())
    if include_timeline:
        metrics["timeline"] = timeline
    return metrics


def visual_dependency_report(metrics_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "correct" not in metrics_by_mode:
        raise ValueError("metrics_by_mode must include correct")
    correct_metrics = metrics_by_mode["correct"]
    correct = float(correct_metrics["full_mse"])
    report: dict[str, Any] = {"correct_full_mse": correct}
    for mode in ("zero", "same_episode_shift", "cross_episode"):
        if mode in metrics_by_mode:
            mode_metrics = metrics_by_mode[mode]
            value = float(mode_metrics["full_mse"])
            report[f"{mode}_full_mse"] = value
            report[f"{mode}_gap"] = value - correct
            for subset in ("event", "regular"):
                key = f"{subset}_full_mse"
                if key in mode_metrics and key in correct_metrics:
                    report[f"{mode}_{subset}_gap"] = float(mode_metrics[key]) - float(correct_metrics[key])
            if "demand_score_mean" in mode_metrics and "demand_score_mean" in correct_metrics:
                report[f"{mode}_demand_shift"] = float(mode_metrics["demand_score_mean"]) - float(correct_metrics["demand_score_mean"])
            if "mean_solver_steps" in mode_metrics and "mean_solver_steps" in correct_metrics:
                report[f"{mode}_solver_steps_shift"] = float(mode_metrics["mean_solver_steps"]) - float(correct_metrics["mean_solver_steps"])
            for field_key in ("visual_gate_mean", "applied_visual_correction_norm_mean"):
                if field_key in mode_metrics and field_key in correct_metrics:
                    report[f"{mode}_{field_key}_shift"] = float(mode_metrics[field_key]) - float(correct_metrics[field_key])
    return report
