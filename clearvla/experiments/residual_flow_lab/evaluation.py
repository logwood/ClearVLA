from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.data.normalizer import ZScoreNormalizer
from clearvla.evaluation.metrics import compute_metrics
from .model import ResidualFlowLabModel


def _subset_metrics(
    *,
    pred: np.ndarray,
    source: np.ndarray,
    future: np.ndarray,
    prior: np.ndarray,
    past: np.ndarray,
    mask: np.ndarray,
    normalizer: ZScoreNormalizer,
) -> dict[str, float]:
    final = compute_metrics(pred_norm=pred[mask], target_norm=future[mask], prior_norm=prior[mask], past_norm=past[mask], normalizer=normalizer)
    source_metrics = compute_metrics(pred_norm=source[mask], target_norm=future[mask], prior_norm=prior[mask], past_norm=past[mask], normalizer=normalizer)
    out = {
        "full_mse": float(final["full_mse"]),
        "full_rmse": float(final["full_rmse"]),
        "full_mae": float(final["full_mae"]),
        "normalized_mae": float(final["normalized_mae"]),
        "first_mse": float(final["first_mse"]),
        "first_mae": float(final["first_mae"]),
        "first4_mse": float(final["first4_mse"]),
        "first4_mae": float(final["first4_mae"]),
        "arm_full_rmse": float(final.get("arm_full_rmse", float("nan"))),
        "arm_first_rmse": float(final.get("arm_first_rmse", float("nan"))),
        "arm_first4_rmse": float(final.get("arm_first4_rmse", float("nan"))),
        "gripper_full_rmse": float(final["gripper_full_rmse"]),
        "source_full_mse": float(source_metrics["full_mse"]),
    }
    out["final_minus_source_mse"] = out["full_mse"] - out["source_full_mse"]
    out["relative_gain_vs_source"] = 1.0 - out["full_mse"] / max(out["source_full_mse"], 1e-12)
    return out


@torch.no_grad()
def evaluate_residual_flow_model(
    model: ResidualFlowLabModel,
    loader: DataLoader,
    *,
    device: torch.device,
    normalizer: ZScoreNormalizer,
    integration_steps: int = 4,
) -> dict[str, Any]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    event_target: list[np.ndarray] = []
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        source, _ = model.predict_source(batch["past"], batch["prior"])
        prepared = model.prepare_visual(batch["visual_tokens"])
        flow_memory = model.prepare_flow_memory(prepared)
        pred = model.integrate_prepared(
            past=batch["past"],
            prior=batch["prior"],
            prepared_visual=prepared,
            steps=integration_steps,
            learned_source=source,
            prepared_flow=flow_memory,
        )
        zeros = torch.zeros_like(source)
        batch_size = source.shape[0]
        probe = model.predict_residual_velocity_prepared(
            past=batch["past"],
            learned_source=source,
            prepared_visual=prepared,
            residual_state=zeros,
            bridge_time=torch.zeros((batch_size,), device=device, dtype=source.dtype),
            step_size=torch.full((batch_size,), 1.0 / float(integration_steps), device=device, dtype=source.dtype),
            noise_level=torch.zeros((batch_size,), device=device, dtype=source.dtype),
            prepared_flow=flow_memory,
        )
        for key, value in probe.diagnostics.items():
            diagnostics[key].append(float(value.detach().cpu()))
        rows["pred"].append(pred.cpu().numpy())
        rows["source"].append(source.cpu().numpy())
        rows["future"].append(batch["future"].cpu().numpy())
        rows["prior"].append(batch["prior"].cpu().numpy())
        rows["past"].append(batch["past"].cpu().numpy())
        if "event_flag" in batch:
            event_target.append(batch["event_flag"].cpu().numpy())
    joined = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    metrics = compute_metrics(
        pred_norm=joined["pred"], target_norm=joined["future"], prior_norm=joined["prior"], past_norm=joined["past"], normalizer=normalizer,
    )
    source_metrics = compute_metrics(
        pred_norm=joined["source"], target_norm=joined["future"], prior_norm=joined["prior"], past_norm=joined["past"], normalizer=normalizer,
    )
    keys = (
        "full_mse", "full_rmse", "full_mae", "normalized_mae",
        "first_mse", "first_rmse", "first_mae",
        "first4_mse", "first4_rmse", "first4_mae",
        "delta_mse", "delta_rmse", "delta_mae",
        "relative_mse_improvement_vs_prior", "per_dim_rmse", "per_dim_mae", "per_dim_nrmse",
        "per_horizon_rmse", "per_horizon_mae",
        "arm_full_rmse", "arm_first_rmse", "arm_first4_rmse", "arm_full_mae",
        "arm_full_rmse_deg_if_rad", "arm_first_rmse_deg_if_rad", "arm_first4_rmse_deg_if_rad",
        "gripper_full_rmse", "gripper_first_rmse", "gripper_first4_rmse", "gripper_full_mae",
    )
    for key in keys:
        if key in source_metrics:
            metrics[f"source_{key}"] = source_metrics[key]
    source_mse = float(source_metrics["full_mse"])
    final_mse = float(metrics["full_mse"])
    metrics["final_minus_source_mse"] = final_mse - source_mse
    metrics["relative_gain_vs_source"] = 1.0 - final_mse / max(source_mse, 1e-12)
    metrics["integration_steps"] = int(integration_steps)
    for key, values in diagnostics.items():
        metrics[f"diag_{key}"] = float(np.mean(values)) if values else float("nan")
    if event_target:
        y = np.concatenate(event_target)
        for label, mask in (("event", y >= 0.5), ("regular", y < 0.5)):
            if bool(mask.any()):
                subset = _subset_metrics(
                    pred=joined["pred"], source=joined["source"], future=joined["future"], prior=joined["prior"], past=joined["past"], mask=mask, normalizer=normalizer,
                )
                for key, value in subset.items():
                    metrics[f"{label}_{key}"] = value
    return metrics


def visual_dependency_report(metrics_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "correct" not in metrics_by_mode:
        raise ValueError("metrics_by_mode must include correct")
    correct = metrics_by_mode["correct"]
    report: dict[str, Any] = {
        "correct_full_mse": float(correct["full_mse"]),
        "correct_source_full_mse": float(correct["source_full_mse"]),
        "correct_relative_gain_vs_source": float(correct["relative_gain_vs_source"]),
    }
    for mode in ("zero", "same_episode_shift", "cross_episode"):
        if mode not in metrics_by_mode:
            continue
        item = metrics_by_mode[mode]
        report[f"{mode}_gap"] = float(item["full_mse"]) - float(correct["full_mse"])
        report[f"{mode}_relative_gain_vs_source"] = float(item["relative_gain_vs_source"])
        for subset in ("event", "regular"):
            key = f"{subset}_full_mse"
            if key in item and key in correct:
                report[f"{mode}_{subset}_gap"] = float(item[key]) - float(correct[key])
    return report
