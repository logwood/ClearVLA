from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.evaluation.metrics import compute_metrics
from .codec import RDTLiteCodecs
from .model import ObjectiveName, RDTLiteModel
from .schedule import CosineDiffusionSchedule


def _subset_metrics(
    *,
    pred: np.ndarray,
    future: np.ndarray,
    prior: np.ndarray,
    past: np.ndarray,
    mask: np.ndarray,
    codecs: RDTLiteCodecs,
) -> dict[str, float]:
    rows = compute_metrics(
        pred_norm=pred[mask],
        target_norm=future[mask],
        prior_norm=prior[mask],
        past_norm=past[mask],
        normalizer=codecs.action_normalizer,
    )
    keys = (
        "full_mse",
        "full_rmse",
        "full_mae",
        "normalized_mae",
        "first_mse",
        "first_rmse",
        "first_mae",
        "first4_mse",
        "first4_rmse",
        "first4_mae",
        "arm_full_rmse",
        "arm_first_rmse",
        "arm_first4_rmse",
        "gripper_full_rmse",
        "gripper_first_rmse",
        "gripper_first4_rmse",
    )
    return {key: float(rows[key]) for key in keys if key in rows}


@torch.no_grad()
def evaluate_rdt_lite_model(
    model: RDTLiteModel,
    loader: DataLoader,
    *,
    objective: ObjectiveName,
    device: torch.device,
    codecs: RDTLiteCodecs,
    sampling_steps: int,
    diffusion_schedule: CosineDiffusionSchedule | None = None,
    eval_seed: int = 1729,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_seed))
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    event_target: list[np.ndarray] = []
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        prepared = model.prepare_visual(batch["visual_tokens"])
        shape = (batch["state_history"].shape[0], model.config.chunk_len, model.config.action_dim)
        noise = torch.randn(
            shape, device=device, dtype=batch["state_history"].dtype, generator=generator
        )
        pred_code = model.sample_actions_prepared(
            objective=objective,
            state_history=batch["state_history"],
            prepared=prepared,
            steps=sampling_steps,
            diffusion_schedule=diffusion_schedule,
            initial_noise=noise,
        )
        if objective == "rdt_denoise":
            probe_time = torch.full(
                (shape[0],),
                float((diffusion_schedule.train_timesteps if diffusion_schedule else 1000) - 1),
                device=device,
                dtype=batch["state_history"].dtype,
            )
        else:
            probe_time = torch.ones((shape[0],), device=device, dtype=batch["state_history"].dtype)
        probe = model.forward_prepared(
            state_history=batch["state_history"],
            noisy_actions=noise,
            time=probe_time,
            prepared=prepared,
        )
        for key, value in probe.diagnostics.items():
            diagnostics[key].append(float(value.detach().cpu()))

        pred_code_np = pred_code.cpu().numpy()
        current_state_raw_np = batch["current_state_raw"].cpu().numpy()
        pred_abs_norm = codecs.decode_target_to_action_norm(pred_code_np, current_state_raw_np)
        rows["pred"].append(pred_abs_norm)
        rows["future"].append(batch["future"].cpu().numpy())
        rows["prior"].append(batch["prior"].cpu().numpy())
        rows["past"].append(batch["past"].cpu().numpy())
        if "event_flag" in batch:
            event_target.append(batch["event_flag"].cpu().numpy())

    joined = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    metrics = compute_metrics(
        pred_norm=joined["pred"],
        target_norm=joined["future"],
        prior_norm=joined["prior"],
        past_norm=joined["past"],
        normalizer=codecs.action_normalizer,
    )
    metrics["objective"] = str(objective)
    metrics["sampling_steps"] = int(sampling_steps)
    metrics["action_representation"] = codecs.action_representation
    for key, values in diagnostics.items():
        metrics[f"diag_{key}"] = float(np.mean(values)) if values else float("nan")
    if event_target:
        y = np.concatenate(event_target)
        for label, mask in (("event", y >= 0.5), ("regular", y < 0.5)):
            if bool(mask.any()):
                subset = _subset_metrics(
                    pred=joined["pred"],
                    future=joined["future"],
                    prior=joined["prior"],
                    past=joined["past"],
                    mask=mask,
                    codecs=codecs,
                )
                for key, value in subset.items():
                    metrics[f"{label}_{key}"] = value
    return metrics


def visual_dependency_report(metrics_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "correct" not in metrics_by_mode:
        raise ValueError("metrics_by_mode must include correct")
    correct = metrics_by_mode["correct"]
    report: dict[str, Any] = {"correct_full_mse": float(correct["full_mse"])}
    for mode in ("zero", "same_episode_shift", "cross_episode"):
        if mode not in metrics_by_mode:
            continue
        item = metrics_by_mode[mode]
        report[f"{mode}_gap"] = float(item["full_mse"]) - float(correct["full_mse"])
        for subset in ("event", "regular"):
            key = f"{subset}_full_mse"
            if key in item and key in correct:
                report[f"{mode}_{subset}_gap"] = float(item[key]) - float(correct[key])
    return report
