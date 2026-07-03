from __future__ import annotations

"""Episode-level arm motion diagnostics for V36 policy outputs.

The gripper table answers whether open/close timing is right.  This utility
answers the analogous arm question: whether predicted joint motion and optional
end-effector motion have the right direction, magnitude, phase, and first4
execution behavior.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.observed_state_lab.alicia_urdf_fk import (
    URDFFKChain, default_alicia_urdf_path, rotation_angle, vector_cosine,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v36 import prepare_v36_policy_sample, decode
from clearvla.experiments.observed_state_lab.policy_v36 import V36PolicySystem
from clearvla.experiments.observed_state_lab.world_runtime import autocast_context, jsonable


@dataclass(frozen=True)
class ArmMotionTableConfig:
    episode_idx: int
    action_offset: int = 0
    policy_horizon: int = 24
    gripper_index: int = -1
    first_k: int = 4
    inference_steps: int = 5
    motion_window: int = 8
    motion_source: str = "auto"  # auto | ee | joint
    motion_quantile: float = 0.80
    motion_min_value: float | None = None
    phase_merge_gap: int = 8
    ratio_eps: float = 1e-6
    urdf_path: str | None = None
    urdf_variant: str = "gripper_50mm"
    base_link: str = "base_link"
    end_link: str = "tool0"
    enable_fk: bool = True


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=axis)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    out = dot / np.maximum(an * bn, 1e-9)
    both_small = (an < 1e-9) & (bn < 1e-9)
    one_small = ((an < 1e-9) ^ (bn < 1e-9))
    out = np.where(both_small, 1.0, out)
    out = np.where(one_small, 0.0, out)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _safe_ratio(num: np.ndarray, den: np.ndarray, *, eps: float) -> np.ndarray:
    """Return num/den but mark near-zero target-denominator cases as NaN.

    This avoids misleading million-scale ratios during stationary periods.  The
    time accumulator skips NaNs, so ratios are averaged only where the target
    motion is meaningful.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.where(np.abs(den) >= float(eps), num / den, np.nan).astype(np.float32)


def _angle_deg_from_cos(cosine: np.ndarray) -> np.ndarray:
    cos = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    return np.degrees(np.arccos(cos)).astype(np.float32)


def _vector_progress_lateral(pred: np.ndarray, target: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Metrics for a predicted vector against a target vector.

    progress is the scalar projection of pred on target, normalized by
    ||target||, i.e. 1.0 means full progress along the target direction, 0.0
    means no forward progress, and negative means moving backward.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    t_norm = np.linalg.norm(target, axis=-1)
    p_norm = np.linalg.norm(pred, axis=-1)
    dot = np.sum(pred * target, axis=-1)
    valid = t_norm >= float(eps)
    progress = np.where(valid, dot / np.maximum(t_norm * t_norm, 1e-12), np.nan)
    proj = progress[..., None] * target
    lateral = np.linalg.norm(pred - proj, axis=-1)
    cosine = dot / np.maximum(p_norm * t_norm, 1e-12)
    cosine = np.where(valid & (p_norm >= float(eps)), cosine, np.nan)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    lateral = np.where(valid, lateral, np.nan)
    ratio = np.where(valid, p_norm / np.maximum(t_norm, 1e-12), np.nan)
    return progress.astype(np.float32), lateral.astype(np.float32), angle.astype(np.float32), ratio.astype(np.float32)


def _row_xyz(row: dict[str, Any], scope: str, role: str) -> np.ndarray | None:
    keys = [f"{scope}_{role}_ee_x_mean", f"{scope}_{role}_ee_y_mean", f"{scope}_{role}_ee_z_mean"]
    if not all(k in row for k in keys):
        return None
    vals = np.asarray([float(row[k]) for k in keys], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return None
    return vals


class _TimeAccumulator:
    def __init__(self) -> None:
        self._rows: dict[int, dict[str, float]] = {}

    def _row(self, t: int) -> dict[str, float]:
        if int(t) not in self._rows:
            self._rows[int(t)] = {"abs_t": float(int(t))}
        return self._rows[int(t)]

    def add(self, t: int, scope: str, values: dict[str, float]) -> None:
        row = self._row(t)
        row[f"{scope}_count"] = row.get(f"{scope}_count", 0.0) + 1.0
        for key, value in values.items():
            if value is None:
                continue
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                continue
            row[f"{scope}_{key}_sum"] = row.get(f"{scope}_{key}_sum", 0.0) + v
            row[f"{scope}_{key}_sumsq"] = row.get(f"{scope}_{key}_sumsq", 0.0) + v * v

    def finalize(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in sorted(self._rows):
            row = dict(self._rows[t])
            row["abs_t"] = int(t)
            for scope in ("all", "first"):
                count = float(row.get(f"{scope}_count", 0.0))
                if count <= 0:
                    continue
                row[f"{scope}_count"] = int(count)
                keys = sorted(
                    key[len(f"{scope}_") : -4]
                    for key in list(row)
                    if key.startswith(f"{scope}_") and key.endswith("_sum")
                )
                for key in keys:
                    total = float(row.pop(f"{scope}_{key}_sum"))
                    sumsq = float(row.pop(f"{scope}_{key}_sumsq"))
                    mean = total / count
                    var = max(0.0, sumsq / count - mean * mean)
                    row[f"{scope}_{key}_mean"] = mean
                    row[f"{scope}_{key}_std"] = math.sqrt(var)
            out.append(row)
        return out


def _resolve_arm_indices(action_dim: int, gripper_index: int) -> list[int]:
    gi = int(gripper_index)
    if gi < 0:
        gi = int(action_dim) + gi
    return [i for i in range(int(action_dim)) if i != gi]


def _make_fk(config: ArmMotionTableConfig) -> URDFFKChain | None:
    if not bool(config.enable_fk):
        return None
    path = Path(config.urdf_path) if config.urdf_path else default_alicia_urdf_path(config.urdf_variant)
    return URDFFKChain(path, base_link=config.base_link, end_link=config.end_link)


def _fk_sequences(fk: URDFFKChain, arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # arm shape (..., 6)
    pos = fk.positions(arm)
    rot = fk.rotations(arm)
    return pos.astype(np.float32), rot.astype(np.float32)


def collect_arm_motion_predictions_for_episode(
    *,
    system: V36PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    config: ArmMotionTableConfig,
    max_batches: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    system.eval()
    acc = _TimeAccumulator()
    window_rows: list[dict[str, Any]] = []
    fk = _make_fk(config)
    fk_meta: dict[str, Any] = {"enabled": fk is not None}
    if fk is not None:
        fk_meta.update({
            "urdf_path": str(fk.urdf_path),
            "base_link": fk.base_link,
            "end_link": fk.end_link,
            "active_joint_names": list(fk.active_joint_names),
        })

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches and batch_index > max_batches:
                break
            episode_np = batch["episode_idx"].detach().cpu().numpy().astype(np.int64)
            mask = episode_np == int(config.episode_idx)
            if not bool(mask.any()):
                continue
            sample = prepare_v36_policy_sample(
                batch, conditioner=conditioner, system=system, camera_names=camera_names,
                device=device, dtype=dtype,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(36136 + batch_index)
            noise = torch.randn(sample["policy_action"].shape, generator=generator, device=device, dtype=sample["visual"].dtype)
            with autocast_context(device, dtype):
                pred_pack = system.sample(
                    sample["visual"], sample["history_state"], sample["executed_action_history"],
                    sample["state"], steps=int(config.inference_steps), noise=noise, use_proposal=True,
                    return_event_logits=True,
                )
            assert isinstance(pred_pack, dict)
            pred_raw = decode(action_normalizer, pred_pack["action"])
            target_raw = sample["policy_action_raw"].detach().cpu().numpy()
            current_raw = sample["state_raw"].detach().cpu().numpy()
            centers = batch["center"].detach().cpu().numpy().astype(np.int64)
            sample_indices = batch["sample_index"].detach().cpu().numpy().astype(np.int64)
            arm_indices = _resolve_arm_indices(pred_raw.shape[-1], config.gripper_index)
            pred_arm = pred_raw[..., arm_indices].astype(np.float32)
            target_arm = target_raw[..., arm_indices].astype(np.float32)
            current_arm = current_raw[..., arm_indices].astype(np.float32)
            pred_boundary = np.concatenate([current_arm[:, None, :], pred_arm[:, :-1, :]], axis=1)
            target_boundary = np.concatenate([current_arm[:, None, :], target_arm[:, :-1, :]], axis=1)
            pred_delta = pred_arm - pred_boundary
            target_delta = target_arm - target_boundary
            joint_error = pred_arm - target_arm
            joint_rmse = np.sqrt(np.mean(joint_error.astype(np.float64) ** 2, axis=-1)).astype(np.float32)
            target_delta_norm = _safe_norm(target_delta, axis=-1).astype(np.float32)
            pred_delta_norm = _safe_norm(pred_delta, axis=-1).astype(np.float32)
            joint_delta_cosine = _cosine_rows(pred_delta, target_delta)
            joint_delta_norm_ratio = _safe_ratio(pred_delta_norm, target_delta_norm, eps=config.ratio_eps)
            joint_delta_error = _safe_norm(pred_delta - target_delta, axis=-1).astype(np.float32)
            joint_delta_progress, joint_delta_lateral, joint_delta_angle_deg, _ = _vector_progress_lateral(
                pred_delta, target_delta, eps=config.ratio_eps
            )

            ee_pack: dict[str, np.ndarray] = {}
            if fk is not None:
                if pred_arm.shape[-1] != fk.dof:
                    raise ValueError(f"FK chain expects {fk.dof} joints {fk.active_joint_names}, got arm dim {pred_arm.shape[-1]}")
                pred_ee, pred_R = _fk_sequences(fk, pred_arm)
                target_ee, target_R = _fk_sequences(fk, target_arm)
                current_ee, current_R = _fk_sequences(fk, current_arm)
                pred_ee_boundary = np.concatenate([current_ee[:, None, :], pred_ee[:, :-1, :]], axis=1)
                target_ee_boundary = np.concatenate([current_ee[:, None, :], target_ee[:, :-1, :]], axis=1)
                pred_ee_delta = pred_ee - pred_ee_boundary
                target_ee_delta = target_ee - target_ee_boundary
                ee_pos_error = _safe_norm(pred_ee - target_ee, axis=-1).astype(np.float32)
                target_ee_delta_norm = _safe_norm(target_ee_delta, axis=-1).astype(np.float32)
                pred_ee_delta_norm = _safe_norm(pred_ee_delta, axis=-1).astype(np.float32)
                ee_delta_cosine = _cosine_rows(pred_ee_delta, target_ee_delta)
                ee_delta_norm_ratio = _safe_ratio(pred_ee_delta_norm, target_ee_delta_norm, eps=config.ratio_eps)
                ee_delta_error = _safe_norm(pred_ee_delta - target_ee_delta, axis=-1).astype(np.float32)
                ee_delta_progress, ee_delta_lateral, ee_delta_angle_deg, _ = _vector_progress_lateral(
                    pred_ee_delta, target_ee_delta, eps=config.ratio_eps
                )
                pred_ee_disp = pred_ee - current_ee[:, None, :]
                target_ee_disp = target_ee - current_ee[:, None, :]
                target_ee_disp_norm = _safe_norm(target_ee_disp, axis=-1).astype(np.float32)
                pred_ee_disp_norm = _safe_norm(pred_ee_disp, axis=-1).astype(np.float32)
                ee_disp_error = _safe_norm(pred_ee_disp - target_ee_disp, axis=-1).astype(np.float32)
                ee_disp_cosine = _cosine_rows(pred_ee_disp, target_ee_disp)
                ee_disp_progress, ee_disp_lateral, ee_disp_angle_deg, ee_disp_norm_ratio = _vector_progress_lateral(
                    pred_ee_disp, target_ee_disp, eps=config.ratio_eps
                )
                # Rotation error per horizon.
                flat_pred_R = pred_R.reshape(-1, 3, 3)
                flat_target_R = target_R.reshape(-1, 3, 3)
                rot_err = np.asarray([
                    rotation_angle(tR.T @ pR) for pR, tR in zip(flat_pred_R, flat_target_R)
                ], dtype=np.float32).reshape(pred_R.shape[:-2])
                ee_pack = {
                    "pred_ee": pred_ee, "target_ee": target_ee,
                    "pred_ee_delta": pred_ee_delta, "target_ee_delta": target_ee_delta,
                    "pred_ee_disp": pred_ee_disp, "target_ee_disp": target_ee_disp,
                    "ee_pos_error": ee_pos_error,
                    "ee_error_xyz": pred_ee - target_ee,
                    "target_ee_delta_norm": target_ee_delta_norm,
                    "pred_ee_delta_norm": pred_ee_delta_norm,
                    "ee_delta_cosine": ee_delta_cosine,
                    "ee_delta_norm_ratio": ee_delta_norm_ratio,
                    "ee_delta_error": ee_delta_error,
                    "ee_delta_progress": ee_delta_progress,
                    "ee_delta_lateral_error": ee_delta_lateral,
                    "ee_delta_angle_deg": ee_delta_angle_deg,
                    "target_ee_disp_norm": target_ee_disp_norm,
                    "pred_ee_disp_norm": pred_ee_disp_norm,
                    "ee_disp_error": ee_disp_error,
                    "ee_disp_cosine": ee_disp_cosine,
                    "ee_disp_norm_ratio": ee_disp_norm_ratio,
                    "ee_disp_progress": ee_disp_progress,
                    "ee_disp_lateral_error": ee_disp_lateral,
                    "ee_disp_angle_deg": ee_disp_angle_deg,
                    "ee_rot_error_rad": rot_err,
                }

            for row_i in np.flatnonzero(mask):
                center = int(centers[row_i])
                window_record: dict[str, Any] = {
                    "sample_index": int(sample_indices[row_i]),
                    "episode_idx": int(episode_np[row_i]),
                    "center": center,
                    "arm_indices": [int(x) for x in arm_indices],
                    "target_joint_delta_norm": [float(x) for x in target_delta_norm[row_i].tolist()],
                    "pred_joint_delta_norm": [float(x) for x in pred_delta_norm[row_i].tolist()],
                    "joint_delta_cosine": [float(x) for x in joint_delta_cosine[row_i].tolist()],
                    "joint_delta_norm_ratio": [float(x) for x in joint_delta_norm_ratio[row_i].tolist()],
                    "joint_delta_error": [float(x) for x in joint_delta_error[row_i].tolist()],
                    "joint_delta_progress": [float(x) for x in joint_delta_progress[row_i].tolist()],
                    "joint_delta_lateral_error": [float(x) for x in joint_delta_lateral[row_i].tolist()],
                    "joint_delta_angle_deg": [float(x) for x in joint_delta_angle_deg[row_i].tolist()],
                    "joint_rmse": [float(x) for x in joint_rmse[row_i].tolist()],
                }
                if fk is not None:
                    window_record.update({
                        "target_ee": [[float(v) for v in xyz] for xyz in ee_pack["target_ee"][row_i].tolist()],
                        "pred_ee": [[float(v) for v in xyz] for xyz in ee_pack["pred_ee"][row_i].tolist()],
                        "target_ee_delta_norm": [float(x) for x in ee_pack["target_ee_delta_norm"][row_i].tolist()],
                        "pred_ee_delta_norm": [float(x) for x in ee_pack["pred_ee_delta_norm"][row_i].tolist()],
                        "ee_delta_cosine": [float(x) for x in ee_pack["ee_delta_cosine"][row_i].tolist()],
                        "ee_delta_norm_ratio": [float(x) for x in ee_pack["ee_delta_norm_ratio"][row_i].tolist()],
                        "ee_delta_error": [float(x) for x in ee_pack["ee_delta_error"][row_i].tolist()],
                        "ee_delta_progress": [float(x) for x in ee_pack["ee_delta_progress"][row_i].tolist()],
                        "ee_delta_lateral_error": [float(x) for x in ee_pack["ee_delta_lateral_error"][row_i].tolist()],
                        "ee_delta_angle_deg": [float(x) for x in ee_pack["ee_delta_angle_deg"][row_i].tolist()],
                        "target_ee_disp_norm": [float(x) for x in ee_pack["target_ee_disp_norm"][row_i].tolist()],
                        "pred_ee_disp_norm": [float(x) for x in ee_pack["pred_ee_disp_norm"][row_i].tolist()],
                        "ee_disp_error": [float(x) for x in ee_pack["ee_disp_error"][row_i].tolist()],
                        "ee_disp_cosine": [float(x) for x in ee_pack["ee_disp_cosine"][row_i].tolist()],
                        "ee_disp_norm_ratio": [float(x) for x in ee_pack["ee_disp_norm_ratio"][row_i].tolist()],
                        "ee_disp_progress": [float(x) for x in ee_pack["ee_disp_progress"][row_i].tolist()],
                        "ee_disp_lateral_error": [float(x) for x in ee_pack["ee_disp_lateral_error"][row_i].tolist()],
                        "ee_disp_angle_deg": [float(x) for x in ee_pack["ee_disp_angle_deg"][row_i].tolist()],
                        "ee_pos_error": [float(x) for x in ee_pack["ee_pos_error"][row_i].tolist()],
                        "ee_rot_error_rad": [float(x) for x in ee_pack["ee_rot_error_rad"][row_i].tolist()],
                    })
                window_rows.append(window_record)
                for h in range(min(config.policy_horizon, pred_arm.shape[1])):
                    abs_t = center + int(config.action_offset) + h
                    values: dict[str, float] = {
                        "joint_rmse": float(joint_rmse[row_i, h]),
                        "joint_max_abs_error": float(np.max(np.abs(joint_error[row_i, h]))),
                        "target_joint_delta_norm": float(target_delta_norm[row_i, h]),
                        "pred_joint_delta_norm": float(pred_delta_norm[row_i, h]),
                        "joint_delta_cosine": float(joint_delta_cosine[row_i, h]),
                        "joint_delta_norm_ratio": float(joint_delta_norm_ratio[row_i, h]),
                        "joint_delta_error": float(joint_delta_error[row_i, h]),
                        "joint_delta_progress": float(joint_delta_progress[row_i, h]),
                        "joint_delta_lateral_error": float(joint_delta_lateral[row_i, h]),
                        "joint_delta_angle_deg": float(joint_delta_angle_deg[row_i, h]),
                    }
                    for j, src_idx in enumerate(arm_indices):
                        values[f"target_j{src_idx}"] = float(target_arm[row_i, h, j])
                        values[f"pred_j{src_idx}"] = float(pred_arm[row_i, h, j])
                        values[f"target_delta_j{src_idx}"] = float(target_delta[row_i, h, j])
                        values[f"pred_delta_j{src_idx}"] = float(pred_delta[row_i, h, j])
                    if fk is not None:
                        te = ee_pack["target_ee"][row_i, h]
                        pe = ee_pack["pred_ee"][row_i, h]
                        values.update({
                            "target_ee_x": float(te[0]), "target_ee_y": float(te[1]), "target_ee_z": float(te[2]),
                            "pred_ee_x": float(pe[0]), "pred_ee_y": float(pe[1]), "pred_ee_z": float(pe[2]),
                            "ee_error_x": float(ee_pack["ee_error_xyz"][row_i, h, 0]),
                            "ee_error_y": float(ee_pack["ee_error_xyz"][row_i, h, 1]),
                            "ee_error_z": float(ee_pack["ee_error_xyz"][row_i, h, 2]),
                            "ee_pos_error": float(ee_pack["ee_pos_error"][row_i, h]),
                            "target_ee_delta_norm": float(ee_pack["target_ee_delta_norm"][row_i, h]),
                            "pred_ee_delta_norm": float(ee_pack["pred_ee_delta_norm"][row_i, h]),
                            "ee_delta_cosine": float(ee_pack["ee_delta_cosine"][row_i, h]),
                            "ee_delta_norm_ratio": float(ee_pack["ee_delta_norm_ratio"][row_i, h]),
                            "ee_delta_error": float(ee_pack["ee_delta_error"][row_i, h]),
                            "ee_delta_progress": float(ee_pack["ee_delta_progress"][row_i, h]),
                            "ee_delta_lateral_error": float(ee_pack["ee_delta_lateral_error"][row_i, h]),
                            "ee_delta_angle_deg": float(ee_pack["ee_delta_angle_deg"][row_i, h]),
                            "target_ee_disp_norm": float(ee_pack["target_ee_disp_norm"][row_i, h]),
                            "pred_ee_disp_norm": float(ee_pack["pred_ee_disp_norm"][row_i, h]),
                            "ee_disp_error": float(ee_pack["ee_disp_error"][row_i, h]),
                            "ee_disp_cosine": float(ee_pack["ee_disp_cosine"][row_i, h]),
                            "ee_disp_norm_ratio": float(ee_pack["ee_disp_norm_ratio"][row_i, h]),
                            "ee_disp_progress": float(ee_pack["ee_disp_progress"][row_i, h]),
                            "ee_disp_lateral_error": float(ee_pack["ee_disp_lateral_error"][row_i, h]),
                            "ee_disp_angle_deg": float(ee_pack["ee_disp_angle_deg"][row_i, h]),
                            "ee_rot_error_rad": float(ee_pack["ee_rot_error_rad"][row_i, h]),
                        })
                    acc.add(abs_t, "all", values)
                    if h < int(config.first_k):
                        acc.add(abs_t, "first", values)
    return window_rows, acc.finalize(), fk_meta


def _motion_value(row: dict[str, Any], source: str) -> float:
    if source == "ee":
        return float(row.get("all_target_ee_delta_norm_mean", 0.0))
    return float(row.get("all_target_joint_delta_norm_mean", 0.0))


def choose_motion_source(rows: Sequence[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    if any("all_target_ee_delta_norm_mean" in row for row in rows):
        return "ee"
    return "joint"


def find_motion_segments(
    episode_rows: Sequence[dict[str, Any]],
    *,
    source: str,
    quantile: float,
    min_value: float | None,
) -> list[dict[str, Any]]:
    rows = sorted(episode_rows, key=lambda x: int(x["abs_t"]))
    vals = np.asarray([_motion_value(r, source) for r in rows], dtype=np.float64)
    positive = vals[vals > 1e-12]
    if min_value is None:
        threshold = float(np.quantile(positive, float(quantile))) if positive.size else float("inf")
    else:
        threshold = float(min_value)
    segments: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for row in rows:
        if _motion_value(row, source) >= threshold:
            active.append(row)
        else:
            if active:
                segments.append(_motion_segment(active, source, threshold))
                active = []
    if active:
        segments.append(_motion_segment(active, source, threshold))
    return segments


def _motion_segment(rows: Sequence[dict[str, Any]], source: str, threshold: float) -> dict[str, Any]:
    peak = max(rows, key=lambda r: _motion_value(r, source))
    return {
        "motion_type": f"{source}_motion_peak",
        "event_t": int(peak["abs_t"]),
        "span_start": int(rows[0]["abs_t"]),
        "span_end": int(rows[-1]["abs_t"]),
        "span_len": int(int(rows[-1]["abs_t"]) - int(rows[0]["abs_t"]) + 1),
        "motion_source": source,
        "motion_threshold": float(threshold),
        "target_peak_value": float(_motion_value(peak, source)),
    }


def build_motion_centered_table(
    episode_rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    motion_window: int,
) -> list[dict[str, Any]]:
    by_time = {int(row["abs_t"]): row for row in episode_rows}
    grouped: dict[int, list[dict[str, Any]]] = {}
    numeric_keys: set[str] = set()
    for event in events:
        event_t = int(event["event_t"])
        for rel in range(-int(motion_window), int(motion_window) + 1):
            row = by_time.get(event_t + rel)
            if row is None:
                continue
            grouped.setdefault(rel, []).append(row)
            numeric_keys.update(k for k, v in row.items() if isinstance(v, (int, float)) and k != "abs_t")
    out: list[dict[str, Any]] = []
    for rel, rows in sorted(grouped.items()):
        record: dict[str, Any] = {"motion_type": "motion_peak", "rel_t": int(rel), "n_events": len(rows)}
        for key in sorted(numeric_keys):
            vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
            if vals:
                record[key] = float(np.mean(vals))
        out.append(record)
    return out


def _sum_range(by_time: dict[int, dict[str, Any]], start: int, end: int, key: str) -> float:
    return float(sum(float(by_time.get(t, {}).get(key, 0.0)) for t in range(int(start), int(end) + 1)))


def _peak_in_window(
    by_time: dict[int, dict[str, Any]],
    event_t: int,
    *,
    source: str,
    scope: str,
    motion_window: int,
) -> tuple[float, int | str]:
    key = f"{scope}_pred_{source}_delta_norm_mean" if source == "ee" else f"{scope}_pred_joint_delta_norm_mean"
    best = -float("inf"); best_rel: int | None = None
    for rel in range(-int(motion_window), int(motion_window) + 1):
        row = by_time.get(event_t + rel)
        if row is None or key not in row:
            continue
        val = float(row[key])
        if val > best:
            best = val; best_rel = rel
    return (float(best) if best > -float("inf") else float("nan"), best_rel if best_rel is not None else "")


def build_motion_phase_summary(
    episode_rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    source: str,
    motion_window: int,
    ratio_eps: float = 1e-6,
) -> list[dict[str, Any]]:
    by_time = {int(row["abs_t"]): row for row in episode_rows}
    out: list[dict[str, Any]] = []
    target_key = "all_target_ee_delta_norm_mean" if source == "ee" else "all_target_joint_delta_norm_mean"
    pred_key = "all_pred_ee_delta_norm_mean" if source == "ee" else "all_pred_joint_delta_norm_mean"
    first_pred_key = "first_pred_ee_delta_norm_mean" if source == "ee" else "first_pred_joint_delta_norm_mean"
    cosine_key = "all_ee_delta_cosine_mean" if source == "ee" else "all_joint_delta_cosine_mean"
    first_cosine_key = "first_ee_delta_cosine_mean" if source == "ee" else "first_joint_delta_cosine_mean"
    err_key = "all_ee_pos_error_mean" if source == "ee" else "all_joint_rmse_mean"
    first_err_key = "first_ee_pos_error_mean" if source == "ee" else "first_joint_rmse_mean"
    for event_id, event in enumerate(events):
        event_t = int(event["event_t"])
        start = int(event["span_start"]); end = int(event["span_end"])
        row = by_time.get(event_t, {})
        all_peak, all_rel = _peak_in_window(by_time, event_t, source=source, scope="all", motion_window=motion_window)
        first_peak, first_rel = _peak_in_window(by_time, event_t, source=source, scope="first", motion_window=motion_window)
        target_sum = _sum_range(by_time, start, end, target_key)
        pred_sum = _sum_range(by_time, start, end, pred_key)
        first_pred_sum = _sum_range(by_time, start, end, first_pred_key)
        record = {
            "event_id": int(event_id),
            **event,
            "target_cumulative_motion": float(target_sum),
            "all_pred_cumulative_motion": float(pred_sum),
            "first_pred_cumulative_motion": float(first_pred_sum),
            "all_cumulative_motion_ratio": float(pred_sum / target_sum) if target_sum >= float(ratio_eps) else float("nan"),
            "first_cumulative_motion_ratio": float(first_pred_sum / target_sum) if target_sum >= float(ratio_eps) else float("nan"),
            "all_pred_peak_value": all_peak,
            "all_pred_peak_rel_t": all_rel,
            "first_pred_peak_value": first_peak,
            "first_pred_peak_rel_t": first_rel,
            "all_delta_cosine_at_peak": float(row.get(cosine_key, float("nan"))),
            "first_delta_cosine_at_peak": float(row.get(first_cosine_key, float("nan"))),
            "all_error_at_peak": float(row.get(err_key, float("nan"))),
            "first_error_at_peak": float(row.get(first_err_key, float("nan"))),
        }
        if source == "ee":
            record.update(_phase_net_metrics(by_time, start, end, pred_scope="all", eps=float(ratio_eps)))
            record.update(_phase_net_metrics(by_time, start, end, pred_scope="first", eps=float(ratio_eps)))
        out.append(record)
    return out



def merge_motion_segments(events: Sequence[dict[str, Any]], *, merge_gap: int) -> list[dict[str, Any]]:
    """Merge nearby micro motion peaks into coarser motion phases."""
    ordered = sorted(events, key=lambda e: int(e["span_start"]))
    if not ordered:
        return []
    merged: list[dict[str, Any]] = []
    cur_events: list[dict[str, Any]] = [dict(ordered[0])]

    def flush() -> None:
        start = min(int(e["span_start"]) for e in cur_events)
        end = max(int(e["span_end"]) for e in cur_events)
        peak = max(cur_events, key=lambda e: float(e.get("target_peak_value", 0.0)))
        source = str(peak.get("motion_source", "unknown"))
        merged.append({
            "motion_type": f"{source}_motion_phase",
            "event_t": int(peak["event_t"]),
            "span_start": int(start),
            "span_end": int(end),
            "span_len": int(end - start + 1),
            "motion_source": source,
            "motion_threshold": float(peak.get("motion_threshold", float("nan"))),
            "target_peak_value": float(peak.get("target_peak_value", float("nan"))),
            "num_micro_segments": int(len(cur_events)),
            "micro_event_ts": [int(e["event_t"]) for e in cur_events],
        })

    for event in ordered[1:]:
        gap = int(event["span_start"]) - max(int(e["span_end"]) for e in cur_events)
        if gap <= int(merge_gap):
            cur_events.append(dict(event))
        else:
            flush()
            cur_events = [dict(event)]
    flush()
    return merged


def _phase_net_metrics(
    by_time: dict[int, dict[str, Any]],
    start: int,
    end: int,
    *,
    pred_scope: str,
    eps: float,
) -> dict[str, float]:
    """Net EE displacement metrics for a phase using episode-time coordinates."""
    start_row = by_time.get(int(start), {})
    end_row = by_time.get(int(end), {})
    target_start = _row_xyz(start_row, "all", "target")
    target_end = _row_xyz(end_row, "all", "target")
    pred_start = _row_xyz(start_row, pred_scope, "pred")
    pred_end = _row_xyz(end_row, pred_scope, "pred")
    out: dict[str, float] = {}
    if target_start is None or target_end is None or pred_start is None or pred_end is None:
        return out
    target_disp = target_end - target_start
    pred_disp = pred_end - pred_start
    err = pred_disp - target_disp
    t_norm = float(np.linalg.norm(target_disp))
    p_norm = float(np.linalg.norm(pred_disp))
    e_norm = float(np.linalg.norm(err))
    out[f"{pred_scope}_phase_target_net_disp"] = t_norm
    out[f"{pred_scope}_phase_pred_net_disp"] = p_norm
    out[f"{pred_scope}_phase_net_disp_error"] = e_norm
    if t_norm >= eps:
        progress = float(np.dot(pred_disp, target_disp) / max(t_norm * t_norm, 1e-12))
        lateral = float(np.linalg.norm(pred_disp - progress * target_disp))
        cos = float(np.dot(pred_disp, target_disp) / max(p_norm * t_norm, 1e-12)) if p_norm >= eps else float("nan")
        out[f"{pred_scope}_phase_net_progress"] = progress
        out[f"{pred_scope}_phase_net_lateral_error"] = lateral
        out[f"{pred_scope}_phase_net_angle_deg"] = float(math.degrees(math.acos(max(-1.0, min(1.0, cos))))) if math.isfinite(cos) else float("nan")
        out[f"{pred_scope}_phase_net_relative_error"] = float(e_norm / max(t_norm, 1e-12))
    return out


def _series_stats(rows: Sequence[dict[str, Any]], key: str, *, scale: float = 1.0) -> dict[str, float]:
    vals = np.asarray([float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))], dtype=np.float64)
    vals = vals[np.isfinite(vals)] * float(scale)
    if vals.size == 0:
        return {}
    return {
        f"{key}_mean": float(np.mean(vals)),
        f"{key}_median": float(np.median(vals)),
        f"{key}_p90": float(np.quantile(vals, 0.90)),
        f"{key}_p95": float(np.quantile(vals, 0.95)),
        f"{key}_max": float(np.max(vals)),
    }


def build_episode_summary(episode_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small human-readable summary table in cm/deg/mm where appropriate."""
    out: list[dict[str, Any]] = []
    for scope in ("first", "all"):
        subset = [r for r in episode_rows if int(r.get(f"{scope}_count", 0)) > 0]
        if not subset:
            continue
        rec: dict[str, Any] = {"scope": scope, "n_time_bins": len(subset)}
        for key, scale in [
            (f"{scope}_ee_pos_error_mean", 100.0),
            (f"{scope}_ee_delta_error_mean", 100.0),
            (f"{scope}_ee_disp_error_mean", 100.0),
            (f"{scope}_ee_delta_lateral_error_mean", 100.0),
            (f"{scope}_ee_disp_lateral_error_mean", 100.0),
            (f"{scope}_ee_rot_error_rad_mean", 180.0 / math.pi),
            (f"{scope}_joint_rmse_mean", 180.0 / math.pi),
            (f"{scope}_joint_delta_error_mean", 180.0 / math.pi),
            (f"{scope}_target_ee_delta_norm_mean", 1000.0),
            (f"{scope}_pred_ee_delta_norm_mean", 1000.0),
            (f"{scope}_target_ee_disp_norm_mean", 100.0),
            (f"{scope}_pred_ee_disp_norm_mean", 100.0),
            (f"{scope}_ee_disp_progress_mean", 1.0),
            (f"{scope}_ee_delta_progress_mean", 1.0),
            (f"{scope}_ee_delta_angle_deg_mean", 1.0),
            (f"{scope}_ee_disp_angle_deg_mean", 1.0),
        ]:
            stats = _series_stats(subset, key, scale=scale)
            # Rename units for the main human-facing columns while keeping the original key stem.
            for stat_key, value in stats.items():
                rec[stat_key] = value
        # Coordinate bias/absolute error components in cm.
        for axis in ("x", "y", "z"):
            key = f"{scope}_ee_error_{axis}_mean"
            vals = np.asarray([float(r[key]) for r in subset if key in r], dtype=np.float64)
            vals = vals[np.isfinite(vals)] * 100.0
            if vals.size:
                rec[f"{key}_cm_mean"] = float(np.mean(vals))
                rec[f"{key}_cm_abs_mean"] = float(np.mean(np.abs(vals)))
                rec[f"{key}_cm_p95_abs"] = float(np.quantile(np.abs(vals), 0.95))
        out.append(rec)
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ["episode_idx", "event_id", "motion_type", "event_t", "abs_t", "rel_t", "span_start", "span_end", "span_len", "n_events"]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key); seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key); seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def write_arm_motion_tables(
    *,
    out_prefix: Path,
    config: ArmMotionTableConfig,
    window_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    fk_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = choose_motion_source(episode_rows, config.motion_source)
    events = find_motion_segments(
        episode_rows, source=source, quantile=config.motion_quantile, min_value=config.motion_min_value,
    )
    centered = build_motion_centered_table(episode_rows, events, motion_window=config.motion_window)
    phase = build_motion_phase_summary(
        episode_rows, events, source=source, motion_window=config.motion_window, ratio_eps=config.ratio_eps,
    )
    merged_events = merge_motion_segments(events, merge_gap=config.phase_merge_gap)
    merged_centered = build_motion_centered_table(episode_rows, merged_events, motion_window=config.motion_window)
    merged_phase = build_motion_phase_summary(
        episode_rows, merged_events, source=source, motion_window=config.motion_window, ratio_eps=config.ratio_eps,
    )
    episode_summary = build_episode_summary(episode_rows)
    paths = {
        "episode_arm_time_csv": str(out_prefix.with_suffix(".episode_arm_time.csv")),
        "episode_summary_csv": str(out_prefix.with_suffix(".episode_summary.csv")),
        "arm_motion_centered_csv": str(out_prefix.with_suffix(".arm_motion_centered.csv")),
        "arm_phase_summary_csv": str(out_prefix.with_suffix(".arm_phase_summary.csv")),
        "arm_merged_phase_centered_csv": str(out_prefix.with_suffix(".arm_merged_phase_centered.csv")),
        "arm_merged_phase_summary_csv": str(out_prefix.with_suffix(".arm_merged_phase_summary.csv")),
        "window_arm_jsonl": str(out_prefix.with_suffix(".windows_arm.jsonl")),
        "meta_json": str(out_prefix.with_suffix(".arm_meta.json")),
    }
    write_csv(Path(paths["episode_arm_time_csv"]), episode_rows)
    write_csv(Path(paths["episode_summary_csv"]), episode_summary)
    write_csv(Path(paths["arm_motion_centered_csv"]), centered)
    write_csv(Path(paths["arm_phase_summary_csv"]), phase)
    write_csv(Path(paths["arm_merged_phase_centered_csv"]), merged_centered)
    write_csv(Path(paths["arm_merged_phase_summary_csv"]), merged_phase)
    write_jsonl(Path(paths["window_arm_jsonl"]), window_rows)
    summary = {
        "schema": "clearvla-arm-motion-table-v1",
        "episode_idx": int(config.episode_idx),
        "num_windows": len(window_rows),
        "num_time_bins": len(episode_rows),
        "num_motion_segments": len(events),
        "num_motion_phases": len(merged_events),
        "motion_source": source,
        "config": config.__dict__,
        "fk": fk_meta or {},
        "paths": paths,
    }
    Path(paths["meta_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["meta_json"]).write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
