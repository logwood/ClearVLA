from __future__ import annotations

"""Episode-level gripper event table utilities.

These utilities are intentionally analysis-only.  They do not define training
losses or deployment behavior; they turn window-level policy predictions into
human-readable tables so gripper timing failures can be inspected directly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import inspect
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.observed_state_lab.policy_runtime_v36 import (
    prepare_v36_policy_sample,
    gripper_event_labels,
    decode,
)
from clearvla.experiments.observed_state_lab.policy_v36 import V36PolicySystem
from clearvla.experiments.observed_state_lab.world_runtime import autocast_context, jsonable


EVENT_NAMES = {0: "hold", 1: "open", 2: "close"}
DIRECTION_TO_NAME = {-1: "open", 1: "close"}


@dataclass(frozen=True)
class GripperTableConfig:
    episode_idx: int
    action_offset: int = 0
    policy_horizon: int = 24
    gripper_index: int = -1
    event_threshold: float = 0.10
    tolerance: int = 2
    event_window: int = 8
    first_k: int = 4
    target_event_rate_min: float = 0.50
    gripper_close_value: float = 1.7459820890426636
    closed_threshold: float | None = None
    inference_steps: int = 5

    @property
    def effective_closed_threshold(self) -> float:
        if self.closed_threshold is not None:
            return float(self.closed_threshold)
        return 0.5 * float(self.gripper_close_value)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return (y / np.maximum(y.sum(axis=axis, keepdims=True), 1e-12)).astype(np.float32)


def event_labels_from_delta(delta: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros_like(delta, dtype=np.int64)
    labels = np.where(delta <= -float(threshold), 1, labels)
    labels = np.where(delta >= float(threshold), 2, labels)
    return labels


def signed_event_rate(row: dict[str, Any], direction: int, prefix: str) -> float:
    if direction > 0:
        return float(row.get(f"{prefix}_close_event_rate", 0.0))
    return float(row.get(f"{prefix}_open_event_rate", 0.0))


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
            row[f"{scope}_{key}_sum"] = row.get(f"{scope}_{key}_sum", 0.0) + float(value)
            row[f"{scope}_{key}_sumsq"] = row.get(f"{scope}_{key}_sumsq", 0.0) + float(
                value
            ) * float(value)

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
                    for key in row
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


def collect_window_predictions_for_episode(
    *,
    system: V36PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    config: GripperTableConfig,
    max_batches: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the policy and collect window rows plus an episode-time table."""
    system.eval()
    acc = _TimeAccumulator()
    window_rows: list[dict[str, Any]] = []
    gidx = int(config.gripper_index)
    if gidx < 0:
        # The concrete action dim is known only from data.  We resolve after reading tensors.
        resolved_gidx: int | None = None
    else:
        resolved_gidx = gidx

    sample_supports_action_state = "action_state" in inspect.signature(system.sample).parameters
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches and batch_index > max_batches:
                break
            episode_np = batch["episode_idx"].detach().cpu().numpy().astype(np.int64)
            mask = episode_np == int(config.episode_idx)
            if not bool(mask.any()):
                continue
            sample = prepare_v36_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(36036 + batch_index)
            if bool(getattr(getattr(system, "codec", None), "uses_arm_manifold", False)):
                noise = system.codec.sample_noise(
                    sample["policy_action"].shape[0],
                    generator=generator,
                    device=device,
                    dtype=sample["visual"].dtype,
                    action_state=sample["action_state"],
                )
            else:
                noise = torch.randn(
                    sample["policy_action"].shape,
                    generator=generator,
                    device=device,
                    dtype=sample["visual"].dtype,
                )
            with autocast_context(device, dtype):
                sample_kwargs = {
                    "steps": int(config.inference_steps),
                    "noise": noise,
                    "use_proposal": True,
                    "return_event_logits": True,
                }
                if sample_supports_action_state:
                    sample_kwargs["action_state"] = sample["action_state"]
                pred_pack = system.sample(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    **sample_kwargs,
                )
            assert isinstance(pred_pack, dict)
            pred_raw = decode(action_normalizer, pred_pack["action"])
            target_raw = sample["policy_action_raw"].detach().cpu().numpy()
            current_raw = sample["state_raw"].detach().cpu().numpy()
            centers = batch["center"].detach().cpu().numpy().astype(np.int64)
            sample_indices = batch["sample_index"].detach().cpu().numpy().astype(np.int64)
            logits = pred_pack["event_logits"].detach().float().cpu().numpy()
            probs = softmax_np(logits, axis=-1)
            labels = (
                gripper_event_labels(
                    target_raw=sample["policy_action_raw"],
                    current_raw=sample["state_raw"],
                    gripper_index=gidx,
                    threshold=config.event_threshold,
                )
                .detach()
                .cpu()
                .numpy()
            )
            if resolved_gidx is None:
                resolved_gidx = pred_raw.shape[-1] + gidx
            gi = int(resolved_gidx)
            pred_g = pred_raw[..., gi].astype(np.float32)
            target_g = target_raw[..., gi].astype(np.float32)
            current_g = current_raw[..., gi].astype(np.float32)
            pred_boundary = np.concatenate([current_g[:, None], pred_g[:, :-1]], axis=1)
            target_boundary = np.concatenate([current_g[:, None], target_g[:, :-1]], axis=1)
            pred_delta = pred_g - pred_boundary
            target_delta = target_g - target_boundary
            pred_label = event_labels_from_delta(pred_delta, config.event_threshold)
            closed_thr = float(config.effective_closed_threshold)

            for row_i in np.flatnonzero(mask):
                center = int(centers[row_i])
                window_row: dict[str, Any] = {
                    "sample_index": int(sample_indices[row_i]),
                    "episode_idx": int(episode_np[row_i]),
                    "center": center,
                    "current_gripper": float(current_g[row_i]),
                    "target_events": [EVENT_NAMES[int(x)] for x in labels[row_i].tolist()],
                    "pred_events": [EVENT_NAMES[int(x)] for x in pred_label[row_i].tolist()],
                    "event_head_events": [
                        EVENT_NAMES[int(x)] for x in logits[row_i].argmax(axis=-1).tolist()
                    ],
                    "target_gripper": [float(x) for x in target_g[row_i].tolist()],
                    "pred_gripper": [float(x) for x in pred_g[row_i].tolist()],
                    "target_delta": [float(x) for x in target_delta[row_i].tolist()],
                    "pred_delta": [float(x) for x in pred_delta[row_i].tolist()],
                }
                window_rows.append(window_row)
                for h in range(min(config.policy_horizon, pred_g.shape[1])):
                    abs_t = center + int(config.action_offset) + h
                    values = {
                        "target_g": float(target_g[row_i, h]),
                        "pred_g": float(pred_g[row_i, h]),
                        "target_delta": float(target_delta[row_i, h]),
                        "pred_delta": float(pred_delta[row_i, h]),
                        "target_close_event_rate": float(labels[row_i, h] == 2),
                        "target_open_event_rate": float(labels[row_i, h] == 1),
                        "pred_close_event_rate": float(pred_label[row_i, h] == 2),
                        "pred_open_event_rate": float(pred_label[row_i, h] == 1),
                        "target_closed_rate": float(target_g[row_i, h] >= closed_thr),
                        "pred_closed_rate": float(pred_g[row_i, h] >= closed_thr),
                        "event_head_close_prob": float(probs[row_i, h, 2]),
                        "event_head_open_prob": float(probs[row_i, h, 1]),
                        "event_head_hold_prob": float(probs[row_i, h, 0]),
                    }
                    acc.add(abs_t, "all", values)
                    if h < int(config.first_k):
                        acc.add(abs_t, "first", values)
    return window_rows, acc.finalize()


def _row_at(rows_by_time: dict[int, dict[str, Any]], t: int) -> dict[str, Any] | None:
    return rows_by_time.get(int(t))


def find_target_event_segments(
    episode_rows: Sequence[dict[str, Any]],
    *,
    event_rate_min: float,
) -> list[dict[str, Any]]:
    """Group adjacent target event bins into unique close/open transition segments."""
    rows = sorted(episode_rows, key=lambda x: int(x["abs_t"]))
    segments: list[dict[str, Any]] = []
    for direction, name in ((1, "close"), (-1, "open")):
        active: list[dict[str, Any]] = []
        for row in rows:
            rate = signed_event_rate(row, direction, "all_target")
            # Fallback for old row names after finalize: target_close_event_rate_mean etc.
            if rate == 0.0:
                key = (
                    "all_target_close_event_rate_mean"
                    if direction > 0
                    else "all_target_open_event_rate_mean"
                )
                rate = float(row.get(key, 0.0))
            if rate >= float(event_rate_min):
                active.append(row)
                continue
            if active:
                segments.append(_segment_from_rows(active, direction, name))
                active = []
        if active:
            segments.append(_segment_from_rows(active, direction, name))
    return sorted(segments, key=lambda x: int(x["event_t"]))


def _segment_from_rows(rows: Sequence[dict[str, Any]], direction: int, name: str) -> dict[str, Any]:
    def signed_target_delta(row: dict[str, Any]) -> float:
        return float(direction) * float(row.get("all_target_delta_mean", 0.0))

    peak = max(rows, key=signed_target_delta)
    return {
        "event_type": name,
        "direction": int(direction),
        "event_t": int(peak["abs_t"]),
        "span_start": int(rows[0]["abs_t"]),
        "span_end": int(rows[-1]["abs_t"]),
        "span_len": int(int(rows[-1]["abs_t"]) - int(rows[0]["abs_t"]) + 1),
        "target_peak_delta": float(peak.get("all_target_delta_mean", 0.0)),
        "target_event_rate_at_peak": float(
            peak.get(
                "all_target_close_event_rate_mean"
                if direction > 0
                else "all_target_open_event_rate_mean",
                0.0,
            )
        ),
    }


def build_event_centered_table(
    episode_rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    event_window: int,
) -> list[dict[str, Any]]:
    by_time = {int(row["abs_t"]): row for row in episode_rows}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    numeric_keys: set[str] = set()
    for event in events:
        event_t = int(event["event_t"])
        event_type = str(event["event_type"])
        for rel in range(-int(event_window), int(event_window) + 1):
            row = _row_at(by_time, event_t + rel)
            if row is None:
                continue
            key = (event_type, rel)
            grouped.setdefault(key, []).append(row)
            numeric_keys.update(
                k for k, v in row.items() if isinstance(v, (int, float)) and k != "abs_t"
            )
    out: list[dict[str, Any]] = []
    for (event_type, rel), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        record: dict[str, Any] = {"event_type": event_type, "rel_t": rel, "n_events": len(rows)}
        for key in sorted(numeric_keys):
            vals = [
                float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))
            ]
            if vals:
                record[key] = float(np.mean(vals))
        out.append(record)
    return out


def _peak_for_event(
    by_time: dict[int, dict[str, Any]],
    event_t: int,
    direction: int,
    *,
    event_window: int,
    scope: str,
    event_threshold: float,
    tolerance: int,
) -> dict[str, Any]:
    delta_key = f"{scope}_pred_delta_mean"
    head_key = (
        f"{scope}_event_head_close_prob_mean"
        if direction > 0
        else f"{scope}_event_head_open_prob_mean"
    )
    pred_g_key = f"{scope}_pred_g_mean"
    pred_closed_key = f"{scope}_pred_closed_rate_mean"
    best_delta = -float("inf")
    best_delta_rel: int | None = None
    best_head = -float("inf")
    best_head_rel: int | None = None
    for rel in range(-int(event_window), int(event_window) + 1):
        row = by_time.get(int(event_t + rel))
        if row is None:
            continue
        if delta_key in row:
            score = float(direction) * float(row[delta_key])
            if score > best_delta:
                best_delta = score
                best_delta_rel = rel
        if head_key in row:
            score = float(row[head_key])
            if score > best_head:
                best_head = score
                best_head_rel = rel
    if best_delta == -float("inf"):
        best_delta = float("nan")
    if best_head == -float("inf"):
        best_head = float("nan")
    row_before = by_time.get(int(event_t - 1))
    row_after = by_time.get(int(event_t + 1))
    hit = bool(
        best_delta_rel is not None
        and best_delta >= float(event_threshold)
        and abs(best_delta_rel) <= int(tolerance)
    )
    early = bool(
        best_delta_rel is not None
        and best_delta >= float(event_threshold)
        and best_delta_rel < -int(tolerance)
    )
    late = bool(
        best_delta_rel is not None
        and best_delta >= float(event_threshold)
        and best_delta_rel > int(tolerance)
    )
    miss = bool(best_delta_rel is None or best_delta < float(event_threshold))
    return {
        f"{scope}_pred_peak_signed_delta": float(best_delta),
        f"{scope}_pred_peak_rel_t": best_delta_rel if best_delta_rel is not None else "",
        f"{scope}_event_head_peak_prob": float(best_head),
        f"{scope}_event_head_peak_rel_t": best_head_rel if best_head_rel is not None else "",
        f"{scope}_delta_hit_within_tol": int(hit),
        f"{scope}_delta_early": int(early),
        f"{scope}_delta_late": int(late),
        f"{scope}_delta_miss": int(miss),
        f"{scope}_pred_g_before": float(row_before.get(pred_g_key, float("nan")))
        if row_before
        else float("nan"),
        f"{scope}_pred_g_after": float(row_after.get(pred_g_key, float("nan")))
        if row_after
        else float("nan"),
        f"{scope}_pred_closed_before_rate": float(row_before.get(pred_closed_key, float("nan")))
        if row_before
        else float("nan"),
        f"{scope}_pred_closed_after_rate": float(row_after.get(pred_closed_key, float("nan")))
        if row_after
        else float("nan"),
    }


def build_per_event_table(
    episode_rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    event_window: int,
    event_threshold: float,
    tolerance: int,
) -> list[dict[str, Any]]:
    by_time = {int(row["abs_t"]): row for row in episode_rows}
    out: list[dict[str, Any]] = []
    for event_id, event in enumerate(events):
        event_t = int(event["event_t"])
        direction = int(event["direction"])
        row = by_time.get(event_t, {})
        record: dict[str, Any] = {
            "event_id": event_id,
            **event,
            "target_g_before": float(
                by_time.get(event_t - 1, {}).get("all_target_g_mean", float("nan"))
            ),
            "target_g_at": float(row.get("all_target_g_mean", float("nan"))),
            "target_g_after": float(
                by_time.get(event_t + 1, {}).get("all_target_g_mean", float("nan"))
            ),
        }
        for scope in ("all", "first"):
            record.update(
                _peak_for_event(
                    by_time,
                    event_t,
                    direction,
                    event_window=event_window,
                    scope=scope,
                    event_threshold=event_threshold,
                    tolerance=tolerance,
                )
            )
        out.append(record)
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "episode_idx",
        "event_id",
        "event_type",
        "direction",
        "event_t",
        "abs_t",
        "rel_t",
        "span_start",
        "span_end",
        "span_len",
        "n_events",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return value


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def write_gripper_tables(
    *,
    out_prefix: Path,
    config: GripperTableConfig,
    window_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    events = find_target_event_segments(episode_rows, event_rate_min=config.target_event_rate_min)
    centered = build_event_centered_table(episode_rows, events, event_window=config.event_window)
    per_event = build_per_event_table(
        episode_rows,
        events,
        event_window=config.event_window,
        event_threshold=config.event_threshold,
        tolerance=config.tolerance,
    )
    paths = {
        "episode_time_csv": str(out_prefix.with_suffix(".episode_time.csv")),
        "event_centered_csv": str(out_prefix.with_suffix(".event_centered.csv")),
        "per_event_csv": str(out_prefix.with_suffix(".per_event.csv")),
        "window_jsonl": str(out_prefix.with_suffix(".windows.jsonl")),
        "meta_json": str(out_prefix.with_suffix(".meta.json")),
    }
    write_csv(Path(paths["episode_time_csv"]), episode_rows)
    write_csv(Path(paths["event_centered_csv"]), centered)
    write_csv(Path(paths["per_event_csv"]), per_event)
    write_jsonl(Path(paths["window_jsonl"]), window_rows)
    summary = {
        "schema": "clearvla-gripper-event-table-v1",
        "episode_idx": int(config.episode_idx),
        "num_windows": len(window_rows),
        "num_time_bins": len(episode_rows),
        "num_unique_event_segments": len(events),
        "num_close_segments": sum(1 for x in events if x["event_type"] == "close"),
        "num_open_segments": sum(1 for x in events if x["event_type"] == "open"),
        "config": config.__dict__,
        "paths": paths,
    }
    Path(paths["meta_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["meta_json"]).write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
