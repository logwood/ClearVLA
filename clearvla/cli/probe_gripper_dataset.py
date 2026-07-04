from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clearvla.data.hdf5_episode import find_hdf5_files
from clearvla.data.schema import ACTION_ALIASES, STATE_ALIASES, list_hdf5_datasets, resolve_key


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean", "std")}
    qs = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "min": float(np.min(x)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0)


def _summarize_counter(counter: dict[str, float]) -> dict[str, float]:
    total = float(sum(counter.values()))
    out: dict[str, float] = {"total": total}
    for key, value in sorted(counter.items()):
        out[f"{key}_count"] = float(value)
        out[f"{key}_rate"] = _safe_ratio(float(value), total)
    return out


def _summarize_window_event_counts(counter: dict[str, float], total_windows: float) -> dict[str, float]:
    out: dict[str, float] = {"total": float(total_windows)}
    for key in ("none", "any", "close", "open", "both"):
        value = float(counter.get(key, 0.0))
        out[f"{key}_count"] = value
        out[f"{key}_rate"] = _safe_ratio(value, total_windows)
    return out


def _summarize_boundary_event_direction(counter: dict[str, float]) -> dict[str, float]:
    close_keys = ("close_from_low", "close_from_mid", "close_from_high")
    open_keys = ("open_from_low", "open_from_mid", "open_from_high")
    close_total = float(sum(counter.get(key, 0.0) for key in close_keys))
    open_total = float(sum(counter.get(key, 0.0) for key in open_keys))
    out: dict[str, float] = {"close_total": close_total, "open_total": open_total}
    for key in close_keys:
        value = float(counter.get(key, 0.0))
        out[f"{key}_rate_within_close"] = _safe_ratio(value, close_total)
    for key in open_keys:
        value = float(counter.get(key, 0.0))
        out[f"{key}_rate_within_open"] = _safe_ratio(value, open_total)
    return out


def _resolve_unit(raw: np.ndarray, requested: str) -> tuple[str, float]:
    if requested == "degree":
        return "degree", 1.0
    if requested == "radian":
        return "radian", 180.0 / math.pi
    if requested == "raw":
        return "raw", 1.0
    finite = np.asarray(raw, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
    if max_abs <= 3.5:
        return "radian_auto", 180.0 / math.pi
    return "degree_auto", 1.0


def _event_labels(delta: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(delta.shape, dtype=np.int8)
    labels = np.where(delta >= float(threshold), 1, labels)
    labels = np.where(delta <= -float(threshold), -1, labels)
    return labels


def _event_runs(labels: np.ndarray, delta: np.ndarray) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    runs: list[dict[str, float]] = []
    i = 0
    while i < labels.size:
        direction = int(labels[i])
        if direction == 0:
            i += 1
            continue
        j = i + 1
        while j < labels.size and int(labels[j]) == direction:
            j += 1
        segment = delta[i:j]
        runs.append({
            "direction": float(direction),
            "start": float(i),
            "length": float(j - i),
            "signed_change": float(segment.sum()),
            "abs_change": float(np.abs(segment).sum()),
            "max_abs_step": float(np.max(np.abs(segment))) if segment.size else 0.0,
        })
        i = j
    return runs


def _load_gripper_arrays(
    path: Path,
    *,
    action_key: str,
    state_key: str | None,
    gripper_index: int,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    datasets = list_hdf5_datasets(str(path))
    resolved_action = resolve_key(datasets, action_key, ACTION_ALIASES, required=True)
    assert resolved_action is not None
    resolved_state = resolve_key(datasets, state_key, STATE_ALIASES, required=False) if state_key else None
    with h5py.File(path, "r") as f:
        actions = np.asarray(f[resolved_action], dtype=np.float32)
        states = np.asarray(f[resolved_state], dtype=np.float32) if resolved_state is not None else actions.copy()
    if actions.ndim != 2:
        raise ValueError(f"{path}: action must have [T,D], got {actions.shape}")
    if states.ndim != 2 or states.shape != actions.shape:
        raise ValueError(f"{path}: state must align with action, got state={states.shape} action={actions.shape}")
    gi = int(gripper_index)
    if gi < 0:
        gi = int(actions.shape[1]) + gi
    if gi < 0 or gi >= int(actions.shape[1]):
        raise ValueError(f"{path}: gripper index {gripper_index} resolved to {gi}, action_dim={actions.shape[1]}")
    return actions[:, gi].astype(np.float64), states[:, gi].astype(np.float64), resolved_action, resolved_state


def _segment_name(index: int, horizon: int) -> str:
    if index < 4:
        return "h00_03"
    if index < 8:
        return "h04_07"
    if index < 16:
        return "h08_15"
    return f"h16_{max(int(horizon) - 1, 16):02d}"


def probe_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root)
    files = find_hdf5_files(root, args.glob)
    if int(args.max_episodes) > 0:
        files = files[: int(args.max_episodes)]

    raw_action_values: list[np.ndarray] = []
    raw_state_values: list[np.ndarray] = []
    skipped: list[dict[str, str]] = []
    loaded: list[tuple[Path, np.ndarray, np.ndarray, str, str | None]] = []
    for path in files:
        try:
            action_g, state_g, resolved_action, resolved_state = _load_gripper_arrays(
                path,
                action_key=str(args.action_key),
                state_key=args.state_key,
                gripper_index=int(args.gripper_dim_index),
            )
            if action_g.size < int(args.min_length):
                skipped.append({"path": str(path), "reason": f"too_short={action_g.size}"})
                continue
            raw_action_values.append(action_g)
            raw_state_values.append(state_g)
            loaded.append((path, action_g, state_g, resolved_action, resolved_state))
        except Exception as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})

    if not loaded:
        raise RuntimeError(f"No usable episodes under {root}; skipped={skipped[:5]}")

    raw_all = np.concatenate([x for x in raw_action_values] + [x for x in raw_state_values])
    unit, scale = _resolve_unit(raw_all, str(args.unit))
    event_threshold = float(args.event_threshold_deg)
    hold_threshold = float(args.hold_threshold_deg)
    angle_max = float(args.angle_max_deg)
    closed_threshold = float(args.closed_threshold_deg) if args.closed_threshold_deg is not None else 0.5 * angle_max
    open_threshold = float(args.open_threshold_deg) if args.open_threshold_deg is not None else 0.1 * angle_max

    action_values: list[np.ndarray] = []
    state_values: list[np.ndarray] = []
    action_step_delta: list[np.ndarray] = []
    state_step_delta: list[np.ndarray] = []
    state_to_action_delta: list[np.ndarray] = []
    event_step_delta: list[np.ndarray] = []
    hold_step_abs_delta: list[np.ndarray] = []
    runs: list[dict[str, float]] = []
    episode_rows: list[dict[str, Any]] = []
    window_event_counts = {"none": 0.0, "any": 0.0, "open": 0.0, "close": 0.0, "both": 0.0}
    total_windows = 0.0
    first_event_hist = np.zeros(int(args.horizon), dtype=np.float64)
    segment_counts: dict[str, float] = {}
    segment_event_counts: dict[str, float] = {}
    boundary_counts = {"low": 0.0, "mid": 0.0, "high": 0.0}
    event_from_boundary = {
        "close_from_low": 0.0,
        "close_from_mid": 0.0,
        "close_from_high": 0.0,
        "open_from_low": 0.0,
        "open_from_mid": 0.0,
        "open_from_high": 0.0,
    }
    all_window_event_delta: list[np.ndarray] = []
    all_window_hold_abs_delta: list[np.ndarray] = []

    horizon = int(args.horizon)
    stride = max(int(args.stride), 1)
    for path, action_raw, state_raw, resolved_action, resolved_state in loaded:
        action_g = action_raw * scale
        state_g = state_raw * scale
        action_values.append(action_g)
        state_values.append(state_g)
        if action_g.size > 1:
            action_delta = action_g[1:] - action_g[:-1]
            state_delta = state_g[1:] - state_g[:-1]
            action_step_delta.append(action_delta)
            state_step_delta.append(state_delta)
            labels = _event_labels(action_delta, event_threshold)
            event_step_delta.append(action_delta[labels != 0])
            hold_step_abs_delta.append(np.abs(action_delta[np.abs(action_delta) < hold_threshold]))
            runs.extend(_event_runs(labels, action_delta))
        state_to_action_delta.append(action_g - state_g)

        local_labels = _event_labels(action_g[1:] - action_g[:-1], event_threshold) if action_g.size > 1 else np.zeros(0, dtype=np.int8)
        close_steps = int(np.sum(local_labels > 0))
        open_steps = int(np.sum(local_labels < 0))
        hold_steps = int(local_labels.size - close_steps - open_steps)
        episode_rows.append({
            "path": str(path),
            "length": int(action_g.size),
            "action_key": resolved_action,
            "state_key": resolved_state,
            "action_gripper": _quantiles(action_g),
            "state_gripper": _quantiles(state_g),
            "action_step_delta": _quantiles(action_g[1:] - action_g[:-1]) if action_g.size > 1 else {},
            "state_to_action_delta": _quantiles(action_g - state_g),
            "close_step_count": close_steps,
            "open_step_count": open_steps,
            "hold_step_count": hold_steps,
            "event_step_rate": _safe_ratio(close_steps + open_steps, max(local_labels.size, 1)),
        })

        max_start = int(action_g.size) - horizon
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            total_windows += 1.0
            future = action_g[start : start + horizon]
            boundary = np.concatenate([state_g[start : start + 1], future[:-1]])
            delta = future - boundary
            labels = _event_labels(delta, event_threshold)
            has_close = bool(np.any(labels > 0))
            has_open = bool(np.any(labels < 0))
            has_any = has_close or has_open
            window_event_counts["any" if has_any else "none"] += 1.0
            if has_close:
                window_event_counts["close"] += 1.0
            if has_open:
                window_event_counts["open"] += 1.0
            if has_close and has_open:
                window_event_counts["both"] += 1.0
            idx = np.flatnonzero(labels != 0)
            if idx.size:
                first_event_hist[int(idx[0])] += 1.0
            all_window_event_delta.append(delta[labels != 0])
            all_window_hold_abs_delta.append(np.abs(delta[labels == 0]))

            for h in range(horizon):
                seg = _segment_name(h, horizon)
                segment_counts[seg] = segment_counts.get(seg, 0.0) + 1.0
                if int(labels[h]) != 0:
                    segment_event_counts[seg] = segment_event_counts.get(seg, 0.0) + 1.0
                b = float(boundary[h])
                if b <= open_threshold:
                    bucket = "low"
                elif b >= closed_threshold:
                    bucket = "high"
                else:
                    bucket = "mid"
                boundary_counts[bucket] += 1.0
                if int(labels[h]) > 0:
                    event_from_boundary[f"close_from_{bucket}"] += 1.0
                elif int(labels[h]) < 0:
                    event_from_boundary[f"open_from_{bucket}"] += 1.0

    action_all = np.concatenate(action_values)
    state_all = np.concatenate(state_values)
    action_delta_all = np.concatenate(action_step_delta) if action_step_delta else np.asarray([], dtype=np.float64)
    state_delta_all = np.concatenate(state_step_delta) if state_step_delta else np.asarray([], dtype=np.float64)
    state_to_action_all = np.concatenate(state_to_action_delta) if state_to_action_delta else np.asarray([], dtype=np.float64)
    event_delta_all = np.concatenate([x for x in event_step_delta if x.size]) if any(x.size for x in event_step_delta) else np.asarray([], dtype=np.float64)
    hold_abs_all = np.concatenate([x for x in hold_step_abs_delta if x.size]) if any(x.size for x in hold_step_abs_delta) else np.asarray([], dtype=np.float64)
    window_event_delta_all = np.concatenate([x for x in all_window_event_delta if x.size]) if any(x.size for x in all_window_event_delta) else np.asarray([], dtype=np.float64)
    window_hold_abs_all = np.concatenate([x for x in all_window_hold_abs_delta if x.size]) if any(x.size for x in all_window_hold_abs_delta) else np.asarray([], dtype=np.float64)

    action_labels = _event_labels(action_delta_all, event_threshold)
    close_steps = float(np.sum(action_labels > 0))
    open_steps = float(np.sum(action_labels < 0))
    event_steps = close_steps + open_steps
    total_steps = float(action_labels.size)
    high_rate = _safe_ratio(float(np.sum(action_all >= closed_threshold)), float(action_all.size))
    low_rate = _safe_ratio(float(np.sum(action_all <= open_threshold)), float(action_all.size))
    mid_rate = 1.0 - high_rate - low_rate

    run_lengths = np.asarray([row["length"] for row in runs], dtype=np.float64)
    run_abs = np.asarray([row["abs_change"] for row in runs], dtype=np.float64)
    first_total = float(first_event_hist.sum())
    first_hist = {
        f"h{h:02d}": _safe_ratio(float(first_event_hist[h]), first_total)
        for h in range(horizon)
        if first_event_hist[h] > 0
    }
    segment_rates = {
        key: _safe_ratio(segment_event_counts.get(key, 0.0), count)
        for key, count in sorted(segment_counts.items())
    }

    supervision_hint: list[str] = []
    event_rate = _safe_ratio(event_steps, total_steps)
    close_open_balance = _safe_ratio(min(close_steps, open_steps), max(close_steps, open_steps))
    boundary_direction_rates = _summarize_boundary_event_direction(event_from_boundary)
    repeated_close_rate = float(boundary_direction_rates["close_from_high_rate_within_close"])
    repeated_open_rate = float(boundary_direction_rates["open_from_low_rate_within_open"])
    hold_abs_p95 = _quantiles(window_hold_abs_all).get("p95", float("nan"))
    event_abs_p50 = _quantiles(np.abs(window_event_delta_all)).get("p50", float("nan"))
    if event_rate < 0.05:
        supervision_hint.append("Events are sparse: do not rely on uniform gripper MSE; use event-window weighting.")
    if repeated_close_rate > 0.20 or repeated_open_rate > 0.20:
        supervision_hint.append("Events can originate inside the same coarse open/closed bucket; avoid hard binary state penalties.")
    if np.isfinite(hold_abs_p95) and hold_abs_p95 > 0.25 * event_threshold:
        supervision_hint.append("Hold regions have non-trivial drift: include a hold-stability term if model over-triggers.")
    if np.isfinite(event_abs_p50) and event_abs_p50 > 2.0 * event_threshold:
        supervision_hint.append("Event magnitudes are well above threshold: matched event-delta Huber is meaningful.")
    if close_open_balance < 0.50:
        supervision_hint.append("Open/close events are imbalanced: report direction-specific metrics and avoid one shared positive weight.")
    if mid_rate > 0.25:
        supervision_hint.append("Many gripper values are intermediate: avoid hard binary closed/open assumptions; keep continuous output primary.")

    return {
        "schema": "clearvla-gripper-dataset-probe-v1",
        "config": {
            "data_root": str(root),
            "glob": str(args.glob),
            "action_key": str(args.action_key),
            "state_key": args.state_key,
            "gripper_dim_index": int(args.gripper_dim_index),
            "horizon": horizon,
            "stride": stride,
            "unit": unit,
            "scale_to_degrees": scale,
            "angle_max_deg": angle_max,
            "event_threshold_deg": event_threshold,
            "hold_threshold_deg": hold_threshold,
            "open_threshold_deg": open_threshold,
            "closed_threshold_deg": closed_threshold,
        },
        "files": {
            "matched": len(files),
            "loaded": len(loaded),
            "skipped": len(skipped),
            "skipped_examples": skipped[:10],
        },
        "global": {
            "action_gripper_deg": _quantiles(action_all),
            "state_gripper_deg": _quantiles(state_all),
            "action_step_delta_deg": _quantiles(action_delta_all),
            "state_step_delta_deg": _quantiles(state_delta_all),
            "state_to_action_delta_deg": _quantiles(state_to_action_all),
            "low_open_rate": low_rate,
            "mid_rate": mid_rate,
            "high_closed_rate": high_rate,
        },
        "transition_steps": {
            "total_step_deltas": total_steps,
            "event_step_rate": event_rate,
            "hold_step_rate": 1.0 - event_rate,
            "close_step_count": close_steps,
            "open_step_count": open_steps,
            "close_step_rate": _safe_ratio(close_steps, total_steps),
            "open_step_rate": _safe_ratio(open_steps, total_steps),
            "close_open_min_over_max": close_open_balance,
            "event_delta_deg": _quantiles(event_delta_all),
            "event_abs_delta_deg": _quantiles(np.abs(event_delta_all)),
            "hold_abs_delta_deg": _quantiles(hold_abs_all),
            "event_run_length": _quantiles(run_lengths),
            "event_run_abs_change_deg": _quantiles(run_abs),
        },
        "windows": {
            "window_event_counts": _summarize_window_event_counts(window_event_counts, total_windows),
            "first_event_position_rate": first_hist,
            "segment_event_rates": segment_rates,
            "window_event_delta_deg": _quantiles(window_event_delta_all),
            "window_event_abs_delta_deg": _quantiles(np.abs(window_event_delta_all)),
            "window_hold_abs_delta_deg": _quantiles(window_hold_abs_all),
        },
        "state_conditioning": {
            "boundary_bucket_counts": _summarize_counter(boundary_counts),
            "event_from_boundary_counts": _summarize_counter(event_from_boundary),
            "event_from_boundary_by_direction": boundary_direction_rates,
            "repeat_close_from_high_rate": repeated_close_rate,
            "repeat_open_from_low_rate": repeated_open_rate,
        },
        "supervision_hint": supervision_hint,
        "episodes": episode_rows[: int(args.episode_rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe gripper state/transition statistics directly from HDF5 episodes.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=25)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--unit", choices=("auto", "degree", "radian", "raw"), default="auto")
    parser.add_argument("--angle-max-deg", type=float, default=100.0)
    parser.add_argument("--event-threshold-deg", type=float, default=5.0)
    parser.add_argument("--hold-threshold-deg", type=float, default=1.0)
    parser.add_argument("--open-threshold-deg", type=float, default=None)
    parser.add_argument("--closed-threshold-deg", type=float, default=None)
    parser.add_argument("--episode-rows", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    result = probe_dataset(args)
    text = json.dumps(_jsonable(result), ensure_ascii=False, indent=int(args.indent), sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
