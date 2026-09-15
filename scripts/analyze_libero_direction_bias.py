"""Audit directional support in one converted LIBERO task.

The report separates four quantities that are easy to conflate:

* raw expert OSC_POSE command signs over complete demonstrations;
* commands that are actually visible to ClearVLA's legal 48-step windows;
* pre-close approach versus post-close transport commands; and
* task geometry reconstructed from the original MuJoCo demonstration states.

The script is read-only.  Its HDF5-only analysis needs NumPy and h5py.  Passing
``--raw-demo-hdf5`` additionally requires the LIBERO simulator environment and
recovers the target/plate positions from every recorded simulator state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


DEMO_PATTERN = re.compile(r"_demo_(\d+)$")
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def _finite(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("report value is not finite")
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = _finite(left, name="correlation left", ndim=1)
    y = _finite(right, name="correlation right", ndim=1)
    if x.shape != y.shape or x.size < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return _float(np.corrcoef(x, y)[0, 1])


def _direction(values: Iterable[float] | np.ndarray, *, epsilon: float) -> dict[str, Any]:
    array = _finite(list(values) if not isinstance(values, np.ndarray) else values,
                    name="direction values").reshape(-1)
    count = int(array.size)
    if count == 0:
        return {
            "count": 0,
            "positive_fraction": None,
            "negative_fraction": None,
            "near_zero_fraction": None,
            "positive_abs_mass_fraction": None,
            "negative_abs_mass_fraction": None,
            "mean": None,
            "median": None,
            "rms": None,
            "quantiles": None,
        }
    positive = array > float(epsilon)
    negative = array < -float(epsilon)
    near_zero = ~(positive | negative)
    positive_mass = float(np.maximum(array, 0.0).sum())
    negative_mass = float(np.maximum(-array, 0.0).sum())
    absolute_mass = positive_mass + negative_mass
    quantiles = np.quantile(array, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return {
        "count": count,
        "epsilon": float(epsilon),
        "positive_fraction": _float(positive.mean()),
        "negative_fraction": _float(negative.mean()),
        "near_zero_fraction": _float(near_zero.mean()),
        "positive_abs_mass_fraction": (
            None if absolute_mass <= 1e-15 else _float(positive_mass / absolute_mass)
        ),
        "negative_abs_mass_fraction": (
            None if absolute_mass <= 1e-15 else _float(negative_mass / absolute_mass)
        ),
        "mean": _float(array.mean()),
        "median": _float(np.median(array)),
        "rms": _float(np.sqrt(np.mean(np.square(array)))),
        "min": _float(array.min()),
        "max": _float(array.max()),
        "quantiles": {
            name: _float(value)
            for name, value in zip(
                ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                quantiles,
            )
        },
    }


def _first_sustained(mask: np.ndarray, *, width: int, start: int = 0) -> int | None:
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if width <= 0:
        raise ValueError("sustain width must be positive")
    for index in range(max(0, int(start)), max(0, len(flags) - width + 1)):
        if bool(flags[index : index + width].all()):
            return int(index)
    return None


def _demo_index(episode_id: str) -> int:
    match = DEMO_PATTERN.search(str(episode_id))
    if match is None:
        raise ValueError(f"cannot parse demo index from {episode_id!r}")
    return int(match.group(1))


@dataclass
class Episode:
    split: str
    episode_id: str
    demo_index: int
    path: Path
    action: np.ndarray
    state: np.ndarray
    valid_center_start: int
    valid_center_end: int
    first_close: int | None
    first_reopen: int | None

    @property
    def length(self) -> int:
        return int(self.action.shape[0])

    @property
    def centers(self) -> np.ndarray:
        return np.arange(
            int(self.valid_center_start), int(self.valid_center_end) + 1, dtype=np.int64
        )


def _load_episode(
    root: Path,
    split: str,
    episode_id: str,
    *,
    close_threshold: float,
    sustain: int,
) -> Episode:
    path = root / f"{episode_id}.hdf5"
    if not path.is_file():
        alternate = root / f"{episode_id}.h5"
        if not alternate.is_file():
            raise FileNotFoundError(path)
        path = alternate
    with h5py.File(path, "r") as stream:
        action = _finite(stream["action"][...], name=f"{episode_id}.action", ndim=2)
        state = _finite(stream["state"][...], name=f"{episode_id}.state", ndim=2)
        if action.shape[1] != 7 or state.shape[0] != action.shape[0] or state.shape[1] < 3:
            raise ValueError(f"{episode_id} has an invalid action/state ABI")
        low = int(stream.attrs.get("valid_center_start", 24))
        high = int(stream.attrs.get("valid_center_end", len(action) - 49))
        source_demo = stream.attrs.get("source_demo")
        if isinstance(source_demo, bytes):
            source_demo = source_demo.decode("utf-8")
    expected_demo = _demo_index(episode_id)
    if source_demo is not None and str(source_demo) != f"demo_{expected_demo}":
        raise ValueError(
            f"{episode_id} source_demo={source_demo!r} disagrees with its identity"
        )
    if low < 24 or high > len(action) - 49 or high < low:
        raise ValueError(
            f"{episode_id} legal centers [{low},{high}] do not own -24...+48"
        )
    close = _first_sustained(action[:, 6] > close_threshold, width=sustain)
    reopen = (
        None
        if close is None
        else _first_sustained(
            action[:, 6] < -close_threshold, width=sustain, start=close + sustain
        )
    )
    return Episode(
        split=str(split),
        episode_id=str(episode_id),
        demo_index=expected_demo,
        path=path,
        action=action,
        state=state,
        valid_center_start=low,
        valid_center_end=high,
        first_close=close,
        first_reopen=reopen,
    )


def _slice_before_close(episode: Episode, values: np.ndarray) -> np.ndarray:
    stop = episode.length if episode.first_close is None else int(episode.first_close)
    return np.asarray(values[:stop])


def _slice_after_close(episode: Episode, values: np.ndarray) -> np.ndarray:
    if episode.first_close is None:
        return np.asarray(values[:0])
    start = int(episode.first_close)
    stop = episode.length if episode.first_reopen is None else int(episode.first_reopen)
    return np.asarray(values[start:stop])


def _window_horizon_values(episode: Episode, *, dim: int) -> np.ndarray:
    rows = [episode.action[center : center + 24, dim] for center in episode.centers]
    return np.concatenate(rows, axis=0) if rows else np.empty(0, dtype=np.float64)


def _actual_eef_delta(episode: Episode, *, dim: int) -> np.ndarray:
    return np.diff(episode.state[:, dim])


def _aggregate(episodes: Sequence[Episode], *, epsilon: float) -> dict[str, Any]:
    def combine(parts: Iterable[np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in parts]
        arrays = [value for value in arrays if value.size]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)

    action_dy = combine(episode.action[:, 1] for episode in episodes)
    approach_dy = combine(
        _slice_before_close(episode, episode.action[:, 1]) for episode in episodes
    )
    transport_dy = combine(
        _slice_after_close(episode, episode.action[:, 1]) for episode in episodes
    )
    center_first_dy = combine(
        episode.action[episode.centers, 1] for episode in episodes
    )
    window_dy = combine(
        _window_horizon_values(episode, dim=1) for episode in episodes
    )
    eef_dy = combine(_actual_eef_delta(episode, dim=1) for episode in episodes)
    approach_eef_dy = combine(
        _slice_before_close(episode, _actual_eef_delta(episode, dim=1))
        for episode in episodes
    )
    net_to_close = []
    approach_command_sum = []
    for episode in episodes:
        close = episode.length - 1 if episode.first_close is None else episode.first_close
        close_state = min(int(close), episode.length - 1)
        net_to_close.append(float(episode.state[close_state, 1] - episode.state[0, 1]))
        approach_command_sum.append(float(episode.action[: int(close), 1].sum()))
    return {
        "episode_count": len(episodes),
        "demonstration_length_min_mean_max": [
            int(min(episode.length for episode in episodes)),
            _float(np.mean([episode.length for episode in episodes])),
            int(max(episode.length for episode in episodes)),
        ],
        "legal_window_count": int(sum(len(episode.centers) for episode in episodes)),
        "episodes_with_sustained_close": int(
            sum(episode.first_close is not None for episode in episodes)
        ),
        "episodes_with_sustained_reopen": int(
            sum(episode.first_reopen is not None for episode in episodes)
        ),
        "expert_action_dy_all_rows": _direction(action_dy, epsilon=epsilon),
        "expert_action_dy_pre_close_approach": _direction(
            approach_dy, epsilon=epsilon
        ),
        "expert_action_dy_close_to_reopen_transport": _direction(
            transport_dy, epsilon=epsilon
        ),
        "legal_window_first_action_dy": _direction(center_first_dy, epsilon=epsilon),
        "legal_window_24_step_dy_with_window_multiplicity": _direction(
            window_dy, epsilon=epsilon
        ),
        "observed_eef_delta_y_all_rows_m": _direction(eef_dy, epsilon=1e-5),
        "observed_eef_delta_y_pre_close_m": _direction(
            approach_eef_dy, epsilon=1e-5
        ),
        "per_demo_pre_close_net_eef_y_m": _direction(
            np.asarray(net_to_close), epsilon=1e-4
        ),
        "per_demo_pre_close_cumulative_action_dy": _direction(
            np.asarray(approach_command_sum), epsilon=epsilon
        ),
    }


def _body_position(env: Any, name: str) -> np.ndarray:
    inner = getattr(env, "env", env)
    simulator = getattr(inner, "sim", None)
    if simulator is None:
        raise ValueError("LIBERO environment has no MuJoCo simulator")
    body_id = int(simulator.model.body_name2id(str(name)))
    value = _finite(simulator.data.body_xpos[body_id], name=f"body {name}", ndim=1)
    if value.shape != (3,):
        raise ValueError(f"body {name!r} position has shape {value.shape}")
    return value.copy()


def _recover_layouts(
    episodes: Sequence[Episode],
    *,
    raw_demo_hdf5: Path,
    suite_name: str,
    task_id: int,
    target_body: str,
    plate_body: str,
    image_side: int,
    scan_object_trajectories: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_map = benchmark.get_benchmark_dict()
    suite = benchmark_map[str(suite_name)](task_order_index=0)
    task = suite.get_task(int(task_id))
    env = OffScreenRenderEnv(
        bddl_file_name=str(suite.get_task_bddl_file_path(int(task_id))),
        camera_heights=int(image_side),
        camera_widths=int(image_side),
    )
    layouts: dict[str, dict[str, Any]] = {}
    integrity_max_action_delta = 0.0
    try:
        env.reset()
        setter = getattr(env, "set_state", None)
        if not callable(setter):
            raise ValueError("LIBERO environment does not expose set_state")
        inner = getattr(env, "env", env)
        simulator = getattr(inner, "sim", None)
        if simulator is None:
            raise ValueError("LIBERO environment has no MuJoCo simulator")
        with h5py.File(raw_demo_hdf5, "r") as raw:
            data = raw.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError("raw LIBERO HDF5 has no data group")
            for episode in episodes:
                demo_name = f"demo_{episode.demo_index}"
                demo = data.get(demo_name)
                if not isinstance(demo, h5py.Group):
                    raise ValueError(f"raw LIBERO HDF5 has no data/{demo_name}")
                raw_states = _finite(
                    demo["states"][...], name=f"raw {demo_name}.states", ndim=2
                )
                raw_actions = _finite(
                    demo["actions"][...], name=f"raw {demo_name}.actions", ndim=2
                )
                if raw_states.shape[0] != episode.length or raw_actions.shape != episode.action.shape:
                    raise ValueError(
                        f"raw/converted length mismatch for {episode.episode_id}: "
                        f"states={raw_states.shape}, actions={raw_actions.shape}, "
                        f"converted={episode.action.shape}"
                    )
                integrity_max_action_delta = max(
                    integrity_max_action_delta,
                    float(np.max(np.abs(raw_actions - episode.action))),
                )
                setter(np.asarray(raw_states[0], dtype=np.float64))
                simulator.forward()
                target_initial = _body_position(env, target_body)
                plate_initial = _body_position(env, plate_body)
                eef_initial = np.asarray(episode.state[0, :3], dtype=np.float64)
                target_y = np.asarray([target_initial[1]], dtype=np.float64)
                plate_y = np.asarray([plate_initial[1]], dtype=np.float64)
                if scan_object_trajectories:
                    target_rows = []
                    plate_rows = []
                    for state in raw_states:
                        setter(np.asarray(state, dtype=np.float64))
                        simulator.forward()
                        target_rows.append(_body_position(env, target_body))
                        plate_rows.append(_body_position(env, plate_body))
                    target_trajectory = np.stack(target_rows, axis=0)
                    plate_trajectory = np.stack(plate_rows, axis=0)
                    target_y = target_trajectory[:, 1]
                    plate_y = plate_trajectory[:, 1]
                else:
                    target_trajectory = target_initial[None]
                    plate_trajectory = plate_initial[None]
                close = episode.length - 1 if episode.first_close is None else episode.first_close
                pre_stop = max(0, min(int(close), episode.length))
                target_pre = (
                    target_y[:pre_stop]
                    if target_y.size > 1
                    else np.full(pre_stop, target_initial[1], dtype=np.float64)
                )
                target_minus_eef_y_pre = target_pre - episode.state[:pre_stop, 1]
                post_start = min(int(close), episode.length - 1)
                post_stop = (
                    episode.length
                    if episode.first_reopen is None
                    else min(int(episode.first_reopen), episode.length)
                )
                post_count = max(0, post_stop - post_start)
                target_post = (
                    target_y[post_start:post_stop]
                    if target_y.size > 1
                    else np.full(post_count, target_initial[1], dtype=np.float64)
                )
                plate_post = (
                    plate_y[post_start:post_stop]
                    if plate_y.size > 1
                    else np.full(post_count, plate_initial[1], dtype=np.float64)
                )
                layouts[episode.episode_id] = {
                    "split": episode.split,
                    "demo_index": episode.demo_index,
                    "initial_target_xyz_m": target_initial.tolist(),
                    "initial_plate_xyz_m": plate_initial.tolist(),
                    "initial_eef_xyz_m": eef_initial.tolist(),
                    "initial_target_minus_eef_xyz_m": (
                        target_initial - eef_initial
                    ).tolist(),
                    "initial_plate_minus_target_xyz_m": (
                        plate_initial - target_initial
                    ).tolist(),
                    "pre_close_target_minus_eef_y": target_minus_eef_y_pre.tolist(),
                    "close_to_reopen_plate_minus_target_y": (
                        plate_post - target_post
                    ).tolist(),
                    "target_trajectory_y_min_max_m": [
                        _float(target_y.min()),
                        _float(target_y.max()),
                    ],
                    "target_final_minus_initial_y_m": _float(
                        target_y[-1] - target_y[0]
                    ),
                    "plate_trajectory_max_displacement_m": _float(
                        np.linalg.norm(plate_trajectory - plate_trajectory[0], axis=1).max()
                    ),
                }
    finally:
        env.close()
    return layouts, {
        "suite": str(suite_name),
        "task_id": int(task_id),
        "task_name": str(getattr(task, "name", "")),
        "raw_demo_hdf5": str(raw_demo_hdf5.resolve()),
        "target_body": str(target_body),
        "plate_body": str(plate_body),
        "scan_object_trajectories": bool(scan_object_trajectories),
        "raw_converted_action_max_abs_delta": _float(integrity_max_action_delta),
    }


def _geometry_aggregate(
    episodes: Sequence[Episode],
    layouts: Mapping[str, Mapping[str, Any]],
    *,
    epsilon_m: float,
) -> dict[str, Any]:
    target_y = np.asarray(
        [layouts[episode.episode_id]["initial_target_xyz_m"][1] for episode in episodes],
        dtype=np.float64,
    )
    eef_y = np.asarray(
        [layouts[episode.episode_id]["initial_eef_xyz_m"][1] for episode in episodes],
        dtype=np.float64,
    )
    target_minus_eef_y = target_y - eef_y
    plate_minus_target_y = np.asarray(
        [
            layouts[episode.episode_id]["initial_plate_minus_target_xyz_m"][1]
            for episode in episodes
        ],
        dtype=np.float64,
    )
    pre_relative = np.concatenate(
        [
            np.asarray(
                layouts[episode.episode_id]["pre_close_target_minus_eef_y"],
                dtype=np.float64,
            )
            for episode in episodes
        ]
    )
    post_relative_parts = [
        np.asarray(
            layouts[episode.episode_id]["close_to_reopen_plate_minus_target_y"],
            dtype=np.float64,
        )
        for episode in episodes
    ]
    post_relative = np.concatenate([v for v in post_relative_parts if v.size])
    cumulative_approach = np.asarray(
        [
            episode.action[
                : episode.length if episode.first_close is None else episode.first_close, 1
            ].sum()
            for episode in episodes
        ],
        dtype=np.float64,
    )
    net_eef_to_close = np.asarray(
        [
            episode.state[
                min(
                    episode.length - 1,
                    episode.length - 1
                    if episode.first_close is None
                    else episode.first_close,
                ),
                1,
            ]
            - episode.state[0, 1]
            for episode in episodes
        ],
        dtype=np.float64,
    )
    return {
        "initial_target_y_m": _direction(target_y, epsilon=epsilon_m),
        "initial_eef_y_m": _direction(eef_y, epsilon=epsilon_m),
        "initial_target_minus_eef_y_m": _direction(
            target_minus_eef_y, epsilon=epsilon_m
        ),
        "initial_plate_minus_target_y_m": _direction(
            plate_minus_target_y, epsilon=epsilon_m
        ),
        "per_row_pre_close_target_minus_eef_y_m": _direction(
            pre_relative, epsilon=epsilon_m
        ),
        "per_row_close_to_reopen_plate_minus_target_y_m": _direction(
            post_relative, epsilon=epsilon_m
        ),
        "correlations_across_demonstrations": {
            "initial_target_y_vs_initial_target_minus_eef_y": _correlation(
                target_y, target_minus_eef_y
            ),
            "initial_target_minus_eef_y_vs_pre_close_cumulative_action_dy": _correlation(
                target_minus_eef_y, cumulative_approach
            ),
            "initial_target_minus_eef_y_vs_pre_close_net_eef_y": _correlation(
                target_minus_eef_y, net_eef_to_close
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-demo-hdf5", type=Path, default=None)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--target-body", default="akita_black_bowl_1_main")
    parser.add_argument("--plate-body", default="plate_1_main")
    parser.add_argument("--image-side", type=int, default=64)
    parser.add_argument("--action-epsilon", type=float, default=0.01)
    parser.add_argument("--geometry-epsilon-m", type=float, default=0.001)
    parser.add_argument("--close-threshold", type=float, default=0.1)
    parser.add_argument("--sustain", type=int, default=3)
    parser.add_argument("--scan-object-trajectories", action="store_true")
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.action_epsilon < 0.0 or args.geometry_epsilon_m < 0.0:
        raise ValueError("direction epsilons must be non-negative")
    if args.sustain <= 0 or args.image_side <= 0:
        raise ValueError("sustain and image side must be positive")
    with args.split_manifest.open("r", encoding="utf-8") as stream:
        split_payload = json.load(stream)
    raw_splits = split_payload.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("split manifest has no splits mapping")
    split_order = ("train", "val", "test")
    episodes: list[Episode] = []
    for split in split_order:
        names = raw_splits.get(split)
        if not isinstance(names, list) or not names:
            raise ValueError(f"split manifest has no non-empty {split!r} list")
        for name in names:
            episodes.append(
                _load_episode(
                    args.converted_root,
                    split,
                    str(name),
                    close_threshold=float(args.close_threshold),
                    sustain=int(args.sustain),
                )
            )
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("split manifest repeats an episode")

    payload: dict[str, Any] = {
        "schema": "clearvla-libero-direction-support-audit-v1",
        "converted_root": str(args.converted_root.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "action_contract": {
            "names": list(ACTION_NAMES),
            "dy_index": 1,
            "chart": "official normalized OSC_POSE relative command",
            "gripper_sign_for_phase_only": {
                "positive": "close",
                "negative": "open",
                "threshold": float(args.close_threshold),
                "sustained_rows": int(args.sustain),
            },
            "meaningful_action_direction_epsilon": float(args.action_epsilon),
        },
        "window_contract": {
            "history": "-24...-1",
            "future_teacher": "+4...+48",
            "policy_action_rows": "center...center+23",
            "legal_centers": "HDF5 valid_center_start...valid_center_end inclusive",
            "stride": 1,
        },
        "splits": {
            split: _aggregate(
                [episode for episode in episodes if episode.split == split],
                epsilon=float(args.action_epsilon),
            )
            for split in split_order
        },
        "episodes": {
            episode.episode_id: {
                "split": episode.split,
                "demo_index": episode.demo_index,
                "length": episode.length,
                "legal_center_start": episode.valid_center_start,
                "legal_center_end": episode.valid_center_end,
                "legal_window_count": len(episode.centers),
                "first_sustained_close": episode.first_close,
                "first_sustained_reopen": episode.first_reopen,
                "all_action_dy": _direction(
                    episode.action[:, 1], epsilon=float(args.action_epsilon)
                ),
                "pre_close_action_dy": _direction(
                    _slice_before_close(episode, episode.action[:, 1]),
                    epsilon=float(args.action_epsilon),
                ),
                "legal_window_first_action_dy": _direction(
                    episode.action[episode.centers, 1],
                    epsilon=float(args.action_epsilon),
                ),
            }
            for episode in episodes
        },
    }

    if args.raw_demo_hdf5 is not None:
        layouts, layout_contract = _recover_layouts(
            episodes,
            raw_demo_hdf5=args.raw_demo_hdf5,
            suite_name=str(args.suite),
            task_id=int(args.task_id),
            target_body=str(args.target_body),
            plate_body=str(args.plate_body),
            image_side=int(args.image_side),
            scan_object_trajectories=bool(args.scan_object_trajectories),
        )
        payload["layout_contract"] = layout_contract
        payload["geometry_by_split"] = {
            split: _geometry_aggregate(
                [episode for episode in episodes if episode.split == split],
                layouts,
                epsilon_m=float(args.geometry_epsilon_m),
            )
            for split in split_order
        }
        for episode in episodes:
            payload["episodes"][episode.episode_id]["geometry"] = layouts[
                episode.episode_id
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    compact = {
        "output": str(args.output.resolve()),
        "splits": {
            split: {
                "episodes": payload["splits"][split]["episode_count"],
                "windows": payload["splits"][split]["legal_window_count"],
                "all_dy": payload["splits"][split]["expert_action_dy_all_rows"],
                "pre_close_dy": payload["splits"][split][
                    "expert_action_dy_pre_close_approach"
                ],
                "window_first_dy": payload["splits"][split][
                    "legal_window_first_action_dy"
                ],
            }
            for split in split_order
        },
    }
    if "geometry_by_split" in payload:
        compact["geometry_by_split"] = payload["geometry_by_split"]
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
