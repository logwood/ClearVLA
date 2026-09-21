"""Read-only attribution report for a recorded StackCube rollout.

The recorder stores policy observations before an action and physics telemetry
after that action.  This tool keeps that boundary explicit: post-step opening
at row ``t`` is compared with the physical state at row ``t+1`` rather than
with the same-row observation.  Results are printed to stdout; no audit JSON is
created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _rows(stream: h5py.File) -> list[dict[str, Any]]:
    dataset = stream.get("sim/metrics_json")
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError("episode has no sim/metrics_json telemetry stream")
    result: list[dict[str, Any]] = []
    for raw in dataset:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(str(raw))
        if not isinstance(value, dict):
            raise ValueError("sim/metrics_json rows must be objects")
        result.append(value)
    return result


def _matrix(rows: list[dict[str, Any]], key: str, width: int) -> np.ndarray | None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (len(rows), width) or not np.isfinite(array).all():
        return None
    return array


def _vector(rows: list[dict[str, Any]], key: str) -> np.ndarray | None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (len(rows),) or not np.isfinite(array).all():
        return None
    return array


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def analyze(path: str | Path, *, event_threshold: float = 0.10) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as stream:
        action = np.asarray(stream["action"], dtype=np.float64)
        state = np.asarray(stream["state"], dtype=np.float64)
        action_state = np.asarray(stream["action_state"], dtype=np.float64)
        success = bool(np.asarray(stream["sim/success"]).any())
        rows = _rows(stream)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError("action must be [T,7]")
    if state.shape != action.shape or action_state.shape != action.shape:
        raise ValueError("state/action/action_state arrays are not aligned")
    if len(rows) != len(action):
        raise ValueError("metrics rows are not aligned with actions")

    tcp = _matrix(rows, "telemetry_tcp_pose", 7)
    cube_a = _matrix(rows, "telemetry_cube_a_pose", 7)
    cube_b = _matrix(rows, "telemetry_cube_b_pose", 7)
    opening = _vector(rows, "telemetry_gripper_opening")
    contact = _matrix(rows, "telemetry_cube_a_contact_force", 3)
    impulse = _matrix(rows, "telemetry_cube_a_contact_impulse", 3)
    finger1 = _matrix(rows, "telemetry_cube_a_finger1_contact_force", 3)
    finger2 = _matrix(rows, "telemetry_cube_a_finger2_contact_force", 3)
    contact_count = _vector(rows, "telemetry_contact_count")
    finger_qpos = _matrix(rows, "telemetry_gripper_finger_qpos", 2)
    finger_qvel = _matrix(rows, "telemetry_gripper_finger_qvel", 2)
    latency = _vector(rows, "telemetry_policy_latency_s")
    clip_fraction = _vector(rows, "telemetry_policy_clip_fraction")
    ik_fallback = _vector(rows, "telemetry_ik_fallback")
    controller_target_delta = _vector(
        rows, "telemetry_controller_arm_target_delta_qpos_norm"
    )
    first_events: list[int] = []
    trace_present = False
    for row_index, row in enumerate(rows):
        raw = row.get("telemetry_policy_raw_action_chunk")
        if raw is None:
            continue
        trace_present = True
        chunk = np.asarray(raw, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise ValueError("malformed telemetry_policy_raw_action_chunk")
        first = np.asarray(row.get("telemetry_policy_raw_action"), dtype=np.float64)
        executed_copy = np.asarray(
            row.get("telemetry_policy_executed_action"), dtype=np.float64
        )
        boundary = np.asarray(
            row.get("telemetry_policy_boundary_action"), dtype=np.float64
        )
        history_last = np.asarray(
            row.get("telemetry_policy_history_last_executed_action"), dtype=np.float64
        )
        if (
            first.shape != (7,)
            or executed_copy.shape != (7,)
            or boundary.shape != (7,)
            or history_last.shape != (7,)
            or not np.isfinite(first).all()
            or not np.isfinite(executed_copy).all()
            or not np.isfinite(boundary).all()
            or not np.isfinite(history_last).all()
            or not np.allclose(chunk[0], first, rtol=0.0, atol=1e-6)
            or not np.allclose(executed_copy, action[row_index], rtol=0.0, atol=1e-6)
            or not np.allclose(history_last, boundary, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("policy trace row-zero/boundary copy is inconsistent")
        executed_telemetry = row.get("telemetry_executed_action")
        if executed_telemetry is not None:
            executed_telemetry_array = np.asarray(executed_telemetry, dtype=np.float64)
            if (
                executed_telemetry_array.shape != (7,)
                or not np.isfinite(executed_telemetry_array).all()
                or not np.allclose(
                    executed_telemetry_array, action[row_index], rtol=0.0, atol=1e-6
                )
            ):
                raise ValueError("physics telemetry executed action is inconsistent")
        boundary = float(row.get("telemetry_policy_gripper_boundary", np.nan))
        if not np.isfinite(boundary):
            raise ValueError("policy trace gripper boundary is not finite")
        grip = chunk[:, -1]
        previous = np.concatenate(([boundary], grip[:-1]))
        events = np.flatnonzero(np.abs(grip - previous) >= float(event_threshold))
        if events.size:
            first_events.append(int(events[0]))
    if trace_present and any(
        row.get("telemetry_policy_raw_action_chunk") is None for row in rows
    ):
        raise ValueError("policy trace is present only on a subset of episode rows")

    distance = None
    horizontal = None
    vertical = None
    if tcp is not None and cube_a is not None:
        delta = tcp[:, :3] - cube_a[:, :3]
        distance = np.linalg.norm(delta, axis=1)
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        vertical = np.abs(delta[:, 2])

    executed_gripper = action[:, -1]
    executed_boundary = np.concatenate(([action_state[0, -1]], executed_gripper[:-1]))
    executed_delta = executed_gripper - executed_boundary
    executed_events = np.flatnonzero(np.abs(executed_delta) >= float(event_threshold))
    raw_event_steps = [
        index
        for index, row in enumerate(rows)
        if int(row.get("telemetry_policy_gripper_event_count", 0)) > 0
    ]

    result: dict[str, Any] = {
        "path": str(source),
        "steps": int(action.shape[0]),
        "success": success,
        "min_distance_m": None if distance is None else float(distance.min()),
        "min_distance_step": None if distance is None else int(distance.argmin()),
        # Report horizontal/vertical components at the same row as the
        # minimum Euclidean distance.  Independent component minima are not a
        # physically co-located pose and can otherwise be misleading.
        "min_distance_horizontal_m": (
            None if horizontal is None else float(horizontal[int(distance.argmin())])
        ),
        "min_distance_vertical_m": (
            None if vertical is None else float(vertical[int(distance.argmin())])
        ),
        "first_executed_event_step": None if not executed_events.size else int(executed_events[0]),
        "raw_event_decision_count": len(raw_event_steps),
        "policy_trace_present": trace_present,
        "raw_first_event_index_median": None if not first_events else float(np.median(first_events)),
        "raw_first_event_index_min": None if not first_events else int(min(first_events)),
        "max_finger_force_n": None
        if finger1 is None or finger2 is None
        else float(max(np.linalg.norm(finger1, axis=1).max(), np.linalg.norm(finger2, axis=1).max())),
        "max_cube_contact_force_n": None
        if contact is None
        else float(np.linalg.norm(contact, axis=1).max()),
        "max_cube_contact_impulse": None
        if impulse is None
        else float(np.linalg.norm(impulse, axis=1).max()),
        "max_contact_count": None if contact_count is None else int(contact_count.max()),
        "opening_min_m": None if opening is None else float(opening.min()),
        "opening_max_m": None if opening is None else float(opening.max()),
        "opening_final_m": None if opening is None else float(opening[-1]),
        "policy_latency_median_s": None if latency is None else float(np.median(latency)),
        "policy_latency_p95_s": None if latency is None else float(np.quantile(latency, 0.95)),
        "policy_clip_fraction_mean": None
        if clip_fraction is None
        else float(np.mean(clip_fraction)),
        "ik_fallback_fraction": None
        if ik_fallback is None
        else float(np.mean(ik_fallback > 0.5)),
        "ik_fallback_steps": None
        if ik_fallback is None
        else int(np.count_nonzero(ik_fallback > 0.5)),
        "controller_target_delta_qpos_norm_rms": None
        if controller_target_delta is None
        else float(np.sqrt(np.mean(controller_target_delta**2))),
        "controller_target_delta_qpos_norm_max": None
        if controller_target_delta is None
        else float(np.max(controller_target_delta)),
    }
    if tcp is not None:
        # ``state[t]`` is the pre-action TCP position, while telemetry TCP is
        # sampled immediately after executing action[t].  This is the realized
        # one-step arm response; it is not a prediction of the controller.
        tcp_response = tcp[:, :3] - state[:, :3]
        tcp_response_norm = np.linalg.norm(tcp_response, axis=1)
        command_translation = action[:, :3]
        command_translation_norm = np.linalg.norm(command_translation, axis=1)
        result["tcp_position_response_rms_m"] = float(
            np.sqrt(np.mean(tcp_response_norm**2))
        )
        result["tcp_position_response_max_m"] = float(np.max(tcp_response_norm))
        result["tcp_position_response_nonzero_fraction"] = float(
            np.mean(tcp_response_norm > 1e-6)
        )
        active = command_translation_norm > 1e-6
        if np.any(active):
            result["tcp_response_to_normalized_translation_ratio_m"] = float(
                np.median(tcp_response_norm[active] / command_translation_norm[active])
            )
            response_active = tcp_response[active]
            command_active = command_translation[active]
            denom = np.linalg.norm(response_active, axis=1) * np.linalg.norm(
                command_active, axis=1
            )
            valid = denom > 1e-10
            if np.any(valid):
                result["tcp_response_command_translation_cosine_mean"] = float(
                    np.mean(
                        np.sum(response_active[valid] * command_active[valid], axis=1)
                        / denom[valid]
                    )
                )
    if finger_qpos is not None:
        qpos_opening = finger_qpos.sum(axis=1)
        result["finger_qpos_min_m"] = float(finger_qpos.min())
        result["finger_qpos_max_m"] = float(finger_qpos.max())
        result["finger_qpos_opening_delta_rms_m"] = float(
            np.sqrt(np.mean(np.diff(qpos_opening, prepend=qpos_opening[0]) ** 2))
        )
        result["finger_qpos_opening_step_count"] = int(
            np.count_nonzero(np.abs(np.diff(qpos_opening)) > 1e-6)
        )
        if opening is not None:
            result["opening_vs_finger_qpos_max_abs_m"] = float(
                np.max(np.abs(opening - qpos_opening))
            )
    if finger_qvel is not None:
        result["finger_qvel_max_abs_m_s"] = float(np.abs(finger_qvel).max())
        result["finger_qvel_rms_m_s"] = float(np.sqrt(np.mean(finger_qvel**2)))
    if opening is not None:
        # ``state[t]`` is the pre-action observation and ``opening[t]`` is the
        # post-action physical measurement.  This gives the realized one-step
        # gripper response without assuming that the policy command is itself
        # a position or a velocity.
        opening_response = opening - state[:, -1]
        result["opening_response_rms_m"] = float(
            np.sqrt(np.mean(opening_response**2))
        )
        result["opening_response_max_abs_m"] = float(np.abs(opening_response).max())
        changed = np.flatnonzero(np.abs(opening_response) > 1e-6)
        result["first_opening_response_step"] = (
            None if not changed.size else int(changed[0])
        )
    if cube_a is not None:
        result["cube_a_z_delta_m"] = float(cube_a[-1, 2] - cube_a[0, 2])
        result["cube_a_xy_delta_m"] = float(np.linalg.norm(cube_a[-1, :2] - cube_a[0, :2]))
    if opening is not None and state.shape[0] > 1:
        aligned = state[1:, -1]
        result["poststep_opening_vs_next_state_max_abs_m"] = float(
            np.max(np.abs(opening[:-1] - aligned))
        )
    if raw_event_steps and distance is not None:
        first_raw_step = raw_event_steps[0]
        result["first_raw_event_distance_m"] = float(distance[first_raw_step])
        result["first_raw_event_step"] = int(first_raw_step)

    print(f"episode: {source}")
    print(f"steps={result['steps']} success={result['success']}")
    print(
        "distance: min="
        f"{_fmt(result['min_distance_m'])} m at step {result['min_distance_step']} "
        f"(horizontal={_fmt(result['min_distance_horizontal_m'])}, "
        f"vertical={_fmt(result['min_distance_vertical_m'])})"
    )
    print(
        "gripper: first executed event="
        f"{result['first_executed_event_step']}; raw-event decisions="
        f"{result['raw_event_decision_count']}; raw first-index median="
        f"{_fmt(result['raw_first_event_index_median'])}"
    )
    print(
        "contact: max finger force="
        f"{_fmt(result['max_finger_force_n'])} N; max cube force="
        f"{_fmt(result['max_cube_contact_force_n'])} N; "
        f"max contacts={result['max_contact_count']}"
    )
    print(
        "opening: "
        f"{_fmt(result['opening_min_m'])}..{_fmt(result['opening_max_m'])} m, "
        f"final={_fmt(result['opening_final_m'])}; "
        "post-step/next-state max="
        f"{_fmt(result.get('poststep_opening_vs_next_state_max_abs_m'))} m"
    )
    if result.get("opening_response_rms_m") is not None:
        print(
            "gripper response: pre-state->post-step RMS/max="
            f"{_fmt(result['opening_response_rms_m'])}/"
            f"{_fmt(result['opening_response_max_abs_m'])} m; "
            f"first nonzero step={result['first_opening_response_step']}"
        )
    if result.get("finger_qvel_rms_m_s") is not None:
        print(
            "gripper joints: qpos range="
            f"{_fmt(result.get('finger_qpos_min_m'))}.."
            f"{_fmt(result.get('finger_qpos_max_m'))} m; qvel RMS/max="
            f"{_fmt(result['finger_qvel_rms_m_s'])}/"
            f"{_fmt(result['finger_qvel_max_abs_m_s'])} m/s; "
            f"qpos opening changes={result.get('finger_qpos_opening_step_count')}"
        )
    print(
        "runtime: latency median/p95="
        f"{_fmt(result['policy_latency_median_s'])}/"
        f"{_fmt(result['policy_latency_p95_s'])} s; mean row-zero clip="
        f"{_fmt(result['policy_clip_fraction_mean'])}"
    )
    if result.get("tcp_position_response_rms_m") is not None:
        print(
            "arm response: TCP position RMS/max="
            f"{_fmt(result['tcp_position_response_rms_m'])}/"
            f"{_fmt(result['tcp_position_response_max_m'])} m; nonzero fraction="
            f"{_fmt(result['tcp_position_response_nonzero_fraction'])}; "
            "response/normalized-translation median="
            f"{_fmt(result.get('tcp_response_to_normalized_translation_ratio_m'))}"
        )
    if result.get("ik_fallback_steps") is not None:
        print(
            "controller: IK fallback steps="
            f"{result['ik_fallback_steps']}/{result['steps']} (fraction="
            f"{_fmt(result.get('ik_fallback_fraction'))}); target-delta qpos RMS/max="
            f"{_fmt(result.get('controller_target_delta_qpos_norm_rms'))}/"
            f"{_fmt(result.get('controller_target_delta_qpos_norm_max'))}"
        )
    if not trace_present:
        print(
            "flag: policy proposal trace is absent; raw chunk timing, latency, "
            "and row-zero boundary attribution are unavailable"
        )
    if result.get("first_raw_event_distance_m") is not None:
        print(
            "attribution: first raw chunk event at step "
            f"{result['first_raw_event_step']} while TCP-cube distance was "
            f"{_fmt(result['first_raw_event_distance_m'])} m"
        )
    if (
        result.get("first_raw_event_distance_m") is not None
        and float(result["first_raw_event_distance_m"]) > 0.04
    ):
        print("flag: gripper event is proposed before spatial convergence (second-layer timing issue)")
    if result.get("max_finger_force_n") is not None and float(result["max_finger_force_n"]) < 1e-3:
        print("flag: no finger-cube contact force was observed")
    if result.get("cube_a_z_delta_m") is not None and abs(float(result["cube_a_z_delta_m"])) < 1e-4:
        print("flag: cube height did not change")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--event-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if not np.isfinite(args.event_threshold) or args.event_threshold < 0.0:
        parser.error("--event-threshold must be finite and non-negative")
    analyze(args.episode, event_threshold=args.event_threshold)


if __name__ == "__main__":
    main()
