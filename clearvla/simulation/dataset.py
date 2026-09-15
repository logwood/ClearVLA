"""Atomic ClearVLA-compatible HDF5 recording for simulator trajectories."""

from __future__ import annotations

import json
import os
import tempfile
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from clearvla.data.hdf5_episode import load_episode

from .contracts import (
    ACTION_DIM,
    EnvironmentDescriptor,
    PolicyObservation,
    StepResult,
)

SIM_DATASET_SCHEMA = "clearvla-simulation-dataset-v1"
SIM_EPISODE_SCHEMA = "clearvla-simulation-episode-v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_dataset_manifest(root: Path, descriptor: EnvironmentDescriptor) -> Path:
    descriptor.validate()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_manifest.json"
    payload = {
        "schema": SIM_DATASET_SCHEMA,
        "environment": descriptor.to_dict(),
        "clearvla_hdf5": {
            "action_key": "action",
            "action_state_key": "action_state",
            "state_key": "state",
            "top_camera_key": "observations/images/cam_high",
            "wrist_camera_key": "observations/images/cam_right_wrist",
            "action_dim": ACTION_DIM,
            "state_dim": 7,
            "image_layout": "THWC_uint8_rgb",
        },
        "policy_evidence": ["rgb", "state", "action_state"],
        "evaluator_only": ["reward", "terminated", "truncated", "sim/metrics_json"],
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"dataset manifest at {path} belongs to different simulator semantics")
    else:
        _atomic_json(path, payload)
    return path


@dataclass(frozen=True)
class RecordedEpisode:
    path: Path
    steps: int
    success: bool
    reward_sum: float


class EpisodeRecorder:
    """Collect one episode and commit it atomically as a flat HDF5 file."""

    def __init__(
        self,
        root: str | Path,
        *,
        descriptor: EnvironmentDescriptor,
        episode_id: str,
        seed: int,
        instruction: str,
        collector: str,
        source_revision: str = "unknown",
        provenance: Mapping[str, Any] | None = None,
        valid_center_bounds: tuple[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.descriptor = descriptor
        self.episode_id = str(episode_id)
        self.seed = int(seed)
        self.instruction = str(instruction)
        self.collector = str(collector)
        self.source_revision = str(source_revision)
        self.provenance = dict(provenance or {})
        self.valid_center_bounds = valid_center_bounds
        if not self.episode_id or any(char in self.episode_id for char in "/\\"):
            raise ValueError("episode_id must be a non-empty filename-safe value")
        if not self.instruction.strip():
            raise ValueError("simulation episode instruction must be non-empty")
        if not self.collector.strip():
            raise ValueError("simulation episode collector must be non-empty")
        ensure_dataset_manifest(self.root, descriptor)
        self._observations: list[PolicyObservation] = []
        self._actions: list[np.ndarray] = []
        self._steps: list[StepResult] = []

    def append(
        self,
        observation: PolicyObservation,
        action: np.ndarray,
        result: StepResult,
    ) -> None:
        observation.validate()
        result.validate()
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("recorded action must be one finite [7] vector")
        if self._steps and (self._steps[-1].terminated or self._steps[-1].truncated):
            raise RuntimeError("cannot append after an episode terminal")
        self._observations.append(observation.copied())
        self._actions.append(value.copy())
        self._steps.append(result)

    def finish(self) -> RecordedEpisode:
        if not self._steps:
            raise ValueError("cannot record an empty simulation episode")
        destination = self.root / f"{self.episode_id}.hdf5"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite recorded episode: {destination}")
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.root
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            self._write(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        success = any(bool(step.evaluation.metrics.get("success", False)) for step in self._steps)
        return RecordedEpisode(
            path=destination,
            steps=len(self._steps),
            success=success,
            reward_sum=float(sum(step.reward for step in self._steps)),
        )

    def _write(self, path: Path) -> None:
        if self.valid_center_bounds is not None:
            lo, hi = self.valid_center_bounds
            if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < len(self._steps):
                raise ValueError("recorded policy center bounds are invalid")
        states = np.stack([row.state for row in self._observations]).astype(np.float32)
        action_states = np.stack([row.action_state for row in self._observations]).astype(
            np.float32
        )
        actions = np.stack(self._actions).astype(np.float32)
        if states.shape != actions.shape or action_states.shape != actions.shape:
            raise RuntimeError("recorded state/action rows lost [T,7] alignment")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as stream:
            stream.attrs["schema"] = SIM_EPISODE_SCHEMA
            stream.attrs["environment_descriptor_json"] = _json(self.descriptor.to_dict())
            stream.attrs["instruction"] = self.instruction
            stream.attrs["collector"] = self.collector
            stream.attrs["seed"] = self.seed
            stream.attrs["source_revision"] = self.source_revision
            if self.provenance:
                stream.attrs["provenance_json"] = _json(self.provenance)
            if self.valid_center_bounds is not None:
                stream.attrs["valid_center_start"], stream.attrs["valid_center_end"] = self.valid_center_bounds
            stream.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            stream.create_dataset("action", data=actions)
            stream.create_dataset("action_state", data=action_states)
            stream.create_dataset("state", data=states)
            image_group = stream.require_group("observations/images")
            image_group.create_dataset(
                "cam_high",
                data=np.stack([row.rgb["top"] for row in self._observations]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            image_group.create_dataset(
                "cam_right_wrist",
                data=np.stack([row.rgb["wrist"] for row in self._observations]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            sim = stream.require_group("sim")
            sim.create_dataset(
                "reward", data=np.asarray([row.reward for row in self._steps], dtype=np.float32)
            )
            sim.create_dataset(
                "terminated",
                data=np.asarray([row.terminated for row in self._steps], dtype=bool),
            )
            sim.create_dataset(
                "truncated",
                data=np.asarray([row.truncated for row in self._steps], dtype=bool),
            )
            sim.create_dataset(
                "success",
                data=np.asarray(
                    [bool(row.evaluation.metrics.get("success", False)) for row in self._steps],
                    dtype=bool,
                ),
            )
            sim.create_dataset(
                "metrics_json",
                data=np.asarray(
                    [_json(dict(row.evaluation.metrics)) for row in self._steps],
                    dtype=object,
                ),
                dtype=string_dtype,
            )


def validate_recorded_episode(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with h5py.File(source, "r") as stream:
        if stream.attrs.get("schema") != SIM_EPISODE_SCHEMA:
            raise ValueError(f"{source} is not a {SIM_EPISODE_SCHEMA} episode")
        action = np.asarray(stream["action"], dtype=np.float32)
        action_state = np.asarray(stream["action_state"], dtype=np.float32)
        state = np.asarray(stream["state"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            raise ValueError("recorded action must be [T,7]")
        if action_state.shape != action.shape or state.shape != action.shape:
            raise ValueError("recorded action/action-state/state arrays must align")
        transition_error = 0.0
        if action.shape[0] > 1:
            transition_error = float(np.max(np.abs(action[:-1] - action_state[1:]), initial=0.0))
            if transition_error > 1e-6:
                raise ValueError(
                    "recorded action[t] must equal the next observation's "
                    f"action_state[t+1], max error={transition_error:.6g}"
                )
        for camera, key in (
            ("top", "observations/images/cam_high"),
            ("wrist", "observations/images/cam_right_wrist"),
        ):
            image = stream.get(key)
            if not isinstance(image, h5py.Dataset):
                raise ValueError(f"recorded {camera} camera dataset is missing")
            if image.shape[0] != action.shape[0] or image.shape[-1] != 3:
                raise ValueError(f"recorded {camera} camera is not aligned THWC RGB")
        if not all(np.isfinite(value).all() for value in (action, action_state, state)):
            raise ValueError("recorded policy arrays contain NaN or infinity")
        policy_trace = False
        policy_trace_schema: str | None = None
        metrics_dataset = stream.get("sim/metrics_json")
        if isinstance(metrics_dataset, h5py.Dataset):
            metric_rows: list[dict[str, Any]] = []
            for raw in metrics_dataset:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    value = json.loads(str(raw))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("recorded simulator metrics are not valid JSON") from error
                if not isinstance(value, Mapping):
                    raise ValueError("recorded simulator metrics must be JSON objects")
                metric_rows.append(dict(value))
            if len(metric_rows) != int(action.shape[0]):
                raise ValueError("recorded simulator metrics are not aligned to actions")
            trace_keys = {
                "telemetry_policy_trace_schema",
                "telemetry_policy_raw_action_chunk",
                "telemetry_policy_raw_action",
                "telemetry_policy_boundary_action",
                "telemetry_policy_history_last_executed_action",
                "telemetry_policy_executed_action",
                "telemetry_policy_latency_s",
                "telemetry_policy_step",
            }
            policy_trace = any(trace_keys.intersection(row) for row in metric_rows)
            if policy_trace:
                required = trace_keys
                for index, row in enumerate(metric_rows):
                    if not required.issubset(row):
                        missing = sorted(required.difference(row))
                        raise ValueError(
                            f"policy trace row {index} is missing fields: {missing}"
                        )
                    schema = str(row["telemetry_policy_trace_schema"])
                    if policy_trace_schema is None:
                        policy_trace_schema = schema
                    elif schema != policy_trace_schema:
                        raise ValueError("policy trace schema changes within an episode")
                    try:
                        chunk = np.asarray(
                            row["telemetry_policy_raw_action_chunk"], dtype=np.float32
                        )
                        first = np.asarray(
                            row["telemetry_policy_raw_action"], dtype=np.float32
                        )
                        boundary = np.asarray(
                            row["telemetry_policy_boundary_action"], dtype=np.float32
                        )
                        history_last = np.asarray(
                            row["telemetry_policy_history_last_executed_action"],
                            dtype=np.float32,
                        )
                        executed = np.asarray(
                            row["telemetry_policy_executed_action"], dtype=np.float32
                        )
                        latency = float(row["telemetry_policy_latency_s"])
                        step = int(row["telemetry_policy_step"])
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError(f"policy trace row {index} is malformed") from error
                    if (
                        chunk.ndim != 2
                        or chunk.shape[1] != ACTION_DIM
                        or chunk.shape[0] < 1
                        or first.shape != (ACTION_DIM,)
                        or boundary.shape != (ACTION_DIM,)
                        or history_last.shape != (ACTION_DIM,)
                        or executed.shape != (ACTION_DIM,)
                        or not np.isfinite(chunk).all()
                        or not np.isfinite(first).all()
                        or not np.isfinite(boundary).all()
                        or not np.isfinite(history_last).all()
                        or not np.isfinite(executed).all()
                        or not np.isfinite(latency)
                        or latency < 0.0
                        or step != index
                    ):
                        raise ValueError(f"policy trace row {index} violates its shape/timing contract")
                    if not np.allclose(chunk[0], first, rtol=0.0, atol=1e-6):
                        raise ValueError(f"policy trace row {index} row-zero copy is inconsistent")
                    if not np.allclose(executed, action[index], rtol=0.0, atol=1e-6):
                        raise ValueError(f"policy trace row {index} executed copy is inconsistent")
                    if not np.allclose(history_last, boundary, rtol=0.0, atol=1e-6):
                        raise ValueError(f"policy trace row {index} history boundary is inconsistent")
        return {
            "schema": SIM_EPISODE_SCHEMA,
            "steps": int(action.shape[0]),
            "success": bool(np.asarray(stream["sim/success"]).any()),
            "instruction": str(stream.attrs["instruction"]),
            "collector": str(stream.attrs["collector"]),
            "action_state_transition_max_abs_error": transition_error,
            "policy_trace": policy_trace,
            "policy_trace_schema": policy_trace_schema,
        }


def audit_simulation_dataset(root: str | Path) -> dict[str, Any]:
    """Validate every episode against one immutable simulator descriptor."""

    source = Path(root)
    manifest_path = source / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"simulation dataset has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SIM_DATASET_SCHEMA:
        raise ValueError(f"{manifest_path} is not a {SIM_DATASET_SCHEMA} manifest")
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("simulation dataset manifest has no environment descriptor")
    files = sorted((*source.glob("*.h5"), *source.glob("*.hdf5")))
    if not files:
        raise FileNotFoundError(f"simulation dataset has no HDF5 episodes under {source}")
    summaries: list[dict[str, Any]] = []
    collectors: Counter[str] = Counter()
    for path in files:
        with h5py.File(path, "r") as stream:
            raw_descriptor = stream.attrs.get("environment_descriptor_json")
            if raw_descriptor is None:
                raise ValueError(f"{path} has no serialized environment descriptor")
            if isinstance(raw_descriptor, bytes):
                raw_descriptor = raw_descriptor.decode("utf-8")
            descriptor = json.loads(str(raw_descriptor))
        if descriptor != dict(environment):
            raise ValueError(f"{path} environment descriptor differs from its dataset manifest")
        summary = validate_recorded_episode(path)
        loaded = load_episode(
            path,
            cameras=("top", "wrist"),
            action_key="action",
            action_state_key="action_state",
            state_key="state",
            camera_key_overrides={
                "top": "observations/images/cam_high",
                "wrist": "observations/images/cam_right_wrist",
            },
        )
        if loaded.length != int(summary["steps"]):
            raise ValueError(f"{path} changes length at the ClearVLA loader boundary")
        if loaded.action_state_key != "action_state" or loaded.state_key != "state":
            raise ValueError(f"{path} did not retain explicit simulator state boundaries")
        summaries.append(summary)
        collectors[str(summary["collector"])] += 1
    return {
        "schema": SIM_DATASET_SCHEMA,
        "root": str(source.resolve()),
        "environment": dict(environment),
        "episodes": len(summaries),
        "clearvla_loader_compatible_episodes": len(summaries),
        "steps": sum(int(row["steps"]) for row in summaries),
        "successful_episodes": sum(bool(row["success"]) for row in summaries),
        "action_state_transition_max_abs_error": max(
            float(row["action_state_transition_max_abs_error"]) for row in summaries
        ),
        "collectors": dict(sorted(collectors.items())),
    }


def _main() -> None:
    parser = ArgumentParser(description="Audit one ClearVLA simulation dataset root")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_simulation_dataset(args.root), indent=2, sort_keys=True))


__all__ = [
    "SIM_DATASET_SCHEMA",
    "SIM_EPISODE_SCHEMA",
    "EpisodeRecorder",
    "RecordedEpisode",
    "audit_simulation_dataset",
    "ensure_dataset_manifest",
    "validate_recorded_episode",
]


if __name__ == "__main__":
    _main()
