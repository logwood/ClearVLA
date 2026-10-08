"""Explicit native dataset semantics, independent of model tensor capacity.

A contract describes an adapter; it does not assert that an arbitrary robot or
outlet is supported by a particular checkpoint. Legacy profiles and hashes are
unchanged. This v2 reader never guesses measured state from an action array.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .hdf5_episode import LoadedEpisode, load_episode


@dataclass(frozen=True)
class Timebase:
    unit: str = "local-control-step"
    seconds_per_step: float | None = None

    def validate(self) -> None:
        if self.unit not in {"local-control-step", "seconds"}:
            raise ValueError("unknown source timebase")
        if self.seconds_per_step is not None and (
            not math.isfinite(self.seconds_per_step) or self.seconds_per_step <= 0
        ):
            raise ValueError("seconds per control step must be finite and positive")
        if self.unit == "seconds" and self.seconds_per_step is not None:
            raise ValueError("second-valued timestamps must not be scaled a second time")

    def duration_seconds(self, value: float) -> float:
        self.validate()
        if not math.isfinite(value) or value < 0:
            raise ValueError("duration must be finite and nonnegative")
        if self.unit == "seconds":
            return float(value)
        if self.seconds_per_step is None:
            raise ValueError("unknown control frequency cannot manufacture seconds")
        return float(value * self.seconds_per_step)


@dataclass(frozen=True)
class NativeChannel:
    name: str
    unit: str
    frame: str
    kind: str

    def validate(self) -> None:
        if any(
            not isinstance(s, str) or not s.strip()
            for s in (self.name, self.unit, self.frame, self.kind)
        ):
            raise ValueError("native channels require named units, frame and meaning")


@dataclass(frozen=True)
class EffectorSpec:
    name: str
    command_indices: tuple[int, ...]
    kind: str
    open_direction: int | None = None

    def validate(self, action_width: int) -> None:
        if not self.name.strip() or self.kind not in {
            "binary",
            "position",
            "velocity",
            "force",
            "suction",
        }:
            raise ValueError("invalid effector specification")
        if (
            not self.command_indices
            or len(set(self.command_indices)) != len(self.command_indices)
            or any(type(i) is not int or not 0 <= i < action_width for i in self.command_indices)
        ):
            raise ValueError("effector indices must reference declared command channels")
        if self.open_direction is not None and (
            type(self.open_direction) is not int or self.open_direction not in (-1, 1)
        ):
            raise ValueError("effector polarity is either explicit +/-1 or unknown")


@dataclass(frozen=True)
class NativeArrayProfile:
    name: str
    source_action_dim: int
    source_state_dim: int
    action_indices: tuple[int, ...]
    state_indices: tuple[int, ...]
    action_channels: tuple[NativeChannel, ...]
    state_channels: tuple[NativeChannel, ...]
    command_reference: str
    camera_names: tuple[str, ...]
    effectors: tuple[EffectorSpec, ...] = ()
    timebase: Timebase = Timebase()
    schema: str = "clearvla-native-array-profile-v2"

    @property
    def output_dim(self) -> int:
        return len(self.action_indices)

    @property
    def state_output_dim(self) -> int:
        return len(self.state_indices)

    def validate(self) -> None:
        if self.schema != "clearvla-native-array-profile-v2" or not self.name.strip():
            raise ValueError("unknown native array profile")
        for indices, width, channels in (
            (self.action_indices, self.source_action_dim, self.action_channels),
            (self.state_indices, self.source_state_dim, self.state_channels),
        ):
            if type(width) is not int or width < 1 or not indices or len(indices) != len(channels):
                raise ValueError("native projection channels and source width disagree")
            if len(set(indices)) != len(indices) or any(
                type(i) is not int or not 0 <= i < width for i in indices
            ):
                raise ValueError("native projection indices are invalid")
            if len({channel.name for channel in channels}) != len(channels):
                raise ValueError("native channel names must be unique within each chart")
            for channel in channels:
                channel.validate()
        if self.command_reference not in {
            "absolute",
            "measured_state",
            "held_target",
            "velocity",
            "torque",
        }:
            raise ValueError("native command reference must be explicit, not inferred from shape")
        if (
            not self.camera_names
            or len(set(self.camera_names)) != len(self.camera_names)
            or any(not name.strip() for name in self.camera_names)
        ):
            raise ValueError("camera names must be distinct declared sources")
        for effector in self.effectors:
            effector.validate(self.output_dim)
        owned = [index for effector in self.effectors for index in effector.command_indices]
        if len(owned) != len(set(owned)):
            raise ValueError("effectors cannot silently share a native command channel")
        self.timebase.validate()

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def project_episode(self, episode: LoadedEpisode) -> LoadedEpisode:
        """Project distinct state/command arrays; never synthesize previous commands."""
        self.validate()
        arrays = (episode.actions_raw, episode.states_raw, episode.action_states_raw)
        widths = (self.source_action_dim, self.source_state_dim, self.source_action_dim)
        for name, value, width in zip(
            ("action", "observed state", "previous command"), arrays, widths, strict=True
        ):
            if value is None or value.ndim != 2 or value.shape != (episode.length, width):
                raise ValueError(f"{name} must be explicitly stored at its native width {width}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite observed values")
        actions, states, previous = arrays
        assert actions is not None and states is not None and previous is not None
        if tuple(episode.camera_keys) != self.camera_names:
            raise ValueError("episode camera order differs from the native contract")
        # Row zero is the explicitly stored reset/command boundary. Later
        # rows must agree with the actually submitted previous command.
        if episode.length > 1 and not np.allclose(previous[1:], actions[:-1], atol=1e-6, rtol=0):
            raise ValueError("previous-command boundary is not aligned to action rows")
        return replace(
            episode,
            actions_raw=np.ascontiguousarray(actions[:, self.action_indices], dtype=np.float32),
            states_raw=np.ascontiguousarray(states[:, self.state_indices], dtype=np.float32),
            action_states_raw=np.ascontiguousarray(
                previous[:, self.action_indices], dtype=np.float32
            ),
            source_action_dim=self.source_action_dim,
            source_state_dim=self.source_state_dim,
            data_profile=self.name,
        )

    def read_hdf5(
        self,
        path: Path,
        *,
        state_key: str,
        previous_command_key: str,
        camera_keys: dict[str, str],
        action_key: str = "action",
    ) -> LoadedEpisode:
        if not state_key or not previous_command_key or set(camera_keys) != set(self.camera_names):
            raise ValueError("v2 read requires explicit state/previous-command/camera keys")
        episode = load_episode(
            path,
            cameras=self.camera_names,
            action_key=action_key,
            state_key=state_key,
            action_state_key=previous_command_key,
            camera_key_overrides=camera_keys,
            array_contract="independent-native-v2",
        )
        return self.project_episode(episode)


@dataclass(frozen=True)
class EpisodeBoundary:
    last_observation: bool
    environment_terminal: bool | None = None
    truncated: bool | None = None
    task_success: bool | None = None
    annotated_goal_observation: bool | None = None

    def validate(self) -> None:
        for value in asdict(self).values():
            if value is not None and type(value) is not bool:
                raise ValueError("episode boundary fields are Boolean or explicitly unknown")
        if type(self.last_observation) is not bool:
            raise ValueError("last-observation status must be explicitly Boolean")
        # Independent flags can coincide at a time limit. Neither implies
        # task success or a goal observation.


def observed_time_indices(
    timestamps: np.ndarray, available_at: np.ndarray, *, now: float
) -> np.ndarray:
    """Select only evidence both captured and available by the decision time."""
    if timestamps.ndim != 1 or available_at.shape != timestamps.shape:
        raise ValueError("source timestamps and availability must align")
    if (
        not np.isfinite(now)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(available_at).all()
    ):
        raise ValueError("source clocks must be finite")
    if np.any(np.diff(timestamps) < 0) or np.any(available_at < timestamps):
        raise ValueError("source clock order or causal availability is invalid")
    return np.flatnonzero((timestamps <= now) & (available_at <= now))
