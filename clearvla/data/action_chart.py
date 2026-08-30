"""Explicit source-to-policy action/state charts for dataset adaptation.

The profile boundary is deliberately data owned.  It selects native source
coordinates and, when necessary, expresses observed qpos in the command chart
used by the action normalizer.  It does not change a model codec or claim that
an output width is supported by the active policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np

from .hdf5_episode import LoadedEpisode


@dataclass(frozen=True)
class ActionStateChartProfile:
    """One versioned projection from native HDF5 arrays to policy arrays."""

    name: str
    source_action_dim: int
    source_state_dim: int
    action_indices: tuple[int, ...]
    state_indices: tuple[int, ...]
    state_to_action_scale: tuple[float, ...]
    gripper_indices: tuple[int, ...]
    action_chart: str
    state_chart: str

    @property
    def output_dim(self) -> int:
        return len(self.action_indices)

    def validate(self) -> None:
        if not self.name or not self.action_chart or not self.state_chart:
            raise ValueError("action/state profile identities must be non-empty")
        if self.source_action_dim <= 0 or self.source_state_dim <= 0:
            raise ValueError("source action/state dimensions must be positive")
        if not self.action_indices or len(self.action_indices) != len(self.state_indices):
            raise ValueError("action/state profile projections must have equal nonzero width")
        if len(self.state_to_action_scale) != self.output_dim:
            raise ValueError("state-to-action scale must align with the output chart")
        if len(set(self.action_indices)) != self.output_dim:
            raise ValueError("action projection indices must be unique")
        if len(set(self.state_indices)) != self.output_dim:
            raise ValueError("state projection indices must be unique")
        if min(self.action_indices) < 0 or max(self.action_indices) >= self.source_action_dim:
            raise ValueError("action projection index is outside the source chart")
        if min(self.state_indices) < 0 or max(self.state_indices) >= self.source_state_dim:
            raise ValueError("state projection index is outside the source chart")
        if not self.gripper_indices or len(set(self.gripper_indices)) != len(
            self.gripper_indices
        ):
            raise ValueError("profile gripper indices must be non-empty and unique")
        if min(self.gripper_indices) < 0 or max(self.gripper_indices) >= self.output_dim:
            raise ValueError("profile gripper index is outside the output chart")
        scale = np.asarray(self.state_to_action_scale, dtype=np.float64)
        if not np.isfinite(scale).all() or bool(np.any(scale <= 0.0)):
            raise ValueError("state-to-action scales must be finite and positive")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def project_episode(self, episode: LoadedEpisode) -> LoadedEpisode:
        """Return an episode whose arrays all have one explicit output meaning."""

        self.validate()
        states = episode.states_raw
        if states is None:
            raise ValueError(f"{episode.path}: profile {self.name} requires qpos/state")
        actions = np.asarray(episode.actions_raw, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        if int(actions.shape[1]) != self.source_action_dim:
            raise ValueError(
                f"{episode.path}: profile {self.name} requires action width "
                f"{self.source_action_dim}, got {actions.shape[1]}"
            )
        if int(states.shape[1]) != self.source_state_dim:
            raise ValueError(
                f"{episode.path}: profile {self.name} requires state width "
                f"{self.source_state_dim}, got {states.shape[1]}"
            )
        projected_action = np.ascontiguousarray(actions[:, self.action_indices])
        projected_state = np.ascontiguousarray(states[:, self.state_indices])
        action_state = np.ascontiguousarray(
            projected_state
            * np.asarray(self.state_to_action_scale, dtype=np.float32)[None, :]
        )
        for name, value in (
            ("projected action", projected_action),
            ("projected state", projected_state),
            ("state in action chart", action_state),
        ):
            if tuple(value.shape) != (episode.length, self.output_dim):
                raise AssertionError(f"{name} has an inconsistent projected shape")
            if not np.isfinite(value).all():
                raise ValueError(f"{episode.path}: {name} contains non-finite values")
        return replace(
            episode,
            actions_raw=projected_action,
            states_raw=projected_state,
            action_states_raw=action_state,
            source_action_dim=int(actions.shape[1]),
            source_state_dim=int(states.shape[1]),
            data_profile=self.name,
        )


_RDT_LEFT_QPOS_GRIPPER_SCALE = 4.7908
_RDT_RIGHT_QPOS_GRIPPER_SCALE = 4.7888
_RDT_LEFT_ACTION_GRIPPER_SCALE = 11.8997
_RDT_RIGHT_ACTION_GRIPPER_SCALE = 13.9231


def _profile(
    *,
    name: str,
    action_indices: Sequence[int],
    state_indices: Sequence[int],
    state_to_action_scale: Sequence[float],
    gripper_indices: Sequence[int],
    action_chart: str,
    state_chart: str,
    source_dim: int,
) -> ActionStateChartProfile:
    result = ActionStateChartProfile(
        name=name,
        source_action_dim=int(source_dim),
        source_state_dim=int(source_dim),
        action_indices=tuple(int(value) for value in action_indices),
        state_indices=tuple(int(value) for value in state_indices),
        state_to_action_scale=tuple(float(value) for value in state_to_action_scale),
        gripper_indices=tuple(int(value) for value in gripper_indices),
        action_chart=action_chart,
        state_chart=state_chart,
    )
    result.validate()
    return result


ACTION_STATE_CHART_PROFILES: dict[str, ActionStateChartProfile] = {
    "identity_7d_pen": _profile(
        name="identity_7d_pen",
        action_indices=range(7),
        state_indices=range(7),
        state_to_action_scale=(1.0,) * 7,
        gripper_indices=(6,),
        action_chart="pen_native_6_joint_plus_gripper_command",
        state_chart="pen_native_6_joint_plus_gripper_qpos",
        source_dim=7,
    ),
    "rdt_right_arm_action_chart_v1": _profile(
        name="rdt_right_arm_action_chart_v1",
        action_indices=range(7, 14),
        state_indices=range(7, 14),
        state_to_action_scale=(1.0,) * 6
        + (_RDT_RIGHT_ACTION_GRIPPER_SCALE / _RDT_RIGHT_QPOS_GRIPPER_SCALE,),
        gripper_indices=(6,),
        action_chart="rdt_native_right_6_joint_plus_command_gripper",
        state_chart="rdt_native_right_6_joint_plus_qpos_gripper",
        source_dim=14,
    ),
    "rdt_left_arm_action_chart_v1": _profile(
        name="rdt_left_arm_action_chart_v1",
        action_indices=range(0, 7),
        state_indices=range(0, 7),
        state_to_action_scale=(1.0,) * 6
        + (_RDT_LEFT_ACTION_GRIPPER_SCALE / _RDT_LEFT_QPOS_GRIPPER_SCALE,),
        gripper_indices=(6,),
        action_chart="rdt_native_left_6_joint_plus_command_gripper",
        state_chart="rdt_native_left_6_joint_plus_qpos_gripper",
        source_dim=14,
    ),
    "rdt_bimanual_action_chart_v1": _profile(
        name="rdt_bimanual_action_chart_v1",
        action_indices=range(14),
        state_indices=range(14),
        state_to_action_scale=(1.0,) * 6
        + (_RDT_LEFT_ACTION_GRIPPER_SCALE / _RDT_LEFT_QPOS_GRIPPER_SCALE,)
        + (1.0,) * 6
        + (_RDT_RIGHT_ACTION_GRIPPER_SCALE / _RDT_RIGHT_QPOS_GRIPPER_SCALE,),
        gripper_indices=(6, 13),
        action_chart="rdt_native_bimanual_command_chart",
        state_chart="rdt_native_bimanual_qpos_chart",
        source_dim=14,
    ),
}


def resolve_action_state_profile(name: str) -> ActionStateChartProfile:
    try:
        return ACTION_STATE_CHART_PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown action/state data profile {name!r}; "
            f"known={sorted(ACTION_STATE_CHART_PROFILES)}"
        ) from error


def project_episodes(
    episodes: Sequence[LoadedEpisode],
    profile: ActionStateChartProfile,
) -> list[LoadedEpisode]:
    if not episodes:
        raise ValueError("cannot project an empty episode inventory")
    return [profile.project_episode(episode) for episode in episodes]


__all__ = [
    "ACTION_STATE_CHART_PROFILES",
    "ActionStateChartProfile",
    "project_episodes",
    "resolve_action_state_profile",
]
