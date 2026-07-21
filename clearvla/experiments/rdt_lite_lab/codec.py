from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.data.normalizer import ZScoreNormalizer

ActionRepresentation = Literal["absolute", "relative_to_current"]


@dataclass(frozen=True)
class RDTLiteCodecs:
    """Explicit state/action coordinate system for the direct-action lab.

    The reference model receives normalized measured states and predicts an
    encoded action trajectory.  The codec keeps those two coordinate systems
    separate.  This prevents the model implementation from silently assuming
    that qpos and emitted actions share the same statistics or semantics.
    """

    state_normalizer: ZScoreNormalizer
    action_normalizer: ZScoreNormalizer
    target_normalizer: ZScoreNormalizer
    action_representation: ActionRepresentation = "absolute"

    def validate(self) -> None:
        if self.action_representation not in ("absolute", "relative_to_current"):
            raise ValueError(f"unsupported action_representation={self.action_representation!r}")
        state_dim = int(self.state_normalizer.mean.shape[-1])
        action_dim = int(self.action_normalizer.mean.shape[-1])
        target_dim = int(self.target_normalizer.mean.shape[-1])
        if state_dim != action_dim or action_dim != target_dim:
            raise ValueError(
                f"state/action/target dims must match, got {state_dim}/{action_dim}/{target_dim}"
            )

    @property
    def action_dim(self) -> int:
        self.validate()
        return int(self.action_normalizer.mean.shape[-1])

    def encode_state(self, raw_state: np.ndarray) -> np.ndarray:
        return self.state_normalizer.encode(raw_state)

    def encode_action_absolute(self, raw_action: np.ndarray) -> np.ndarray:
        return self.action_normalizer.encode(raw_action)

    def encode_target(
        self, raw_future_action: np.ndarray, raw_current_state: np.ndarray
    ) -> np.ndarray:
        if self.action_representation == "absolute":
            target_raw = raw_future_action
        elif self.action_representation == "relative_to_current":
            target_raw = raw_future_action - np.asarray(raw_current_state, dtype=np.float32)
        else:  # pragma: no cover - guarded by validate
            raise ValueError(self.action_representation)
        return self.target_normalizer.encode(np.asarray(target_raw, dtype=np.float32))

    def decode_target_raw(
        self, encoded_target: np.ndarray, raw_current_state: np.ndarray
    ) -> np.ndarray:
        decoded = self.target_normalizer.decode(encoded_target)
        if self.action_representation == "absolute":
            return decoded
        if self.action_representation == "relative_to_current":
            state = np.asarray(raw_current_state, dtype=np.float32)
            while state.ndim < decoded.ndim:
                state = np.expand_dims(state, axis=-2)
            return (decoded + state).astype(np.float32)
        raise ValueError(self.action_representation)  # pragma: no cover

    def decode_target_to_action_norm(
        self, encoded_target: np.ndarray, raw_current_state: np.ndarray
    ) -> np.ndarray:
        return self.action_normalizer.encode(
            self.decode_target_raw(encoded_target, raw_current_state)
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "clearvla-rdt-lite-codecs-v1",
            "action_representation": self.action_representation,
            "state_normalizer": self.state_normalizer.to_dict(),
            "action_normalizer": self.action_normalizer.to_dict(),
            "target_normalizer": self.target_normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RDTLiteCodecs":
        out = cls(
            action_representation=str(payload.get("action_representation", "absolute")),  # type: ignore[arg-type]
            state_normalizer=ZScoreNormalizer.from_dict(payload["state_normalizer"]),  # type: ignore[arg-type]
            action_normalizer=ZScoreNormalizer.from_dict(payload["action_normalizer"]),  # type: ignore[arg-type]
            target_normalizer=ZScoreNormalizer.from_dict(payload["target_normalizer"]),  # type: ignore[arg-type]
        )
        out.validate()
        return out


def _valid_centers(
    episode: LoadedEpisode,
    *,
    chunk_len: int,
    past_len: int,
    state_history_len: int,
    obs_horizon: int,
    state_offset: int,
    image_offset: int,
    action_offset: int,
    stride: int,
) -> range:
    start = max(
        past_len - action_offset,
        state_history_len - 1 - state_offset,
        obs_horizon - 1 - image_offset,
    )
    stop = min(
        episode.length - chunk_len - action_offset + 1,
        episode.length - state_offset,
        episode.length - image_offset,
    )
    return range(int(start), int(stop), int(stride))


def fit_rdt_lite_codecs(
    episodes: list[LoadedEpisode],
    train_ids: list[int],
    *,
    action_representation: ActionRepresentation,
    chunk_len: int,
    past_len: int,
    state_history_len: int,
    obs_horizon: int,
    state_offset: int = 0,
    image_offset: int = 0,
    action_offset: int = 0,
    stride: int = 1,
) -> RDTLiteCodecs:
    if not train_ids:
        raise ValueError("train_ids must be non-empty")
    action_normalizer = ZScoreNormalizer.fit([episodes[index].actions_raw for index in train_ids])
    state_arrays: list[np.ndarray] = []
    for index in train_ids:
        state = episodes[index].states_raw
        if state is None:
            raise ValueError(f"episode {episodes[index].path} has no measured state")
        state_arrays.append(state)
    state_normalizer = ZScoreNormalizer.fit(state_arrays)

    if action_representation == "absolute":
        target_normalizer = action_normalizer
    elif action_representation == "relative_to_current":
        relative_targets: list[np.ndarray] = []
        for index in train_ids:
            episode = episodes[index]
            assert episode.states_raw is not None
            chunks: list[np.ndarray] = []
            for center in _valid_centers(
                episode,
                chunk_len=chunk_len,
                past_len=past_len,
                state_history_len=state_history_len,
                obs_horizon=obs_horizon,
                state_offset=state_offset,
                image_offset=image_offset,
                action_offset=action_offset,
                stride=stride,
            ):
                action_center = center + action_offset
                state_center = center + state_offset
                chunks.append(
                    episode.actions_raw[action_center : action_center + chunk_len]
                    - episode.states_raw[state_center][None]
                )
            if chunks:
                relative_targets.append(np.concatenate(chunks, axis=0).astype(np.float32))
        target_normalizer = ZScoreNormalizer.fit(relative_targets)
    else:  # pragma: no cover - argparse and validate guard this
        raise ValueError(action_representation)

    out = RDTLiteCodecs(
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        target_normalizer=target_normalizer,
        action_representation=action_representation,
    )
    out.validate()
    return out


def apply_rdt_lite_codecs(episodes: list[LoadedEpisode], codecs: RDTLiteCodecs) -> None:
    codecs.validate()
    for episode in episodes:
        if episode.states_raw is None:
            raise ValueError(f"episode {episode.path} has no measured state")
        episode.actions_norm = codecs.action_normalizer.encode(episode.actions_raw)
        episode.states_norm = codecs.state_normalizer.encode(episode.states_raw)
