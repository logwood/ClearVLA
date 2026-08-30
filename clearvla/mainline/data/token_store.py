"""Read-only mmap access to the dense DINO token cache used by mainline.

Cache construction remains a data-preparation utility.  Training owns only a
strict reader so the active graph does not import an old experiment package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.preprocessing import PreprocessConfig

DINO_TOKEN_CACHE_VERSION = "rdt2-dinov2-dense-token-v1"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _episode_dir(cache_dir: Path, episode_key: str) -> Path:
    return cache_dir / episode_key


def _tokens_path(cache_dir: Path, episode_key: str) -> Path:
    return _episode_dir(cache_dir, episode_key) / "tokens.float16.npy"


def _meta_path(cache_dir: Path, episode_key: str) -> Path:
    return _episode_dir(cache_dir, episode_key) / "meta.json"


@dataclass(frozen=True)
class DinoTokenEpisodeMeta:
    cache_version: str
    episode_stem: str
    num_frames: int
    cameras: tuple[str, ...]
    token_dim: int
    tokens_per_camera: int
    dtype: str
    dinov2_model: str
    decoded_preprocessing: dict[str, Any]
    source_fingerprint: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "DinoTokenEpisodeMeta":
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = cls(
            cache_version=str(payload["cache_version"]),
            episode_stem=str(payload["episode_stem"]),
            num_frames=int(payload["num_frames"]),
            cameras=tuple(str(value) for value in payload["cameras"]),
            token_dim=int(payload["token_dim"]),
            tokens_per_camera=int(payload["tokens_per_camera"]),
            dtype=str(payload["dtype"]),
            dinov2_model=str(payload["dinov2_model"]),
            decoded_preprocessing=dict(payload["decoded_preprocessing"]),
            source_fingerprint=dict(payload["source_fingerprint"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.cache_version != DINO_TOKEN_CACHE_VERSION:
            raise ValueError(f"unsupported DINO cache version {self.cache_version!r}")
        if self.num_frames <= 0 or self.token_dim <= 0 or self.tokens_per_camera <= 0:
            raise ValueError("DINO cache dimensions must be positive")
        if not self.episode_stem or not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("DINO cache episode/camera identity is invalid")
        if self.dtype != "float16":
            raise ValueError("DINO cache storage dtype must be float16")
        if not self.dinov2_model:
            raise ValueError("DINO cache model identity is missing")
        PreprocessConfig.from_dict(self.decoded_preprocessing)
        if set(self.source_fingerprint) != {"resolved_path", "size_bytes", "mtime_ns"}:
            raise ValueError("DINO cache source fingerprint is incomplete")


class DinoV2TokenStore:
    """Lazy, episode-grouped mmap reader for ``[T,C,P,D]`` token arrays."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        episodes: Sequence[LoadedEpisode],
        camera_names: tuple[str, ...],
        preprocessing: PreprocessConfig,
        dinov2_model: str,
    ) -> None:
        if not episodes:
            raise ValueError("DINO token store requires at least one episode")
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episodes)
        self.camera_names = tuple(camera_names)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("requested DINO cameras must be non-empty and unique")
        self.preprocessing = preprocessing
        self.dinov2_model = str(dinov2_model)
        self._arrays: dict[int, np.ndarray] = {}
        self._meta = {
            index: self._validate_episode(index, episode)
            for index, episode in enumerate(self.episodes)
        }
        first = self._meta[0]
        self.storage_camera_names = first.cameras
        self._camera_indices = tuple(
            self.storage_camera_names.index(camera) for camera in self.camera_names
        )
        self.token_dim = first.token_dim
        self.tokens_per_camera = first.tokens_per_camera
        for meta in self._meta.values():
            if meta.cameras != self.storage_camera_names:
                raise ValueError("DINO cache camera inventory differs across episodes")
            if (meta.token_dim, meta.tokens_per_camera) != (
                self.token_dim,
                self.tokens_per_camera,
            ):
                raise ValueError("DINO cache geometry differs across episodes")

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _validate_episode(self, episode_idx: int, episode: LoadedEpisode) -> DinoTokenEpisodeMeta:
        del episode_idx
        meta_path = _meta_path(self.cache_dir, episode.cache_key)
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing DINO token metadata: {meta_path}")
        meta = DinoTokenEpisodeMeta.from_path(meta_path)
        if meta.episode_stem != episode.cache_key or meta.num_frames != episode.length:
            raise ValueError(f"DINO cache episode mismatch for {episode.cache_key}")
        missing_cameras = [camera for camera in self.camera_names if camera not in meta.cameras]
        if missing_cameras:
            raise ValueError(
                f"DINO cache cameras {meta.cameras} do not contain {missing_cameras}"
            )
        if meta.decoded_preprocessing != self.preprocessing.to_dict():
            raise ValueError("DINO and decoded-image preprocessing do not match")
        if meta.dinov2_model != self.dinov2_model:
            raise ValueError(f"DINO cache model {meta.dinov2_model!r} != {self.dinov2_model!r}")
        if meta.source_fingerprint != _source_fingerprint(episode.path):
            raise ValueError(f"source HDF5 changed after caching: {episode.path}")
        token_path = _tokens_path(self.cache_dir, episode.cache_key)
        if not token_path.is_file():
            raise FileNotFoundError(f"missing DINO token array: {token_path}")
        array = np.load(token_path, mmap_mode="r")
        expected = (
            episode.length,
            len(meta.cameras),
            meta.tokens_per_camera,
            meta.token_dim,
        )
        if tuple(array.shape) != expected or array.dtype != np.float16:
            raise ValueError(
                f"invalid DINO cache {token_path}: {array.shape}/{array.dtype}, expected {expected}/float16"
            )
        return meta

    def _array(self, episode_idx: int) -> np.ndarray:
        index = int(episode_idx)
        if not 0 <= index < len(self.episodes):
            raise IndexError(f"episode index {index} is outside the token store")
        if index not in self._arrays:
            self._arrays[index] = np.load(
                _tokens_path(self.cache_dir, self.episodes[index].cache_key),
                mmap_mode="r",
            )
        return self._arrays[index]

    def load_batch(self, sample_keys: Tensor | Sequence[Sequence[int]]) -> Tensor:
        keys = torch.as_tensor(sample_keys, dtype=torch.long).cpu().numpy()
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(f"sample keys must be [N,2], got {keys.shape}")
        result = np.empty(
            (
                len(keys),
                len(self.camera_names),
                self.tokens_per_camera,
                self.token_dim,
            ),
            dtype=np.float16,
        )
        episode_ids = keys[:, 0].astype(np.int64, copy=False)
        frame_ids = keys[:, 1].astype(np.int64, copy=False)
        for episode_idx in np.unique(episode_ids):
            positions = np.flatnonzero(episode_ids == episode_idx)
            array = self._array(int(episode_idx))
            frames = frame_ids[positions]
            if frames.size and (int(frames.min()) < 0 or int(frames.max()) >= len(array)):
                raise IndexError(f"DINO frame index outside episode {episode_idx}")
            cached = np.asarray(array[frames], dtype=np.float16)
            result[positions] = cached[:, self._camera_indices]
        return torch.from_numpy(np.ascontiguousarray(result))


__all__ = ["DINO_TOKEN_CACHE_VERSION", "DinoTokenEpisodeMeta", "DinoV2TokenStore"]
