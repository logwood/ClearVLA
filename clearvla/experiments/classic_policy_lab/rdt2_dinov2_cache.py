from __future__ import annotations

"""Strict mmap-backed DINOv2 dense-token cache for RDT2-FM experiments.

The frozen DINOv2 encoder is expensive but deterministic.  Formal experiments
should encode each decoded frame once, persist per-camera patch tokens, and let
RDT2-FM training read only the dense token cache.
"""

from dataclasses import asdict, dataclass
import json
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


def _episode_dir(cache_dir: Path, episode_stem: str) -> Path:
    return Path(cache_dir) / episode_stem


def _tokens_path(cache_dir: Path, episode_stem: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / "tokens.float16.npy"


def _meta_path(cache_dir: Path, episode_stem: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / "meta.json"


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

    def validate(self) -> None:
        if self.cache_version != DINO_TOKEN_CACHE_VERSION:
            raise ValueError(f"unsupported DINO token cache version={self.cache_version!r}")
        if not self.episode_stem:
            raise ValueError("episode_stem must be non-empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must be non-empty and unique")
        if self.token_dim <= 0 or self.tokens_per_camera <= 0:
            raise ValueError("token_dim and tokens_per_camera must be positive")
        if self.dtype != "float16":
            raise ValueError(f"DINO token cache dtype must be float16, got {self.dtype!r}")
        if not self.dinov2_model:
            raise ValueError("dinov2_model must be non-empty")
        PreprocessConfig.from_dict(dict(self.decoded_preprocessing))
        required = {"resolved_path", "size_bytes", "mtime_ns"}
        if set(self.source_fingerprint) != required:
            raise ValueError(f"source_fingerprint must contain exactly {sorted(required)}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cameras"] = list(self.cameras)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DinoTokenEpisodeMeta":
        out = cls(
            cache_version=str(data["cache_version"]),
            episode_stem=str(data["episode_stem"]),
            num_frames=int(data["num_frames"]),
            cameras=tuple(str(x) for x in data["cameras"]),
            token_dim=int(data["token_dim"]),
            tokens_per_camera=int(data["tokens_per_camera"]),
            dtype=str(data["dtype"]),
            dinov2_model=str(data["dinov2_model"]),
            decoded_preprocessing=dict(data["decoded_preprocessing"]),
            source_fingerprint=dict(data["source_fingerprint"]),
        )
        out.validate()
        return out


class DinoV2TokenStore:
    """Lazy mmap reader for per-frame, per-camera DINOv2 patch tokens.

    Stored array shape per episode: ``[T, Cam, Patch, Dim]``.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        episodes: Sequence[LoadedEpisode],
        camera_names: tuple[str, ...],
        preprocessing: PreprocessConfig,
        dinov2_model: str | None = None,
    ) -> None:
        if not camera_names:
            raise ValueError("camera_names must be non-empty")
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episodes)
        self.camera_names = tuple(camera_names)
        self.preprocessing = preprocessing
        self.dinov2_model = None if dinov2_model is None else str(dinov2_model)
        self._arrays: dict[int, np.ndarray] = {}
        self._meta: dict[int, DinoTokenEpisodeMeta] = {}
        for episode_idx, episode in enumerate(self.episodes):
            self._meta[episode_idx] = self.validate_episode(episode_idx, episode)
        first = self._meta[0]
        self.token_dim = int(first.token_dim)
        self.tokens_per_camera = int(first.tokens_per_camera)
        for meta in self._meta.values():
            if meta.token_dim != self.token_dim or meta.tokens_per_camera != self.tokens_per_camera:
                raise ValueError("DINO token cache shape differs across episodes")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def validate_episode(self, episode_idx: int, episode: LoadedEpisode) -> DinoTokenEpisodeMeta:
        meta_path = _meta_path(self.cache_dir, episode.stem)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"missing DINO token cache metadata: {meta_path}. "
                "Run `python -m clearvla.cli.build_dinov2_token_cache ...` first."
            )
        meta = DinoTokenEpisodeMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.episode_stem != episode.stem:
            raise ValueError(f"DINO token stem mismatch: {meta.episode_stem!r} != {episode.stem!r}")
        if meta.num_frames != episode.length:
            raise ValueError(
                f"DINO token frame count mismatch for {episode.stem}: {meta.num_frames} != {episode.length}"
            )
        if meta.cameras != self.camera_names:
            raise ValueError(f"DINO token cameras mismatch: {meta.cameras} != {self.camera_names}")
        if meta.decoded_preprocessing != self.preprocessing.to_dict():
            raise ValueError(
                f"DINO token preprocessing mismatch: cached={meta.decoded_preprocessing}, "
                f"requested={self.preprocessing.to_dict()}"
            )
        if self.dinov2_model is not None and meta.dinov2_model != self.dinov2_model:
            raise ValueError(
                f"DINO model mismatch: cached={meta.dinov2_model!r}, requested={self.dinov2_model!r}"
            )
        expected_fingerprint = _source_fingerprint(episode.path)
        if meta.source_fingerprint != expected_fingerprint:
            raise ValueError(
                f"DINO token source fingerprint mismatch for {episode.path}; "
                "the source HDF5 changed after token-cache construction"
            )
        path = _tokens_path(self.cache_dir, episode.stem)
        if not path.exists():
            raise FileNotFoundError(f"missing DINO token mmap: {path}")
        array = np.load(path, mmap_mode="r")
        expected = (episode.length, len(self.camera_names), meta.tokens_per_camera, meta.token_dim)
        if tuple(array.shape) != expected or array.dtype != np.float16:
            raise ValueError(
                f"invalid DINO token mmap {path}: shape={array.shape}, dtype={array.dtype}; expected={expected}, float16"
            )
        return meta

    def _array(self, episode_idx: int) -> np.ndarray:
        idx = int(episode_idx)
        if idx < 0 or idx >= len(self.episodes):
            raise IndexError(f"episode_idx={idx} outside [0,{len(self.episodes)})")
        if idx not in self._arrays:
            self._arrays[idx] = np.load(
                _tokens_path(self.cache_dir, self.episodes[idx].stem), mmap_mode="r"
            )
        return self._arrays[idx]

    def load_batch(self, sample_keys: Tensor | Sequence[Sequence[int]]) -> Tensor:
        """Load a batch of frame tokens with episode-grouped mmap reads.

        Earlier versions performed one Python-level mmap slice per requested
        frame.  V38.3 groups requests by episode and uses vectorized frame
        indexing, which substantially reduces Python overhead and small random
        reads when a batch contains repeated episode ids.  The output order is
        exactly the input order.
        """
        keys = torch.as_tensor(sample_keys, dtype=torch.long).cpu().numpy()
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(f"sample_keys must be [B,2] episode/frame pairs, got {keys.shape}")
        n = int(keys.shape[0])
        if n == 0:
            empty = np.empty(
                (0, len(self.camera_names), self.tokens_per_camera, self.token_dim),
                dtype=np.float16,
            )
            return torch.from_numpy(empty)
        out = np.empty(
            (n, len(self.camera_names), self.tokens_per_camera, self.token_dim), dtype=np.float16
        )
        episode_ids = keys[:, 0].astype(np.int64, copy=False)
        frame_ids = keys[:, 1].astype(np.int64, copy=False)
        for episode_idx in np.unique(episode_ids):
            positions = np.flatnonzero(episode_ids == episode_idx)
            array = self._array(int(episode_idx))
            frames = frame_ids[positions]
            if frames.size and (int(frames.min()) < 0 or int(frames.max()) >= array.shape[0]):
                bad = int(frames[(frames < 0) | (frames >= array.shape[0])][0])
                raise IndexError(
                    f"frame_idx={bad} outside episode {episode_idx} length={array.shape[0]}"
                )
            out[positions] = np.asarray(array[frames], dtype=np.float16)
        return torch.from_numpy(np.ascontiguousarray(out))


def load_episode_token_meta(cache_dir: Path, episode_stem: str) -> DinoTokenEpisodeMeta:
    """Read and validate one persisted metadata record."""
    path = _meta_path(Path(cache_dir), str(episode_stem))
    if not path.exists():
        raise FileNotFoundError(f"missing DINO token cache metadata: {path}")
    return DinoTokenEpisodeMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))


def episode_tokens_exist(cache_dir: Path, episode_stem: str) -> bool:
    return (
        _meta_path(Path(cache_dir), str(episode_stem)).exists()
        and _tokens_path(Path(cache_dir), str(episode_stem)).exists()
    )


def save_episode_tokens(
    *,
    cache_dir: Path,
    episode: LoadedEpisode,
    camera_names: tuple[str, ...],
    preprocessing: PreprocessConfig,
    dinov2_model: str,
    tokens: np.ndarray,
    rebuild: bool = False,
) -> DinoTokenEpisodeMeta:
    """Persist one token array with a strict metadata contract."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [T,Cam,Patch,Dim], got {tokens.shape}")
    if tokens.shape[0] != episode.length or tokens.shape[1] != len(camera_names):
        raise ValueError("token episode/camera dimensions do not match source episode")
    episode_root = _episode_dir(Path(cache_dir), episode.stem)
    meta_path = _meta_path(Path(cache_dir), episode.stem)
    path = _tokens_path(Path(cache_dir), episode.stem)
    if meta_path.exists() and path.exists() and not rebuild:
        return DinoTokenEpisodeMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    episode_root.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=tokens.shape)
    mmap[:] = tokens.astype(np.float16, copy=False)
    mmap.flush()
    del mmap
    meta = DinoTokenEpisodeMeta(
        cache_version=DINO_TOKEN_CACHE_VERSION,
        episode_stem=episode.stem,
        num_frames=episode.length,
        cameras=tuple(camera_names),
        token_dim=int(tokens.shape[3]),
        tokens_per_camera=int(tokens.shape[2]),
        dtype="float16",
        dinov2_model=str(dinov2_model),
        decoded_preprocessing=preprocessing.to_dict(),
        source_fingerprint=_source_fingerprint(episode.path),
    )
    meta.validate()
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return meta


__all__ = [
    "DINO_TOKEN_CACHE_VERSION",
    "DinoTokenEpisodeMeta",
    "DinoV2TokenStore",
    "load_episode_token_meta",
    "episode_tokens_exist",
    "save_episode_tokens",
]
