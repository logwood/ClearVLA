from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.image_io import decode_image_value
from clearvla.vision.preprocessing import PreprocessConfig, apply_preprocess


DECODED_IMAGE_CACHE_VERSION = "decoded-image-v1"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _camera_filename(camera: str) -> str:
    if not camera or any(ch in camera for ch in "/\\"):
        raise ValueError(f"Invalid camera name={camera!r}")
    return f"{camera}.uint8.npy"


@dataclass(frozen=True)
class DecodedImageEpisodeMeta:
    cache_version: str
    episode_stem: str
    num_frames: int
    cameras: tuple[str, ...]
    camera_keys: dict[str, str]
    camera_shapes_hwc: dict[str, tuple[int, int, int]]
    dtype: str
    preprocessing: dict[str, Any]
    source_fingerprint: dict[str, Any]

    def validate(self) -> None:
        if self.cache_version != DECODED_IMAGE_CACHE_VERSION:
            raise ValueError(
                f"Unsupported decoded-image cache version={self.cache_version!r}; "
                f"expected={DECODED_IMAGE_CACHE_VERSION!r}"
            )
        if not self.episode_stem:
            raise ValueError("episode_stem must be non-empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not self.cameras:
            raise ValueError("decoded-image cache requires at least one camera")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"duplicate cameras={self.cameras}")
        if set(self.camera_keys) != set(self.cameras):
            raise ValueError("camera_keys do not match cameras")
        if set(self.camera_shapes_hwc) != set(self.cameras):
            raise ValueError("camera_shapes_hwc do not match cameras")
        for camera, shape in self.camera_shapes_hwc.items():
            if len(shape) != 3 or int(shape[2]) != 3 or min(int(x) for x in shape) <= 0:
                raise ValueError(f"camera={camera!r} invalid HWC shape={shape}")
        if self.dtype != "uint8":
            raise ValueError(f"decoded-image cache dtype must be uint8, got {self.dtype!r}")
        PreprocessConfig.from_dict(dict(self.preprocessing))
        required_fingerprint = {"resolved_path", "size_bytes", "mtime_ns"}
        if set(self.source_fingerprint) != required_fingerprint:
            raise ValueError(f"source_fingerprint must contain exactly {sorted(required_fingerprint)}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cameras"] = list(self.cameras)
        out["camera_shapes_hwc"] = {
            camera: list(shape) for camera, shape in sorted(self.camera_shapes_hwc.items())
        }
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodedImageEpisodeMeta":
        out = cls(
            cache_version=str(data["cache_version"]),
            episode_stem=str(data["episode_stem"]),
            num_frames=int(data["num_frames"]),
            cameras=tuple(str(x) for x in data["cameras"]),
            camera_keys={str(k): str(v) for k, v in dict(data["camera_keys"]).items()},
            camera_shapes_hwc={
                str(camera): (int(shape[0]), int(shape[1]), int(shape[2]))
                for camera, shape in dict(data["camera_shapes_hwc"]).items()
            },
            dtype=str(data["dtype"]),
            preprocessing=dict(data["preprocessing"]),
            source_fingerprint=dict(data["source_fingerprint"]),
        )
        out.validate()
        return out


def _episode_dir(cache_dir: Path, episode_stem: str) -> Path:
    return cache_dir / episode_stem


def _meta_path(cache_dir: Path, episode_stem: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / "meta.json"


def build_episode_decoded_cache(
    episode: LoadedEpisode,
    *,
    cache_dir: Path,
    camera_names: tuple[str, ...],
    preprocessing: PreprocessConfig,
    rebuild: bool = False,
) -> DecodedImageEpisodeMeta:
    """Decode one HDF5 episode once into per-camera native/preprocessed uint8 mmap arrays."""
    if not camera_names:
        raise ValueError("camera_names must be non-empty")
    episode_root = _episode_dir(cache_dir, episode.stem)
    meta_path = _meta_path(cache_dir, episode.stem)
    if meta_path.exists() and not rebuild:
        store = DecodedImageStore(cache_dir, camera_names=camera_names, preprocessing=preprocessing)
        return store.validate_episode(episode)

    episode_root.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, tuple[int, int, int]] = {}
    arrays: dict[str, np.memmap] = {}
    with h5py.File(episode.path, "r") as f:
        for camera in camera_names:
            if camera not in episode.camera_keys:
                raise KeyError(f"{episode.path}: unresolved camera={camera!r}")
            dataset = f[episode.camera_keys[camera]]
            first = apply_preprocess(decode_image_value(dataset[0]), preprocessing)
            shape = (int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))
            if shape[2] != 3:
                raise ValueError(f"camera={camera!r} expected RGB HWC, got {shape}")
            shapes[camera] = shape
            path = episode_root / _camera_filename(camera)
            mm = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.uint8,
                shape=(episode.length, *shape),
            )
            mm[0] = first
            arrays[camera] = mm
            for idx in range(1, episode.length):
                frame = apply_preprocess(decode_image_value(dataset[idx]), preprocessing)
                if tuple(frame.shape) != shape:
                    raise ValueError(
                        f"{episode.path}: camera={camera!r} frame={idx} shape={tuple(frame.shape)} "
                        f"!= first frame shape={shape}. Configure explicit preprocessing."
                    )
                mm[idx] = frame
            mm.flush()
            del arrays[camera]

    meta = DecodedImageEpisodeMeta(
        cache_version=DECODED_IMAGE_CACHE_VERSION,
        episode_stem=episode.stem,
        num_frames=episode.length,
        cameras=tuple(camera_names),
        camera_keys={camera: episode.camera_keys[camera] for camera in camera_names},
        camera_shapes_hwc=shapes,
        dtype="uint8",
        preprocessing=preprocessing.to_dict(),
        source_fingerprint=_source_fingerprint(episode.path),
    )
    meta.validate()
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return meta


def build_all_decoded_caches(
    episodes: list[LoadedEpisode],
    *,
    cache_dir: Path,
    camera_names: tuple[str, ...],
    preprocessing: PreprocessConfig,
    rebuild: bool = False,
) -> list[DecodedImageEpisodeMeta]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for episode in episodes:
        meta = build_episode_decoded_cache(
            episode,
            cache_dir=cache_dir,
            camera_names=camera_names,
            preprocessing=preprocessing,
            rebuild=rebuild,
        )
        metas.append(meta)
    return metas


class DecodedImageStore:
    """Strict mmap-backed decoded-image store.

    Each camera is stored separately, so native camera resolutions may differ.
    The store validates source HDF5 fingerprint, camera keys, frame count and
    preprocessing before returning a frame.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        camera_names: tuple[str, ...],
        preprocessing: PreprocessConfig,
    ) -> None:
        if not camera_names:
            raise ValueError("camera_names must be non-empty")
        self.cache_dir = Path(cache_dir)
        self.camera_names = tuple(camera_names)
        self.preprocessing = preprocessing
        self._meta: dict[str, DecodedImageEpisodeMeta] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    def validate_episode(self, episode: LoadedEpisode) -> DecodedImageEpisodeMeta:
        meta_path = _meta_path(self.cache_dir, episode.stem)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing decoded-image cache metadata: {meta_path}. "
                "Run `python -m clearvla.cli.build_decoded_image_cache ...` first."
            )
        meta = DecodedImageEpisodeMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.episode_stem != episode.stem:
            raise ValueError(f"decoded-image stem mismatch: {meta.episode_stem!r} != {episode.stem!r}")
        if meta.num_frames != episode.length:
            raise ValueError(f"decoded-image frame count mismatch: {meta.num_frames} != {episode.length}")
        if meta.cameras != self.camera_names:
            raise ValueError(f"decoded-image cameras mismatch: {meta.cameras} != {self.camera_names}")
        if meta.preprocessing != self.preprocessing.to_dict():
            raise ValueError(
                f"decoded-image preprocessing mismatch: cached={meta.preprocessing}, "
                f"requested={self.preprocessing.to_dict()}"
            )
        expected_keys = {camera: episode.camera_keys[camera] for camera in self.camera_names}
        if meta.camera_keys != expected_keys:
            raise ValueError(f"decoded-image camera key mismatch: cached={meta.camera_keys}, expected={expected_keys}")
        fingerprint = _source_fingerprint(episode.path)
        if meta.source_fingerprint != fingerprint:
            raise ValueError(
                f"decoded-image source fingerprint mismatch for {episode.path}; "
                "the source HDF5 changed after cache construction"
            )
        for camera in self.camera_names:
            path = _episode_dir(self.cache_dir, episode.stem) / _camera_filename(camera)
            if not path.exists():
                raise FileNotFoundError(f"Missing decoded-image mmap: {path}")
            array = np.load(path, mmap_mode="r")
            expected_shape = (episode.length, *meta.camera_shapes_hwc[camera])
            if tuple(array.shape) != expected_shape or array.dtype != np.uint8:
                raise ValueError(
                    f"Invalid decoded-image mmap {path}: shape={tuple(array.shape)}, dtype={array.dtype}; "
                    f"expected={expected_shape}, uint8"
                )
        self._meta[episode.stem] = meta
        return meta

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        state["_meta"] = {}
        return state

    def _array(self, episode: LoadedEpisode, camera: str) -> np.ndarray:
        key = (episode.stem, camera)
        if key not in self._arrays:
            if episode.stem not in self._meta:
                self.validate_episode(episode)
            path = _episode_dir(self.cache_dir, episode.stem) / _camera_filename(camera)
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def load_window(self, episode: LoadedEpisode, indices: np.ndarray) -> dict[str, torch.Tensor]:
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"indices must be non-empty [H], got shape={indices.shape}")
        if int(indices.min()) < 0 or int(indices.max()) >= episode.length:
            raise IndexError(
                f"indices range [{int(indices.min())},{int(indices.max())}] outside episode length={episode.length}"
            )
        output: dict[str, torch.Tensor] = {}
        for camera in self.camera_names:
            # mmap slice -> private contiguous buffer [H,Hc,Wc,3]
            frames = np.asarray(self._array(episode, camera)[indices], dtype=np.uint8)
            chw = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
            output[camera] = torch.from_numpy(chw)
        return output
