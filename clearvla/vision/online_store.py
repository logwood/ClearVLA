from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.image_io import decode_image_value
from clearvla.vision.preprocessing import PreprocessConfig, apply_preprocess


class OnlineVisualStore:
    """Online pixel provider for end-to-end trainable visual stems.

    The provider has two explicit modes:

    1. mmap decoded-image cache when ``decoded_cache_dir`` is supplied;
    2. direct HDF5 fallback with bounded process-local frame LRU and bounded
       process-local HDF5 handle cache.

    The returned tensors are camera-local ``[H,3,Ih,Iw]`` arrays. Camera
    resolutions may differ when preprocessing preserves native resolution.
    """

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...],
        preprocessing: PreprocessConfig,
        decoded_cache_dir: Path | None = None,
        frame_lru_capacity: int = 512,
        open_file_capacity: int = 8,
    ) -> None:
        if not camera_names:
            raise ValueError("OnlineVisualStore requires at least one camera")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError(f"Duplicate camera names: {camera_names}")
        if frame_lru_capacity < 0:
            raise ValueError("frame_lru_capacity must be non-negative")
        if open_file_capacity <= 0:
            raise ValueError("open_file_capacity must be positive")
        self.camera_names = tuple(camera_names)
        self.preprocessing = preprocessing
        self.decoded_cache_dir = None if decoded_cache_dir is None else Path(decoded_cache_dir)
        self.frame_lru_capacity = int(frame_lru_capacity)
        self.open_file_capacity = int(open_file_capacity)
        self._decoded_store = (
            None
            if self.decoded_cache_dir is None
            else DecodedImageStore(
                self.decoded_cache_dir,
                camera_names=self.camera_names,
                preprocessing=self.preprocessing,
            )
        )
        self._pid = os.getpid()
        self._frame_lru: OrderedDict[tuple[str, str, int], np.ndarray] = OrderedDict()
        self._h5_lru: OrderedDict[str, h5py.File] = OrderedDict()
        self._stats = {
            "decoded_mmap_windows": 0,
            "hdf5_windows": 0,
            "frame_lru_hits": 0,
            "frame_lru_misses": 0,
            "hdf5_file_opens": 0,
        }

    @property
    def uses_decoded_cache(self) -> bool:
        return self._decoded_store is not None

    def validate_episode(self, episode: LoadedEpisode) -> None:
        if self._decoded_store is not None:
            self._decoded_store.validate_episode(episode)

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        self.close()
        self._pid = pid
        self._frame_lru = OrderedDict()
        self._h5_lru = OrderedDict()
        self._stats = {
            "decoded_mmap_windows": 0,
            "hdf5_windows": 0,
            "frame_lru_hits": 0,
            "frame_lru_misses": 0,
            "hdf5_file_opens": 0,
        }

    def close(self) -> None:
        for handle in self._h5_lru.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_lru.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_frame_lru"] = OrderedDict()
        state["_h5_lru"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def stats(self) -> dict[str, int | bool]:
        return {**self._stats, "uses_decoded_cache": self.uses_decoded_cache}

    def _h5(self, path: Path) -> h5py.File:
        self._ensure_process_local()
        key = str(path.resolve())
        handle = self._h5_lru.pop(key, None)
        if handle is None:
            handle = h5py.File(path, "r")
            self._stats["hdf5_file_opens"] += 1
        self._h5_lru[key] = handle
        while len(self._h5_lru) > self.open_file_capacity:
            _, old = self._h5_lru.popitem(last=False)
            old.close()
        return handle

    def _direct_frame(self, episode: LoadedEpisode, camera: str, idx: int) -> np.ndarray:
        self._ensure_process_local()
        key = (str(episode.path.resolve()), camera, int(idx))
        cached = self._frame_lru.pop(key, None)
        if cached is not None:
            self._frame_lru[key] = cached
            self._stats["frame_lru_hits"] += 1
            return cached
        self._stats["frame_lru_misses"] += 1
        handle = self._h5(episode.path)
        try:
            dataset = handle[episode.camera_keys[camera]]
        except KeyError as exc:
            raise KeyError(f"{episode.path}: unresolved camera={camera!r}") from exc
        frame = apply_preprocess(decode_image_value(dataset[int(idx)]), self.preprocessing)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.frame_lru_capacity > 0:
            self._frame_lru[key] = frame
            while len(self._frame_lru) > self.frame_lru_capacity:
                self._frame_lru.popitem(last=False)
        return frame

    def load_window(self, episode: LoadedEpisode, indices: np.ndarray) -> dict[str, torch.Tensor]:
        """Load one temporal window as camera-local tensors ``[H,3,Ih,Iw]``."""
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"indices must be non-empty [H], got shape={indices.shape}")
        if int(indices.min()) < 0 or int(indices.max()) >= episode.length:
            raise IndexError(
                f"indices range [{int(indices.min())},{int(indices.max())}] outside episode length={episode.length}"
            )
        if self._decoded_store is not None:
            self._stats["decoded_mmap_windows"] += 1
            return self._decoded_store.load_window(episode, indices)

        self._stats["hdf5_windows"] += 1
        output: dict[str, torch.Tensor] = {}
        for camera in self.camera_names:
            frames = [self._direct_frame(episode, camera, int(idx)) for idx in indices]
            shapes = {tuple(frame.shape) for frame in frames}
            if len(shapes) != 1:
                raise ValueError(
                    f"{episode.path}: camera={camera!r} frames have inconsistent HWC shapes={sorted(shapes)}. "
                    "Configure explicit preprocessing."
                )
            array = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
            output[camera] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.uint8))
        return output
