from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.online_store import OnlineVisualStore
from .teacher import PatchTeacherConfig, PatchTeacherLike


VISION_LATENT_CACHE_VERSION = "vision-usage-latent-v2"
_SUPPORTED_CACHE_VERSIONS = {"vision-usage-latent-v1", VISION_LATENT_CACHE_VERSION}
_COMPLETE_MARKER = "COMPLETE"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _teacher_fingerprint(config: PatchTeacherConfig) -> str:
    """Record cache provenance at build time.

    This is deliberately provenance, not a runtime dependency check.  Training
    consumes static latent arrays and must not fail because a local teacher
    repository was later moved or edited.
    """
    payload = config.to_dict()
    if config.torch_hub_source == "local" and config.local_repo_dir:
        repo = Path(config.local_repo_dir).expanduser().resolve()
        payload["local_repo_fingerprint"] = {
            "resolved_path": str(repo),
            "exists": repo.exists(),
            "mtime_ns": int(repo.stat().st_mtime_ns) if repo.exists() else None,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_camera(camera: str) -> str:
    if not camera or any(ch in camera for ch in "/\\"):
        raise ValueError(f"invalid camera name={camera!r}")
    return camera


def _episode_dir(cache_dir: Path, episode_stem: str) -> Path:
    return cache_dir / episode_stem


def _meta_path(cache_dir: Path, episode_stem: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / "meta.json"


def _complete_path(cache_dir: Path, episode_stem: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / _COMPLETE_MARKER


def _tokens_path(cache_dir: Path, episode_stem: str, camera: str) -> Path:
    return _episode_dir(cache_dir, episode_stem) / f"{_safe_camera(camera)}.tokens.float16.npy"


def _tokens_path_in_root(root: Path, camera: str) -> Path:
    return root / f"{_safe_camera(camera)}.tokens.float16.npy"


@dataclass(frozen=True)
class VisionLatentEpisodeMeta:
    cache_version: str
    episode_stem: str
    num_frames: int
    cameras: tuple[str, ...]
    patch_grid: tuple[int, int]
    patch_count: int
    token_dim: int
    dtype: str
    teacher_config: dict[str, Any]
    teacher_fingerprint: str
    source_fingerprint: dict[str, Any]

    def validate(self) -> None:
        if self.cache_version not in _SUPPORTED_CACHE_VERSIONS:
            raise ValueError(
                f"unsupported latent cache version={self.cache_version!r}; "
                f"supported={sorted(_SUPPORTED_CACHE_VERSIONS)!r}"
            )
        if not self.episode_stem:
            raise ValueError("episode_stem must be non-empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"invalid cameras={self.cameras}")
        gh, gw = int(self.patch_grid[0]), int(self.patch_grid[1])
        if gh <= 0 or gw <= 0 or int(self.patch_count) != gh * gw:
            raise ValueError(f"invalid patch grid/count: grid={self.patch_grid}, count={self.patch_count}")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if self.dtype != "float16":
            raise ValueError(f"latent cache dtype must be float16, got {self.dtype!r}")
        # Keep the build-time teacher description as provenance, but only check
        # shape-bearing resolved values.  Do not recompute a runtime fingerprint.
        config = PatchTeacherConfig.from_dict(dict(self.teacher_config))
        if config.patch_grid() != self.patch_grid:
            raise ValueError(f"teacher patch grid mismatch: {config.patch_grid()} != {self.patch_grid}")
        if config.resolved_token_dim() != self.token_dim:
            raise ValueError(f"teacher token dim mismatch: {config.resolved_token_dim()} != {self.token_dim}")
        if not self.teacher_fingerprint:
            raise ValueError("teacher_fingerprint must be non-empty")
        required_source = {"resolved_path", "size_bytes", "mtime_ns"}
        if set(self.source_fingerprint) != required_source:
            raise ValueError(f"source_fingerprint must contain exactly {sorted(required_source)}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cameras"] = list(self.cameras)
        out["patch_grid"] = list(self.patch_grid)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionLatentEpisodeMeta":
        out = cls(
            cache_version=str(data["cache_version"]),
            episode_stem=str(data["episode_stem"]),
            num_frames=int(data["num_frames"]),
            cameras=tuple(str(x) for x in data["cameras"]),
            patch_grid=(int(data["patch_grid"][0]), int(data["patch_grid"][1])),
            patch_count=int(data["patch_count"]),
            token_dim=int(data["token_dim"]),
            dtype=str(data["dtype"]),
            teacher_config=dict(data["teacher_config"]),
            teacher_fingerprint=str(data["teacher_fingerprint"]),
            source_fingerprint=dict(data["source_fingerprint"]),
        )
        out.validate()
        return out


def _write_episode_cache_atomic(
    episode: LoadedEpisode,
    *,
    cache_dir: Path,
    camera_names: tuple[str, ...],
    teacher: PatchTeacherLike,
    teacher_config: PatchTeacherConfig,
    visual_store: OnlineVisualStore,
    device: torch.device,
    batch_frames: int,
) -> VisionLatentEpisodeMeta:
    expected_grid = teacher_config.patch_grid()
    expected_dim = teacher_config.resolved_token_dim()
    patch_count = expected_grid[0] * expected_grid[1]
    root = _episode_dir(cache_dir, episode.stem)
    tmp_root = cache_dir / f".{episode.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    backup_root = cache_dir / f".{episode.stem}.backup-{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    try:
        for camera in camera_names:
            tokens_mm = np.lib.format.open_memmap(
                _tokens_path_in_root(tmp_root, camera),
                mode="w+",
                dtype=np.float16,
                shape=(episode.length, patch_count, expected_dim),
            )
            for start in range(0, episode.length, batch_frames):
                stop = min(start + batch_frames, episode.length)
                indices = np.arange(start, stop, dtype=np.int64)
                frames = visual_store.load_window(episode, indices)[camera].to(device=device, non_blocking=True)
                tokens, _ = teacher.encode(frames)
                if tuple(tokens.shape) != (stop - start, patch_count, expected_dim):
                    raise RuntimeError(
                        f"camera={camera!r} teacher tokens shape={tuple(tokens.shape)} "
                        f"!= {(stop-start, patch_count, expected_dim)}"
                    )
                if not torch.isfinite(tokens).all():
                    raise FloatingPointError(f"camera={camera!r} teacher returned non-finite latent")
                tokens_mm[start:stop] = tokens.detach().to(dtype=torch.float16, device="cpu").numpy()
            tokens_mm.flush()
            del tokens_mm

        meta = VisionLatentEpisodeMeta(
            cache_version=VISION_LATENT_CACHE_VERSION,
            episode_stem=episode.stem,
            num_frames=episode.length,
            cameras=tuple(camera_names),
            patch_grid=expected_grid,
            patch_count=patch_count,
            token_dim=expected_dim,
            dtype="float16",
            teacher_config=teacher_config.to_dict(),
            teacher_fingerprint=_teacher_fingerprint(teacher_config),
            source_fingerprint=_source_fingerprint(episode.path),
        )
        meta.validate()
        (tmp_root / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        (tmp_root / _COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")

        if root.exists():
            root.replace(backup_root)
        tmp_root.replace(root)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        return meta
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        if backup_root.exists() and not root.exists():
            backup_root.replace(root)
        raise


@torch.inference_mode()
def build_episode_vision_latent_cache(
    episode: LoadedEpisode,
    *,
    cache_dir: Path,
    camera_names: tuple[str, ...],
    teacher: PatchTeacherLike,
    teacher_config: PatchTeacherConfig,
    visual_store: OnlineVisualStore,
    device: torch.device,
    batch_frames: int = 32,
    rebuild: bool = False,
) -> VisionLatentEpisodeMeta:
    """Encode one episode into strict frame-local patch-token mmap arrays."""
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    if not camera_names:
        raise ValueError("camera_names must be non-empty")
    teacher_config.validate()
    expected_grid = teacher_config.patch_grid()
    expected_dim = teacher_config.resolved_token_dim()
    if tuple(teacher.patch_grid) != expected_grid:
        raise ValueError(f"teacher.patch_grid={teacher.patch_grid} != config={expected_grid}")
    if int(teacher.token_dim) != expected_dim:
        raise ValueError(f"teacher.token_dim={teacher.token_dim} != config={expected_dim}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path(cache_dir, episode.stem)
    if meta_path.exists() and not rebuild:
        meta = VisionLatentCacheStore(cache_dir, camera_names=camera_names).validate_episode(episode)
        if meta.patch_grid != expected_grid or meta.token_dim != expected_dim:
            raise ValueError(
                "existing latent cache shape is incompatible with requested teacher: "
                f"cached grid/dim={meta.patch_grid}/{meta.token_dim}, requested={expected_grid}/{expected_dim}; "
                "pass --rebuild to replace it"
            )
        return meta
    return _write_episode_cache_atomic(
        episode,
        cache_dir=cache_dir,
        camera_names=camera_names,
        teacher=teacher,
        teacher_config=teacher_config,
        visual_store=visual_store,
        device=device,
        batch_frames=batch_frames,
    )


@torch.inference_mode()
def build_all_vision_latent_caches(
    episodes: list[LoadedEpisode],
    *,
    cache_dir: Path,
    camera_names: tuple[str, ...],
    teacher: PatchTeacherLike,
    teacher_config: PatchTeacherConfig,
    visual_store: OnlineVisualStore,
    device: torch.device,
    batch_frames: int = 32,
    rebuild: bool = False,
) -> list[VisionLatentEpisodeMeta]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        build_episode_vision_latent_cache(
            episode,
            cache_dir=cache_dir,
            camera_names=camera_names,
            teacher=teacher,
            teacher_config=teacher_config,
            visual_store=visual_store,
            device=device,
            batch_frames=batch_frames,
            rebuild=rebuild,
        )
        for episode in episodes
    ]


class VisionLatentCacheStore:
    """Strict mmap store for frozen teacher patch tokens.

    The store validates static cache assets.  It intentionally does not require
    the original teacher runtime configuration: training consumes cached arrays,
    not the teacher implementation.
    """

    def __init__(self, cache_dir: Path, *, camera_names: tuple[str, ...]) -> None:
        if not camera_names:
            raise ValueError("camera_names must be non-empty")
        self.cache_dir = Path(cache_dir)
        self.camera_names = tuple(camera_names)
        self._metas: dict[str, VisionLatentEpisodeMeta] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_metas"] = {}
        state["_arrays"] = {}
        return state

    def validate_episode(self, episode: LoadedEpisode) -> VisionLatentEpisodeMeta:
        path = _meta_path(self.cache_dir, episode.stem)
        if not path.exists():
            raise FileNotFoundError(
                f"missing vision latent cache metadata: {path}. "
                "Run `python -m clearvla.cli.build_vision_latent_cache ...` first."
            )
        meta = VisionLatentEpisodeMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if meta.cache_version == VISION_LATENT_CACHE_VERSION and not _complete_path(self.cache_dir, episode.stem).exists():
            raise ValueError(f"incomplete latent cache: {_episode_dir(self.cache_dir, episode.stem)}")
        if meta.episode_stem != episode.stem:
            raise ValueError(f"latent cache stem mismatch: {meta.episode_stem!r} != {episode.stem!r}")
        if meta.num_frames != episode.length:
            raise ValueError(f"latent cache frame count mismatch: {meta.num_frames} != {episode.length}")
        if meta.cameras != self.camera_names:
            raise ValueError(f"latent cache cameras mismatch: {meta.cameras} != {self.camera_names}")
        if meta.source_fingerprint != _source_fingerprint(episode.path):
            raise ValueError(f"latent cache source fingerprint mismatch for {episode.path}")
        for camera in self.camera_names:
            tokens = np.load(_tokens_path(self.cache_dir, episode.stem, camera), mmap_mode="r")
            expected_tokens = (episode.length, meta.patch_count, meta.token_dim)
            if tuple(tokens.shape) != expected_tokens or tokens.dtype != np.float16:
                raise ValueError(
                    f"invalid latent token mmap camera={camera!r}: shape={tuple(tokens.shape)} dtype={tokens.dtype}; "
                    f"expected={expected_tokens}, float16"
                )
        self._metas[episode.stem] = meta
        return meta

    def validate_consistent(self, episodes: list[LoadedEpisode]) -> VisionLatentEpisodeMeta:
        if not episodes:
            raise ValueError("episodes must be non-empty")
        metas = [self.validate_episode(episode) for episode in episodes]
        first = metas[0]
        for meta in metas[1:]:
            if meta.patch_grid != first.patch_grid or meta.token_dim != first.token_dim:
                raise ValueError("latent cache patch specification differs across episodes")
            if meta.teacher_fingerprint != first.teacher_fingerprint:
                raise ValueError("latent cache teacher provenance differs across episodes")
        return first

    def _array(self, episode: LoadedEpisode, camera: str) -> np.ndarray:
        key = (episode.stem, camera)
        if key not in self._arrays:
            if episode.stem not in self._metas:
                self.validate_episode(episode)
            self._arrays[key] = np.load(_tokens_path(self.cache_dir, episode.stem, camera), mmap_mode="r")
        return self._arrays[key]

    def load_tokens(self, episode: LoadedEpisode, indices: np.ndarray) -> np.ndarray:
        """Return copied float32 patch tokens ``[H,V,P,C]``."""
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"indices must be non-empty [H], got {indices.shape}")
        if int(indices.min()) < 0 or int(indices.max()) >= episode.length:
            raise IndexError(f"indices range [{int(indices.min())},{int(indices.max())}] outside T={episode.length}")
        tokens = np.stack([
            np.asarray(self._array(episode, camera)[indices], dtype=np.float32)
            for camera in self.camera_names
        ], axis=1)
        return np.ascontiguousarray(tokens)
