from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class EventBalancedSamplerConfig:
    batch_size: int
    event_fraction: float = 0.50
    batches_per_epoch: int | None = None
    seed: int = 0
    drop_last: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.event_fraction <= 1.0:
            raise ValueError("event_fraction must be in [0,1]")
        if self.batches_per_epoch is not None and self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive when set")


class EventBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic event-aware window sampler.

    It deliberately uses replacement inside each pool so rare event windows are
    not diluted by the large number of smooth sliding windows.
    """

    def __init__(self, is_event: np.ndarray, config: EventBalancedSamplerConfig) -> None:
        config.validate()
        flags = np.asarray(is_event, dtype=bool)
        if flags.ndim != 1 or len(flags) == 0:
            raise ValueError("is_event must be a non-empty flat array")
        self.flags = flags
        self.config = config
        self.event_indices = np.flatnonzero(flags)
        self.regular_indices = np.flatnonzero(~flags)
        if len(self.event_indices) == 0:
            raise ValueError("event-aware sampler requires at least one event window")
        if len(self.regular_indices) == 0:
            raise ValueError("event-aware sampler requires at least one regular window")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.config.batches_per_epoch is not None:
            return int(self.config.batches_per_epoch)
        if self.config.drop_last:
            return len(self.flags) // self.config.batch_size
        return math.ceil(len(self.flags) / self.config.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        event_count = int(round(self.config.batch_size * self.config.event_fraction))
        event_count = min(max(event_count, 0), self.config.batch_size)
        regular_count = self.config.batch_size - event_count
        for _ in range(len(self)):
            pieces: list[np.ndarray] = []
            if event_count:
                pieces.append(rng.choice(self.event_indices, size=event_count, replace=True))
            if regular_count:
                pieces.append(rng.choice(self.regular_indices, size=regular_count, replace=True))
            batch = np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.int64)
            rng.shuffle(batch)
            yield [int(x) for x in batch]


@dataclass(frozen=True)
class TrajectoryBlockSamplerConfig:
    block_size: int
    event_fraction: float = 0.50
    blocks_per_epoch: int | None = None
    seed: int = 0

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.event_fraction <= 1.0:
            raise ValueError("event_fraction must be in [0,1]")
        if self.blocks_per_epoch is not None and self.blocks_per_epoch <= 0:
            raise ValueError("blocks_per_epoch must be positive when set")


class TrajectoryBlockBatchSampler(Sampler[list[int]]):
    """Event-aware sampler that never mixes unrelated trajectory windows in a batch.

    Each yielded batch is one contiguous block inside a single episode. Blocks
    are selected around event or regular anchors, but temporal order inside the
    block is preserved. This avoids reverting to global window-level shuffle.
    """

    def __init__(
        self,
        refs: list[object],
        is_event: np.ndarray,
        config: TrajectoryBlockSamplerConfig,
    ) -> None:
        config.validate()
        flags = np.asarray(is_event, dtype=bool)
        if flags.shape != (len(refs),) or len(refs) == 0:
            raise ValueError("is_event must align with non-empty refs")
        self.refs = refs
        self.flags = flags
        self.config = config
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for dataset_index, ref in enumerate(refs):
            episode_idx = int(getattr(ref, "episode_idx"))
            groups.setdefault(episode_idx, []).append(dataset_index)
        self.groups = {episode: tuple(indices) for episode, indices in groups.items()}
        self.index_to_group_pos: dict[int, tuple[tuple[int, ...], int]] = {}
        for indices in self.groups.values():
            for pos, dataset_index in enumerate(indices):
                self.index_to_group_pos[dataset_index] = (indices, pos)
        self.event_indices = np.flatnonzero(flags)
        self.regular_indices = np.flatnonzero(~flags)
        if len(self.event_indices) == 0 or len(self.regular_indices) == 0:
            raise ValueError("trajectory-block sampler requires event and regular windows")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.config.blocks_per_epoch is not None:
            return int(self.config.blocks_per_epoch)
        return math.ceil(len(self.refs) / self.config.block_size)

    def _block_around(self, dataset_index: int) -> list[int]:
        indices, pos = self.index_to_group_pos[int(dataset_index)]
        size = min(self.config.block_size, len(indices))
        start = max(0, min(pos - size // 2, len(indices) - size))
        return [int(value) for value in indices[start : start + size]]

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        for _ in range(len(self)):
            pool = (
                self.event_indices
                if rng.random() < self.config.event_fraction
                else self.regular_indices
            )
            anchor = int(rng.choice(pool))
            yield self._block_around(anchor)


@dataclass(frozen=True)
class TrajectorySequentialSamplerConfig:
    block_size: int
    seed: int = 0

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


class TrajectorySequentialBatchSampler(Sampler[list[int]]):
    """Shuffle episode order while traversing each selected trajectory in order."""

    def __init__(self, refs: list[object], config: TrajectorySequentialSamplerConfig) -> None:
        config.validate()
        if not refs:
            raise ValueError("refs must be non-empty")
        self.refs = refs
        self.config = config
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for dataset_index, ref in enumerate(refs):
            groups.setdefault(int(getattr(ref, "episode_idx")), []).append(dataset_index)
        self.groups = {episode: tuple(indices) for episode, indices in groups.items()}

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.config.block_size) for indices in self.groups.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        episodes = list(self.groups)
        rng.shuffle(episodes)
        for episode in episodes:
            indices = self.groups[episode]
            for start in range(0, len(indices), self.config.block_size):
                yield [int(value) for value in indices[start : start + self.config.block_size]]


@dataclass(frozen=True)
class TrajectoryShuffledBlockSamplerConfig:
    block_size: int
    seed: int = 0

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


class TrajectoryShuffledBlockBatchSampler(Sampler[list[int]]):
    """Shuffle natural contiguous blocks globally without replacement.

    This keeps mmap locality inside a batch while avoiding the strong
    cross-batch correlation of full episode-wise sequential traversal.
    Every window is emitted exactly once per epoch.
    """

    def __init__(self, refs: list[object], config: TrajectoryShuffledBlockSamplerConfig) -> None:
        config.validate()
        if not refs:
            raise ValueError("refs must be non-empty")
        self.refs = refs
        self.config = config
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for dataset_index, ref in enumerate(refs):
            groups.setdefault(int(getattr(ref, "episode_idx")), []).append(dataset_index)
        blocks: list[tuple[int, ...]] = []
        for episode in sorted(groups):
            indices = groups[episode]
            for start in range(0, len(indices), config.block_size):
                blocks.append(
                    tuple(int(value) for value in indices[start : start + config.block_size])
                )
        self.blocks = tuple(blocks)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        order = np.arange(len(self.blocks))
        rng.shuffle(order)
        for index in order.tolist():
            yield list(self.blocks[int(index)])
