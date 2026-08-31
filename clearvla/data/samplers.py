from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

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
class InformationBalancedSamplerConfig:
    batch_size: int
    uniform_fraction: float = 0.50
    event_fraction: float = 0.125
    motion_quantile: float = 0.70
    batches_per_epoch: int | None = None
    seed: int = 0
    drop_last: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0,1]")
        if not 0.0 <= self.event_fraction <= 1.0:
            raise ValueError("event_fraction must be in [0,1]")
        if self.uniform_fraction + self.event_fraction > 1.0:
            raise ValueError("uniform_fraction + event_fraction cannot exceed 1")
        if not 0.0 <= self.motion_quantile <= 1.0:
            raise ValueError("motion_quantile must be in [0,1]")
        if self.batches_per_epoch is not None and self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive when set")


class InformationBalancedBatchSampler(Sampler[list[int]]):
    """Mix uniform coverage with bounded motion/event strata.

    The uniform lane is drawn without replacement from a shuffled permutation.
    Informative lanes may repeat rare windows, but never replace the uniform
    lane.  If no informative distinction exists, the sampler becomes an exact
    ordinary shuffled traversal of the dataset.
    """

    def __init__(
        self,
        motion_score: np.ndarray,
        is_event: np.ndarray,
        config: InformationBalancedSamplerConfig,
    ) -> None:
        config.validate()
        score = np.asarray(motion_score, dtype=np.float64)
        events = np.asarray(is_event, dtype=bool)
        if score.ndim != 1 or len(score) == 0 or events.shape != score.shape:
            raise ValueError("motion_score and is_event must be aligned non-empty vectors")
        if not np.isfinite(score).all() or (score < 0.0).any():
            raise ValueError("motion_score must be finite and non-negative")
        self.motion_score = score
        self.is_event = events
        self.config = config
        self.epoch = 0
        threshold = float(np.quantile(score, config.motion_quantile))
        spread = float(score.max() - score.min())
        self.motion_indices = (
            np.flatnonzero(score >= threshold)
            if spread > max(1e-12, abs(float(score.mean())) * 1e-8)
            else np.empty((0,), dtype=np.int64)
        )
        self.event_indices = np.flatnonzero(events)
        self.all_indices = np.arange(len(score), dtype=np.int64)
        self.informative = bool(len(self.motion_indices) or len(self.event_indices))

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.config.batches_per_epoch is not None:
            return int(self.config.batches_per_epoch)
        if self.config.drop_last:
            return len(self.all_indices) // self.config.batch_size
        return math.ceil(len(self.all_indices) / self.config.batch_size)

    @property
    def summary(self) -> dict[str, float | int]:
        return {
            "windows": int(len(self.all_indices)),
            "motion_windows": int(len(self.motion_indices)),
            "event_windows": int(len(self.event_indices)),
            "motion_threshold": float(
                np.quantile(self.motion_score, self.config.motion_quantile)
            ),
            "uniform_fraction": float(self.config.uniform_fraction),
            "event_fraction": float(self.config.event_fraction),
            "motion_fraction": float(
                1.0 - self.config.uniform_fraction - self.config.event_fraction
            ),
            "fallback_uniform": int(not self.informative),
        }

    @staticmethod
    def _choice(
        rng: np.random.Generator,
        pool: np.ndarray,
        count: int,
        selected: set[int],
        fallback: np.ndarray,
    ) -> list[int]:
        if count <= 0:
            return []
        available = np.asarray(
            [int(x) for x in pool if int(x) not in selected], dtype=np.int64
        )
        primary_count = min(count, len(available))
        rows: list[int] = []
        if primary_count:
            rows.extend(
                int(value)
                for value in rng.choice(available, size=primary_count, replace=False)
            )
        remaining = count - len(rows)
        if remaining:
            occupied = selected | set(rows)
            fallback_available = np.asarray(
                [int(x) for x in fallback if int(x) not in occupied], dtype=np.int64
            )
            if len(fallback_available) == 0:
                fallback_available = fallback
            rows.extend(
                int(value)
                for value in rng.choice(
                    fallback_available,
                    size=remaining,
                    replace=len(fallback_available) < remaining,
                )
            )
        return rows

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        permutation = self.all_indices.copy()
        rng.shuffle(permutation)
        if not self.informative:
            for start in range(0, len(permutation), self.config.batch_size):
                batch = permutation[start : start + self.config.batch_size]
                if len(batch) < self.config.batch_size and self.config.drop_last:
                    break
                yield [int(value) for value in batch]
            return

        uniform_count = int(round(self.config.batch_size * self.config.uniform_fraction))
        event_count = int(round(self.config.batch_size * self.config.event_fraction))
        uniform_count = min(max(uniform_count, 1), self.config.batch_size)
        event_count = min(max(event_count, 0), self.config.batch_size - uniform_count)
        motion_count = self.config.batch_size - uniform_count - event_count
        cursor = 0
        for _ in range(len(self)):
            if cursor + uniform_count > len(permutation):
                rng.shuffle(permutation)
                cursor = 0
            batch = [int(value) for value in permutation[cursor : cursor + uniform_count]]
            cursor += uniform_count
            selected = set(batch)
            event_rows = self._choice(
                rng,
                self.event_indices,
                event_count,
                selected,
                self.all_indices,
            )
            batch.extend(event_rows)
            selected.update(event_rows)
            motion_rows = self._choice(
                rng,
                self.motion_indices,
                motion_count,
                selected,
                self.all_indices,
            )
            batch.extend(motion_rows)
            rng.shuffle(batch)
            yield batch


class TaskBalancedInformationBatchSampler(Sampler[list[int]]):
    """Balance task identity before drawing the existing information lanes.

    Task identity is CPU-side sampling metadata only.  It never enters a
    dataset sample or model input.  Every batch owns the same fixed number of
    slots as the ordinary information sampler; the slots are first assigned
    round-robin over tasks, then assigned uniform/event/motion lanes using the
    existing configured fractions.  With eight tasks and batch size eight,
    every batch therefore contains exactly one row from every task.

    Uniform rows traverse a task-local shuffled permutation before repeating.
    Event and motion rows may repeat, matching the established informative
    lane semantics.  A task with no event or motion distinction falls back to
    its uniform lane rather than borrowing a row from another task.
    """

    def __init__(
        self,
        motion_score: np.ndarray,
        is_event: np.ndarray,
        task_index: np.ndarray,
        task_names: Sequence[str],
        config: InformationBalancedSamplerConfig,
    ) -> None:
        config.validate()
        score = np.asarray(motion_score, dtype=np.float64)
        events = np.asarray(is_event, dtype=bool)
        tasks = np.asarray(task_index, dtype=np.int64)
        names = tuple(str(value) for value in task_names)
        if score.ndim != 1 or len(score) == 0:
            raise ValueError("motion_score must be a non-empty flat vector")
        if events.shape != score.shape or tasks.shape != score.shape:
            raise ValueError("motion, event and task vectors must align")
        if not np.isfinite(score).all() or (score < 0.0).any():
            raise ValueError("motion_score must be finite and non-negative")
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("task names must be non-empty and unique")
        if (tasks < 0).any() or (tasks >= len(names)).any():
            raise ValueError("task indices must identify the declared task order")
        self.motion_score = score
        self.is_event = events
        self.task_index = tasks
        self.task_names = names
        self.config = config
        self.epoch = 0
        self.all_indices = np.arange(len(score), dtype=np.int64)
        self.task_pools: tuple[np.ndarray, ...] = tuple(
            np.flatnonzero(tasks == task) for task in range(len(names))
        )
        if any(len(pool) == 0 for pool in self.task_pools):
            missing = [names[index] for index, pool in enumerate(self.task_pools) if not len(pool)]
            raise ValueError(f"task-balanced sampler has empty tasks: {missing}")
        motion_pools: list[np.ndarray] = []
        event_pools: list[np.ndarray] = []
        motion_thresholds: list[float] = []
        for pool in self.task_pools:
            task_score = score[pool]
            threshold = float(np.quantile(task_score, config.motion_quantile))
            spread = float(task_score.max() - task_score.min())
            motion_thresholds.append(threshold)
            motion_pools.append(
                pool[task_score >= threshold]
                if spread > max(1e-12, abs(float(task_score.mean())) * 1e-8)
                else np.empty((0,), dtype=np.int64)
            )
            event_pools.append(pool[events[pool]])
        self.motion_pools = tuple(motion_pools)
        self.event_pools = tuple(event_pools)
        self.motion_thresholds = tuple(motion_thresholds)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.config.batches_per_epoch is not None:
            return int(self.config.batches_per_epoch)
        if self.config.drop_last:
            return len(self.all_indices) // self.config.batch_size
        return math.ceil(len(self.all_indices) / self.config.batch_size)

    @property
    def summary(self) -> dict[str, object]:
        total_slots = len(self) * int(self.config.batch_size)
        task_count = len(self.task_names)
        projected_floor = total_slots // task_count
        projected_ceil = math.ceil(total_slots / task_count)
        return {
            "schema": "clearvla-task-balanced-information-sampler-v1",
            "windows": int(len(self.all_indices)),
            "task_count": task_count,
            "task_order": list(self.task_names),
            "batch_size": int(self.config.batch_size),
            "batches_per_epoch": int(len(self)),
            "uniform_fraction": float(self.config.uniform_fraction),
            "event_fraction": float(self.config.event_fraction),
            "motion_fraction": float(
                1.0 - self.config.uniform_fraction - self.config.event_fraction
            ),
            "projected_samples_per_task_min": int(projected_floor),
            "projected_samples_per_task_max": int(projected_ceil),
            "projected_task_sample_count_gap_max": int(
                projected_ceil - projected_floor
            ),
            "tasks": [
                {
                    "task_id": name,
                    "windows": int(len(self.task_pools[index])),
                    "event_windows": int(len(self.event_pools[index])),
                    "motion_windows": int(len(self.motion_pools[index])),
                    "motion_threshold": float(self.motion_thresholds[index]),
                }
                for index, name in enumerate(self.task_names)
            ],
        }

    @staticmethod
    def _lane_counts(config: InformationBalancedSamplerConfig) -> tuple[int, int, int]:
        uniform = int(round(config.batch_size * config.uniform_fraction))
        event = int(round(config.batch_size * config.event_fraction))
        uniform = min(max(uniform, 1), config.batch_size)
        event = min(max(event, 0), config.batch_size - uniform)
        return uniform, event, config.batch_size - uniform - event

    @staticmethod
    def _draw_pool(
        rng: np.random.Generator,
        pool: np.ndarray,
        selected: set[int],
    ) -> int | None:
        available = np.asarray(
            [int(value) for value in pool if int(value) not in selected],
            dtype=np.int64,
        )
        if not len(available):
            return None
        return int(rng.choice(available))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003)
        task_order = np.arange(len(self.task_names), dtype=np.int64)
        rng.shuffle(task_order)
        uniform_rows = [pool.copy() for pool in self.task_pools]
        uniform_cursor = [0 for _ in self.task_pools]
        for rows in uniform_rows:
            rng.shuffle(rows)

        def draw_uniform(task: int, selected: set[int]) -> int:
            pool = uniform_rows[task]
            attempts = 0
            while attempts <= len(pool):
                if uniform_cursor[task] >= len(pool):
                    rng.shuffle(pool)
                    uniform_cursor[task] = 0
                value = int(pool[uniform_cursor[task]])
                uniform_cursor[task] += 1
                attempts += 1
                if value not in selected:
                    return value
            # A repeated task can legally exhaust every unique row inside one
            # oversized batch.  Repetition is then explicit and local to that
            # task rather than silently borrowing another task's sample.
            return int(rng.choice(self.task_pools[task]))

        uniform_count, event_count, motion_count = self._lane_counts(self.config)
        lane_template = np.asarray(
            [0] * uniform_count + [1] * event_count + [2] * motion_count,
            dtype=np.int8,
        )
        task_cursor = 0
        for _batch in range(len(self)):
            task_slots = [
                int(task_order[(task_cursor + offset) % len(task_order)])
                for offset in range(self.config.batch_size)
            ]
            task_cursor = (task_cursor + self.config.batch_size) % len(task_order)
            lanes = lane_template.copy()
            rng.shuffle(lanes)
            selected: set[int] = set()
            batch: list[int] = []
            for task, lane in zip(task_slots, lanes, strict=True):
                if int(lane) == 1:
                    value = self._draw_pool(rng, self.event_pools[task], selected)
                elif int(lane) == 2:
                    value = self._draw_pool(rng, self.motion_pools[task], selected)
                else:
                    value = None
                if value is None:
                    value = draw_uniform(task, selected)
                batch.append(value)
                selected.add(value)
            paired = list(zip(task_slots, batch, strict=True))
            rng.shuffle(paired)
            yield [value for _task, value in paired]


class TaskStratifiedBatchSampler(Sampler[list[int]]):
    """Deterministic bounded validation panel with equal rows per task."""

    def __init__(
        self,
        task_index: np.ndarray,
        task_names: Sequence[str],
        *,
        samples_per_task: int,
        batch_size: int,
    ) -> None:
        tasks = np.asarray(task_index, dtype=np.int64)
        names = tuple(str(value) for value in task_names)
        if tasks.ndim != 1 or not len(tasks):
            raise ValueError("validation task indices must be a non-empty vector")
        if not names or len(set(names)) != len(names):
            raise ValueError("validation task names must be non-empty and unique")
        if samples_per_task <= 0 or batch_size <= 0:
            raise ValueError("validation samples per task and batch size must be positive")
        if (tasks < 0).any() or (tasks >= len(names)).any():
            raise ValueError("validation task indices are outside the task registry")
        per_task: list[np.ndarray] = []
        for task in range(len(names)):
            pool = np.flatnonzero(tasks == task)
            if not len(pool):
                raise ValueError(f"validation task has no windows: {names[task]}")
            count = min(int(samples_per_task), len(pool))
            positions = np.linspace(0, len(pool) - 1, num=count, dtype=np.int64)
            selected = pool[positions]
            if len(np.unique(selected)) != count:
                raise AssertionError("stratified validation selected duplicate rows")
            per_task.append(selected)
        order: list[int] = []
        for row in range(max(len(values) for values in per_task)):
            for values in per_task:
                if row < len(values):
                    order.append(int(values[row]))
        self.task_index = tasks
        self.task_names = names
        self.samples_per_task = int(samples_per_task)
        self.batch_size = int(batch_size)
        self.per_task = tuple(per_task)
        self.order = tuple(order)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    @property
    def summary(self) -> dict[str, object]:
        return {
            "schema": "clearvla-task-stratified-validation-panel-v1",
            "task_order": list(self.task_names),
            "requested_samples_per_task": self.samples_per_task,
            "selected_samples": len(self.order),
            "selected_samples_per_task": {
                name: int(len(self.per_task[index]))
                for index, name in enumerate(self.task_names)
            },
            "batch_size": self.batch_size,
            "batches": len(self),
        }

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.order), self.batch_size):
            yield list(self.order[start : start + self.batch_size])


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
