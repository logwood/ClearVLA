from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from .window_boundaries import (
    CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    CAUSAL_PREFIX_V1,
    PREFIX_REGION,
    STRICT_REGION,
    TAIL_REGION,
)


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
    event_scope: str = "window_any"

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
        if self.event_scope not in {"window_any", "first_action", "disabled"}:
            raise ValueError(
                "event_scope must be window_any, first_action or disabled"
            )
        if self.event_scope == "disabled" and self.event_fraction != 0.0:
            raise ValueError("disabled event_scope requires event_fraction=0")


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
    def summary(self) -> dict[str, float | int | str]:
        uniform = min(max(int(round(self.config.batch_size * self.config.uniform_fraction)), 1), self.config.batch_size)
        event_target = min(self.config.batch_size * self.config.event_fraction, self.config.batch_size - uniform)
        scheduled = math.floor(len(self) * event_target + 1e-10) if self.informative else 0
        return {
            "windows": int(len(self.all_indices)),
            "motion_windows": int(len(self.motion_indices)),
            "event_windows": int(len(self.event_indices)),
            "event_scope": self.config.event_scope,
            "motion_threshold": float(
                np.quantile(self.motion_score, self.config.motion_quantile)
            ),
            "uniform_fraction": float(self.config.uniform_fraction),
            "event_fraction": float(self.config.event_fraction),
            "motion_fraction": float(
                1.0 - self.config.uniform_fraction - self.config.event_fraction
            ),
            "fallback_uniform": int(not self.informative),
            "event_rows_per_epoch_scheduled": scheduled,
            "effective_dedicated_event_fraction": scheduled / max(len(self) * self.config.batch_size, 1),
            "event_fractional_carry": int(event_target != int(event_target)),
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
        uniform_count = min(max(uniform_count, 1), self.config.batch_size)
        event_target = min(self.config.batch_size * self.config.event_fraction, self.config.batch_size - uniform_count)
        cursor = 0
        for batch_index in range(len(self)):
            # Carry fractional event mass across batches. B4/default alternates
            # 0/1 instead of round(0.5)==0 forever; integral B8 stays bit-exact.
            event_count = (math.floor((batch_index + 1) * event_target + 1e-10)
                           - math.floor(batch_index * event_target + 1e-10))
            motion_count = self.config.batch_size - uniform_count - event_count
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


class BoundaryAwareInformationBatchSampler(Sampler[list[int]]):
    """Reserve exact prefix/tail slots around an information-balanced strict lane.

    The epoch length is deliberately anchored to the historical strict-window
    count divided by the *full* batch size.  Adding boundary supervision thus
    changes neither optimizer steps per epoch nor schedule duration.  The
    remaining strict slots retain the existing information sampler, while
    prefix and tail pools traverse independent shuffled cycles.
    """

    def __init__(
        self,
        motion_score: np.ndarray,
        is_event: np.ndarray,
        boundary_region: Sequence[str],
        *,
        contract: str,
        config: InformationBalancedSamplerConfig,
        release_first_events: np.ndarray | None = None,
        release_first_fraction: float = 0.0,
    ) -> None:
        config.validate()
        score = np.asarray(motion_score, dtype=np.float64)
        events = np.asarray(is_event, dtype=bool)
        regions = np.asarray(tuple(str(value) for value in boundary_region), dtype=object)
        if score.ndim != 1 or not len(score) or events.shape != score.shape:
            raise ValueError("boundary sampler motion/event vectors must align")
        if regions.shape != score.shape:
            raise ValueError("boundary sampler regions must align with dataset windows")
        if not np.isfinite(score).all() or (score < 0.0).any():
            raise ValueError("boundary sampler motion scores must be finite and non-negative")
        release_fraction = float(release_first_fraction)
        if not np.isfinite(release_fraction) or not 0.0 <= release_fraction <= 1.0:
            raise ValueError("release_first_fraction must be finite and in [0,1]")
        if release_first_events is None:
            release_flags = np.zeros((len(score),), dtype=bool)
        else:
            release_flags = np.asarray(release_first_events, dtype=bool)
            if release_flags.shape != score.shape:
                raise ValueError(
                    "release-first event flags must align with boundary sampler windows"
                )
        selected_contract = str(contract)
        if selected_contract == CAUSAL_PREFIX_V1:
            quotas = {PREFIX_REGION: 1, TAIL_REGION: 0, STRICT_REGION: 7}
        elif selected_contract == CAUSAL_PREFIX_TERMINAL_SUFFIX_V2:
            quotas = {PREFIX_REGION: 1, TAIL_REGION: 1, STRICT_REGION: 6}
        else:
            raise ValueError(
                "boundary-aware sampling requires a causal LIBERO boundary contract"
            )
        if config.batch_size != 8 or sum(quotas.values()) != config.batch_size:
            raise ValueError("controlled LIBERO boundary sampling requires B8")
        pools = {
            name: np.flatnonzero(regions == name).astype(np.int64, copy=False)
            for name in (PREFIX_REGION, STRICT_REGION, TAIL_REGION)
        }
        for name, quota in quotas.items():
            if quota and len(pools[name]) < quota:
                raise ValueError(
                    f"boundary sampler region {name!r} has {len(pools[name])} rows, "
                    f"fewer than its per-batch quota {quota}"
                )
        strict_pool = pools[STRICT_REGION]
        # The historical boundary sampler used the strict-window count as its
        # epoch length.  That is intentionally retained when no override is
        # present so old runs replay identically.  A formal full-coverage run
        # may explicitly request the complete dataset-window budget; this is
        # a sampling control, not an implicit change to the boundary quotas.
        if config.batches_per_epoch is None:
            batches = math.ceil(len(strict_pool) / config.batch_size)
            epoch_length_reference = "ceil(strict_window_count/full_batch_size)"
        else:
            batches = int(config.batches_per_epoch)
            epoch_length_reference = "explicit_configured_batches_per_epoch"
        if batches <= 0:
            raise ValueError("boundary sampler has no strict baseline windows")
        strict_config = InformationBalancedSamplerConfig(
            batch_size=quotas[STRICT_REGION],
            uniform_fraction=config.uniform_fraction,
            event_fraction=config.event_fraction,
            motion_quantile=config.motion_quantile,
            batches_per_epoch=batches,
            seed=config.seed,
            drop_last=False,
            event_scope=config.event_scope,
        )
        self.contract = selected_contract
        self.config = config
        self.quotas = quotas
        self.pools = pools
        self.release_first_fraction = release_fraction
        self.release_first_pool = np.intersect1d(
            pools[TAIL_REGION],
            np.flatnonzero(release_flags).astype(np.int64, copy=False),
            assume_unique=True,
        )
        self.non_release_tail_pool = np.setdiff1d(
            pools[TAIL_REGION],
            self.release_first_pool,
            assume_unique=True,
        )
        self.strict_sampler = InformationBalancedBatchSampler(
            score[strict_pool],
            events[strict_pool],
            strict_config,
        )
        self.epoch = 0
        self.batches_per_epoch = int(batches)
        self.epoch_length_reference = epoch_length_reference

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self.strict_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    @property
    def summary(self) -> dict[str, object]:
        release_target = self.quotas[TAIL_REGION] * self.release_first_fraction
        release_scheduled = (
            math.floor(len(self) * release_target + 1e-10)
            if len(self.release_first_pool)
            else 0
        )
        return {
            "schema": "clearvla-boundary-aware-information-sampler-v2",
            "contract": self.contract,
            "batch_size": int(self.config.batch_size),
            "batches_per_epoch": len(self),
            "epoch_length_reference": self.epoch_length_reference,
            "region_window_counts": {
                name: int(len(pool)) for name, pool in self.pools.items()
            },
            "region_quota_per_batch": dict(self.quotas),
            "region_rows_per_epoch": {
                name: int(quota * len(self)) for name, quota in self.quotas.items()
            },
            "release_first_action_fraction": float(self.release_first_fraction),
            "release_first_action_windows": int(len(self.release_first_pool)),
            "release_first_action_rows_per_epoch_scheduled": int(release_scheduled),
            "release_first_action_lane": (
                "terminal_tail_quota"
                if self.release_first_fraction > 0.0 and len(self.release_first_pool)
                else "disabled_or_empty"
            ),
            "strict_information_sampling": self.strict_sampler.summary,
            "prefix_tail_cycle_replacement": "only_after_full_region_cycle",
            "within_batch_duplicates": False,
        }

    @staticmethod
    def _draw_cycle(
        rng: np.random.Generator,
        pool: np.ndarray,
        permutation: np.ndarray,
        cursor: int,
        count: int,
    ) -> tuple[list[int], np.ndarray, int]:
        if count <= 0:
            return [], permutation, cursor
        if len(pool) < count:
            raise ValueError("boundary region pool is smaller than its batch quota")
        if cursor + count > len(permutation):
            permutation = pool.copy()
            rng.shuffle(permutation)
            cursor = 0
        rows = [int(value) for value in permutation[cursor : cursor + count]]
        return rows, permutation, cursor + count

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003 + 79_019)
        permutations = {
            name: pool.copy() for name, pool in self.pools.items()
        }
        cursors = {name: 0 for name in self.pools}
        for values in permutations.values():
            rng.shuffle(values)

        release_permutation = self.release_first_pool.copy()
        release_cursor = 0
        non_release_tail_permutation = self.non_release_tail_pool.copy()
        non_release_tail_cursor = 0
        if len(release_permutation):
            rng.shuffle(release_permutation)
        if len(non_release_tail_permutation):
            rng.shuffle(non_release_tail_permutation)

        strict_pool = self.pools[STRICT_REGION]
        if self.strict_sampler.informative:
            strict_batches: Iterator[list[int]] = iter(self.strict_sampler)
        else:
            # The ordinary sampler intentionally emits one exact traversal when
            # all scores are indistinguishable.  This controlled wrapper still
            # needs the fixed historical number of batches, so cycle the
            # strict pool explicitly in that degenerate case.
            strict_batches = iter(())
        release_target = self.quotas[TAIL_REGION] * self.release_first_fraction
        for _batch_index in range(len(self)):
            if self.strict_sampler.informative:
                strict_local = next(strict_batches)
                strict_rows = [int(strict_pool[index]) for index in strict_local]
            else:
                strict_rows, permutations[STRICT_REGION], cursors[STRICT_REGION] = (
                    self._draw_cycle(
                        rng,
                        strict_pool,
                        permutations[STRICT_REGION],
                        cursors[STRICT_REGION],
                        self.quotas[STRICT_REGION],
                    )
                )
            batch = list(strict_rows)
            # Prefix remains a plain causal-reset cycle.  For the terminal
            # tail, optionally spend a deterministic fraction of the quota on
            # a directional opening transition—the exact row receding-horizon
            # deployment executes next.  The fallback preserves the ordinary
            # tail cycle when a split has no release rows.
            prefix_rows, permutations[PREFIX_REGION], cursors[PREFIX_REGION] = (
                self._draw_cycle(
                    rng,
                    self.pools[PREFIX_REGION],
                    permutations[PREFIX_REGION],
                    cursors[PREFIX_REGION],
                    self.quotas[PREFIX_REGION],
                )
            )
            batch.extend(prefix_rows)
            release_count = (
                math.floor((_batch_index + 1) * release_target + 1e-10)
                - math.floor(_batch_index * release_target + 1e-10)
            )
            if not len(self.release_first_pool):
                release_count = 0
            tail_count = self.quotas[TAIL_REGION] - release_count
            tail_rows: list[int] = []
            if release_count:
                tail_rows, release_permutation, release_cursor = self._draw_cycle(
                    rng,
                    self.release_first_pool,
                    release_permutation,
                    release_cursor,
                    release_count,
                )
            if tail_count:
                normal_pool = (
                    self.non_release_tail_pool
                    if release_count and len(self.non_release_tail_pool) >= tail_count
                    else self.pools[TAIL_REGION]
                )
                normal_permutation = (
                    non_release_tail_permutation
                    if normal_pool is self.non_release_tail_pool
                    else permutations[TAIL_REGION]
                )
                normal_cursor = (
                    non_release_tail_cursor
                    if normal_pool is self.non_release_tail_pool
                    else cursors[TAIL_REGION]
                )
                normal_rows, normal_permutation, normal_cursor = self._draw_cycle(
                    rng,
                    normal_pool,
                    normal_permutation,
                    normal_cursor,
                    tail_count,
                )
                tail_rows.extend(normal_rows)
                if normal_pool is self.non_release_tail_pool:
                    non_release_tail_permutation, non_release_tail_cursor = (
                        normal_permutation,
                        normal_cursor,
                    )
                else:
                    permutations[TAIL_REGION], cursors[TAIL_REGION] = (
                        normal_permutation,
                        normal_cursor,
                    )
            batch.extend(tail_rows)
            if len(batch) != self.config.batch_size or len(set(batch)) != len(batch):
                raise AssertionError("boundary-aware sampler violated its B8 quota/uniqueness")
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


class EvenlySpacedPanelBatchSampler(Sampler[list[int]]):
    """Bound one read-only panel while covering its complete ordered pool.

    Prefix and terminal-tail validation pools are short but still expensive at
    deploy-time ODE sampling.  Selecting evenly spaced rows avoids measuring
    only the first validation episode while keeping the panel to a fixed,
    explicitly serialized number of batches.
    """

    def __init__(self, length: int, *, batch_size: int, max_batches: int) -> None:
        if int(length) <= 0 or int(batch_size) <= 0 or int(max_batches) <= 0:
            raise ValueError("panel length, batch size and max batches must be positive")
        self.source_rows = int(length)
        self.batch_size = int(batch_size)
        self.max_batches = int(max_batches)
        selected_count = min(self.source_rows, self.batch_size * self.max_batches)
        positions = np.linspace(
            0,
            self.source_rows - 1,
            num=selected_count,
            dtype=np.int64,
        )
        if len(np.unique(positions)) != selected_count:
            raise AssertionError("evenly spaced panel selection produced duplicates")
        self.order = tuple(int(value) for value in positions)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    @property
    def summary(self) -> dict[str, object]:
        return {
            "schema": "clearvla-evenly-spaced-validation-panel-v1",
            "source_rows": self.source_rows,
            "selected_rows": len(self.order),
            "batch_size": self.batch_size,
            "batches": len(self),
            "selection": "inclusive_linspace_over_ordered_dataset",
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
