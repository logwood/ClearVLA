"""One-batch CUDA prefetch for typed training inputs.

The prefetcher changes only when an already selected worker batch is copied to
the accelerator.  Batch selection, conversion, model execution and every RNG
owner remain outside this module.  The source object is retained until its
copy-stream event completes so pinned host storage cannot be recycled while a
DMA transfer is still reading it.

This is deliberately converter-driven rather than model-driven.  A versioned
data adapter can keep constructing its own :class:`TrainingBatch`; changing a
decoder, block layout or loss graph does not change this transport boundary.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

from ..interfaces import TrainingBatch

SourceBatch = TypeVar("SourceBatch")


def _unpinned_cpu_tensor_paths(value: Any, *, path: str = "batch") -> list[str]:
    """Return pageable CPU tensor paths without touching tensor values."""

    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and not value.is_pinned():
            return [path]
        return []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: list[str] = []
        for field in dataclasses.fields(value):
            result.extend(
                _unpinned_cpu_tensor_paths(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
            )
        return result
    if isinstance(value, Mapping):
        result = []
        for name, item in value.items():
            result.extend(
                _unpinned_cpu_tensor_paths(item, path=f"{path}.{name}")
            )
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(
                _unpinned_cpu_tensor_paths(item, path=f"{path}[{index}]")
            )
        return result
    return []


def _record_cuda_tree(value: Any, stream: torch.cuda.Stream) -> None:
    """Keep side-stream allocations alive through their consumer stream use."""

    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _record_cuda_tree(getattr(value, field.name), stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_cuda_tree(item, stream)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _record_cuda_tree(item, stream)


@dataclass(frozen=True)
class _PrefetchedBatch(Generic[SourceBatch]):
    batch: TrainingBatch
    source: SourceBatch
    ready: torch.cuda.Event
    source_wait_seconds: float
    conversion_enqueue_seconds: float


class CudaTrainingBatchPrefetcher(Iterator[TrainingBatch], Generic[SourceBatch]):
    """Overlap conversion of batch ``n + 1`` with computation of batch ``n``.

    ``converter`` is invoked under a private CUDA stream and must return the
    same typed batch that the non-prefetched path would return.  The consumer's
    current stream waits on a readiness event before the batch is exposed; no
    host synchronization is inserted on the steady-state path.

    The source iterable is advanced one batch ahead.  Callers that serialize a
    live sampler/iterator must account for that prefetched row before enabling
    this transport in resumable production training.
    """

    def __init__(
        self,
        source: Iterable[SourceBatch],
        *,
        converter: Callable[[SourceBatch], TrainingBatch],
        device: torch.device,
        require_pinned_source: bool = True,
        max_retained_sources: int = 2,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA batch prefetch requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA batch prefetch was requested without CUDA")
        if max_retained_sources < 1:
            raise ValueError("max_retained_sources must be positive")
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        self._source = iter(source)
        self._converter = converter
        self._require_pinned_source = bool(require_pinned_source)
        self._max_retained_sources = int(max_retained_sources)
        with torch.cuda.device(self.device):
            self.copy_stream = torch.cuda.Stream(device=self.device)
        self._retained_sources: list[tuple[SourceBatch, torch.cuda.Event]] = []
        self._closed = False
        self._next = self._preload()
        self.last_source_wait_seconds = 0.0
        self.last_conversion_enqueue_seconds = 0.0

    def __iter__(self) -> CudaTrainingBatchPrefetcher[SourceBatch]:
        return self

    def _preload(self) -> _PrefetchedBatch[SourceBatch] | None:
        wait_started = time.perf_counter()
        try:
            source = next(self._source)
        except StopIteration:
            return None
        source_wait_seconds = time.perf_counter() - wait_started
        if self._require_pinned_source:
            unpinned = _unpinned_cpu_tensor_paths(source)
            if unpinned:
                preview = ", ".join(unpinned[:4])
                suffix = "" if len(unpinned) <= 4 else f" (+{len(unpinned) - 4} more)"
                raise ValueError(
                    "CUDA batch prefetch requires pinned worker tensors; "
                    f"pageable tensors: {preview}{suffix}"
                )

        conversion_started = time.perf_counter()
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            batch = self._converter(source)
            if not isinstance(batch, TrainingBatch):
                raise TypeError(
                    "CUDA batch prefetch converter must return TrainingBatch, "
                    f"got {type(batch).__name__}"
                )
            ready = torch.cuda.Event(blocking=False, interprocess=False)
            ready.record(self.copy_stream)
        conversion_enqueue_seconds = time.perf_counter() - conversion_started
        return _PrefetchedBatch(
            batch=batch,
            source=source,
            ready=ready,
            source_wait_seconds=source_wait_seconds,
            conversion_enqueue_seconds=conversion_enqueue_seconds,
        )

    def _release_completed_sources(self) -> None:
        retained: list[tuple[SourceBatch, torch.cuda.Event]] = []
        for source, ready in self._retained_sources:
            if not ready.query():
                retained.append((source, ready))
        self._retained_sources = retained
        # A normal training step makes the oldest event ready long before this
        # limit.  The bounded fallback protects unusual consumers that drain
        # the iterator without doing GPU work, while still guaranteeing host
        # storage lifetime rather than relying on allocator implementation
        # details.
        while len(self._retained_sources) > self._max_retained_sources:
            _source, ready = self._retained_sources.pop(0)
            ready.synchronize()

    def __next__(self) -> TrainingBatch:
        if self._closed or self._next is None:
            raise StopIteration
        current = self._next
        with torch.cuda.device(self.device):
            consumer_stream = torch.cuda.current_stream(self.device)
            consumer_stream.wait_event(current.ready)
            _record_cuda_tree(current.batch, consumer_stream)
        self._retained_sources.append((current.source, current.ready))
        self.last_source_wait_seconds = current.source_wait_seconds
        self.last_conversion_enqueue_seconds = current.conversion_enqueue_seconds
        # Enqueue the following H2D transfer before returning the current batch
        # so its DMA can overlap all work submitted by the consumer.
        self._next = self._preload()
        self._release_completed_sources()
        return current.batch

    def close(self) -> None:
        """Finish any enqueued transfer before releasing pinned source owners."""

        if self._closed:
            return
        self._closed = True
        with torch.cuda.device(self.device):
            self.copy_stream.synchronize()
        self._next = None
        self._retained_sources.clear()

    def __enter__(self) -> CudaTrainingBatchPrefetcher[SourceBatch]:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "CudaTrainingBatchPrefetcher",
]
