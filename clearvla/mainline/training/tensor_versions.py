"""Cheap mutation stamps for reusable current-observation training graphs.

No tensor data are copied, hashed or reduced. PyTorch's version counters catch
normal in-place operations and parameter loading/optimizer updates. Inference
tensors have no counter; .data/raw-memory writes and buffers modified outside
this engine are not guarded. These callers must still rebuild the graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass

import torch
from torch import Tensor, nn


def _online_tensors(value: object, prefix: str = "online") -> Iterator[tuple[str, Tensor]]:
    """Traverse declared input records, not cached/teacher/future objects."""
    if isinstance(value, Tensor):
        yield prefix, value
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _online_tensors(getattr(value, item.name), prefix + "." + item.name)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _online_tensors(child, f"{prefix}[{key}]")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _online_tensors(child, f"{prefix}[{index}]")


def _version(tensor: Tensor) -> int | None:
    # PyTorch explicitly omits this counter on inference tensors. Do not
    # fabricate a counter, copy their storage or break inference-mode readers.
    return None if torch.is_inference(tensor) else tensor._version


@dataclass(frozen=True)
class _TensorStamp:
    name: str
    tensor: Tensor = field(repr=False, compare=False)
    version: int | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool

    @classmethod
    def capture(cls, name: str, tensor: Tensor) -> _TensorStamp:
        return cls(
            name,
            tensor,
            _version(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
        )

    def matches(self, name: str, tensor: Tensor) -> bool:
        return (
            self.name == name
            and self.tensor is tensor
            and self.version == _version(tensor)
            and self.shape == tuple(tensor.shape)
            and self.dtype == tensor.dtype
            and self.device == tensor.device
            and self.requires_grad == tensor.requires_grad
        )


@dataclass(frozen=True)
class TrainingSourceVersions:
    """Runtime-only source/parameter stamp; not a checkpoint or content digest."""

    online: tuple[_TensorStamp, ...] = field(repr=False, compare=False)
    parameters: tuple[_TensorStamp, ...] = field(repr=False, compare=False)

    @classmethod
    def capture(cls, online: object, model: nn.Module) -> TrainingSourceVersions:
        return cls(
            tuple(_TensorStamp.capture(name, value) for name, value in _online_tensors(online)),
            tuple(_TensorStamp.capture(name, value) for name, value in model.named_parameters()),
        )

    @property
    def unversioned_names(self) -> tuple[str, ...]:
        """Report inference tensors for which in-place detection is unavailable."""
        return tuple(x.name for x in (*self.online, *self.parameters) if x.version is None)

    @staticmethod
    def _validate(
        stamps: tuple[_TensorStamp, ...], values: Iterator[tuple[str, Tensor]], label: str
    ) -> None:
        count = 0
        for count, (name, value) in enumerate(values, start=1):
            if count > len(stamps) or not stamps[count - 1].matches(name, value):
                raise ValueError(f"encoded training graph {label} changed: {name}; re-encode")
        if count != len(stamps):
            raise ValueError(f"encoded training graph {label} inventory changed; re-encode")

    def validate(self, online: object, model: nn.Module) -> None:
        self._validate(self.online, _online_tensors(online), "online tensor")
        # Buffers are deliberately excluded: set_training_step fills the
        # canonical execution-phase buffer even when the value is unchanged.
        # Parameter values AND identities cover ordinary external optimizers,
        # load_state_dict, replacement Parameters and requires_grad changes.
        self._validate(self.parameters, iter(model.named_parameters()), "model parameter")
