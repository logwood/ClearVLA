from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from clearvla.mainline.training.acceleration_contract import (
    GenericTrainingAccelerationAdapter,
    module_structure_signature,
    resolve_training_acceleration_adapter,
)


class _ToyPolicy(nn.Module):
    def __init__(self, width: int, *, with_adapter: bool = False) -> None:
        super().__init__()
        self.block = nn.Linear(4, width)
        self.head = nn.Linear(width, 2)
        self._with_adapter = with_adapter

    def get_training_acceleration_adapter(self):
        if not self._with_adapter:
            return None
        return _ToyAdapter()


@dataclass(frozen=True)
class _ToyAdapter:
    name: str = "toy-v1"

    def prepare(self, engine: object) -> None:
        del engine

    def topology_signature(self, engine: object) -> tuple[str]:
        del engine
        return ("static",)

    def structure_signature(self, engine: object) -> tuple[str]:
        del engine
        return ("toy",)


def test_structure_signature_changes_when_a_branch_changes_width() -> None:
    narrow = module_structure_signature(_ToyPolicy(8))
    wide = module_structure_signature(_ToyPolicy(16))
    assert narrow != wide


def test_generic_adapter_uses_model_hooks_without_model_specific_imports() -> None:
    model = _ToyPolicy(8)
    engine = type("Engine", (), {"model": model})()
    adapter = GenericTrainingAccelerationAdapter()
    assert adapter.topology_signature(engine) == ("static",)
    assert adapter.structure_signature(engine) == module_structure_signature(model)


def test_versioned_adapter_is_resolved_by_capability_not_class_name() -> None:
    model = _ToyPolicy(8, with_adapter=True)
    adapter = resolve_training_acceleration_adapter(model)
    assert isinstance(adapter, _ToyAdapter)
    assert adapter.name == "toy-v1"


def test_default_resolution_is_safe_for_a_model_without_adapter() -> None:
    model = _ToyPolicy(8)
    adapter = resolve_training_acceleration_adapter(model)
    assert isinstance(adapter, GenericTrainingAccelerationAdapter)


def test_structure_signature_ignores_parameter_values() -> None:
    model = _ToyPolicy(8)
    before = module_structure_signature(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    assert module_structure_signature(model) == before
