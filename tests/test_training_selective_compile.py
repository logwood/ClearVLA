from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from clearvla.mainline.training.acceleration_adapters import (
    MainlineTrainingAccelerationAdapter,
)
from clearvla.mainline.training.acceleration_contract import (
    TrainingCompilePlan,
    TrainingCompileRegion,
    TrainingEagerBoundary,
)
from clearvla.mainline.training.cuda_graph import CudaGraphTrainingStepRunner
from clearvla.mainline.training.selective_compile import (
    AppliedTrainingCompilePlan,
    apply_training_compile_plan,
    get_applied_training_compile_plan,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)

    def _unit_rms(self, value: torch.Tensor) -> torch.Tensor:
        return value.square().mean().sqrt()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value) + self._unit_rms(value)


class _CompileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(value))


def _plan(*, acceptance: str = "candidate") -> TrainingCompilePlan:
    return TrainingCompilePlan(
        name="toy-selective-v1",
        regions=(TrainingCompileRegion(name="body", module_paths=("block",)),),
        eager_boundaries=(
            TrainingEagerBoundary(
                module_path="block.norm",
                reason="preserve-eager-norm",
            ),
            TrainingEagerBoundary(
                module_path="block",
                method_name="_unit_rms",
                reason="preserve-eager-rms",
            ),
        ),
        acceptance=acceptance,  # type: ignore[arg-type]
    )


def _fake_disable(method: Any, *, recursive: bool = True, reason: str = "") -> Any:
    del recursive, reason

    def disabled(*args: Any, **kwargs: Any) -> Any:
        return method(*args, **kwargs)

    return disabled


def test_candidate_plan_is_rejected_before_any_callable_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CompileModel()
    engine = SimpleNamespace(model=model)
    compile_calls = 0

    def fake_compile(method: Any, **kwargs: Any) -> Any:
        nonlocal compile_calls
        del kwargs
        compile_calls += 1
        return method

    monkeypatch.setattr(torch, "compile", fake_compile)
    before = model(torch.randn(2, 4))
    with pytest.raises(PermissionError, match="only a candidate"):
        apply_training_compile_plan(engine, _plan(), profile="toy")
    after = model(torch.randn(2, 4))
    assert compile_calls == 0
    assert "forward" not in model.block.__dict__
    assert get_applied_training_compile_plan(engine) is None
    assert before.shape == after.shape


def test_plan_installs_idempotently_and_records_complete_audit_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CompileModel()
    engine = SimpleNamespace(model=model)
    calls: list[str] = []

    def fake_compile(method: Any, **kwargs: Any) -> Any:
        del kwargs
        calls.append("compile")

        def compiled(*args: Any, **call_kwargs: Any) -> Any:
            return method(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "disable", _fake_disable)
    value = torch.randn(2, 4)
    expected = model(value)
    applied = apply_training_compile_plan(
        engine,
        _plan(),
        profile="toy",
        allow_candidate=True,
    )
    actual = model(value)
    repeated = apply_training_compile_plan(
        engine,
        _plan(),
        profile="toy",
        allow_candidate=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert repeated is applied
    assert calls == ["compile"]
    summary = applied.summary()
    assert summary["rng_policy"] == _plan().rng_policy
    assert summary["numerical_policy"] == _plan().numerical_policy
    assert len(str(summary["signature_sha256"])) == 64


def test_missing_path_and_unknown_option_fail_before_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CompileModel()
    engine = SimpleNamespace(model=model)

    def unexpected_compile(method: Any, **kwargs: Any) -> Any:
        del method, kwargs
        raise AssertionError("torch.compile must not run after failed preflight")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    missing = replace(
        _plan(),
        regions=(TrainingCompileRegion(name="body", module_paths=("missing",)),),
        eager_boundaries=(),
    )
    with pytest.raises(ValueError, match="missing modules"):
        apply_training_compile_plan(
            engine,
            missing,
            profile="missing",
            allow_candidate=True,
        )

    unknown_option = replace(
        _plan(),
        regions=(
            TrainingCompileRegion(
                name="body",
                module_paths=("block",),
                options=(("clearvla.this_option_does_not_exist", True),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unsupported Inductor options"):
        apply_training_compile_plan(
            engine,
            unknown_option,
            profile="unknown-option",
            allow_candidate=True,
        )
    assert "forward" not in model.block.__dict__


def test_partial_compile_failure_restores_methods_and_transform_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CompileModel()
    engine = SimpleNamespace(model=model)
    rollback_plan = TrainingCompilePlan(
        name="rollback",
        regions=(
            TrainingCompileRegion(
                name="body",
                module_paths=("block", "head"),
            ),
        ),
        acceptance="strict-gated",
    )
    compile_calls = 0

    def flaky_compile(method: Any, **kwargs: Any) -> Any:
        nonlocal compile_calls
        del kwargs
        compile_calls += 1
        if compile_calls == 2:
            raise RuntimeError("synthetic compile failure")
        return method

    monkeypatch.setattr(torch, "compile", flaky_compile)
    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        apply_training_compile_plan(engine, rollback_plan, profile="rollback")
    assert "forward" not in model.block.__dict__
    assert "forward" not in model.head.__dict__
    assert not hasattr(model.block, "_clearvla_training_compile_transforms")
    assert not hasattr(model.head, "_clearvla_training_compile_transforms")
    assert get_applied_training_compile_plan(engine) is None


class _NamedContainer(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _layout_model(*, modular: bool) -> nn.Module:
    model = _NamedContainer()
    encoder = _NamedContainer()
    encoder.flow = _NamedContainer()
    encoder.flow.encoder = _NamedContainer()
    encoder.flow.context = _NamedContainer()
    encoder.flow.update = _NamedContainer()
    encoder.flow.delta_head = _NamedContainer()
    encoder.flow.initial_residual = _NamedContainer()
    encoder.raw_flow = _NamedContainer()
    encoder.raw_flow.pyramid = _NamedContainer()
    observation = _NamedContainer()
    if modular:
        observation.compiler = _NamedContainer()
        observation.compiler.encoder = encoder
    else:
        observation.encoder = encoder
    model.observation = observation

    decoder = _NamedContainer()
    decoder.blocks = nn.ModuleList((_Block(), _Block()))
    layer_heads = nn.ModuleList((_NamedContainer(), _NamedContainer()))
    if modular:
        model.execution_bottom = _NamedContainer()
        model.execution_bottom.decoder = decoder
        model.execution_bottom.layer_contract_heads = layer_heads
        model.grounding = _NamedContainer()
        model.grounding.blocks = nn.ModuleList((_Block(),))
        model.p1 = _NamedContainer()
        model.p1.dynamic_policy_block = _Block()
        model.world = _NamedContainer()
        model.world.dynamics = _NamedContainer()
        model.world.dynamics.w1 = _Block()
        model.world.dynamics.w2 = _Block()
    else:
        model.bottom = _NamedContainer()
        model.bottom.decoder = decoder
        model.bottom.layer_contract_heads = layer_heads
        model.bottom.p1_policy_block = _Block()
        model.top = _NamedContainer()
        model.top.grounding_blocks = nn.ModuleList((_Block(),))
        model.top.dynamics = _NamedContainer()
        model.top.dynamics.w1 = _Block()
        model.top.dynamics.w2 = _Block()
    model.transition = _NamedContainer()
    model.transition.v120_transition = _Block()
    return model


@pytest.mark.parametrize(
    ("modular", "expected_prefix"),
    ((True, "clearvla-modular"), (False, "clearvla-legacy")),
)
def test_mainline_adapter_builds_layout_specific_partitioned_plan(
    modular: bool,
    expected_prefix: str,
) -> None:
    model = _layout_model(modular=modular)
    engine = SimpleNamespace(model=model)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        "partitioned-combined",
    )
    plan.validate()
    assert plan.name.startswith(expected_prefix)
    assert {region.name for region in plan.regions} == {
        "visual",
        "mainline",
        "mmdit",
    }
    assert plan.eager_boundaries


def test_cuda_graph_capture_identity_changes_with_compile_plan_signature() -> None:
    runner = object.__new__(CudaGraphTrainingStepRunner)
    runner.engine = SimpleNamespace()
    runner._acceleration_adapter = SimpleNamespace(
        name="adapter",
        topology_signature=lambda _engine: ("static",),
    )
    runner._structure_signature = ("structure",)
    runner._applied_compile_plan = None
    batch = torch.zeros(2, 3)
    eager_identity = runner._capture_identity(batch)
    plan = _plan(acceptance="strict-gated")
    runner._applied_compile_plan = AppliedTrainingCompilePlan(
        profile="toy",
        plan_name=plan.name,
        acceptance=plan.acceptance,
        rng_policy=plan.rng_policy,
        numerical_policy=plan.numerical_policy,
        signature=plan.signature(),
        signature_sha256="0" * 64,
        compiled_module_paths=("block",),
        eager_boundaries=(("block.norm", "forward"),),
        wrapper_setup_seconds=0.0,
        model_identity=0,
    )
    compiled_identity = runner._capture_identity(batch)
    assert compiled_identity != eager_identity


def test_cuda_graph_invalidate_resolves_adapter_for_replaced_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import clearvla.mainline.training.cuda_graph as cuda_graph_module

    new_model = _CompileModel()
    old_adapter = SimpleNamespace(name="old")
    prepared: list[tuple[str, nn.Module]] = []

    class _NewAdapter:
        name = "new"

        def prepare(self, engine: Any) -> None:
            prepared.append((self.name, engine.model))

        def structure_signature(self, engine: Any) -> tuple[str, int]:
            return (self.name, id(engine.model))

    new_adapter = _NewAdapter()
    monkeypatch.setattr(
        cuda_graph_module,
        "resolve_training_acceleration_adapter",
        lambda model: new_adapter if model is new_model else old_adapter,
    )
    runner = object.__new__(CudaGraphTrainingStepRunner)
    runner.engine = SimpleNamespace(model=new_model)
    runner._acceleration_adapter = old_adapter
    runner._release_capture = lambda: None
    runner._configure_compile_plan = lambda: None
    runner.invalidate("model-replaced")
    assert runner._acceleration_adapter is new_adapter
    assert prepared == [("new", new_model)]
    assert runner._structure_signature == ("new", id(new_model))
