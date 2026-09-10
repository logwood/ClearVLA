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

    def _variance_floored_centered_norm(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return value, value.mean()

    def _configured_typed_microgrid_expectation(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        return value + 1.0

    def _configured_typed_microgrid_rgb_detail_contraction(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        return value + 2.0

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
    assert summary["compiled_targets"] == [["block", "forward"]]


def test_plan_can_compile_one_non_forward_method_without_replacing_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CompileModel()
    engine = SimpleNamespace(model=model)
    original_forward = model.block.forward
    original_method = model.block._configured_typed_microgrid_expectation
    calls: list[str] = []

    def fake_compile(method: Any, **kwargs: Any) -> Any:
        del kwargs
        calls.append(method.__name__)

        def compiled(*args: Any, **call_kwargs: Any) -> Any:
            return method(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    plan = TrainingCompilePlan(
        name="toy-method-target",
        regions=(
            TrainingCompileRegion(
                name="microgrid",
                module_paths=("block",),
                method_name="_configured_typed_microgrid_expectation",
            ),
        ),
        acceptance="strict-gated",
    )

    applied = apply_training_compile_plan(engine, plan, profile="toy-method")

    assert calls == ["_configured_typed_microgrid_expectation"]
    assert model.block.forward == original_forward
    assert model.block._configured_typed_microgrid_expectation != original_method
    torch.testing.assert_close(
        model.block._configured_typed_microgrid_expectation(torch.zeros(1)),
        torch.ones(1),
        rtol=0.0,
        atol=0.0,
    )
    assert applied.compiled_module_paths == ("block",)
    assert applied.compiled_targets == (
        ("block", "_configured_typed_microgrid_expectation"),
    )


def test_retain_graph_safe_plan_disables_aot_buffer_donation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch._functorch import config as aot_autograd_config

    model = _CompileModel()
    engine = SimpleNamespace(model=model)
    monkeypatch.setattr(aot_autograd_config, "donated_buffer", True)
    monkeypatch.setattr(torch, "compile", lambda method, **_kwargs: method)
    monkeypatch.setattr(torch.compiler, "disable", _fake_disable)
    plan = replace(_plan(), disable_aot_autograd_buffer_donation=True)

    applied = apply_training_compile_plan(
        engine,
        plan,
        profile="toy-retain-graph",
        allow_candidate=True,
    )

    assert aot_autograd_config.donated_buffer is False
    assert applied.disable_aot_autograd_buffer_donation is True
    assert applied.summary()["aot_autograd_buffer_donation"] == "disabled"


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
        disable_aot_autograd_buffer_donation=True,
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
    from torch._functorch import config as aot_autograd_config

    monkeypatch.setattr(aot_autograd_config, "donated_buffer", True)
    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        apply_training_compile_plan(engine, rollback_plan, profile="rollback")
    assert "forward" not in model.block.__dict__
    assert "forward" not in model.head.__dict__
    assert not hasattr(model.block, "_clearvla_training_compile_transforms")
    assert not hasattr(model.head, "_clearvla_training_compile_transforms")
    assert get_applied_training_compile_plan(engine) is None
    assert aot_autograd_config.donated_buffer is True


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
    decoder.evidence_adapter = _Block()
    decoder.organizer = _Block()
    decoder.operator_contractions = nn.ModuleList((_Block(), _Block()))
    decoder.execution_controller = _Block()
    layer_heads = nn.ModuleList((_NamedContainer(), _NamedContainer()))
    if modular:
        model.conditioning = _NamedContainer()
        model.conditioning.history_proposal = _Block()
        model.bridge = _NamedContainer()
        model.bridge.query_encoder = _Block()
        model.intent = _NamedContainer()
        model.intent.organizer = _Block()
        model.intent.coarse_action = _Block()
        model.policy_compiler = _NamedContainer()
        model.policy_compiler.effect_reader = _Block()
        model.policy_compiler.consequence = _Block()
        model.policy_compiler.plan_compiler = _Block()
        model.execution_bottom = _NamedContainer()
        model.execution_bottom.decoder = decoder
        model.execution_bottom.layer_contract_heads = layer_heads
        model.grounding = _NamedContainer()
        model.grounding.blocks = nn.ModuleList((_Block(),))
        model.p1 = _NamedContainer()
        model.p1.factual_reader = _Block()
        model.p1.factual_reader.typed_local_refiners = nn.ModuleList(
            (_Block(), _Block())
        )
        model.p1.dynamic_policy_block = _Block()
        model.world = _NamedContainer()
        model.world.dynamics = _NamedContainer()
        model.world.dynamics.w1 = _Block()
        model.world.dynamics.w2 = _Block()
    else:
        model.history_proposal = _Block()
        model.factual_reader = _Block()
        model.factual_reader.typed_local_refiners = nn.ModuleList(
            (_Block(), _Block())
        )
        model.bottom = _NamedContainer()
        model.bottom.decoder = decoder
        model.bottom.layer_contract_heads = layer_heads
        model.bottom.query_encoder = _Block()
        model.bottom.p1_policy_block = _Block()
        model.top = _NamedContainer()
        model.top.grounding_blocks = nn.ModuleList((_Block(),))
        model.top.intent = _Block()
        model.top.coarse_action = _Block()
        model.top.effect_reader = _Block()
        model.top.consequence = _Block()
        model.top.plan_compiler = _Block()
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
    assert any(
        boundary.method_name == "_variance_floored_centered_norm"
        and boundary.reason
        == "preserve-eager-centered-normalization-reduction"
        for boundary in plan.eager_boundaries
    )


@pytest.mark.parametrize(
    ("scope", "expected_paths"),
    (
        ("conditioning", {"conditioning.history_proposal"}),
        ("bridge", {"bridge.query_encoder"}),
        ("intent", {"intent.organizer", "intent.coarse_action"}),
        ("factual", {"p1.factual_reader"}),
        ("factual-microgrid", {"p1.factual_reader"}),
        ("factual-microgrid-values", {"p1.factual_reader"}),
        (
            "factual-refiners",
            {
                "p1.factual_reader.typed_local_refiners.0",
                "p1.factual_reader.typed_local_refiners.1",
            },
        ),
        (
            "policy",
            {
                "policy_compiler.effect_reader",
                "policy_compiler.consequence",
                "policy_compiler.plan_compiler",
            },
        ),
        (
            "execution",
            {
                "execution_bottom.decoder.evidence_adapter",
                "execution_bottom.decoder.organizer",
                "execution_bottom.decoder.operator_contractions.0",
                "execution_bottom.decoder.operator_contractions.1",
                "execution_bottom.decoder.execution_controller",
            },
        ),
        (
            "mainline-grounding",
            {"grounding.blocks.0"},
        ),
        (
            "mainline-grounding-safe",
            {"grounding.blocks.0"},
        ),
        ("mainline-p1", {"p1.dynamic_policy_block"}),
        ("mainline-world", {"world.dynamics.w1", "world.dynamics.w2"}),
        ("mainline-transition", {"transition.v120_transition"}),
        (
            "mainline-layer-heads",
            {
                "execution_bottom.layer_contract_heads.0",
                "execution_bottom.layer_contract_heads.1",
            },
        ),
    ),
)
def test_mainline_adapter_resolves_semantic_compile_families(
    scope: str,
    expected_paths: set[str],
) -> None:
    model = _layout_model(modular=True)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        f"precision-{scope}",
    )

    plan.validate()
    assert {region.name for region in plan.regions} == {scope}
    assert set(plan.regions[0].module_paths) == expected_paths


@pytest.mark.parametrize("modular", (False, True))
def test_mainline_subfamilies_recompose_existing_mainline_order(modular: bool) -> None:
    model = _layout_model(modular=modular)
    adapter = MainlineTrainingAccelerationAdapter()
    engine = SimpleNamespace(model=model)

    full = adapter.compile_plan(engine, "precision-mainline")
    split = adapter.compile_plan(
        engine,
        "precision-mainline-grounding+mainline-p1+mainline-world+"
        "mainline-transition+mainline-layer-heads",
    )

    full_paths = tuple(path for region in full.regions for path in region.module_paths)
    split_paths = tuple(path for region in split.regions for path in region.module_paths)
    assert split_paths == full_paths


@pytest.mark.parametrize(
    ("modular", "expected_path"),
    (
        (True, "grounding.blocks.0"),
        (False, "top.grounding_blocks.0"),
    ),
)
def test_grounding_safe_uses_only_version_owned_normalization_boundary(
    modular: bool,
    expected_path: str,
) -> None:
    model = _layout_model(modular=modular)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "precision-mainline-grounding-safe",
    )

    plan.validate()
    assert plan.regions[0].module_paths == (expected_path,)
    assert plan.eager_boundaries == (
        TrainingEagerBoundary(
            module_path=expected_path,
            method_name="_variance_floored_centered_norm",
            reason="preserve-eager-centered-normalization-reduction",
        ),
    )


@pytest.mark.parametrize("modular", (False, True))
def test_mixed_safe_profile_recomposes_mainline_without_duplicate_targets(
    modular: bool,
) -> None:
    model = _layout_model(modular=modular)
    adapter = MainlineTrainingAccelerationAdapter()
    engine = SimpleNamespace(model=model)
    full_mainline = adapter.compile_plan(engine, "precision-mainline")
    mixed = adapter.compile_plan(
        engine,
        "precision-mainline-grounding-safe+mainline-p1+mainline-world+"
        "mainline-transition+mainline-layer-heads+factual-shell-safe",
    )

    mixed.validate()
    full_paths = tuple(
        path for region in full_mainline.regions for path in region.module_paths
    )
    mixed_paths = tuple(
        path for region in mixed.regions for path in region.module_paths
    )
    factual_path = "p1.factual_reader" if modular else "factual_reader"
    assert mixed_paths == (*full_paths, factual_path)
    grounding_prefix = "grounding.blocks." if modular else "top.grounding_blocks."
    grounding_boundaries = tuple(
        boundary
        for boundary in mixed.eager_boundaries
        if boundary.module_path.startswith(grounding_prefix)
    )
    assert grounding_boundaries
    assert all(
        boundary.method_name == "_variance_floored_centered_norm"
        for boundary in grounding_boundaries
    )
    assert mixed.eager_boundaries[-1].module_path == factual_path


def test_mainline_adapter_composes_semantic_families_without_changing_aliases() -> None:
    model = _layout_model(modular=True)
    adapter = MainlineTrainingAccelerationAdapter()

    composed = adapter.compile_plan(
        SimpleNamespace(model=model),
        "precision-mainline+intent+factual",
    )
    combined = adapter.compile_plan(
        SimpleNamespace(model=model),
        "precision-combined",
    )

    composed.validate()
    assert {region.name for region in composed.regions} == {
        "mainline",
        "intent",
        "factual",
    }
    # ``combined`` predates the semantic-family sweep.  Keep its exact three
    # region identities so an old accepted fingerprint cannot silently move.
    assert {region.name for region in combined.regions} == {
        "visual",
        "mainline",
        "mmdit",
    }


def test_mainline_adapter_hybrid_composition_partitions_only_new_families() -> None:
    model = _layout_model(modular=True)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "hybrid-mainline+intent+factual",
    )

    plan.validate()
    assert {region.name for region in plan.regions} == {
        "mainline",
        "intent",
        "factual",
    }
    boundary_paths = {boundary.module_path for boundary in plan.eager_boundaries}
    assert any(path.startswith("intent.") for path in boundary_paths)
    assert any(path.startswith("p1.factual_reader.") for path in boundary_paths)
    assert not any(path.startswith("grounding.blocks.") for path in boundary_paths)
    assert (
        plan.numerical_policy
        == "partition-intent+factual-and-force-precision-mainline"
    )


def test_mainline_adapter_partitions_factual_refiners_below_eager_parent() -> None:
    model = _layout_model(modular=False)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "hybrid-mainline+factual-refiners",
    )

    plan.validate()
    regions = {region.name: region for region in plan.regions}
    assert set(regions) == {"mainline", "factual-refiners"}
    assert regions["factual-refiners"].module_paths == (
        "factual_reader.typed_local_refiners.0",
        "factual_reader.typed_local_refiners.1",
    )
    assert "factual_reader" not in {
        path for region in plan.regions for path in region.module_paths
    }
    assert not any(
        boundary.module_path == "factual_reader"
        for boundary in plan.eager_boundaries
    )
    assert (
        plan.numerical_policy
        == "partition-factual-refiners-and-force-precision-mainline"
    )


def test_mainline_adapter_targets_only_factual_microgrid_method() -> None:
    model = _layout_model(modular=False)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "hybrid-mainline+factual-microgrid",
    )

    plan.validate()
    regions = {region.name: region for region in plan.regions}
    assert set(regions) == {"mainline", "factual-microgrid"}
    microgrid = regions["factual-microgrid"]
    assert microgrid.module_paths == ("factual_reader",)
    assert (
        microgrid.method_name
        == "_configured_typed_microgrid_expectation"
    )
    assert not any(
        boundary.module_path == "factual_reader"
        for boundary in plan.eager_boundaries
    )
    assert (
        plan.numerical_policy
        == "partition-factual-microgrid-and-force-precision-mainline"
    )


def test_mainline_adapter_targets_only_normalized_factual_value_contraction() -> None:
    model = _layout_model(modular=False)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "hybrid-mainline+factual-microgrid-values",
    )

    plan.validate()
    regions = {region.name: region for region in plan.regions}
    assert set(regions) == {"mainline", "factual-microgrid-values"}
    contraction = regions["factual-microgrid-values"]
    assert contraction.module_paths == ("factual_reader",)
    assert contraction.method_name == (
        "_configured_typed_microgrid_rgb_detail_contraction"
    )
    assert not any(
        boundary.module_path == "factual_reader"
        for boundary in plan.eager_boundaries
    )
    assert plan.numerical_policy == (
        "partition-factual-microgrid-values-and-force-precision-mainline"
    )


@pytest.mark.parametrize(
    ("family", "eager_method", "reason"),
    (
        (
            "factual-shell-eager-values",
            "_configured_typed_microgrid_rgb_detail_contraction",
            "preserve-eager-normalized-rgb-detail-contraction",
        ),
        (
            "factual-shell-eager-microgrid",
            "_configured_typed_microgrid_expectation",
            "preserve-eager-complete-microgrid-expectation",
        ),
    ),
)
def test_mainline_adapter_compiles_factual_shell_around_explicit_eager_method(
    family: str,
    eager_method: str,
    reason: str,
) -> None:
    model = _layout_model(modular=False)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        f"hybrid-mainline+{family}",
    )

    plan.validate()
    regions = {region.name: region for region in plan.regions}
    assert set(regions) == {"mainline", family}
    shell = regions[family]
    assert shell.module_paths == ("factual_reader",)
    assert shell.method_name == "forward"
    assert plan.eager_boundaries == (
        TrainingEagerBoundary(
            module_path="factual_reader",
            method_name=eager_method,
            reason=reason,
        ),
    )
    assert plan.numerical_policy == (
        f"partition-{family}-and-force-precision-mainline"
    )


def test_factual_shell_eager_island_is_owned_by_family_not_strategy() -> None:
    model = _layout_model(modular=True)
    family = "factual-shell-eager-values"

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        f"precision-{family}",
    )

    plan.validate()
    assert {region.name for region in plan.regions} == {family}
    assert plan.eager_boundaries == (
        TrainingEagerBoundary(
            module_path="p1.factual_reader",
            method_name=(
                "_configured_typed_microgrid_rgb_detail_contraction"
            ),
            reason="preserve-eager-normalized-rgb-detail-contraction",
        ),
    )
    assert plan.rng_policy == "eager-boundary-or-explicit-input"


def test_factual_shell_safe_prefers_narrow_value_boundary() -> None:
    model = _layout_model(modular=True)

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "precision-mainline+factual-shell-safe",
    )

    plan.validate()
    assert plan.eager_boundaries == (
        TrainingEagerBoundary(
            module_path="p1.factual_reader",
            method_name=(
                "_configured_typed_microgrid_rgb_detail_contraction"
            ),
            reason="preserve-eager-normalized-rgb-detail-contraction",
        ),
    )


def test_factual_shell_safe_falls_back_to_complete_microgrid_boundary() -> None:
    model = _layout_model(modular=False)
    model.factual_reader._configured_typed_microgrid_rgb_detail_contraction = None

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "precision-mainline+factual-shell-safe",
    )

    plan.validate()
    assert plan.eager_boundaries == (
        TrainingEagerBoundary(
            module_path="factual_reader",
            method_name="_configured_typed_microgrid_expectation",
            reason="preserve-eager-complete-microgrid-expectation",
        ),
    )


def test_atomic_factual_family_requires_adapter_opt_in() -> None:
    model = _layout_model(modular=True)
    adapter = MainlineTrainingAccelerationAdapter()
    engine = SimpleNamespace(model=model)

    with pytest.raises(ValueError, match="cannot resolve families"):
        adapter.compile_plan(engine, "precision-mainline+factual-atomic")

    original = model.p1.factual_reader._configured_typed_microgrid_rgb_detail_contraction
    state_keys = tuple(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    buffer_ids = tuple(id(buffer) for buffer in model.buffers())
    assert adapter.set_atomic_factual_contraction(engine, enabled=True) == 1
    assert adapter.set_atomic_factual_contraction(engine, enabled=True) == 0

    plan = adapter.compile_plan(engine, "precision-mainline+factual-atomic")
    plan.validate()
    atomic_region = next(
        region for region in plan.regions if region.name == "factual-atomic"
    )
    assert atomic_region.module_paths == ("p1.factual_reader",)
    assert atomic_region.method_name == "forward"
    assert not any(
        boundary.module_path == "p1.factual_reader"
        for boundary in plan.eager_boundaries
    )
    assert tuple(model.state_dict()) == state_keys
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(id(buffer) for buffer in model.buffers()) == buffer_ids

    assert adapter.set_atomic_factual_contraction(engine, enabled=False) == 1
    assert adapter.set_atomic_factual_contraction(engine, enabled=False) == 0
    restored = model.p1.factual_reader._configured_typed_microgrid_rgb_detail_contraction
    assert restored == original


def test_mainline_adapter_semantic_families_resolve_legacy_paths() -> None:
    model = _layout_model(modular=False)
    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        SimpleNamespace(model=model),
        "precision-conditioning+bridge+intent+factual+policy+execution",
    )

    plan.validate()
    paths = {path for region in plan.regions for path in region.module_paths}
    assert "history_proposal" in paths
    assert "bottom.query_encoder" in paths
    assert {"top.intent", "top.coarse_action"}.issubset(paths)
    assert "factual_reader" in paths
    assert {
        "top.effect_reader",
        "top.consequence",
        "top.plan_compiler",
    }.issubset(paths)
    assert "bottom.decoder.execution_controller" in paths


@pytest.mark.parametrize(
    "strategy", ("fast", "precision", "partitioned")
)
def test_mainline_adapter_core_scope_excludes_visual_family(strategy: str) -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        f"{strategy}-core",
    )

    plan.validate()
    assert {region.name for region in plan.regions} == {"mainline", "mmdit"}
    assert all(
        not path.startswith("observation.")
        for region in plan.regions
        for path in region.module_paths
    )
    assert bool(plan.eager_boundaries) is (strategy == "partitioned")
    assert plan.disable_aot_autograd_buffer_donation is True


@pytest.mark.parametrize(
    "strategy", ("fast", "precision", "partitioned")
)
@pytest.mark.parametrize(
    ("scope", "expected_suffix"),
    (("flow", ".flow"), ("raw", ".raw_flow")),
)
def test_mainline_adapter_can_isolate_visual_owner_compile_scopes(
    strategy: str,
    scope: str,
    expected_suffix: str,
) -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        f"{strategy}-{scope}",
    )

    plan.validate()
    assert {region.name for region in plan.regions} == {scope}
    assert plan.regions[0].module_paths
    assert all(
        expected_suffix in path
        for path in plan.regions[0].module_paths
    )
    assert plan.numerical_policy == {
        "fast": "unrestricted-inductor-candidate",
        "precision": "force-precision-with-inductor-reduction-order",
        "partitioned": "partition-known-reductions-and-force-precision",
    }[strategy]


@pytest.mark.parametrize(
    ("profile", "expected_families", "donation_disabled"),
    (
        ("hybrid-mainline-flow", {"flow", "mainline"}, False),
        ("hybrid-core-flow", {"flow", "mainline", "mmdit"}, True),
        ("hybrid-mainline-mmdit", {"mainline", "mmdit"}, True),
    ),
)
def test_mainline_adapter_builds_hybrid_flow_profiles(
    profile: str,
    expected_families: set[str],
    donation_disabled: bool,
) -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    plan = MainlineTrainingAccelerationAdapter().compile_plan(engine, profile)

    plan.validate()
    assert {region.name for region in plan.regions} == expected_families
    if "flow" in expected_families:
        flow_region = next(region for region in plan.regions if region.name == "flow")
        assert "observation.compiler.encoder.flow" not in flow_region.module_paths
        assert "observation.compiler.encoder.flow.encoder" in flow_region.module_paths
        assert plan.numerical_policy == "partition-flow-and-force-precision-mainline"
    else:
        assert plan.numerical_policy == "partition-mmdit-and-force-precision-mainline"
    assert plan.disable_aot_autograd_buffer_donation is donation_disabled


def test_mainline_adapter_rejects_ambiguous_hybrid_profile() -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    with pytest.raises(ValueError, match="hybrid selective compile profiles"):
        MainlineTrainingAccelerationAdapter().compile_plan(
            engine,
            "hybrid-mainline",
        )


def test_mainline_adapter_autotunes_without_relaxing_precision_options() -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    precision = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        "precision-mainline",
    )
    autotuned = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        "autotune-mainline",
    )

    precision_options = dict(precision.regions[0].options)
    autotuned_options = dict(autotuned.regions[0].options)
    assert autotuned_options.items() >= precision_options.items()
    assert autotuned_options["max_autotune"] is True
    assert autotuned_options["coordinate_descent_tuning"] is True
    assert autotuned.regions[0].mode is None
    assert not autotuned.eager_boundaries
    assert (
        autotuned.numerical_policy
        == "force-precision-with-autotuned-kernel-schedules"
    )


def test_mainline_adapter_fast_rng_keeps_only_random_fallback_guard() -> None:
    model = _layout_model(modular=True)
    engine = SimpleNamespace(model=model)

    plan = MainlineTrainingAccelerationAdapter().compile_plan(
        engine,
        "fast-rng-mainline",
    )

    plan.validate()
    assert dict(plan.regions[0].options) == {"fallback_random": True}
    assert plan.regions[0].mode is None
    assert plan.numerical_policy == "unrestricted-inductor-with-fallback-random"


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
