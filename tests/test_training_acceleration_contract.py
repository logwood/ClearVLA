from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from test_mainline_policy import _config as mainline_test_config
from torch import nn

from clearvla.mainline.train import (
    _overrides as apply_training_overrides,
)
from clearvla.mainline.train import (
    _parser as training_parser,
)
from clearvla.mainline.train import (
    _resolve_mainline_training_acceleration_adapter,
)
from clearvla.mainline.training.acceleration_adapters import (
    MainlineTrainingAccelerationAdapter,
    _set_module_cudnn_default_guard,
)
from clearvla.mainline.training.acceleration_contract import (
    GenericTrainingAccelerationAdapter,
    TrainingCompilePlan,
    TrainingCompileRegion,
    TrainingEagerBoundary,
    module_structure_signature,
    resolve_training_acceleration_adapter,
    resolve_training_compile_plan,
    set_cudnn_forward_benchmark_phase,
)
from clearvla.mainline.v120_core.flow_dino_evidence import _DenseRawFlowRefiner


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


class _CudnnFlagProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.seen.append(bool(torch.backends.cudnn.benchmark))
        return value


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


def test_legacy_adapter_needs_no_compile_plan_until_profile_is_requested() -> None:
    model = _ToyPolicy(8, with_adapter=True)
    engine = type("Engine", (), {"model": model})()
    adapter = resolve_training_acceleration_adapter(model)
    assert resolve_training_compile_plan(adapter, engine, None) is None
    try:
        resolve_training_compile_plan(adapter, engine, "precision-mainline")
    except ValueError as error:
        assert "does not support compile profile" in str(error)
    else:
        raise AssertionError("an adapter without compile_plan accepted a profile")


def test_compile_plan_signature_covers_regions_options_and_boundaries() -> None:
    base = TrainingCompilePlan(
        name="toy-plan",
        regions=(
            TrainingCompileRegion(
                name="body",
                module_paths=("block",),
                options=(("force_same_precision", True),),
            ),
        ),
        eager_boundaries=(
            TrainingEagerBoundary(
                module_path="block.norm",
                reason="preserve-eager-norm",
            ),
        ),
    )
    changed_region = replace(
        base,
        regions=(replace(base.regions[0], module_paths=("head",)),),
        eager_boundaries=(),
    )
    changed_options = replace(
        base,
        regions=(
            replace(
                base.regions[0],
                options=(("force_same_precision", False),),
            ),
        ),
    )
    changed_boundary = replace(
        base,
        eager_boundaries=(
            replace(base.eager_boundaries[0], reason="different-policy"),
        ),
    )
    changed_donation = replace(
        base,
        disable_aot_autograd_buffer_donation=True,
    )
    changed_method = replace(
        base,
        regions=(
            replace(
                base.regions[0],
                method_name="_configured_typed_microgrid_expectation",
            ),
        ),
    )
    signatures = {
        base.signature(),
        changed_region.signature(),
        changed_options.signature(),
        changed_boundary.signature(),
        changed_donation.signature(),
        changed_method.signature(),
    }
    assert len(signatures) == 6


def test_mainline_adapter_can_disable_only_selected_checkpoint_surfaces() -> None:
    raw_modules = {
        name: nn.Identity()
        for name in ("pyramid", "mid", "high", "early_masked_raw_context")
    }
    for module in raw_modules.values():
        module.activation_checkpoint = True
    raw_flow = SimpleNamespace(**raw_modules)
    factual_reader = nn.Identity()
    factual_reader.raw_activation_checkpoint = True
    model = SimpleNamespace(
        observation=SimpleNamespace(
            compiler=SimpleNamespace(encoder=SimpleNamespace(raw_flow=raw_flow))
        ),
        factual_reader=factual_reader,
    )
    engine = SimpleNamespace(model=model)
    adapter = MainlineTrainingAccelerationAdapter()

    changed = adapter.set_activation_checkpointing(
        engine,
        enabled=False,
        scopes=("raw_high", "p1"),
    )

    assert changed == 2
    assert raw_modules["high"].activation_checkpoint is False
    assert factual_reader.raw_activation_checkpoint is False
    assert raw_modules["pyramid"].activation_checkpoint is True
    assert raw_modules["mid"].activation_checkpoint is True
    assert raw_modules["early_masked_raw_context"].activation_checkpoint is True


def test_mainline_adapter_installs_selective_raw_mid_checkpoint_policy() -> None:
    mid = nn.Identity()
    mid.activation_checkpoint = True
    mid.checkpoint_save_operations = ()
    model = SimpleNamespace(
        observation=SimpleNamespace(
            compiler=SimpleNamespace(
                encoder=SimpleNamespace(raw_flow=SimpleNamespace(mid=mid))
            )
        )
    )
    engine = SimpleNamespace(model=model)
    adapter = MainlineTrainingAccelerationAdapter()

    changed = adapter.set_selective_checkpoint_save_operations(
        engine,
        scope="raw_mid",
        operations=("convolution", "linear", "convolution"),
    )

    assert changed == 1
    assert mid.checkpoint_save_operations == ("convolution", "linear")


def test_selective_raw_mid_checkpoint_preserves_cpu_update_surface() -> None:
    torch.manual_seed(197)
    reference = _DenseRawFlowRefiner(
        8,
        8,
        radius=1,
        uncertainty_floor=0.05,
        activation_checkpoint=True,
        preserve_uncertain_seed=True,
        bounded_coordinates=True,
        normalization_floor=0.10,
    )
    candidate = copy.deepcopy(reference)
    candidate.checkpoint_save_operations = (
        "convolution",
        "linear",
        "grid_sample",
    )
    input_rows = (
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 2, 4, 4),
        torch.rand(2, 1, 4, 4),
    )

    def run(module: _DenseRawFlowRefiner):
        inputs = tuple(value.detach().clone().requires_grad_(True) for value in input_rows)
        result = module(*inputs)
        tensors = (
            result.flow,
            result.information,
            result.uncertainty,
            result.correlation_entropy,
            result.correlation_margin,
            *result.iterations,
        )
        loss = sum(value.float().square().mean() for value in tensors)
        loss.backward()
        return (
            tuple(value.detach().clone() for value in tensors),
            tuple(value.grad.detach().clone() for value in inputs),
            tuple(
                parameter.grad.detach().clone()
                for parameter in module.parameters()
            ),
        )

    expected = run(reference)
    actual = run(candidate)
    for actual_group, expected_group in zip(actual, expected, strict=True):
        assert len(actual_group) == len(expected_group)
        for actual_value, expected_value in zip(
            actual_group,
            expected_group,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=0.0,
                atol=0.0,
            )


def test_mainline_adapter_context_reuse_is_one_optional_version_hook() -> None:
    decoder = SimpleNamespace()
    engine = SimpleNamespace(
        model=SimpleNamespace(
            execution_bottom=SimpleNamespace(decoder=decoder),
        )
    )
    adapter = MainlineTrainingAccelerationAdapter()

    assert adapter.set_context_reuse(engine, enabled=True) == 3
    assert decoder._reuse_prepared_block_contexts is True
    assert decoder._reuse_prepared_controller_context is True
    assert decoder._reuse_terminal_candidate_velocity is True


def test_mainline_adapter_scopes_cudnn_benchmark_without_state_abi_change() -> None:
    probes = {
        name: _CudnnFlagProbe()
        for name in ("flow", "raw_pyramid", "raw_mid", "raw_high", "raw_context")
    }
    encoder = SimpleNamespace(
        flow=probes["flow"],
        raw_flow=SimpleNamespace(
            pyramid=probes["raw_pyramid"],
            mid=probes["raw_mid"],
            high=probes["raw_high"],
        ),
        early_masked_raw_context=probes["raw_context"],
    )
    engine = SimpleNamespace(
        model=SimpleNamespace(
            observation=SimpleNamespace(compiler=SimpleNamespace(encoder=encoder))
        )
    )
    adapter = MainlineTrainingAccelerationAdapter()
    before_keys = {
        name: tuple(probe.state_dict()) for name, probe in probes.items()
    }
    original_benchmark = bool(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = False
    try:
        changed = adapter.set_cudnn_benchmark_scopes(
            engine,
            enabled=True,
            scopes=("flow", "raw_mid", "raw_high"),
        )
        assert changed == 3
        value = torch.ones(1)
        for probe in probes.values():
            probe(value)
        assert probes["flow"].seen == [True]
        assert probes["raw_mid"].seen == [True]
        assert probes["raw_high"].seen == [True]
        assert probes["raw_pyramid"].seen == [False]
        assert probes["raw_context"].seen == [False]
        assert torch.backends.cudnn.benchmark is False
        assert {
            name: tuple(probe.state_dict()) for name, probe in probes.items()
        } == before_keys

        changed = adapter.set_cudnn_benchmark_scopes(
            engine,
            enabled=False,
            scopes=("flow", "raw_mid", "raw_high"),
        )
        assert changed == 3
        for probe in probes.values():
            probe(value)
        assert all(probe.seen[-1] is False for probe in probes.values())
    finally:
        torch.backends.cudnn.benchmark = original_benchmark


def test_mainline_adapter_rejects_unknown_cudnn_benchmark_scope() -> None:
    adapter = MainlineTrainingAccelerationAdapter()
    engine = SimpleNamespace(model=SimpleNamespace(observation=None))
    try:
        adapter.set_cudnn_benchmark_scopes(
            engine,
            enabled=True,
            scopes=("whole_model",),
        )
    except ValueError as error:
        assert "unknown=['whole_model']" in str(error)
    else:
        raise AssertionError("unknown cuDNN benchmark scope was accepted")


def test_cudnn_default_guard_is_balanced_across_repeated_forward_calls() -> None:
    events: list[tuple[str, bool]] = []

    class FlagConv(nn.Conv2d):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            events.append(("forward", bool(torch.backends.cudnn.benchmark)))
            return super().forward(value)

    module = FlagConv(3, 4, 3, padding=1, bias=False)
    before_keys = tuple(module.state_dict())
    assert _set_module_cudnn_default_guard(module, enabled=True) is True
    module.weight.register_hook(
        lambda gradient: (
            events.append(("backward", bool(torch.backends.cudnn.benchmark))),
            gradient,
        )[1]
    )
    original_benchmark = bool(torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.benchmark = True
        value = torch.randn(2, 3, 8, 8, requires_grad=True)
        output = module(value) + module(value)
        assert torch.backends.cudnn.benchmark is True
        output.sum().backward()
        assert events == [
            ("forward", False),
            ("forward", False),
            ("backward", True),
        ]
        assert torch.backends.cudnn.benchmark is True
        assert tuple(module.state_dict()) == before_keys
        assert _set_module_cudnn_default_guard(module, enabled=False) is True
        assert tuple(module.state_dict()) == before_keys
    finally:
        torch.backends.cudnn.benchmark = original_benchmark


def test_forward_benchmark_phase_composes_with_default_conv_guard() -> None:
    guarded = _CudnnFlagProbe()
    ordinary = _CudnnFlagProbe()
    assert _set_module_cudnn_default_guard(guarded, enabled=True) is True
    engine = SimpleNamespace()

    def forward(value: torch.Tensor) -> torch.Tensor:
        return ordinary(guarded(value))

    engine._forward = forward
    original_benchmark = bool(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = False
    try:
        assert set_cudnn_forward_benchmark_phase(engine, enabled=True) is True
        assert engine._forward(torch.ones(1)).item() == 1.0
        assert guarded.seen == [False]
        assert ordinary.seen == [True]
        assert torch.backends.cudnn.benchmark is False

        def failing_forward(_value: torch.Tensor) -> torch.Tensor:
            assert torch.backends.cudnn.benchmark is True
            raise RuntimeError("expected failure")

        assert set_cudnn_forward_benchmark_phase(engine, enabled=False) is True
        engine._forward = failing_forward
        assert set_cudnn_forward_benchmark_phase(engine, enabled=True) is True
        with pytest.raises(RuntimeError, match="expected failure"):
            engine._forward(torch.ones(1))
        assert torch.backends.cudnn.benchmark is False
        assert set_cudnn_forward_benchmark_phase(engine, enabled=False) is True
    finally:
        torch.backends.cudnn.benchmark = original_benchmark


@torch.no_grad()
def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


@torch.no_grad()
def _assert_optimizer_state_exact(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert actual.keys() == expected.keys()
    assert actual["param_groups"] == expected["param_groups"]
    actual_state = actual["state"]
    expected_state = expected["state"]
    assert isinstance(actual_state, dict)
    assert isinstance(expected_state, dict)
    assert actual_state.keys() == expected_state.keys()
    for parameter_id in actual_state:
        actual_row = actual_state[parameter_id]
        expected_row = expected_state[parameter_id]
        assert isinstance(actual_row, dict)
        assert isinstance(expected_row, dict)
        assert actual_row.keys() == expected_row.keys()
        for name in actual_row:
            actual_value = actual_row[name]
            expected_value = expected_row[name]
            if isinstance(actual_value, torch.Tensor):
                assert isinstance(expected_value, torch.Tensor)
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


@torch.no_grad()
def _initialize_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN prewarm requires CUDA")
def test_mainline_adapter_default_prewarm_preserves_training_state_and_rng() -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    class Pyramid(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
            high = torch.nn.functional.gelu(self.conv(value))
            return high, torch.nn.functional.avg_pool2d(high, 2)

    model = nn.Module()
    model.observation = nn.Module()
    model.observation.compiler = nn.Module()
    model.observation.compiler.encoder = nn.Module()
    model.observation.compiler.encoder.raw_flow = nn.Module()
    model.observation.compiler.encoder.raw_flow.pyramid = Pyramid()
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _initialize_optimizer_state(model, optimizer)
    flow_generator = torch.Generator(device=device).manual_seed(8_311)
    condition_generator = torch.Generator(device=device).manual_seed(8_312)
    engine = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32,
        train_flow_generator=flow_generator,
        train_condition_generator=condition_generator,
    )
    batch = SimpleNamespace(
        online=SimpleNamespace(
            observation=SimpleNamespace(
                raw_rgb=torch.randn(2, 3, 1, 3, 16, 16, device=device)
            )
        )
    )
    torch.manual_seed(8_313)
    torch.cuda.manual_seed_all(8_314)
    before_model = _clone_state_dict(model)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(device).clone()
    before_flow_rng = flow_generator.get_state().clone()
    before_condition_rng = condition_generator.get_state().clone()
    original_benchmark = bool(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = False
    try:
        changed = MainlineTrainingAccelerationAdapter().prewarm_cudnn_default_scopes(
            engine,
            batch,
            scopes=("raw_pyramid",),
        )
        assert changed == 1
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.benchmark = original_benchmark

    after_model = model.state_dict()
    assert after_model.keys() == before_model.keys()
    for name, expected in before_model.items():
        assert torch.equal(after_model[name], expected), name
    _assert_optimizer_state_exact(optimizer.state_dict(), before_optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), before_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), before_cuda_rng)
    assert torch.equal(flow_generator.get_state(), before_flow_rng)
    assert torch.equal(condition_generator.get_state(), before_condition_rng)


def test_training_cli_serializes_explicit_acceleration_policy() -> None:
    args = training_parser().parse_args(
        [
            "--cuda-graph-training",
            "--training-acceleration-context-reuse",
            "--disable-training-checkpoint-scope",
            "p1",
            "--disable-training-checkpoint-scope",
            "raw_high",
            "--raw-mid-checkpoint-save-operation",
            "convolution",
            "--raw-mid-checkpoint-save-operation",
            "linear",
        ]
    )
    assert args.cuda_graph_training is True
    assert args.training_acceleration_context_reuse is True
    assert args.disable_training_checkpoint_scope == ["p1", "raw_high"]
    assert args.raw_mid_checkpoint_save_operation == ["convolution", "linear"]


def test_smoke_cli_can_schedule_graph_and_diagnostic_batches() -> None:
    args = training_parser().parse_args(["--smoke", "--log-every", "2"])

    config = apply_training_overrides(mainline_test_config(), args)

    assert config.runtime.log_every == 2


def test_training_entry_installs_current_adapter_for_plain_mainline_policy() -> None:
    model = nn.Linear(2, 2)

    adapter = _resolve_mainline_training_acceleration_adapter(model)  # type: ignore[arg-type]

    assert adapter.name == "clearvla-mainline-cuda-graph-v1"
    assert isinstance(
        model.training_acceleration_adapter,
        MainlineTrainingAccelerationAdapter,
    )
