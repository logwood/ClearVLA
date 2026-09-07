from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
import torch

import clearvla.mainline.model.v120_p1 as p1_module
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.v120_p1 import (
    _FunctionalOwnershipLocalRefiner,
    _UtilityPrecisionLocalRefiner,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.randn(2, 2, 9, 3, requires_grad=True),
        "learned_detail": torch.randn(2, 2, 9, 12, requires_grad=True),
        "coordinates": torch.randn(2, 2, 9, 2, requires_grad=True),
        **{
            name: torch.randn(2, 2, 8, requires_grad=True)
            for name in ("query", "semantic", "appearance", "geometry")
        },
        "future_transport": torch.randn(2, 2, 5, requires_grad=True),
    }


def _refiner() -> _UtilityPrecisionLocalRefiner:
    return _UtilityPrecisionLocalRefiner(width=16, raw_dim=12, route_dim=8, depth=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_empty_pixels_are_zero_and_typed_conditions_cannot_change_protected_read(dtype: torch.dtype) -> None:
    torch.manual_seed(8201)
    refiner, inputs = _refiner(), _inputs()
    records = []
    hook = refiner.delta_router.register_forward_pre_hook(lambda _m, args: records.append(args[:2]))
    try:
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            refiner(**inputs, collect_diagnostics=False)
            neutral = dict(inputs)
            for name in ("semantic", "appearance", "geometry", "future_transport"):
                neutral[name] = torch.zeros_like(inputs[name])
            output, _ = refiner(**neutral, collect_diagnostics=False)
            zero, _ = refiner(
                **{
                    **inputs,
                    "rgb": torch.zeros_like(inputs["rgb"]),
                    "learned_detail": torch.zeros_like(inputs["learned_detail"]),
                },
                collect_diagnostics=False,
            )
    finally:
        hook.remove()
    # Optional selectors cannot alter the protected policy/base read.
    assert torch.equal(records[0][0], records[1][0])
    # Zero selector metadata does not imply absent pixel evidence. An optional
    # owner may still read it; do not reinstall the rejected difference rule.
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(zero) == 0


def _historical_refiner(commit: str) -> type:
    # Use the exact Git-owned class, not a handwritten near-equivalent oracle.
    # This is a local unit test: no checkout, model load or remote call occurs.
    path = "clearvla/mainline/model/v120_p1.py"
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"missing fixed regression source {commit}:{path}"
    definition = next(
        node for node in ast.parse(result.stdout).body
        if isinstance(node, ast.ClassDef) and node.name == "_UtilityPrecisionLocalRefiner"
    )
    namespace = dict(vars(p1_module))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), f"{commit}:{path}", "exec"), namespace)
    return namespace["_UtilityPrecisionLocalRefiner"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_matches_recovery_forward_and_all_parameter_input_vjps(dtype: torch.dtype) -> None:
    torch.manual_seed(8203)
    current, inputs = _refiner(), _inputs()
    recovery = _historical_refiner("0973f192")(
        width=16, raw_dim=12, route_dim=8, depth=2
    )
    recovery.load_state_dict(current.state_dict(), strict=True)
    records = []
    for refiner in (current, recovery):
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            output, metrics = refiner(**inputs, collect_diagnostics=True)
        named = tuple(refiner.named_parameters())
        grads = torch.autograd.grad(
            output.float().square().mean(), (*inputs.values(), *(p for _, p in named)),
            allow_unused=True,
        )
        records.append((output, metrics, grads))
    assert torch.equal(records[0][0], records[1][0])
    assert records[0][1].keys() == records[1][1].keys()
    for key, value in records[0][1].items():
        assert torch.equal(value, records[1][1][key]), key
    for left, right in zip(records[0][2], records[1][2], strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.isfinite(left).all() and torch.equal(left, right)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_restoration_changes_optional_read_only_not_schema32_protected_carrier(dtype: torch.dtype) -> None:
    torch.manual_seed(8204)
    current, inputs = _refiner(), _inputs()
    previous = _historical_refiner("5dbe5ee")(
        width=16, raw_dim=12, route_dim=8, depth=2
    )
    previous.load_state_dict(current.state_dict(), strict=True)
    records = []
    for refiner in (previous, current):
        hook = refiner.delta_router.register_forward_pre_hook(
            lambda _m, args: records.append(tuple(x.detach().clone() for x in args[:2]))
        )
        try:
            with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
                refiner(**inputs, collect_diagnostics=False)
        finally:
            hook.remove()
    assert torch.equal(records[0][0], records[1][0])
    assert not torch.equal(records[0][1], records[1][1])


def test_independent_read_has_distinct_identity_but_retains_w_s_and_q5() -> None:
    selection = ComponentSelection.from_config(ExperimentConfig())
    assert selection.p1 == "v120_factual_independent_precision_dynamic_p1_v1"
    assert selection.intent == "stateless_object_intent_current_once_diag_invariant_v1"
    assert selection.world == "object_candidate_w12_typed_interval_qk_v1"
    assert "p2_independent_precision_v1" in ARCHITECTURE_MANIFEST.components.top
    assert "p2_policy_relative_precision_v1" not in ARCHITECTURE_MANIFEST.components.top


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_every_typed_owner_has_finite_vjp_without_more_parameters_or_reads(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(8202)
    refiner, inputs = _refiner(), _inputs()
    parent = _FunctionalOwnershipLocalRefiner(width=16, raw_dim=12, route_dim=8, depth=2)
    assert {k: v.shape for k, v in refiner.state_dict().items()} == {
        k: v.shape for k, v in parent.state_dict().items()
    }
    calls = []
    hooks = [
        attention.register_forward_hook(lambda *_: calls.append(1))
        for attention in (*refiner.token_attn, refiner.read_attn)
    ]
    try:
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            output, _ = refiner(**inputs, collect_diagnostics=True)
        output.float().square().mean().backward()
    finally:
        for hook in hooks:
            hook.remove()
    assert len(calls) == 3
    assert torch.isfinite(output).all()
    for name, parameter in refiner.named_parameters():
        if name.startswith(
            (
                "owner_conditions.",
                "owner_outputs.",
                "geometry_key.",
                "rgb_value.",
                "detail_value.",
                "coordinate_key.",
            )
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    for name, value in inputs.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        assert value.grad.abs().sum() > 0, name


@pytest.mark.parametrize("zero_lane,owner", [("learned_detail", "semantic"), ("rgb", "appearance")])
def test_p2_cannot_borrow_values_from_the_wrong_modality(zero_lane: str, owner: str) -> None:
    inputs = _inputs()
    inputs[zero_lane] = torch.zeros_like(inputs[zero_lane])
    _, metrics = _refiner()(**inputs, collect_diagnostics=True)
    assert metrics[f"flow_jepa_typed_p2_{owner}_contribution_rms"] == 0
