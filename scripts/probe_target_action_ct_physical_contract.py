"""Causal/ABI checks for the target-action controlled-transition outlet."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearvla.mainline import config as config_module  # noqa: E402, I001
from clearvla.mainline import interfaces  # noqa: E402
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec  # noqa: E402
from clearvla.mainline.model.components import OutletAdapter  # noqa: E402
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy  # noqa: E402
from clearvla.mainline.model.types import PhysicalTransitionInnovation  # noqa: E402
from clearvla.mainline.v120_core.time_domain_mmdit import (  # noqa: E402
    TerminalActionController,
)
from clearvla.tools import mainline_equivalence as equivalence  # noqa: E402


MODE = "target_action_bottleneck_v1"


def _gradient_rms(parameter: nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().square().mean().sqrt())


def _expect_value_error(callable_: Any) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    return False


def _terminal_contract(seed: int) -> tuple[dict[str, bool], dict[str, float]]:
    torch.manual_seed(seed)
    batch, candidates, horizon, hidden, physical = 2, 3, 24, 8, 18
    head = nn.Linear(hidden, physical, bias=False)
    nn.init.zeros_(head.weight)
    state = torch.randn(batch, horizon, hidden)
    delta = torch.stack(
        (
            torch.full((horizon, physical), 0.25),
            torch.full((horizon, physical), -0.75),
        ),
        dim=0,
    )
    continuous = TerminalActionController(
        action_norm=nn.Identity(),
        velocity_head=head,
        optional_command_head=None,
        optional_event_head=None,
        motion_head=nn.Linear(hidden, 1),
        arm_dim=6,
    )
    continuous_value = continuous.predict_candidate_velocity(
        state,
        physical_transition_delta=delta,
    )
    binary = TerminalActionController(
        action_norm=nn.Identity(),
        velocity_head=head,
        optional_command_head=nn.Sequential(nn.Identity()),
        optional_event_head=None,
        motion_head=nn.Linear(hidden, 1),
        arm_dim=6,
    )
    binary_value = binary.predict_candidate_velocity(
        state,
        physical_transition_delta=delta,
    )
    expected_binary = torch.cat((delta[..., :12], torch.zeros_like(delta[..., 12:])), dim=-1)

    stacked_state = state[:, None].expand(-1, candidates, -1, -1).reshape(
        batch * candidates, horizon, hidden
    )
    stacked_delta = delta[:, None].expand(-1, candidates, -1, -1).reshape(
        batch * candidates, horizon, physical
    )
    stacked_value = continuous.predict_candidate_velocity(
        stacked_state,
        physical_transition_delta=stacked_delta,
    ).reshape(batch, candidates, horizon, physical)
    expected_stacked = delta[:, None].expand(-1, candidates, -1, -1)

    row = torch.tensor([1], dtype=torch.long)
    row_value = continuous.predict_candidate_velocity(
        state.index_select(0, row),
        physical_transition_delta=delta.index_select(0, row),
    )
    checks = {
        "continuous_terminal_adds_delta_exactly_once": torch.equal(
            continuous_value, delta
        ),
        "binary_terminal_masks_compatibility_channels_after_delta": torch.equal(
            binary_value, expected_binary
        ),
        "candidate_flatten_is_batch_major_and_delta_shared": torch.equal(
            stacked_value, expected_stacked
        ),
        "row_subset_keeps_the_matching_delta": torch.equal(
            row_value, delta.index_select(0, row)
        ),
        "terminal_rejects_nonfinite_delta": _expect_value_error(
            lambda: continuous.predict_candidate_velocity(
                state,
                physical_transition_delta=delta.clone().masked_fill(
                    torch.zeros_like(delta, dtype=torch.bool).index_fill(2, torch.tensor([0]), True),
                    float("nan"),
                ),
            )
        ),
    }
    metrics = {
        "continuous_addition_error_max": float(
            (continuous_value - delta).detach().abs().amax()
        ),
        "binary_masked_tail_max_abs": float(
            binary_value[..., 12:].detach().abs().amax()
        ),
        "candidate_mapping_error_max": float(
            (stacked_value - expected_stacked).detach().abs().amax()
        ),
    }
    return checks, metrics


def _terminal_callsite_contract() -> tuple[dict[str, bool], dict[str, int]]:
    source_path = (
        REPO_ROOT
        / "clearvla"
        / "mainline"
        / "v120_core"
        / "time_domain_mmdit.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"predict_candidate_velocity", "read_heads"}:
            calls.append(node)
    missing = [
        call.lineno
        for call in calls
        if "physical_transition_delta"
        not in {keyword.arg for keyword in call.keywords}
    ]
    text = source_path.read_text(encoding="utf-8")
    checks = {
        "every_terminal_read_has_explicit_delta_kwarg": not missing,
        "all_ten_terminal_reads_are_covered": len(calls) == 10,
        "candidate_delta_uses_expand_not_repeat": (
            text.count("candidate_transition_delta = (") == 2
            and ".expand(-1, candidate_count, -1, -1)" in text
            and "physical_transition_delta.repeat" not in text
        ),
    }
    return checks, {
        "terminal_read_call_count": len(calls),
        "terminal_read_missing_delta_count": len(missing),
    }


def run(*, fixture: Path, seed: int) -> dict[str, object]:
    modules = {"config": config_module, "interfaces": interfaces}
    base = equivalence.build_reduced_equivalence_config(modules)
    config = replace(base, top=replace(base.top, p2_spatial_intent_mode=MODE))
    config.validate()
    if not fixture.is_file():
        equivalence.create_fixture(fixture, seed=seed)
    payload = equivalence._load_fixture(fixture)
    batch = equivalence._training_batch(modules, config, payload)
    torch.manual_seed(seed)
    model = ClearVLAMainlinePolicy(config).to(
        device=torch.device("cpu"), dtype=torch.float32
    )
    noisy = payload["tensors"]["dynamic_physical_field"].clone()
    time = torch.tensor([0.5], dtype=torch.float32)

    captured_intent_kwargs: dict[str, Any] = {}

    def capture_intent_inputs(_module, _args, kwargs):
        captured_intent_kwargs.update(kwargs)

    input_handle = model.intent.organizer.register_forward_pre_hook(
        capture_intent_inputs,
        with_kwargs=True,
    )
    model.eval()
    with torch.no_grad():
        cache, training_state, _ = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=True,
            collect_diagnostics=False,
        )
    input_handle.remove()

    transition_rows: list[PhysicalTransitionInnovation] = []

    def capture_transition(_module, _inputs, output):
        transition_rows.append(output[0])

    transition_handle = model.transition.register_forward_hook(capture_transition)
    with torch.no_grad():
        zero_output = model.velocity(
            cache,
            noisy_action_field=noisy.clone(),
            time=time.clone(),
            collect_diagnostics=True,
        )
    transition_handle.remove()
    if len(transition_rows) != 1:
        raise RuntimeError("CT probe did not capture exactly one transition")
    zero_transition = transition_rows[0]
    if not isinstance(zero_transition, PhysicalTransitionInnovation):
        raise TypeError("target-action CT exported its legacy private state")

    head = model.transition.physical_delta_head
    lift = model.execution_bottom.transition_delta_lift
    if head is None or not isinstance(lift, nn.Linear):
        raise RuntimeError("target-action CT public readout is incomplete")
    head_started_zero = int(torch.count_nonzero(head.weight.detach()).item()) == 0
    basis_head = model.transition.v120_transition.basis_head[-1]
    if not isinstance(basis_head, nn.Linear):
        raise TypeError("controlled transition basis head changed type")

    model.train()
    target = torch.linspace(
        -1.0,
        1.0,
        noisy.numel(),
        dtype=noisy.dtype,
    ).reshape_as(noisy)
    model.zero_grad(set_to_none=True)
    first_output = model.velocity(
        cache,
        noisy_action_field=noisy.clone(),
        time=time.clone(),
        collect_diagnostics=False,
    )
    F.mse_loss(first_output.bottom.physical_velocity, target).backward()
    first_head_gradient = _gradient_rms(head.weight)
    first_basis_gradient = _gradient_rms(basis_head.weight)
    first_lift_gradient = _gradient_rms(lift.weight)

    with torch.no_grad():
        row = torch.linspace(-0.02, 0.02, head.weight.numel()).reshape_as(
            head.weight
        )
        head.weight.copy_(row)
    model.zero_grad(set_to_none=True)
    opened_transition_rows: list[PhysicalTransitionInnovation] = []

    def capture_opened_transition(_module, _inputs, output):
        opened_transition_rows.append(output[0])

    opened_handle = model.transition.register_forward_hook(capture_opened_transition)
    second_output = model.velocity(
        cache,
        noisy_action_field=noisy.clone(),
        time=time.clone(),
        collect_diagnostics=False,
    )
    opened_handle.remove()
    F.mse_loss(second_output.bottom.physical_velocity, target).backward()
    second_head_gradient = _gradient_rms(head.weight)
    second_basis_gradient = _gradient_rms(basis_head.weight)
    second_lift_gradient = _gradient_rms(lift.weight)
    if len(opened_transition_rows) != 1:
        raise RuntimeError("opened CT probe did not capture exactly one transition")
    opened_delta_rms = float(
        opened_transition_rows[0].delta_v.detach().float().square().mean().sqrt()
    )

    model.eval()
    runtime_calls: dict[str, int] = {}
    controller = model.execution_bottom.decoder.terminal_controller
    original_apply = controller._apply_physical_transition_delta
    active_mode = ""

    def counted_apply(velocity: Tensor, delta: Tensor | None) -> Tensor:
        runtime_calls[active_mode] = runtime_calls.get(active_mode, 0) + 1
        if delta is None or tuple(delta.shape) != tuple(velocity.shape):
            raise RuntimeError("runtime terminal read lost the physical CT delta")
        return original_apply(velocity, delta)

    controller._apply_physical_transition_delta = counted_apply
    original_dynamic = model.execution_bottom.decoder.dynamic_block_route_enabled
    try:
        with torch.no_grad():
            active_mode = "dynamic"
            model.velocity(
                cache,
                noisy_action_field=noisy.clone(),
                time=time.clone(),
                collect_diagnostics=False,
            )
            active_mode = "dynamic_fastpath"
            model.velocity(
                cache,
                noisy_action_field=noisy.clone(),
                time=time.clone(),
                deployment_fastpath=True,
                collect_diagnostics=False,
            )
            model.execution_bottom.decoder.dynamic_block_route_enabled = False
            active_mode = "fixed"
            model.velocity(
                cache,
                noisy_action_field=noisy.clone(),
                time=time.clone(),
                collect_diagnostics=False,
            )
            active_mode = "fixed_diagnostics"
            model.velocity(
                cache,
                noisy_action_field=noisy.clone(),
                time=time.clone(),
                collect_diagnostics=True,
            )
    finally:
        controller._apply_physical_transition_delta = original_apply
        model.execution_bottom.decoder.dynamic_block_route_enabled = original_dynamic

    organizer = model.intent.organizer
    original_names = organizer.camera_names
    original_canonical = organizer.canonical_camera_names
    original_permutation = organizer.camera_canonical_permutation
    facts = captured_intent_kwargs["facts"]
    cameras = int(facts.camera_coordinates.shape[2])
    camera_permutation = torch.arange(cameras - 1, -1, -1, dtype=torch.long)
    permuted_facts = replace(
        facts,
        camera_coordinates=facts.camera_coordinates.index_select(2, camera_permutation),
        camera_transport_prior=facts.camera_transport_prior.index_select(
            2, camera_permutation
        ),
        camera_support=facts.camera_support.index_select(2, camera_permutation),
        camera_validity=facts.camera_validity.index_select(2, camera_permutation),
        log_camera_validity=facts.log_camera_validity.index_select(
            2, camera_permutation
        ),
    )
    try:
        organizer.camera_names = tuple(original_names[index] for index in camera_permutation)
        organizer.canonical_camera_names = tuple(sorted(organizer.camera_names))
        organizer.camera_canonical_permutation = tuple(
            organizer.camera_names.index(name)
            for name in organizer.canonical_camera_names
        )
        permuted_kwargs = dict(captured_intent_kwargs)
        permuted_kwargs["facts"] = permuted_facts
        with torch.no_grad():
            permuted_intent, _ = organizer(**permuted_kwargs)
    finally:
        organizer.camera_names = original_names
        organizer.canonical_camera_names = original_canonical
        organizer.camera_canonical_permutation = original_permutation

    invalid_facts = replace(
        facts,
        validity=facts.validity.clone().masked_fill(
            torch.zeros_like(facts.validity, dtype=torch.bool).index_fill(
                1, torch.tensor([0]), True
            ),
            float("nan"),
        ),
    )
    invalid_kwargs = dict(captured_intent_kwargs)
    invalid_kwargs["facts"] = invalid_facts

    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    binary_outlet = OutletAdapter(codec, selection="calvin_7d_binary_v1")
    test_delta = torch.randn(2, 24, codec.physical_dim)
    binary_delta = binary_outlet.prepare_transition_delta(test_delta)
    nonfinite_delta = test_delta.clone()
    nonfinite_delta[0, 0, 0] = float("nan")
    original_tail_start = model.execution_bottom.binary_transition_tail_start
    model.execution_bottom.binary_transition_tail_start = 12
    try:
        bottom_rejects_unmasked_binary = _expect_value_error(
            lambda: model.execution_bottom._transition_views(
                PhysicalTransitionInnovation(delta_v=test_delta)
            )
        )
        bottom_accepts_masked_binary = True
        try:
            model.execution_bottom._transition_views(
                PhysicalTransitionInnovation(delta_v=binary_delta)
            )
        except ValueError:
            bottom_accepts_masked_binary = False
    finally:
        model.execution_bottom.binary_transition_tail_start = original_tail_start

    terminal_checks, terminal_metrics = _terminal_contract(seed + 1)
    callsite_checks, callsite_metrics = _terminal_callsite_contract()
    checks = {
        **terminal_checks,
        **callsite_checks,
        "ct_public_type_has_no_private_g3_or_coefficients": (
            {field.name for field in fields(PhysicalTransitionInnovation)}
            == {"delta_v", "layout", "chart"}
        ),
        "ct_public_delta_is_24_by_physical_width": tuple(zero_transition.delta_v.shape[1:])
        == (24, model.outlet_adapter.physical_dim),
        "ct_and_bottom_start_exactly_neutral": (
            head_started_zero
            and float(zero_transition.delta_v.abs().amax()) == 0.0
            and float(
                zero_output.metrics["controlled_transition_public_physical_delta_rms"]
            )
            == 0.0
        ),
        "zero_head_gets_direct_task_gradient": first_head_gradient > 0.0,
        "zero_head_blocks_upstream_and_lift_on_first_step": (
            first_basis_gradient == 0.0 and first_lift_gradient == 0.0
        ),
        "opened_head_routes_task_gradient_into_private_ct": second_basis_gradient > 0.0,
        "opened_head_routes_task_gradient_into_evidence_lift": second_lift_gradient > 0.0,
        "all_runtime_modes_apply_delta_at_terminal_reads": (
            set(runtime_calls) == {
                "dynamic",
                "dynamic_fastpath",
                "fixed",
                "fixed_diagnostics",
            }
            and all(value > 0 for value in runtime_calls.values())
        ),
        "camera_data_and_names_joint_permutation_preserves_target_fact": torch.equal(
            training_state.top.intent.target_summary,
            permuted_intent.target_summary,
        ),
        "nonfinite_object_validity_is_rejected_before_sanitization": _expect_value_error(
            lambda: organizer(**invalid_kwargs)
        ),
        "binary_outlet_masks_delta_before_bottom_evidence": float(
            binary_delta[..., 12:].abs().amax()
        )
        == 0.0,
        "binary_bottom_rejects_unmasked_public_delta": bottom_rejects_unmasked_binary,
        "binary_bottom_accepts_the_masked_public_delta": bottom_accepts_masked_binary,
        "outlet_rejects_nonfinite_delta_before_masking": _expect_value_error(
            lambda: binary_outlet.prepare_transition_delta(nonfinite_delta)
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"target-action CT contract failed: {failed}")
    return {
        "schema": "clearvla-target-action-ct-physical-contract-v1",
        "seed": int(seed),
        "mode": MODE,
        "fixture": str(fixture.resolve()),
        "checks": checks,
        "metrics": {
            **terminal_metrics,
            **callsite_metrics,
            "first_head_gradient_rms": first_head_gradient,
            "first_basis_gradient_rms": first_basis_gradient,
            "first_lift_gradient_rms": first_lift_gradient,
            "second_head_gradient_rms": second_head_gradient,
            "second_basis_gradient_rms": second_basis_gradient,
            "second_lift_gradient_rms": second_lift_gradient,
            "opened_delta_rms": float(
                opened_delta_rms
            ),
        },
        "runtime_terminal_read_calls": runtime_calls,
        "scope_note": (
            "This checks CT's narrow physical ABI, direct task-gradient staging, "
            "terminal application coverage, binary masking and canonical camera seam. "
            "It does not establish real-scene target correctness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(Path(tempfile.gettempdir()) / "clearvla_target_action_ct_fixture.pt"),
    )
    parser.add_argument("--seed", type=int, default=202_609_22)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(fixture=args.fixture, seed=args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
