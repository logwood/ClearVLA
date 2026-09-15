from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import Tensor

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import (
    CurrentObservation,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.sampling import sample_refined_cached_action
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_IMPLEMENTATION,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
)
from clearvla.mainline.v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
)


class _TerminalHarness:
    @staticmethod
    def normalize(action: Tensor) -> Tensor:
        return action

    @staticmethod
    def predict_candidate_velocity(action: Tensor) -> Tensor:
        return torch.stack(
            (
                action[..., 0] + 0.25 * action[..., 1],
                action[..., 1] - 0.50 * action[..., 2],
            ),
            dim=-1,
        )


class _CandidateHarness:
    """Minimal owner semantics for exercising the two candidate algorithms."""

    training = False
    dynamic_block_route_enabled = True
    dwell_mode = "learned"
    identity_candidate_enabled = True
    max_dwell = 2
    horizon = 4
    blocks = (object(), object(), object())
    terminal_controller = _TerminalHarness()
    config = SimpleNamespace(physical_action_dim=2)
    _select_scale_rows = staticmethod(
        EvidenceLatentMMDiTActionDecoder._select_scale_rows
    )

    def __init__(self) -> None:
        self.operation_calls = 0

    def _apply_selected_native_operations(
        self,
        action: Tensor,
        *,
        block_index: Tensor,
        repeat_count: Tensor,
        **_unused: object,
    ) -> tuple[Tensor, dict[str, Tensor], list[dict[str, Tensor]]]:
        result = action
        for repeat_index in range(self.max_dwell):
            rows = torch.nonzero(
                repeat_count > repeat_index, as_tuple=False
            ).flatten()
            if int(rows.numel()) == 0:
                continue
            self.operation_calls += 1
            selected = result.index_select(0, rows)
            owner = block_index.index_select(0, rows).to(dtype=selected.dtype)
            owner = owner[:, None, None]
            update = torch.tanh(selected * 0.2 + owner + 0.125)
            result = result.index_copy(0, rows, selected + update)
        zero = result.new_zeros(())
        return result, {"update": zero}, []


def _candidate_inputs(
    harness: _CandidateHarness,
    action: Tensor,
    *,
    decision_index: int,
) -> dict[str, object]:
    batch = int(action.shape[0])
    depth = len(harness.blocks)
    blocks = torch.arange(depth, dtype=torch.long).repeat_interleave(
        harness.max_dwell
    )
    repeats = torch.arange(harness.max_dwell, dtype=torch.long).repeat(depth)
    blocks = torch.cat((blocks, torch.tensor([depth], dtype=torch.long)))
    repeats = torch.cat((repeats, torch.zeros(1, dtype=torch.long)))
    candidate_blocks = blocks[None].expand(batch, -1)
    candidate_repeats = repeats[None].expand(batch, -1)
    candidate_mask = candidate_blocks >= decision_index
    baseline_velocity = harness.terminal_controller.predict_candidate_velocity(action)
    evidence = torch.randn(batch, 3, 5)
    condition = torch.randn(batch, 5)
    return {
        "baseline_velocity": baseline_velocity,
        "candidate_blocks": candidate_blocks,
        "candidate_repeats": candidate_repeats,
        "candidate_mask": candidate_mask,
        "evidence_tokens": evidence,
        "evidence_value_tokens": evidence + 0.1,
        "global_condition": condition,
        "evidence_key_bias": torch.zeros(batch, 3),
        "evidence_scale": 1.0,
        "capacity_ratios": torch.ones(batch, depth),
        "identity_boundary": False,
        "prepared_factors": None,
    }


def _neutral_action(
    harness: _CandidateHarness,
    action: Tensor,
    *,
    decision_index: int,
) -> Tensor:
    batch = int(action.shape[0])
    neutral, _, _ = harness._apply_selected_native_operations(
        action,
        block_index=torch.full((batch,), decision_index, dtype=torch.long),
        repeat_count=torch.ones(batch, dtype=torch.long),
    )
    return neutral


def _authoritative_candidates(
    harness: _CandidateHarness,
    action: Tensor,
    inputs: dict[str, object],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return EvidenceLatentMMDiTActionDecoder._run_differentiable_native_candidates(
        harness,  # type: ignore[arg-type]
        action,
        **inputs,
    )


def _fast_candidates(
    harness: _CandidateHarness,
    action: Tensor,
    inputs: dict[str, object],
    *,
    neutral_action: Tensor,
    decision_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    chart = EvidenceLatentMMDiTActionDecoder._validate_deployment_candidate_prefix_chart(
        harness,  # type: ignore[arg-type]
        action,
        candidate_blocks=inputs["candidate_blocks"],  # type: ignore[arg-type]
        candidate_repeats=inputs["candidate_repeats"],  # type: ignore[arg-type]
    )
    return EvidenceLatentMMDiTActionDecoder._run_deployment_candidate_prefix_reuse(
        harness,  # type: ignore[arg-type]
        action,
        neutral_action=neutral_action,
        decision_index=decision_index,
        candidate_chart=chart,
        baseline_velocity=inputs["baseline_velocity"],  # type: ignore[arg-type]
        evidence_tokens=inputs["evidence_tokens"],  # type: ignore[arg-type]
        evidence_value_tokens=inputs["evidence_value_tokens"],  # type: ignore[arg-type]
        global_condition=inputs["global_condition"],  # type: ignore[arg-type]
        evidence_key_bias=inputs["evidence_key_bias"],  # type: ignore[arg-type]
        evidence_scale=inputs["evidence_scale"],  # type: ignore[arg-type]
        capacity_ratios=inputs["capacity_ratios"],  # type: ignore[arg-type]
        identity_boundary=inputs["identity_boundary"],  # type: ignore[arg-type]
        prepared_factors=inputs["prepared_factors"],  # type: ignore[arg-type]
    )


def test_prefix_reuse_matches_authoritative_actions_and_invalidates_neutral() -> None:
    torch.manual_seed(5100)
    harness = _CandidateHarness()
    first_action = torch.randn(2, harness.horizon, 5)
    second_action = first_action + torch.linspace(0.1, 0.7, 5)
    decision_index = 1

    fast_outputs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    with torch.no_grad():
        for action in (first_action, second_action):
            inputs = _candidate_inputs(
                harness,
                action,
                decision_index=decision_index,
            )
            neutral = _neutral_action(
                harness,
                action,
                decision_index=decision_index,
            )
            authoritative = _authoritative_candidates(harness, action, inputs)
            fast = _fast_candidates(
                harness,
                action,
                inputs,
                neutral_action=neutral,
                decision_index=decision_index,
            )
            for expected, actual in zip(authoritative, fast, strict=True):
                torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
            fast_outputs.append(fast)

    assert not torch.equal(fast_outputs[0][0], fast_outputs[1][0])
    assert not hasattr(harness, "_candidate_prefix_neutral_action")


def test_prefix_reuse_reduces_exact_block_invocations_from_21_to_12() -> None:
    torch.manual_seed(5102)
    harness = _CandidateHarness()
    action = torch.randn(1, harness.horizon, 5)
    authoritative_calls = 0
    fast_calls = 0

    with torch.no_grad():
        for decision_index in range(len(harness.blocks)):
            inputs = _candidate_inputs(
                harness,
                action,
                decision_index=decision_index,
            )
            harness.operation_calls = 0
            _neutral_action(harness, action, decision_index=decision_index)
            _authoritative_candidates(harness, action, inputs)
            authoritative_calls += harness.operation_calls

            harness.operation_calls = 0
            neutral = _neutral_action(
                harness,
                action,
                decision_index=decision_index,
            )
            _fast_candidates(
                harness,
                action,
                inputs,
                neutral_action=neutral,
                decision_index=decision_index,
            )
            fast_calls += harness.operation_calls

    assert authoritative_calls == 21
    assert fast_calls == 12


def test_prefix_reuse_shape_and_deployment_guards_fail_closed() -> None:
    harness = _CandidateHarness()
    action = torch.randn(1, harness.horizon, 5)
    inputs = _candidate_inputs(harness, action, decision_index=0)
    neutral = _neutral_action(harness, action, decision_index=0)

    with pytest.raises(ValueError, match="eval mode"):
        _fast_candidates(
            harness,
            action,
            inputs,
            neutral_action=neutral,
            decision_index=0,
        )

    with torch.no_grad(), pytest.raises(ValueError, match="neutral candidate"):
        _fast_candidates(
            harness,
            action,
            inputs,
            neutral_action=neutral[:, :-1],
            decision_index=0,
        )

    malformed = dict(inputs)
    malformed["candidate_blocks"] = inputs["candidate_blocks"][:, :-1]  # type: ignore[index]
    malformed["candidate_repeats"] = inputs["candidate_repeats"][:, :-1]  # type: ignore[index]
    malformed["candidate_mask"] = inputs["candidate_mask"][:, :-1]  # type: ignore[index]
    with torch.no_grad(), pytest.raises(ValueError, match="canonical global"):
        _fast_candidates(
            harness,
            action,
            malformed,
            neutral_action=neutral,
            decision_index=0,
        )

    wrong_order = dict(inputs)
    wrong_blocks = inputs["candidate_blocks"].clone()  # type: ignore[union-attr]
    wrong_blocks[:, 1] = 1
    wrong_blocks[:, 2] = 0
    wrong_order["candidate_blocks"] = wrong_blocks
    with torch.no_grad(), pytest.raises(ValueError, match="block-major/dwell-major"):
        _fast_candidates(
            harness,
            action,
            wrong_order,
            neutral_action=neutral,
            decision_index=0,
        )

    injected_mask = dict(inputs)
    injected_mask["candidate_mask"] = torch.zeros_like(  # type: ignore[arg-type]
        inputs["candidate_mask"]
    )
    with torch.no_grad():
        derived_mask = _fast_candidates(
            harness,
            action,
            injected_mask,
            neutral_action=neutral,
            decision_index=0,
        )[2]
    torch.testing.assert_close(
        derived_mask,
        inputs["candidate_blocks"] >= 0,  # type: ignore[operator]
        atol=0.0,
        rtol=0.0,
    )

    wrong_prediction = dict(inputs)
    wrong_prediction["baseline_velocity"] = inputs["baseline_velocity"][:, :-1]  # type: ignore[index]
    with torch.no_grad(), pytest.raises(ValueError, match="velocity-head"):
        _fast_candidates(
            harness,
            action,
            wrong_prediction,
            neutral_action=neutral,
            decision_index=0,
        )

    harness.dwell_mode = "fixed"
    try:
        with torch.no_grad(), pytest.raises(ValueError, match="learned global"):
            _fast_candidates(
                harness,
                action,
                inputs,
                neutral_action=neutral,
                decision_index=0,
            )
    finally:
        harness.dwell_mode = "learned"

    harness.training = True
    try:
        with torch.no_grad(), pytest.raises(ValueError, match="eval mode"):
            _fast_candidates(
                harness,
                action,
                inputs,
                neutral_action=neutral,
                decision_index=0,
            )
    finally:
        harness.training = False


def _tiny_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            action_basis_tokens=2,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
        ),
        observation=replace(
            base.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(
            base.top,
            grounder_iterations=2,
            teacher_key_dim=8,
            goal_condition_dropout=0.0,
            action_history_condition_dropout=0.0,
        ),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=1,
            controller_heads=4,
        ),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    config.validate()
    return config


def _tiny_arm_only_config() -> ExperimentConfig:
    base = _tiny_config()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE_ARM_ONLY_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
            bspine_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        ),
    )
    config.validate()
    return config


def _online_input(config: ExperimentConfig) -> OnlinePolicyInput:
    dims = config.dimensions
    batch = 1
    return OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=torch.randn(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
            ),
            raw_rgb=torch.rand(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                3,
                48,
                48,
            ),
        ),
        history=ObservableHistory(
            state=torch.randn(batch, dims.state_dim),
            action_state=torch.randn(batch, dims.action_dim),
            codec_gripper_boundary=torch.randn(batch, 1),
            state_history=torch.randn(
                batch, dims.state_history_length, dims.state_dim
            ),
            executed_action_history=torch.randn(
                batch,
                dims.executed_history_length,
                dims.action_dim,
            ),
        ),
        goal=GoalCondition(
            tokens=torch.randn(batch, 6, dims.goal_token_dim),
            mask=torch.ones(batch, 6, dtype=torch.bool),
        ),
    )


@pytest.mark.parametrize("arm_only", (False, True), ids=("raw", "arm-only-spine"))
def test_full_two_pass_action_parity_and_default_opt_in_boundary(
    arm_only: bool,
) -> None:
    torch.manual_seed(5101)
    config = _tiny_arm_only_config() if arm_only else _tiny_config()
    model = ClearVLAMainlinePolicy(config).eval()
    if arm_only:
        spine = model.execution_bottom.decoder.spine
        assert spine is not None
        with torch.no_grad():
            for parameter in spine.parameters():
                parameter.fill_(0.01)
    model.set_training_step(1200)
    with torch.no_grad():
        cache, _, _ = model.encode_online(_online_input(config))
    initial_noise = torch.randn(
        1,
        config.dimensions.action_horizon,
        model.outlet_adapter.physical_dim,
    )
    decoder = model.execution_bottom.decoder

    with mock.patch.object(
        decoder,
        "_run_deployment_candidate_prefix_reuse",
        wraps=decoder._run_deployment_candidate_prefix_reuse,
    ) as fastpath:
        authoritative = sample_refined_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=initial_noise,
            collect_diagnostics=True,
            dtype=torch.float32,
        )
        assert fastpath.call_count == 0
        optimized = sample_refined_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=initial_noise,
            collect_diagnostics=True,
            dtype=torch.float32,
            deployment_fastpath=True,
        )

    expected_dynamic_decisions = (
        2
        * (config.runtime.inference_steps + 1)
        * len(decoder.blocks)
    )
    assert fastpath.call_count == expected_dynamic_decisions
    for expected, actual in (
        (authoritative.action, optimized.action),
        (authoritative.physical_field, optimized.physical_field),
        (authoritative.motion_logits, optimized.motion_logits),
        (authoritative.step_times, optimized.step_times),
    ):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert set(optimized.metrics) == set(authoritative.metrics)
    for name in authoritative.metrics:
        torch.testing.assert_close(
            optimized.metrics[name],
            authoritative.metrics[name],
            atol=0.0,
            rtol=0.0,
            msg=lambda message, metric=name: f"{metric}: {message}",
        )
    assert not hasattr(decoder, "_candidate_prefix_neutral_action")
