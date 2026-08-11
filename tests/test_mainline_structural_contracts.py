from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from unittest import mock

import torch
import torch.nn.functional as F

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import CurrentObservation
from clearvla.mainline.model.bottom import (
    ReadOnlyEvidenceMMDiTBlock,
    TypedEvidenceBank,
)
from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler
from clearvla.mainline.model.grounding import DenseObjectGrounder, dense_chart_from_local_facts
from clearvla.mainline.model.intent import FuturePlanRecognizer
from clearvla.mainline.model.observation import (
    CurrentObservationCompiler,
    PatchFlowField,
    RecurrentLocalFlow,
    _flow_parameter_to_displacement,
    _sample_feature_chart,
)
from clearvla.mainline.model.policy import OnlinePolicyCache
from clearvla.mainline.model.teacher import ObjectFutureTeacher
from clearvla.mainline.model.top import DeploymentTopCache, ObjectIntentDynamicsTop
from clearvla.mainline.model.types import (
    FutureObjectDynamics,
    LocalFactSet,
    ObjectFactualDock,
)
from clearvla.mainline.training.losses import flow_geometry_terms, future_dynamics_terms
from clearvla.mainline.v120_core.refinement import NestedLowRankContractionBank


def _assert_same_typed_value(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
        return
    if is_dataclass(left):
        for field in fields(left):
            _assert_same_typed_value(getattr(left, field.name), getattr(right, field.name))
        return
    assert left == right


def test_feature_chart_sampling_owns_the_fp16_bf16_boundary() -> None:
    feature = torch.randn(1, 2, 8, 4, 4, dtype=torch.float16)
    coordinates = torch.zeros(1, 2, 2, 2, 4, 2, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sampled, valid = _sample_feature_chart(feature, coordinates)
    assert sampled.dtype == torch.float32
    assert tuple(sampled.shape) == (1, 2, 2, 2, 4, 8)
    assert bool(valid.all())


def _local_facts(
    *,
    batch: int = 1,
    cameras: int = 1,
    side: int = 2,
    content: int = 16,
    route: int = 8,
    hidden: int = 32,
    valid: bool = True,
    observed: bool = True,
) -> LocalFactSet:
    hypotheses = 4
    prefix = (batch, cameras, side, side, hypotheses)
    owner = torch.full(prefix, 1.0 / hypotheses)
    return LocalFactSet(
        public_scene_base=torch.randn(batch, cameras, side, side, hidden),
        target_dino_content=torch.randn(batch, cameras, side, side, content),
        cell_observed=torch.full((batch, cameras, side, side, 1), observed, dtype=torch.bool),
        content_slots=torch.randn(*prefix, content),
        semantic_slots=torch.randn(*prefix, route),
        appearance_slots=torch.randn(*prefix, route),
        geometry_slots=torch.randn(*prefix, route),
        semantic_owner_probs=owner,
        appearance_owner_probs=owner,
        geometry_owner_probs=owner,
        slot_coordinates=torch.tanh(torch.randn(*prefix, 2)),
        slot_support=torch.full(prefix, 0.1),
        slot_validity=torch.full((*prefix, 1), float(valid)),
        slot_transport_prior=torch.zeros(*prefix, 2),
    )


def test_masked_reconstruction_keeps_target_without_reopening_candidates() -> None:
    torch.manual_seed(1)
    local = _local_facts(
        content=8,
        route=4,
        hidden=16,
        valid=False,
        observed=False,
    )
    public = local.public_scene_base.detach().clone().requires_grad_(True)
    candidates = local.content_slots.detach().clone().requires_grad_(True)
    local = type(local)(
        **{
            **local.__dict__,
            "public_scene_base": public,
            "content_slots": candidates,
        }
    )
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    facts, metrics = grounder(local)
    assert torch.count_nonzero(facts.dense_chart.dino_content) > 0
    assert metrics["object_grounding_masked_reconstruction_mse"] > 0
    facts.reconstruction_error.backward()
    assert public.grad is not None and public.grad.abs().sum() > 0
    assert candidates.grad is None or torch.count_nonzero(candidates.grad) == 0
    reconstruction_grad = sum(
        parameter.grad.abs().sum()
        for parameter in grounder.reconstruction_query.parameters()
        if parameter.grad is not None
    )
    assert reconstruction_grad > 0


def test_future_recognizer_keeps_four_interval_whole_segment_targets() -> None:
    torch.manual_seed(2)
    batch, intervals, objects, content, cameras = 1, 4, 3, 8, 1
    scalar = torch.ones(batch, intervals, objects, 1)
    teacher = FutureObjectDynamics(
        current_reference=torch.randn(batch, objects, content),
        successor_content=torch.randn(batch, intervals, objects, content),
        semantic_delta=torch.randn(batch, intervals, objects, content),
        transport_mean=torch.randn(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.rand(batch, intervals, objects, cameras, 3),
        visibility=torch.zeros_like(scalar),
        persistence=torch.zeros_like(scalar),
        uncertainty=torch.zeros_like(scalar),
        reliability=scalar,
        validity=scalar[:, :, :, None],
        future_address=torch.full((batch, intervals, objects, cameras, 2, 2), 0.25),
        object_coordinates=torch.zeros(batch, objects, cameras, 2),
    )
    recognizer = FuturePlanRecognizer(
        hidden=16,
        action_dim=2,
        state_dim=2,
        content_dim=content,
        heads=4,
    )
    result = recognizer(
        future_action=torch.randn(batch, 48, 2),
        future_state=torch.randn(batch, 48, 2),
        teacher=teacher,
    )
    assert tuple(result.interval_targets.shape) == (batch, intervals, 16)
    assert tuple(result.action_summary.shape) == (batch, intervals, 2)
    assert tuple(result.state_summary.shape) == (batch, intervals, 2)
    assert tuple(result.effect_summary.shape) == (batch, intervals, content)


def test_future_recognizer_supervises_neutral_objects_without_reliability_discount() -> None:
    batch, intervals, objects, content, cameras = 1, 4, 2, 8, 1
    current = torch.randn(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    teacher = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.ones(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=torch.ones_like(scalar),
        reliability=scalar,
        validity=torch.ones(batch, intervals, objects, cameras, 1),
        future_address=torch.zeros(batch, intervals, objects, cameras, 2, 2),
        object_coordinates=torch.zeros(batch, objects, cameras, 2),
    )
    recognizer = FuturePlanRecognizer(
        hidden=16,
        action_dim=2,
        state_dim=2,
        content_dim=content,
        heads=4,
    )
    result = recognizer(
        future_action=torch.randn(batch, 48, 2),
        future_state=torch.randn(batch, 48, 2),
        teacher=teacher,
    )
    torch.testing.assert_close(result.effect_summary, torch.ones_like(result.effect_summary))


def test_final_object_posterior_is_recomputed_after_last_slot_update() -> None:
    torch.manual_seed(21)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=3,
    )
    with mock.patch.object(
        grounder,
        "_competition",
        wraps=grounder._competition,
    ) as competition:
        facts, _ = grounder(
            _local_facts(content=8, route=4, hidden=16),
        )
    facts.validate()
    assert competition.call_count == grounder.iterations + 1


def test_grounding_diagnostics_switch_is_math_equivalent() -> None:
    torch.manual_seed(211)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=2,
    )
    local = _local_facts(content=8, route=4, hidden=16)
    with_metrics, metrics = grounder(local, collect_diagnostics=True)
    without_metrics, silent = grounder(local, collect_diagnostics=False)
    assert metrics
    assert silent == {}
    _assert_same_typed_value(with_metrics, without_metrics)


def test_teacher_diagnostics_switch_is_math_equivalent() -> None:
    torch.manual_seed(212)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    facts, _ = grounder(
        _local_facts(content=8, route=4, hidden=16),
        collect_diagnostics=False,
    )
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    supports = torch.randn(1, 12, 1, 2, 2, 8)
    offsets = torch.arange(4, 49, 4)[None]
    with_metrics, metrics = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=offsets,
        collect_diagnostics=True,
    )
    without_metrics, silent = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=offsets,
        collect_diagnostics=False,
    )
    assert metrics
    assert silent == {}
    _assert_same_typed_value(with_metrics, without_metrics)


def test_teacher_scales_four_frame_flow_in_physical_horizon_units() -> None:
    teacher = ObjectFutureTeacher(
        content_dim=8,
        key_dim=4,
        flow_reference_frames=4,
    )
    offsets = torch.arange(4, 49, 4)[None]
    torch.testing.assert_close(
        teacher._flow_horizon_scale(offsets),
        torch.arange(1, 13, dtype=torch.float32)[None],
    )


class _FixedFlow(torch.nn.Module):
    def __init__(self, displacement: float) -> None:
        super().__init__()
        self.displacement = float(displacement)

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        *,
        compute_backward: bool = True,
    ) -> PatchFlowField:
        batch, cameras, _, rows, columns = previous.shape
        flow = previous.new_zeros(batch, cameras, 2, rows, columns)
        flow[:, :, 0] = self.displacement
        scalar = previous.new_full((batch, cameras, 1, rows, columns), 0.5)
        result = PatchFlowField(
            forward=flow,
            backward=-flow if compute_backward else None,
            confidence=scalar,
            uncertainty=scalar,
            occlusion=torch.zeros_like(scalar),
            refinement_sequence=(flow,),
        )
        result.validate()
        return result


def test_recurrent_flow_exports_current_aligned_physical_displacements() -> None:
    estimator = RecurrentLocalFlow(
        feature_dim=4,
        iterations=2,
        radius=1,
        uncertainty_floor=0.01,
    )
    previous = torch.zeros(1, 1, 4, 3, 5)
    current = torch.ones_like(previous)
    inverse_parameter = torch.zeros(1, 2, 3, 5)
    inverse_parameter[:, 0] = 0.30
    inverse_parameter[:, 1] = -0.20
    source_forward_parameter = torch.zeros_like(inverse_parameter)
    source_forward_parameter[:, 0] = -0.15
    source_forward_parameter[:, 1] = 0.25
    scalar = torch.full((1, 1, 3, 5), 0.5)
    inverse_sequence = (0.5 * inverse_parameter, inverse_parameter)
    with mock.patch.object(
        estimator,
        "_estimate",
        side_effect=(
            (
                inverse_parameter,
                scalar,
                scalar,
                torch.zeros_like(scalar),
                inverse_sequence,
            ),
            (
                source_forward_parameter,
                scalar,
                scalar,
                torch.zeros_like(scalar),
                (source_forward_parameter,),
            ),
        ),
    ) as estimate:
        field = estimator(previous, current, compute_backward=True)

    first_source, first_target = estimate.call_args_list[0].args
    second_source, second_target = estimate.call_args_list[1].args
    assert torch.equal(first_source, current.reshape_as(first_source))
    assert torch.equal(first_target, previous.reshape_as(first_target))
    assert torch.equal(second_source, previous.reshape_as(second_source))
    assert torch.equal(second_target, current.reshape_as(second_target))
    torch.testing.assert_close(
        field.forward[:, 0],
        -_flow_parameter_to_displacement(inverse_parameter),
    )
    assert field.backward is not None
    torch.testing.assert_close(
        field.backward[:, 0],
        -_flow_parameter_to_displacement(source_forward_parameter),
    )
    for actual, parameter in zip(field.refinement_sequence, inverse_sequence, strict=True):
        torch.testing.assert_close(
            actual[:, 0],
            -_flow_parameter_to_displacement(parameter),
        )
    # The recurrent parameter is not the exported displacement: border-aware
    # conversion is part of the estimator ABI and keeps chart edges valid.
    assert not torch.equal(field.forward[:, 0], -inverse_parameter)


def test_vectorized_flow_neighbourhood_matches_scalar_sampling_order() -> None:
    torch.manual_seed(34)
    estimator = RecurrentLocalFlow(
        feature_dim=4,
        iterations=1,
        radius=2,
        uncertainty_floor=0.01,
    )
    target = torch.randn(2, 4, 5, 7)
    parameter = 0.2 * torch.randn(2, 2, 5, 7)
    expected = torch.stack(
        tuple(
            estimator._sample(target, parameter, dx=dx, dy=dy)
            for dy in range(-estimator.radius, estimator.radius + 1)
            for dx in range(-estimator.radius, estimator.radius + 1)
        ),
        dim=1,
    )
    actual = estimator._sample_neighbourhood(target, parameter)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_flow_refinement_uses_two_grid_samples_per_iteration() -> None:
    torch.manual_seed(35)
    estimator = RecurrentLocalFlow(
        feature_dim=4,
        iterations=3,
        radius=2,
        uncertainty_floor=0.01,
    )
    source = torch.randn(2, 4, 5, 7)
    target = torch.randn_like(source)
    with mock.patch(
        "clearvla.mainline.model.observation.F.grid_sample",
        wraps=F.grid_sample,
    ) as grid_sample:
        estimator._estimate(source, target)
    assert grid_sample.call_count == 2 * estimator.iterations


def test_current_fact_coordinates_do_not_double_apply_previous_to_current_flow() -> None:
    torch.manual_seed(22)
    config = ExperimentConfig()
    compiler = CurrentObservationCompiler(config).eval()
    observation = CurrentObservation(
        dino_history=torch.randn(
            1,
            config.dimensions.visual_history_length,
            config.dimensions.num_cameras,
            config.dimensions.patches_per_camera,
            config.dimensions.visual_token_dim,
        ),
        raw_rgb=torch.rand(
            1,
            config.dimensions.visual_history_length,
            config.dimensions.num_cameras,
            3,
            32,
            32,
        ),
    )
    compiler.flow = _FixedFlow(0.0)
    zero, _ = compiler(observation)
    compiler.flow = _FixedFlow(0.35)
    moved, _ = compiler(observation)
    assert torch.equal(
        zero.local_facts.slot_coordinates,
        moved.local_facts.slot_coordinates,
    )
    assert not torch.equal(
        zero.local_facts.slot_transport_prior,
        moved.local_facts.slot_transport_prior,
    )


def test_causal_dino_history_changes_owner_evidence_without_changing_current_target() -> None:
    torch.manual_seed(222)
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
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
            flow_iterations=1,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        bottom=replace(base.bottom, controller_heads=4),
    )
    compiler = CurrentObservationCompiler(config).eval()
    compiler.flow = _FixedFlow(0.0)
    dino = torch.randn(
        1,
        config.dimensions.visual_history_length,
        config.dimensions.num_cameras,
        config.dimensions.patches_per_camera,
        config.dimensions.visual_token_dim,
    )
    raw = torch.rand(
        1,
        config.dimensions.visual_history_length,
        config.dimensions.num_cameras,
        3,
        32,
        32,
    )
    baseline, _ = compiler(CurrentObservation(dino_history=dino, raw_rgb=raw))
    changed_history = dino.clone()
    changed_history[:, 0, :, :8] += 2.0
    changed, _ = compiler(
        CurrentObservation(dino_history=changed_history, raw_rgb=raw)
    )
    # The supervised current DINO target is still the same final frame, while
    # causal history changes G's owner evidence through an aligned innovation.
    torch.testing.assert_close(
        baseline.local_facts.target_dino_content,
        changed.local_facts.target_dino_content,
    )
    assert not torch.equal(
        baseline.local_facts.semantic_owner_probs,
        changed.local_facts.semantic_owner_probs,
    )


def test_flow_geometry_keeps_literal_rgb_anchor_when_learned_features_collapse() -> None:
    torch.manual_seed(33)
    config = ExperimentConfig()
    compiler = CurrentObservationCompiler(config).eval()
    observation = CurrentObservation(
        dino_history=torch.randn(
            1,
            config.dimensions.visual_history_length,
            config.dimensions.num_cameras,
            config.dimensions.patches_per_camera,
            config.dimensions.visual_token_dim,
        ),
        raw_rgb=torch.zeros(
            1,
            config.dimensions.visual_history_length,
            config.dimensions.num_cameras,
            3,
            32,
            32,
        ),
    )
    evidence, _ = compiler(observation)
    rows, columns = evidence.flow.forward.shape[-2:]
    previous_rgb = torch.zeros(1, config.dimensions.num_cameras, 3, rows, columns)
    current_rgb = torch.zeros_like(previous_rgb)
    previous_rgb[..., 1] = 1.0
    current_rgb[..., 2] = 1.0
    displacement = 2.0 / float(columns - 1)
    forward = torch.zeros_like(evidence.flow.forward)
    forward[:, :, 0] = displacement
    backward = -forward
    scalar = torch.full_like(evidence.flow.confidence, 0.5)
    anchored = replace(
        evidence,
        detail_features=torch.zeros_like(evidence.detail_features),
        previous_detail_features=torch.zeros_like(evidence.previous_detail_features),
        earlier_detail_features=torch.zeros_like(evidence.earlier_detail_features),
        literal_rgb=current_rgb,
        previous_literal_rgb=previous_rgb,
        earlier_literal_rgb=previous_rgb,
        flow=PatchFlowField(
            forward=forward,
            backward=backward,
            confidence=scalar,
            uncertainty=scalar,
            occlusion=torch.zeros_like(scalar),
            refinement_sequence=(forward,),
        ),
        earlier_flow=PatchFlowField(
            forward=torch.zeros_like(forward),
            backward=torch.zeros_like(backward),
            confidence=scalar,
            uncertainty=scalar,
            occlusion=torch.zeros_like(scalar),
            refinement_sequence=(torch.zeros_like(forward),),
        ),
    )
    terms = flow_geometry_terms(anchored)
    assert terms["flow_feature_warp"] == 0.0
    assert terms["flow_photometric_warp"] < terms["flow_photometric_zero_warp"]
    assert terms["flow_warp"] > 0.0


def test_chunked_future_support_pool_matches_full_fp32_pool() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            visual_token_dim=16,
            goal_token_dim=16,
        ),
    )
    compiler = CurrentObservationCompiler(config)
    tokens = torch.randn(
        2,
        config.dimensions.future_supports,
        config.dimensions.num_cameras,
        config.dimensions.patches_per_camera,
        config.dimensions.visual_token_dim,
    ).half()
    actual = compiler.teacher_supports(tokens)
    side = int(config.dimensions.patches_per_camera**0.5)
    chart = tokens.float().reshape(-1, side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
    expected = F.adaptive_avg_pool2d(
        chart,
        (config.observation.grid_size, config.observation.grid_size),
    )
    expected = expected.permute(0, 2, 3, 1).reshape_as(actual)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_w_transport_moves_probability_instead_of_reweighting_source() -> None:
    side = 7
    current = torch.zeros(1, 1, 1, side, side)
    current[..., side // 2, side // 2] = 1.0
    transport = torch.tensor([[[[[0.5, 0.0]]]]])
    moved = ObjectFutureDynamicsCompiler._transport_address(current, transport)
    axis = torch.linspace(-1.0, 1.0, side)
    source_x = (current * axis[None, None, None, None]).sum()
    moved_x = (moved * axis[None, None, None, None, None]).sum()
    assert moved_x > source_x + 0.25
    assert torch.allclose(moved.sum(), current.sum(), atol=1e-5)


def test_g_prototype_loss_penalizes_overlapping_reads_without_forcing_diversity() -> None:
    decoded = torch.tensor([[[-1.0], [1.0]]])
    target = torch.tensor([[[[[-1.0], [1.0]]]]])
    owned = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    overlapping = torch.full_like(owned, 0.5)
    owned_loss = DenseObjectGrounder._conditional_prototype_error(decoded, target, owned)
    overlap_loss = DenseObjectGrounder._conditional_prototype_error(
        decoded,
        target,
        overlapping,
    )
    assert torch.count_nonzero(owned_loss) == 0
    assert overlap_loss > 1.0

    # This is conditional reconstruction, not an artificial slot-diversity
    # quota: identical facts and identical prototypes remain a legal optimum.
    identical = torch.zeros_like(decoded)
    identical_target = torch.zeros_like(target)
    identical_loss = DenseObjectGrounder._conditional_prototype_error(
        identical,
        identical_target,
        overlapping,
    )
    assert torch.count_nonzero(identical_loss) == 0


def test_g_host_context_changes_only_binder_keys_not_candidate_values() -> None:
    torch.manual_seed(221)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    first = dense_chart_from_local_facts(
        _local_facts(cameras=2, content=8, route=4, hidden=16)
    )
    second = replace(
        first,
        public_scene_base=first.public_scene_base + 1000.0 * torch.randn_like(
            first.public_scene_base
        ),
    )
    first_key, first_value = grounder._candidate_tokens(first)
    second_key, second_value = grounder._candidate_tokens(second)
    # Hosted G context must reach the online ownership competition.
    assert not torch.allclose(first_key, second_key)
    # It remains key-only: exported/aggregated candidate values are not a
    # public carrier copied into every K slot.
    torch.testing.assert_close(first_value, second_value)


def test_w_zero_initialized_camera_mass_residual_preserves_current_camera_prior() -> None:
    torch.manual_seed(25)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=8,
        route_dim=4,
        heads=4,
    )
    field = dynamics._field(
        facts=facts,
        hidden=torch.zeros(1, 2, 4, 16),
    )
    expected = facts.object_to_chart[:, None].expand(-1, 2, -1, -1, -1, -1)
    assert torch.allclose(field.future_address, expected, atol=1e-5, rtol=1e-5)


def test_w_successor_innovation_has_no_detach_minus_current_ghost_gradient() -> None:
    torch.manual_seed(29)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    facts.content.retain_grad()
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=16, content_dim=8, route_dim=4, heads=4
    )
    field = dynamics._field(
        facts=facts,
        hidden=torch.zeros(1, 2, 4, 16, requires_grad=True),
    )
    innovation = field.successor_content - field.current_reference[:, None]
    assert torch.count_nonzero(innovation) == 0
    innovation.sum().backward()
    assert facts.content.grad is None or torch.count_nonzero(facts.content.grad) == 0


def test_w_receives_completed_intent_and_coarse_action_as_distinct_inputs() -> None:

    torch.manual_seed(31)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    )
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    blank_intent = replace(
        intent,
        interval_queries=torch.zeros_like(intent.interval_queries),
    )
    signal_intent = replace(
        blank_intent,
        interval_queries=torch.randn_like(intent.interval_queries),
    )
    coarse = top.coarse_action(blank_intent)
    zero_action = replace(coarse, tokens=torch.zeros_like(coarse.tokens))
    signal_action = replace(coarse, tokens=torch.randn_like(coarse.tokens))

    signal_zero, _ = top.dynamics._base(
        facts,
        signal_intent,
        zero_action,
        collect_diagnostics=False,
    )
    blank_zero, _ = top.dynamics._base(
        facts,
        blank_intent,
        zero_action,
        collect_diagnostics=False,
    )
    blank_action, _ = top.dynamics._base(
        facts,
        blank_intent,
        signal_action,
        collect_diagnostics=False,
    )
    assert not torch.equal(signal_zero, blank_zero)
    assert not torch.equal(blank_action, blank_zero)


def test_stateless_intent_is_repeatable_without_frame_progress_input() -> None:

    torch.manual_seed(30)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    )
    facts, _ = top.grounder(_local_facts(cameras=2))
    kwargs = dict(
        goal_tokens=torch.zeros(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.zeros(1, 3, 7),
        state=torch.zeros(1, 7),
        executed_history=torch.zeros(1, 3, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    first, _ = top.intent(**kwargs)
    second, _ = top.intent(**kwargs)
    torch.testing.assert_close(first.interval_queries, second.interval_queries)
    torch.testing.assert_close(first.temporal_queries, second.temporal_queries)


def test_goal_changes_interval_intent_without_rewriting_object_facts() -> None:

    torch.manual_seed(32)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    )
    facts, _ = top.grounder(_local_facts(cameras=2))
    common = dict(
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.zeros(1, 3, 7),
        state=torch.zeros(1, 7),
        executed_history=torch.zeros(1, 3, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    zero_goal, _ = top.intent(
        goal_tokens=torch.zeros(1, 6, 12),
        **common,
    )
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        **common,
    )
    torch.testing.assert_close(intent.object_tokens, zero_goal.object_tokens)
    assert not torch.equal(intent.interval_queries, zero_goal.interval_queries)


def test_future_neutral_fallback_remains_supervised_when_reliability_is_zero() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    current = torch.randn(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    validity = torch.ones(batch, intervals, objects, cameras, 1)
    address = torch.zeros(batch, intervals, objects, cameras, 2, 2)
    address[..., 0, 0] = 1.0
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        validity=validity,
        future_address=address,
        object_coordinates=torch.zeros(batch, objects, cameras, 2),
    )
    changed_address = torch.zeros_like(address)
    changed_address[..., 1, 1] = 1.0
    prediction = replace(
        target,
        future_address=changed_address,
        transport_mean=torch.ones_like(target.transport_mean),
    )
    unreliable = future_dynamics_terms(prediction, target)
    assert unreliable["future_transport"] > 0
    assert "future_address" not in unreliable

    # The teacher's null association already turns content into the current
    # fact/zero-delta fallback.  That fallback remains an actual supervised W
    # target instead of being discounted a second time by reliability.
    changed_successor = replace(
        target,
        successor_content=target.successor_content + 1.0,
        semantic_delta=target.semantic_delta + 1.0,
    )
    unreliable_content = future_dynamics_terms(changed_successor, target)
    assert unreliable_content["future_successor"] > 0
    assert unreliable_content["future_semantic_delta"] > 0

    reliable_target = replace(target, reliability=torch.ones_like(scalar))
    reliable_prediction = replace(prediction, reliability=torch.ones_like(scalar))
    reliable = future_dynamics_terms(reliable_prediction, reliable_target)
    torch.testing.assert_close(unreliable["future_transport"], reliable["future_transport"])


def test_future_interval_transition_penalizes_temporal_collapse_not_common_offset() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    current = torch.zeros(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    interval = torch.arange(intervals, dtype=torch.float32)[None, :, None, None]
    semantic = interval.expand(batch, intervals, objects, content).clone()
    address = torch.zeros(batch, intervals, objects, cameras, 2, 2)
    address[..., 0, 0] = 1.0
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=semantic,
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        validity=torch.ones(batch, intervals, objects, cameras, 1),
        future_address=address,
        object_coordinates=torch.zeros(batch, objects, cameras, 2),
    )
    shifted = replace(target, semantic_delta=semantic + 7.0)
    collapsed = replace(target, semantic_delta=torch.zeros_like(semantic))

    torch.testing.assert_close(
        future_dynamics_terms(shifted, target)["future_transition"],
        torch.zeros(()),
    )
    assert future_dynamics_terms(collapsed, target)["future_transition"] > 0


def test_teacher_reliability_falls_for_semantically_opposed_supports() -> None:
    torch.manual_seed(3)
    local = _local_facts(content=8, route=4, hidden=16)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    facts, _ = grounder(local)
    weight = facts.semantic_candidate_assignment.flatten(2).float()
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    candidates = facts.dense_chart.candidate_content.flatten(1, -2).float()
    semantic_current = torch.einsum("bkn,bnd->bkd", weight, candidates)
    offsets = torch.arange(4, 49, 4)[None]
    high = semantic_current[:, 0, None, None, None, None].expand(-1, 12, 1, 2, 2, -1)
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    high_target, _ = teacher(
        facts=facts,
        future_supports=high,
        future_offsets=offsets,
    )
    low_target, _ = teacher(
        facts=facts,
        future_supports=-high,
        future_offsets=offsets,
    )
    assert high_target.reliability[:, :, 0].mean() > low_target.reliability[:, :, 0].mean()


def test_teacher_diffuse_visible_track_falls_back_continuously_to_current_fact() -> None:
    torch.manual_seed(31)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(content=8, route=4, hidden=16))
    future_value = torch.randn(1, 1, 1, 1, 1, 8)
    supports = future_value.expand(1, 12, 1, 2, 2, 8).clone()
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    target, _ = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=torch.arange(4, 49, 4)[None],
    )
    full_delta = future_value.reshape(1, 1, 1, 8) - target.current_reference[:, None]
    observed_delta = target.successor_content - target.current_reference[:, None]
    confidence = (observed_delta * full_delta).sum(dim=-1, keepdim=True) / full_delta.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    torch.testing.assert_close(
        observed_delta,
        confidence * full_delta,
        atol=2e-5,
        rtol=2e-5,
    )
    assert bool((confidence >= 0.0).all())
    assert bool((confidence < 1.0).all())
    address_mass = target.future_address.sum(dim=(-3, -2, -1))
    torch.testing.assert_close(address_mass, torch.ones_like(address_mass))


def test_teacher_track_is_equivariant_to_global_object_relabeling() -> None:
    torch.manual_seed(26)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=2,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    supports = torch.randn(1, 12, 2, 2, 2, 8)
    offsets = torch.arange(4, 49, 4)[None]
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    target, _ = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=offsets,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    relabeled, _ = teacher(
        facts=facts.permute(permutation),
        future_supports=supports,
        future_offsets=offsets,
    )
    expected = target.permute(permutation)
    for field in fields(FutureObjectDynamics):
        assert torch.allclose(
            getattr(relabeled, field.name),
            getattr(expected, field.name),
            atol=1e-6,
            rtol=1e-6,
        ), field.name


def test_global_object_axis_survives_s_w_and_p_without_order_dependence() -> None:
    torch.manual_seed(27)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    ).eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    goal_tokens = torch.randn(1, 6, 12)
    goal_mask = torch.ones(1, 6, dtype=torch.bool)
    state_history = torch.randn(1, 3, 7)
    state = torch.randn(1, 7)
    executed_history = torch.randn(1, 3, 7)

    def organize(current_facts):
        return top.intent(
            goal_tokens=goal_tokens,
            goal_mask=goal_mask,
            state_history=state_history,
            state=state,
            executed_history=executed_history,
            facts=current_facts,
            collect_diagnostics=False,
        )[0]

    intent = organize(facts)
    permutation = torch.tensor([3, 1, 0, 2])
    relabeled_facts = facts.permute(permutation)
    relabeled_intent = organize(relabeled_facts)
    expected_intent = intent.permute(permutation)
    for field in fields(type(intent)):
        assert torch.allclose(
            getattr(relabeled_intent, field.name),
            getattr(expected_intent, field.name),
            atol=2e-5,
            rtol=2e-5,
        ), field.name

    coarse = top.coarse_action(intent)
    relabeled_coarse = top.coarse_action(relabeled_intent)
    assert torch.allclose(
        relabeled_coarse.tokens,
        coarse.tokens,
        atol=2e-5,
        rtol=2e-5,
    )

    _, w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent,
        action=coarse,
        collect_diagnostics=False,
    )
    dynamics, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent,
        action=coarse,
        w1_state=w1,
        collect_diagnostics=False,
    )
    _, relabeled_w1, _ = top.dynamics.forward_w1(
        facts=relabeled_facts,
        intent=relabeled_intent,
        action=relabeled_coarse,
        collect_diagnostics=False,
    )
    relabeled_dynamics, _ = top.dynamics.forward_w2(
        facts=relabeled_facts,
        intent=relabeled_intent,
        action=relabeled_coarse,
        w1_state=relabeled_w1,
        collect_diagnostics=False,
    )
    expected_dynamics = dynamics.permute(permutation)
    for field in fields(FutureObjectDynamics):
        assert torch.allclose(
            getattr(relabeled_dynamics, field.name),
            getattr(expected_dynamics, field.name),
            atol=2e-5,
            rtol=2e-5,
        ), field.name

    batch, horizon, basis, objects, hidden = 1, 24, 2, 4, 32
    fact_by_object = torch.randn(batch, horizon, basis, objects, hidden)
    posterior = torch.softmax(torch.randn(batch, horizon, basis, objects + 1), dim=-1)
    dock = ObjectFactualDock(
        fact_by_object=fact_by_object,
        object_posterior=posterior[..., :-1],
        null_posterior=posterior[..., -1:],
        chart_posterior=facts.object_to_chart[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1, -1
        ),
        camera_coordinates=facts.camera_coordinates[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1
        ),
        aggregate_fact=torch.einsum("btqk,btqkh->btqh", posterior[..., :-1], fact_by_object),
    )
    action_query = torch.randn(batch, horizon, basis, hidden)
    compiled, _ = top.compile_policy(
        DeploymentTopCache(intent=intent, predicted_dynamics=dynamics),
        factual_dock=dock,
        action_query=action_query,
    )
    relabeled_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=relabeled_intent,
            predicted_dynamics=relabeled_dynamics,
        ),
        factual_dock=dock.permute(permutation),
        action_query=action_query,
    )
    for name in ("effect",):
        assert torch.allclose(
            getattr(relabeled_compiled, name),
            getattr(compiled, name),
            atol=2e-5,
            rtol=2e-5,
        )
    for name in ("factual_base", "effect", "interaction", "protected_consequence"):
        assert torch.allclose(
            getattr(relabeled_compiled.consequence, name),
            getattr(compiled.consequence, name),
            atol=2e-5,
            rtol=2e-5,
        ), name
    for name in ("protected_base", "precision", "temporal", "state_change"):
        assert torch.allclose(
            getattr(relabeled_compiled.plan, name),
            getattr(compiled.plan, name),
            atol=2e-5,
            rtol=2e-5,
        ), name


def test_neutral_w_preserves_current_precision_and_temporal_without_w_interaction() -> None:
    torch.manual_seed(4)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    )
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    horizon, basis, objects, hidden = 24, 2, 4, 32
    posterior = torch.softmax(torch.randn(1, horizon, basis, objects + 1), dim=-1)
    fact_by_object = torch.randn(1, horizon, basis, objects, hidden)
    dock = ObjectFactualDock(
        fact_by_object=fact_by_object,
        object_posterior=posterior[..., :-1],
        null_posterior=posterior[..., -1:],
        chart_posterior=context.facts.object_to_chart[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1, -1
        ),
        camera_coordinates=context.facts.camera_coordinates[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1
        ),
        aggregate_fact=torch.einsum("btqk,btqkh->btqh", posterior[..., :-1], fact_by_object),
    )
    deployment = DeploymentTopCache(
        intent=context.intent,
        predicted_dynamics=FutureObjectDynamics.neutral(context.facts),
    )
    action_query = torch.randn(1, horizon, basis, hidden)
    compiled, _ = top.compile_policy(
        deployment,
        factual_dock=dock,
        action_query=action_query,
    )
    neutral_other_query, _ = top.compile_policy(
        deployment,
        factual_dock=dock,
        action_query=torch.randn(1, horizon, basis, hidden),
    )
    assert torch.count_nonzero(compiled.effect) == 0
    assert torch.count_nonzero(compiled.plan.precision) > 0
    assert torch.count_nonzero(compiled.plan.temporal) > 0
    assert torch.count_nonzero(compiled.plan.state_change) > 0
    # V120 keeps noisy-action modulation in its typed temporal lane.  The
    # protected factual consequence remains available independently.
    assert not torch.equal(compiled.plan.temporal, neutral_other_query.plan.temporal)
    identity_only_intent = replace(
        context.intent,
        interval_queries=context.intent.interval_queries + 1000.0 * torch.randn_like(
            context.intent.interval_queries
        ),
    )
    identity_only_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=identity_only_intent,
            predicted_dynamics=deployment.predicted_dynamics,
        ),
        factual_dock=dock,
        action_query=action_query,
    )
    # Cumulative/identity-bearing S queries are addresses internal to S.  P2
    # consumes only the observed interval innovation, so changing identity
    # alone cannot recreate a fixed temporal route prior.
    torch.testing.assert_close(
        compiled.effect,
        identity_only_compiled.effect,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        compiled.consequence.protected_consequence,
        dock.aggregate_fact,
        atol=0.0,
        rtol=0.0,
    )


def test_supervised_successor_innovation_crosses_w_to_p2_without_current_bypass() -> None:
    torch.manual_seed(28)
    top = ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
    )
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    horizon, basis, objects, hidden = 24, 2, 4, 32
    posterior = torch.softmax(torch.randn(1, horizon, basis, objects + 1), dim=-1)
    fact_by_object = torch.randn(1, horizon, basis, objects, hidden)
    dock = ObjectFactualDock(
        fact_by_object=fact_by_object,
        object_posterior=posterior[..., :-1],
        null_posterior=posterior[..., -1:],
        chart_posterior=context.facts.object_to_chart[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1, -1
        ),
        camera_coordinates=context.facts.camera_coordinates[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1
        ),
        aggregate_fact=torch.einsum("btqk,btqkh->btqh", posterior[..., :-1], fact_by_object),
    )
    neutral = FutureObjectDynamics.neutral(context.facts)
    changed = replace(
        neutral,
        semantic_delta=neutral.semantic_delta
        + 0.25 * torch.randn_like(neutral.semantic_delta),
    )
    action_query = torch.randn(1, horizon, basis, hidden)
    neutral_effect, _ = top.effect_reader(
        action_query,
        neutral,
        context.intent,
        collect_diagnostics=False,
    )
    changed_effect, _ = top.effect_reader(
        action_query,
        changed,
        context.intent,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(neutral_effect) == 0
    assert torch.count_nonzero(changed_effect) > 0


def test_bottom_optional_values_preserve_zero_and_do_not_expand_near_zero() -> None:
    torch.manual_seed(5)
    block = ReadOnlyEvidenceMMDiTBlock(
        hidden=16,
        heads=4,
        expansion=2.0,
        dropout=0.0,
        residual_scale_max=0.25,
        residual_scale_init=0.05,
        normalization_floor=0.25,
    )
    selector = torch.randn(2, 7, 16)
    action = torch.zeros(2, 5, 16)
    condition = torch.zeros(2, 16)
    zero = TypedEvidenceBank(
        selector=selector,
        value=torch.zeros_like(selector),
        lane_ranges={"test": (0, 7)},
    )
    zero_update, _ = block(
        action,
        zero,
        condition,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_update) == 0
    near = TypedEvidenceBank(
        selector=selector,
        value=1e-8 * torch.randn_like(selector),
        lane_ranges={"test": (0, 7)},
    )
    near_update, _ = block(
        action,
        near,
        condition,
        collect_diagnostics=False,
    )
    assert near_update.abs().amax() < 1e-5


def test_active_v120_capacity_is_full_identity_and_nested_nonexpansive() -> None:
    torch.manual_seed(23)
    operator = NestedLowRankContractionBank(
        hidden_size=16,
        condition_size=8,
        stage_count=1,
        rank=4,
        group_count=4,
        depth_logit_init=2.0,
    )
    update = torch.randn(2, 5, 16)
    condition = torch.randn(2, 8)
    stage = torch.zeros(2, dtype=torch.long)
    closed, closed_metrics = operator(
        update,
        condition,
        stage,
        depth_ratio_override=0.0,
    )
    basis = operator.prepare_factors()[0]
    expected_closed = update.float() - torch.einsum(
        "bnr,hr->bnh",
        torch.einsum("bnh,hr->bnr", update.float(), basis),
        basis,
    )
    torch.testing.assert_close(closed.float(), expected_closed, atol=2e-6, rtol=2e-6)
    # Capacity is ordered rank retention, not a host-residual amplitude gate.
    assert torch.count_nonzero(closed) > 0
    assert closed_metrics["effective_depth"] == 0

    full, metrics = operator(
        update,
        condition,
        stage,
        depth_ratio_override=1.0,
    )
    torch.testing.assert_close(full, update, atol=2e-6, rtol=2e-6)
    assert metrics["effective_depth"] == 4
    assert metrics["nonexpansive_violation"] < 1e-6

    middle, middle_metrics = operator(
        update,
        condition,
        stage,
        depth_ratio_override=0.375,
    )
    input_rms = update.float().square().mean(dim=(1, 2)).sqrt()
    output_rms = middle.float().square().mean(dim=(1, 2)).sqrt()
    assert bool((output_rms <= input_rms + 1e-6).all())
    assert middle_metrics["nonexpansive_violation"] < 1e-6


def test_active_v120_capacity_reuses_explicit_prepared_basis_within_a_forward() -> None:
    torch.manual_seed(231)
    operator = NestedLowRankContractionBank(
        hidden_size=16,
        condition_size=8,
        stage_count=1,
        rank=4,
        group_count=4,
        depth_logit_init=2.0,
    ).eval()
    update = torch.randn(2, 5, 16)
    condition = torch.randn(2, 8)
    stage = torch.zeros(2, dtype=torch.long)
    capacity = torch.full((2,), 0.75)
    prepared = operator.prepare_factors()
    with mock.patch("torch.linalg.qr", wraps=torch.linalg.qr) as qr:
        with torch.no_grad():
            first, _ = operator(
                update,
                condition,
                stage,
                depth_ratio_override=capacity,
                prepared_factors=prepared,
                collect_diagnostics=False,
            )
            second, _ = operator(
                update,
                condition,
                stage,
                depth_ratio_override=capacity,
                prepared_factors=prepared,
                collect_diagnostics=False,
            )
        assert qr.call_count == 0
        assert torch.equal(first, second)
        with torch.no_grad():
            operator(
                update,
                condition,
                stage,
                depth_ratio_override=capacity,
                collect_diagnostics=False,
            )
        assert qr.call_count == 1


def test_deployment_cache_has_no_source_or_training_charts() -> None:
    assert {field.name for field in fields(OnlinePolicyCache)} == {
        "top",
        "factual_dock",
        "transition_source",
        "history",
        "executed_memory",
        "action_history_keep",
    }
    assert {field.name for field in fields(DeploymentTopCache)} == {
        "intent",
        "predicted_dynamics",
    }
