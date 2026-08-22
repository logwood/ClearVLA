from __future__ import annotations

import copy
import inspect
import math
from dataclasses import fields, is_dataclass, replace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import CurrentObservation
from clearvla.mainline.model.bottom import (
    ReadOnlyEvidenceMMDiTBlock,
    TypedEvidenceBank,
)
from clearvla.mainline.model.compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
)
from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler
from clearvla.mainline.model.grounding import (
    DenseObjectGrounder,
    _conditional_k_reconstruction_assignment,
    dense_chart_from_local_facts,
)
from clearvla.mainline.model.intent import (
    TYPED_INTENT_NAMES,
    CoarseActionIntent,
    DirectIntentFutureSupervisor,
    StatelessObjectIntentOrganizer,
    _CrossRead,
    _SelfBlock,
)
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
    INTERVAL_BOUNDS,
    FutureObjectDynamics,
    LocalFactSet,
    PolicyIntentDock,
)
from clearvla.mainline.training.losses import flow_geometry_terms, future_dynamics_terms
from clearvla.mainline.v120_core.profile import build_v120_visual_config
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


def _object_top() -> ObjectIntentDynamicsTop:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=12,
            action_basis_tokens=2,
        ),
        bottom=replace(base.bottom, controller_heads=4),
    )
    config.validate()
    return ObjectIntentDynamicsTop(
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
        core_config=build_v120_visual_config(config),
    )


def test_grounder_owns_only_one_dense_reconstruction_objective() -> None:
    torch.manual_seed(1)
    local = _local_facts(
        content=8,
        route=4,
        hidden=16,
        valid=True,
        observed=True,
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
    assert metrics["object_grounding_reconstruction_mse"] > 0
    assert not any(
        name in " ".join(metrics)
        for name in ("prototype", "masked_reconstruction", "typed_consistency")
    )
    facts.reconstruction_error.backward()
    # V120's binder does not inject the already-consumed public G chart into
    # every K candidate a second time.
    assert public.grad is None
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0


def test_grounder_reconstructs_the_independent_observed_dino_chart() -> None:
    local = _local_facts(content=8, route=4, hidden=16, observed=True)
    chart = dense_chart_from_local_facts(local)
    torch.testing.assert_close(
        chart.dino_content,
        local.target_dino_content,
        atol=0.0,
        rtol=0.0,
    )
    self_mixture = (
        local.content_slots
        * local.semantic_owner_probs[..., None]
    ).sum(dim=-2)
    assert not torch.equal(chart.dino_content, self_mixture)

    masked = replace(
        local,
        cell_observed=torch.zeros_like(local.cell_observed),
    )
    _, metrics = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(masked)
    assert float(metrics["object_grounding_reconstruction_mse"]) == 0.0


def test_grounder_reconstruction_assignment_is_null_independent_and_zero_safe() -> None:
    torch.manual_seed(201)
    conditional_k = torch.softmax(torch.randn(2, 7, 4), dim=-1)
    local_prior = torch.rand(2, 7, 1)
    validity = torch.ones_like(local_prior)
    assignment = _conditional_k_reconstruction_assignment(
        conditional_k,
        local_prior,
        validity,
    )
    torch.testing.assert_close(
        assignment.sum(dim=-1, keepdim=True),
        local_prior,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Changing object-vs-null mass cannot change the conditional-K posterior
    # presented to reconstruction.
    high_k_mass = 0.9 * conditional_k
    low_k_mass = 0.001 * conditional_k
    high_conditional = high_k_mass / high_k_mass.sum(dim=-1, keepdim=True)
    low_conditional = low_k_mass / low_k_mass.sum(dim=-1, keepdim=True)
    scaled = _conditional_k_reconstruction_assignment(
        low_conditional,
        local_prior,
        validity,
    )
    high = _conditional_k_reconstruction_assignment(
        high_conditional,
        local_prior,
        validity,
    )
    torch.testing.assert_close(scaled, assignment, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(high, assignment, atol=1.0e-6, rtol=1.0e-6)

    zeros = _conditional_k_reconstruction_assignment(
        torch.zeros_like(conditional_k),
        local_prior,
        validity,
    )
    assert torch.equal(zeros, torch.zeros_like(zeros))

    invalid = _conditional_k_reconstruction_assignment(
        conditional_k,
        local_prior,
        torch.zeros_like(validity),
    )
    assert torch.equal(invalid, torch.zeros_like(invalid))


def test_grounder_dense_reconstruction_uses_exported_conditional_k_content() -> None:
    torch.manual_seed(202)
    local = _local_facts(content=8, route=4, hidden=16, observed=True)
    facts, metrics = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(local)
    assignment = facts.candidate_assignment.float()
    k_mass = assignment.sum(dim=1, keepdim=True)
    conditional_k = torch.where(
        k_mass > 1.0e-8,
        assignment / k_mass.clamp_min(1.0e-8),
        torch.zeros_like(assignment),
    )
    local_prior = facts.dense_chart.candidate_owner_prior[:, None].float()
    reconstruction_owner = (conditional_k * local_prior).sum(dim=-1)
    expected = facts.public_content[:, None, None, None] + torch.einsum(
        "bkcyx,bkd->bcyxd",
        reconstruction_owner,
        facts.content_innovation.float(),
    )
    torch.testing.assert_close(
        facts.reconstructed_dino.float(), expected, atol=2.0e-6, rtol=1.0e-5
    )
    assert float(metrics["object_grounding_reconstruction_object_mass_mean"]) > 0.99
    assert float(metrics["object_grounding_reconstruction_active_fraction"]) == 1.0


def test_typed_compatibility_votes_before_one_physical_k_binding() -> None:
    grounder = DenseObjectGrounder(
        hidden=4,
        content_dim=4,
        route_dim=2,
        objects=2,
        iterations=1,
    )
    slots = torch.tensor([[[2.0, -1.0, -1.0, 0.0], [-1.0, 2.0, -1.0, 0.0]]])
    views = torch.zeros(1, 1, 3, 4)
    views[..., 0, :] = slots[:, :1]
    views[..., 1, :] = slots[:, 1:2]
    views[..., 2, :] = slots[:, 1:2]
    validity = torch.ones(1, 1, 1)
    prior = torch.ones(1, 1, 1)
    owner, typed_owner, mass, null, _ = grounder._competition(
        slots, views, validity, prior
    )
    assert tuple(typed_owner.shape) == (1, 1, 3, 3)
    assert owner[0, 0, 1] > owner[0, 0, 0]

    changed = views.clone()
    changed[..., 0, :] = 4.0 * slots[:, :1]
    changed_owner, _, changed_mass, changed_null, _ = grounder._competition(
        slots, changed, validity, prior
    )
    assert changed_owner[0, 0, 0] > owner[0, 0, 0]
    torch.testing.assert_close(
        mass.sum(dim=-1) + null,
        torch.ones_like(null),
    )
    torch.testing.assert_close(
        changed_mass.sum(dim=-1) + changed_null,
        torch.ones_like(changed_null),
    )


def test_intent_supervisor_uses_canonical_four_interval_physical_targets() -> None:
    torch.manual_seed(2)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=1))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    batch, intervals, objects, content, cameras = 1, 4, facts.objects, 16, 1
    scalar = torch.ones(batch, intervals, objects, 1)
    teacher = FutureObjectDynamics(
        current_reference=torch.randn(batch, objects, content),
        successor_content=torch.randn(batch, intervals, objects, content),
        semantic_delta=torch.randn(batch, intervals, objects, content),
        transport_mean=torch.randn(batch, intervals, objects, 2),
        transport_covariance=torch.rand(batch, intervals, objects, 3),
        visibility=torch.zeros_like(scalar),
        persistence=torch.zeros_like(scalar),
        uncertainty=torch.zeros_like(scalar),
        reliability=scalar,
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=scalar,
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    future_state = torch.randn(batch, 48, 7)
    intent_boundary = FutureObjectDynamics.neutral(facts)
    result = top.intent_supervisor(
        intent=intent,
        intent_boundary=intent_boundary,
        future_state=future_state,
        teacher=teacher,
        current_loss_support=torch.ones(batch, objects, cameras, 1),
    )
    expected_state = torch.stack(
        [future_state[:, lower - 1 : upper].mean(dim=1) for lower, upper in INTERVAL_BOUNDS],
        dim=1,
    )
    torch.testing.assert_close(result.state_target, expected_state)
    assert tuple(result.state_prediction.shape) == (batch, intervals, 7)
    assert tuple(result.semantic_prediction.shape) == (batch, intervals, objects, content)
    assert tuple(result.status_prediction.shape) == (batch, intervals, objects, 2)
    assert tuple(result.transport_prediction.shape) == (batch, intervals, objects, 2)
    torch.testing.assert_close(
        result.semantic_prediction,
        intent_boundary.semantic_delta,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.transport_prediction,
        intent_boundary.transport_mean,
        atol=0.0,
        rtol=0.0,
    )
    assert not any(
        name.startswith(("semantic_head", "status_head", "transport_head"))
        for name, _ in top.intent_supervisor.named_parameters()
    )
    assert "future_action" not in inspect.signature(
        DirectIntentFutureSupervisor.forward
    ).parameters


def test_intent_typed_loss_ignores_reliability_and_selector_validity() -> None:
    torch.manual_seed(22)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=1))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    batch, intervals, objects, content, cameras = 1, 4, facts.objects, 16, 1
    current = torch.randn(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    teacher = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.ones(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=torch.ones_like(scalar),
        reliability=scalar,
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=torch.zeros(batch, intervals, objects, 1),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    kwargs = dict(
        intent=intent,
        intent_boundary=FutureObjectDynamics.neutral(facts),
        future_state=torch.randn(batch, 48, 7),
        teacher=teacher,
        current_loss_support=torch.ones(batch, objects, cameras, 1),
    )
    result = top.intent_supervisor(**kwargs)
    changed = top.intent_supervisor(
        **{
            **kwargs,
            "teacher": replace(
                teacher,
                reliability=torch.ones_like(teacher.reliability),
                future_selector_validity=torch.ones_like(
                    teacher.future_selector_validity
                ),
            ),
        }
    )
    torch.testing.assert_close(result.semantic_target, torch.ones_like(result.semantic_target))
    torch.testing.assert_close(result.typed_loss, changed.typed_loss)


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


def test_g3_common_k_residual_cannot_change_object_vs_null_mass() -> None:
    """G3 owns conditional K identity, never the parent existence decision."""

    torch.manual_seed(211)
    local = _local_facts(cameras=2, content=8, route=4, hidden=16)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=2,
    ).eval()
    baseline, baseline_metrics = grounder(local)

    def common_residual(pair: torch.Tensor) -> torch.Tensor:
        return torch.ones(*pair.shape[:-1], 1, device=pair.device, dtype=pair.dtype)

    with mock.patch.object(
        grounder.g3_residual,
        "forward",
        side_effect=common_residual,
    ):
        common, common_metrics = grounder(local)
    torch.testing.assert_close(
        common.candidate_assignment,
        baseline.candidate_assignment,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        common.null_assignment,
        baseline.null_assignment,
        atol=2e-7,
        rtol=2e-7,
    )
    assert float(baseline_metrics["object_grounding_g3_null_identity_error"]) == 0.0
    assert float(common_metrics["object_grounding_g3_null_identity_error"]) == 0.0


def test_typed_grounding_cannot_resurrect_zero_physical_support() -> None:
    torch.manual_seed(212)
    local = _local_facts(content=8, route=4, hidden=16)
    validity = local.slot_validity.clone()
    validity[:, :, 0, 0, 0] = 0.0
    # Make the invalid hypothesis maximally attractive in every typed view.
    semantic = local.semantic_owner_probs.clone()
    appearance = local.appearance_owner_probs.clone()
    geometry = local.geometry_owner_probs.clone()
    for prior in (semantic, appearance, geometry):
        prior[:, :, 0, 0] = 0.0
        prior[:, :, 0, 0, 0] = 1.0
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(
        replace(
            local,
            slot_validity=validity,
            semantic_owner_probs=semantic,
            appearance_owner_probs=appearance,
            geometry_owner_probs=geometry,
        )
    )
    for assignment in (
        facts.candidate_assignment,
        facts.semantic_candidate_assignment,
        facts.appearance_candidate_assignment,
        facts.geometry_candidate_assignment,
    ):
        assert torch.count_nonzero(assignment[:, :, :, 0, 0, 0]) == 0


def test_object_camera_reduction_uses_joint_evidence_mass_not_width() -> None:
    torch.manual_seed(213)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    coordinates = torch.zeros_like(facts.camera_coordinates)
    coordinates[:, :, 0] = -1.0
    coordinates[:, :, 1] = 1.0
    evidence = torch.zeros_like(facts.camera_evidence_mass)
    evidence[:, :, 0] = 0.75
    evidence[:, :, 1] = 0.25
    changed = replace(
        facts,
        camera_coordinates=coordinates,
        camera_evidence_mass=evidence,
        camera_support=torch.flip(facts.camera_support, dims=(2,)),
        camera_validity=torch.ones_like(facts.camera_validity),
    )
    torch.testing.assert_close(
        changed.coordinates,
        torch.full_like(changed.coordinates, -0.5),
        atol=0.0,
        rtol=0.0,
    )


def test_object_content_boundary_is_one_public_base_plus_k_innovations() -> None:
    torch.manual_seed(214)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    ).eval()
    facts, _ = grounder(_local_facts(cameras=2, content=8, route=4, hidden=16))
    assert facts.public_content.dtype == facts.content.dtype
    torch.testing.assert_close(
        facts.public_content[:, None] + facts.content_innovation,
        facts.content,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = facts.permute(permutation)
    torch.testing.assert_close(permuted.public_content, facts.public_content)
    torch.testing.assert_close(
        permuted.content_innovation,
        facts.content_innovation[:, permutation],
    )


def test_s_projects_public_scene_once_and_k_only_as_content_innovations() -> None:
    torch.manual_seed(215)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    public, private = top.intent._object_tokens(facts)
    shift = torch.full_like(facts.public_content, 0.125)
    shifted = replace(
        facts,
        public_content=facts.public_content + shift,
        content=facts.content + shift[:, None],
    )
    shifted_public, shifted_private = top.intent._object_tokens(shifted)
    torch.testing.assert_close(private, shifted_private, atol=2e-6, rtol=2e-6)
    assert not torch.equal(public, shifted_public)


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


def test_teacher_camera_relative_moments_do_not_invent_static_transport() -> None:
    axis = torch.tensor((-1.0, 0.0, 1.0))
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinate = torch.stack((x, y), dim=-1)[None].expand(2, -1, -1, -1)
    posterior = torch.zeros(1, 1, 1, 2, 3, 3)
    # One static object occupies different image coordinates in the two
    # cameras, while the future association changes camera mass. Subtracting
    # a separately reduced global current coordinate would invent +0.5 flow.
    posterior[0, 0, 0, 0, 1, 0] = 0.2
    posterior[0, 0, 0, 1, 1, 2] = 0.7
    current = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])
    transport, covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=posterior,
        null_probability=torch.full((1, 1, 1, 1), 0.1),
        candidate_coordinate=coordinate,
        current_camera_coordinate=current,
    )
    torch.testing.assert_close(transport, torch.zeros_like(transport))
    torch.testing.assert_close(covariance, torch.zeros_like(covariance))

    permutation = torch.tensor((1, 0))
    permuted_transport, permuted_covariance = (
        ObjectFutureTeacher._relative_geometry_moments(
            candidate_posterior=posterior[:, :, :, permutation],
            null_probability=torch.full((1, 1, 1, 1), 0.1),
            candidate_coordinate=coordinate[permutation],
            current_camera_coordinate=current[:, :, permutation],
        )
    )
    torch.testing.assert_close(permuted_transport, transport)
    torch.testing.assert_close(permuted_covariance, covariance)


def test_teacher_covariance_contains_identity_null_second_moment() -> None:
    coordinate = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])
    posterior = torch.zeros(1, 1, 1, 1, 1, 2)
    posterior[..., 1] = 0.5
    current = torch.zeros(1, 1, 1, 2)
    transport, covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=posterior,
        null_probability=torch.full((1, 1, 1, 1), 0.5),
        candidate_coordinate=coordinate,
        current_camera_coordinate=current,
    )
    torch.testing.assert_close(transport, torch.tensor([[[[0.5, 0.0]]]]))
    torch.testing.assert_close(covariance, torch.tensor([[[[0.25, 0.0, 0.0]]]]))


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


def test_grounder_has_no_learned_typed_or_prototype_shortcut_heads() -> None:
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    names = {name for name, _ in grounder.named_parameters()}
    assert not any("prototype" in name for name in names)
    assert not any("typed_verifier" in name for name in names)
    assert not any("masked" in name for name in names)


def test_grounder_does_not_reinject_public_chart_into_object_candidates() -> None:
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
    first_candidate = grounder._candidate_tokens(first)
    second_candidate = grounder._candidate_tokens(second)
    torch.testing.assert_close(first_candidate, second_candidate)
    names = {name for name, _ in grounder.named_parameters()}
    assert not any("public_address_key" in name for name in names)


def test_w_zero_initialized_visibility_preserves_current_selector_support() -> None:
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
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=torch.zeros(1, 2, 4, 3, 16),
    )
    torch.testing.assert_close(
        field.future_selector_validity,
        (facts.validity * facts.existence.detach())[:, None].expand(-1, 2, -1, -1),
    )

    # Predicted visibility is a status value, not authority to erase the
    # semantic/geometry candidates supervised in the same W field.
    with torch.no_grad():
        dynamics.visibility_head.weight.fill_(20.0)
    appearance = torch.zeros(1, 2, 4, 3, 16)
    appearance[..., 1, :] = 1.0
    suppressed = dynamics._field(
        facts=facts,
        hidden=torch.zeros(1, 2, 4, 16),
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=appearance,
    )
    torch.testing.assert_close(
        suppressed.future_selector_validity,
        field.future_selector_validity,
        atol=0.0,
        rtol=0.0,
    )


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
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=torch.zeros(1, 2, 4, 3, 16),
    )
    innovation = field.successor_content - field.current_reference[:, None]
    assert torch.count_nonzero(innovation) == 0
    innovation.sum().backward()
    assert facts.content.grad is None or torch.count_nonzero(facts.content.grad) == 0


def test_w_typed_common_residual_values_drive_only_owned_output_fields() -> None:
    torch.manual_seed(291)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=16, content_dim=8, route_dim=4, heads=4
    )
    with torch.no_grad():
        dynamics.delta_head.weight.fill_(0.05)
        dynamics.transport_head.weight.fill_(0.05)
        dynamics.covariance_head.weight.fill_(0.05)
        dynamics.visibility_head.weight.fill_(0.05)
        dynamics.persistence_head.weight.fill_(0.05)
    hidden = torch.zeros(1, 2, 4, 16)
    zero = torch.zeros(1, 2, 4, 3, 16)
    common = torch.zeros(1, 4, 3, 16)
    baseline = dynamics._field(
        facts=facts,
        hidden=hidden,
        typed_common=common,
        typed_interval_residual=zero,
    )

    def changed(type_index: int) -> FutureObjectDynamics:
        sidecars = zero.clone()
        sidecars[..., type_index, :] = torch.randn_like(
            sidecars[..., type_index, :]
        )
        return dynamics._field(
            facts=facts,
            hidden=hidden,
            typed_common=common,
            typed_interval_residual=sidecars,
        )

    semantic = changed(0)
    assert not torch.equal(semantic.semantic_delta, baseline.semantic_delta)
    torch.testing.assert_close(semantic.transport_mean, baseline.transport_mean)
    torch.testing.assert_close(semantic.visibility, baseline.visibility)

    appearance = changed(1)
    assert not torch.equal(appearance.visibility, baseline.visibility)
    assert not torch.equal(appearance.persistence, baseline.persistence)
    torch.testing.assert_close(appearance.semantic_delta, baseline.semantic_delta)
    torch.testing.assert_close(appearance.transport_mean, baseline.transport_mean)

    geometry = changed(2)
    assert not torch.equal(geometry.transport_mean, baseline.transport_mean)
    assert not torch.equal(
        geometry.transport_covariance, baseline.transport_covariance
    )
    torch.testing.assert_close(geometry.semantic_delta, baseline.semantic_delta)
    torch.testing.assert_close(geometry.visibility, baseline.visibility)
    for field in (semantic, appearance, geometry):
        torch.testing.assert_close(field.uncertainty, baseline.uncertainty)


def test_w_field_decoder_cannot_republicize_completed_typed_fields() -> None:
    torch.manual_seed(292)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(_local_facts(cameras=2, content=8, route=4, hidden=16))
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=16, content_dim=8, route_dim=4, heads=4
    )
    with torch.no_grad():
        dynamics.delta_head.weight.fill_(0.05)
        dynamics.transport_head.weight.fill_(0.05)
        dynamics.visibility_head.weight.fill_(0.05)
        dynamics.persistence_head.weight.fill_(0.05)
    zero = torch.zeros(1, 2, 4, 3, 16)
    public_a = torch.randn(1, 2, 4, 16)
    public_b = torch.randn_like(public_a)
    zero_a = dynamics._field(
        facts=facts,
        hidden=public_a,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=zero,
    )
    zero_b = dynamics._field(
        facts=facts,
        hidden=public_b,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=zero,
    )
    for name in (
        "semantic_delta",
        "transport_mean",
        "visibility",
        "persistence",
    ):
        torch.testing.assert_close(getattr(zero_a, name), getattr(zero_b, name))
        assert torch.count_nonzero(getattr(zero_a, name)) == 0

    typed = torch.randn_like(zero)
    typed_a = dynamics._field(
        facts=facts,
        hidden=public_a,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=typed,
    )
    typed_b = dynamics._field(
        facts=facts,
        hidden=public_b,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_residual=typed,
    )
    torch.testing.assert_close(typed_a.semantic_delta, typed_b.semantic_delta)
    torch.testing.assert_close(typed_a.transport_mean, typed_b.transport_mean)
    torch.testing.assert_close(typed_a.visibility, typed_b.visibility)


def test_w_typed_values_cross_w1_and_w2_and_preserve_exact_zero() -> None:
    torch.manual_seed(293)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    coarse = top.coarse_action(intent.action_dock())
    _, common_input, residual_input, _ = top.dynamics._base(
        facts,
        intent.world_dock(),
        coarse,
        collect_diagnostics=False,
    )
    _, working, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(working.common_typed, common_input)
    assert not torch.equal(working.near_residual_typed, residual_input[:, :2])
    completed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=working,
        collect_diagnostics=False,
    )
    completed.validate()

    zero_typed = replace(
        intent,
        typed_common_mass=torch.zeros_like(intent.typed_common_mass),
        typed_common_value=torch.zeros_like(intent.typed_common_value),
        typed_interval_residual_mass=torch.zeros_like(
            intent.typed_interval_residual_mass
        ),
        typed_interval_residual_value=torch.zeros_like(
            intent.typed_interval_residual_value
        ),
        typed_common_policy_components=torch.zeros_like(
            intent.typed_common_policy_components
        ),
        typed_interval_residual_policy_components=torch.zeros_like(
            intent.typed_interval_residual_policy_components
        ),
        policy_interval_context=intent.public_interval_carrier,
        policy_interval_innovation=intent.interval_condition_innovation,
    )
    zero_coarse = top.coarse_action(zero_typed.action_dock())
    _, zero_working, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=zero_typed.world_dock(),
        action=zero_coarse,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_working.common_typed) == 0
    assert torch.count_nonzero(zero_working.near_residual_typed) == 0
    assert torch.count_nonzero(zero_working.far_residual_typed) == 0
    zero_completed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=zero_typed.world_dock(),
        action=zero_coarse,
        w1_state=zero_working,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_completed.semantic_delta) == 0
    assert torch.count_nonzero(zero_completed.transport_mean) == 0
    assert torch.count_nonzero(zero_completed.visibility) == 0
    assert torch.count_nonzero(zero_completed.persistence) == 0


def test_w_receives_completed_intent_and_coarse_action_as_distinct_inputs() -> None:

    torch.manual_seed(31)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    blank_intent = replace(
        intent.world_dock(),
        interval_condition_innovation=torch.zeros_like(
            intent.interval_condition_innovation
        ),
    )
    signal_intent = replace(
        blank_intent,
        interval_condition_innovation=torch.randn_like(
            intent.interval_condition_innovation
        ),
    )
    coarse = top.coarse_action(intent.action_dock())
    zero_action = replace(coarse, tokens=torch.zeros_like(coarse.tokens))
    signal_action = replace(coarse, tokens=torch.randn_like(coarse.tokens))

    signal_zero, _, _, _ = top.dynamics._base(
        facts,
        signal_intent,
        zero_action,
        collect_diagnostics=False,
    )
    blank_zero, _, _, _ = top.dynamics._base(
        facts,
        blank_intent,
        zero_action,
        collect_diagnostics=False,
    )
    blank_action, _, _, _ = top.dynamics._base(
        facts,
        blank_intent,
        signal_action,
        collect_diagnostics=False,
    )
    assert not torch.equal(signal_zero, blank_zero)
    assert not torch.equal(blank_action, blank_zero)


def test_stateless_intent_is_repeatable_without_frame_progress_input() -> None:

    torch.manual_seed(30)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=2))
    kwargs = dict(
        goal_tokens=torch.zeros(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.zeros(1, 3, 7),
        state=torch.zeros(1, 7),
        executed_history=torch.zeros(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    first, _ = top.intent(**kwargs)
    second, _ = top.intent(**kwargs)
    torch.testing.assert_close(first.interval_queries, second.interval_queries)
    torch.testing.assert_close(first.temporal_queries, second.temporal_queries)


def test_goal_changes_interval_intent_without_rewriting_object_facts() -> None:

    torch.manual_seed(32)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=2))
    common = dict(
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.zeros(1, 3, 7),
        state=torch.zeros(1, 7),
        executed_history=torch.zeros(1, 8, 7),
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
    validity = torch.ones(batch, intervals, objects, 1)
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=validity,
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    prediction = replace(
        target,
        transport_mean=torch.ones_like(target.transport_mean),
    )
    current_support = torch.ones(batch, objects, cameras, 1)
    unreliable = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=current_support,
    )
    assert unreliable["future_transport"] > 0

    # The teacher's null association already turns content into the current
    # fact/zero-delta fallback.  That fallback remains an actual supervised W
    # target instead of being discounted a second time by reliability.
    changed_successor = replace(
        target,
        successor_content=target.successor_content + 1.0,
        semantic_delta=target.semantic_delta + 1.0,
    )
    unreliable_content = future_dynamics_terms(
        changed_successor,
        target,
        current_loss_support=current_support,
    )
    assert unreliable_content["future_semantic_common"] > 0
    assert unreliable_content["future_semantic_residual"] == 0
    assert unreliable_content["future_semantic_delta"] > 0

    reliable_target = replace(target, reliability=torch.ones_like(scalar))
    reliable_prediction = replace(prediction, reliability=torch.ones_like(scalar))
    reliable = future_dynamics_terms(
        reliable_prediction,
        reliable_target,
        current_loss_support=current_support,
    )
    torch.testing.assert_close(unreliable["future_transport"], reliable["future_transport"])

    # Predicted/target future support is diagnostic only: it is neither a loss
    # mask nor P2 routing authority, so status cannot self-mask its own value.
    selector_changed_target = replace(
        target,
        future_selector_validity=torch.zeros_like(target.future_selector_validity),
    )
    selector_changed_prediction = replace(
        prediction,
        future_selector_validity=torch.zeros_like(prediction.future_selector_validity),
    )
    selector_changed = future_dynamics_terms(
        selector_changed_prediction,
        selector_changed_target,
        current_loss_support=current_support,
    )
    for name in unreliable:
        torch.testing.assert_close(selector_changed[name], unreliable[name])

    unsupported = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=torch.zeros_like(current_support),
    )
    torch.testing.assert_close(unsupported["future_dynamics"], torch.zeros(()))


def test_training_support_uses_physical_camera_validity_not_assignment_mass() -> None:
    source = inspect.getsource(ObjectIntentDynamicsTop.build_training_targets)
    assert source.count("context.facts.camera_validity") == 2
    assert "camera_evidence_mass" not in source


def test_future_interval_transition_penalizes_temporal_collapse_not_common_offset() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    current = torch.zeros(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    interval = torch.arange(intervals, dtype=torch.float32)[None, :, None, None]
    semantic = interval.expand(batch, intervals, objects, content).clone()
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=semantic,
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=torch.ones(batch, intervals, objects, 1),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    shifted = replace(target, semantic_delta=semantic + 7.0)
    collapsed = replace(target, semantic_delta=torch.zeros_like(semantic))

    torch.testing.assert_close(
        future_dynamics_terms(
            shifted,
            target,
            current_loss_support=torch.ones(batch, objects, cameras, 1),
        )["future_transition"],
        torch.zeros(()),
    )
    assert (
        future_dynamics_terms(
            collapsed,
            target,
            current_loss_support=torch.ones(batch, objects, cameras, 1),
        )["future_transition"]
        > 0
    )


def test_semantic_deduplication_preserves_historical_gradient_coefficients() -> None:
    """Removing successor duplication must not amplify normalized pressure."""

    batch, intervals, objects, cameras, content = 1, 4, 1, 1, 4
    current = torch.zeros(batch, objects, content)
    scalar = torch.zeros(batch, intervals, objects, 1)
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=torch.ones(batch, intervals, objects, 1),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    semantic = torch.full_like(target.semantic_delta, 1.0e-3)
    prediction = replace(
        target,
        successor_content=current[:, None] + semantic,
        semantic_delta=semantic,
    )
    terms = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=torch.ones(batch, objects, cameras, 1),
    )
    raw = torch.tensor(0.5e-6)
    normalized = torch.tensor(0.5)
    direction = torch.tensor(0.25)
    compatibility_row = raw + (5.0 / 11.0) * normalized + (1.0 / 22.0) * direction
    # All four intervals carry the same error, hence only the protected common
    # half of the fixed common/residual budget is active.
    expected_semantic = 0.5 * compatibility_row
    torch.testing.assert_close(terms["future_semantic_delta"], expected_semantic)
    torch.testing.assert_close(
        terms["future_dynamics"],
        0.55 * expected_semantic,
    )


def test_teacher_partial_matching_prefers_a_local_semantic_peak_over_diffuse_opposition() -> None:
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
    opposed = -semantic_current[:, 0, None, None, None, None].expand(
        -1, 12, 1, 2, 2, -1
    ).clone()
    high = opposed.clone()
    high[:, :, 0, 0, 0] = semantic_current[:, 0, None].expand(-1, 12, -1)
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    high_target, _ = teacher(
        facts=facts,
        future_supports=high,
        future_offsets=offsets,
    )
    low_target, _ = teacher(
        facts=facts,
        future_supports=opposed,
        future_offsets=offsets,
    )
    assert high_target.reliability[:, :, 0].mean() > low_target.reliability[:, :, 0].mean()
    torch.testing.assert_close(
        high_target.semantic_delta,
        high_target.successor_content - high_target.current_reference[:, None],
    )
    torch.testing.assert_close(
        low_target.semantic_delta,
        low_target.successor_content - low_target.current_reference[:, None],
    )


def test_teacher_count_calibrated_partial_matching_uses_dustbin_for_diffuse_scores() -> None:
    objects, candidates = 4, 128
    diffuse = torch.full(
        (1, 2, objects, candidates),
        -math.log(float(candidates)),
    )
    diffuse_real, diffuse_null, diffuse_error = ObjectFutureTeacher._partial_assignment(
        diffuse
    )
    sharp = diffuse.clone()
    for object_index in range(objects):
        sharp[..., object_index, object_index] += 8.0
    sharp_real, sharp_null, sharp_error = ObjectFutureTeacher._partial_assignment(
        sharp
    )

    assert float(diffuse_null.mean()) > 0.5
    assert float(sharp_null.mean()) < float(diffuse_null.mean())
    torch.testing.assert_close(
        diffuse_real.sum(dim=-1, keepdim=True) + diffuse_null,
        torch.ones_like(diffuse_null),
        atol=1e-5,
        rtol=0.0,
    )
    assert float(diffuse_error.max()) < 1e-5
    assert float(sharp_error.max()) < 1e-5


def test_diffuse_teacher_keeps_raw_successor_without_reliability_contraction() -> None:
    """Entropy may lower reliability but must not contract physical content."""

    torch.manual_seed(33)
    local = _local_facts(content=8, route=4, hidden=16)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    facts, _ = grounder(local)
    teacher = ObjectFutureTeacher(content_dim=8, key_dim=4)
    with torch.no_grad():
        teacher.semantic_content_key.weight.zero_()
        teacher.appearance_content_key.weight.zero_()

    rows, columns = facts.object_to_chart.shape[-2:]
    support_value = torch.linspace(-0.75, 0.75, 8)
    supports = support_value.reshape(1, 1, 1, 1, 1, 8).expand(
        1, 4, 1, rows, columns, 8
    )
    target, _ = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=torch.tensor((6, 12, 24, 40)),
    )

    visibility = 1.0 + target.visibility.float()
    expected_successor = (
        visibility * support_value[None, None, None].float()
        + (1.0 - visibility) * target.current_reference[:, None].float()
    )
    torch.testing.assert_close(target.successor_content.float(), expected_successor)
    torch.testing.assert_close(
        target.semantic_delta.float(),
        target.successor_content.float() - target.current_reference[:, None].float(),
    )
    assert torch.isfinite(target.transport_mean).all()
    assert torch.isfinite(target.transport_covariance).all()
    assert target.reliability.mean() < visibility.mean()


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
        atol=5e-5,
        rtol=1e-2,
    )
    assert bool((confidence >= 0.0).all())
    assert bool((confidence < 1.0).all())
    assert torch.isfinite(target.transport_mean).all()


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


def test_teacher_camera_relabeling_preserves_object_geometry() -> None:
    torch.manual_seed(261)
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
    camera_permutation = torch.tensor([1, 0])
    chart = facts.dense_chart
    permuted_chart = type(chart)(
        **{
            field.name: getattr(chart, field.name)[:, camera_permutation]
            for field in fields(type(chart))
        }
    )
    permuted_facts = replace(
        facts,
        dense_chart=permuted_chart,
        camera_coordinates=facts.camera_coordinates[:, :, camera_permutation],
        camera_transport_prior=facts.camera_transport_prior[:, :, camera_permutation],
        camera_support=facts.camera_support[:, :, camera_permutation],
        camera_validity=facts.camera_validity[:, :, camera_permutation],
        camera_evidence_mass=facts.camera_evidence_mass[:, :, camera_permutation],
        object_to_chart=facts.object_to_chart[:, :, camera_permutation],
        candidate_assignment=facts.candidate_assignment[:, :, camera_permutation],
        semantic_candidate_assignment=(
            facts.semantic_candidate_assignment[:, :, camera_permutation]
        ),
        appearance_candidate_assignment=(
            facts.appearance_candidate_assignment[:, :, camera_permutation]
        ),
        geometry_candidate_assignment=(
            facts.geometry_candidate_assignment[:, :, camera_permutation]
        ),
        null_assignment=facts.null_assignment[:, camera_permutation],
        reconstructed_dino=facts.reconstructed_dino[:, camera_permutation],
    )
    permuted_facts.validate()
    relabeled, _ = teacher(
        facts=permuted_facts,
        future_supports=supports[:, :, camera_permutation],
        future_offsets=offsets,
    )
    for field in fields(FutureObjectDynamics):
        expected = getattr(target, field.name)
        torch.testing.assert_close(
            getattr(relabeled, field.name),
            expected,
            atol=2e-6,
            rtol=2e-6,
            msg=field.name,
        )


def test_global_object_axis_survives_s_w_and_p_without_order_dependence() -> None:
    torch.manual_seed(27)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    goal_tokens = torch.randn(1, 6, 12)
    goal_mask = torch.ones(1, 6, dtype=torch.bool)
    state_history = torch.randn(1, 3, 7)
    state = torch.randn(1, 7)
    executed_history = torch.randn(1, 8, 7)

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

    coarse = top.coarse_action(intent.action_dock())
    relabeled_coarse = top.coarse_action(relabeled_intent.action_dock())
    assert torch.allclose(
        relabeled_coarse.tokens,
        coarse.tokens,
        atol=2e-5,
        rtol=2e-5,
    )

    _, w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=False,
    )
    dynamics, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=w1,
        collect_diagnostics=False,
    )
    _, relabeled_w1, _ = top.dynamics.forward_w1(
        facts=relabeled_facts,
        intent=relabeled_intent.world_dock(),
        action=relabeled_coarse,
        collect_diagnostics=False,
    )
    relabeled_dynamics, _ = top.dynamics.forward_w2(
        facts=relabeled_facts,
        intent=relabeled_intent.world_dock(),
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

    batch, horizon, basis, hidden = 1, 24, 2, 32
    p1_fact = torch.randn(batch, horizon, basis, hidden)
    action_query = torch.randn(batch, horizon, basis, hidden)
    compiled, _ = top.compile_policy(
        DeploymentTopCache(intent=intent, predicted_dynamics=dynamics),
        p1_fact=p1_fact,
        p1_precision_innovation=p1_fact,
        action_query=action_query,
    )
    relabeled_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=relabeled_intent,
            predicted_dynamics=relabeled_dynamics,
        ),
        p1_fact=p1_fact,
        p1_precision_innovation=p1_fact,
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


def test_s_owns_per_type_object_relevance_and_fixed_zero_null_values() -> None:
    torch.manual_seed(37)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    inputs = {
        "goal_tokens": torch.randn(1, 6, 12),
        "goal_mask": torch.ones(1, 6, dtype=torch.bool),
        "state_history": torch.randn(1, 3, 7),
        "state": torch.randn(1, 7),
        "executed_history": torch.randn(1, 8, 7),
        "collect_diagnostics": False,
    }

    def organize(current_facts):
        return top.intent(facts=current_facts, **inputs)[0]

    intent = organize(facts)
    assert tuple(intent.typed_common_mass.shape[:3]) == (1, 4, 3)
    assert tuple(intent.typed_common_value.shape[:3]) == (1, 4, 3)
    assert tuple(intent.typed_interval_residual_mass.shape[:4]) == (1, 4, 4, 3)
    assert tuple(intent.typed_interval_residual_value.shape[:4]) == (1, 4, 4, 3)

    semantic_facts = replace(
        facts,
        semantic=facts.semantic.roll(1, dims=-1) + 0.13,
    )
    semantic_intent = organize(semantic_facts)
    torch.testing.assert_close(
        semantic_intent.public_interval_carrier,
        intent.public_interval_carrier,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(
        semantic_intent.typed_common_value[..., 0, :],
        intent.typed_common_value[..., 0, :],
    )
    torch.testing.assert_close(
        semantic_intent.typed_common_mass[..., 1:, :],
        intent.typed_common_mass[..., 1:, :],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        semantic_intent.typed_common_value[..., 1:, :],
        intent.typed_common_value[..., 1:, :],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        semantic_intent.typed_interval_residual_value[..., 1:, :],
        intent.typed_interval_residual_value[..., 1:, :],
        atol=0.0,
        rtol=0.0,
    )

    zero_semantic = organize(replace(facts, semantic=torch.zeros_like(facts.semantic)))
    assert torch.count_nonzero(zero_semantic.typed_common_value[..., 0, :]) == 0
    assert torch.count_nonzero(
        zero_semantic.typed_interval_residual_value[..., 0, :]
    ) == 0
    assert torch.count_nonzero(
        zero_semantic.typed_common_policy_components[..., 0, :]
    ) == 0
    assert torch.count_nonzero(
        zero_semantic.typed_interval_residual_policy_components[..., 0, :]
    ) == 0

    invalid_facts = replace(facts, validity=torch.zeros_like(facts.validity))
    invalid_intent = organize(invalid_facts)
    assert torch.count_nonzero(invalid_intent.typed_common_value) == 0
    assert torch.count_nonzero(invalid_intent.typed_interval_residual_value) == 0
    assert torch.count_nonzero(invalid_intent.typed_common_policy_components) == 0
    assert torch.count_nonzero(
        invalid_intent.typed_interval_residual_policy_components
    ) == 0
    torch.testing.assert_close(
        invalid_intent.policy_interval_context,
        invalid_intent.public_interval_carrier,
        atol=0.0,
        rtol=0.0,
    )
    coarse = top.coarse_action(invalid_intent.action_dock())
    _, _, _, w_metrics = top.dynamics._base(
        invalid_facts,
        invalid_intent.world_dock(),
        coarse,
        collect_diagnostics=True,
    )
    assert float(w_metrics["object_w_typed_common_state_rms"]) == 0.0
    assert float(w_metrics["object_w_typed_interval_residual_state_rms"]) == 0.0


def test_s_zero_goal_is_an_exact_language_null_not_a_learned_query_value() -> None:
    torch.manual_seed(370)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.zeros(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(intent.protected_goal_set) == 0
    _, _, _, metrics = top.dynamics._base(
        facts,
        intent.world_dock(),
        top.coarse_action(intent.action_dock()),
        collect_diagnostics=True,
    )
    assert float(metrics["object_w_goal_innovation_rms"]) == 0.0


def test_s_typed_selector_uses_real_interval_innovation_without_forcing_it() -> None:
    torch.manual_seed(371)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    zero = torch.zeros(1, 4, 32)
    zero_common_mass, zero_common_value, zero_residual_mass, zero_residual_value, *_ = top.intent._typed_relevance(
        interval_condition_innovation=zero,
        facts=facts,
    )
    assert torch.count_nonzero(zero_common_mass) == 0
    assert torch.count_nonzero(zero_common_value) == 0
    assert torch.count_nonzero(zero_residual_mass) == 0
    assert torch.count_nonzero(zero_residual_value) == 0

    common = torch.randn(1, 1, 32).expand(-1, 4, -1).clone()
    (
        common_mass,
        common_value,
        common_residual_mass,
        common_residual_value,
        _,
        _,
        common_score,
        common_differential_score,
        _,
        _,
        _,
        common_denominator,
        common_differential_denominator,
    ) = top.intent._typed_relevance(
        interval_condition_innovation=common,
        facts=facts,
    )
    assert torch.count_nonzero(common_mass) > 0
    assert torch.count_nonzero(common_value) > 0
    assert torch.count_nonzero(common_residual_mass) == 0
    assert torch.count_nonzero(common_residual_value) == 0
    assert torch.count_nonzero(common_differential_score) == 0
    assert float(common_denominator.amin()) >= 0.25
    assert float(common_differential_denominator.amin()) >= 0.25
    assert float(common_score.detach().abs().max()) <= 1.0

    tiny_mass, *_ = top.intent._typed_relevance(
        interval_condition_innovation=common * 1e-6,
        facts=facts,
    )
    assert float(tiny_mass.detach().amax()) < float(
        common_mass.detach().amax()
    ) * 1e-3

    differential = common.clone()
    differential[:, 2] = differential[:, 2] + torch.randn_like(differential[:, 2])
    (
        _,
        _,
        differential_mass,
        differential_value,
        _,
        _,
        _,
        raw_differential_score,
        _,
        _,
        _,
        _,
        differential_denominator,
    ) = top.intent._typed_relevance(
        interval_condition_innovation=differential,
        facts=facts,
    )
    assert float(
        differential_mass.detach().std(dim=1, unbiased=False).mean()
    ) > 0.0
    torch.testing.assert_close(
        differential_value.float().mean(dim=1),
        torch.zeros_like(differential_value[:, 0]).float(),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert torch.count_nonzero(raw_differential_score) > 0
    assert float(differential_denominator.amin()) >= 0.25
    assert float(raw_differential_score.detach().abs().max()) <= 1.0


def test_s_history_is_a_typed_time_union_not_fake_state_action_pairs() -> None:
    state_history = torch.tensor([[[80.0], [40.0], [999.0]]])
    current_state = torch.tensor([[1.0]])
    executed = torch.tensor(
        [[[24.0], [16.0], [12.0], [8.0], [6.0], [4.0], [2.0], [1.0]]]
    )
    packed, delta = StatelessObjectIntentOrganizer._paired_history(
        state_history,
        current_state,
        executed,
    )
    assert tuple(packed.shape) == (1, 11, 5)
    assert tuple(delta.shape) == (1, 11, 1)
    expected_offsets = torch.tensor(
        [-1.0, -2 / 3, -0.5, -1 / 3, -1 / 3, -1 / 4, -1 / 6, -1 / 6, -1 / 12, -1 / 24, 0.0]
    )
    torch.testing.assert_close(packed[0, :, -2], expected_offsets)
    expected_owner = torch.tensor(
        [-1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1],
        dtype=packed.dtype,
    )
    torch.testing.assert_close(packed[0, :, -1], expected_owner)
    state_rows = packed[0, :, -1] > 0
    action_rows = ~state_rows
    assert torch.count_nonzero(packed[0, state_rows, 1]) == 0
    assert torch.count_nonzero(packed[0, action_rows, 0]) == 0
    # The stale offset-zero history row is replaced, not appended or paired.
    assert 999.0 not in packed
    assert int((packed[0, :, 0] == 1.0).sum()) == 1


def test_coarse_action_has_no_typed_value_bypass_or_raw_g_reread() -> None:
    torch.manual_seed(372)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    action_field_names = {field.name for field in fields(intent.action_dock())}
    assert "interval_condition_innovation" in action_field_names
    assert "public_interval_carrier" not in action_field_names
    assert "public_scene_memory" in action_field_names
    assert "object_innovation_memory" in action_field_names
    assert "typed_interval_object_value" not in action_field_names
    assert "typed_interval_object_value" not in inspect.getsource(
        type(top.coarse_action).forward
    )
    assert "facts." not in inspect.getsource(type(top.coarse_action).forward)

    zero_typed = replace(
        intent,
        typed_common_mass=torch.zeros_like(intent.typed_common_mass),
        typed_common_value=torch.zeros_like(intent.typed_common_value),
        typed_interval_residual_mass=torch.zeros_like(
            intent.typed_interval_residual_mass
        ),
        typed_interval_residual_value=torch.zeros_like(
            intent.typed_interval_residual_value
        ),
        typed_common_policy_components=torch.zeros_like(
            intent.typed_common_policy_components
        ),
        typed_interval_residual_policy_components=torch.zeros_like(
            intent.typed_interval_residual_policy_components
        ),
        policy_interval_context=intent.public_interval_carrier,
        policy_interval_innovation=intent.interval_condition_innovation,
    )
    coarse = top.coarse_action(intent.action_dock())
    zero_coarse = top.coarse_action(zero_typed.action_dock())
    torch.testing.assert_close(coarse.tokens, zero_coarse.tokens, atol=0.0, rtol=0.0)

    dock = intent.action_dock()
    empty_dock = replace(
        dock,
        interval_condition_innovation=torch.zeros_like(
            dock.interval_condition_innovation
        ),
        history_memory=torch.zeros_like(dock.history_memory),
        public_scene_memory=torch.zeros_like(dock.public_scene_memory),
        object_innovation_memory=torch.zeros_like(dock.object_innovation_memory),
    )
    empty_coarse = top.coarse_action(empty_dock)
    assert torch.count_nonzero(empty_coarse.tokens) == 0
    assert torch.count_nonzero(empty_coarse.action_prediction) == 0

    with_typed, with_common, with_residual, _ = top.dynamics._base(
        facts,
        intent.world_dock(),
        coarse,
        collect_diagnostics=False,
    )
    without_typed, without_common, without_residual, _ = top.dynamics._base(
        facts,
        zero_typed.world_dock(),
        coarse,
        collect_diagnostics=False,
    )
    # Typed evidence must not be re-publicized into W's common carrier.  It
    # crosses the boundary only through field-owned sidecars.
    torch.testing.assert_close(with_typed, without_typed, atol=0.0, rtol=0.0)
    assert not torch.equal(with_common, without_common)
    assert not torch.equal(with_residual, without_residual)


def test_p2_common_and_residual_keys_do_not_duplicate_typed_policy_context() -> None:
    torch.manual_seed(373)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    dock = intent.policy_dock()
    torch.testing.assert_close(
        dock.common_key,
        intent.interval_condition_innovation.mean(dim=1),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dock.interval_residual_key,
        intent.interval_condition_innovation
        - intent.interval_condition_innovation.mean(dim=1, keepdim=True),
        atol=0.0,
        rtol=0.0,
    )
    changed = replace(
        intent,
        policy_interval_innovation=(
            intent.policy_interval_innovation
            + torch.randn_like(intent.policy_interval_innovation)
        ),
    )
    torch.testing.assert_close(
        changed.policy_dock().common_key,
        dock.common_key,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        changed.policy_dock().interval_residual_key,
        dock.interval_residual_key,
        atol=0.0,
        rtol=0.0,
    )


def test_typed_owner_relabeling_is_equivariant_through_coarse_action_and_w() -> None:
    torch.manual_seed(39)
    top = _object_top().eval()
    relabeled_top = copy.deepcopy(top)
    semantic_query = relabeled_top.intent.typed_relevance_queries[0]
    appearance_query = relabeled_top.intent.typed_relevance_queries[1]
    relabeled_top.intent.typed_relevance_queries[0] = appearance_query
    relabeled_top.intent.typed_relevance_queries[1] = semantic_query
    semantic_projection = relabeled_top.intent.object_semantic
    appearance_projection = relabeled_top.intent.object_appearance
    relabeled_top.intent.object_semantic = appearance_projection
    relabeled_top.intent.object_appearance = semantic_projection
    semantic_projection = relabeled_top.dynamics.object_semantic
    appearance_projection = relabeled_top.dynamics.object_appearance
    relabeled_top.dynamics.object_semantic = appearance_projection
    relabeled_top.dynamics.object_appearance = semantic_projection

    facts, _ = top.grounder(_local_facts(cameras=2))
    relabeled_facts = replace(
        facts,
        semantic=facts.appearance,
        appearance=facts.semantic,
    )
    inputs = {
        "goal_tokens": torch.randn(1, 6, 12),
        "goal_mask": torch.ones(1, 6, dtype=torch.bool),
        "state_history": torch.randn(1, 3, 7),
        "state": torch.randn(1, 7),
        "executed_history": torch.randn(1, 8, 7),
        "collect_diagnostics": False,
    }
    intent = top.intent(facts=facts, **inputs)[0]
    relabeled_intent = relabeled_top.intent(facts=relabeled_facts, **inputs)[0]

    torch.testing.assert_close(
        relabeled_intent.public_interval_carrier,
        intent.public_interval_carrier,
    )
    torch.testing.assert_close(
        relabeled_intent.typed_common_mass[..., (0, 1), :],
        intent.typed_common_mass[..., (1, 0), :],
    )
    torch.testing.assert_close(
        relabeled_intent.typed_common_value[..., (0, 1), :],
        intent.typed_common_value[..., (1, 0), :],
    )
    torch.testing.assert_close(
        relabeled_intent.typed_interval_residual_mass[..., (0, 1), :],
        intent.typed_interval_residual_mass[..., (1, 0), :],
    )
    torch.testing.assert_close(
        relabeled_intent.typed_interval_residual_value[..., (0, 1), :],
        intent.typed_interval_residual_value[..., (1, 0), :],
    )
    torch.testing.assert_close(
        relabeled_intent.policy_interval_context,
        intent.policy_interval_context,
    )

    coarse = top.coarse_action(intent.action_dock())
    relabeled_coarse = relabeled_top.coarse_action(relabeled_intent.action_dock())
    torch.testing.assert_close(relabeled_coarse.tokens, coarse.tokens)

    _, w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=False,
    )
    dynamics, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=w1,
        collect_diagnostics=False,
    )
    _, relabeled_w1, _ = relabeled_top.dynamics.forward_w1(
        facts=relabeled_facts,
        intent=relabeled_intent.world_dock(),
        action=relabeled_coarse,
        collect_diagnostics=False,
    )
    relabeled_dynamics, _ = relabeled_top.dynamics.forward_w2(
        facts=relabeled_facts,
        intent=relabeled_intent.world_dock(),
        action=relabeled_coarse,
        w1_state=relabeled_w1,
        collect_diagnostics=False,
    )
    for field in fields(FutureObjectDynamics):
        torch.testing.assert_close(
            getattr(relabeled_dynamics, field.name),
            getattr(dynamics, field.name),
            msg=field.name,
        )


def test_public_intent_match_cannot_train_optional_typed_relevance() -> None:
    torch.manual_seed(38)
    top = _object_top()
    facts, _ = top.grounder(_local_facts(cameras=2))
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    intent.public_interval_carrier.float().square().mean().backward()
    for parameter in top.intent.typed_relevance_queries.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert top.intent.typed_temperature_logit.grad is None


def test_typed_values_have_one_s_to_w_ingress() -> None:
    top = _object_top()
    assert not hasattr(top.coarse_action, "typed_router")
    assert not hasattr(top.dynamics, "typed_router")
    coarse_source = inspect.getsource(type(top.coarse_action).forward)
    assert "typed_" not in coarse_source
    assert "facts." not in coarse_source
    source = inspect.getsource(type(top.dynamics)._base)
    for forbidden in ("facts.semantic", "facts.appearance", "facts.geometry"):
        assert forbidden not in source
    assert "intent.typed_common_value" in source
    assert "intent.typed_interval_residual_value" in source


def test_removed_coarse_typed_path_preserves_retained_initializer_rng() -> None:
    """A structural deletion must not silently re-seed all later modules."""

    hidden, action_dim, route_dim, heads = 32, 7, 8, 4
    torch.manual_seed(901)
    current = CoarseActionIntent(
        hidden=hidden,
        action_dim=action_dim,
        route_dim=route_dim,
        heads=heads,
    )
    current_next = torch.rand(8)

    torch.manual_seed(901)
    legacy_query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
    legacy_intent_read = _CrossRead(hidden, heads)
    legacy_object_read = _CrossRead(hidden, heads)
    legacy_history_read = _CrossRead(hidden, heads)
    legacy_typed_inputs = nn.ModuleList(
        nn.Linear(route_dim, hidden, bias=False) for _ in TYPED_INTENT_NAMES
    )
    legacy_typed_memory_norm = nn.ModuleList(
        nn.LayerNorm(hidden, elementwise_affine=False)
        for _ in TYPED_INTENT_NAMES
    )
    legacy_typed_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
    legacy_typed_reads = nn.ModuleList(
        nn.MultiheadAttention(
            hidden,
            heads,
            bias=False,
            dropout=0.0,
            batch_first=True,
        )
        for _ in TYPED_INTENT_NAMES
    )
    legacy_typed_router = nn.Linear(
        hidden, len(TYPED_INTENT_NAMES), bias=False
    )
    legacy_block = _SelfBlock(hidden, heads)
    legacy_action_head = nn.Linear(hidden, action_dim, bias=False)
    legacy_next = torch.rand(8)
    del (
        legacy_typed_inputs,
        legacy_typed_memory_norm,
        legacy_typed_query_norm,
        legacy_typed_reads,
        legacy_typed_router,
    )

    torch.testing.assert_close(current.query, legacy_query, atol=0.0, rtol=0.0)
    for current_reader, legacy_reader in (
        (current.intent_read, legacy_intent_read),
        (current.object_read, legacy_object_read),
        (current.history_read, legacy_history_read),
        (current.block, legacy_block),
        (current.action_head, legacy_action_head),
    ):
        for name, value in current_reader.state_dict().items():
            torch.testing.assert_close(
                value,
                legacy_reader.state_dict()[name],
                atol=0.0,
                rtol=0.0,
            )
    torch.testing.assert_close(current_next, legacy_next, atol=0.0, rtol=0.0)


def test_neutral_w_preserves_current_precision_and_temporal_without_w_interaction() -> None:
    torch.manual_seed(4)
    top = _object_top()
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    horizon, basis, hidden = 24, 2, 32
    p1_fact = torch.randn(1, horizon, basis, hidden)
    deployment = DeploymentTopCache(
        intent=context.intent,
        predicted_dynamics=FutureObjectDynamics.neutral(context.facts),
    )
    action_query = torch.randn(1, horizon, basis, hidden)
    compiled, _ = top.compile_policy(
        deployment,
        p1_fact=p1_fact,
        p1_precision_innovation=p1_fact,
        action_query=action_query,
    )
    neutral_other_query, _ = top.compile_policy(
        deployment,
        p1_fact=p1_fact,
        p1_precision_innovation=p1_fact,
        action_query=torch.randn(1, horizon, basis, hidden),
    )
    assert torch.count_nonzero(compiled.effect) == 0
    assert not hasattr(compiled.plan, "factual")
    assert torch.count_nonzero(compiled.plan.precision) > 0
    assert torch.count_nonzero(compiled.plan.temporal) > 0
    assert torch.count_nonzero(compiled.plan.state_change) > 0
    # V120 keeps noisy-action modulation in its typed temporal lane.  The
    # protected factual consequence remains available independently.
    assert not torch.equal(compiled.plan.temporal, neutral_other_query.plan.temporal)
    identity_only_intent = replace(
        context.intent,
        policy_interval_context=context.intent.interval_queries
        + 1000.0
        * torch.randn_like(
            context.intent.interval_queries
        ),
    )
    identity_only_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=identity_only_intent,
            predicted_dynamics=deployment.predicted_dynamics,
        ),
        p1_fact=p1_fact,
        p1_precision_innovation=p1_fact,
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
        p1_fact,
        atol=0.0,
        rtol=0.0,
    )


def test_p2_equal_candidate_evidence_has_no_fixed_half_null_prior() -> None:
    batch, intervals, objects, content, hidden = 1, 4, 4, 6, 8
    scalar = torch.ones(batch, intervals, objects, 1)
    dynamics = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=torch.zeros(batch, intervals, objects, content),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=torch.zeros_like(scalar),
        persistence=torch.zeros_like(scalar),
        uncertainty=torch.zeros_like(scalar),
        reliability=torch.zeros_like(scalar),
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=scalar,
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    intent = PolicyIntentDock(
        common_key=torch.zeros(batch, hidden),
        interval_residual_key=torch.zeros(batch, intervals, hidden),
        typed_common_object_value=torch.zeros(batch, objects, 3, 4),
        typed_interval_residual_value=torch.zeros(
            batch, intervals, objects, 3, 4
        ),
        temporal_control=torch.zeros(batch, 24, hidden),
        state_change_evidence=torch.zeros(batch, hidden),
    )
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=content, route_dim=4
    )
    value, metrics = reader(
        torch.zeros(batch, 24, 2, hidden),
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    assert torch.count_nonzero(value) == 0
    assert float(metrics["object_p2_residual_null_mass"]) < 0.5
    assert float(metrics["object_p2_protected_common_rms"]) == 0.0


def test_p2_protected_common_survives_when_interval_residual_is_exactly_zero() -> None:
    """The common W field is mandatory evidence, not another null candidate."""

    batch, intervals, objects, content, hidden = 1, 4, 3, 6, 8
    common_semantic = torch.randn(batch, objects, content)
    common_transport = 0.1 * torch.randn(batch, objects, 2)
    common_status = -0.2 * torch.ones(batch, objects, 1)
    semantic = common_semantic[:, None].expand(-1, intervals, -1, -1).clone()
    transport = common_transport[:, None].expand(-1, intervals, -1, -1).clone()
    status = common_status[:, None].expand(-1, intervals, -1, -1).clone()
    dynamics = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=status,
        persistence=status,
        uncertainty=torch.zeros(batch, intervals, objects, 1),
        reliability=torch.zeros(batch, intervals, objects, 1),
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=torch.ones(batch, intervals, objects, 1),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    intent = PolicyIntentDock(
        common_key=torch.zeros(batch, hidden),
        interval_residual_key=torch.zeros(batch, intervals, hidden),
        typed_common_object_value=torch.zeros(batch, objects, 3, 4),
        typed_interval_residual_value=torch.zeros(
            batch, intervals, objects, 3, 4
        ),
        temporal_control=torch.zeros(batch, 24, hidden),
        state_change_evidence=torch.zeros(batch, hidden),
    )
    reader = ObjectFutureEffectReader(
        hidden=hidden,
        content_dim=content,
        route_dim=4,
    )
    with torch.no_grad():
        reader.semantic_value.weight.fill_(0.125)
        reader.transport_value.weight.fill_(0.125)
        reader.status_value.weight.fill_(0.125)
    value, metrics = reader(
        torch.zeros(batch, 24, 2, hidden),
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    dynamics.validate_effect_decomposition()
    assert torch.count_nonzero(dynamics.semantic_interval_residual) == 0
    assert torch.count_nonzero(dynamics.transport_interval_residual) == 0
    assert torch.count_nonzero(dynamics.visibility_interval_residual) == 0
    assert torch.count_nonzero(value) > 0
    assert float(metrics["object_p2_protected_common_rms"]) > 0.0
    assert float(metrics["object_p2_optional_residual_rms"]) == 0.0


def test_p2_effect_gradient_reaches_w_heads_and_s_typed_queries() -> None:
    """Prove the online S -> W -> P2 path is one differentiable chain."""

    torch.manual_seed(406)
    top = _object_top()
    with torch.no_grad():
        top.dynamics.delta_head.weight.fill_(0.02)
        top.dynamics.transport_head.weight.fill_(0.02)
        top.dynamics.visibility_head.weight.fill_(0.02)
        top.dynamics.persistence_head.weight.fill_(0.02)
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    effect, _ = top.effect_reader(
        torch.randn(1, 24, 2, 32),
        context.predicted_dynamics,
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    effect.float().square().mean().backward()

    def nonzero(parameter: nn.Parameter) -> bool:
        return parameter.grad is not None and bool(
            torch.count_nonzero(parameter.grad.detach())
        )

    assert nonzero(top.effect_reader.semantic_value.weight)
    assert nonzero(top.dynamics.delta_head.weight)
    assert nonzero(top.intent.typed_relevance_queries[0].weight)


def test_p2_disappearance_status_is_not_self_masked_by_future_visibility() -> None:
    batch, intervals, objects, content, hidden = 1, 4, 4, 6, 8
    scalar = torch.zeros(batch, intervals, objects, 1)
    dynamics = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=torch.zeros(batch, intervals, objects, content),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=-torch.ones_like(scalar),
        persistence=-torch.ones_like(scalar),
        uncertainty=torch.zeros_like(scalar),
        reliability=torch.zeros_like(scalar),
        current_selector_validity=torch.ones(batch, objects, 1),
        future_selector_validity=torch.zeros_like(scalar),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    intent = PolicyIntentDock(
        common_key=torch.zeros(batch, hidden),
        interval_residual_key=torch.zeros(batch, intervals, hidden),
        typed_common_object_value=torch.zeros(batch, objects, 3, 4),
        typed_interval_residual_value=torch.zeros(
            batch, intervals, objects, 3, 4
        ),
        temporal_control=torch.zeros(batch, 24, hidden),
        state_change_evidence=torch.zeros(batch, hidden),
    )
    reader = ObjectFutureEffectReader(
        hidden=hidden,
        content_dim=content,
        route_dim=4,
    )
    with torch.no_grad():
        reader.status_value.weight.fill_(0.25)
    value, metrics = reader(
        torch.zeros(batch, 24, 2, hidden),
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    assert torch.count_nonzero(value) > 0
    assert float(metrics["object_p2_status_common_selected_value_rms"]) > 0.0
    # Future selector validity is diagnostic-only at this boundary.  All
    # optional residuals retain the same current physical support.
    assert float(metrics["object_p2_status_residual_null_mass"]) < 0.5
    assert float(metrics["object_p2_semantic_residual_null_mass"]) < 0.5
    assert float(metrics["object_p2_geometry_residual_null_mass"]) < 0.5


def test_p2_invalid_objects_have_exactly_zero_common_and_residual_support() -> None:
    batch, intervals, objects, content, hidden = 1, 4, 4, 6, 8
    scalar = torch.zeros(batch, intervals, objects, 1)
    validity = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
    baseline = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=torch.zeros(batch, intervals, objects, content),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 3),
        visibility=scalar,
        persistence=scalar,
        uncertainty=scalar,
        reliability=scalar,
        current_selector_validity=validity,
        future_selector_validity=torch.ones_like(scalar),
        object_coordinates=torch.zeros(batch, objects, 2),
    )
    semantic = baseline.semantic_delta.clone()
    transport = baseline.transport_mean.clone()
    visibility = baseline.visibility.clone()
    persistence = baseline.persistence.clone()
    semantic[:, :, 1:] = 1000.0
    transport[:, :, 1:] = 1000.0
    visibility[:, :, 1:] = -1.0
    persistence[:, :, 1:] = -1.0
    invalid_changed = replace(
        baseline,
        successor_content=baseline.current_reference[:, None] + semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        visibility=visibility,
        persistence=persistence,
    )
    intent = PolicyIntentDock(
        common_key=torch.zeros(batch, hidden),
        interval_residual_key=torch.zeros(batch, intervals, hidden),
        typed_common_object_value=torch.zeros(batch, objects, 3, 4),
        typed_interval_residual_value=torch.zeros(
            batch, intervals, objects, 3, 4
        ),
        temporal_control=torch.zeros(batch, 24, hidden),
        state_change_evidence=torch.zeros(batch, hidden),
    )
    reader = ObjectFutureEffectReader(
        hidden=hidden,
        content_dim=content,
        route_dim=4,
    )
    query = torch.zeros(batch, 24, 2, hidden)
    baseline_value, _ = reader(
        query, baseline, intent, collect_diagnostics=False
    )
    changed_value, _ = reader(
        query, invalid_changed, intent, collect_diagnostics=False
    )
    torch.testing.assert_close(changed_value, baseline_value, atol=0.0, rtol=0.0)


def test_p2_complementary_fusion_has_variance_preserving_base_and_exact_zero() -> None:
    torch.manual_seed(405)
    reader = ObjectFutureEffectReader(hidden=16, content_dim=8, route_dim=4)
    selected = torch.randn(2, 3, 2, 3, 16)
    with torch.no_grad():
        reader.type_contrast_scale.zero_()
    fused, base, contrast, residual = reader._fuse_complementary_values(selected)
    expected_base = selected.float().sum(dim=-2) / (3.0**0.5)
    torch.testing.assert_close(base, expected_base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(fused.float(), base, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(residual) == 0
    torch.testing.assert_close(
        contrast.sum(dim=-2),
        torch.zeros_like(base),
        atol=2.0e-6,
        rtol=0.0,
    )

    zeros = torch.zeros_like(selected)
    fused_zero, base_zero, contrast_zero, residual_zero = (
        reader._fuse_complementary_values(zeros)
    )
    assert torch.equal(fused_zero, zeros[..., 0, :])
    assert torch.equal(base_zero, zeros[..., 0, :].float())
    assert torch.equal(contrast_zero, zeros.float())
    assert torch.equal(residual_zero, zeros[..., 0, :].float())

    shared = torch.randn(2, 3, 2, 1, 16)
    identical = shared.expand(-1, -1, -1, 3, -1).clone()
    with torch.no_grad():
        reader.type_contrast_scale.fill_(1.0e-4)
    fused_same, base_same, _, residual_same = reader._fuse_complementary_values(
        identical
    )
    torch.testing.assert_close(
        base_same,
        (3.0**0.5) * shared[..., 0, :].float(),
        atol=2.0e-6,
        rtol=1.0e-6,
    )
    assert torch.count_nonzero(residual_same) == 0
    torch.testing.assert_close(fused_same.float(), base_same, atol=0.0, rtol=0.0)


def test_p2_complementary_fusion_starts_near_uniform_without_type_selector() -> None:
    torch.manual_seed(406)
    reader = ObjectFutureEffectReader(hidden=32, content_dim=8, route_dim=4)
    assert not hasattr(reader, "type_query")
    selected = torch.randn(2, 24, 2, 3, 32, requires_grad=True)
    fused, base, _, residual = reader._fuse_complementary_values(selected)
    residual_ratio = residual.square().mean().sqrt() / base.square().mean().sqrt()
    assert float(residual_ratio.detach()) < 1.0e-3
    fused.float().sum().backward()
    assert selected.grad is not None
    for type_index in range(3):
        assert torch.count_nonzero(selected.grad[..., type_index, :]) > 0


def test_p2_complementary_fusion_owns_cpu_bf16_boundary() -> None:
    torch.manual_seed(407)
    reader = ObjectFutureEffectReader(hidden=16, content_dim=8, route_dim=4)
    selected = torch.randn(1, 4, 2, 3, 16, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fused, base, contrast, residual = reader._fuse_complementary_values(selected)
    assert fused.dtype == torch.bfloat16
    assert base.dtype == torch.float32
    assert contrast.dtype == torch.float32
    assert residual.dtype == torch.float32
    assert torch.isfinite(fused).all()


def test_p2_semantic_intervention_cannot_change_geometry_or_status_selector() -> None:
    torch.manual_seed(403)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    query = torch.randn(1, 24, 2, 32)
    baseline = context.predicted_dynamics
    _, metrics = top.effect_reader(
        query,
        baseline,
        context.intent.policy_dock(),
        collect_diagnostics=True,
    )
    changed = replace(
        baseline,
        semantic_delta=baseline.semantic_delta
        + torch.randn_like(baseline.semantic_delta),
    )
    _, changed_metrics = top.effect_reader(
        query,
        changed,
        context.intent.policy_dock(),
        collect_diagnostics=True,
    )
    assert not torch.equal(
        metrics["object_p2_semantic_residual_null_mass"],
        changed_metrics["object_p2_semantic_residual_null_mass"],
    )
    torch.testing.assert_close(
        metrics["object_p2_geometry_residual_null_mass"],
        changed_metrics["object_p2_geometry_residual_null_mass"],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        metrics["object_p2_status_residual_null_mass"],
        changed_metrics["object_p2_status_residual_null_mass"],
        atol=0.0,
        rtol=0.0,
    )


def test_p3_optional_lanes_are_source_exclusive_and_zero_preserving() -> None:
    torch.manual_seed(402)
    batch, horizon, basis, hidden = 1, 24, 4, 16
    compiler = ObjectPolicyPlanCompiler(
        hidden=hidden,
        horizon=horizon,
        basis=basis,
    )
    action_query = torch.randn(batch, horizon, basis, hidden)
    common_fact = torch.randn(batch, horizon, 1, hidden).expand(
        -1, -1, basis, -1
    )
    effect = torch.randn_like(common_fact)
    consequence = ObjectConsequenceState(
        factual_base=common_fact,
        effect=effect,
        interaction=torch.zeros_like(effect),
        protected_consequence=common_fact + effect,
    )
    zero_intent = PolicyIntentDock(
        common_key=torch.zeros(batch, hidden),
        interval_residual_key=torch.zeros(batch, 4, hidden),
        typed_common_object_value=torch.zeros(batch, 4, 3, 4),
        typed_interval_residual_value=torch.zeros(batch, 4, 4, 3, 4),
        temporal_control=torch.zeros(batch, horizon, hidden),
        state_change_evidence=torch.zeros(batch, hidden),
    )
    bank, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=common_fact,
        consequence=consequence,
        intent=zero_intent,
        action_query=action_query,
    )
    assert not hasattr(bank, "factual")
    # Precision reads the cached high-resolution P1 innovation directly; it
    # must not disappear merely because the observation is shared across
    # action bases.  Temporal requires both S and consequence.
    assert torch.count_nonzero(bank.precision) > 0
    assert torch.count_nonzero(bank.temporal) == 0
    assert torch.count_nonzero(bank.state_change) == 0
    assert torch.count_nonzero(bank.effect) > 0

    zero_precision, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=torch.zeros_like(common_fact),
        consequence=consequence,
        intent=zero_intent,
        action_query=action_query,
    )
    assert torch.count_nonzero(zero_precision.precision) == 0
    torch.testing.assert_close(
        zero_precision.temporal,
        bank.temporal,
        atol=0.0,
        rtol=0.0,
    )

    active_intent = replace(
        zero_intent,
        temporal_control=torch.randn(batch, horizon, hidden),
    )
    active, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=common_fact,
        consequence=consequence,
        intent=active_intent,
        action_query=action_query,
    )
    assert torch.count_nonzero(active.temporal) > 0
    no_consequence, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=common_fact,
        consequence=replace(
            consequence,
            protected_consequence=torch.zeros_like(
                consequence.protected_consequence
            ),
        ),
        intent=active_intent,
        action_query=action_query,
    )
    assert torch.count_nonzero(no_consequence.temporal) == 0
    changed_consequence = replace(
        consequence,
        protected_consequence=consequence.protected_consequence
        + torch.randn_like(consequence.protected_consequence),
    )
    changed, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=common_fact,
        consequence=changed_consequence,
        intent=active_intent,
        action_query=action_query,
    )
    torch.testing.assert_close(changed.precision, active.precision, atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed.effect, active.effect, atol=0.0, rtol=0.0)
    assert not torch.equal(changed.temporal, active.temporal)

    changed_precision, _ = compiler(
        p1_fact=common_fact,
        p1_precision_innovation=common_fact + torch.randn_like(common_fact),
        consequence=consequence,
        intent=active_intent,
        action_query=action_query,
    )
    assert not torch.equal(changed_precision.precision, active.precision)
    torch.testing.assert_close(
        changed_precision.temporal,
        active.temporal,
        atol=0.0,
        rtol=0.0,
    )


def test_supervised_successor_innovation_crosses_w_to_p2_without_current_bypass() -> None:
    torch.manual_seed(28)
    top = _object_top()
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    horizon, basis, hidden = 24, 2, 32
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
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    changed_effect, _ = top.effect_reader(
        action_query,
        changed,
        context.intent.policy_dock(),
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
        "role_table",
    }
    assert {field.name for field in fields(DeploymentTopCache)} == {
        "intent",
        "predicted_dynamics",
    }
