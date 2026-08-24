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
    TypedP2EffectRead,
    ZeroPreservingObjectConsequence,
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
    ObservableIntentStateSupervisor,
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
    CompletedP1PolicyState,
    FutureObjectDynamics,
    LocalFactSet,
    PolicyIntentDock,
)
from clearvla.mainline.training.losses import flow_geometry_terms, future_dynamics_terms
from clearvla.mainline.v120_core.profile import build_v120_visual_config
from clearvla.mainline.v120_core.refinement import NestedLowRankContractionBank
from clearvla.mainline.v120_core.role_delta_attnres import RoleDeltaAttnRes


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


def test_lane_local_bottom_selector_preserves_parent_initialization_stream() -> None:
    """Removing serialized source rows must not rerandomize the live bottom."""

    torch.manual_seed(37001)
    parent = RoleDeltaAttnRes(
        16,
        8,
        max_sources=20,
        max_value_rms=0.35,
        normalization_floor=0.25,
    )
    parent_next = nn.Linear(16, 16, bias=False)

    torch.manual_seed(37001)
    current = RoleDeltaAttnRes(
        16,
        8,
        max_sources=4,
        initialization_source_rows=20,
        max_value_rms=0.35,
        normalization_floor=0.25,
    )
    current_next = nn.Linear(16, 16, bias=False)

    torch.testing.assert_close(
        current.source_key,
        parent.source_key[:4],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        current.query_proj.weight,
        parent.query_proj.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        current.key_proj.weight,
        parent.key_proj.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        current_next.weight,
        parent_next.weight,
        atol=0.0,
        rtol=0.0,
    )


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


def test_grounder_reconstruction_uses_canonical_k_content_plus_public_position() -> None:
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
    canonical_part = facts.public_content[:, None, None, None] + torch.einsum(
        "bkcyx,bkd->bcyxd",
        reconstruction_owner,
        facts.content_innovation.float(),
    )
    public_position = facts.reconstructed_dino.float() - canonical_part
    torch.testing.assert_close(
        public_position.mean(dim=(1, 2, 3)),
        torch.zeros_like(public_position.mean(dim=(1, 2, 3))),
        atol=2.0e-6,
        rtol=0.0,
    )
    # Both added capacities are exact zero at initialization, preserving the
    # former forward distribution while remaining in the dense objective.
    assert float(metrics["object_grounding_public_position_rms"]) == 0.0
    assert float(metrics["object_grounding_canonical_slot_residual_rms"]) == 0.0
    assert float(metrics["object_grounding_reconstruction_object_mass_mean"]) > 0.99
    assert float(metrics["object_grounding_reconstruction_active_fraction"]) == 1.0


def test_semantic_and_appearance_only_correct_conditional_k_mass() -> None:
    grounder = DenseObjectGrounder(
        hidden=4,
        content_dim=4,
        route_dim=2,
        objects=2,
        iterations=1,
    )
    slots = torch.tensor([[[2.0, -1.0, -1.0, 0.0], [-1.0, 2.0, -1.0, 0.0]]])
    content = torch.zeros(1, 1, 4)
    views = torch.zeros(1, 1, 3, 4)
    validity = torch.ones(1, 1, 1)
    prior = torch.ones(1, 1, 1)
    owner, typed_owner, mass, null, _ = grounder._competition(
        slots, content, views, validity, prior
    )
    assert tuple(typed_owner.shape) == (1, 1, 3, 3)
    torch.testing.assert_close(
        mass.sum(dim=-1) + null,
        torch.ones_like(null),
    )

    for type_index in (0, 1):
        changed = views.clone()
        changed[..., type_index, :] = 4.0 * slots[:, :1]
        changed_owner, changed_typed, changed_mass, changed_null, _ = (
            grounder._competition(slots, content, changed, validity, prior)
        )
        assert changed_owner[0, 0, 0] > owner[0, 0, 0]
        torch.testing.assert_close(
            changed_owner[..., :2].sum(dim=-1),
            owner[..., :2].sum(dim=-1),
            atol=1.0e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            changed_owner[..., 2], owner[..., 2], atol=1.0e-7, rtol=0.0
        )
        torch.testing.assert_close(
            changed_typed[..., type_index, :2].sum(dim=-1),
            owner[..., :2].sum(dim=-1),
            atol=1.0e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            changed_typed[..., type_index, 2],
            owner[..., 2],
            atol=1.0e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            changed_mass.sum(dim=-1) + changed_null,
            torch.ones_like(changed_null),
        )


def test_geometry_cannot_vote_on_physical_k_identity() -> None:
    torch.manual_seed(203)
    grounder = DenseObjectGrounder(
        hidden=4,
        content_dim=4,
        route_dim=2,
        objects=2,
        iterations=1,
    )
    slots = torch.randn(1, 2, 4)
    content = torch.randn(1, 3, 4)
    views = torch.randn(1, 3, 3, 4)
    validity = torch.ones(1, 3, 1)
    prior = torch.ones(1, 3, 1)
    owner, typed_owner, mass, null, read = grounder._competition(
        slots, content, views, validity, prior
    )

    changed = views.clone()
    changed[..., 2, :] = 100.0 * torch.randn_like(changed[..., 2, :])
    changed_owner, changed_typed, changed_mass, changed_null, changed_read = (
        grounder._competition(slots, content, changed, validity, prior)
    )
    torch.testing.assert_close(changed_owner, owner, atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed_mass, mass, atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed_null, null, atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed_read, read, atol=0.0, rtol=0.0)
    assert not torch.equal(changed_typed[..., 2, :], typed_owner[..., 2, :])


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
    batch, intervals = 1, 4
    current_state = torch.randn(batch, 7)
    future_state = torch.randn(batch, 48, 7)
    result = top.intent_supervisor(
        intent=intent,
        current_state=current_state,
        future_state=future_state,
    )
    interval_mean = torch.stack(
        [future_state[:, lower - 1 : upper].mean(dim=1) for lower, upper in INTERVAL_BOUNDS],
        dim=1,
    )
    expected_state = interval_mean - torch.cat(
        (current_state[:, None], interval_mean[:, :-1]), dim=1
    )
    torch.testing.assert_close(result.state_target, expected_state)
    assert tuple(result.state_prediction.shape) == (batch, intervals, 7)
    assert set(dict(top.intent_supervisor.named_parameters())) == {"state_head.weight"}
    parameters = inspect.signature(ObservableIntentStateSupervisor.forward).parameters
    assert set(parameters) == {"self", "intent", "current_state", "future_state"}
    shift = torch.randn(batch, 1, 7)
    shifted = top.intent_supervisor(
        intent=intent,
        current_state=current_state + shift[:, 0],
        future_state=future_state + shift,
    )
    torch.testing.assert_close(shifted.state_target, result.state_target)


def test_intent_supervisor_has_no_w_or_teacher_target_boundary() -> None:
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
    result = top.intent_supervisor(
        intent=intent,
        current_state=torch.randn(1, 7),
        future_state=torch.randn(1, 48, 7),
    )
    assert result.loss.ndim == 0
    assert not hasattr(result, "typed_loss")
    assert not hasattr(result, "semantic_prediction")


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
        camera_chart_availability=torch.ones_like(
            facts.camera_chart_availability
        ),
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
        candidate_coordinate=coordinate,
        current_camera_coordinate=current,
        null_probability=torch.zeros(1, 1, 1, 1),
        null_camera_measure=torch.full((1, 1, 2, 1), 0.5),
    )
    torch.testing.assert_close(transport, torch.zeros_like(transport))
    torch.testing.assert_close(covariance, torch.zeros_like(covariance))

    permutation = torch.tensor((1, 0))
    permuted_transport, permuted_covariance = (
        ObjectFutureTeacher._relative_geometry_moments(
            candidate_posterior=posterior[:, :, :, permutation],
            candidate_coordinate=coordinate[permutation],
            current_camera_coordinate=current[:, :, permutation],
            null_probability=torch.zeros(1, 1, 1, 1),
            null_camera_measure=torch.full((1, 1, 2, 1), 0.5)[
                :, :, permutation
            ],
        )
    )
    torch.testing.assert_close(permuted_transport, transport[:, :, :, permutation])
    torch.testing.assert_close(permuted_covariance, covariance[:, :, :, permutation])


def test_teacher_camera_moments_include_identity_null_mass() -> None:
    coordinate = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])
    posterior = torch.zeros(1, 1, 1, 1, 1, 2)
    posterior[..., 1] = 0.5
    current = torch.zeros(1, 1, 1, 2)
    transport, covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=posterior,
        candidate_coordinate=coordinate,
        current_camera_coordinate=current,
        null_probability=torch.full((1, 1, 1, 1), 0.5),
        null_camera_measure=torch.ones(1, 1, 1, 1),
    )
    scaled_transport, scaled_covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=0.1 * posterior,
        candidate_coordinate=coordinate,
        current_camera_coordinate=current,
        null_probability=torch.full((1, 1, 1, 1), 0.95),
        null_camera_measure=torch.ones(1, 1, 1, 1),
    )
    torch.testing.assert_close(transport, torch.tensor([[[[[0.5, 0.0]]]]]))
    torch.testing.assert_close(
        covariance,
        torch.tensor([[[[[0.25, 0.0, 0.0]]]]]),
    )
    torch.testing.assert_close(
        scaled_transport,
        torch.tensor([[[[[0.05, 0.0]]]]]),
    )
    torch.testing.assert_close(
        scaled_covariance,
        torch.tensor([[[[[0.0475, 0.0, 0.0]]]]]),
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


def test_grounder_canonical_slot_residual_is_exported_and_supervised() -> None:
    """The decoded K residual is the canonical fact, not a private recon head."""

    torch.manual_seed(2201)
    local = _local_facts(cameras=2, content=8, route=4, hidden=16)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    baseline, _ = grounder(local)
    assert torch.count_nonzero(grounder.decode_content_residual.weight) == 0
    with torch.no_grad():
        grounder.decode_content_residual.weight.copy_(
            0.05 * torch.randn_like(grounder.decode_content_residual.weight)
        )
    canonical, metrics = grounder(local)
    assert not torch.equal(canonical.content, baseline.content)
    assert not torch.equal(canonical.content_innovation, baseline.content_innovation)
    assert float(metrics["object_grounding_canonical_slot_residual_rms"]) > 0.0
    canonical.reconstruction_error.backward()
    gradient = grounder.decode_content_residual.weight.grad
    assert gradient is not None and torch.count_nonzero(gradient) > 0
    position_gradient = grounder.decode_public_position.weight.grad
    assert position_gradient is not None and torch.count_nonzero(position_gradient) > 0


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
        g3_public_scene_audit=(
            first.g3_public_scene_audit
            + 1000.0 * torch.randn_like(first.g3_public_scene_audit)
        ),
    )
    first_candidate = grounder._candidate_tokens(first)
    second_candidate = grounder._candidate_tokens(second)
    torch.testing.assert_close(first_candidate, second_candidate)
    names = {name for name, _ in grounder.named_parameters()}
    assert not any("public_address_key" in name for name in names)


def test_w_exports_no_unobserved_status_and_keeps_current_chart_authority() -> None:
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
        typed_interval_innovation=torch.zeros(1, 2, 4, 3, 16),
    )
    assert {field.name for field in fields(FutureObjectDynamics)} == {
        "current_reference",
        "successor_content",
        "semantic_delta",
        "transport_mean",
        "transport_covariance",
        "chart_availability",
        "camera_coordinates",
        "camera_chart_availability",
        "camera_weights",
    }
    assert not hasattr(dynamics, "visibility_head")
    assert not hasattr(dynamics, "persistence_head")
    assert not hasattr(field, "future_selector_validity")
    assert not hasattr(field, "visibility")
    assert not hasattr(field, "persistence")
    torch.testing.assert_close(
        field.chart_availability,
        facts.chart_availability,
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        field.camera_chart_availability,
        facts.camera_chart_availability,
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
        typed_interval_innovation=torch.zeros(1, 2, 4, 3, 16),
    )
    innovation = field.successor_content - field.current_reference[:, None]
    assert torch.count_nonzero(innovation) == 0
    innovation.sum().backward()
    assert facts.content.grad is None or torch.count_nonzero(facts.content.grad) == 0


def test_w_appearance_conditions_semantic_zero_preservingly_with_gradients() -> None:
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
    semantic = torch.randn(1, 2, 4, 16, requires_grad=True)
    appearance = torch.randn_like(semantic, requires_grad=True)
    conditioned, modulation, denominator = dynamics._appearance_condition_semantic(
        semantic,
        appearance,
    )
    assert torch.count_nonzero(modulation) > 0
    assert float(denominator.detach().amin()) >= 0.25
    semantic_gradient, appearance_gradient = torch.autograd.grad(
        conditioned.float().square().sum(),
        (semantic, appearance),
    )
    assert torch.count_nonzero(semantic_gradient) > 0
    assert torch.count_nonzero(appearance_gradient) > 0

    appearance_zero, modulation_zero, _ = dynamics._appearance_condition_semantic(
        semantic.detach(),
        torch.zeros_like(appearance),
    )
    torch.testing.assert_close(appearance_zero, semantic.detach(), atol=0.0, rtol=0.0)
    assert torch.count_nonzero(modulation_zero) == 0
    semantic_zero, semantic_zero_modulation, _ = (
        dynamics._appearance_condition_semantic(
            torch.zeros_like(semantic),
            appearance.detach(),
        )
    )
    assert torch.count_nonzero(semantic_zero) == 0
    assert torch.count_nonzero(semantic_zero_modulation) == 0

    # Appearance is no longer decoded as an unsupervised status value.  It can
    # modulate the semantic owner, while geometry remains the sole camera-value
    # owner and retains the real C axis.
    with torch.no_grad():
        dynamics.delta_head.weight.fill_(0.05)
        dynamics.transport_head.weight.fill_(0.05)
        dynamics.covariance_head.weight.fill_(0.05)
    hidden = torch.zeros(1, 2, 4, 16)
    common = torch.zeros(1, 4, 3, 16)
    interval = torch.zeros(1, 2, 4, 3, 16)
    interval[..., 0, :] = semantic.detach()
    semantic_only = dynamics._field(
        facts=facts,
        hidden=hidden,
        typed_common=common,
        typed_interval_innovation=interval,
    )
    interval[..., 1, :] = appearance.detach()
    with_appearance = dynamics._field(
        facts=facts,
        hidden=hidden,
        typed_common=common,
        typed_interval_innovation=interval,
    )
    assert not torch.equal(with_appearance.semantic_delta, semantic_only.semantic_delta)
    torch.testing.assert_close(
        with_appearance.transport_mean,
        semantic_only.transport_mean,
    )
    interval.zero_()
    interval[..., 2, :] = torch.randn_like(interval[..., 2, :])
    geometry = dynamics._field(
        facts=facts,
        hidden=hidden,
        typed_common=common,
        typed_interval_innovation=interval,
    )
    assert torch.count_nonzero(geometry.semantic_delta) == 0
    assert geometry.transport_mean.shape == (1, 2, 4, 2, 2)
    assert geometry.transport_covariance.shape == (1, 2, 4, 2, 3)


def test_w_camera_geometry_head_is_equivariant_local_and_psd() -> None:
    """W decodes each real camera with one shared, zero-preserving head."""

    torch.manual_seed(2911)
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
        dynamics.camera_geometry_condition.weight.zero_()
        dynamics.camera_geometry_condition.weight[0, 16] = 1.0
        dynamics.transport_head.weight.zero_()
        dynamics.transport_head.weight[0, 0] = 0.5
        dynamics.covariance_head.weight.normal_(mean=0.0, std=0.05)
    typed_common = torch.zeros(1, 4, 3, 16)
    typed_common[..., 2, :] = 1.0
    typed_residual = torch.zeros(1, 2, 4, 3, 16)
    hidden = torch.zeros(1, 2, 4, 16)
    coordinates = torch.zeros_like(facts.camera_coordinates)
    first_facts = replace(facts, camera_coordinates=coordinates)
    second_coordinates = coordinates.clone()
    second_coordinates[:, :, 0, 0] = 1.0
    second_facts = replace(facts, camera_coordinates=second_coordinates)
    first = dynamics._field(
        facts=first_facts,
        hidden=hidden,
        typed_common=typed_common,
        typed_interval_innovation=typed_residual,
    )
    second = dynamics._field(
        facts=second_facts,
        hidden=hidden,
        typed_common=typed_common,
        typed_interval_innovation=typed_residual,
    )
    assert not torch.equal(second.transport_mean[..., 0, :], first.transport_mean[..., 0, :])
    torch.testing.assert_close(
        second.transport_mean[..., 1, :],
        first.transport_mean[..., 1, :],
        atol=0.0,
        rtol=0.0,
    )

    camera_permutation = torch.tensor([1, 0])
    permuted_facts = replace(
        second_facts,
        camera_coordinates=second_facts.camera_coordinates[:, :, camera_permutation],
        camera_transport_prior=(
            second_facts.camera_transport_prior[:, :, camera_permutation]
        ),
        camera_support=second_facts.camera_support[:, :, camera_permutation],
        camera_chart_availability=(
            second_facts.camera_chart_availability[:, :, camera_permutation]
        ),
        camera_evidence_mass=(
            second_facts.camera_evidence_mass[:, :, camera_permutation]
        ),
    )
    permuted = dynamics._field(
        facts=permuted_facts,
        hidden=hidden,
        typed_common=typed_common,
        typed_interval_innovation=typed_residual,
    )
    torch.testing.assert_close(
        permuted.transport_mean,
        second.transport_mean[:, :, :, camera_permutation],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        permuted.transport_covariance,
        second.transport_covariance[:, :, :, camera_permutation],
        atol=0.0,
        rtol=0.0,
    )
    covariance = second.transport_covariance.float()
    floor = (2.0 / 7.0) ** 2
    assert bool((covariance[..., 0] >= floor).all())
    assert bool((covariance[..., 2] >= floor).all())
    assert bool((covariance[..., (0, 2)] <= 1.0).all())
    determinant = covariance[..., 0] * covariance[..., 2] - covariance[..., 1].square()
    assert bool((determinant >= -1.0e-7).all())


def test_w_camera_covariance_retains_fp32_psd_under_bf16_autocast() -> None:
    torch.manual_seed(2912)
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
    with torch.no_grad():
        dynamics.covariance_head.weight.normal_(mean=0.0, std=1.5)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        field = dynamics._field(
            facts=facts,
            hidden=torch.zeros(1, 4, 4, 16),
            typed_common=torch.randn(1, 4, 3, 16),
            typed_interval_innovation=torch.randn(1, 4, 4, 3, 16),
        )
    assert field.transport_covariance.dtype == torch.float32
    xx, xy, yy = field.transport_covariance.unbind(dim=-1)
    assert bool((xx * yy - xy.square() >= -1.0e-7).all())


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
    zero = torch.zeros(1, 2, 4, 3, 16)
    public_a = torch.randn(1, 2, 4, 16)
    public_b = torch.randn_like(public_a)
    zero_a = dynamics._field(
        facts=facts,
        hidden=public_a,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_innovation=zero,
    )
    zero_b = dynamics._field(
        facts=facts,
        hidden=public_b,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_innovation=zero,
    )
    for name in ("semantic_delta", "transport_mean"):
        torch.testing.assert_close(getattr(zero_a, name), getattr(zero_b, name))
        assert torch.count_nonzero(getattr(zero_a, name)) == 0

    typed = torch.randn_like(zero)
    typed_a = dynamics._field(
        facts=facts,
        hidden=public_a,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_innovation=typed,
    )
    typed_b = dynamics._field(
        facts=facts,
        hidden=public_b,
        typed_common=torch.zeros(1, 4, 3, 16),
        typed_interval_innovation=typed,
    )
    torch.testing.assert_close(typed_a.semantic_delta, typed_b.semantic_delta)
    torch.testing.assert_close(typed_a.transport_mean, typed_b.transport_mean)


def test_w_full_base_interaction_is_zero_preserving_and_condition_sensitive() -> None:
    torch.manual_seed(2921)
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=16, content_dim=8, route_dim=4, heads=4
    )
    with torch.no_grad():
        dynamics.typed_base_interaction.weight.copy_(torch.eye(16))
    typed = torch.randn(2, 4, 3, 16)
    base = torch.randn(2, 4, 16)
    first, first_interaction, first_denominator = dynamics._interact_with_base(
        typed, base
    )
    second, _, _ = dynamics._interact_with_base(typed, base.roll(1, dims=0))
    assert not torch.equal(first, second)
    assert torch.count_nonzero(first_interaction) > 0

    _, tiny_interaction, tiny_denominator = dynamics._interact_with_base(
        typed * 1.0e-6, base
    )
    assert float(
        (
            tiny_interaction.float().square().mean().sqrt()
            / first_interaction.float().square().mean().sqrt().clamp_min(1.0e-8)
        ).detach()
    ) < 1.0e-4
    assert float(first_denominator.detach().amin()) >= 0.25
    assert float(tiny_denominator.detach().amin()) >= 0.25

    zero = torch.zeros_like(typed)
    zero_output, zero_interaction, zero_denominator = dynamics._interact_with_base(
        zero, base
    )
    assert torch.equal(zero_output, zero)
    assert torch.equal(zero_interaction, zero)
    torch.testing.assert_close(
        zero_denominator, torch.full_like(zero_denominator, 0.25)
    )


def test_w_typed_base_interaction_is_deferred_to_each_completed_owner() -> None:
    torch.manual_seed(2922)
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
    with torch.no_grad():
        top.dynamics.typed_base_interaction.weight.zero_()
    base_zero = top.dynamics._base(
        facts,
        intent.world_dock(),
        coarse,
        collect_diagnostics=False,
    )[:3]
    with torch.no_grad():
        top.dynamics.typed_base_interaction.weight.copy_(torch.eye(32))
    base_active = top.dynamics._base(
        facts,
        intent.world_dock(),
        coarse,
        collect_diagnostics=False,
    )[:3]
    for before_owner, after_owner in zip(base_zero, base_active, strict=True):
        torch.testing.assert_close(
            before_owner,
            after_owner,
            atol=0.0,
            rtol=0.0,
        )

    _, working, w1_metrics = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=True,
    )
    assert float(w1_metrics["object_w_common_base_interaction_rms"]) > 0.0
    assert float(w1_metrics["object_w_interval_base_interaction_rms"]) > 0.0
    _, w2_metrics = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=working,
        collect_diagnostics=True,
    )
    assert float(w2_metrics["object_w2_common_processing_delta_rms"]) == 0.0


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
    _, working, w1_metrics = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=True,
    )
    assert not torch.equal(working.common_typed, common_input)
    assert not torch.equal(working.near_interval_innovation, residual_input[:, :2])
    assert float(w1_metrics["object_w1_common_processing_delta_rms"]) > 0.0
    completed, completed_metrics = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=working,
        collect_diagnostics=True,
    )
    completed.validate()
    assert float(completed_metrics["object_w2_common_processing_delta_rms"]) == 0.0

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
    assert torch.count_nonzero(zero_working.near_interval_innovation) == 0
    assert torch.count_nonzero(zero_working.far_interval_innovation) == 0
    zero_completed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=zero_typed.world_dock(),
        action=zero_coarse,
        w1_state=zero_working,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_completed.semantic_delta) == 0
    assert torch.count_nonzero(zero_completed.transport_mean) == 0


def test_w2_near_innovation_updates_far_without_rereading_common() -> None:
    """W2 reads near innovation while forwarding W1 common unchanged."""

    torch.manual_seed(294)
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
    quiet_field, working, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=False,
    )
    assert quiet_field is None
    assert not hasattr(working, "near_field")
    with torch.no_grad():
        top.dynamics.delta_head.weight.normal_(mean=0.0, std=0.05)
    near_innovation = (
        working.near_interval_innovation.detach().clone().requires_grad_(True)
    )
    captured: dict[str, torch.Tensor] = {}
    original_field = top.dynamics._field

    def capture_field(**kwargs):
        captured["typed_common"] = kwargs["typed_common"]
        captured["typed_interval_innovation"] = kwargs[
            "typed_interval_innovation"
        ]
        return original_field(**kwargs)

    with mock.patch.object(top.dynamics, "_field", side_effect=capture_field):
        completed, metrics = top.dynamics.forward_w2(
            facts=facts,
            intent=intent.world_dock(),
            action=coarse,
            w1_state=replace(working, near_interval_innovation=near_innovation),
            collect_diagnostics=True,
        )
    torch.testing.assert_close(
        captured["typed_common"],
        working.common_typed,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured["typed_interval_innovation"][:, :2],
        near_innovation,
        atol=0.0,
        rtol=0.0,
    )
    far_gradient = torch.autograd.grad(
        completed.semantic_delta[:, 2:].square().sum(),
        near_innovation,
        allow_unused=False,
    )[0]
    assert torch.count_nonzero(far_gradient.detach()) > 0
    assert float(metrics["object_w2_near_to_far_innovation_update_rms"]) > 0.0


def test_w2_preserves_near_rows_and_retains_same_direction_far_innovation() -> None:
    """Far cannot rewrite near, and its common-mode direction is not erased."""

    torch.manual_seed(2941)
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
    with torch.no_grad():
        top.dynamics.delta_head.weight.normal_(mean=0.0, std=0.05)
        top.dynamics.transport_head.weight.normal_(mean=0.0, std=0.05)
        top.dynamics.covariance_head.weight.normal_(mean=0.0, std=0.05)
        top.dynamics.typed_base_interaction.weight.normal_(mean=0.0, std=0.05)
    near_field, working, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        collect_diagnostics=True,
    )
    assert near_field is not None
    far_base = working.far_base.detach().clone().requires_grad_(True)
    far_typed = (
        working.far_interval_innovation.detach().clone().requires_grad_(True)
    )
    completed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=replace(
            working,
            far_base=far_base,
            far_interval_innovation=far_typed,
        ),
        collect_diagnostics=False,
    )
    for name in (
        "successor_content",
        "semantic_delta",
        "transport_mean",
        "transport_covariance",
    ):
        torch.testing.assert_close(
            getattr(completed, name)[:, :2],
            getattr(near_field, name),
            # The 2-row and 4-row shared Linear kernels may accumulate in a
            # different FP order; causal equality is proved exactly by JVP
            # below rather than by requiring bitwise GEMM identity.
            atol=2.0e-6,
            rtol=0.0,
            msg=name,
        )
    near_public = sum(
        getattr(completed, name)[:, :2].float().square().sum()
        for name in (
            "successor_content",
            "semantic_delta",
            "transport_mean",
            "transport_covariance",
        )
    )
    far_to_near = torch.autograd.grad(
        near_public,
        (far_base, far_typed),
        retain_graph=True,
        allow_unused=True,
    )
    for gradient in far_to_near:
        if gradient is not None:
            assert torch.count_nonzero(gradient.detach()) == 0

    same_direction = torch.randn_like(working.far_interval_innovation[:, :1]).expand_as(
        working.far_interval_innovation
    )
    shifted_field, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=intent.world_dock(),
        action=coarse,
        w1_state=replace(
            working,
            far_interval_innovation=(
                working.far_interval_innovation + 0.2 * same_direction
            ),
        ),
        collect_diagnostics=False,
    )
    assert not torch.equal(
        shifted_field.semantic_delta[:, 2:],
        completed.semantic_delta[:, 2:],
    )
    assert not torch.equal(
        shifted_field.semantic_common,
        completed.semantic_common,
    )
    assert not hasattr(top.dynamics, "_complete_with_far_owned_zero_mean")
    assert not hasattr(top.dynamics, "_maybe_apply_far_owned_gauge")


def test_w_block_common_and_interval_owners_have_zero_cross_jacobian() -> None:
    """Shared W parameters must not turn owner separation into shared state."""

    torch.manual_seed(295)
    compiler = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=8,
        route_dim=4,
        heads=4,
    ).eval()
    common = torch.randn(1, 3, 3, 16, requires_grad=True)
    residual = torch.randn(1, 4, 3, 3, 16, requires_grad=True)
    completed_common, completed_residual = (
        compiler._run_separated_owned_typed_block(
            compiler.w1,
            common,
            residual,
            causal_interval=True,
        )
    )
    common_from_residual = torch.autograd.grad(
        completed_common.square().sum(),
        residual,
        retain_graph=True,
        allow_unused=True,
    )[0]
    residual_from_common = torch.autograd.grad(
        completed_residual.square().sum(),
        common,
        allow_unused=True,
    )[0]
    if common_from_residual is not None:
        assert torch.count_nonzero(common_from_residual.detach()) == 0
    if residual_from_common is not None:
        assert torch.count_nonzero(residual_from_common.detach()) == 0


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
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.ones(batch, objects, cameras, 1),
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

    unsupported = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=torch.zeros_like(current_support),
    )
    torch.testing.assert_close(unsupported["future_dynamics"], torch.zeros(()))


def test_training_support_uses_physical_camera_validity_not_assignment_mass() -> None:
    source = inspect.getsource(ObjectIntentDynamicsTop.build_training_targets)
    assert source.count("context.facts.camera_chart_availability") == 1
    assert "camera_evidence_mass" not in source


def test_future_transport_interval_diagnostic_uses_camera_support() -> None:
    """An invalid camera must not contaminate the per-interval audit row."""

    batch, intervals, objects, cameras, content = 1, 4, 1, 2, 4
    current = torch.zeros(batch, objects, content)
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.full((batch, objects, cameras, 1), 0.5),
    )
    invalid_camera_error = torch.zeros_like(target.transport_mean)
    invalid_camera_error[:, 0, :, 1] = 1.0
    prediction = replace(target, transport_mean=invalid_camera_error)
    camera_support = torch.tensor([[[[1.0], [0.0]]]])
    masked_terms = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=camera_support,
        collect_diagnostics=True,
    )
    torch.testing.assert_close(masked_terms["future_transport"], torch.zeros(()))
    for index in range(intervals):
        torch.testing.assert_close(
            masked_terms[f"future_interval_{index}_transport"],
            torch.zeros(()),
        )

    supported_terms = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=torch.ones_like(camera_support),
        collect_diagnostics=True,
    )
    assert supported_terms["future_interval_0_transport"] > 0


def test_future_interval_transition_penalizes_temporal_collapse_not_common_offset() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    current = torch.zeros(batch, objects, content)
    interval = torch.arange(intervals, dtype=torch.float32)[None, :, None, None]
    semantic = interval.expand(batch, intervals, objects, content).clone()
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=semantic,
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.ones(batch, objects, cameras, 1),
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
    target = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None].expand(-1, intervals, -1, -1),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.ones(batch, objects, cameras, 1),
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
    high_target, high_metrics = teacher(
        facts=facts,
        future_supports=high,
        future_offsets=offsets,
    )
    low_target, low_metrics = teacher(
        facts=facts,
        future_supports=opposed,
        future_offsets=offsets,
    )
    assert high_metrics["object_teacher_reliability"] > low_metrics[
        "object_teacher_reliability"
    ]
    assert high_metrics["object_teacher_dustbin_probability"] < low_metrics[
        "object_teacher_dustbin_probability"
    ]
    torch.testing.assert_close(
        high_target.semantic_delta,
        high_target.successor_content - high_target.current_reference[:, None],
    )
    torch.testing.assert_close(
        low_target.semantic_delta,
        low_target.successor_content - low_target.current_reference[:, None],
    )
    for target in (high_target, low_target):
        assert not hasattr(target, "uncertainty")
        assert not hasattr(target, "reliability")
        torch.testing.assert_close(
            target.chart_availability,
            facts.chart_availability.detach(),
            atol=0.0,
            rtol=0.0,
        )
    for metrics in (high_metrics, low_metrics):
        assert "object_teacher_uncertainty" in metrics
        assert "object_teacher_reliability" in metrics
        assert not any("visibility" in name or "persistence" in name for name in metrics)
        assert "object_teacher_future_selector_validity" not in metrics


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
    target, metrics = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=torch.tensor((6, 12, 24, 40)),
    )

    full_delta = (
        support_value[None, None, None].float()
        - target.current_reference[:, None].float()
    )
    observed_delta = (
        target.successor_content.float() - target.current_reference[:, None].float()
    )
    matched_mass = (
        observed_delta * full_delta
    ).sum(dim=-1, keepdim=True) / full_delta.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    torch.testing.assert_close(
        observed_delta,
        matched_mass * full_delta,
        atol=5.0e-5,
        rtol=1.0e-2,
    )
    torch.testing.assert_close(
        target.semantic_delta.float(),
        target.successor_content.float() - target.current_reference[:, None].float(),
    )
    assert torch.isfinite(target.transport_mean).all()
    assert torch.isfinite(target.transport_covariance).all()
    teacher_covariance = target.transport_covariance.float()
    assert bool(
        (
            teacher_covariance[..., 0] * teacher_covariance[..., 2]
            - teacher_covariance[..., 1].square()
            >= -1.0e-7
        ).all()
    )
    assert metrics["object_teacher_reliability"] < 1.0
    torch.testing.assert_close(
        target.chart_availability,
        facts.chart_availability.detach(),
        atol=0.0,
        rtol=0.0,
    )


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
        camera_chart_availability=(
            facts.camera_chart_availability[:, :, camera_permutation]
        ),
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
        if field.name in {
            "transport_mean",
            "transport_covariance",
            "camera_coordinates",
            "camera_chart_availability",
            "camera_weights",
        }:
            camera_axis = 3 if field.name.startswith("transport_") else 2
            expected = expected.index_select(camera_axis, camera_permutation)
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
    policy_residual = torch.randn_like(p1_fact)
    p1_state = CompletedP1PolicyState(
        factual_base=p1_fact,
        policy_query_residual=policy_residual,
    )
    compiled, _ = top.compile_policy(
        DeploymentTopCache(intent=intent, predicted_dynamics=dynamics),
        p1_state=p1_state,
        action_query=action_query,
    )
    relabeled_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=relabeled_intent,
            predicted_dynamics=relabeled_dynamics,
        ),
        p1_state=p1_state,
        action_query=action_query,
    )
    assert torch.allclose(
        relabeled_compiled.consequence.effect_by_type,
        compiled.consequence.effect_by_type,
        atol=2e-5,
        rtol=2e-5,
    )
    for name in (
        "factual_base",
        "effect_by_type",
        "interaction_by_type",
        "protected_consequence",
    ):
        assert torch.allclose(
            getattr(relabeled_compiled.consequence, name),
            getattr(compiled.consequence, name),
            atol=2e-5,
            rtol=2e-5,
        ), name
    for name in (
        "protected_base",
        "precision",
        "effect_semantic",
        "effect_geometry",
        "temporal_semantic",
        "temporal_geometry",
        "state_change",
    ):
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

    invalid_facts = replace(
        facts,
        chart_availability=torch.zeros_like(facts.chart_availability),
    )
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
    assert float(w_metrics["object_w_typed_interval_innovation_state_rms"]) == 0.0


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
        differential_common_value,
        differential_mass,
        differential_value,
        _,
        _,
        differential_common_score,
        raw_differential_score,
        _,
        _,
        typed_temperature,
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

    # The nonlinear differential signal generally has a nonzero interval
    # mean.  Schema37 transfers that mean into the typed common owner before
    # applying one shared, one-sided scale to the centered residual.  The mean
    # is therefore retained exactly rather than being discarded as a gauge.
    raw_common_signal = torch.tanh(
        differential_common_score * typed_temperature[None, None]
    )
    raw_residual_signal = torch.tanh(
        raw_differential_score * typed_temperature[None, None, None]
    )
    residual_mean = raw_residual_signal.mean(dim=1)
    assert torch.count_nonzero(residual_mean) > 0
    centered_residual = raw_residual_signal - residual_mean[:, None]
    scaled_residual = centered_residual / centered_residual.abs().amax(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)
    typed_route = torch.stack(
        (facts.semantic, facts.appearance, facts.geometry),
        dim=2,
    )
    common_validity = facts.chart_availability.float()[:, :, None, :]
    expected_common_value = (
        raw_common_signal + residual_mean
    )[..., None] * common_validity * typed_route
    expected_residual_value = (
        scaled_residual[..., None]
        * common_validity[:, None]
        * typed_route[:, None]
    )
    torch.testing.assert_close(
        differential_common_value,
        expected_common_value,
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        differential_value,
        expected_residual_value,
        atol=2.0e-6,
        rtol=0.0,
    )


def test_s_weak_k_mass_is_not_normalized_into_a_strong_policy_value() -> None:
    torch.manual_seed(3711)
    top = _object_top().eval()
    facts, _ = top.grounder(_local_facts(cameras=2))
    interval = torch.randn(1, 4, 32)
    weak_facts = replace(
        facts,
        chart_availability=torch.full_like(facts.chart_availability, 1.0e-4),
    )
    double_facts = replace(
        facts,
        chart_availability=torch.full_like(facts.chart_availability, 2.0e-4),
    )
    weak = top.intent._typed_relevance(
        interval_condition_innovation=interval,
        facts=weak_facts,
    )
    doubled = top.intent._typed_relevance(
        interval_condition_innovation=interval,
        facts=double_facts,
    )
    # Both K-mass denominators stay on their clamp-at-one branch.  Doubling
    # weak observable support therefore doubles, rather than renormalizes away,
    # each common/residual policy component.
    for weak_component, doubled_component in zip(
        weak[4:6],
        doubled[4:6],
        strict=True,
    ):
        assert torch.count_nonzero(weak_component) > 0
        torch.testing.assert_close(
            2.0 * weak_component,
            doubled_component,
            atol=2.0e-6,
            rtol=2.0e-5,
        )


def test_s_final_common_residual_owners_stay_canonical_under_bf16() -> None:
    torch.manual_seed(3712)
    top = _object_top().eval()
    local_facts = _local_facts(cameras=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        context, metrics = top.build_online_context(
            local_facts=local_facts,
            goal_tokens=torch.randn(1, 6, 12),
            goal_mask=torch.ones(1, 6, dtype=torch.bool),
            state_history=torch.randn(1, 3, 7),
            state=torch.randn(1, 7),
            executed_history=torch.randn(1, 8, 7),
            collect_diagnostics=True,
        )
    intent = context.intent
    assert intent.public_common_condition.dtype == torch.float32
    assert intent.public_interval_residual_condition.dtype == torch.float32
    torch.testing.assert_close(
        intent.public_common_condition[:, None]
        + intent.public_interval_residual_condition,
        intent.interval_condition_innovation.float(),
        atol=2.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        intent.public_interval_residual_condition.mean(dim=1),
        torch.zeros_like(intent.public_common_condition),
        atol=2.0e-7,
        rtol=0.0,
    )
    assert intent.typed_common_policy_components.dtype == torch.float32
    assert intent.typed_interval_residual_policy_components.dtype == torch.float32
    torch.testing.assert_close(
        intent.typed_interval_residual_policy_components.mean(dim=1),
        torch.zeros_like(intent.typed_common_policy_components),
        atol=2.0e-7,
        rtol=0.0,
    )
    assert float(metrics["object_intent_public_reconstruction_max_abs"]) <= 2.0e-7
    assert float(metrics["object_intent_public_residual_mean_max_abs"]) <= 2.0e-7
    assert float(metrics["object_intent_typed_policy_residual_mean_rms"]) <= 2.0e-7
    assert float(metrics["object_w_typed_interval_input_mean_rms"]) <= 2.0e-7


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
        intent.public_common_condition,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dock.interval_residual_key,
        intent.public_interval_residual_condition,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dock.common_key[:, None] + dock.interval_residual_key,
        intent.interval_condition_innovation,
        atol=2.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dock.interval_residual_key.mean(dim=1),
        torch.zeros_like(dock.common_key),
        atol=2.0e-7,
        rtol=0.0,
    )
    factual = intent.factual_dock()
    torch.testing.assert_close(
        factual.typed_interval_context,
        intent.typed_common_policy_components[:, None]
        + intent.typed_interval_residual_policy_components,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        factual.goal_interval_context,
        intent.goal_interval_context,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        factual.history_interval_context,
        intent.history_interval_context,
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


def test_s_typed_owner_relabeling_is_equivariant_through_coarse_action() -> None:
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


def test_neutral_w_preserves_precision_but_zeros_effect_and_temporal() -> None:
    torch.manual_seed(4)
    top = _object_top()
    facts, _ = top.grounder(_local_facts())
    intent, _ = top.intent(
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    horizon, basis, hidden = 24, 2, 32
    p1_fact = torch.randn(1, horizon, basis, hidden)
    deployment = DeploymentTopCache(
        intent=intent,
        predicted_dynamics=FutureObjectDynamics.neutral(facts),
    )
    action_query = torch.randn(1, horizon, basis, hidden)
    policy_residual = torch.randn_like(p1_fact)
    p1_state = CompletedP1PolicyState(
        factual_base=p1_fact,
        policy_query_residual=policy_residual,
    )
    compiled, _ = top.compile_policy(
        deployment,
        p1_state=p1_state,
        action_query=action_query,
    )
    other_action_query = torch.randn(1, horizon, basis, hidden)
    neutral_other_query, _ = top.compile_policy(
        deployment,
        p1_state=CompletedP1PolicyState(
            factual_base=p1_fact,
            policy_query_residual=policy_residual,
        ),
        action_query=other_action_query,
    )
    assert torch.count_nonzero(compiled.consequence.effect_by_type) == 0
    assert not hasattr(compiled.plan, "factual")
    assert torch.count_nonzero(compiled.plan.precision) > 0
    for name in (
        "effect_semantic",
        "effect_geometry",
        "temporal_semantic",
        "temporal_geometry",
    ):
        assert torch.count_nonzero(getattr(compiled.plan, name)) == 0
    assert compiled.plan.source_names == (
        "p3_precision",
        "p3_effect_semantic",
        "p3_effect_geometry",
        "p3_temporal_semantic",
        "p3_temporal_geometry",
        "p3_state_change",
    )
    assert torch.count_nonzero(compiled.plan.state_change) > 0
    torch.testing.assert_close(
        compiled.plan.temporal,
        neutral_other_query.plan.temporal,
        atol=0.0,
        rtol=0.0,
    )
    identity_only_intent = replace(
        intent,
        policy_interval_context=intent.interval_queries
        + 1000.0
        * torch.randn_like(
            intent.interval_queries
        ),
    )
    identity_only_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=identity_only_intent,
            predicted_dynamics=deployment.predicted_dynamics,
        ),
        p1_state=p1_state,
        action_query=action_query,
    )
    # Cumulative/identity-bearing S queries are addresses internal to S.  P2
    # consumes only the observed interval innovation, so changing identity
    # alone cannot recreate a fixed temporal route prior.
    torch.testing.assert_close(
        compiled.consequence.effect_by_type,
        identity_only_compiled.consequence.effect_by_type,
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
    batch, intervals, objects, cameras, content, hidden = 1, 4, 4, 3, 6, 8
    dynamics = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=torch.zeros(batch, intervals, objects, content),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.ones(batch, objects, cameras, 1),
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
    assert torch.count_nonzero(value.effect_by_type) == 0
    semantic_null = torch.tensor(1.0 / float(intervals * objects + 1))
    geometry_null = torch.tensor(1.0 / float(intervals * objects * cameras + 1))
    aggregate_null = 0.5 * (semantic_null + geometry_null)
    torch.testing.assert_close(
        metrics["object_p2_semantic_residual_null_mass"],
        semantic_null,
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        metrics["object_p2_geometry_residual_null_mass"],
        geometry_null,
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        metrics["object_p2_residual_null_mass"],
        aggregate_null,
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        metrics["object_p2_type_interval_null_mass"],
        aggregate_null,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert float(metrics["object_p2_protected_common_rms"]) == 0.0


def test_p2_protected_common_survives_when_interval_residual_is_exactly_zero() -> None:
    """The common W field is mandatory evidence, not another null candidate."""

    batch, intervals, objects, content, hidden = 1, 4, 3, 6, 8
    common_semantic = torch.randn(batch, objects, content)
    common_transport = 0.1 * torch.randn(batch, objects, 1, 2)
    semantic = common_semantic[:, None].expand(-1, intervals, -1, -1).clone()
    transport = common_transport[:, None].expand(
        -1, intervals, -1, -1, -1
    ).clone()
    dynamics = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        transport_covariance=torch.zeros(batch, intervals, objects, 1, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, 1, 2),
        camera_chart_availability=torch.ones(batch, objects, 1, 1),
        camera_weights=torch.ones(batch, objects, 1, 1),
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
    value, metrics = reader(
        torch.zeros(batch, 24, 2, hidden),
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    dynamics.validate_effect_decomposition()
    assert torch.count_nonzero(dynamics.semantic_interval_residual) == 0
    assert torch.count_nonzero(dynamics.transport_interval_residual) == 0
    assert torch.count_nonzero(value.effect_by_type) > 0
    assert float(metrics["object_p2_protected_common_rms"]) > 0.0
    assert float(metrics["object_p2_optional_residual_rms"]) == 0.0


def test_p2_effect_gradient_reaches_w_heads_and_s_typed_queries() -> None:
    """Prove the online S -> W -> P2 path is one differentiable chain."""

    torch.manual_seed(406)
    top = _object_top()
    with torch.no_grad():
        top.dynamics.delta_head.weight.fill_(0.02)
        top.dynamics.transport_head.weight.fill_(0.02)
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
    effect.physical_sum.float().square().mean().backward()

    def nonzero(parameter: nn.Parameter) -> bool:
        return parameter.grad is not None and bool(
            torch.count_nonzero(parameter.grad.detach())
        )

    assert nonzero(top.effect_reader.semantic_value.weight)
    assert nonzero(top.dynamics.delta_head.weight)
    assert nonzero(top.intent.typed_relevance_queries[0].weight)


def test_appearance_receives_ordinary_semantic_future_gradient() -> None:
    """Appearance conditions the supervised successor instead of a dead status."""

    torch.manual_seed(4061)
    top = _object_top()
    with torch.no_grad():
        top.dynamics.delta_head.weight.fill_(0.02)
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    target = FutureObjectDynamics.neutral(context.facts)
    terms = future_dynamics_terms(
        context.predicted_dynamics,
        target,
        current_loss_support=context.facts.camera_chart_availability,
    )
    terms["future_semantic_delta"].backward()
    parameter = top.dynamics.object_appearance.weight
    assert parameter.grad is not None
    assert torch.count_nonzero(parameter.grad.detach()) > 0


def test_p2_and_future_loss_have_no_unobserved_status_branch() -> None:
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    target = FutureObjectDynamics.neutral(context.facts)
    terms = future_dynamics_terms(
        context.predicted_dynamics,
        target,
        current_loss_support=context.facts.camera_chart_availability,
    )
    assert not any("visibility" in name or "persistence" in name for name in terms)
    assert "future_selector_validity" not in inspect.getsource(
        ObjectFutureEffectReader.forward
    )


def test_p2_real_camera_mixture_is_camera_permutation_invariant() -> None:
    torch.manual_seed(4041)
    batch, intervals, objects, cameras, content, hidden = 2, 4, 3, 3, 6, 8
    dynamics = FutureObjectDynamics(
        current_reference=torch.randn(batch, objects, content),
        successor_content=torch.randn(batch, intervals, objects, content),
        semantic_delta=torch.randn(batch, intervals, objects, content),
        transport_mean=0.1 * torch.randn(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, cameras, 3),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.tanh(
            torch.randn(batch, objects, cameras, 2)
        ),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
        camera_weights=torch.rand(batch, objects, cameras, 1),
    )
    intent = PolicyIntentDock(
        common_key=torch.randn(batch, hidden),
        interval_residual_key=torch.randn(batch, intervals, hidden),
        typed_common_object_value=torch.randn(batch, objects, 3, 4),
        typed_interval_residual_value=torch.randn(
            batch, intervals, objects, 3, 4
        ),
        temporal_control=torch.randn(batch, 24, hidden),
        state_change_evidence=torch.randn(batch, hidden),
    )
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=content, route_dim=4
    ).eval()
    query = torch.randn(batch, 24, 2, hidden)
    baseline, baseline_metrics = reader(
        query, dynamics, intent, collect_diagnostics=True
    )
    camera_permutation = torch.tensor([2, 0, 1])
    relabeled = replace(
        dynamics,
        camera_coordinates=dynamics.camera_coordinates[:, :, camera_permutation],
        camera_chart_availability=(
            dynamics.camera_chart_availability[:, :, camera_permutation]
        ),
        camera_weights=dynamics.camera_weights[:, :, camera_permutation],
        transport_mean=dynamics.transport_mean[:, :, :, camera_permutation],
        transport_covariance=(
            dynamics.transport_covariance[:, :, :, camera_permutation]
        ),
    )
    permuted, permuted_metrics = reader(
        query, relabeled, intent, collect_diagnostics=True
    )
    torch.testing.assert_close(
        permuted.effect_by_type,
        baseline.effect_by_type,
        atol=2.0e-6,
        rtol=1.0e-6,
    )
    for name in (
        "object_p2_coordinate_score_abs",
        "object_p2_coordinate_score_max_abs",
        "object_p2_camera_mixture_effective_count",
        "object_p2_camera_support_fraction",
        "object_p2_camera_coordinate_variation",
    ):
        torch.testing.assert_close(
            permuted_metrics[name], baseline_metrics[name], atol=2.0e-6, rtol=1.0e-6
        )

    # Breaking the paired camera relabeling changes the joint KxC geometry
    # hypotheses.  Geometry must respond while semantic remains independent of
    # camera order, proving transport was not averaged before selection.
    unpaired = replace(
        dynamics,
        transport_mean=dynamics.transport_mean[:, :, :, camera_permutation],
    )
    unpaired_read, unpaired_metrics = reader(
        query,
        unpaired,
        intent,
        collect_diagnostics=True,
    )
    for suffix in (
        "common_selected_value_rms",
        "residual_selected_value_rms",
        "common_posterior_entropy",
        "common_posterior_max",
        "residual_null_mass",
        "interval_posterior_entropy",
        "interval_posterior_max",
        "within_interval_object_posterior_entropy",
        "within_interval_object_posterior_max",
    ):
        torch.testing.assert_close(
            unpaired_metrics[f"object_p2_semantic_{suffix}"],
            baseline_metrics[f"object_p2_semantic_{suffix}"],
            atol=0.0,
            rtol=0.0,
        )
    assert not torch.equal(unpaired_read.geometry, baseline.geometry)

    no_camera = replace(
        dynamics,
        camera_weights=torch.zeros_like(dynamics.camera_weights),
        camera_chart_availability=torch.zeros_like(
            dynamics.camera_chart_availability
        ),
    )
    _, no_camera_metrics = reader(
        query, no_camera, intent, collect_diagnostics=True
    )
    assert float(no_camera_metrics["object_p2_coordinate_score_abs"]) == 0.0
    assert float(no_camera_metrics["object_p2_coordinate_score_max_abs"]) == 0.0
    assert float(no_camera_metrics["object_p2_camera_mixture_effective_count"]) == 0.0
    assert float(no_camera_metrics["object_p2_camera_support_fraction"]) == 0.0
    assert float(no_camera_metrics["object_p2_camera_coordinate_variation"]) == 0.0


def test_p2_invalid_objects_have_exactly_zero_common_and_residual_support() -> None:
    batch, intervals, objects, content, hidden = 1, 4, 4, 6, 8
    validity = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])
    baseline = FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=torch.zeros(batch, intervals, objects, content),
        semantic_delta=torch.zeros(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, 1, 2),
        transport_covariance=torch.zeros(batch, intervals, objects, 1, 3),
        chart_availability=validity,
        camera_coordinates=torch.zeros(batch, objects, 1, 2),
        camera_chart_availability=torch.ones(batch, objects, 1, 1),
        camera_weights=torch.ones(batch, objects, 1, 1),
    )
    semantic = baseline.semantic_delta.clone()
    transport = baseline.transport_mean.clone()
    semantic[:, :, 1:] = 1000.0
    transport[:, :, 1:] = 1000.0
    invalid_changed = replace(
        baseline,
        successor_content=baseline.current_reference[:, None] + semantic,
        semantic_delta=semantic,
        transport_mean=transport,
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
    torch.testing.assert_close(
        changed_value.effect_by_type,
        baseline_value.effect_by_type,
        atol=0.0,
        rtol=0.0,
    )


def test_p2_complementary_fusion_preserves_single_owner_and_exact_zero() -> None:
    torch.manual_seed(405)
    reader = ObjectFutureEffectReader(hidden=16, content_dim=8, route_dim=4)
    selected = torch.randn(2, 3, 2, 2, 16)
    read, raw_sum, shared_scale = reader._fuse_complementary_values(selected)
    expected = selected * shared_scale[..., None, :].to(dtype=selected.dtype)
    torch.testing.assert_close(read.effect_by_type, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(raw_sum, selected.sum(dim=-2), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        read.physical_sum,
        raw_sum * shared_scale.to(dtype=raw_sum.dtype),
        atol=2.0e-6,
        rtol=0.0,
    )

    zeros = torch.zeros_like(selected)
    read_zero, raw_zero, zero_scale = (
        reader._fuse_complementary_values(zeros)
    )
    assert torch.equal(read_zero.effect_by_type, zeros)
    assert torch.equal(raw_zero, zeros[..., 0, :])
    torch.testing.assert_close(zero_scale, torch.ones_like(zero_scale))

    semantic_only = selected.clone()
    semantic_only[..., 1, :].zero_()
    semantic_read, _, _ = (
        reader._fuse_complementary_values(semantic_only)
    )
    torch.testing.assert_close(
        semantic_read.physical_sum,
        semantic_read.semantic,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(semantic_read.geometry) == 0


def test_p2_complementary_fusion_starts_near_uniform_without_type_selector() -> None:
    torch.manual_seed(406)
    reader = ObjectFutureEffectReader(hidden=32, content_dim=8, route_dim=4)
    assert not hasattr(reader, "type_query")
    assert not hasattr(reader, "type_contrast_down")
    selected = torch.randn(2, 24, 2, 2, 32, requires_grad=True)
    read, _, _ = reader._fuse_complementary_values(selected)
    read.physical_sum.float().sum().backward()
    assert selected.grad is not None
    for type_index in range(2):
        assert torch.count_nonzero(selected.grad[..., type_index, :]) > 0


def test_p2_complementary_fusion_owns_cpu_bf16_boundary() -> None:
    torch.manual_seed(407)
    reader = ObjectFutureEffectReader(hidden=16, content_dim=8, route_dim=4)
    selected = torch.randn(1, 4, 2, 2, 16, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        read, raw_sum, shared_scale = reader._fuse_complementary_values(selected)
    assert read.effect_by_type.dtype == torch.bfloat16
    assert raw_sum.dtype == torch.bfloat16
    assert shared_scale.dtype == torch.float32
    assert torch.isfinite(read.effect_by_type).all()


def test_p2_complementary_value_contract_is_one_sided_and_exact_zero() -> None:
    reader = ObjectFutureEffectReader(hidden=16, content_dim=8, route_dim=4)
    limit = reader.COMPLEMENTARY_VALUE_MAX_RMS

    zero = torch.zeros(2, 3, 1, 2, 16)
    zero_read, _, zero_scale = reader._fuse_complementary_values(zero)
    assert torch.equal(zero_read.effect_by_type, zero)
    torch.testing.assert_close(zero_scale, torch.ones_like(zero_scale))

    tiny = torch.full((2, 3, 1, 2, 16), 1.0e-4)
    tiny_read, _, tiny_scale = reader._fuse_complementary_values(tiny)
    assert bool((tiny_scale <= 1.0).all())
    torch.testing.assert_close(
        tiny_read.effect_by_type,
        tiny,
        atol=1.0e-7,
        rtol=1.0e-6,
    )

    large = torch.full((2, 3, 1, 2, 16), 100.0)
    large_read, _, large_scale = reader._fuse_complementary_values(large)
    contracted_rms = large_read.physical_sum.float().square().mean(dim=-1).sqrt()
    assert bool((large_scale < 1.0).all())
    assert float(contracted_rms.amax()) <= limit * 1.0001


def test_p2_consumer_type_names_map_to_the_matching_s_owner() -> None:
    """P2 semantic/geometry must read the matching S semantic/geometry."""

    torch.manual_seed(410)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    intent = context.intent.policy_dock()
    captured: list[list[torch.Tensor]] = [[], []]
    handles = []
    for type_index, module in enumerate(top.effect_reader.typed_intent_key):
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, inputs, index=type_index: captured[index].append(
                    inputs[0].detach().clone()
                )
            )
        )
    try:
        top.effect_reader(
            torch.randn(1, 24, 2, 32),
            context.predicted_dynamics,
            intent,
            collect_diagnostics=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    for p2_index, s_index in enumerate((0, 2)):
        assert len(captured[p2_index]) == 2
        torch.testing.assert_close(
            captured[p2_index][0],
            intent.typed_common_object_value[..., s_index, :],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            captured[p2_index][1],
            intent.typed_interval_residual_value[..., s_index, :],
            atol=0.0,
            rtol=0.0,
        )


def test_p2_semantic_intervention_cannot_change_geometry_selector() -> None:
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
    # Semantic W evidence owns semantic interval/object selection only.  It
    # cannot move geometry's interval or within-interval object posterior.
    assert not torch.equal(
        metrics["object_p2_semantic_interval_w_score_abs"],
        changed_metrics["object_p2_semantic_interval_w_score_abs"],
    )
    assert not torch.equal(
        metrics["object_p2_semantic_residual_null_mass"],
        changed_metrics["object_p2_semantic_residual_null_mass"],
    )
    assert not torch.equal(
        metrics["object_p2_semantic_residual_selected_value_rms"],
        changed_metrics["object_p2_semantic_residual_selected_value_rms"],
    )
    for suffix in (
        "interval_w_score_abs",
        "residual_null_mass",
        "interval_posterior_entropy",
        "interval_posterior_max",
        "within_interval_object_posterior_entropy",
        "within_interval_object_posterior_max",
    ):
        torch.testing.assert_close(
            metrics[f"object_p2_geometry_{suffix}"],
            changed_metrics[f"object_p2_geometry_{suffix}"],
            atol=0.0,
            rtol=0.0,
        )


def test_p2_matched_interval_routes_read_typed_s_and_w_object_evidence() -> None:
    """Each active owner adds its own typed-S and W temporal evidence."""

    torch.manual_seed(408)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    query = torch.randn(1, 24, 2, 32)
    dynamics = context.predicted_dynamics
    intent = context.intent.policy_dock()
    baseline, baseline_metrics = top.effect_reader(
        query,
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    semantic_delta = dynamics.semantic_delta + torch.randn_like(
        dynamics.semantic_delta
    )
    changed_dynamics = replace(
        dynamics,
        successor_content=dynamics.current_reference[:, None] + semantic_delta,
        semantic_delta=semantic_delta,
        transport_mean=dynamics.transport_mean
        + 0.2 * torch.randn_like(dynamics.transport_mean),
    )
    changed_intent = replace(
        intent,
        typed_interval_residual_value=(
            intent.typed_interval_residual_value
            + torch.randn_like(intent.typed_interval_residual_value)
        ),
    )
    changed, changed_metrics = top.effect_reader(
        query,
        changed_dynamics,
        changed_intent,
        collect_diagnostics=True,
    )
    assert not torch.equal(changed.effect_by_type, baseline.effect_by_type)
    for name in (
        "object_p2_type_interval_typed_score_abs",
        "object_p2_type_interval_w_score_abs",
        "object_p2_type_interval_score_abs",
        "object_p2_type_interval_null_mass",
    ):
        assert not torch.equal(changed_metrics[name], baseline_metrics[name])
    # Distinct typed/W evidence is now allowed—and expected—to produce
    # distinct semantic and geometry time posteriors.
    assert float(
        changed_metrics["object_p2_type_interval_disagreement_max_abs"]
    ) > 0.0


def test_p2_public_interval_prior_responds_to_public_s_interval_keys() -> None:
    """The shared public prior remains live for both matched type routes."""

    torch.manual_seed(409)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    query = torch.randn(1, 24, 2, 32)
    dynamics = context.predicted_dynamics
    intent = context.intent.policy_dock()
    _, baseline_metrics = top.effect_reader(
        query,
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    changed_intent = replace(
        intent,
        interval_residual_key=intent.interval_residual_key.roll(1, dims=1),
    )
    _, changed_metrics = top.effect_reader(
        query,
        dynamics,
        changed_intent,
        collect_diagnostics=True,
    )
    baseline_mass = torch.stack(
        [
            baseline_metrics[f"object_p2_residual_interval_{index}_mass"]
            for index in range(4)
        ]
    )
    changed_mass = torch.stack(
        [
            changed_metrics[f"object_p2_residual_interval_{index}_mass"]
            for index in range(4)
        ]
    )
    assert not torch.allclose(changed_mass, baseline_mass, atol=1.0e-7, rtol=0.0)
    for type_name in ("semantic", "geometry"):
        baseline_typed_mass = torch.stack(
            [
                baseline_metrics[
                    f"object_p2_{type_name}_residual_interval_{index}_mass"
                ]
                for index in range(4)
            ]
        )
        changed_typed_mass = torch.stack(
            [
                changed_metrics[
                    f"object_p2_{type_name}_residual_interval_{index}_mass"
                ]
                for index in range(4)
            ]
        )
        assert not torch.allclose(
            changed_typed_mass,
            baseline_typed_mass,
            atol=1.0e-7,
            rtol=0.0,
        )


def test_p2_residual_retention_metrics_are_support_aware_and_complementary() -> None:
    torch.manual_seed(411)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 8, 7),
    )
    query = torch.randn(1, 24, 2, 32)
    dynamics = context.predicted_dynamics
    interval_scale = torch.tensor(
        [-1.0, -0.25, 0.25, 1.0],
        dtype=dynamics.semantic_delta.dtype,
    ).view(1, 4, 1, 1)
    semantic_delta = dynamics.semantic_delta + interval_scale * torch.randn_like(
        dynamics.semantic_delta[:, :1]
    )
    dynamics = replace(
        dynamics,
        semantic_delta=semantic_delta,
        successor_content=dynamics.current_reference[:, None] + semantic_delta,
    )
    _, metrics = top.effect_reader(
        query,
        dynamics,
        context.intent.policy_dock(),
        collect_diagnostics=True,
    )
    retained = metrics["object_p2_residual_retained_rms_ratio"]
    cancelled = metrics["object_p2_residual_cancelled_rms_fraction"]
    support = metrics["object_p2_residual_cancellation_support_fraction"]
    assert 0.0 <= float(retained) <= 1.0
    assert 0.0 <= float(cancelled) <= 1.0
    assert 0.0 < float(support) <= 1.0
    torch.testing.assert_close(
        retained + cancelled,
        torch.ones_like(retained),
        atol=1.0e-6,
        rtol=0.0,
    )

    _, neutral_metrics = top.effect_reader(
        query,
        FutureObjectDynamics.neutral(context.facts),
        context.intent.policy_dock(),
        collect_diagnostics=True,
    )
    assert float(
        neutral_metrics["object_p2_residual_cancellation_support_fraction"]
    ) == 0.0
    assert float(neutral_metrics["object_p2_residual_retained_rms_ratio"]) == 0.0
    assert float(
        neutral_metrics["object_p2_residual_cancelled_rms_fraction"]
    ) == 0.0


def test_consequence_preserves_typed_effect_and_uses_one_physical_sum() -> None:
    torch.manual_seed(401)
    batch, horizon, basis, hidden = 1, 24, 4, 16
    factual = torch.randn(batch, horizon, basis, hidden)
    effect_by_type = torch.randn(batch, horizon, basis, 2, hidden)
    module = ZeroPreservingObjectConsequence(hidden)
    state, _ = module(
        factual_base=factual,
        effect=TypedP2EffectRead(effect_by_type=effect_by_type),
    )
    state.validate()
    torch.testing.assert_close(
        state.effect,
        effect_by_type.sum(dim=-2),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        state.interaction,
        state.interaction_by_type.sum(dim=-2),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        state.protected_consequence,
        factual + state.effect + state.interaction,
        atol=0.0,
        rtol=0.0,
    )

    zero = torch.zeros_like(effect_by_type)
    neutral, _ = module(
        factual_base=factual,
        effect=TypedP2EffectRead(effect_by_type=zero),
    )
    assert torch.count_nonzero(neutral.effect_by_type) == 0
    assert torch.count_nonzero(neutral.interaction_by_type) == 0
    torch.testing.assert_close(
        neutral.protected_consequence,
        factual,
        atol=0.0,
        rtol=0.0,
    )
    try:
        module(factual_base=factual, effect=effect_by_type)  # type: ignore[arg-type]
    except TypeError as error:
        assert "TypedP2EffectRead" in str(error)
    else:
        raise AssertionError("Schema37 consequence must reject an untyped effect")


def test_p3_six_optional_lanes_are_source_exclusive_and_zero_preserving() -> None:
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
    effect_by_type = torch.randn(batch, horizon, basis, 2, hidden)
    interaction_by_type = torch.zeros_like(effect_by_type)
    consequence = ObjectConsequenceState(
        factual_base=common_fact,
        effect_by_type=effect_by_type,
        interaction_by_type=interaction_by_type,
        protected_consequence=common_fact + effect_by_type.sum(dim=-2),
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
        p1_factual_detail=common_fact,
        consequence=consequence,
        intent=zero_intent,
        action_query=action_query,
    )
    assert not hasattr(bank, "factual")
    assert bank.source_names == (
        "p3_precision",
        "p3_effect_semantic",
        "p3_effect_geometry",
        "p3_temporal_semantic",
        "p3_temporal_geometry",
        "p3_state_change",
    )
    role = bank.as_policy_role_bank(source_depth=7)
    assert role.values.shape == (batch, 6, horizon, basis, hidden)
    assert role.source_names == bank.source_names
    assert role.source_depths == (7,) * 6
    for index, name in enumerate(
        (
            "precision",
            "effect_semantic",
            "effect_geometry",
            "temporal_semantic",
            "temporal_geometry",
            "state_change",
        )
    ):
        torch.testing.assert_close(
            role.values[:, index],
            getattr(bank, name),
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        bank.effect,
        bank.effect_semantic + bank.effect_geometry,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        bank.temporal,
        bank.temporal_semantic + bank.temporal_geometry,
        atol=0.0,
        rtol=0.0,
    )
    # Precision reads the cached high-resolution P1 innovation directly; it
    # must not disappear merely because the observation is shared across
    # action bases. Temporal requires both S and W effect.
    assert torch.count_nonzero(bank.precision) > 0
    assert torch.count_nonzero(bank.temporal_semantic) == 0
    assert torch.count_nonzero(bank.temporal_geometry) == 0
    assert torch.count_nonzero(bank.state_change) == 0
    assert torch.count_nonzero(bank.effect_semantic) > 0
    assert torch.count_nonzero(bank.effect_geometry) > 0

    zero_precision, _ = compiler(
        p1_factual_detail=torch.zeros_like(common_fact),
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
        p1_factual_detail=common_fact,
        consequence=consequence,
        intent=active_intent,
        action_query=action_query,
    )
    assert torch.count_nonzero(active.temporal_semantic) > 0
    assert torch.count_nonzero(active.temporal_geometry) > 0
    no_effect, _ = compiler(
        p1_factual_detail=common_fact,
        consequence=replace(
            consequence,
            effect_by_type=torch.zeros_like(consequence.effect_by_type),
            interaction_by_type=torch.zeros_like(consequence.interaction_by_type),
            protected_consequence=consequence.factual_base,
        ),
        intent=active_intent,
        action_query=action_query,
    )
    assert torch.count_nonzero(no_effect.effect_semantic) == 0
    assert torch.count_nonzero(no_effect.effect_geometry) == 0
    assert torch.count_nonzero(no_effect.temporal_semantic) == 0
    assert torch.count_nonzero(no_effect.temporal_geometry) == 0
    changed_factual_base = common_fact + torch.randn_like(common_fact)
    changed_consequence = replace(
        consequence,
        factual_base=changed_factual_base,
        protected_consequence=(
            changed_factual_base + consequence.effect + consequence.interaction
        ),
    )
    changed, _ = compiler(
        p1_factual_detail=common_fact,
        consequence=changed_consequence,
        intent=active_intent,
        action_query=action_query,
    )
    for name in (
        "precision",
        "effect_semantic",
        "effect_geometry",
        "temporal_semantic",
        "temporal_geometry",
        "state_change",
    ):
        torch.testing.assert_close(
            getattr(changed, name),
            getattr(active, name),
            atol=0.0,
            rtol=0.0,
        )

    # The dynamic P1 policy block refines P2's query upstream.  P3 accepts no
    # live P1 residual argument, so precision has exactly one P1-owned value
    # source: the cached high-resolution factual detail.
    assert "p1_policy_residual" not in inspect.signature(compiler.forward).parameters


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
    assert torch.count_nonzero(neutral_effect.effect_by_type) == 0
    assert torch.count_nonzero(changed_effect.effect_by_type) > 0


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
