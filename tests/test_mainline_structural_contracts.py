from __future__ import annotations

import copy
import inspect
from dataclasses import fields, is_dataclass, replace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import clearvla.mainline.model.grounding as grounding_module
import clearvla.mainline.model.intent as intent_module
import clearvla.mainline.model.types as model_types
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


def _p1_state(
    factual_base: torch.Tensor,
    policy_query_residual: torch.Tensor | None = None,
):
    completed_type = getattr(model_types, "CompletedP1PolicyState", None)
    assert completed_type is not None
    return completed_type(
        factual_base=factual_base,
        policy_query_residual=(
            torch.zeros_like(factual_base)
            if policy_query_residual is None
            else policy_query_residual
        ),
    )


def _future_dynamics(
    *,
    batch: int = 1,
    intervals: int = 4,
    objects: int = 2,
    cameras: int = 2,
    content: int = 8,
    semantic_delta: torch.Tensor | None = None,
    transport_mean: torch.Tensor | None = None,
    transport_covariance: torch.Tensor | None = None,
    chart_availability: torch.Tensor | None = None,
    camera_chart_availability: torch.Tensor | None = None,
) -> FutureObjectDynamics:
    """Construct the active supervised W/P2 ABI without status aliases."""

    current = torch.zeros(batch, objects, content)
    semantic = (
        torch.zeros(batch, intervals, objects, content)
        if semantic_delta is None
        else semantic_delta
    )
    transport = (
        torch.zeros(batch, intervals, objects, cameras, 2)
        if transport_mean is None
        else transport_mean
    )
    covariance = (
        torch.zeros(batch, intervals, objects, cameras, 3, dtype=torch.float32)
        if transport_covariance is None
        else transport_covariance.float()
    )
    object_support = (
        torch.ones(batch, objects, 1) if chart_availability is None else chart_availability
    )
    camera_support = (
        torch.ones(batch, objects, cameras, 1)
        if camera_chart_availability is None
        else camera_chart_availability
    )
    return FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None] + semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        transport_covariance=covariance,
        chart_availability=object_support,
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=camera_support,
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


def test_grounder_reconstructs_only_the_independent_observed_dino_target() -> None:
    torch.manual_seed(201)
    local = _local_facts(content=8, route=4, hidden=16, observed=True)
    target = local.target_dino_content.detach().clone().requires_grad_(True)
    local = replace(local, target_dino_content=target)
    chart = dense_chart_from_local_facts(local)
    torch.testing.assert_close(
        chart.dino_content,
        target.detach(),
        atol=0.0,
        rtol=0.0,
    )
    assert not chart.dino_content.requires_grad
    self_mixture = (
        local.content_slots * local.semantic_owner_probs[..., None]
    ).sum(dim=-2)
    assert not torch.equal(chart.dino_content, self_mixture)

    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    observed_facts, _ = grounder(local, collect_diagnostics=False)
    observed_facts.reconstruction_error.backward()
    assert target.grad is None

    masked = replace(
        local,
        cell_observed=torch.zeros_like(local.cell_observed),
    )
    masked_facts, metrics = grounder(masked)
    assert float(masked_facts.reconstruction_error.detach()) == 0.0
    assert float(metrics["object_grounding_reconstruction_mse"]) == 0.0


def test_conditional_k_reconstruction_assignment_is_fp32_null_free_and_zero_safe() -> None:
    torch.manual_seed(202)
    conditional_k = torch.softmax(
        torch.randn(2, 7, 4, dtype=torch.float32),
        dim=-1,
    )
    local_prior = torch.rand(2, 7, 1, dtype=torch.float16)
    validity = torch.ones_like(local_prior)
    validity[:, 3] = 0.0
    assignment = grounding_module._conditional_k_reconstruction_assignment(
        conditional_k,
        local_prior,
        validity,
    )
    assert assignment.dtype == torch.float32
    torch.testing.assert_close(
        assignment.sum(dim=-1, keepdim=True),
        local_prior.float() * validity.float(),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.equal(assignment[:, 3], torch.zeros_like(assignment[:, 3]))

    high_real_mass = 0.9 * conditional_k.float()
    low_real_mass = 0.001 * conditional_k.float()
    high_conditional = high_real_mass / high_real_mass.sum(dim=-1, keepdim=True)
    low_conditional = low_real_mass / low_real_mass.sum(dim=-1, keepdim=True)
    high = grounding_module._conditional_k_reconstruction_assignment(
        high_conditional,
        local_prior,
        validity,
    )
    low = grounding_module._conditional_k_reconstruction_assignment(
        low_conditional,
        local_prior,
        validity,
    )
    torch.testing.assert_close(high, assignment, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(low, assignment, atol=1.0e-6, rtol=1.0e-6)

    with pytest.raises(ValueError, match="conditional owner"):
        grounding_module._conditional_k_reconstruction_assignment(
            conditional_k[..., 0],
            local_prior,
            validity,
        )
    with pytest.raises(ValueError, match="candidate prior"):
        grounding_module._conditional_k_reconstruction_assignment(
            conditional_k,
            local_prior[:, :-1],
            validity,
        )


def test_g3_changes_only_conditional_k_and_preserves_parent_real_null_mass() -> None:
    torch.manual_seed(203)
    local = _local_facts(cameras=2, content=8, route=4, hidden=16)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=2,
    ).eval()
    baseline, _ = grounder(local, collect_diagnostics=False)

    def distinct_k_residual(pair: torch.Tensor) -> torch.Tensor:
        pattern = torch.linspace(
            -2.0,
            2.0,
            grounder.objects,
            device=pair.device,
            dtype=pair.dtype,
        ).reshape(1, 1, grounder.objects, 1)
        return pattern.expand(pair.shape[0], pair.shape[1], -1, -1)

    with mock.patch.object(
        grounder.g3_residual,
        "forward",
        side_effect=distinct_k_residual,
    ):
        changed, _ = grounder(local, collect_diagnostics=False)

    assert not torch.equal(changed.candidate_assignment, baseline.candidate_assignment)
    torch.testing.assert_close(
        changed.candidate_assignment.sum(dim=1),
        baseline.candidate_assignment.sum(dim=1),
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    torch.testing.assert_close(
        changed.null_assignment,
        baseline.null_assignment,
        atol=2.0e-7,
        rtol=2.0e-7,
    )


def test_grounder_slot_residual_is_exported_before_reconstruction() -> None:
    torch.manual_seed(204)
    local = _local_facts(content=8, route=4, hidden=16, observed=True)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    ).eval()
    baseline, _ = grounder(local, collect_diagnostics=False)
    captured: dict[str, torch.Tensor] = {}

    def exported_residual(slots: torch.Tensor) -> torch.Tensor:
        object_value = torch.linspace(
            -0.2,
            0.2,
            grounder.objects,
            device=slots.device,
            dtype=slots.dtype,
        ).reshape(1, grounder.objects, 1)
        feature_value = torch.linspace(
            0.5,
            1.0,
            grounder.content_dim,
            device=slots.device,
            dtype=slots.dtype,
        ).reshape(1, 1, grounder.content_dim)
        value = object_value * feature_value
        captured["value"] = value.expand(slots.shape[0], -1, -1)
        return captured["value"]

    with mock.patch.object(
        grounder.decode_content_residual,
        "forward",
        side_effect=exported_residual,
    ):
        changed, _ = grounder(local, collect_diagnostics=False)

    residual = captured["value"]
    torch.testing.assert_close(
        changed.content - baseline.content,
        residual,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    physical_assignment = changed.candidate_assignment.float()
    physical_real_mass = physical_assignment.sum(dim=1, keepdim=True)
    conditional_k = torch.where(
        physical_real_mass > 1.0e-8,
        physical_assignment / physical_real_mass.clamp_min(1.0e-8),
        torch.zeros_like(physical_assignment),
    )
    reconstruction_owner = (
        conditional_k
        * changed.dense_chart.candidate_owner_prior[:, None].float()
        * changed.dense_chart.candidate_validity[..., 0][:, None].float()
    ).sum(dim=-1)
    expected_delta = torch.einsum(
        "bkcyx,bkd->bcyxd",
        reconstruction_owner,
        residual.float(),
    )
    torch.testing.assert_close(
        changed.reconstructed_dino.float() - baseline.reconstructed_dino.float(),
        expected_delta,
        atol=3.0e-6,
        rtol=3.0e-6,
    )


def test_g02_reconstruction_gradients_reach_unique_content_and_assignment_owners() -> None:
    torch.manual_seed(205)
    local = _local_facts(content=8, route=4, hidden=16, observed=True)
    target = local.target_dino_content.detach().clone().requires_grad_(True)
    candidates = local.content_slots.detach().clone().requires_grad_(True)
    local = replace(
        local,
        target_dino_content=target,
        content_slots=candidates,
    )
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    facts, _ = grounder(local, collect_diagnostics=False)
    facts.reconstruction_error.backward()
    assert target.grad is None
    assert candidates.grad is not None and candidates.grad.abs().sum() > 0
    assert (
        grounder.decode_content_residual.weight.grad is not None
        and grounder.decode_content_residual.weight.grad.abs().sum() > 0
    )
    g3_output = grounder.g3_residual[-1]
    assert isinstance(g3_output, torch.nn.Linear)
    assert g3_output.weight.grad is not None and g3_output.weight.grad.abs().sum() > 0


def test_g02_retains_all_schema25_physical_binder_inputs() -> None:
    torch.manual_seed(206)
    grounder = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )
    chart = dense_chart_from_local_facts(
        _local_facts(cameras=2, content=8, route=4, hidden=16)
    )
    baseline = grounder._candidate_tokens(chart)
    perturbations = {
        "candidate_content": torch.randn_like(chart.candidate_content),
        "candidate_semantic": torch.randn_like(chart.candidate_semantic),
        "candidate_appearance": torch.randn_like(chart.candidate_appearance),
        "candidate_geometry": torch.randn_like(chart.candidate_geometry),
        "candidate_coordinates": 0.25 * torch.randn_like(chart.candidate_coordinates),
    }
    for name, delta in perturbations.items():
        changed = grounder._candidate_tokens(
            replace(chart, **{name: getattr(chart, name) + delta})
        )
        assert not torch.equal(changed, baseline), name


def test_future_recognizer_keeps_four_interval_whole_segment_targets() -> None:
    torch.manual_seed(2)
    batch, intervals, objects, content, cameras = 1, 4, 3, 8, 1
    current = torch.randn(batch, objects, content)
    semantic_delta = torch.randn(batch, intervals, objects, content)
    teacher = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None] + semantic_delta,
        semantic_delta=semantic_delta,
        transport_mean=torch.randn(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(
            batch, intervals, objects, cameras, 3, dtype=torch.float32
        ),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
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
        current_loss_support=torch.ones(batch, objects, cameras, 1),
    )
    assert tuple(result.interval_targets.shape) == (batch, intervals, 16)
    assert tuple(result.action_summary.shape) == (batch, intervals, 2)
    assert tuple(result.state_summary.shape) == (batch, intervals, 2)
    assert tuple(result.effect_summary.shape) == (batch, intervals, content)


def test_future_recognizer_supervises_neutral_objects_from_current_support() -> None:
    batch, intervals, objects, content, cameras = 1, 4, 2, 8, 1
    current = torch.randn(batch, objects, content)
    teacher = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None] + torch.ones(batch, intervals, objects, content),
        semantic_delta=torch.ones(batch, intervals, objects, content),
        transport_mean=torch.zeros(batch, intervals, objects, cameras, 2),
        transport_covariance=torch.zeros(
            batch, intervals, objects, cameras, 3, dtype=torch.float32
        ),
        chart_availability=torch.ones(batch, objects, 1),
        camera_coordinates=torch.zeros(batch, objects, cameras, 2),
        camera_chart_availability=torch.ones(batch, objects, cameras, 1),
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
        current_loss_support=torch.ones(batch, objects, cameras, 1),
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
        null_probability=torch.full((1, 1, 1, 1), 0.1),
        null_camera_measure=torch.full((1, 1, 2, 1), 0.5),
    )
    assert tuple(transport.shape) == (1, 1, 1, 2, 2)
    assert tuple(covariance.shape) == (1, 1, 1, 2, 3)
    torch.testing.assert_close(transport, torch.zeros_like(transport))
    torch.testing.assert_close(covariance, torch.zeros_like(covariance))

    permutation = torch.tensor((1, 0))
    permuted_transport, permuted_covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=posterior[:, :, :, permutation],
        candidate_coordinate=coordinate[permutation],
        current_camera_coordinate=current[:, :, permutation],
        null_probability=torch.full((1, 1, 1, 1), 0.1),
        null_camera_measure=torch.full((1, 1, 2, 1), 0.5)[:, :, permutation],
    )
    torch.testing.assert_close(permuted_transport, transport[:, :, :, permutation])
    torch.testing.assert_close(
        permuted_covariance,
        covariance[:, :, :, permutation],
    )


def test_teacher_null_identity_is_inside_each_camera_geometry_moment() -> None:
    coordinate = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    posterior = torch.zeros(1, 1, 1, 1, 1, 2)
    posterior[..., 0, 1] = 0.25
    transport, covariance = ObjectFutureTeacher._relative_geometry_moments(
        candidate_posterior=posterior,
        candidate_coordinate=coordinate,
        current_camera_coordinate=torch.zeros(1, 1, 1, 2),
        null_probability=torch.full((1, 1, 1, 1), 0.75),
        null_camera_measure=torch.ones(1, 1, 1, 1),
    )
    torch.testing.assert_close(
        transport,
        torch.tensor([[[[[0.25, 0.0]]]]]),
    )
    torch.testing.assert_close(
        covariance,
        torch.tensor([[[[[0.1875, 0.0, 0.0]]]]]),
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


def test_future_dynamics_abi_retains_camera_geometry_and_has_no_status_alias() -> None:
    field = _future_dynamics(cameras=3)
    assert {row.name for row in fields(field)} == {
        "current_reference",
        "successor_content",
        "semantic_delta",
        "transport_mean",
        "transport_covariance",
        "chart_availability",
        "camera_coordinates",
        "camera_chart_availability",
    }
    field.validate()
    assert tuple(field.transport_mean.shape) == (1, 4, 2, 3, 2)
    assert tuple(field.transport_covariance.shape) == (1, 4, 2, 3, 3)
    assert field.transport_covariance.dtype == torch.float32
    for removed in (
        "visibility",
        "persistence",
        "uncertainty",
        "reliability",
        "future_selector_validity",
        "future_address",
        "object_coordinates",
    ):
        assert not hasattr(field, removed)


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


def test_w_camera_field_is_psd_and_unavailable_camera_is_exact_zero() -> None:
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
    typed_common = torch.randn(1, 4, 3, 16)
    typed_interval = torch.randn(1, 2, 4, 3, 16)
    camera_validity = facts.camera_validity.clone()
    camera_validity[:, :, 0] = 0.0
    masked_facts = replace(facts, camera_validity=camera_validity)
    field = dynamics._field(
        facts=masked_facts,
        typed_common=typed_common,
        typed_interval_innovation=typed_interval,
    )
    assert field.transport_covariance.dtype == torch.float32
    xx = field.transport_covariance[..., 0]
    xy = field.transport_covariance[..., 1]
    yy = field.transport_covariance[..., 2]
    assert bool((xx >= 0.0).all())
    assert bool((yy >= 0.0).all())
    assert bool((xx * yy - xy.square() >= -1.0e-7).all())
    assert torch.count_nonzero(field.transport_mean[:, :, :, 0]) == 0
    assert torch.count_nonzero(field.transport_covariance[:, :, :, 0]) == 0
    names = {name for name, _ in dynamics.named_parameters()}
    assert not any(
        status in name for name in names for status in ("visibility", "persistence", "uncertainty")
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
    typed_common = torch.zeros(1, 4, 3, 16, requires_grad=True)
    typed_interval = torch.zeros(1, 2, 4, 3, 16, requires_grad=True)
    field = dynamics._field(
        facts=facts,
        typed_common=typed_common,
        typed_interval_innovation=typed_interval,
    )
    innovation = field.successor_content - field.current_reference[:, None]
    assert torch.count_nonzero(innovation) == 0
    innovation.sum().backward()
    assert facts.content.grad is None or torch.count_nonzero(facts.content.grad) == 0


def test_w_receives_completed_intent_and_coarse_action_as_distinct_inputs() -> None:

    torch.manual_seed(31)
    top = _object_top()
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
        intent.world_dock(),
        public_interval_carrier=torch.zeros_like(intent.public_interval_carrier),
    )
    signal_intent = replace(
        blank_intent,
        public_interval_carrier=torch.randn_like(intent.public_interval_carrier),
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


def test_w_common_is_written_once_and_zero_innovation_returns_common() -> None:
    torch.manual_seed(311)
    top = _object_top().eval()
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
    coarse = top.coarse_action(intent.action_dock())
    world = replace(
        intent.world_dock(),
        typed_common_value=torch.randn_like(intent.typed_common_value),
        typed_interval_residual_value=torch.zeros_like(intent.typed_interval_residual_value),
    )
    with torch.no_grad():
        top.dynamics.delta_head.weight.normal_(std=0.1)
        top.dynamics.transport_head.weight.normal_(std=0.1)

    w1_interval_counts: list[int] = []

    def record_w1_interval_count(_module, args) -> None:
        w1_interval_counts.append(int(args[0].shape[1]))

    handle = top.dynamics.w1.register_forward_pre_hook(record_w1_interval_count)
    try:
        _, working, _ = top.dynamics.forward_w1(
            facts=facts,
            intent=world,
            action=coarse,
            collect_diagnostics=False,
        )
        field, _ = top.dynamics.forward_w2(
            facts=facts,
            intent=world,
            action=coarse,
            w1_state=working,
            collect_diagnostics=False,
        )
    finally:
        handle.remove()

    assert w1_interval_counts.count(1) == 1
    assert torch.count_nonzero(working.near_interval_innovation) == 0
    for index in range(1, field.intervals):
        torch.testing.assert_close(
            field.semantic_delta[:, index],
            field.semantic_delta[:, 0],
        )
        torch.testing.assert_close(
            field.transport_mean[:, index],
            field.transport_mean[:, 0],
        )

    zero_world = replace(
        world,
        typed_common_value=torch.zeros_like(world.typed_common_value),
    )
    _, zero_working, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=zero_world,
        action=coarse,
        collect_diagnostics=False,
    )
    zero_field, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=zero_world,
        action=coarse,
        w1_state=zero_working,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_working.common_typed) == 0
    assert torch.count_nonzero(zero_field.semantic_delta) == 0
    assert torch.count_nonzero(zero_field.transport_mean) == 0


def test_w2_far_reads_w1_but_cannot_rewrite_common_or_near() -> None:
    torch.manual_seed(312)
    top = _object_top().eval()
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
    coarse = top.coarse_action(intent.action_dock())
    with torch.no_grad():
        top.dynamics.delta_head.weight.normal_(std=0.1)
        top.dynamics.transport_head.weight.normal_(std=0.1)
    world = intent.world_dock()
    _, baseline_w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=world,
        action=coarse,
        collect_diagnostics=False,
    )
    baseline, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=world,
        action=coarse,
        w1_state=baseline_w1,
        collect_diagnostics=False,
    )

    far_value = world.typed_interval_residual_value.clone()
    far_value[:, 2:] += torch.randn_like(far_value[:, 2:])
    far_world = replace(world, typed_interval_residual_value=far_value)
    _, changed_w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=far_world,
        action=coarse,
        collect_diagnostics=False,
    )
    changed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=far_world,
        action=coarse,
        w1_state=changed_w1,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(changed_w1.common_typed, baseline_w1.common_typed)
    torch.testing.assert_close(
        changed_w1.near_interval_innovation,
        baseline_w1.near_interval_innovation,
    )
    torch.testing.assert_close(changed.semantic_delta[:, :2], baseline.semantic_delta[:, :2])
    assert not torch.equal(changed.semantic_delta[:, 2:], baseline.semantic_delta[:, 2:])

    near_value = world.typed_interval_residual_value.clone()
    near_value[:, :2] += torch.randn_like(near_value[:, :2])
    near_world = replace(world, typed_interval_residual_value=near_value)
    _, near_w1, _ = top.dynamics.forward_w1(
        facts=facts,
        intent=near_world,
        action=coarse,
        collect_diagnostics=False,
    )
    near_changed, _ = top.dynamics.forward_w2(
        facts=facts,
        intent=near_world,
        action=coarse,
        w1_state=near_w1,
        collect_diagnostics=False,
    )
    assert not torch.equal(
        near_changed.semantic_delta[:, 2:],
        baseline.semantic_delta[:, 2:],
    )


def test_w_camera_prior_perturbation_stays_on_its_camera_axis() -> None:
    torch.manual_seed(313)
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
        dynamics.transport_head.weight.normal_(std=0.1)
    typed_common = torch.zeros(1, 4, 3, 16)
    typed_common[..., 2, :] = torch.randn(1, 4, 16)
    typed_interval = torch.zeros(1, 2, 4, 3, 16)
    baseline = dynamics._field(
        facts=facts,
        typed_common=typed_common,
        typed_interval_innovation=typed_interval,
    )
    changed_prior = facts.camera_transport_prior.clone()
    changed_prior[:, :, 1] += 0.5
    changed = dynamics._field(
        facts=replace(facts, camera_transport_prior=changed_prior),
        typed_common=typed_common,
        typed_interval_innovation=typed_interval,
    )
    torch.testing.assert_close(
        changed.transport_mean[:, :, :, 0],
        baseline.transport_mean[:, :, :, 0],
    )
    assert not torch.equal(
        changed.transport_mean[:, :, :, 1],
        baseline.transport_mean[:, :, :, 1],
    )


def test_w_appearance_conditions_semantic_but_cannot_create_it() -> None:
    semantic = torch.randn(2, 3, 4)
    appearance = torch.randn_like(semantic)
    zero_semantic, _ = ObjectFutureDynamicsCompiler._appearance_condition_semantic(
        torch.zeros_like(semantic),
        appearance,
    )
    identity_semantic, _ = ObjectFutureDynamicsCompiler._appearance_condition_semantic(
        semantic,
        torch.zeros_like(appearance),
    )
    conditioned, _ = ObjectFutureDynamicsCompiler._appearance_condition_semantic(
        semantic,
        appearance,
    )
    assert torch.count_nonzero(zero_semantic) == 0
    torch.testing.assert_close(identity_semantic, semantic)
    assert not torch.equal(conditioned, semantic)


def test_p2_consumes_camera_covariance_and_zero_support_is_exact_zero() -> None:
    torch.manual_seed(314)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    neutral = FutureObjectDynamics.neutral(context.facts)
    transport = neutral.transport_mean.clone()
    transport[:, :, 0, 1, 0] = 0.8
    coordinates = torch.zeros_like(neutral.camera_coordinates)
    covariance = neutral.transport_covariance.clone()
    broad_covariance = covariance.clone()
    broad_covariance[:, :, 0, 1, 0] = 9.0
    broad_covariance[:, :, 0, 1, 2] = 9.0
    available = replace(
        neutral,
        transport_mean=transport,
        transport_covariance=covariance,
        camera_coordinates=coordinates,
        chart_availability=torch.ones_like(neutral.chart_availability),
        camera_chart_availability=torch.ones_like(neutral.camera_chart_availability),
    )
    broadened = replace(available, transport_covariance=broad_covariance)
    reader = top.effect_reader
    with torch.no_grad():
        for projection in reader.source_query:
            projection.weight.zero_()
        for projection in reader.source_key:
            projection.weight.zero_()
        reader.public_interval_key.weight.zero_()
        for projection in reader.typed_intent_key:
            projection.weight.zero_()
        reader.coordinate_query.weight.zero_()
        reader.semantic_value.weight.zero_()
        reader.transport_value.weight.zero_()
        reader.transport_value.weight[0, 0] = 1.0
    action_query = torch.zeros(1, 24, 2, 32)
    ordinary, _ = reader(
        action_query,
        available,
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    broad, _ = reader(
        action_query,
        broadened,
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    assert not torch.equal(broad, ordinary)
    assert not hasattr(reader, "status_value")
    assert not hasattr(reader, "type_query")
    assert not any("null" in name for name, _ in reader.named_parameters())

    unavailable = replace(
        available,
        chart_availability=torch.zeros_like(available.chart_availability),
        camera_chart_availability=torch.zeros_like(available.camera_chart_availability),
    )
    zero, _ = reader(
        action_query,
        unavailable,
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero) == 0

    reader.zero_grad(set_to_none=True)
    gradient_query = action_query.clone().requires_grad_(True)
    differentiable_zero, _ = reader(
        gradient_query,
        unavailable,
        context.intent.policy_dock(),
        collect_diagnostics=False,
    )
    differentiable_zero.sum().backward()
    assert gradient_query.grad is not None
    assert torch.isfinite(gradient_query.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in reader.parameters()
    )


def test_p2_policy_dock_exposes_existing_typed_metadata_by_identity() -> None:
    torch.manual_seed(315)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    dock = context.intent.policy_dock()
    assert dock.typed_common_value is context.intent.typed_common_value
    assert (
        dock.typed_interval_residual_value
        is context.intent.typed_interval_residual_value
    )


def test_p2_spatial_selection_retains_interval_and_s_cannot_select_w() -> None:
    torch.manual_seed(316)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    semantic = torch.zeros(1, 4, 4, 16)
    semantic[:, 2, 0, 0] = 1.0
    dynamics = _future_dynamics(
        content=16,
        objects=4,
        semantic_delta=semantic,
        chart_availability=torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]),
        camera_chart_availability=torch.zeros(1, 4, 2, 1),
    )
    reader = top.effect_reader
    with torch.no_grad():
        for projection in reader.source_query:
            projection.weight.zero_()
        for projection in reader.source_key:
            projection.weight.zero_()
        reader.semantic_value.weight.zero_()
        reader.semantic_value.weight[0, 0] = 1.0
        reader.transport_value.weight.zero_()
        reader.coordinate_query.weight.zero_()
    action_query = torch.zeros(1, 24, 2, 32)
    dock = context.intent.policy_dock()
    selected, _ = reader.spatial_select(
        action_query,
        dynamics,
        dock,
        collect_diagnostics=False,
    )
    selected.validate()
    assert tuple(selected.value.shape) == (1, 24, 2, 4, 2, 32)
    assert tuple(selected.support.shape) == (1, 4, 2)
    assert selected.support.dtype == torch.bool
    torch.testing.assert_close(
        selected.value[0, 0, 0, :, 0, 0],
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )

    changed_dock = replace(
        dock,
        interval_key=torch.randn_like(dock.interval_key),
        typed_common_value=torch.randn_like(dock.typed_common_value),
        typed_interval_residual_value=torch.randn_like(
            dock.typed_interval_residual_value
        ),
    )
    changed, _ = reader.spatial_select(
        action_query,
        dynamics,
        changed_dock,
        collect_diagnostics=False,
    )
    for name in ("key", "value", "common_value", "residual_value", "support"):
        torch.testing.assert_close(getattr(changed, name), getattr(selected, name))
    assert not torch.equal(changed.selected_s_context, selected.selected_s_context)


def test_p2_physical_terminal_has_no_null_or_type_competition() -> None:
    torch.manual_seed(317)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    semantic = torch.zeros(1, 4, 4, 16)
    semantic[..., 0] = 1.0
    transport = torch.zeros(1, 4, 4, 2, 2)
    transport[..., 0] = 2.0
    both = _future_dynamics(
        content=16,
        objects=4,
        semantic_delta=semantic,
        transport_mean=transport,
    )
    semantic_only = replace(
        both,
        transport_mean=torch.zeros_like(transport),
        camera_chart_availability=torch.zeros_like(
            both.camera_chart_availability
        ),
    )
    geometry_only = replace(
        both,
        semantic_delta=torch.zeros_like(semantic),
        successor_content=both.current_reference[:, None].expand(-1, 4, -1, -1),
    )
    zero_w = replace(
        geometry_only,
        transport_mean=torch.zeros_like(transport),
    )
    reader = top.effect_reader
    with torch.no_grad():
        for projection in reader.source_query:
            projection.weight.zero_()
        for projection in reader.source_key:
            projection.weight.zero_()
        reader.public_interval_key.weight.zero_()
        for projection in reader.typed_intent_key:
            projection.weight.zero_()
        reader.coordinate_query.weight.zero_()
        reader.semantic_value.weight.zero_()
        reader.semantic_value.weight[0, 0] = 1.0
        reader.transport_value.weight.zero_()
        reader.transport_value.weight[0, 0] = 1.0
    action_query = torch.randn(1, 24, 2, 32)
    dock = context.intent.policy_dock()
    semantic_raw, _ = reader(
        action_query,
        semantic_only,
        dock,
        collect_diagnostics=False,
    )
    geometry_raw, _ = reader(
        action_query,
        geometry_only,
        dock,
        collect_diagnostics=False,
    )
    combined_raw, _ = reader(
        action_query,
        both,
        dock,
        collect_diagnostics=False,
    )
    neutral_raw, _ = reader(
        action_query,
        zero_w,
        replace(
            dock,
            interval_key=torch.randn_like(dock.interval_key),
            typed_common_value=torch.randn_like(dock.typed_common_value),
            typed_interval_residual_value=torch.randn_like(
                dock.typed_interval_residual_value
            ),
        ),
        collect_diagnostics=False,
    )
    torch.testing.assert_close(semantic_raw[..., 0], torch.ones_like(semantic_raw[..., 0]))
    torch.testing.assert_close(
        geometry_raw[..., 0],
        torch.full_like(geometry_raw[..., 0], 2.0),
    )
    torch.testing.assert_close(combined_raw, semantic_raw + geometry_raw)
    assert torch.count_nonzero(neutral_raw) == 0
    assert not any(
        token in name
        for name, _ in reader.named_parameters()
        for token in ("null", "type_query", "type_gain")
    )


def test_p2_reverse_path_reaches_each_legal_w_s_and_action_owner() -> None:
    torch.manual_seed(318)
    top = _object_top().eval()
    context, _ = top.build_online_context(
        local_facts=_local_facts(cameras=2),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    semantic = torch.randn(1, 4, 4, 16, requires_grad=True)
    transport = (0.1 * torch.randn(1, 4, 4, 2, 2)).requires_grad_(True)
    dynamics = _future_dynamics(
        content=16,
        objects=4,
        semantic_delta=semantic,
        transport_mean=transport,
    )
    dock = context.intent.policy_dock()
    typed_common = dock.typed_common_value.detach().clone().requires_grad_(True)
    typed_residual = (
        dock.typed_interval_residual_value.detach().clone().requires_grad_(True)
    )
    action_query = torch.randn(1, 24, 2, 32, requires_grad=True)
    value, _ = top.effect_reader(
        action_query,
        dynamics,
        replace(
            dock,
            typed_common_value=typed_common,
            typed_interval_residual_value=typed_residual,
        ),
        collect_diagnostics=False,
    )
    gradients = torch.autograd.grad(
        value.square().mean(),
        (semantic, transport, typed_common, typed_residual, action_query),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_stateless_intent_is_repeatable_without_frame_progress_input() -> None:

    torch.manual_seed(30)
    top = _object_top()
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
    top = _object_top()
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


def test_future_objectives_use_only_semantic_camera_geometry_and_current_support() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    target = _future_dynamics(
        batch=batch,
        intervals=intervals,
        objects=objects,
        cameras=cameras,
        content=content,
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
    for removed in (
        "future_successor",
        "future_visibility",
        "future_persistence",
        "future_uncertainty",
    ):
        assert removed not in unreliable

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
    assert unreliable_content["future_semantic_delta"] > 0

    unsupported = future_dynamics_terms(
        prediction,
        target,
        current_loss_support=torch.zeros_like(current_support),
    )
    torch.testing.assert_close(unsupported["future_dynamics"], torch.zeros(()))


def test_future_interval_transition_penalizes_temporal_collapse_not_common_offset() -> None:
    batch, intervals, objects, cameras, content = 1, 4, 2, 1, 8
    interval = torch.arange(intervals, dtype=torch.float32)[None, :, None, None]
    semantic = interval.expand(batch, intervals, objects, content).clone()
    target = _future_dynamics(
        batch=batch,
        intervals=intervals,
        objects=objects,
        cameras=cameras,
        content=content,
        semantic_delta=semantic,
    )
    shifted_semantic = semantic + 7.0
    shifted = replace(
        target,
        semantic_delta=shifted_semantic,
        successor_content=target.current_reference[:, None] + shifted_semantic,
    )
    collapsed = replace(
        target,
        semantic_delta=torch.zeros_like(semantic),
        successor_content=target.current_reference[:, None].expand(-1, intervals, -1, -1),
    )

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


def test_teacher_association_audit_falls_for_semantically_opposed_supports() -> None:
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
    high_target, high_metrics = teacher(
        facts=facts,
        future_supports=high,
        future_offsets=offsets,
        collect_diagnostics=True,
    )
    low_target, low_metrics = teacher(
        facts=facts,
        future_supports=-high,
        future_offsets=offsets,
        collect_diagnostics=True,
    )
    assert (
        high_metrics["object_teacher_association_real_mass"]
        > low_metrics["object_teacher_association_real_mass"]
    )
    torch.testing.assert_close(
        high_target.semantic_delta,
        high_target.successor_content - high_target.current_reference[:, None],
    )
    torch.testing.assert_close(
        low_target.semantic_delta,
        low_target.successor_content - low_target.current_reference[:, None],
    )


def test_teacher_keeps_row_softmax_and_exports_no_status_or_address() -> None:
    """W-02 changes physical moments, not the Schema25 association backend."""

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

    supports = torch.randn(1, 4, 1, 2, 2, 8)
    target, _ = teacher(
        facts=facts,
        future_supports=supports,
        future_offsets=torch.tensor((6, 12, 24, 40)),
    )
    torch.testing.assert_close(
        target.semantic_delta,
        target.successor_content - target.current_reference[:, None],
    )
    assert tuple(target.transport_mean.shape) == (1, 4, 4, 1, 2)
    assert tuple(target.transport_covariance.shape) == (1, 4, 4, 1, 3)
    source = inspect.getsource(ObjectFutureTeacher)
    assert "torch.softmax(torch.cat((candidate_flat, null_logit)" in source
    assert "_partial_assignment" not in source
    for removed in (
        "visibility",
        "persistence",
        "uncertainty",
        "reliability",
        "future_selector_validity",
        "future_address",
    ):
        assert not hasattr(target, removed)


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


def test_teacher_camera_relabeling_permutes_the_physical_geometry_axis() -> None:
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
        if field.name in ("transport_mean", "transport_covariance"):
            expected = expected[:, :, :, camera_permutation]
        elif field.name in ("camera_coordinates", "camera_chart_availability"):
            expected = expected[:, :, camera_permutation]
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
        p1_state=_p1_state(p1_fact),
        action_query=action_query,
    )
    relabeled_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=relabeled_intent,
            predicted_dynamics=relabeled_dynamics,
        ),
        p1_state=_p1_state(p1_fact),
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
    for name in (
        "protected_base",
        "protected_policy_precision",
        "precision",
        "temporal",
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
        "executed_history": torch.randn(1, 3, 7),
        "collect_diagnostics": False,
    }

    def organize(current_facts):
        return top.intent(facts=current_facts, **inputs)[0]

    intent = organize(facts)
    assert tuple(intent.typed_relevance_mass.shape[:4]) == (1, 4, 4, 3)
    assert tuple(intent.typed_relevance_value.shape[:4]) == (1, 4, 4, 3)

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
        semantic_intent.typed_relevance_value[..., 0, :],
        intent.typed_relevance_value[..., 0, :],
    )
    torch.testing.assert_close(
        semantic_intent.typed_relevance_mass[..., 1:, :],
        intent.typed_relevance_mass[..., 1:, :],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        semantic_intent.typed_relevance_value[..., 1:, :],
        intent.typed_relevance_value[..., 1:, :],
        atol=0.0,
        rtol=0.0,
    )

    zero_semantic = organize(replace(facts, semantic=torch.zeros_like(facts.semantic)))
    assert torch.count_nonzero(zero_semantic.typed_relevance_value[..., 0, :]) == 0
    assert torch.count_nonzero(zero_semantic.typed_policy_components[..., 0, :]) == 0

    invalid_facts = replace(facts, validity=torch.zeros_like(facts.validity))
    invalid_intent = organize(invalid_facts)
    assert torch.count_nonzero(invalid_intent.typed_relevance_value) == 0
    assert torch.count_nonzero(invalid_intent.typed_policy_components) == 0
    torch.testing.assert_close(
        invalid_intent.policy_interval_context,
        invalid_intent.public_interval_carrier,
        atol=0.0,
        rtol=0.0,
    )
    coarse = top.coarse_action(invalid_intent.action_dock())
    _, common, residual, _ = top.dynamics._base(
        invalid_facts,
        invalid_intent.world_dock(),
        coarse,
        collect_diagnostics=True,
    )
    assert torch.count_nonzero(common) == 0
    assert torch.count_nonzero(residual) == 0


def test_s_interval_common_residual_is_lossless_axis_preserving_and_vjp_exact() -> None:
    torch.manual_seed(371)
    source = torch.randn(2, 4, 5, 3, 7, requires_grad=True)
    common, residual = intent_module._interval_common_residual(source)
    assert tuple(common.shape) == (2, 5, 3, 7)
    assert tuple(residual.shape) == tuple(source.shape)
    reconstructed = common[:, None] + residual
    torch.testing.assert_close(reconstructed, source, atol=2.0e-7, rtol=2.0e-7)
    torch.testing.assert_close(
        residual.sum(dim=1),
        torch.zeros_like(common),
        atol=5.0e-7,
        rtol=0.0,
    )

    cotangent = torch.randn_like(source)
    (source_vjp,) = torch.autograd.grad(
        reconstructed,
        source,
        grad_outputs=cotangent,
    )
    torch.testing.assert_close(source_vjp, cotangent, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="four interval"):
        intent_module._interval_common_residual(torch.randn(2, 3, 5))


def test_s_integrated_common_residual_reconstructs_unchanged_schema25_scoring() -> None:
    torch.manual_seed(372)
    top = _object_top().eval()
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
    raw_mass, raw_value, raw_components, _, _ = top.intent._typed_relevance(
        public_interval_carrier=intent.public_interval_carrier,
        facts=facts,
    )
    torch.testing.assert_close(
        intent.typed_common_mass[:, None] + intent.typed_interval_residual_mass,
        raw_mass,
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    torch.testing.assert_close(
        intent.typed_common_value[:, None] + intent.typed_interval_residual_value,
        raw_value,
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    torch.testing.assert_close(intent.typed_policy_components, raw_components)
    torch.testing.assert_close(
        intent.typed_interval_residual_mass.sum(dim=1),
        torch.zeros_like(intent.typed_common_mass),
        atol=5.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        intent.typed_interval_residual_value.sum(dim=1),
        torch.zeros_like(intent.typed_common_value),
        atol=5.0e-7,
        rtol=0.0,
    )
    assert tuple(intent.typed_common_value.shape[:3]) == (1, 4, 3)
    assert tuple(intent.typed_interval_residual_value.shape[:4]) == (1, 4, 4, 3)


def test_coarse_action_cannot_duplicate_the_typed_world_ingress() -> None:
    torch.manual_seed(373)
    top = _object_top().eval()
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
    action_field_names = {field.name for field in fields(intent.action_dock())}
    assert not any("typed" in name for name in action_field_names)
    assert "typed_" not in inspect.getsource(type(top.coarse_action).forward)

    zero_typed = replace(
        intent,
        typed_common_mass=torch.zeros_like(intent.typed_common_mass),
        typed_common_value=torch.zeros_like(intent.typed_common_value),
        typed_interval_residual_mass=torch.zeros_like(intent.typed_interval_residual_mass),
        typed_interval_residual_value=torch.zeros_like(intent.typed_interval_residual_value),
        typed_policy_components=torch.zeros_like(intent.typed_policy_components),
        policy_interval_context=intent.public_interval_carrier,
    )
    coarse = top.coarse_action(intent.action_dock())
    zero_coarse = top.coarse_action(zero_typed.action_dock())
    torch.testing.assert_close(coarse.tokens, zero_coarse.tokens, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        coarse.action_prediction,
        zero_coarse.action_prediction,
        atol=0.0,
        rtol=0.0,
    )

    with_typed_base, with_common, with_residual, _ = top.dynamics._base(
        facts,
        intent.world_dock(),
        coarse,
        collect_diagnostics=False,
    )
    without_typed_base, without_common, without_residual, _ = top.dynamics._base(
        facts,
        zero_typed.world_dock(),
        zero_coarse,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(with_typed_base, without_typed_base)
    assert not torch.equal(with_common, without_common)
    assert not torch.equal(with_residual, without_residual)


def test_s_typed_world_and_coarse_action_gradients_have_distinct_owners() -> None:
    torch.manual_seed(374)
    top = _object_top()
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
    coarse = top.coarse_action(intent.action_dock())
    coarse_typed_grads = torch.autograd.grad(
        coarse.tokens.float().square().mean(),
        (intent.typed_common_value, intent.typed_interval_residual_value),
        allow_unused=True,
        retain_graph=True,
    )
    assert coarse_typed_grads == (None, None)

    world_dock = intent.world_dock()
    independent_common = world_dock.typed_common_value.detach().clone().requires_grad_(True)
    independent_residual = (
        world_dock.typed_interval_residual_value.detach().clone().requires_grad_(True)
    )
    independent_world_dock = replace(
        world_dock,
        typed_common_value=independent_common,
        typed_interval_residual_value=independent_residual,
    )
    generic, common, residual, _ = top.dynamics._base(
        facts,
        independent_world_dock,
        coarse,
        collect_diagnostics=False,
    )
    common_grad, residual_grad, action_grad = torch.autograd.grad(
        (
            generic.float().square().mean()
            + common.float().square().mean()
            + residual.float().square().mean()
        ),
        (
            independent_common,
            independent_residual,
            coarse.tokens,
        ),
        allow_unused=True,
    )
    assert common_grad is not None and common_grad.abs().sum() > 0
    assert residual_grad is not None and residual_grad.abs().sum() > 0
    assert action_grad is not None and action_grad.abs().sum() > 0


def test_s03_keeps_future_owner_supervision_outside_online_intent() -> None:
    signature = inspect.signature(intent_module.StatelessObjectIntentOrganizer.forward)
    assert not any(
        name in signature.parameters
        for name in ("future_supports", "future_state", "teacher", "future_action")
    )
    source = inspect.getsource(intent_module)
    for forbidden in (
        "DirectIntentFutureSupervisor",
        "ObservableIntentStateSupervisor",
        "IntentFutureSupervision",
    ):
        assert forbidden not in source


def test_w_semantic_and_appearance_have_fixed_nonalias_roles() -> None:
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
    with torch.no_grad():
        decoded = torch.randn_like(top.dynamics.delta_head.weight)
        top.dynamics.delta_head.weight.copy_(decoded)
        relabeled_top.dynamics.delta_head.weight.copy_(decoded)

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
        "executed_history": torch.randn(1, 3, 7),
        "collect_diagnostics": False,
    }
    intent = top.intent(facts=facts, **inputs)[0]
    relabeled_intent = relabeled_top.intent(facts=relabeled_facts, **inputs)[0]

    torch.testing.assert_close(
        relabeled_intent.public_interval_carrier,
        intent.public_interval_carrier,
    )
    torch.testing.assert_close(
        relabeled_intent.typed_relevance_mass[..., (0, 1), :],
        intent.typed_relevance_mass[..., (1, 0), :],
    )
    torch.testing.assert_close(
        relabeled_intent.typed_relevance_value[..., (0, 1), :],
        intent.typed_relevance_value[..., (1, 0), :],
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
    assert not torch.equal(relabeled_dynamics.semantic_delta, dynamics.semantic_delta)
    torch.testing.assert_close(
        relabeled_dynamics.transport_mean,
        dynamics.transport_mean,
    )
    torch.testing.assert_close(
        relabeled_dynamics.transport_covariance,
        dynamics.transport_covariance,
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
        executed_history=torch.randn(1, 3, 7),
        facts=facts,
        collect_diagnostics=False,
    )
    intent.public_interval_carrier.float().square().mean().backward()
    for parameter in top.intent.typed_relevance_queries.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert top.intent.typed_temperature_logit.grad is None


def test_coarse_action_and_w_have_no_raw_typed_fact_reread() -> None:
    top = _object_top()
    for name in (
        "semantic_read",
        "appearance_read",
        "geometry_read",
        "typed_router",
    ):
        assert not hasattr(top.coarse_action, name)
    assert not hasattr(top.dynamics, "typed_router")
    source = inspect.getsource(type(top.dynamics)._base)
    for forbidden in ("facts.semantic", "facts.appearance", "facts.geometry"):
        assert forbidden not in source
    assert "intent.typed_common_value" in source
    assert "intent.typed_interval_residual_value" in source


def test_neutral_w_preserves_current_precision_and_temporal_without_w_interaction() -> None:
    torch.manual_seed(4)
    top = _object_top()
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
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
        p1_state=_p1_state(p1_fact),
        action_query=action_query,
    )
    neutral_other_query, _ = top.compile_policy(
        deployment,
        p1_state=_p1_state(p1_fact),
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
        policy_interval_context=context.intent.interval_queries
        + 1000.0 * torch.randn_like(context.intent.interval_queries),
    )
    identity_only_compiled, _ = top.compile_policy(
        DeploymentTopCache(
            intent=identity_only_intent,
            predicted_dynamics=deployment.predicted_dynamics,
        ),
        p1_state=_p1_state(p1_fact),
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


def test_supervised_successor_innovation_crosses_w_to_p2_without_current_bypass() -> None:
    torch.manual_seed(28)
    top = _object_top()
    context, _ = top.build_online_context(
        local_facts=_local_facts(),
        goal_tokens=torch.randn(1, 6, 12),
        goal_mask=torch.ones(1, 6, dtype=torch.bool),
        state_history=torch.randn(1, 3, 7),
        state=torch.randn(1, 7),
        executed_history=torch.randn(1, 3, 7),
    )
    horizon, basis, hidden = 24, 2, 32
    neutral = FutureObjectDynamics.neutral(context.facts)
    changed_delta = neutral.semantic_delta + 0.25 * torch.randn_like(neutral.semantic_delta)
    changed = replace(
        neutral,
        semantic_delta=changed_delta,
        successor_content=neutral.current_reference[:, None] + changed_delta,
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
