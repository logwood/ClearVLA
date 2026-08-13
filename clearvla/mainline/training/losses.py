"""Single owner of active ClearVLA objectives and loss accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import ActionSupervision, ObservableHistory
from ..model.action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from ..model.observation_contract import (
    ObservationEvidence,
    PatchFlowField,
    _coordinate_grid,
)
from ..model.policy import PolicyStepOutput
from ..model.types import FutureObjectDynamics, ObjectTopTrainingTargets
from ..v120_core.gauges import masked_candidate_center

_STANDARD_GAMMA = getattr(torch, "_standard_gamma")


@dataclass(frozen=True)
class FlowMatchingState:
    time: Tensor  # [B]
    source_physical_noise: Tensor  # [B,T,Aphysical]
    noisy_physical: Tensor  # [B,T,Aphysical]
    target_physical: Tensor  # [B,T,Aphysical]
    target_physical_velocity: Tensor  # [B,T,Aphysical]


def balanced_event_row_weights(event_mask: Tensor, horizon_weight: Tensor) -> Tensor:
    """Return audit-only inverse-root-frequency gripper weights.

    Information-balanced sampling operates at the *window* level.  A selected
    event window still contains mostly hold rows.  Schema 20 keeps the exact
    V120 action and decoded objectives formal, so this geometry is serialized
    only as a counterfactual audit.  Root-frequency balancing is intentionally
    milder than inverse-frequency weighting, and the final normalization makes
    ``mean(weight * horizon_weight)`` exactly match the original horizon
    budget.  It is never registered in ``loss_contrib_*``.
    """

    if event_mask.ndim != 2 or horizon_weight.ndim != 1:
        raise ValueError("event balancing requires [B,T] mask and [T] horizon weights")
    if int(event_mask.shape[1]) != int(horizon_weight.shape[0]):
        raise ValueError("event rows and horizon weights do not align")
    event = event_mask.to(device=horizon_weight.device, dtype=torch.float32)
    hold = 1.0 - event
    event_count = event.sum()
    hold_count = hold.sum()
    total = event_count + hold_count
    event_scale = torch.sqrt(total / (2.0 * event_count.clamp_min(1.0)))
    hold_scale = torch.sqrt(total / (2.0 * hold_count.clamp_min(1.0)))
    raw = event * event_scale + hold * hold_scale
    both_classes = (event_count > 0.0) & (hold_count > 0.0)
    raw = torch.where(both_classes, raw, torch.ones_like(raw))
    step = horizon_weight.float()[None]
    denominator = step.expand_as(raw).sum().clamp_min(1.0)
    normalization = (raw * step).sum() / denominator
    balanced = raw / normalization.clamp_min(1e-8)
    # A batch with only one class has no imbalance to repair and must retain
    # the original gripper objective bit-exactly.
    return torch.where(both_classes, balanced, torch.ones_like(balanced))


def event_positive_class_weights(event_target: Tensor, *, positive_boost: float) -> Tensor:
    """Return the V120 additive event boost (hold=1, positive=1+boost)."""

    if float(positive_boost) < 0.0:
        raise ValueError("event positive boost must be non-negative")
    return 1.0 + (event_target != 0).to(dtype=torch.float32) * float(positive_boost)


def sample_flow_matching(
    target: Tensor,
    *,
    action_state: Tensor,
    codec: PhysicalActionFieldCodec,
    distribution: str,
    generator: torch.Generator | None = None,
) -> FlowMatchingState:
    if target.ndim != 3:
        raise ValueError("flow-matching native action target must be [B,T,A]")
    if distribution != "v120_mirrored_beta_1_5_1":
        raise ValueError(
            "the mainline supports only the mirrored V120 beta_1_5_1 flow time"
        )
    # ``_standard_gamma`` accepts a generator, unlike Distribution.sample,
    # and keeps training/resume RNG ownership explicit.
    concentration = target.new_full((target.shape[0],), 1.5, dtype=torch.float32)
    numerator = _STANDARD_GAMMA(concentration, generator=generator)
    denominator = numerator + _STANDARD_GAMMA(
        target.new_ones(target.shape[0], dtype=torch.float32),
        generator=generator,
    )
    # V120 uses ``t_v120=1`` at noise and samples Beta(1.5, 1.0), with a
    # public endpoint contraction to [0.001, 1).  The independent mainline
    # uses the opposite chart (0=noise, 1=clean), so the density must be
    # mirrored as well as the bridge algebra.  Keep the two owned Gamma draws
    # in their existing order so resume RNG ownership does not change.
    v120_time = numerator / denominator.clamp_min(1e-8)
    v120_time = v120_time * 0.999 + 0.001
    time = 1.0 - v120_time
    target_physical = codec.encode(target, action_state)
    noise = codec.sample_noise(
        int(target.shape[0]),
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    alpha = time.to(dtype=target.dtype)[:, None, None]
    noisy = (1.0 - alpha) * noise + alpha * target_physical
    return FlowMatchingState(
        time=time,
        source_physical_noise=noise,
        noisy_physical=noisy,
        target_physical=target_physical,
        target_physical_velocity=target_physical - noise,
    )


def _warp(value: Tensor, displacement: Tensor) -> Tensor:
    """Sample ``value`` using a true normalized-coordinate displacement."""

    if value.ndim != 5 or displacement.ndim != 5:
        raise ValueError("flow warp requires [B,C,F,H,W] and [B,C,2,H,W]")
    batch, cameras, channels, height, width = value.shape
    if tuple(displacement.shape) != (batch, cameras, 2, height, width):
        raise ValueError("flow and feature charts do not align")
    base = _coordinate_grid(
        height,
        width,
        device=value.device,
        dtype=torch.float32,
    )[None].expand(batch * cameras, -1, -1, -1)
    flow = displacement.reshape(batch * cameras, 2, height, width)
    grid = base + flow.float().permute(0, 2, 3, 1)
    sampled = F.grid_sample(
        value.reshape(batch * cameras, channels, height, width),
        grid.to(dtype=value.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(batch, cameras, channels, height, width)


def _flow_pair_terms(
    *,
    flow: PatchFlowField,
    previous: Tensor,
    current: Tensor,
    previous_literal_rgb: Tensor,
    current_literal_rgb: Tensor,
) -> dict[str, Tensor]:
    """One adjacent causal-flow objective in shared physical units."""

    if flow.backward is None:
        raise ValueError("flow geometry objectives require the training-only backward field")
    height, width = flow.forward.shape[-2:]

    def rgb_chart(value: Tensor) -> Tensor:
        batch, cameras, channels = value.shape[:3]
        if tuple(value.shape[-2:]) == (height, width):
            return value
        resized = F.interpolate(
            value.reshape(batch * cameras, channels, *value.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        return resized.reshape(batch, cameras, channels, height, width)

    previous_rgb = rgb_chart(previous_literal_rgb)
    current_rgb = rgb_chart(current_literal_rgb)
    # ``forward`` is previous->current motion indexed on current cells.  A
    # backward sampling warp reconstructs each current cell from
    # ``current_coordinate - forward`` in the previous chart.
    warped_previous = _warp(previous, -flow.forward)
    channels = float(current.shape[2]) ** 0.5
    feature_residual = (
        torch.linalg.vector_norm(
            warped_previous.float() - current.float(),
            dim=2,
        )
        / channels
    )
    feature_identity_residual = (
        torch.linalg.vector_norm(
            current.float() - previous.float(),
            dim=2,
        )
        / channels
    )
    warped_previous_rgb = _warp(previous_rgb, -flow.forward)
    rgb_channels = float(current_rgb.shape[2]) ** 0.5
    photometric_residual = (
        torch.linalg.vector_norm(
            warped_previous_rgb.float() - current_rgb.float(),
            dim=2,
        )
        / rgb_channels
    )
    photometric_identity_residual = (
        torch.linalg.vector_norm(
            current_rgb.float() - previous_rgb.float(),
            dim=2,
        )
        / rgb_channels
    )
    feature_warp = feature_residual.mean()
    photometric_warp = photometric_residual.mean()
    # Feature matching keeps the learned spatial representation coherent;
    # literal RGB prevents the shared trainable encoder from lowering the
    # entire geometry objective by erasing temporal detail.  This reuses the
    # existing outer flow-warp budget rather than adding a non-zero-flow quota.
    warp = 0.5 * (feature_warp + photometric_warp)
    # Learned flow must beat the zero-flow explanation where observable change
    # exists, but a truly static patch is allowed to retain identity flow.
    motion_weight = (
        photometric_identity_residual.detach() / (photometric_identity_residual.detach() + 0.05)
    ).clamp(0, 1)
    advantage = (
        0.05
        * F.softplus((photometric_residual - photometric_identity_residual.detach()) / 0.05)
        * motion_weight
    ).sum() / motion_weight.sum().clamp_min(1.0)
    static_weight = 1.0 - motion_weight
    static_identity = (
        torch.linalg.vector_norm(flow.forward.float(), dim=2) * static_weight
    ).sum() / static_weight.sum().clamp_min(1.0)
    # Backward flow is indexed on the previous chart.  Pull it onto the
    # current chart through the same inverse address before checking the two
    # directed transports cancel.
    sampled_backward = _warp(flow.backward, -flow.forward)
    # ``sqrt(sum(x**2))`` has an undefined derivative at the exact zero-flow
    # initialization.  ``vector_norm`` defines the zero subgradient and keeps
    # the identity solution legal without a numerical epsilon loss floor.
    cycle = torch.linalg.vector_norm(
        flow.forward.float() + sampled_backward.float(),
        dim=2,
    ).mean()
    dx = flow.forward[..., :, 1:] - flow.forward[..., :, :-1]
    dy = flow.forward[..., 1:, :] - flow.forward[..., :-1, :]
    feature_dx = current[..., :, 1:] - current[..., :, :-1]
    feature_dy = current[..., 1:, :] - current[..., :-1, :]
    smooth = 0.5 * (
        (dx.float().abs() * torch.exp(-feature_dx.float().abs().mean(dim=2, keepdim=True))).mean()
        + (dy.float().abs() * torch.exp(-feature_dy.float().abs().mean(dim=2, keepdim=True))).mean()
    )
    uncertainty_target = photometric_residual.detach().mean(dim=(-2, -1), keepdim=True).unsqueeze(2)
    uncertainty = F.smooth_l1_loss(
        flow.uncertainty.float(),
        uncertainty_target.expand_as(flow.uncertainty),
    )
    sequence_rows = []
    for estimate in flow.refinement_sequence:
        sequence_rows.append(
            (
                torch.linalg.vector_norm(
                    _warp(previous_rgb, -estimate).float() - current_rgb.float(),
                    dim=2,
                )
                / rgb_channels
            ).mean()
        )
    refinement = torch.stack(sequence_rows).mean()
    return {
        "flow_warp": warp,
        "flow_feature_warp": feature_warp,
        "flow_feature_zero_warp": feature_identity_residual.mean(),
        "flow_photometric_warp": photometric_warp,
        "flow_photometric_zero_warp": photometric_identity_residual.mean(),
        "flow_identity_advantage": advantage,
        "flow_static_identity": static_identity,
        "flow_cycle": cycle,
        "flow_smoothness": smooth,
        "flow_uncertainty": uncertainty,
        "flow_refinement_sequence": refinement,
    }


def flow_geometry_terms(evidence: ObservationEvidence) -> dict[str, Tensor]:
    """Average both -8->-4 and -4->0 observable geometry objectives."""

    evidence.validate()
    # The active restored observation contract publishes the source-resolved
    # V120 ledger.  The preserved pre-extraction observation prototype does
    # not own that optional field and remains useful for isolated geometry
    # regressions; absence means "compute the explicit fallback below", not
    # an invalid runtime contract.
    native_flow_losses = getattr(evidence, "native_flow_losses", None)
    if native_flow_losses is not None:
        native = native_flow_losses
        # Preserve the source-resolved V120 arithmetic instead of rebuilding
        # another photometric/feature objective from compatibility charts.
        # The V120 SEA-RAFT core already batches both adjacent pairs and both
        # directions in every scalar below.
        return {
            "flow_warp": native["flow_jepa_warp_loss"],
            "flow_identity_advantage": native[
                "flow_jepa_identity_advantage_loss"
            ],
            "flow_static_identity": native["flow_jepa_static_identity_loss"],
            "flow_cycle": native["flow_jepa_cycle_loss"],
            "flow_smoothness": native["flow_jepa_smoothness_loss"],
            "flow_uncertainty": native["flow_jepa_uncertainty_nll"],
            "flow_refinement_sequence": native[
                "flow_jepa_refinement_sequence_loss"
            ],
        }
    recent = _flow_pair_terms(
        flow=evidence.flow,
        previous=evidence.previous_detail_features,
        current=evidence.detail_features,
        previous_literal_rgb=evidence.previous_literal_rgb,
        current_literal_rgb=evidence.literal_rgb,
    )
    if evidence.earlier_detail_features is None:
        raise ValueError(
            "explicit flow fallback requires real t=-8 detail features"
        )
    earlier = _flow_pair_terms(
        flow=evidence.earlier_flow,
        previous=evidence.earlier_detail_features,
        current=evidence.previous_detail_features,
        previous_literal_rgb=evidence.earlier_literal_rgb,
        current_literal_rgb=evidence.previous_literal_rgb,
    )
    result = {
        name: 0.5 * (recent[name] + earlier[name])
        for name in recent
    }
    # Keep the two physical intervals separately visible without changing the
    # outer geometry budget.
    result.update(
        {
            f"flow_recent_{name.removeprefix('flow_')}": value.detach()
            for name, value in recent.items()
        }
    )
    result.update(
        {
            f"flow_earlier_{name.removeprefix('flow_')}": value.detach()
            for name, value in earlier.items()
        }
    )
    return result


def future_dynamics_terms(
    prediction: FutureObjectDynamics,
    target: FutureObjectDynamics,
    *,
    current_loss_support: Tensor,
    collect_diagnostics: bool = False,
) -> dict[str, Tensor]:
    prediction.validate()
    target.validate()
    batch, intervals, objects = prediction.semantic_delta.shape[:3]
    cameras = int(prediction.transport_mean.shape[3])
    if tuple(current_loss_support.shape) != (batch, objects, cameras, 1):
        raise ValueError(
            "current loss support must be [B,K,C,1] and align with future dynamics"
        )
    camera_validity = current_loss_support.detach().float()[:, None].expand(
        -1, intervals, -1, -1, -1
    ).clamp(0.0, 1.0)
    object_validity = camera_validity.amax(dim=3)

    def masked(error: Tensor, weight: Tensor) -> Tensor:
        expanded = weight
        while expanded.ndim < error.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(error)
        return (error * expanded).sum() / expanded.sum().clamp_min(1.0)

    def row_loss(
        prediction_value: Tensor,
        target_value: Tensor,
        *,
        scale_floored: bool,
    ) -> Tensor:
        prediction_f = prediction_value.float()
        target_f = target_value.detach().float()
        raw = F.smooth_l1_loss(
            prediction_f,
            target_f,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        if not scale_floored:
            return raw
        target_rms = target_f.square().mean(dim=-1, keepdim=True).sqrt()
        scale_floor = (
            0.25 * target_rms.mean(dim=(0, 2), keepdim=True)
        ).clamp_min(1e-3)
        scale = torch.sqrt(target_rms.square() + scale_floor.square())
        normalized = F.smooth_l1_loss(
            prediction_f / scale,
            target_f / scale,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        prediction_direction = prediction_f / torch.sqrt(
            prediction_f.square().mean(dim=-1, keepdim=True)
            + scale_floor.square()
        )
        target_direction = target_f / torch.sqrt(
            target_f.square().mean(dim=-1, keepdim=True)
            + scale_floor.square()
        )
        # The historical cosine-like expression was positive even when the
        # prediction exactly equalled the target because the variance floor
        # makes each smoothed direction shorter than unit length.  Compare the
        # two smoothed directions directly so the supervised optimum is an
        # attainable exact zero while retaining the same bounded denominator.
        direction = 0.5 * (
            prediction_direction - target_direction
        ).square().mean(dim=-1, keepdim=True)
        return raw + normalized + 0.10 * direction

    successor_error = row_loss(
        prediction.successor_content,
        target.successor_content,
        scale_floored=False,
    )
    successor = masked(successor_error, object_validity)
    semantic_delta_error = row_loss(
        prediction.semantic_delta,
        target.semantic_delta,
        scale_floored=True,
    )
    semantic_delta = masked(semantic_delta_error, object_validity)
    transport_error = row_loss(
        prediction.transport_mean,
        target.transport_mean,
        scale_floored=False,
    )
    transport = masked(transport_error, camera_validity)
    covariance_error = row_loss(
        prediction.transport_covariance,
        target.transport_covariance,
        scale_floored=False,
    )
    covariance = masked(covariance_error, camera_validity)
    visibility_error = row_loss(
        prediction.visibility,
        target.visibility,
        scale_floored=False,
    )
    visibility = masked(visibility_error, object_validity)
    persistence_error = row_loss(
        prediction.persistence,
        target.persistence,
        scale_floored=False,
    )
    persistence = masked(persistence_error, object_validity)
    uncertainty_error = row_loss(
        prediction.uncertainty,
        target.uncertainty,
        scale_floored=False,
    )
    uncertainty = masked(uncertainty_error, object_validity)
    semantic_transition_prediction = (
        prediction.semantic_delta.float()[:, 1:]
        - prediction.semantic_delta.float()[:, :-1]
    )
    semantic_transition_target = (
        target.semantic_delta.detach().float()[:, 1:]
        - target.semantic_delta.detach().float()[:, :-1]
    )
    transition_error = row_loss(
        semantic_transition_prediction,
        semantic_transition_target,
        scale_floored=True,
    )
    transition_validity = torch.minimum(
        object_validity[:, 1:],
        object_validity[:, :-1],
    )
    transition = masked(transition_error, transition_validity)
    total = (
        0.30 * successor
        + 0.25 * semantic_delta
        + 0.15 * transport
        + 0.05 * covariance
        + 0.08 * visibility
        + 0.07 * persistence
        + 0.10 * uncertainty
    )
    terms = {
        "future_dynamics": total,
        "future_successor": successor,
        "future_semantic_delta": semantic_delta,
        "future_transport": transport,
        "future_covariance": covariance,
        "future_visibility": visibility,
        "future_persistence": persistence,
        "future_uncertainty": uncertainty,
        "future_transition": transition,
    }
    if collect_diagnostics:
        terms["future_current_loss_support"] = camera_validity.mean().detach()
        terms["future_prediction_selector_validity"] = (
            prediction.future_selector_validity.detach().float().mean()
        )
        terms["future_target_selector_validity"] = (
            target.future_selector_validity.detach().float().mean()
        )
        for index in range(prediction.intervals):
            interval_slice = slice(index, index + 1)
            interval_validity = object_validity[:, interval_slice]
            terms[f"future_interval_{index}_successor"] = masked(
                successor_error[:, interval_slice], interval_validity
            ).detach()
            terms[f"future_interval_{index}_semantic_delta"] = masked(
                semantic_delta_error[:, interval_slice], interval_validity
            ).detach()
            terms[f"future_interval_{index}_transport"] = masked(
                transport_error[:, interval_slice],
                camera_validity[:, interval_slice],
            ).detach()
    return terms


def execution_value_terms(
    config: ExperimentConfig,
    codec: PhysicalActionFieldCodec,
    output: PolicyStepOutput,
    flow_state: FlowMatchingState,
) -> dict[str, Tensor]:
    """Restore V120's centered physical candidate-value supervision.

    Candidate action predictions are detached targets.  Only the controller's
    typed ``[arm, gripper]`` value field receives gradients.  Near-tie rows are
    down-weighted by their physical spread, and common candidate offsets are
    removed before both regression and selection diagnostics.
    """

    tensors = output.bottom.decoder_tensors
    names = (
        "evidence_mmd_it_execution_candidate_value_field",
        "evidence_mmd_it_dwell_candidate_pred_velocity",
        "evidence_mmd_it_execution_candidate_value_mask",
        "evidence_mmd_it_execution_baseline_pred_velocity",
    )
    missing = [name for name in names if name not in tensors]
    if missing:
        raise ValueError(
            "restored V120 execution supervision is missing " + ", ".join(missing)
        )
    predicted = tensors[names[0]].float()
    candidates = tensors[names[1]].detach().float()
    valid = tensors[names[2]].detach().bool()
    baseline = tensors[names[3]].detach().float()
    target = flow_state.target_physical_velocity.detach().float()
    if (
        predicted.ndim != 5
        or candidates.ndim != 5
        or valid.ndim != 3
        or baseline.ndim != 4
        or int(predicted.shape[-1]) != 2
    ):
        raise ValueError("V120 execution candidate tensors have invalid ranks")
    if tuple(predicted.shape[:4]) != tuple(candidates.shape[:4]):
        raise ValueError("execution value field and candidate predictions are misaligned")
    if tuple(valid.shape) != tuple(candidates.shape[:3]):
        raise ValueError("execution candidate validity has the wrong shape")
    if tuple(target.shape) != (
        int(candidates.shape[0]),
        int(candidates.shape[3]),
        int(candidates.shape[4]),
    ):
        raise ValueError("execution candidate target has the wrong physical shape")
    if tuple(baseline.shape) != (
        int(candidates.shape[0]),
        int(candidates.shape[1]),
        int(candidates.shape[3]),
        int(candidates.shape[4]),
    ):
        raise ValueError("execution baseline has the wrong physical shape")

    batch, blocks, candidate_count, horizon, physical = candidates.shape
    residual = candidates - target[:, None, None]
    flat = residual.reshape(batch * blocks * candidate_count, horizon, physical)
    parts = codec.split(flat)
    arm_error = 0.5 * (
        parts.arm_absolute.square() + parts.arm_delta.square()
    ).sum(dim=-1) / float(codec.arm_dim)
    gripper_error = parts.gripper_field.square().mean(dim=-1)
    target_value = torch.stack((arm_error, gripper_error), dim=-1).reshape(
        batch,
        blocks,
        candidate_count,
        horizon,
        2,
    )
    target_centered, _ = masked_candidate_center(
        target_value,
        valid,
        candidate_dim=2,
    )
    predicted_centered, predicted_mean = masked_candidate_center(
        predicted,
        valid,
        candidate_dim=2,
    )
    valid_field = valid[..., None, None].expand_as(predicted)
    component_weight = predicted.new_tensor([float(codec.arm_dim), 1.0]) / float(
        codec.arm_dim + 1
    )
    physical_weight = valid_field.float() * component_weight[None, None, None, None]
    active = valid.float().sum(dim=2) > 1.0
    active_float = active.float()
    active_denominator = active_float.sum().clamp_min(1.0)
    row_denominator = (
        valid[..., None]
        .expand(-1, -1, -1, horizon)
        .float()
        .sum(dim=(2, 3))
        .clamp_min(1.0)
    )
    target_spread = torch.sqrt(
        (target_centered.square() * physical_weight).sum(dim=(2, 3, 4))
        / row_denominator
    )
    reliability_scale = (
        (target_spread.detach() * active_float).sum() / active_denominator
    ).clamp_min(1e-6)
    reliability = (
        target_spread / (target_spread + reliability_scale)
    ) * active_float
    reliability_denominator = reliability.sum().clamp_min(1e-6)
    normalization_scale = torch.maximum(
        target_spread.detach(),
        reliability_scale.detach(),
    )
    normalized_target = target_centered / normalization_scale[..., None, None, None]
    value_field = F.smooth_l1_loss(
        predicted_centered,
        normalized_target,
        reduction="none",
        beta=float(config.objectives.execution_value_huber_delta),
    ) * physical_weight
    value_rows = value_field.sum(dim=(2, 3, 4)) / row_denominator
    value_loss = (value_rows * reliability).sum() / reliability_denominator

    predicted_scalar = (
        (predicted * component_weight[None, None, None, None])
        .sum(dim=-1)
        .mean(dim=-1)
    )
    target_scalar = (
        (normalized_target * component_weight[None, None, None, None])
        .sum(dim=-1)
        .mean(dim=-1)
    )
    invalid_max = torch.finfo(predicted_scalar.dtype).max
    predicted_best = predicted_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
    target_best = target_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
    decision_accuracy = (
        (predicted_best == target_best).float() * active_float
    ).sum() / active_denominator
    target_difference = target_scalar[..., :, None] - target_scalar[..., None, :]
    predicted_difference = (
        predicted_scalar[..., :, None] - predicted_scalar[..., None, :]
    )
    pair_mask = valid[..., :, None] & valid[..., None, :]
    pair_mask = pair_mask & torch.triu(
        torch.ones(candidate_count, candidate_count, device=valid.device, dtype=torch.bool),
        diagonal=1,
    )
    informative = pair_mask & target_difference.ne(0.0)
    pairwise_accuracy = (
        (predicted_difference * target_difference > 0.0).float()
        * informative.float()
    ).sum() / informative.float().sum().clamp_min(1.0)
    correlation = (
        predicted_centered * normalized_target * physical_weight
    ).sum() / (
        (predicted_centered.square() * physical_weight).sum().sqrt()
        * (normalized_target.square() * physical_weight).sum().sqrt()
    ).clamp_min(1e-8)
    predicted_rms = (
        (predicted.square() * physical_weight).sum()
        / row_denominator.sum().clamp_min(1.0)
    ).sqrt()
    active_common = active_float[..., None, None, None]
    predicted_common_rms = (
        (
            predicted_mean.square()
            * active_common
            * component_weight[None, None, None, None]
        ).sum()
        / (active_common.sum() * horizon).clamp_min(1.0)
    ).sqrt()
    predicted_standardized_spread = torch.sqrt(
        (predicted_centered.square() * physical_weight).sum(dim=(2, 3, 4))
        / row_denominator
    )
    predicted_spread = predicted_standardized_spread * normalization_scale
    selected_spread = target_spread[active]
    if int(selected_spread.numel()) > 0:
        spread_p25, spread_p50, spread_p75 = (
            torch.quantile(selected_spread, quantile)
            for quantile in (0.25, 0.50, 0.75)
        )
    else:
        spread_p25 = spread_p50 = spread_p75 = target_spread.new_zeros(())
    terminal_valid = valid[..., -1] & active
    operation_scalar = target_scalar[..., :-1].masked_fill(
        ~valid[..., :-1], invalid_max
    )
    terminal_target_margin = target_scalar[..., -1] - operation_scalar.amin(dim=-1)
    predicted_operation = predicted_scalar[..., :-1].masked_fill(
        ~valid[..., :-1], invalid_max
    )
    terminal_predicted_margin = (
        predicted_scalar[..., -1] - predicted_operation.amin(dim=-1)
    )
    terminal_denominator = terminal_valid.float().sum().clamp_min(1.0)
    execution_cost = tensors.get("evidence_mmd_it_execution_cost")
    if not isinstance(execution_cost, Tensor) or execution_cost.ndim != 0:
        execution_cost = value_loss.new_zeros(())
    return {
        "execution_value": value_loss,
        "execution_cost_audit": execution_cost.detach().float(),
        "execution_value_reliability_scale": reliability_scale.detach(),
        "execution_value_reliability": (
            reliability.sum() / active_denominator
        ).detach(),
        "execution_value_target_spread": (
            (target_spread * active_float).sum() / active_denominator
        ).detach(),
        "execution_value_predicted_spread": (
            (predicted_spread * active_float).sum() / active_denominator
        ).detach(),
        "execution_value_predicted_standardized_spread": (
            (predicted_standardized_spread * active_float).sum()
            / active_denominator
        ).detach(),
        "execution_value_target_spread_p25": spread_p25.detach(),
        "execution_value_target_spread_p50": spread_p50.detach(),
        "execution_value_target_spread_p75": spread_p75.detach(),
        "execution_value_correlation": correlation.detach(),
        "execution_value_pairwise_accuracy": pairwise_accuracy.detach(),
        "execution_value_decision_accuracy": decision_accuracy.detach(),
        "execution_value_common_mode_ratio": (
            predicted_common_rms / predicted_rms.clamp_min(1e-8)
        ).detach(),
        "execution_candidate_coverage": valid.float().mean().detach(),
        "execution_terminal_identity_error": (
            candidates[:, :, -1] - baseline
        ).square().mean().sqrt().detach(),
        "execution_terminal_target_cost_margin": (
            (terminal_target_margin * terminal_valid.float()).sum()
            / terminal_denominator
        ).detach(),
        "execution_terminal_predicted_cost_margin": (
            (terminal_predicted_margin * terminal_valid.float()).sum()
            / terminal_denominator
        ).detach(),
        "execution_terminal_target_preferred_fraction": (
            ((terminal_target_margin < 0.0) & terminal_valid).float().sum()
            / terminal_denominator
        ).detach(),
    }


def action_terms(
    config: ExperimentConfig,
    codec: PhysicalActionFieldCodec,
    output: PolicyStepOutput,
    target: ActionSupervision,
    history: ObservableHistory,
    flow_state: FlowMatchingState,
) -> dict[str, Tensor]:
    objective = config.objectives
    raw_grip = target.raw_units[..., -1].float()
    raw_boundary = torch.cat(
        (target.current_raw_units[:, None, -1:].float(), raw_grip[:, :-1, None]),
        dim=1,
    )[..., 0]
    raw_grip_delta = raw_grip - raw_boundary
    event_target = torch.zeros_like(raw_grip_delta, dtype=torch.long)
    event_target = torch.where(
        raw_grip_delta <= -float(objective.gripper_event_threshold),
        torch.ones_like(event_target),
        event_target,
    )
    event_target = torch.where(
        raw_grip_delta >= float(objective.gripper_event_threshold),
        torch.full_like(event_target, 2),
        event_target,
    )
    event_mask = (event_target != 0).to(dtype=torch.float32)
    prediction = output.bottom.physical_velocity.float()
    velocity_target = flow_state.target_physical_velocity.detach().float()
    residual = prediction - velocity_target
    residual_parts = codec.split(residual)
    arm_error = 0.5 * (
        residual_parts.arm_absolute.square() + residual_parts.arm_delta.square()
    )
    gripper_error_unweighted = residual_parts.gripper_field.square().mean(dim=-1)
    horizon_weight = anchor_horizon_weights(
        horizon=config.dimensions.action_horizon,
        tail_emphasis=objective.horizon_tail_emphasis,
        first_step_protection=objective.horizon_first_step_protection,
        device=prediction.device,
    )
    step_weight = horizon_weight[None]
    event_row_weight = balanced_event_row_weights(event_mask, horizon_weight)
    gripper_error = gripper_error_unweighted * event_row_weight
    physical_error_unweighted = (
        arm_error.sum(dim=-1) + gripper_error_unweighted
    ) / float(codec.arm_dim + 1)
    physical_error = (arm_error.sum(dim=-1) + gripper_error) / float(codec.arm_dim + 1)
    flow = (physical_error * step_weight).mean()
    # V120 used no event-row boost in its physical flow objective.  Serialize
    # the balanced counterfactual under an explicit audit name; it is not sent
    # to backward and cannot be mistaken for the recovered formal geometry.
    flow_v120_comparable = (physical_error_unweighted * step_weight).mean()
    arm = (arm_error.mean(dim=-1) * step_weight).mean()
    grip = (gripper_error * step_weight).mean()
    grip_unweighted = (gripper_error_unweighted * step_weight).mean()
    grip_value = (residual_parts.gripper_field[..., 0].square() * step_weight).mean()
    grip_value_balanced = (
        residual_parts.gripper_field[..., 0].square() * event_row_weight * step_weight
    ).mean()
    grip_delta_unweighted = (
        residual_parts.gripper_field[..., 1].square() * step_weight
    ).mean()
    grip_delta = (
        residual_parts.gripper_field[..., 1].square() * event_row_weight * step_weight
    ).mean()
    grip_auxiliary_unweighted = (
        residual_parts.gripper_field[..., 2:].square().mean(dim=-1)
        * step_weight
    ).mean()
    grip_auxiliary = (
        residual_parts.gripper_field[..., 2:].square().mean(dim=-1)
        * event_row_weight
        * step_weight
    ).mean()
    uniform_flow = residual.square().mean()
    native_arm, _ = codec.project_arm_tangent(residual[..., : 2 * codec.arm_dim])
    native_grip = residual_parts.gripper_field[..., 0]
    native_error = (
        native_arm.float().square().sum(dim=-1) + native_grip.float().square()
    ) / float(codec.arm_dim + 1)
    native_flow = (native_error * step_weight).mean()
    remaining = (1.0 - flow_state.time.float())[:, None, None]
    clean_physical = flow_state.noisy_physical.float() + remaining * prediction
    decoded = codec.decode(clean_physical, history.action_state.float())
    decoded_element_error = F.smooth_l1_loss(
        decoded,
        target.normalized.float(),
        reduction="none",
    )
    decoded_gripper_error = decoded_element_error[..., -1]
    decoded_rows = (
        decoded_element_error[..., : codec.arm_dim].sum(dim=-1)
        + decoded_gripper_error * event_row_weight
    ) / float(codec.arm_dim + 1)
    decoded_action = (decoded_rows * step_weight).mean()
    decoded_action_v120_comparable = (
        decoded_element_error.mean(dim=-1) * step_weight
    ).mean()
    boundary = torch.cat(
        (
            history.action_state[:, None].float(),
            target.normalized[:, :-1].float(),
        ),
        dim=1,
    )
    delta = target.normalized.float() - boundary
    predicted_boundary = torch.cat(
        (history.action_state[:, None].float(), decoded[:, :-1]), dim=1
    )
    predicted_delta = decoded - predicted_boundary
    smooth_delta_rows = F.smooth_l1_loss(
        predicted_delta,
        delta,
        reduction="none",
    ).mean(dim=-1)
    smooth_delta = (smooth_delta_rows * step_weight).mean()
    physical_delta_rows = codec.delta_consistency(
        clean_physical,
        history.action_state.float(),
        decoded,
    )
    physical_delta_consistency = (physical_delta_rows * step_weight).mean()

    event_logits = output.bottom.event_logits.float().reshape(-1, 3)
    flat_event = event_target.reshape(-1)
    event_ce = F.cross_entropy(event_logits, flat_event, reduction="none")
    event_pt = torch.exp(-event_ce.detach()).clamp(min=1e-6, max=1.0)
    event_ce = (1.0 - event_pt).pow(float(objective.event_focal_gamma)) * event_ce
    event_positive = event_positive_class_weights(
        flat_event,
        positive_boost=objective.event_positive_boost,
    ).to(dtype=event_ce.dtype)
    event = (
        event_ce
        * event_positive
        * step_weight.expand(int(target.batch), -1).reshape(-1)
    ).mean()
    target_parts = codec.split(flow_state.target_physical.detach())
    motion_target = (
        target_parts.arm_delta.float().norm(dim=-1)
        >= float(objective.arm_motion_threshold)
    ).float()
    motion_rows = F.binary_cross_entropy_with_logits(
        output.bottom.motion_logits.float(), motion_target, reduction="none"
    )
    motion = (motion_rows * step_weight).mean()
    event_mask = event_mask.to(dtype=gripper_error_unweighted.dtype)
    hold_mask = 1.0 - event_mask
    event_denominator = (event_mask * step_weight).sum().clamp_min(1.0)
    hold_denominator = (hold_mask * step_weight).sum().clamp_min(1.0)
    event_gripper_flow = (
        gripper_error_unweighted * event_mask * step_weight
    ).sum() / event_denominator
    hold_gripper_flow = (
        gripper_error_unweighted * hold_mask * step_weight
    ).sum() / hold_denominator
    event_decoded_gripper = (
        decoded_gripper_error * event_mask * step_weight
    ).sum() / event_denominator
    hold_decoded_gripper = (
        decoded_gripper_error * hold_mask * step_weight
    ).sum() / hold_denominator
    event_count = event_mask.sum()
    hold_count = hold_mask.sum()
    event_row_weight_mean = (event_row_weight * event_mask).sum() / event_count.clamp_min(1.0)
    hold_row_weight_mean = (event_row_weight * hold_mask).sum() / hold_count.clamp_min(1.0)
    predicted_event = output.bottom.event_logits.detach().argmax(dim=-1)
    event_positive_target = event_target != 0
    event_positive_prediction = predicted_event != 0
    true_positive = (event_positive_target & event_positive_prediction).float().sum()
    false_positive = ((~event_positive_target) & event_positive_prediction).float().sum()
    false_negative = (event_positive_target & (~event_positive_prediction)).float().sum()
    event_precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    event_recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    event_f1 = 2.0 * event_precision * event_recall / (
        event_precision + event_recall
    ).clamp_min(1e-8)
    predicted_motion = torch.sigmoid(output.bottom.motion_logits.detach().float()) >= 0.5
    target_motion = motion_target >= 0.5
    motion_true_positive = (predicted_motion & target_motion).float().sum()
    motion_false_positive = (predicted_motion & (~target_motion)).float().sum()
    motion_false_negative = ((~predicted_motion) & target_motion).float().sum()
    motion_precision = motion_true_positive / (
        motion_true_positive + motion_false_positive
    ).clamp_min(1.0)
    motion_recall = motion_true_positive / (
        motion_true_positive + motion_false_negative
    ).clamp_min(1.0)
    band_metrics: dict[str, Tensor] = {}
    start = 0
    for end in (4, 12, 24):
        # Preserve the historical metric as an unweighted diagnostic.  The
        # counterfactual event-balanced geometry is reported as audit-only.
        band_metrics[f"action_flow_band_{start + 1}_{end}"] = physical_error_unweighted[
            :, start:end
        ].mean()
        band_metrics[
            f"action_flow_event_balanced_audit_band_{start + 1}_{end}"
        ] = physical_error[:, start:end].mean()
        band_metrics[f"action_horizon_weight_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].mean()
        band_metrics[f"action_horizon_mass_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].sum() / horizon_weight.sum()
        start = end
    return {
        **band_metrics,
        # The formal objective is the exact V120 physical metric.  Event-row
        # balancing remains an audit, not an alternative training geometry.
        "action_flow": flow_v120_comparable,
        "action_flow_v120_comparable": flow_v120_comparable,
        "action_flow_event_balance_delta": flow - flow_v120_comparable,
        "action_flow_event_balanced_audit": flow,
        "action_flow_uniform_field_mse": uniform_flow,
        "action_flow_native": native_flow,
        "action_arm_flow": arm,
        "action_gripper_flow": grip_unweighted,
        "action_gripper_flow_v120_comparable": grip_unweighted,
        "action_gripper_flow_unweighted": grip_unweighted,
        "action_gripper_flow_event_balanced_audit": grip,
        "action_gripper_value_flow": grip_value,
        "action_gripper_value_flow_unweighted": grip_value,
        "action_gripper_value_flow_event_balanced_audit": grip_value_balanced,
        "action_gripper_delta_flow": grip_delta_unweighted,
        "action_gripper_delta_flow_event_balanced_audit": grip_delta,
        "action_gripper_auxiliary_flow": grip_auxiliary_unweighted,
        "action_gripper_auxiliary_flow_event_balanced_audit": grip_auxiliary,
        "action_gripper_event_flow": event_gripper_flow,
        "action_gripper_hold_flow": hold_gripper_flow,
        "action_decoded_gripper_event": event_decoded_gripper,
        "action_decoded_gripper_hold": hold_decoded_gripper,
        "action_gripper_event_row_weight": event_row_weight_mean,
        "action_gripper_hold_row_weight": hold_row_weight_mean,
        "action_gripper_event_rate": event_mask.mean(),
        "decoded_action": decoded_action_v120_comparable,
        "decoded_action_v120_comparable": decoded_action_v120_comparable,
        "decoded_action_event_balance_delta": (
            decoded_action - decoded_action_v120_comparable
        ),
        "decoded_action_event_balanced_audit": decoded_action,
        "smooth_delta": smooth_delta,
        "physical_delta_consistency": physical_delta_consistency,
        "event": event,
        "motion": motion,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "motion_precision": motion_precision,
        "motion_recall": motion_recall,
        "action_horizon_weight_first": horizon_weight[0],
        "action_horizon_weight_tail": horizon_weight[-1],
        "action_flow_first": physical_error_unweighted[:, 0].mean(),
        "action_flow_first4": physical_error_unweighted[:, :4].mean(),
        "action_flow_first8": physical_error_unweighted[:, :8].mean(),
        "action_flow_tail": physical_error_unweighted[:, 8:].mean(),
        "action_flow_event_balanced_audit_first": physical_error[:, 0].mean(),
        "action_flow_event_balanced_audit_first8": physical_error[:, :8].mean(),
        "action_flow_event_balanced_audit_tail": physical_error[:, 8:].mean(),
    }


@dataclass(frozen=True)
class LossLedger:
    total: Tensor
    groups: dict[str, Tensor]
    contributions: dict[str, Tensor]
    terms: dict[str, Tensor]

    def validate(self) -> None:
        if self.total.ndim != 0:
            raise ValueError("total loss must be scalar")
        if set(self.groups) != {"action", "representation", "execution"}:
            raise ValueError("loss ledger has an inactive or unknown group")
        if any(value.ndim != 0 for value in self.groups.values()):
            raise ValueError("loss groups must be scalar")
        if not self.contributions or any(
            value.ndim != 0 for value in self.contributions.values()
        ):
            raise ValueError("loss contributions must be non-empty scalars")


def compose_losses(
    config: ExperimentConfig,
    *,
    policy_output: PolicyStepOutput,
    action_target: ActionSupervision,
    history: ObservableHistory,
    flow_state: FlowMatchingState,
    observation: ObservationEvidence,
    top_targets: ObjectTopTrainingTargets,
    predicted_dynamics: FutureObjectDynamics,
    action_codec: PhysicalActionFieldCodec,
    collect_diagnostics: bool = False,
) -> LossLedger:
    action = action_terms(config, action_codec, policy_output, action_target, history, flow_state)
    execution = execution_value_terms(
        config,
        action_codec,
        policy_output,
        flow_state,
    )
    geometry = flow_geometry_terms(observation)
    if top_targets.teacher_dynamics is None:
        raise ValueError("formal training requires future teacher dynamics")
    future = future_dynamics_terms(
        predicted_dynamics,
        top_targets.teacher_dynamics,
        current_loss_support=top_targets.current_loss_support,
        collect_diagnostics=collect_diagnostics,
    )
    objective = config.objectives
    action_group = (
        action["action_flow"]
        + objective.decoded_action * action["decoded_action"]
        + objective.event * action["event"]
        + objective.motion * action["motion"]
        + objective.smooth_delta * action["smooth_delta"]
        + objective.physical_delta_consistency * action["physical_delta_consistency"]
        + objective.proposal * top_targets.history_proposal_loss
    )
    intent_structure_core = (
        0.25 * top_targets.object_reconstruction_loss
        + 0.35 * top_targets.online_intent_loss
        + 0.20 * top_targets.plan_recognition_loss
        + 0.20 * top_targets.coarse_action_loss
    )
    # Restore V120's interval ledger exactly: half of the existing 0.02
    # budget supervises chronological changes between adjacent W intervals;
    # the other half owns the small G/S/recognizer/coarse scaffold.  Giving
    # the whole budget to the easy scalar scaffold both over-trained it and
    # removed W1/W2's only explicit differentiation pressure.
    interval_structure = 0.50 * future["future_transition"] + 0.50 * intent_structure_core
    representation_group = (
        objective.future_dynamics * future["future_dynamics"]
        + objective.intent_structure * interval_structure
        + objective.flow_warp * geometry["flow_warp"]
        + objective.flow_identity_advantage * geometry["flow_identity_advantage"]
        + objective.flow_static_identity * geometry["flow_static_identity"]
        + objective.flow_cycle * geometry["flow_cycle"]
        + objective.flow_smoothness * geometry["flow_smoothness"]
        + objective.flow_uncertainty * geometry["flow_uncertainty"]
        + objective.flow_refinement_sequence * geometry["flow_refinement_sequence"]
    )
    execution_group = objective.execution_value * execution["execution_value"]
    groups = {
        "action": action_group,
        "representation": representation_group,
        "execution": execution_group,
    }
    contributions = {
        "action_flow": action["action_flow"],
        "decoded_action": objective.decoded_action * action["decoded_action"],
        "event": objective.event * action["event"],
        "motion": objective.motion * action["motion"],
        "smooth_delta": objective.smooth_delta * action["smooth_delta"],
        "physical_delta_consistency": (
            objective.physical_delta_consistency
            * action["physical_delta_consistency"]
        ),
        "proposal": objective.proposal * top_targets.history_proposal_loss,
        "execution_value": execution_group,
        "future_dynamics": objective.future_dynamics * future["future_dynamics"],
        "future_transition": (
            objective.intent_structure * 0.50 * future["future_transition"]
        ),
        "object_reconstruction": (
            objective.intent_structure
            * 0.50
            * 0.25
            * top_targets.object_reconstruction_loss
        ),
        "intent_online": (
            objective.intent_structure
            * 0.50
            * 0.35
            * top_targets.online_intent_loss
        ),
        "intent_recognizer": (
            objective.intent_structure
            * 0.50
            * 0.20
            * top_targets.plan_recognition_loss
        ),
        "coarse_action": (
            objective.intent_structure
            * 0.50
            * 0.20
            * top_targets.coarse_action_loss
        ),
        "flow_warp": objective.flow_warp * geometry["flow_warp"],
        "flow_identity_advantage": (
            objective.flow_identity_advantage * geometry["flow_identity_advantage"]
        ),
        "flow_static_identity": (
            objective.flow_static_identity * geometry["flow_static_identity"]
        ),
        "flow_cycle": objective.flow_cycle * geometry["flow_cycle"],
        "flow_smoothness": objective.flow_smoothness * geometry["flow_smoothness"],
        "flow_uncertainty": objective.flow_uncertainty * geometry["flow_uncertainty"],
        "flow_refinement_sequence": (
            objective.flow_refinement_sequence * geometry["flow_refinement_sequence"]
        ),
    }
    terms = {
        **action,
        **geometry,
        **future,
        **execution,
        "intent_online": top_targets.online_intent_loss,
        "intent_recognizer": top_targets.plan_recognition_loss,
        "object_reconstruction": top_targets.object_reconstruction_loss,
        "coarse_action": top_targets.coarse_action_loss,
        "history_action_proposal": top_targets.history_proposal_loss,
    }
    ledger = LossLedger(
        total=action_group + representation_group + execution_group,
        groups=groups,
        contributions=contributions,
        terms=terms,
    )
    ledger.validate()
    return ledger


__all__ = [
    "FlowMatchingState",
    "LossLedger",
    "action_terms",
    "balanced_event_row_weights",
    "compose_losses",
    "execution_value_terms",
    "event_positive_class_weights",
    "flow_geometry_terms",
    "future_dynamics_terms",
    "sample_flow_matching",
]
