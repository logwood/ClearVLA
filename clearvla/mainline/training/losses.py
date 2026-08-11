"""Single owner of active ClearVLA objectives and loss accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import ActionSupervision, ObservableHistory
from ..model.action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from ..model.observation import ObservationEvidence, PatchFlowField, _coordinate_grid
from ..model.policy import PolicyStepOutput
from ..model.types import FutureObjectDynamics, ObjectTopTrainingTargets

_STANDARD_GAMMA = getattr(torch, "_standard_gamma")


@dataclass(frozen=True)
class FlowMatchingState:
    time: Tensor  # [B]
    source_physical_noise: Tensor  # [B,T,Aphysical]
    noisy_physical: Tensor  # [B,T,Aphysical]
    target_physical: Tensor  # [B,T,Aphysical]
    target_physical_velocity: Tensor  # [B,T,Aphysical]


def balanced_event_row_weights(event_mask: Tensor, horizon_weight: Tensor) -> Tensor:
    """Return inverse-root-frequency gripper weights with exact budget closure.

    Information-balanced sampling operates at the *window* level.  A selected
    event window still contains mostly hold rows, so an auxiliary event head
    does not make the decoded gripper trajectory care about the transition
    row.  This helper balances event/hold rows only inside the gripper part of
    the physical and decoded action objectives.  Root-frequency balancing is
    intentionally milder than inverse-frequency weighting, and the final
    normalization makes ``mean(weight * horizon_weight)`` exactly match the
    original horizon budget.  No model prediction, gate or detached gradient
    surrogate is involved.
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
    if distribution != "beta_1_5_1":
        raise ValueError("the mainline supports only beta_1_5_1 flow time")
    # ``_standard_gamma`` accepts a generator, unlike Distribution.sample,
    # and keeps training/resume RNG ownership explicit.
    concentration = target.new_full((target.shape[0],), 1.5, dtype=torch.float32)
    numerator = _STANDARD_GAMMA(concentration, generator=generator)
    denominator = numerator + _STANDARD_GAMMA(
        target.new_ones(target.shape[0], dtype=torch.float32),
        generator=generator,
    )
    time = numerator / denominator.clamp_min(1e-8)
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
    recent = _flow_pair_terms(
        flow=evidence.flow,
        previous=evidence.previous_detail_features,
        current=evidence.detail_features,
        previous_literal_rgb=evidence.previous_literal_rgb,
        current_literal_rgb=evidence.literal_rgb,
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
    collect_diagnostics: bool = False,
) -> dict[str, Tensor]:
    prediction.validate()
    target.validate()
    validity = target.validity.detach().float()
    object_validity = validity.amax(dim=3)
    reliability_weight = target.reliability.detach().float().clamp(0.0, 1.0)

    def masked(error: Tensor, weight: Tensor) -> Tensor:
        expanded = weight
        while expanded.ndim < error.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(error)
        return (error * expanded).sum() / expanded.sum().clamp_min(1.0)

    successor_error = F.smooth_l1_loss(
        prediction.successor_content.float(),
        target.successor_content.detach().float(),
        reduction="none",
    )
    successor = masked(
        successor_error,
        # Teacher-G has already blended both confident-null and high-entropy
        # associations to the current fact.  Multiplying by reliability here
        # would discount the same uncertainty twice and make W nearly
        # unsupervised on precisely the neutral rows it must learn.
        object_validity,
    )
    delta_target = target.semantic_delta.detach().float()
    delta_scale = delta_target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(0.05)
    semantic_delta_error = F.smooth_l1_loss(
        prediction.semantic_delta.float() / delta_scale,
        delta_target / delta_scale,
        reduction="none",
    )
    semantic_delta = masked(
        semantic_delta_error,
        # Semantic delta is likewise zero-centred after the confidence
        # fallback, so it remains a valid target when association is weak.
        object_validity,
    )
    transport_error = F.smooth_l1_loss(
        prediction.transport_mean.float(),
        target.transport_mean.detach().float(),
        reduction="none",
    )
    transport = masked(
        transport_error,
        # Teacher-G has already converted ambiguous geometry to identity
        # transport.  It must remain supervised, otherwise action gradients
        # can repurpose W transport as a free carrier on low-confidence rows.
        validity,
    )
    covariance = masked(
        F.smooth_l1_loss(
            prediction.transport_covariance.float(),
            target.transport_covariance.detach().float(),
            reduction="none",
        ),
        validity,
    )
    visibility = masked(
        F.smooth_l1_loss(
            prediction.visibility.float(),
            target.visibility.detach().float(),
            reduction="none",
        ),
        object_validity,
    )
    persistence = masked(
        F.smooth_l1_loss(
            prediction.persistence.float(),
            target.persistence.detach().float(),
            reduction="none",
        ),
        object_validity,
    )
    uncertainty = masked(
        F.smooth_l1_loss(
            prediction.uncertainty.float(),
            target.uncertainty.detach().float(),
            reduction="none",
        ),
        object_validity,
    )
    reliability = masked(
        F.smooth_l1_loss(
            prediction.reliability.float(),
            reliability_weight,
            reduction="none",
        ),
        object_validity,
    )

    def address_distribution(address: Tensor, *, detached: bool) -> Tensor:
        candidate = address.detach().float() if detached else address.float()
        candidate = candidate.clamp_min(0.0).flatten(-3)
        total = candidate.sum(dim=-1, keepdim=True)
        candidate = candidate / total.clamp_min(1.0)
        null = (1.0 - candidate.sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        return torch.cat((candidate, null), dim=-1)

    predicted_address = address_distribution(prediction.future_address, detached=False)
    target_address = address_distribution(target.future_address, detached=True)
    address_error = 0.5 * (
        predicted_address.clamp_min(1e-8).sqrt() - target_address.clamp_min(1e-8).sqrt()
    ).square().sum(dim=-1)
    # Teacher-G confidence-blends a null/diffuse association to the current
    # unit-mass address.  That identity fallback is an actual target: masking
    # it by reliability would again leave an action-owned free address on the
    # rows where W is supposed to be neutral.
    address = masked(address_error, object_validity.squeeze(-1))
    total = (
        0.30 * successor
        + 0.22 * semantic_delta
        + 0.15 * transport
        + 0.05 * covariance
        + 0.06 * visibility
        + 0.06 * persistence
        + 0.05 * uncertainty
        + 0.06 * reliability
        + 0.05 * address
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
        "future_reliability": reliability,
        "future_address": address,
    }
    if collect_diagnostics:
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
                transport_error[:, interval_slice], validity[:, interval_slice]
            ).detach()
            terms[f"future_interval_{index}_address"] = masked(
                address_error[:, interval_slice], interval_validity.squeeze(-1)
            ).detach()
    return terms


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
    # V120 used no event-row boost in its physical flow objective.  Keep the
    # new event-balanced objective, but also serialize the exact comparable
    # scale so recovery checks do not mistake a deliberate reweighting for a
    # worse velocity fit.
    flow_v120_comparable = (physical_error_unweighted * step_weight).mean()
    arm = (arm_error.mean(dim=-1) * step_weight).mean()
    grip = (gripper_error * step_weight).mean()
    grip_unweighted = (gripper_error_unweighted * step_weight).mean()
    grip_value = (residual_parts.gripper_field[..., 0].square() * step_weight).mean()
    grip_value_balanced = (
        residual_parts.gripper_field[..., 0].square() * event_row_weight * step_weight
    ).mean()
    grip_delta = (
        residual_parts.gripper_field[..., 1].square() * event_row_weight * step_weight
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
    event_positive = torch.where(
        flat_event != 0,
        event_ce.new_full((), float(objective.event_positive_weight)),
        event_ce.new_ones(()),
    )
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
        # actual event-balanced objective is reported under an explicit name.
        band_metrics[f"action_flow_band_{start + 1}_{end}"] = physical_error_unweighted[
            :, start:end
        ].mean()
        band_metrics[f"action_flow_balanced_band_{start + 1}_{end}"] = physical_error[
            :, start:end
        ].mean()
        band_metrics[f"action_horizon_weight_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].mean()
        band_metrics[f"action_horizon_mass_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].sum() / horizon_weight.sum()
        start = end
    return {
        **band_metrics,
        "action_flow": flow,
        "action_flow_v120_comparable": flow_v120_comparable,
        "action_flow_event_balance_delta": flow - flow_v120_comparable,
        "action_flow_uniform_field_mse": uniform_flow,
        "action_flow_native": native_flow,
        "action_arm_flow": arm,
        "action_gripper_flow": grip,
        "action_gripper_flow_unweighted": grip_unweighted,
        "action_gripper_value_flow": grip_value_balanced,
        "action_gripper_value_flow_unweighted": grip_value,
        "action_gripper_delta_flow": grip_delta,
        "action_gripper_auxiliary_flow": grip_auxiliary,
        "action_gripper_event_flow": event_gripper_flow,
        "action_gripper_hold_flow": hold_gripper_flow,
        "action_decoded_gripper_event": event_decoded_gripper,
        "action_decoded_gripper_hold": hold_decoded_gripper,
        "action_gripper_event_row_weight": event_row_weight_mean,
        "action_gripper_hold_row_weight": hold_row_weight_mean,
        "action_gripper_event_rate": event_mask.mean(),
        "decoded_action": decoded_action,
        "decoded_action_v120_comparable": decoded_action_v120_comparable,
        "decoded_action_event_balance_delta": (
            decoded_action - decoded_action_v120_comparable
        ),
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
        "action_flow_balanced_first": physical_error[:, 0].mean(),
        "action_flow_balanced_first8": physical_error[:, :8].mean(),
        "action_flow_balanced_tail": physical_error[:, 8:].mean(),
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
        if set(self.groups) != {"action", "representation"}:
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
    geometry = flow_geometry_terms(observation)
    if top_targets.teacher_dynamics is None:
        raise ValueError("formal training requires future teacher dynamics")
    future = future_dynamics_terms(
        predicted_dynamics,
        top_targets.teacher_dynamics,
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
    representation_group = (
        objective.future_dynamics * future["future_dynamics"]
        + objective.intent_structure
        * (
            0.25 * top_targets.object_reconstruction_loss
            + 0.35 * top_targets.online_intent_loss
            + 0.20 * top_targets.plan_recognition_loss
            + 0.20 * top_targets.coarse_action_loss
        )
        + objective.flow_warp * geometry["flow_warp"]
        + objective.flow_identity_advantage * geometry["flow_identity_advantage"]
        + objective.flow_static_identity * geometry["flow_static_identity"]
        + objective.flow_cycle * geometry["flow_cycle"]
        + objective.flow_smoothness * geometry["flow_smoothness"]
        + objective.flow_uncertainty * geometry["flow_uncertainty"]
        + objective.flow_refinement_sequence * geometry["flow_refinement_sequence"]
    )
    groups = {"action": action_group, "representation": representation_group}
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
        "future_dynamics": objective.future_dynamics * future["future_dynamics"],
        "object_reconstruction": (
            objective.intent_structure * 0.25 * top_targets.object_reconstruction_loss
        ),
        "intent_online": (
            objective.intent_structure * 0.35 * top_targets.online_intent_loss
        ),
        "intent_recognizer": (
            objective.intent_structure * 0.20 * top_targets.plan_recognition_loss
        ),
        "coarse_action": (
            objective.intent_structure * 0.20 * top_targets.coarse_action_loss
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
        "intent_online": top_targets.online_intent_loss,
        "intent_recognizer": top_targets.plan_recognition_loss,
        "object_reconstruction": top_targets.object_reconstruction_loss,
        "coarse_action": top_targets.coarse_action_loss,
        "history_action_proposal": top_targets.history_proposal_loss,
    }
    ledger = LossLedger(
        total=action_group + representation_group,
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
    "flow_geometry_terms",
    "future_dynamics_terms",
    "sample_flow_matching",
]
