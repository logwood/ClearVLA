"""Single owner of active ClearVLA objectives and loss accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from clearvla.action_solvers.flow_solver.spec import ScheduleSpec
from clearvla.data.action_chart import resolve_action_state_profile

from ..config import ExperimentConfig
from ..gripper_contract import is_binary_gripper_mode
from ..interfaces import ActionSupervision, ObservableHistory
from ..model.action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from ..model.observation_contract import (
    ObservationEvidence,
    PatchFlowField,
    _coordinate_grid,
)
from ..model.policy import PolicyStepOutput
from ..model.types import FlowStepContext, FutureObjectDynamics, ObjectTopTrainingTargets
from ..supervision import quarantine, supported_mean
from ..v120_core.gauges import masked_candidate_center

if TYPE_CHECKING:
    from ..model.components import OutletAdapter

_STANDARD_GAMMA = getattr(torch, "_standard_gamma")


@dataclass(frozen=True)
class FlowMatchingState:
    time: Tensor  # [B]
    source_physical_noise: Tensor  # [B,T,Aphysical]
    noisy_physical: Tensor  # [B,T,Aphysical]
    target_physical: Tensor  # [B,T,Aphysical]
    target_physical_velocity: Tensor  # [B,T,Aphysical]
    # Optional numerical context used only by a schedule-aware bottom run.
    # Keeping this appended/defaulted preserves every historical constructor.
    flow_step_context: FlowStepContext | None = None
    row_valid: Tensor | None = None  # source labels only; never passed to velocity


def _gripper_field_error_terms(
    field: Tensor,
    *,
    compatibility_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return legacy, deployed-lane and owned gripper field errors.

    The six gripper coordinates are one physical degree of freedom expressed
    in two deployed operands (absolute/local-delta) plus four compatibility
    coordinates.  ``compatibility_weight=1`` is algebraically the historical
    six-channel mean.  Lower values preserve the deployed lane's total budget
    while reducing only the compatibility lane; zero therefore does not
    silently weaken the deployed gripper objective by a factor of six.

    The helper accepts arbitrary leading dimensions so it can be shared by
    the direct flow objective and the detached execution-value target.  It
    never changes the field shape or decoder semantics.
    """

    if not isinstance(field, Tensor) or field.ndim < 1:
        raise ValueError("gripper field must be a tensor with a final channel axis")
    channels = int(field.shape[-1])
    if channels != 6:
        raise ValueError(
            "the resolved legacy gripper field requires exactly six channels, "
            f"got {channels}"
        )
    weight = float(compatibility_weight)
    if not math.isfinite(weight):
        raise ValueError("gripper compatibility weight must be finite")
    if weight < 0.0:
        raise ValueError("gripper compatibility weight must be non-negative")

    squared = field.square()
    deployed = squared[..., :2].mean(dim=-1)
    compatibility = squared[..., 2:].mean(dim=-1)
    # Weight by the number of coordinates in each lane.  This makes weight=1
    # exactly the old six-channel mean and keeps the deployed lane's aggregate
    # scale unchanged when compatibility is disabled.
    owned = (2.0 * deployed + (4.0 * weight) * compatibility) / (
        2.0 + 4.0 * weight
    )
    legacy = (2.0 * deployed + 4.0 * compatibility) / 6.0
    return legacy, deployed, compatibility, owned


def balanced_event_row_weights(
    event_mask: Tensor,
    horizon_weight: Tensor,
    row_valid: Tensor | None = None,
) -> Tensor:
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
    if row_valid is not None:
        event = event * row_valid
        hold = hold * row_valid
    event_count = event.sum()
    hold_count = hold.sum()
    total = event_count + hold_count
    event_scale = torch.sqrt(total / (2.0 * event_count.clamp_min(1.0)))
    hold_scale = torch.sqrt(total / (2.0 * hold_count.clamp_min(1.0)))
    raw = event * event_scale + hold * hold_scale
    both_classes = (event_count > 0.0) & (hold_count > 0.0)
    raw = torch.where(both_classes, raw, torch.ones_like(raw))
    step = horizon_weight.float()[None]
    if row_valid is not None:
        step = step * row_valid
    denominator = step.expand_as(raw).sum().clamp_min(1.0)
    normalization = (raw * step).sum() / denominator
    balanced = raw / normalization.clamp_min(1e-8)
    # A batch with only one class has no imbalance to repair and must retain
    # the original gripper objective bit-exactly.
    return torch.where(both_classes, balanced, torch.ones_like(balanced))


def calvin_selective_frame_weights(
    config: ExperimentConfig,
    codec: PhysicalActionFieldCodec | "OutletAdapter",
    normalized_action: Tensor,
    action_state: Tensor,
    event_mask: Tensor,
    horizon_weight: Tensor,
    row_valid: Tensor | None = None,
) -> Tensor:
    """Build bounded CALVIN row weights without changing the policy graph.

    CALVIN's information sampler already selects informative *windows*, but
    the formal action objective historically gave every row inside a selected
    24-step window the same weight (apart from the fixed horizon bands).  This
    helper adds a deliberately small, target-only focus on actual centered TCP
    motion and gripper transition rows.  The result is normalized against the
    existing horizon budget, so enabling it reallocates action supervision
    rather than silently changing the action group's overall scale.

    The selective profile is CALVIN-only.  Pen/RDT and the default ``uniform``
    mode return an exact all-ones tensor, preserving their objective and
    checkpoint behavior.  Event/motion values are detached supervision
    metadata; no future tensor crosses into the deployed model path.
    """

    mode = str(config.objectives.calvin_frame_weight_mode)
    if mode == "uniform":
        return normalized_action.new_ones(normalized_action.shape[:2], dtype=torch.float32)
    if mode != "motion_event_v1":
        raise ValueError(f"unsupported CALVIN frame weighting mode: {mode!r}")
    if str(config.data.data_profile) != "calvin_relative_7d_v1":
        raise ValueError("selective frame weighting is valid only for CALVIN")
    if not bool(getattr(codec, "is_binary_command", False)):
        raise ValueError(
            "CALVIN selective frame weighting requires the configured OutletAdapter"
        )
    if normalized_action.ndim != 3 or action_state.ndim != 2:
        raise ValueError("CALVIN frame weighting received invalid action shapes")
    batch, horizon = normalized_action.shape[:2]
    if tuple(event_mask.shape) != (batch, horizon):
        raise ValueError("CALVIN event rows must align with the action horizon")
    if tuple(horizon_weight.shape) != (horizon,):
        raise ValueError("CALVIN horizon weights must align with the action horizon")

    # OutletAdapter.arm_motion_magnitude removes the z-score representation of
    # raw zero for CALVIN relative commands.  Calling it here, rather than
    # taking a norm of normalized values, prevents a normalizer offset from
    # masquerading as physical motion.
    motion = codec.arm_motion_magnitude(
        normalized_action.detach(),
        action_state.detach(),
    ).float()
    if tuple(motion.shape) != (batch, horizon) or not bool(torch.isfinite(motion).all()):
        raise ValueError("CALVIN centered motion scores must be finite [B,T]")
    motion = motion.clamp_min(0.0)
    # Use the same centered magnitude threshold that owns the motion-head
    # target, rather than a per-window percentile.  Percentiles silently make
    # a one- or two-row arm movement look like a hold window (p90 is still
    # zero), which is exactly the sparse-tail failure this profile is meant to
    # repair.  The bounded ratio keeps tiny numerical motion close to one while
    # giving genuinely active rows a predictable, source-independent boost.
    motion_threshold = max(float(config.objectives.arm_motion_threshold), 1e-6)
    motion_focus = (motion / motion_threshold).clamp(0.0, 1.0)

    event = event_mask.detach().float().clamp(0.0, 1.0)
    radius = int(config.objectives.calvin_frame_event_radius)
    if radius:
        event_focus = F.max_pool1d(
            F.pad(event[:, None], (radius, radius)),
            kernel_size=2 * radius + 1,
            stride=1,
        ).squeeze(1)
    else:
        event_focus = event
    raw = (
        1.0
        + float(config.objectives.calvin_frame_motion_gain) * motion_focus
        + float(config.objectives.calvin_frame_event_gain) * event_focus
    ).clamp_min(1.0)
    raw = raw.clamp_max(float(config.objectives.calvin_frame_max_weight))
    horizon = horizon_weight.detach().float()[None]
    if row_valid is not None:
        horizon = horizon * row_valid
    denominator = (horizon.expand_as(raw)).sum().clamp_min(1e-8)
    weighted_mean = (raw * horizon).sum() / denominator
    weights = raw / weighted_mean.clamp_min(1e-8)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("CALVIN selective frame weights became non-finite")
    return weights


def causal_event_trajectory_mask(event_mask: Tensor) -> Tensor:
    """Select each continuous event row and every later trajectory row."""

    if event_mask.ndim != 2:
        raise ValueError("gripper trajectory event mask must be [B,T]")
    return event_mask.to(dtype=torch.float32).cumsum(dim=1).clamp(max=1.0)


def event_transition_persistence_masks(
    event_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split event transitions from non-event persistence rows.

    An event row owns the local state change. Later non-event rows own
    persistence until another event row takes transition ownership. These are
    loss-selection masks only: they never reset or reconstruct a deployed
    decoder branch. The two masks are disjoint, and a no-event trajectory is
    algebraically zero in both owners.
    """

    if event_mask.ndim != 2:
        raise ValueError("gripper trajectory event mask must be [B,T]")
    event = (event_mask > 0).to(dtype=torch.float32)
    after_or_event = torch.cummax(event, dim=1).values
    persistence = (after_or_event - event).clamp(0.0, 1.0)
    return event, persistence


def stackcube_phase_masks(
    event_target: Tensor,
    gripper_boundary_raw: Tensor,
    *,
    open_class: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return target-only approach, transport and pre-event-hold masks.

    The event convention stores a negative delta as class ``1`` and a
    positive delta as class ``2``.  ``open_class`` maps that sign convention
    to the native outlet direction.  The masks are derived exclusively from
    detached target rows and the causal command boundary; no object state,
    reward or future observation is exposed to the policy.

    Windows beginning after the close event are still recognized as transport
    windows through the closed command boundary.  Windows with no close event
    are not treated as approach rows, avoiding a false approach boost on
    post-release tails.
    """

    if event_target.ndim != 2 or gripper_boundary_raw.ndim != 1:
        raise ValueError("StackCube phase masks require [B,T] events and [B] boundaries")
    if int(event_target.shape[0]) != int(gripper_boundary_raw.shape[0]):
        raise ValueError("StackCube phase mask batch dimensions must align")
    if int(open_class) not in {1, 2}:
        raise ValueError("StackCube open event class must be 1 or 2")
    close_class = 1 if int(open_class) == 2 else 2
    close = event_target == close_class
    open_ = event_target == int(open_class)
    close_seen = close.cumsum(dim=1).to(dtype=torch.bool)
    open_seen = open_.cumsum(dim=1).to(dtype=torch.bool)
    boundary_closed = gripper_boundary_raw[:, None] <= 0.0
    has_close = close.any(dim=1, keepdim=True)
    approach = (~close_seen) & (~open_seen) & (~boundary_closed) & has_close
    transport = (boundary_closed | close_seen) & (~open_seen)
    pre_event_hold = ~(close_seen | open_seen)
    return (
        approach.to(dtype=torch.float32),
        transport.to(dtype=torch.float32),
        pre_event_hold.to(dtype=torch.float32),
    )


def anchored_gripper_persistence(
    absolute: Tensor,
    local_delta: Tensor,
    event_mask: Tensor,
) -> Tensor:
    """Reconstruct post-event segments without pre-event delta leakage."""

    if absolute.ndim != 3 or int(absolute.shape[-1]) != 1:
        raise ValueError("gripper absolute trajectory must be [B,T,1]")
    if tuple(local_delta.shape) != tuple(absolute.shape):
        raise ValueError("gripper local delta must align with absolute trajectory")
    if tuple(event_mask.shape) != tuple(absolute.shape[:2]):
        raise ValueError("gripper event mask must align with trajectory rows")
    batch, horizon = event_mask.shape
    row = torch.arange(horizon, device=event_mask.device, dtype=torch.long)[None]
    latest = torch.where(
        event_mask > 0,
        row.expand(batch, -1),
        torch.full(
            (batch, horizon),
            -1,
            device=event_mask.device,
            dtype=torch.long,
        ),
    )
    latest = torch.cummax(latest, dim=1).values
    gather = latest.clamp_min(0)[..., None]
    anchor = absolute.gather(1, gather)
    prefix = torch.cumsum(local_delta, dim=1)
    prefix_at_event = prefix.gather(1, gather)
    reconstructed = anchor + prefix - prefix_at_event
    return torch.where(
        (latest >= 0)[..., None],
        reconstructed,
        torch.zeros_like(reconstructed),
    )


def sample_flow_matching(
    target: Tensor,
    *,
    action_state: Tensor,
    codec_gripper_boundary: Tensor | None = None,
    codec: PhysicalActionFieldCodec,
    distribution: str,
    generator: torch.Generator | None = None,
    schedule: ScheduleSpec | None = None,
    row_valid: Tensor | None = None,
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
    if row_valid is not None:
        if row_valid.dtype != torch.bool or tuple(row_valid.shape) != tuple(target.shape[:2]):
            raise ValueError("flow label support must be bool [B,T]")
        target = quarantine(target, row_valid)
    target_physical = codec.encode(
        target,
        action_state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    noise = codec.sample_noise(
        int(target.shape[0]),
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    alpha = time.to(dtype=target.dtype)[:, None, None]
    noisy = (1.0 - alpha) * noise + alpha * target_physical
    if row_valid is not None:
        target_physical = quarantine(target_physical, row_valid)
        # Unknown rows remain source noise; they are not trained as a zero or
        # a frozen terminal command. The label mask is NOT an online input.
        noisy = torch.where(row_valid[..., None], noisy, noise)
    flow_step_context: FlowStepContext | None = None
    if schedule is not None:
        schedule.validate()
        # An explicit legacy E5 schedule is still the no-context ABI.  A
        # non-uniform schedule opts into the numerical context contract and
        # uses the sampled continuous time to select its containing interval.
        legacy_boundaries = ScheduleSpec.uniform(5).boundaries
        if tuple(schedule.boundaries) != tuple(legacy_boundaries):
            boundaries = torch.tensor(
                schedule.boundaries,
                device=time.device,
                dtype=time.dtype,
            )
            interval_index = torch.bucketize(
                time,
                boundaries[1:-1],
                right=True,
            )
            step_sizes = boundaries[1:] - boundaries[:-1]
            step_size = step_sizes.index_select(0, interval_index)
            normalized_index = interval_index.to(dtype=time.dtype) / float(
                schedule.interval_count
            )
            flow_step_context = FlowStepContext(
                time=time,
                step_size=step_size,
                normalized_index=normalized_index,
                endpoint=torch.zeros_like(time),
            )
    return FlowMatchingState(
        time=time,
        source_physical_noise=noise,
        noisy_physical=noisy,
        target_physical=target_physical,
        target_physical_velocity=(
            target_physical - noise
            if row_valid is None
            else quarantine(target_physical - noise, row_valid)
        ),
        row_valid=row_valid,
        flow_step_context=flow_step_context,
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
    interval_valid: Tensor | None = None,
    collect_diagnostics: bool = False,
) -> dict[str, Tensor]:
    prediction.validate()
    target.validate()
    batch, intervals, objects = prediction.semantic_delta.shape[:3]
    if current_loss_support.ndim != 4:
        raise ValueError("current loss support must retain [B,K,C,1]")
    cameras = int(current_loss_support.shape[2])
    if tuple(current_loss_support.shape) != (batch, objects, cameras, 1):
        raise ValueError("current loss support must be [B,K,C,1] and align with future dynamics")
    camera_validity = torch.nan_to_num(
        current_loss_support.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )[:, None].expand(-1, intervals, -1, -1, -1).clamp(0.0, 1.0)
    if interval_valid is not None:
        if tuple(interval_valid.shape) != (batch, intervals) or interval_valid.dtype != torch.bool:
            raise ValueError("future interval support must be bool [B,I]")
        camera_validity = camera_validity * interval_valid[:, :, None, None, None]
    object_validity = camera_validity.amax(dim=3)

    def masked(error: Tensor, weight: Tensor) -> Tensor:
        expanded = torch.nan_to_num(
            weight.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        while expanded.ndim < error.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(error)
        # Quarantine unsupported rows before reducing.  Multiplying a NaN
        # error by a zero mask is still NaN and can poison the scalar loss and
        # its backward graph.
        safe_error = torch.where(
            expanded > 0.0,
            error,
            torch.zeros_like(error),
        )
        return (safe_error * expanded).sum() / expanded.sum().clamp_min(1.0)

    def row_loss(
        prediction_value: Tensor,
        target_value: Tensor,
        *,
        support: Tensor,
        scale_floored: bool,
    ) -> tuple[Tensor, Tensor]:
        prediction_f = prediction_value.float()
        target_f = target_value.detach().float()
        support_f = torch.nan_to_num(
            support.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        while support_f.ndim < prediction_f.ndim:
            support_f = support_f.unsqueeze(-1)
        support_f = support_f.expand_as(prediction_f)
        supported = support_f > 0.0
        if bool((supported & ~torch.isfinite(prediction_f)).any()):
            raise ValueError("future dynamics prediction contains non-finite supported values")
        if bool((supported & ~torch.isfinite(target_f)).any()):
            raise ValueError("future dynamics target contains non-finite supported values")
        prediction_f = torch.where(supported, prediction_f, torch.zeros_like(prediction_f))
        target_f = torch.where(supported, target_f, torch.zeros_like(target_f))
        raw = F.smooth_l1_loss(
            prediction_f,
            target_f,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        if not scale_floored:
            return raw, raw
        target_rms = target_f.square().mean(dim=-1, keepdim=True).sqrt()
        if interval_valid is None:
            floor_mean = target_rms.mean(dim=(0, 2), keepdim=True)
        else:
            row_support = support_f[..., :1]
            floor_mean = (target_rms * row_support).sum(dim=(0, 2), keepdim=True) / (
                row_support.sum(dim=(0, 2), keepdim=True).clamp_min(1.0)
            )
        scale_floor = (0.25 * floor_mean).clamp_min(1e-3)
        scale = torch.sqrt(target_rms.square() + scale_floor.square())
        normalized = F.smooth_l1_loss(
            prediction_f / scale,
            target_f / scale,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        prediction_direction = prediction_f / torch.sqrt(
            prediction_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
        )
        target_direction = target_f / torch.sqrt(
            target_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
        )
        # The historical cosine-like expression was positive even when the
        # prediction exactly equalled the target because the variance floor
        # makes each smoothed direction shorter than unit length.  Compare the
        # two smoothed directions directly so the supervised optimum is an
        # attainable exact zero while retaining the same bounded denominator.
        direction = 0.5 * (prediction_direction - target_direction).square().mean(
            dim=-1, keepdim=True
        )
        return raw + normalized + 0.10 * direction, raw

    def transport_row_audits(
        prediction_value: Tensor,
        target_value: Tensor,
        support: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return detached relative-coordinate audits for diagnostic batches.

        Target magnitude is physically meaningful for camera transport and
        therefore cannot redistribute backward responsibility.  The former
        normalized and direction views remain useful observations, but they
        are computed only when diagnostics are requested and never enter the
        active raw-coordinate objective.
        """

        prediction_f = prediction_value.detach().float()
        target_f = target_value.detach().float()
        target_rms = target_f.square().mean(dim=-1, keepdim=True).sqrt()
        floor_mean = target_rms.mean(dim=(0, 2), keepdim=True)
        if interval_valid is not None:
            floor_mean = (target_rms * support).sum(dim=(0, 2), keepdim=True) / support.sum(
                dim=(0, 2), keepdim=True
            ).clamp_min(1.0)
        scale_floor = (0.25 * floor_mean).clamp_min(1e-3)
        scale = torch.sqrt(target_rms.square() + scale_floor.square())
        normalized_audit = F.smooth_l1_loss(
            prediction_f / scale,
            target_f / scale,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        prediction_direction = prediction_f / torch.sqrt(
            prediction_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
        )
        target_direction = target_f / torch.sqrt(
            target_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
        )
        direction_audit = 0.5 * (
            prediction_direction - target_direction
        ).square().mean(dim=-1, keepdim=True)
        return normalized_audit, direction_audit

    def decomposed_loss(
        prediction_value: Tensor,
        target_value: Tensor,
        *,
        weight: Tensor,
        scale_floored: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        prediction_value = prediction_value.float()
        target_value = target_value.detach().float()
        support_f = torch.nan_to_num(
            weight.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        while support_f.ndim < prediction_value.ndim:
            support_f = support_f.unsqueeze(-1)
        support_f = support_f.expand_as(prediction_value)
        supported = support_f > 0.0
        if bool((supported & ~torch.isfinite(prediction_value)).any()):
            raise ValueError("future dynamics prediction contains non-finite supported values")
        if bool((supported & ~torch.isfinite(target_value)).any()):
            raise ValueError("future dynamics target contains non-finite supported values")
        # Invalid intervals/cameras are not values.  Remove them before the
        # interval common-mode reduction so they cannot enter the mean or
        # leave a NaN that a later zero mask would fail to suppress.
        prediction_value = torch.where(
            supported, prediction_value, torch.zeros_like(prediction_value)
        )
        target_value = torch.where(
            supported, target_value, torch.zeros_like(target_value)
        )
        support_count = support_f.sum(dim=1).clamp_min(1.0)
        prediction_common = (prediction_value * support_f).sum(dim=1) / support_count
        target_common = (target_value * support_f).sum(dim=1) / support_count
        prediction_innovation = prediction_value.float() - prediction_common[:, None]
        target_innovation = target_value - target_common[:, None]
        common_error, common_raw_error = row_loss(
            prediction_common[:, None],
            target_common[:, None],
            support=weight.amax(dim=1, keepdim=True)
            if interval_valid is not None
            else weight[:, :1],
            scale_floored=scale_floored,
        )
        innovation_error, innovation_raw_error = row_loss(
            prediction_innovation,
            target_innovation,
            support=weight,
            scale_floored=scale_floored,
        )
        common = masked(
            common_error,
            weight.amax(dim=1, keepdim=True) if interval_valid is not None else weight[:, :1],
        )
        innovation = masked(innovation_error, weight)
        return (
            common,
            innovation,
            common_error,
            innovation_error,
            common_raw_error,
            innovation_raw_error,
        )

    (
        semantic_common,
        semantic_innovation,
        _semantic_common_error,
        semantic_innovation_error,
        _semantic_common_raw_error,
        _semantic_innovation_raw_error,
    ) = decomposed_loss(
        prediction.semantic_delta,
        target.semantic_delta,
        weight=object_validity,
        scale_floored=True,
    )
    semantic_delta = 0.5 * (semantic_common + semantic_innovation)
    (
        transport_common,
        transport_innovation,
        _transport_common_error,
        transport_innovation_error,
        transport_common_raw_error,
        transport_innovation_raw_error,
    ) = decomposed_loss(
        prediction.transport_mean,
        target.transport_mean,
        weight=camera_validity,
        scale_floored=False,
    )
    transport = 0.5 * (transport_common + transport_innovation)
    # Raw coordinate error is both the active measure and the archival audit.
    # No Teacher-magnitude-dependent weight is allowed to reassign physical
    # transport responsibility between common, near or far rows.
    transport_raw_common = masked(
        transport_common_raw_error.detach(),
        camera_validity[:, :1],
    )
    transport_raw_innovation = masked(
        transport_innovation_raw_error.detach(),
        camera_validity,
    )
    transport_raw_coordinate = 0.5 * (
        transport_raw_common + transport_raw_innovation
    )
    covariance_error, _covariance_raw_error = row_loss(
        prediction.transport_covariance,
        target.transport_covariance,
        support=camera_validity,
        scale_floored=False,
    )
    covariance = masked(covariance_error, camera_validity)
    if interval_valid is None:
        semantic_transition_prediction = (
            prediction.semantic_interval_innovation[:, 1:]
            - prediction.semantic_interval_innovation[:, :-1]
        )
        semantic_transition_target = (
            target.semantic_interval_innovation.detach()[:, 1:]
            - target.semantic_interval_innovation.detach()[:, :-1]
        )
    else:
        # Common modes cancel in adjacent differences. Do not first compute
        # an unmasked mean containing unavailable future intervals.
        semantic_prediction = quarantine(prediction.semantic_delta, interval_valid)
        semantic_target = quarantine(target.semantic_delta.detach(), interval_valid)
        semantic_transition_prediction = semantic_prediction[:, 1:] - semantic_prediction[:, :-1]
        semantic_transition_target = semantic_target[:, 1:] - semantic_target[:, :-1]
    transition_validity = torch.minimum(
        object_validity[:, 1:],
        object_validity[:, :-1],
    )
    transition_error, _transition_raw_error = row_loss(
        semantic_transition_prediction,
        semantic_transition_target,
        support=transition_validity,
        scale_floored=True,
    )
    transition = masked(transition_error, transition_validity)
    # The old successor objective was an algebraic duplicate of semantic
    # delta. Its 0.30 budget joins the surviving 0.25 semantic owner. The
    # unobserved status budgets are retired rather than reassigned.
    total = 0.55 * semantic_delta + 0.15 * transport + 0.05 * covariance
    terms = {
        "future_observed_intervals": (
            camera_validity.new_tensor(batch * intervals)
            if interval_valid is None
            else interval_valid.float().sum()
        ),
        "future_supervised_object_intervals": (object_validity > 0).float().sum(),
        "future_dynamics": total,
        "future_semantic_delta": semantic_delta,
        "future_semantic_common": semantic_common,
        "future_semantic_innovation": semantic_innovation,
        "future_transport": transport,
        "future_transport_common": transport_common,
        "future_transport_innovation": transport_innovation,
        "future_transport_raw_coordinate": transport_raw_coordinate.detach(),
        "future_covariance": covariance,
        "future_transition": transition,
    }
    if collect_diagnostics:
        # Diagnostics must obey the same support boundary as the active
        # objective.  In particular, ``FutureObjectDynamics`` exposes
        # interval-common views with the interval axis already reduced, so a
        # full [B,I,K,C,1] mask cannot be broadcast onto them directly.
        def _finite_support(weight: Tensor) -> Tensor:
            return torch.nan_to_num(
                weight.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        def _support_for(value: Tensor, weight: Tensor) -> Tensor:
            support = _finite_support(weight)
            # A reduced common value has lost the interval axis.  Reduce the
            # corresponding support over that axis before appending feature
            # dimensions; ``amax`` preserves the legal-source semantics.
            while support.ndim > value.ndim:
                support = support.amax(dim=1)
            while support.ndim < value.ndim:
                support = support.unsqueeze(-1)
            try:
                return support.expand_as(value)
            except RuntimeError as error:
                raise ValueError(
                    "future diagnostic support does not align with its value: "
                    f"{tuple(weight.shape)} vs {tuple(value.shape)}"
                ) from error

        def diagnostic_rms(value: Tensor, support: Tensor) -> Tensor:
            value_f = value.detach().float()
            support_f = _support_for(value_f, support)
            safe_value = torch.where(
                support_f > 0.0,
                value_f,
                torch.zeros_like(value_f),
            )
            numerator = (safe_value.square() * support_f).sum()
            denominator = support_f.sum().clamp_min(1.0)
            return (numerator / denominator).sqrt()

        def diagnostic_interval_views(
            value: Tensor,
            support: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            """Return sanitized common/innovation views and reduced support."""

            value_f = value.detach().float()
            support_f = _support_for(value_f, support)
            safe_value = torch.where(
                support_f > 0.0,
                value_f,
                torch.zeros_like(value_f),
            )
            count = support_f.sum(dim=1).clamp_min(1.0)
            common = (safe_value * support_f).sum(dim=1) / count
            innovation = safe_value - common[:, None]
            common_support = _finite_support(support).amax(dim=1)
            return common, innovation, common_support

        terms["future_current_loss_support"] = current_loss_support.detach().float().mean()
        terms["future_label_support_fraction"] = (
            camera_validity.new_ones(())
            if interval_valid is None
            else interval_valid.float().mean()
        )
        terms["future_target_semantic_delta_rms"] = (
            diagnostic_rms(target.semantic_delta, object_validity)
        )
        terms["future_target_transport_rms"] = (
            diagnostic_rms(target.transport_mean, camera_validity)
        )
        prediction_transport_common, prediction_transport_innovation, camera_common_support = (
            diagnostic_interval_views(prediction.transport_mean, camera_validity)
        )
        target_transport_common, target_transport_innovation, _ = diagnostic_interval_views(
            target.transport_mean,
            camera_validity,
        )
        terms["future_prediction_transport_common_rms"] = (
            diagnostic_rms(prediction_transport_common, camera_common_support)
        )
        terms["future_target_transport_common_rms"] = (
            diagnostic_rms(target_transport_common, camera_common_support)
        )
        terms["future_prediction_transport_innovation_rms"] = (
            diagnostic_rms(prediction_transport_innovation, camera_validity)
        )
        terms["future_target_transport_innovation_rms"] = (
            diagnostic_rms(target_transport_innovation, camera_validity)
        )
        common_normalized_audit, common_direction_audit = transport_row_audits(
            prediction_transport_common[:, None],
            target_transport_common[:, None],
            camera_common_support[:, None],
        )
        innovation_normalized_audit, innovation_direction_audit = transport_row_audits(
            prediction_transport_innovation,
            target_transport_innovation,
            camera_validity,
        )
        terms["future_transport_normalized_audit"] = 0.5 * (
            masked(common_normalized_audit, camera_validity[:, :1])
            + masked(innovation_normalized_audit, camera_validity)
        )
        terms["future_transport_direction_audit"] = 0.5 * (
            masked(common_direction_audit, camera_validity[:, :1])
            + masked(innovation_direction_audit, camera_validity)
        )
        terms["future_target_covariance_rms"] = (
            diagnostic_rms(target.transport_covariance, camera_validity)
        )
        for index in range(prediction.intervals):
            interval_slice = slice(index, index + 1)
            interval_validity = object_validity[:, interval_slice]
            interval_camera_validity = camera_validity[:, interval_slice]
            terms[f"future_interval_{index}_semantic_delta"] = masked(
                semantic_innovation_error[:, interval_slice], interval_validity
            ).detach()
            terms[f"future_interval_{index}_transport"] = masked(
                transport_innovation_error[:, interval_slice],
                interval_camera_validity,
            ).detach()
    return terms


def execution_value_terms(
    config: ExperimentConfig,
    codec: PhysicalActionFieldCodec | OutletAdapter,
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
    row_valid = flow_state.row_valid
    source_rows = (
        torch.ones(batch, horizon, device=candidates.device, dtype=torch.bool)
        if row_valid is None
        else row_valid
    )
    if tuple(source_rows.shape) != (batch, horizon):
        raise ValueError("execution label support must be [B,T]")
    label_field = source_rows[:, None, None, :, None]
    residual = candidates - target[:, None, None]
    if row_valid is not None:
        residual = torch.where(label_field, residual, torch.zeros_like(residual))
        predicted = torch.where(label_field, predicted, torch.zeros_like(predicted))
    flat = residual.reshape(batch * blocks * candidate_count, horizon, physical)
    parts = codec.split(flat)
    # Both established value/difference arm branches retain symmetric
    # execution-value pressure.  Outlet semantics never replace this field.
    arm_error = 0.5 * (
        parts.arm_absolute.square() + parts.arm_delta.square()
    ).sum(dim=-1) / float(codec.arm_dim)
    (
        gripper_error_legacy,
        gripper_error_deployed,
        gripper_error_compatibility,
        gripper_error_owned,
    ) = _gripper_field_error_terms(
        parts.gripper_field,
        compatibility_weight=config.objectives.gripper_compatibility_weight,
    )
    target_value = torch.stack((arm_error, gripper_error_owned), dim=-1).reshape(
        batch,
        blocks,
        candidate_count,
        horizon,
        2,
    )
    binary_command = is_binary_gripper_mode(config.bottom.gripper_output_mode)
    if binary_command:
        # The continuous gripper field is compatibility/audit-only for a
        # binary outlet;
        # execution candidate values must rank candidates from the arm field
        # alone so random continuous gripper coordinates cannot steer routing.
        target_value = torch.stack(
            (target_value[..., 0], torch.zeros_like(target_value[..., 1])),
            dim=-1,
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
    valid_field = valid[..., None, None].expand_as(predicted) & label_field
    component_weight = (
        predicted.new_tensor([1.0, 0.0])
        if binary_command
        else predicted.new_tensor([float(codec.arm_dim), 1.0])
        / float(codec.arm_dim + 1)
    )
    physical_weight = valid_field.float() * component_weight[None, None, None, None]
    active = (valid.float().sum(dim=2) > 1.0) & source_rows.any(dim=-1)[:, None]
    active_float = active.float()
    active_denominator = active_float.sum().clamp_min(1.0)
    row_denominator = (
        (valid[..., None] & source_rows[:, None, None]).float().sum(dim=(2, 3)).clamp_min(1.0)
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

    def label_time_mean(value: Tensor) -> Tensor:
        if row_valid is None:
            return value.mean(dim=-1)
        return (value * source_rows[:, None, None]).sum(dim=-1) / source_rows.float().sum(dim=-1)[
            :, None, None
        ].clamp_min(1.0)

    predicted_scalar = label_time_mean(
        (predicted * component_weight[None, None, None, None]).sum(dim=-1)
    )
    target_scalar = label_time_mean(
        (normalized_target * component_weight[None, None, None, None]).sum(dim=-1)
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
    common_denominator = active_common.sum() * horizon
    if row_valid is not None:
        active_common = active_common * label_field
        common_denominator = active_common.sum()
    predicted_common_rms = (
        (predicted_mean.square() * active_common * component_weight[None, None, None, None]).sum()
        / common_denominator.clamp_min(1.0)
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
    if execution_cost is None or execution_cost.ndim != 0:
        execution_cost = value_loss.new_zeros(())
    return {
        "execution_value": value_loss,
        "execution_gripper_error_legacy": supported_mean(
            gripper_error_legacy.detach(),
            None
            if row_valid is None
            else source_rows[:, None, None]
            .expand(batch, blocks, candidate_count, horizon)
            .reshape(-1, horizon),
        ),
        "execution_gripper_error_deployed": supported_mean(
            gripper_error_deployed.detach(),
            None
            if row_valid is None
            else source_rows[:, None, None]
            .expand(batch, blocks, candidate_count, horizon)
            .reshape(-1, horizon),
        ),
        "execution_gripper_error_compatibility": supported_mean(
            gripper_error_compatibility.detach(),
            None
            if row_valid is None
            else source_rows[:, None, None]
            .expand(batch, blocks, candidate_count, horizon)
            .reshape(-1, horizon),
        ),
        "execution_gripper_error_owned": supported_mean(
            gripper_error_owned.detach(),
            None
            if row_valid is None
            else source_rows[:, None, None]
            .expand(batch, blocks, candidate_count, horizon)
            .reshape(-1, horizon),
        ),
        "execution_gripper_compatibility_weight": predicted.new_tensor(
            float(config.objectives.gripper_compatibility_weight), dtype=torch.float32
        ),
        "execution_cost_audit": execution_cost.detach().float(),
        "execution_value_reliability_scale": reliability_scale.detach(),
        "execution_value_reliability": (reliability.sum() / active_denominator).detach(),
        "execution_value_target_spread": (
            (target_spread * active_float).sum() / active_denominator
        ).detach(),
        "execution_value_predicted_spread": (
            (predicted_spread * active_float).sum() / active_denominator
        ).detach(),
        "execution_value_predicted_standardized_spread": (
            (predicted_standardized_spread * active_float).sum() / active_denominator
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
        "execution_terminal_identity_error": (candidates[:, :, -1] - baseline)
        .square()
        .mean()
        .sqrt()
        .detach(),
        "execution_terminal_target_cost_margin": (
            (terminal_target_margin * terminal_valid.float()).sum() / terminal_denominator
        ).detach(),
        "execution_terminal_predicted_cost_margin": (
            (terminal_predicted_margin * terminal_valid.float()).sum() / terminal_denominator
        ).detach(),
        "execution_terminal_target_preferred_fraction": (
            ((terminal_target_margin < 0.0) & terminal_valid).float().sum() / terminal_denominator
        ).detach(),
    }


def action_terms(
    config: ExperimentConfig,
    codec: PhysicalActionFieldCodec | OutletAdapter,
    output: PolicyStepOutput,
    target: ActionSupervision,
    history: ObservableHistory,
    flow_state: FlowMatchingState,
    *,
    collect_diagnostics: bool = False,
) -> dict[str, Tensor]:
    objective = config.objectives
    row_valid = target.row_valid
    if row_valid is not None:
        target = replace(
            target,
            normalized=quarantine(target.normalized, row_valid),
            raw_units=quarantine(target.raw_units, row_valid),
        )
    valid = (
        torch.ones_like(target.normalized[..., 0], dtype=torch.bool)
        if row_valid is None
        else row_valid
    )

    def mean_rows(value: Tensor, rows: slice | int = slice(None)) -> Tensor:
        mask = None if row_valid is None else row_valid[:, rows]
        return supported_mean(value, mask)

    binary_command = is_binary_gripper_mode(config.bottom.gripper_output_mode)
    calvin_binary = str(config.bottom.gripper_output_mode) == "calvin_binary_command"
    profile = resolve_action_state_profile(config.data.data_profile)
    # ``event_target`` below is sign-coded (1=negative delta, 2=positive
    # delta), while each source chart has its own native open sign.  Resolve
    # this once at the data boundary so every row-zero/phase diagnostic uses
    # the same meaning.
    open_class = 2 if profile.gripper_open_direction > 0 else 1
    raw_grip = target.raw_units[..., -1].float()
    raw_boundary = torch.cat(
        (
            target.gripper_transition_boundary_raw_units[:, None, -1:].float(),
            raw_grip[:, :-1, None],
        ),
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
    if row_valid is not None:
        event_target = torch.where(row_valid, event_target, torch.zeros_like(event_target))
    event_mask = (event_target != 0).to(dtype=torch.float32)
    # Older probes and compatibility callers construct a minimal bottom
    # namespace without the optional binary-command field. Treat that as the
    # continuous-path default; the binary profile still fails closed below
    # when the command head is genuinely missing.
    command_logits = getattr(output.bottom, "gripper_command_logits", None)
    if binary_command:
        if command_logits is None:
            raise ValueError("binary action loss requires gripper command logits")
        expected_command_shape = (
            int(raw_grip.shape[0]),
            int(raw_grip.shape[1]),
            2,
        )
        if tuple(command_logits.shape) != expected_command_shape:
            raise ValueError(
                "binary gripper command logits must be [B,T,2], got "
                f"{tuple(command_logits.shape)}"
            )
        command_target = (raw_grip >= 0.0).to(dtype=torch.long)
        command_rows = F.cross_entropy(
            command_logits.float().reshape(-1, 2),
            command_target.reshape(-1),
            reduction="none",
        ).reshape_as(raw_grip)
    else:
        command_target = None
        command_rows = torch.zeros_like(raw_grip)
    prediction = output.bottom.physical_velocity.float()
    velocity_target = flow_state.target_physical_velocity.detach().float()
    if (row_valid is None) != (flow_state.row_valid is None):
        raise ValueError("action and flow label support disagree")
    if row_valid is not None and (
        flow_state.row_valid is None or not torch.equal(row_valid, flow_state.row_valid)
    ):
        raise ValueError("action and flow rows must use the same source support")
    residual = prediction - velocity_target
    if row_valid is not None:
        residual = quarantine(residual, row_valid)
    residual_parts = codec.split(residual)
    arm_error = 0.5 * (
        residual_parts.arm_absolute.square() + residual_parts.arm_delta.square()
    )
    (
        gripper_error_legacy,
        gripper_error_deployed,
        gripper_error_compatibility,
        gripper_error_owned,
    ) = _gripper_field_error_terms(
        residual_parts.gripper_field,
        compatibility_weight=objective.gripper_compatibility_weight,
    )
    horizon_weight = anchor_horizon_weights(
        horizon=config.dimensions.action_horizon,
        tail_emphasis=objective.horizon_tail_emphasis,
        first_step_protection=objective.horizon_first_step_protection,
        device=prediction.device,
    )
    horizon_step_weight = horizon_weight[None]
    if row_valid is not None:
        # Preserve legacy horizon weighting, but normalize by the number of
        # observed rows rather than the padded tensor size. No tail dilution.
        horizon_step_weight = (
            horizon_step_weight
            * row_valid
            * (row_valid.numel() / row_valid.float().sum().clamp_min(1.0))
        )
    frame_weight = calvin_selective_frame_weights(
        config,
        codec,
        target.normalized,
        history.action_state,
        event_mask,
        horizon_weight,
        row_valid=row_valid,
    ).to(device=prediction.device, dtype=torch.float32)
    # The selective profile is normalized against the existing horizon mass;
    # this is a row-level reallocation, not a hidden action-group gain.
    step_weight = horizon_step_weight * frame_weight
    phase_mode = str(objective.phase_control_mode)
    if phase_mode == "stackcube_phase_v1":
        phase_approach, phase_transport, phase_pre_event_hold = stackcube_phase_masks(
            event_target.detach(),
            target.gripper_transition_boundary_raw_units[:, -1].detach().float(),
            open_class=open_class,
        )
        phase_raw_weight = (
            1.0
            + float(objective.phase_approach_arm_gain) * phase_approach
            + float(objective.phase_transport_arm_gain) * phase_transport
        )
        phase_mass = step_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        phase_denominator = (phase_raw_weight * step_weight).sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        phase_weight = phase_raw_weight * phase_mass / phase_denominator
    else:
        phase_approach = torch.zeros_like(event_mask)
        phase_transport = torch.zeros_like(event_mask)
        phase_pre_event_hold = torch.zeros_like(event_mask)
        phase_weight = torch.ones_like(step_weight)
    event_row_weight = balanced_event_row_weights(event_mask, horizon_weight, row_valid)
    # ``*_legacy`` remains the exact V120 six-channel audit.  The owned error
    # is the only continuous-gripper field error allowed into the formal
    # physical-flow objective; its lane budget is unchanged at weight=1 and
    # excludes only compatibility coordinates at weight=0.
    gripper_error = gripper_error_owned * event_row_weight
    if binary_command:
        # A binary outlet's formal action objective is arm-only; the command
        # head owns gripper supervision below.  Keep continuous gripper terms
        # available as detached audits, but never let them enter the ledger.
        physical_error_unweighted = arm_error.mean(dim=-1)
        physical_error = physical_error_unweighted
        physical_error_owned_unweighted = physical_error_unweighted
        physical_error_owned = physical_error
    else:
        physical_error_unweighted = (
            arm_error.sum(dim=-1) + gripper_error_legacy
        ) / float(codec.arm_dim + 1)
        physical_error_owned_unweighted = (
            arm_error.sum(dim=-1) + gripper_error_owned
        ) / float(codec.arm_dim + 1)
        physical_error = (arm_error.sum(dim=-1) + gripper_error) / float(codec.arm_dim + 1)
        physical_error_owned = (
            arm_error.sum(dim=-1) + gripper_error
        ) / float(codec.arm_dim + 1)
    flow = (physical_error * step_weight).mean()
    flow_owned_unbalanced = (
        physical_error_owned_unweighted * step_weight
    ).mean()
    # V120 used no event-row boost in its physical flow objective.  Serialize
    # the balanced counterfactual under an explicit audit name; it is not sent
    # to backward and cannot be mistaken for the recovered formal geometry.
    flow_selective_unbalanced = (physical_error_unweighted * step_weight).mean()
    flow_v120_comparable = (physical_error_unweighted * horizon_step_weight).mean()
    arm = (arm_error.mean(dim=-1) * step_weight).mean()
    grip = (
        torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        if binary_command
        else (gripper_error * step_weight).mean()
    )
    # Keep the historical metric names as six-channel audit values.  The
    # explicit deployed/compatibility values below make the active ownership
    # decision visible without changing old log interpretation.
    grip_unweighted = (gripper_error_legacy * horizon_step_weight).mean()
    grip_owned_unweighted = (gripper_error_owned * horizon_step_weight).mean()
    grip_deployed_unweighted = (
        gripper_error_deployed * horizon_step_weight
    ).mean()
    grip_compatibility_unweighted = (
        gripper_error_compatibility * horizon_step_weight
    ).mean()
    grip_owned = (gripper_error * horizon_step_weight).mean()
    owned_deployed_mass = (
        2.0 * gripper_error_deployed * horizon_step_weight
    ).mean()
    owned_compatibility_mass = (
        4.0
        * float(objective.gripper_compatibility_weight)
        * gripper_error_compatibility
        * horizon_step_weight
    ).mean()
    owned_compatibility_fraction = owned_compatibility_mass / (
        owned_deployed_mass + owned_compatibility_mass
    ).clamp_min(1e-8)
    grip_value = (
        residual_parts.gripper_field[..., 0].square() * horizon_step_weight
    ).mean()
    grip_value_balanced = (
        residual_parts.gripper_field[..., 0].square()
        * event_row_weight
        * horizon_step_weight
    ).mean()
    grip_delta_unweighted = (
        residual_parts.gripper_field[..., 1].square() * horizon_step_weight
    ).mean()
    grip_delta = (
        residual_parts.gripper_field[..., 1].square()
        * event_row_weight
        * horizon_step_weight
    ).mean()
    grip_auxiliary_unweighted = (
        residual_parts.gripper_field[..., 2:].square().mean(dim=-1)
        * horizon_step_weight
    ).mean()
    grip_auxiliary = (
        residual_parts.gripper_field[..., 2:].square().mean(dim=-1)
        * event_row_weight
        * horizon_step_weight
    ).mean()
    uniform_flow = mean_rows(residual.square())
    native_arm, _ = codec.project_arm_tangent(residual[..., : 2 * codec.arm_dim])
    native_grip = residual_parts.gripper_field[..., 0]
    native_error = (
        native_arm.float().square().mean(dim=-1)
        if binary_command
        else (
            native_arm.float().square().sum(dim=-1) + native_grip.float().square()
        )
        / float(codec.arm_dim + 1)
    )
    native_flow = (native_error * step_weight).mean()
    if row_valid is not None:
        complete = row_valid.all(dim=1)
        native_flow = supported_mean(native_error * horizon_weight[None], complete)
    remaining = (1.0 - flow_state.time.float())[:, None, None]
    clean_physical = flow_state.noisy_physical.float() + remaining * prediction
    # Unlike the standalone codec compatibility API, the formal loss path
    # must always carry the profile-owned command boundary explicitly.
    codec_gripper_boundary = history.codec_gripper_boundary.float()
    decoded = codec.decode(
        clean_physical,
        history.action_state.float(),
        codec_gripper_boundary=codec_gripper_boundary,
    )
    decoded_element_error = F.smooth_l1_loss(
        decoded,
        target.normalized.float(),
        reduction="none",
    )
    decoded_gripper_error = decoded_element_error[..., -1]
    # Receding-horizon deployment executes row zero and discards the rest of
    # this proposal.  A window-level event objective can therefore teach a
    # release at row 4 while the environment keeps receiving the old closed
    # command.  This narrow auxiliary surface explicitly prices only a
    # target opening transition at row zero; it is disabled for binary-command
    # outlets and remains zero when a batch has no such row.
    first_step_release_mask = ((event_target[:, 0] == open_class) & valid[:, 0]).to(
        dtype=torch.float32
    )
    first_step_release_denominator = first_step_release_mask.sum().clamp_min(1.0)
    first_step_release = (
        (decoded_gripper_error[:, 0] * first_step_release_mask).sum()
        / first_step_release_denominator
    )
    if binary_command:
        first_step_release = first_step_release * 0.0
    decoded_rows_unbalanced = (
        decoded_element_error[..., : codec.arm_dim].mean(dim=-1)
        if binary_command
        else (
            decoded_element_error[..., : codec.arm_dim].sum(dim=-1)
            + decoded_gripper_error
        )
        / float(codec.arm_dim + 1)
    )
    decoded_rows = (
        decoded_element_error[..., : codec.arm_dim].mean(dim=-1)
        if binary_command
        else (
            decoded_element_error[..., : codec.arm_dim].sum(dim=-1)
            + decoded_gripper_error * event_row_weight
        )
        / float(codec.arm_dim + 1)
    )
    decoded_action = (decoded_rows * step_weight).mean()
    decoded_action_selective_unbalanced = (
        decoded_rows_unbalanced * step_weight
    ).mean()
    decoded_action_v120_comparable = (
        decoded_element_error.mean(dim=-1) * horizon_step_weight
    ).mean()
    transition_start = torch.cat(
        (
            history.action_state[:, : codec.arm_dim].float(),
            codec_gripper_boundary,
        ),
        dim=-1,
    )
    boundary = torch.cat(
        (
            transition_start[:, None],
            target.normalized[:, :-1].float(),
        ),
        dim=1,
    )
    delta = target.normalized.float() - boundary
    predicted_boundary = torch.cat(
        (transition_start[:, None], decoded[:, :-1]), dim=1
    )
    predicted_delta = decoded - predicted_boundary
    # For relative-command outlets this is adjacent-command consistency; it is
    # deliberately not reused as W's physical delta. Pen/RDT retain their
    # original calculation.
    smooth_delta_all = F.smooth_l1_loss(
        predicted_delta,
        delta,
        reduction="none",
    )
    smooth_delta_rows = (
        smooth_delta_all[..., : codec.arm_dim].mean(dim=-1)
        if binary_command
        else smooth_delta_all.mean(dim=-1)
    )
    smooth_delta = (smooth_delta_rows * step_weight).mean()
    clean_parts = codec.split(clean_physical)
    if binary_command:
        clean_arm_delta = clean_parts.arm_delta.float()
        decoded_boundary = torch.cat(
            (history.action_state[:, None].float(), decoded[:, :-1]),
            dim=1,
        )
        physical_delta_rows = F.smooth_l1_loss(
            decoded[..., : codec.arm_dim] - decoded_boundary[..., : codec.arm_dim],
            clean_arm_delta,
            reduction="none",
        ).mean(dim=-1)
    else:
        physical_delta_rows = codec.delta_consistency(
            clean_physical,
            history.action_state.float(),
            decoded,
            codec_gripper_boundary=codec_gripper_boundary,
        )
    physical_delta_consistency = (physical_delta_rows * step_weight).mean()
    clean_gripper_absolute = clean_parts.gripper_field[..., :1]
    clean_gripper_local_delta = clean_parts.gripper_field[..., 1:2]
    clean_gripper_cumulative = anchored_gripper_persistence(
        clean_gripper_absolute,
        clean_gripper_local_delta,
        event_mask,
    )
    continuous_gripper_target = target.normalized[..., -1:].float()
    continuous_gripper_target_delta = delta[..., -1:].detach().float()
    # The deployment loop consumes only row zero.  On a no-event row zero the
    # correct command is to hold the current gripper state, even when a later
    # row in the 24-step target contains a close/open transition.  Supervise
    # both the decoded absolute lane and the local-delta lane so this remains
    # effective for the two independently-owned physical coordinates.
    first_step_hold_mask = ((event_target[:, 0] == 0) & valid[:, 0]).to(dtype=torch.float32)
    first_step_hold_local = F.smooth_l1_loss(
        clean_gripper_local_delta[:, 0],
        continuous_gripper_target_delta[:, 0],
        reduction="none",
    )[..., 0]
    first_step_hold_absolute = decoded_gripper_error[:, 0]
    first_step_hold_rows = 0.5 * (
        first_step_hold_local + first_step_hold_absolute
    )
    first_step_hold = (
        first_step_hold_rows * first_step_hold_mask
    ).sum() / first_step_hold_mask.sum().clamp_min(1.0)
    if binary_command or profile.name not in {
        "libero_relative_7d_v1",
        "maniskill_pd_ee_delta_pose_7d_v1",
        "maniskill_pd_ee_delta_pose_7d_v2",
    }:
        # Config validation restricts this auxiliary to continuous
        # relative-command outlets.  The explicit zero also makes
        # compatibility/minimal callers fail closed if they construct an
        # unrelated namespace without running validation.
        first_step_hold = first_step_hold * 0.0
    transition_mask, persistence_mask = event_transition_persistence_masks(
        event_mask
    )
    event_and_after_mask = causal_event_trajectory_mask(event_mask)
    if row_valid is not None:
        transition_mask = transition_mask * row_valid
        persistence_mask = persistence_mask * row_valid
        event_and_after_mask = event_and_after_mask * row_valid
    transition_weight = transition_mask * step_weight
    persistence_weight = persistence_mask * step_weight

    def owned_trajectory_mean(rows: Tensor, weight: Tensor) -> Tensor:
        numerator = (rows * weight).sum()
        denominator = weight.sum()
        return torch.where(
            denominator > 0.0,
            numerator / denominator.clamp_min(1.0),
            numerator * 0.0,
        )

    transition_absolute_rows = F.smooth_l1_loss(
        clean_gripper_absolute,
        continuous_gripper_target,
        reduction="none",
    )[..., 0]
    transition_delta_rows = F.smooth_l1_loss(
        clean_gripper_local_delta,
        continuous_gripper_target_delta,
        reduction="none",
    )[..., 0]
    persistence_absolute_rows = transition_absolute_rows
    persistence_delta_rows = F.smooth_l1_loss(
        clean_gripper_cumulative,
        continuous_gripper_target,
        reduction="none",
    )[..., 0]
    gripper_transition_absolute = owned_trajectory_mean(
        transition_absolute_rows,
        transition_weight,
    )
    gripper_transition_delta = owned_trajectory_mean(
        transition_delta_rows,
        transition_weight,
    )
    gripper_persistence_absolute = owned_trajectory_mean(
        persistence_absolute_rows,
        persistence_weight,
    )
    gripper_persistence_delta = owned_trajectory_mean(
        persistence_delta_rows,
        persistence_weight,
    )
    gripper_trajectory_absolute = 0.5 * (
        gripper_transition_absolute + gripper_persistence_absolute
    )
    gripper_trajectory_delta = 0.5 * (
        gripper_transition_delta + gripper_persistence_delta
    )
    gripper_trajectory = 0.5 * (
        gripper_trajectory_absolute + gripper_trajectory_delta
    )
    if binary_command:
        # Keep the old continuous trajectory quantities below as audit
        # surfaces, but remove them from the formal binary-command objective.
        gripper_trajectory = gripper_trajectory * 0.0
    motion_target = (
        codec.arm_motion_magnitude(
            target.normalized.detach().float(),
            history.action_state.detach().float(),
        )
        >= float(objective.arm_motion_threshold)
    ).float()
    motion_rows = F.binary_cross_entropy_with_logits(
        output.bottom.motion_logits.float(), motion_target, reduction="none"
    )
    motion = (motion_rows * step_weight).mean()
    phase_arm_flow = (
        arm_error.mean(dim=-1) * phase_weight * step_weight
    ).mean()
    pre_event_hold_denominator = (
        phase_pre_event_hold * step_weight
    ).sum().clamp_min(1.0)
    pre_event_hold_local_rows = F.smooth_l1_loss(
        clean_gripper_local_delta,
        continuous_gripper_target_delta,
        reduction="none",
    )[..., 0]
    pre_event_hold_absolute_rows = F.smooth_l1_loss(
        decoded[..., -1],
        target.normalized[..., -1].float(),
        reduction="none",
    )
    # The second term keeps the deployed absolute lane on the same target
    # without introducing a new decoder or runtime gate.
    pre_event_hold_rows = 0.5 * (
        pre_event_hold_local_rows + pre_event_hold_absolute_rows
    )
    pre_event_hold = (
        pre_event_hold_rows * phase_pre_event_hold * step_weight
    ).sum() / pre_event_hold_denominator
    if phase_mode != "stackcube_phase_v1":
        phase_arm_flow = phase_arm_flow * 0.0
        pre_event_hold = pre_event_hold * 0.0
    if binary_command:
        if command_target is None or command_logits is None:
            raise RuntimeError("binary command target/logits were not materialized")
        command_prediction = command_logits.detach().float().argmax(dim=-1)
        command_correct = (command_prediction == command_target).float()
        command_target_positive = (command_target == 1).float()
        command_prediction_positive = (command_prediction == 1).float()
        command_tp = (((command_prediction == 1) & (command_target == 1)) & valid).float().sum()
        command_fp = (((command_prediction == 1) & (command_target == 0)) & valid).float().sum()
        command_fn = (((command_prediction == 0) & (command_target == 1)) & valid).float().sum()
        command_rows_count = valid.float().sum()
        command_accuracy = mean_rows(command_correct)
        command_positive_rate = mean_rows(command_prediction_positive)
        command_target_positive_rate = mean_rows(command_target_positive)
        command_precision = command_tp / (command_tp + command_fp).clamp_min(1.0)
        command_recall = command_tp / (command_tp + command_fn).clamp_min(1.0)
        command_f1 = (
            2.0 * command_precision * command_recall
            / (command_precision + command_recall).clamp_min(1e-8)
        )
        command_loss_unbalanced = (command_rows * step_weight).mean()
        # The sampler owns data coverage; the binary head keeps ordinary CE.
        # Hidden per-batch class reweighting changed the objective between
        # batches and is intentionally absent from the formal path.
        command_loss = command_loss_unbalanced
        command_positive_weight = command_loss.new_ones(())
        command_negative_weight = command_loss.new_ones(())
    else:
        zero_command = prediction.new_zeros((), dtype=torch.float32)
        command_accuracy = zero_command
        command_positive_rate = zero_command
        command_target_positive_rate = zero_command
        command_precision = zero_command
        command_recall = zero_command
        command_f1 = zero_command
        command_rows_count = zero_command
        command_loss = prediction.sum() * 0.0
        command_loss_unbalanced = zero_command
        command_positive_weight = zero_command
        command_negative_weight = zero_command
    event_mask = event_mask.to(dtype=gripper_error_legacy.dtype)
    hold_mask = (1.0 - event_mask) * valid
    event_denominator = (event_mask * step_weight).sum().clamp_min(1.0)
    hold_denominator = (hold_mask * step_weight).sum().clamp_min(1.0)
    event_gripper_flow = (
        gripper_error_legacy * event_mask * step_weight
    ).sum() / event_denominator
    hold_gripper_flow = (
        gripper_error_legacy * hold_mask * step_weight
    ).sum() / hold_denominator
    event_decoded_gripper = (
        decoded_gripper_error * event_mask * step_weight
    ).sum() / event_denominator
    hold_decoded_gripper = (
        decoded_gripper_error * hold_mask * step_weight
    ).sum() / hold_denominator
    event_gripper_flow_owned = (
        gripper_error_owned * event_mask * step_weight
    ).sum() / event_denominator
    hold_gripper_flow_owned = (
        gripper_error_owned * hold_mask * step_weight
    ).sum() / hold_denominator
    event_count = event_mask.sum()
    hold_count = hold_mask.sum()
    event_row_weight_mean = (event_row_weight * event_mask).sum() / event_count.clamp_min(1.0)
    hold_row_weight_mean = (event_row_weight * hold_mask).sum() / hold_count.clamp_min(1.0)
    predicted_motion = torch.sigmoid(output.bottom.motion_logits.detach().float()) >= 0.5
    target_motion = motion_target >= 0.5
    motion_true_positive = ((predicted_motion & target_motion) & valid).float().sum()
    motion_false_positive = ((predicted_motion & (~target_motion)) & valid).float().sum()
    motion_false_negative = (((~predicted_motion) & target_motion) & valid).float().sum()
    motion_precision = motion_true_positive / (
        motion_true_positive + motion_false_positive
    ).clamp_min(1.0)
    motion_recall = motion_true_positive / (
        motion_true_positive + motion_false_negative
    ).clamp_min(1.0)
    gripper_private_metrics: dict[str, Tensor] = {}
    if collect_diagnostics:
        gate = output.bottom.decoder_tensors.get("gripper_private_gate_tensor")
        private_state = output.bottom.decoder_tensors.get(
            "gripper_private_state_tensor"
        )
        state_delta = output.bottom.decoder_tensors.get(
            "gripper_private_state_delta_tensor"
        )
        if (
            not isinstance(gate, Tensor)
            or not isinstance(private_state, Tensor)
            or not isinstance(state_delta, Tensor)
        ):
            raise RuntimeError("diagnostic batch lost gripper-private tensors")
        expected_private = (
            int(target.batch),
            config.dimensions.action_horizon,
        )
        if (
            tuple(gate.shape[:2]) != expected_private
            or tuple(private_state.shape) != tuple(gate.shape)
            or tuple(state_delta.shape) != tuple(gate.shape)
        ):
            raise ValueError("gripper-private diagnostics lost [B,T,H]")
        if tuple(clean_gripper_absolute.shape) != (*expected_private, 1):
            raise ValueError("absolute gripper trajectory must be [B,T,1]")
        if tuple(clean_gripper_cumulative.shape) != (*expected_private, 1):
            raise ValueError("cumulative gripper trajectory must be [B,T,1]")

        close_class = 1 if open_class == 2 else 2
        context_masks = {
            "hold": (event_target == 0) & valid,
            "event": event_target != 0,
            "open": event_target == open_class,
            "close": event_target == close_class,
        }

        def conditional_mean(value: Tensor, mask: Tensor) -> Tensor:
            value_f = value.detach().float()
            expanded = mask.detach().float()
            while expanded.ndim < value_f.ndim:
                expanded = expanded.unsqueeze(-1)
            expanded = expanded.expand_as(value_f)
            return (value_f * expanded).sum() / expanded.sum().clamp_min(1.0)

        def conditional_rms(value: Tensor, mask: Tensor) -> Tensor:
            return conditional_mean(value.detach().float().square(), mask).sqrt()

        def register_conditional_gradient(
            value: Tensor,
            mask: Tensor,
            name: str,
        ) -> None:
            slot = value.new_zeros((), dtype=torch.float32)
            gripper_private_metrics[name] = slot
            if not value.requires_grad:
                return
            mask_f = mask.detach().float()

            def capture(gradient: Tensor) -> Tensor:
                with torch.no_grad():
                    expanded = mask_f
                    while expanded.ndim < gradient.ndim:
                        expanded = expanded.unsqueeze(-1)
                    expanded = expanded.expand_as(gradient)
                    slot.copy_(
                        (
                            (gradient.detach().float().square() * expanded).sum()
                            / expanded.sum().clamp_min(1.0)
                        ).sqrt()
                    )
                return gradient

            value.register_hook(capture)

        gripper_private_metrics.update(
            {
                "gripper_private_gate_signed_mean": gate.detach().float().mean(),
                "gripper_private_gate_saturation_fraction": (
                    gate.detach().float().abs() >= 0.95
                )
                .float()
                .mean(),
                "gripper_trajectory_mask_fraction": event_and_after_mask.detach()
                .float()
                .mean(),
                "gripper_trajectory_transition_mask_fraction": transition_mask.detach()
                .float()
                .mean(),
                "gripper_trajectory_persistence_mask_fraction": persistence_mask.detach()
                .float()
                .mean(),
                "gripper_trajectory_absolute_loss": (
                    gripper_trajectory_absolute.detach()
                ),
                "gripper_trajectory_delta_loss": gripper_trajectory_delta.detach(),
                "gripper_trajectory_transition_absolute_loss": (
                    gripper_transition_absolute.detach()
                ),
                "gripper_trajectory_transition_delta_loss": (
                    gripper_transition_delta.detach()
                ),
                "gripper_trajectory_persistence_absolute_loss": (
                    gripper_persistence_absolute.detach()
                ),
                "gripper_trajectory_persistence_delta_loss": (
                    gripper_persistence_delta.detach()
                ),
                "gripper_trajectory_branch_disagreement_rms": (
                    clean_gripper_absolute.detach().float()
                    - clean_gripper_cumulative.detach().float()
                )
                .square()
                .mean()
                .sqrt(),
            }
        )
        for context_name, context_mask in context_masks.items():
            gripper_private_metrics[
                f"gripper_private_gate_{context_name}_rms"
            ] = conditional_rms(gate, context_mask)
            gripper_private_metrics[
                f"gripper_private_state_delta_{context_name}_rms"
            ] = conditional_rms(state_delta, context_mask)
            gripper_private_metrics[
                f"gripper_trajectory_absolute_{context_name}_rms"
            ] = conditional_rms(clean_gripper_absolute, context_mask)
            gripper_private_metrics[
                f"gripper_trajectory_delta_{context_name}_rms"
            ] = conditional_rms(clean_gripper_cumulative, context_mask)
            register_conditional_gradient(
                gate,
                context_mask,
                f"gradient_tensor_gripper_private_gate_{context_name}_rms",
            )
            register_conditional_gradient(
                private_state,
                context_mask,
                f"gradient_tensor_gripper_private_state_{context_name}_rms",
            )
            register_conditional_gradient(
                clean_gripper_absolute,
                context_mask,
                f"gradient_tensor_gripper_trajectory_absolute_{context_name}_rms",
            )
            register_conditional_gradient(
                clean_gripper_cumulative,
                context_mask,
                f"gradient_tensor_gripper_trajectory_delta_{context_name}_rms",
            )
    band_metrics: dict[str, Tensor] = {}
    start = 0
    for end in (4, 12, 24):
        # Preserve the historical metric as an unweighted diagnostic.  The
        # counterfactual event-balanced geometry is reported as audit-only.
        band_metrics[f"action_flow_band_{start + 1}_{end}"] = mean_rows(
            physical_error_unweighted[:, start:end], slice(start, end)
        )
        band_metrics[f"action_flow_event_balanced_audit_band_{start + 1}_{end}"] = mean_rows(
            physical_error[:, start:end], slice(start, end)
        )
        band_metrics[f"action_horizon_weight_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].mean()
        band_metrics[f"action_horizon_mass_band_{start + 1}_{end}"] = horizon_weight[
            start:end
        ].sum() / horizon_weight.sum()
        band_metrics[f"action_flow_owned_band_{start + 1}_{end}"] = mean_rows(
            physical_error_owned_unweighted[:, start:end], slice(start, end)
        )
        start = end
    selective_mode = str(objective.calvin_frame_weight_mode)
    # Continuous outlets use the owned gripper lane.  At the default weight
    # of one this is algebraically the old V120 six-channel flow; binary
    # outlets keep their established arm-only/command branches.
    # Keep outlet/objective routing explicit. Calvin's opt-in selective profile
    # owns its row reallocation; binary outlets otherwise use the exact arm-only
    # V120 flow, while continuous outlets use the owned gripper lanes.
    if calvin_binary and selective_mode == "motion_event_v1":
        formal_action_flow = flow
    elif binary_command:
        formal_action_flow = flow_v120_comparable
    else:
        formal_action_flow = flow_owned_unbalanced
    return {
        "action_observed_label_rows": valid.float().sum(),
        "action_observed_label_fraction": valid.float().mean(),
        **band_metrics,
        **gripper_private_metrics,
        # The explicit Calvin selective profile owns row reallocation. The
        # default binary and continuous outlets retain their exact V120 paths.
        "action_flow": formal_action_flow,
        "action_flow_v120_comparable": flow_v120_comparable,
        "action_flow_owned_unbalanced": flow_owned_unbalanced,
        "action_flow_owned_delta": flow_owned_unbalanced - flow_v120_comparable,
        "action_flow_selective_unbalanced": flow_selective_unbalanced,
        "action_flow_selective_delta": (flow_selective_unbalanced - flow_v120_comparable),
        "action_flow_event_balance_delta": flow - flow_selective_unbalanced,
        "action_flow_event_balanced_audit": flow,
        "action_flow_uniform_field_mse": uniform_flow,
        "action_flow_native": native_flow,
        "action_flow_native_complete_samples": valid.all(dim=1).float().sum(),
        "action_arm_flow": arm,
        "action_gripper_flow": grip_unweighted,
        "action_gripper_flow_v120_comparable": grip_unweighted,
        "action_gripper_flow_unweighted": grip_unweighted,
        "action_gripper_flow_owned": grip_owned_unweighted,
        "action_gripper_deployed_flow": grip_deployed_unweighted,
        "action_gripper_compatibility_flow": grip_compatibility_unweighted,
        "action_gripper_compatibility_weight": prediction.new_tensor(
            float(objective.gripper_compatibility_weight), dtype=torch.float32
        ),
        "action_gripper_compatibility_fraction": owned_compatibility_fraction.detach(),
        "action_gripper_flow_owned_event_balanced_audit": grip_owned,
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
        "action_gripper_event_flow_owned": event_gripper_flow_owned,
        "action_gripper_hold_flow_owned": hold_gripper_flow_owned,
        "action_decoded_gripper_event": event_decoded_gripper,
        "action_decoded_gripper_hold": hold_decoded_gripper,
        "gripper_first_step_release": first_step_release,
        "gripper_first_step_release_rows": first_step_release_mask.detach().sum(),
        "gripper_first_step_hold": first_step_hold,
        "gripper_first_step_hold_rows": first_step_hold_mask.detach().sum(),
        "action_gripper_event_row_weight": event_row_weight_mean,
        "action_gripper_hold_row_weight": hold_row_weight_mean,
        "action_gripper_event_rate": mean_rows(event_mask),
        # Binary command-state supervision. These are zero-valued, detached
        # compatibility metrics on continuous paths.
        "gripper_command": command_loss,
        "gripper_command_unbalanced_audit": command_loss_unbalanced.detach(),
        "gripper_command_positive_class_weight": command_positive_weight.detach(),
        "gripper_command_negative_class_weight": command_negative_weight.detach(),
        "gripper_command_accuracy": command_accuracy,
        "gripper_command_positive_rate": command_positive_rate,
        "gripper_command_target_positive_rate": command_target_positive_rate,
        "gripper_command_precision": command_precision,
        "gripper_command_recall": command_recall,
        "gripper_command_f1": command_f1,
        "gripper_command_rows": command_rows_count,
        # Continuous outlets preserve the exact V120 metric. A binary outlet's
        # formal decoded objective is arm-only; its compatibility gripper field
        # is an audit and cannot add even a disconnected random scalar to the ledger.
        "decoded_action": decoded_action if binary_command else decoded_action_v120_comparable,
        "decoded_action_v120_comparable": decoded_action_v120_comparable,
        "decoded_action_selective_unbalanced": decoded_action_selective_unbalanced,
        "decoded_action_selective_delta": (
            decoded_action_selective_unbalanced - decoded_action_v120_comparable
        ),
        "decoded_action_event_balance_delta": (
            decoded_action - decoded_action_selective_unbalanced
        ),
        "decoded_action_event_balanced_audit": decoded_action,
        "smooth_delta": smooth_delta,
        "physical_delta_consistency": physical_delta_consistency,
        "gripper_trajectory": gripper_trajectory,
        "gripper_trajectory_absolute": gripper_trajectory_absolute.detach(),
        "gripper_trajectory_delta": gripper_trajectory_delta.detach(),
        "gripper_trajectory_mask_fraction": mean_rows(event_and_after_mask.detach()),
        "gripper_trajectory_transition": (
            0.5 * (gripper_transition_absolute + gripper_transition_delta)
        ).detach(),
        "gripper_trajectory_persistence": (
            0.5 * (gripper_persistence_absolute + gripper_persistence_delta)
        ).detach(),
        "gripper_trajectory_transition_mask_fraction": mean_rows(transition_mask.detach()),
        "gripper_trajectory_persistence_mask_fraction": mean_rows(persistence_mask.detach()),
        "phase_control_mode_code": prediction.new_tensor(
            1.0 if phase_mode == "stackcube_phase_v1" else 0.0,
            dtype=torch.float32,
        ),
        "phase_approach_mask_fraction": phase_approach.detach().mean(),
        "phase_transport_mask_fraction": phase_transport.detach().mean(),
        "phase_pre_event_hold_mask_fraction": phase_pre_event_hold.detach().mean(),
        "phase_arm_flow": phase_arm_flow,
        "phase_gripper_hold": pre_event_hold,
        "motion": motion,
        "motion_precision": motion_precision,
        "motion_recall": motion_recall,
        "action_horizon_weight_first": horizon_weight[0],
        "action_horizon_weight_tail": horizon_weight[-1],
        "action_frame_weight_mode_code": prediction.new_tensor(
            1.0 if selective_mode == "motion_event_v1" else 0.0,
            dtype=torch.float32,
        ),
        "action_frame_weight_mean": mean_rows(frame_weight.detach()),
        "action_frame_weight_min": frame_weight.detach().amin(),
        "action_frame_weight_max": frame_weight.detach().amax(),
        "action_frame_weight_selective_fraction": mean_rows(
            (frame_weight.detach() > 1.0001).float()
        ),
        "action_flow_first": mean_rows(physical_error_unweighted[:, 0], 0),
        "action_flow_first4": mean_rows(physical_error_unweighted[:, :4], slice(None, 4)),
        "action_flow_first8": mean_rows(physical_error_unweighted[:, :8], slice(None, 8)),
        "action_flow_tail": mean_rows(physical_error_unweighted[:, 8:], slice(8, None)),
        "action_flow_event_balanced_audit_first": mean_rows(physical_error[:, 0], 0),
        "action_flow_event_balanced_audit_first8": mean_rows(physical_error[:, :8], slice(None, 8)),
        "action_flow_event_balanced_audit_tail": mean_rows(physical_error[:, 8:], slice(8, None)),
        "action_flow_owned_first": mean_rows(physical_error_owned_unweighted[:, 0], 0),
        "action_flow_owned_first4": mean_rows(
            physical_error_owned_unweighted[:, :4], slice(None, 4)
        ),
        "action_flow_owned_first8": mean_rows(
            physical_error_owned_unweighted[:, :8], slice(None, 8)
        ),
        "action_flow_owned_tail": mean_rows(physical_error_owned_unweighted[:, 8:], slice(8, None)),
        "action_flow_owned_event_balanced_audit_first": mean_rows(physical_error_owned[:, 0], 0),
        "action_flow_owned_event_balanced_audit_first8": mean_rows(
            physical_error_owned[:, :8], slice(None, 8)
        ),
        "action_flow_owned_event_balanced_audit_tail": mean_rows(
            physical_error_owned[:, 8:], slice(8, None)
        ),
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
    action_codec: PhysicalActionFieldCodec | OutletAdapter,
    collect_diagnostics: bool = False,
) -> LossLedger:
    action = action_terms(
        config,
        action_codec,
        policy_output,
        action_target,
        history,
        flow_state,
        collect_diagnostics=collect_diagnostics,
    )
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
        interval_valid=top_targets.future_interval_valid,
        collect_diagnostics=collect_diagnostics,
    )
    objective = config.objectives
    action_group = (
        action["action_flow"]
        + objective.decoded_action * action["decoded_action"]
        + objective.gripper_command * action["gripper_command"]
        + objective.gripper_trajectory * action["gripper_trajectory"]
        + objective.gripper_first_step_release * action["gripper_first_step_release"]
        + objective.gripper_first_step_hold * action["gripper_first_step_hold"]
        + objective.phase_arm_flow * action["phase_arm_flow"]
        + objective.phase_gripper_hold * action["phase_gripper_hold"]
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
        "gripper_command": objective.gripper_command * action["gripper_command"],
        "gripper_trajectory": (
            objective.gripper_trajectory * action["gripper_trajectory"]
        ),
        "gripper_first_step_release": (
            objective.gripper_first_step_release
            * action["gripper_first_step_release"]
        ),
        "gripper_first_step_hold": (
            objective.gripper_first_step_hold * action["gripper_first_step_hold"]
        ),
        "phase_arm_flow": objective.phase_arm_flow * action["phase_arm_flow"],
        "phase_gripper_hold": objective.phase_gripper_hold * action["phase_gripper_hold"],
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
    "anchored_gripper_persistence",
    "action_terms",
    "balanced_event_row_weights",
    "calvin_selective_frame_weights",
    "causal_event_trajectory_mask",
    "compose_losses",
    "event_transition_persistence_masks",
    "execution_value_terms",
    "flow_geometry_terms",
    "future_dynamics_terms",
    "sample_flow_matching",
]
