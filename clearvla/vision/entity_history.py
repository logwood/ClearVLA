"""Causal image correspondence for the entity binder, not persistent object IDs.

The existing directed flow maps current/source cells into an older image.
All past evidence is pulled onto the CURRENT image chart before local candidate
quadrature. Image correspondences are model estimates: their coverage never
changes the producer's observed-cell/label masks or establishes physical identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .candidate_support import sample_candidate_expectation
from .entity_chart import CurrentImageSupport, current_image_grid
from .source_time import VisualSourceTime

NO_ENTITY_HISTORY = "current_only_v1"
CAUSAL_ENTITY_HISTORY = "flow_pulled_history_v1"


def entity_history_metadata(mode: str) -> dict[str, object]:
    if mode not in (NO_ENTITY_HISTORY, CAUSAL_ENTITY_HISTORY):
        raise ValueError("unknown entity history mode")
    return {
        "mode": mode,
        "identity": "current_entities_with_finite_causal_window_not_persistent_ids",
        "flow": "current_source_to_previous_normalized_image_coordinates",
        "time": "actual_control_steps_not_seconds_or_solver_nodes",
        "observed_authority": "source_clock_and_context_mask_only",
        "consumer": "K_competition_keys_and_slot_updates",
        "reset": "recompute_from_the_same_causal_window_in_train_and_online",
    }


@dataclass(frozen=True)
class ObservedEntityHistory:
    """Online-only, full-width observed charts and estimated backward flow."""

    content: Tensor  # [B,T,C,Y,X,D], same normalization as current observed DINO
    observed: Tensor  # bool [B,T,C,Y,X], source clock AND context-dropout support
    source_time: VisualSourceTime
    backward_flow: Tensor  # [B,T-1,C,2,Y,X], on later SOURCE cells, normalized xy
    confidence: Tensor  # [B,T-1,C,1,Y,X], inferred, never observational validity
    occlusion: Tensor  # same shape; not a source mask

    def validate(self) -> None:
        if self.content.ndim != 6:
            raise ValueError("entity history content must be [B,T,C,Y,X,D]")
        b, t, c, y, x, d = self.content.shape
        if min(b, c, y, x, d) < 1:
            raise ValueError("entity history axes must be nonempty")
        self.source_time.validate(batch=b, frames=t, device=self.content.device, strict=True)
        if self.observed.shape != self.content.shape[:-1] or self.observed.dtype != torch.bool:
            raise ValueError("entity history observed support must be Boolean [B,T,C,Y,X]")
        if self.backward_flow.shape != (b, t - 1, c, 2, y, x):
            raise ValueError("entity history flow must be later-source indexed [B,T-1,C,2,Y,X]")
        if (
            self.confidence.shape != (b, t - 1, c, 1, y, x)
            or self.occlusion.shape != self.confidence.shape
        ):
            raise ValueError("entity history flow status lost camera/pair axes")
        for v in (self.observed, self.backward_flow, self.confidence, self.occlusion):
            if v.device != self.content.device:
                raise ValueError("entity history tensors must share device")
        frames = self.source_time.frame_observed[:, :, None, None, None]
        if bool((self.observed & ~frames).any()):
            raise ValueError("duplicate source frame must not masquerade as observed history")
        if not bool(torch.isfinite(self.content.masked_select(self.observed[..., None])).all()):
            raise ValueError("observed entity content must be finite")
        pairs = self.source_time.pair_observed[:, :, None, None, None, None]
        for v in (self.backward_flow, self.confidence, self.occlusion):
            if not bool(torch.isfinite(v.masked_select(pairs.expand_as(v))).all()):
                raise ValueError("real observed pair requires finite estimated flow/status")


@dataclass(frozen=True)
class PulledEntityHistory:
    """Older image evidence in current pixel coordinates; no K axis yet."""

    values: Tensor  # [B,T-1,C,Y,X,R], masked interpolation, NOT coverage-normalized
    coordinates: Tensor  # [B,T-1,C,Y,X,2], inferred older-image locations
    coverage: Tensor  # [B,T-1,C,Y,X,1], interpolated observed source support
    confidence: Tensor  # same prefix, inferred path confidence
    occlusion: Tensor  # same prefix, inferred path occlusion
    offsets: Tensor  # [B,T-1], physical source offsets


def _pull(image: Tensor, coordinates: Tensor) -> Tensor:
    """[B,C,Y,X,D] pull at [B,C,Y,X,2], with explicit FP32 interpolation."""
    b, c, y, x, d = image.shape
    with torch.autocast(device_type=image.device.type, enabled=False):
        sampled = F.grid_sample(
            image.float().permute(0, 1, 4, 2, 3).reshape(b * c, d, y, x),
            coordinates.float().reshape(b * c, y, x, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    return sampled.reshape(b, c, d, y, x).permute(0, 1, 3, 4, 2)


def pull_causal_history(history: ObservedEntityHistory, projected: Tensor) -> PulledEntityHistory:
    """Compose actual adjacent inverse maps, not a scaled latest-flow guess.

    A duplicate frame is an identity step, not a new observation. Leaving the
    image terminates that estimated path; an out-of-view point cannot return
    as an invented observation. Partial masked interpolation retains its
    mass and is never divided by tiny coverage.
    """
    history.validate()
    if projected.shape[:-1] != history.content.shape[:-1]:
        raise ValueError("projected history lost its source axes")
    if projected.device != history.content.device:
        raise ValueError("projected history has a different device")
    b, t, c, y, x, _ = projected.shape
    grid = current_image_grid(y, x, device=projected.device)[None, None].expand(b, c, y, x, 2)
    coordinates = grid
    alive = history.observed[:, -1, ..., None]
    path_confidence = torch.ones_like(coordinates[..., :1])
    path_survival = torch.ones_like(path_confidence)
    safe_values = torch.where(history.observed[..., None], projected, 0.0)
    values: list[Tensor] = []
    points: list[Tensor] = []
    coverages: list[Tensor] = []
    confidences: list[Tensor] = []
    occlusions: list[Tensor] = []
    for i in range(t - 2, -1, -1):
        real_pair = history.source_time.pair_observed[:, i, None, None, None, None]
        flow = torch.where(real_pair, history.backward_flow[:, i], 0.0).permute(0, 1, 3, 4, 2)
        conf = torch.where(real_pair, history.confidence[:, i], 0.0).permute(0, 1, 3, 4, 2)
        occ = torch.where(real_pair, history.occlusion[:, i], 0.0).permute(0, 1, 3, 4, 2)
        safe_coordinates = torch.where(alive, coordinates, 0.0)
        # All of these fields are indexed on the LATER frame of this pair.
        step = _pull(flow, safe_coordinates)
        step_confidence = _pull(conf, safe_coordinates).clamp(0.0, 1.0)
        step_occlusion = _pull(occ, safe_coordinates).clamp(0.0, 1.0)
        coordinates = torch.where(real_pair, safe_coordinates + step, safe_coordinates)
        alive = alive & (coordinates.abs() <= 1.0).all(-1, keepdim=True)
        path_confidence = path_confidence * torch.where(real_pair, step_confidence, 1.0)
        path_survival = path_survival * torch.where(real_pair, 1.0 - step_occlusion, 1.0)
        frame = history.source_time.frame_observed[:, i, None, None, None, None]
        eligible = alive & frame
        safe_coordinates = torch.where(eligible, coordinates, 0.0)
        coverage = torch.where(
            eligible, _pull(history.observed[:, i, ..., None].float(), safe_coordinates), 0.0
        ).clamp(0.0, 1.0)
        values.append(torch.where(eligible, _pull(safe_values[:, i], safe_coordinates), 0.0))
        points.append(torch.where(eligible, safe_coordinates, 0.0))
        coverages.append(coverage)
        confidences.append(torch.where(eligible, path_confidence, 0.0))
        occlusions.append(torch.where(eligible, 1.0 - path_survival, 0.0))
    return PulledEntityHistory(
        torch.stack(values[::-1], 1),
        torch.stack(points[::-1], 1),
        torch.stack(coverages[::-1], 1),
        torch.stack(confidences[::-1], 1),
        torch.stack(occlusions[::-1], 1),
        history.source_time.frame_offsets[:, :-1],
    )


class CausalEntityEvidence(nn.Module):
    """Time-aware, camera-local correspondence evidence for current K grouping.

    Projection happens before expansion over candidates. This avoids a dense
    candidate x past-image x full-DINO-channel tensor. All retained G1 points
    are still consumed by the existing tiled exact quadrature.
    """

    def __init__(self, content_dim: int, route_dim: int) -> None:
        super().__init__()
        self.content_projection = nn.Sequential(
            nn.LayerNorm(content_dim, elementwise_affine=False),
            nn.Linear(content_dim, route_dim, bias=False),
        )
        self.relation = nn.Sequential(
            nn.Linear(2 * route_dim + 7, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, route_dim, bias=False),
        )

    def forward(self, history: ObservedEntityHistory, support: CurrentImageSupport) -> Tensor:
        history.validate()
        support.validate()
        safe = torch.where(history.observed[..., None], history.content, 0.0)
        projected = self.content_projection(safe)
        pulled = pull_causal_history(history, projected)
        age = -pulled.offsets[:, :, None, None, None, None].float()
        duration = age.clamp_min(1.0)
        current = projected[:, -1, None].float()
        current = torch.where(history.observed[:, -1, None, ..., None], current, 0.0)
        coverage = pulled.coverage
        y, x = projected.shape[3:5]
        grid = current_image_grid(y, x, device=projected.device)[None, None, None]
        # Pixel differences are camera-chart evidence, not world-frame velocity.
        fields = torch.cat(
            (
                pulled.values,
                (current * coverage - pulled.values) / duration,
                (grid - pulled.coordinates) * coverage / duration,
                coverage,
                pulled.confidence * coverage,
                pulled.occlusion * coverage,
                coverage / duration,
                coverage * age.log1p(),
            ),
            dim=-1,
        )
        relation = self.relation(fields.to(projected.dtype))
        # Denominator depends ONLY on source frames, never confidence/coverage.
        # Zero correspondence cannot fabricate an observation through an affine bias.
        source_count = history.source_time.frame_observed[:, :-1].sum(1).clamp_min(1)
        chart = relation.sum(1) / source_count[:, None, None, None, None].to(relation.dtype)
        coordinates = torch.where(support.valid[..., None], support.coordinates, 0.0)
        probability = torch.where(support.valid, support.probability, 0.0)
        result = sample_candidate_expectation(chart, coordinates, probability)
        return result
