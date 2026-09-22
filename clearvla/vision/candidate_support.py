"""Spatial support and quadrature for the real progressive G1/G2 reader.

The support of a multimodal address distribution is not its mean coordinate.
This module has no parameters, task input, additional evidence, or top-k quota.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

MOMENT_LOCAL_SUPPORT = "moment_local_v1"
FULL_POSTERIOR_SUPPORT = "full_posterior_lattice_v1"


def candidate_support_metadata(mode: str) -> dict[str, object]:
    if mode not in {MOMENT_LOCAL_SUPPORT, FULL_POSTERIOR_SUPPORT}:
        raise ValueError("unknown progressive candidate support mode")
    return {
        "contract": mode,
        "coordinates": "current_camera_normalized_xy_align_corners",
        "source": "complete_G1_camera_chart"
        if mode == FULL_POSTERIOR_SUPPORT
        else "moment_centered_local_lattice",
        "candidate_count": 64 if mode == FULL_POSTERIOR_SUPPORT else 49,
        "parent_measure": "FP32_log_probability"
        if mode == FULL_POSTERIOR_SUPPORT
        else "moment_gaussian",
        "current_content": "candidate_quadrature"
        if mode == FULL_POSTERIOR_SUPPORT
        else "sample_at_mean",
        "persistent_entity_identity": False,
    }


@dataclass(frozen=True)
class PosteriorCandidateSupport:
    current_coordinates: Tensor
    source_coordinates: Tensor
    relative_coordinates: Tensor
    parent_log_probability: Tensor


def posterior_candidate_support(
    *,
    positions: Tensor,
    logits: Tensor,
    correction: Tensor,
    source_centers: Tensor,
    support: Tensor,
) -> PosteriorCandidateSupport:
    """Keep every G1 support point; rectify points, never replace them by E[x].

    G2 supplies the existing bounded coordinate proposal. The edge-relative
    map is applied separately at each support point, not at a global mean.
    Source coordinates retain the actual source query, so far candidate
    matches do not become fictitious out-of-frame historical observations.
    """
    if positions.ndim != 4 or positions.shape[-1] != 2 or logits.ndim != 6:
        raise ValueError("posterior support expects [B,C,N,2] and [B,C,Y,X,M,N]")
    if (
        tuple(positions.shape[:2]) != tuple(logits.shape[:2])
        or positions.shape[2] != logits.shape[-1]
    ):
        raise ValueError("posterior support lost camera/candidate correspondence")
    prefix = tuple(logits.shape[:-1])
    if tuple(correction.shape) != (*prefix, 2) or tuple(support.shape) != prefix:
        raise ValueError(
            "posterior support correction and support must keep the local hypothesis axes"
        )
    if tuple(source_centers.shape) != (*prefix[:4], 2):
        raise ValueError("posterior support source centers lost query-chart identity")
    if any(x.device != logits.device for x in (positions, correction, support, source_centers)):
        raise ValueError("posterior support must share one device")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        base = positions.float()[:, :, None, None, None].expand(*logits.shape, 2)
        current = base + (1.0 - base.square()).clamp_min(0.0) * correction.float()[..., None, :]
        source = source_centers.float()[..., None, None, :].expand_as(current)
        relative = (current - source) / support.float()[..., None, None].clamp_min(1e-4)
        log_probability = F.log_softmax(logits.float(), dim=-1)
    return PosteriorCandidateSupport(current, source, relative, log_probability)


def sample_candidate_expectation(chart: Tensor, coordinates: Tensor, probability: Tensor) -> Tensor:
    """Read observed content at its actual support, rather than at a barycenter.

    The caller owns whether values are observed DINO or learned current G3
    context. Activation recomputation avoids retaining a full
    [B,C,Y,X,M,N,D] expansion; value, coordinate and probability gradients
    remain ordinary autograd paths. This is
    not a second encoder call or an extra physical/ODE update.
    """
    if chart.ndim != 5 or coordinates.ndim != 7 or coordinates.shape[-1] != 2:
        raise ValueError("candidate expectation needs [B,C,S,S,D] and [B,C,Y,X,M,N,2]")
    if tuple(coordinates.shape[:-1]) != tuple(probability.shape) or tuple(chart.shape[:2]) != tuple(
        coordinates.shape[:2]
    ):
        raise ValueError("candidate expectation value, address and measure do not align")
    b, c, sy, sx, _ = chart.shape
    prefix = tuple(probability.shape[2:-1])
    n = int(probability.shape[-1])

    def read(values: Tensor, points: Tensor, weights: Tensor) -> Tensor:
        with torch.autocast(device_type=values.device.type, enabled=False):
            d = int(values.shape[-1])
            sampled = F.grid_sample(
                values.permute(0, 1, 4, 2, 3).reshape(b * c, d, sy, sx).float(),
                points.float().reshape(b * c, -1, n, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            result = (sampled * weights.float().reshape(b * c, 1, -1, n)).sum(dim=-1)
            return result.transpose(1, 2).reshape(b, c, *prefix, d).to(values.dtype)

    outputs: list[Tensor] = []
    # Channel tiling is exact: no candidate/value channel is discarded. It
    # bounds the transient full-content volume even for the real 768-D chart.
    for start in range(0, int(chart.shape[-1]), 128):
        values = chart[..., start : start + 128]
        if torch.is_grad_enabled() and (
            chart.requires_grad or coordinates.requires_grad or probability.requires_grad
        ):
            output = checkpoint(
                read,
                values,
                coordinates,
                probability,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if not isinstance(output, Tensor):
                raise TypeError("candidate expectation checkpoint must return a Tensor")
            outputs.append(output)
        else:
            outputs.append(read(values, coordinates, probability))
    return torch.cat(outputs, dim=-1)


def posterior_microgrid_expectation(
    route_weights: Tensor,
    fine_logits: Tensor,
    candidate_valid: Tensor,
    coordinates: Tensor,
    radius: Tensor,
    rgb_chart: Tensor,
    detail_chart: Tensor,
    center_rgb: Tensor,
    center_detail: Tensor,
    *,
    microgrid_side: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a real local patch at EACH possible location before marginalizing.

    A fixed 7x7-offset -> 3x3 basis cannot describe a global source lattice.
    Replacing it by a resized global basis would turn a local precision read
    into a whole-camera thumbnail. Keep the same local radius and ordered
    micro cells, while retaining every candidate in their expectation.
    Values outside the observed image are not manufactured by border padding.
    The center cell reuses the exact already sampled RGB/detail values.
    """
    if fine_logits.ndim != 8 or tuple(route_weights.shape) != tuple(fine_logits.shape[:-1]):
        raise ValueError("posterior microgrid route and candidate measures do not align")
    if (
        tuple(coordinates.shape[:-1]) != (fine_logits.shape[0], *fine_logits.shape[3:])
        or coordinates.shape[-1] != 2
    ):
        raise ValueError("posterior microgrid lost actual candidate coordinates")
    if candidate_valid.dtype != torch.bool or tuple(candidate_valid.shape) != tuple(
        coordinates.shape[:-1]
    ):
        raise ValueError("posterior microgrid validity must retain every candidate")
    if tuple(radius.shape) != tuple(coordinates.shape[:-2]):
        raise ValueError("posterior microgrid radius must retain source hypothesis identity")
    if microgrid_side < 1:
        raise ValueError("microgrid side must be positive")
    if tuple(center_rgb.shape[:-1]) != tuple(radius.shape) + (fine_logits.shape[-1],) or tuple(
        center_detail.shape[:-1]
    ) != tuple(center_rgb.shape[:-1]):
        raise ValueError("posterior microgrid cached center values do not match support")
    b, c = coordinates.shape[:2]
    n = int(coordinates.shape[-2])
    prefix = tuple(coordinates.shape[2:-2])

    def sample(chart: Tensor, points: Tensor) -> Tensor:
        if chart.ndim != 5 or tuple(chart.shape[:2]) != (b, c):
            raise ValueError("posterior microgrid dense values lost camera identity")
        d, sy, sx = chart.shape[2:]
        value = F.grid_sample(
            chart.float().reshape(b * c, d, sy, sx),
            points.reshape(b * c, -1, n, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return value.permute(0, 2, 3, 1).reshape(b, c, *prefix, n, d)

    rgb_rows: list[Tensor] = []
    detail_rows: list[Tensor] = []
    coord_rows: list[Tensor] = []
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        axis = (
            torch.linspace(-1.0, 1.0, microgrid_side, device=coordinates.device)
            if microgrid_side > 1
            else coordinates.new_zeros(1).float()
        )
        for y in axis.unbind():
            for x in axis.unbind():
                offset = torch.stack((x, y))
                safe_coordinates = torch.where(
                    candidate_valid[..., None],
                    coordinates.float(),
                    torch.zeros_like(coordinates, dtype=torch.float32),
                )
                safe_radius = torch.where(
                    candidate_valid.any(dim=-1),
                    radius.float(),
                    torch.zeros_like(radius, dtype=torch.float32),
                )
                points = safe_coordinates + safe_radius[..., None, None] * offset
                valid = (points.abs() <= 1.0).all(dim=-1) & candidate_valid
                observed = valid[:, None, None]
                logits = fine_logits.float().masked_fill(~observed, torch.finfo(torch.float32).min)
                supported = observed.any(dim=-1, keepdim=True)
                safe_logits = torch.where(supported, logits, torch.zeros_like(logits))
                # Normalize in logit space; conditioning a tiny probability
                # sum by division can overflow its reverse derivative.
                weights = torch.softmax(safe_logits, dim=-1) * observed.float()
                joint = route_weights.float()[..., None] * weights
                # Use integer loop positions below to avoid a device sync.
                cell_index = len(rgb_rows)
                if microgrid_side % 2 and cell_index == microgrid_side**2 // 2:
                    rgb, detail = center_rgb.float(), center_detail.float()
                else:
                    rgb, detail = sample(rgb_chart, points), sample(detail_chart, points)
                rgb = torch.where(valid[..., None], rgb, torch.zeros_like(rgb))
                detail = torch.where(valid[..., None], detail, torch.zeros_like(detail))
                rgb_rows.append(torch.einsum("bqgcijmk,bcijmkv->bqgv", joint, rgb))
                detail_rows.append(torch.einsum("bqgcijmk,bcijmkv->bqgv", joint, detail))
                coord_rows.append(torch.einsum("bqgcijmk,bcijmkd->bqgd", joint, points))
    return torch.stack(rgb_rows, -2), torch.stack(detail_rows, -2), torch.stack(coord_rows, -2)
