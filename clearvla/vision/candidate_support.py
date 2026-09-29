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


def _weighted_contract(weights: Tensor, values: Tensor) -> Tensor:
    """Contract [B,Q,G,C,...] weights with [B,C,...,D] values.

    The explicit batched matrix product keeps the physical C/I/J/M/K order
    visible and avoids einsum's layout planner materializing intermediate
    transposes for the same contraction.
    """
    if weights.ndim < 4 or values.ndim != weights.ndim - 1:
        raise ValueError("weighted support contraction ranks are misaligned")
    batch, query, group = (
        int(weights.shape[0]),
        int(weights.shape[1]),
        int(weights.shape[2]),
    )
    if tuple(weights.shape[3:]) != tuple(values.shape[1:-1]):
        raise ValueError("weighted support contraction axes are misaligned")
    feature = int(values.shape[-1])
    weight_matrix = weights.reshape(batch, query * group, -1)
    value_matrix = values.reshape(batch, -1, feature)
    return torch.bmm(weight_matrix, value_matrix).reshape(batch, query, group, feature)



def candidate_support_metadata(
    mode: str, *, source_candidate_count: int = 64
) -> dict[str, object]:
    # G1 reads the complete DINO chart, not the 8x8 public query grid.
    # The default preserves the historical 64-patch metadata for callers
    # without a source; all graph/runtime boundaries must pass their source.
    if mode not in {MOMENT_LOCAL_SUPPORT, FULL_POSTERIOR_SUPPORT}:
        raise ValueError("unknown progressive candidate support mode")
    if type(source_candidate_count) is not int or source_candidate_count < 1:
        raise ValueError("source candidate count must be a positive integer")
    return {
        "contract": mode,
        "coordinates": "current_camera_normalized_xy_align_corners",
        "source": "complete_G1_camera_chart"
        if mode == FULL_POSTERIOR_SUPPORT
        else "moment_centered_local_lattice",
        "candidate_count": source_candidate_count if mode == FULL_POSTERIOR_SUPPORT else 49,
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


# Larger candidate tiles reduce grid_sample launch overhead while remaining
# within the validated B8 memory envelope.
_POSTERIOR_CANDIDATE_TILE = 16

@dataclass(frozen=True)
class PosteriorMicrogridCache:
    """Query-independent sampled support reused across P1 query chunks."""
    points: Tensor
    valid: Tensor
    rgb: Tensor
    detail: Tensor | None

    def validate(
        self,
        *,
        coordinates: Tensor,
        candidate_valid: Tensor,
        rgb_chart: Tensor,
        detail_chart: Tensor,
        microgrid_side: int,
    ) -> None:
        cells = int(microgrid_side) ** 2
        expected_prefix = (*coordinates.shape[:-2], cells)
        if tuple(self.points.shape[:-2]) != expected_prefix or self.points.shape[-1] != 2:
            raise ValueError("posterior microgrid cache points have an invalid shape")
        if tuple(self.valid.shape[:-1]) != expected_prefix:
            raise ValueError("posterior microgrid cache validity has an invalid shape")
        if tuple(self.rgb.shape[:-2]) != expected_prefix or self.rgb.shape[-1] != int(rgb_chart.shape[2]):
            raise ValueError("posterior microgrid cache RGB values have an invalid shape")
        if self.detail is not None:
            if tuple(self.detail.shape[:-2]) != expected_prefix or self.detail.shape[-1] != int(detail_chart.shape[2]):
                raise ValueError("posterior microgrid cache detail values have an invalid shape")
        if self.points.device != coordinates.device or self.valid.device != coordinates.device:
            raise ValueError("posterior microgrid cache address tensors use another device")
        if self.rgb.device != rgb_chart.device:
            raise ValueError("posterior microgrid cache RGB values use another device")
        if self.detail is not None and self.detail.device != detail_chart.device:
            raise ValueError("posterior microgrid cache detail values use another device")
        if self.valid.dtype != torch.bool:
            raise ValueError("posterior microgrid cache validity must be bool")
        if tuple(self.valid.shape[:-2]) != tuple(candidate_valid.shape[:-1]):
            raise ValueError("posterior microgrid cache lost candidate identity")


def prepare_posterior_microgrid_cache(
    coordinates: Tensor,
    radius: Tensor,
    candidate_valid: Tensor,
    rgb_chart: Tensor,
    detail_chart: Tensor,
    *,
    microgrid_side: int = 3,
    cache_detail: bool = True,
    detail_dtype: torch.dtype | None = None,
) -> PosteriorMicrogridCache:
    """Sample each query-independent local support point exactly once."""
    if coordinates.ndim != 7 or coordinates.shape[-1] != 2:
        raise ValueError("posterior cache coordinates must be [B,C,I,J,M,K,2]")
    if candidate_valid.dtype is not torch.bool or tuple(candidate_valid.shape) != tuple(coordinates.shape[:-1]):
        raise ValueError("posterior cache candidate validity must retain every candidate")
    if tuple(radius.shape) != tuple(coordinates.shape[:-2]):
        raise ValueError("posterior cache radius must retain source hypothesis identity")
    b, c = coordinates.shape[:2]
    prefix = tuple(coordinates.shape[2:-2])
    n = int(coordinates.shape[-2])
    if microgrid_side < 1:
        raise ValueError("microgrid side must be positive")
    if rgb_chart.ndim != 5 or detail_chart.ndim != 5:
        raise ValueError("posterior cache charts must be [B,C,D,H,W]")
    if tuple(rgb_chart.shape[:2]) != (b, c) or tuple(detail_chart.shape[:2]) != (b, c):
        raise ValueError("posterior cache charts lost camera identity")

    def sample(chart: Tensor, points: Tensor) -> Tensor:
        d, sy, sx = chart.shape[2:]
        n_local = int(points.shape[-2])
        value = F.grid_sample(
            chart.float().reshape(b * c, d, sy, sx),
            points.reshape(b * c, -1, n_local, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return value.permute(0, 2, 3, 1).reshape(b, c, *prefix, n_local, d)

    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        axis = (
            torch.linspace(-1.0, 1.0, microgrid_side, device=coordinates.device)
            if microgrid_side > 1
            else coordinates.new_zeros(1).float()
        )
        safe_coordinates = coordinates.float().masked_fill(
            ~candidate_valid[..., None], 0.0
        )
        safe_radius = radius.float().masked_fill(
            ~candidate_valid.any(dim=-1), 0.0
        )
        point_rows: list[Tensor] = []
        valid_rows: list[Tensor] = []
        rgb_rows: list[Tensor] = []
        detail_rows: list[Tensor] = []
        center_index = microgrid_side**2 // 2
        for cell_index, (y, x) in enumerate(
            ((y, x) for y in axis.unbind() for x in axis.unbind())
        ):
            offset = torch.stack((x, y))
            points = safe_coordinates + safe_radius[..., None, None] * offset
            valid = (points.abs() <= 1.0).all(dim=-1) & candidate_valid
            if microgrid_side % 2 and cell_index == center_index:
                sampled_rgb = torch.zeros(
                    b, c, *prefix, n, int(rgb_chart.shape[2]),
                    device=rgb_chart.device, dtype=torch.float32,
                )
                sampled_detail = torch.zeros(
                    b, c, *prefix, n, int(detail_chart.shape[2]),
                    device=detail_chart.device,
                    dtype=torch.float32 if detail_dtype is None else detail_dtype,
                )
            else:
                sampled_rgb = sample(rgb_chart, points)
                sampled_detail = sample(detail_chart, points) if cache_detail else None
                if sampled_detail is not None and detail_dtype is not None:
                    sampled_detail = sampled_detail.to(dtype=detail_dtype)
            point_rows.append(points)
            valid_rows.append(valid)
            rgb_rows.append(sampled_rgb)
            if cache_detail:
                assert sampled_detail is not None
                detail_rows.append(sampled_detail)
    cache = PosteriorMicrogridCache(
        points=torch.stack(point_rows, dim=-3),
        valid=torch.stack(valid_rows, dim=-2),
        rgb=torch.stack(rgb_rows, dim=-3),
        detail=torch.stack(detail_rows, dim=-3) if cache_detail else None,
    )
    cache.validate(
        coordinates=coordinates,
        candidate_valid=candidate_valid,
        rgb_chart=rgb_chart,
        detail_chart=detail_chart,
        microgrid_side=microgrid_side,
    )
    return cache


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
    cache: PosteriorMicrogridCache | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a real local patch at EACH possible location before marginalizing.

    A fixed 7x7-offset -> 3x3 basis cannot describe a global source lattice.
    Replacing it by a resized global basis would turn a local precision read
    into a whole-camera thumbnail. Keep the same local radius and ordered
    micro cells, while retaining every candidate in their expectation.
    Values outside the observed image are not manufactured by border padding.

    The weighted contraction is tiled over candidate support points. This
    preserves the complete posterior expectation while avoiding a transient
    [B,C,I,J,M,K,D] sampled-value volume during the second chart read.
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
        n_local = int(points.shape[-2])
        value = F.grid_sample(
            chart.float().reshape(b * c, d, sy, sx),
            points.reshape(b * c, -1, n_local, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return value.permute(0, 2, 3, 1).reshape(b, c, *prefix, n_local, d)

    def weighted_dense(chart: Tensor, points: Tensor, weights: Tensor) -> Tensor:
        partials: list[Tensor] = []
        tile = _POSTERIOR_CANDIDATE_TILE
        point_tiles = points.split(tile, dim=-2)
        weight_tiles = weights.split(tile, dim=-1)
        valid_tiles = candidate_valid.split(tile, dim=-1)
        for point_tile, weight_tile, valid_tile in zip(
            point_tiles, weight_tiles, valid_tiles, strict=True
        ):
            sampled = sample(chart, point_tile).masked_fill(
                ~valid_tile[..., None], 0.0
            )
            partials.append(
                torch.einsum(
                    "bqgcijmk,bcijmkv->bqgv",
                    weight_tile,
                    sampled,
                )
            )
        return torch.stack(partials, dim=0).sum(dim=0)

    def weighted_cached_dense(
        sampled_values: Tensor,
        weights: Tensor,
        validity: Tensor,
    ) -> Tensor:
        partials: list[Tensor] = []
        tile = _POSTERIOR_CANDIDATE_TILE
        value_tiles = sampled_values.split(tile, dim=-2)
        weight_tiles = weights.split(tile, dim=-1)
        validity_tiles = validity.split(tile, dim=-1)
        for value_tile, weight_tile, validity_tile in zip(
            value_tiles, weight_tiles, validity_tiles, strict=True
        ):
            safe_values = value_tile.float().masked_fill(
                ~validity_tile[..., None], 0.0
            )
            partials.append(
                _weighted_contract(weight_tile, safe_values)
            )
        return torch.stack(partials, dim=0).sum(dim=0)

    def weighted_center(
        values: Tensor,
        weights: Tensor,
        validity: Tensor,
    ) -> Tensor:
        partials: list[Tensor] = []
        tile = _POSTERIOR_CANDIDATE_TILE
        value_tiles = values.split(tile, dim=-2)
        weight_tiles = weights.split(tile, dim=-1)
        validity_tiles = validity.split(tile, dim=-1)
        for value_tile, weight_tile, validity_tile in zip(
            value_tiles, weight_tiles, validity_tiles, strict=True
        ):
            safe_values = value_tile.float().masked_fill(
                ~validity_tile[..., None], 0.0
            )
            partials.append(
                _weighted_contract(weight_tile, safe_values)
            )
        return torch.stack(partials, dim=0).sum(dim=0)

    if cache is not None:
        cache.validate(
            coordinates=coordinates,
            candidate_valid=candidate_valid,
            rgb_chart=rgb_chart,
            detail_chart=detail_chart,
            microgrid_side=microgrid_side,
        )
    rgb_rows: list[Tensor] = []
    detail_rows: list[Tensor] = []
    coord_rows: list[Tensor] = []
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        axis = (
            torch.linspace(-1.0, 1.0, microgrid_side, device=coordinates.device)
            if microgrid_side > 1
            else coordinates.new_zeros(1).float()
        )
        for cell_index, (y, x) in enumerate(
            ((y, x) for y in axis.unbind() for x in axis.unbind())
        ):
            offset = torch.stack((x, y))
            if cache is None:
                safe_coordinates = coordinates.float().masked_fill(
                    ~candidate_valid[..., None], 0.0
                )
                safe_radius = radius.float().masked_fill(
                    ~candidate_valid.any(dim=-1), 0.0
                )
                points = safe_coordinates + safe_radius[..., None, None] * offset
                valid = (points.abs() <= 1.0).all(dim=-1) & candidate_valid
            else:
                points = cache.points[..., cell_index, :, :]
                valid = cache.valid[..., cell_index, :]
            observed = valid[:, None, None]
            logits = fine_logits.float().masked_fill(
                ~observed, torch.finfo(torch.float32).min
            )
            supported = observed.any(dim=-1, keepdim=True)
            safe_logits = logits.masked_fill(~supported, 0.0)
            weights = torch.softmax(safe_logits, dim=-1) * observed.float()
            joint = route_weights.float()[..., None] * weights
            if microgrid_side % 2 and cell_index == microgrid_side**2 // 2:
                rgb_rows.append(
                    weighted_center(center_rgb, joint, candidate_valid)
                )
                detail_rows.append(
                    weighted_center(center_detail, joint, candidate_valid)
                )
            else:
                if cache is None:
                    rgb_rows.append(weighted_dense(rgb_chart, points, joint))
                    detail_rows.append(weighted_dense(detail_chart, points, joint))
                else:
                    rgb_rows.append(
                        weighted_cached_dense(
                            cache.rgb[..., cell_index, :, :],
                            joint,
                            valid,
                        )
                    )
                    if cache.detail is None:
                        detail_rows.append(weighted_dense(detail_chart, points, joint))
                    else:
                        detail_rows.append(
                            weighted_cached_dense(
                                cache.detail[..., cell_index, :, :],
                                joint,
                                valid,
                            )
                        )
            coord_rows.append(_weighted_contract(joint, points))
    return torch.stack(rgb_rows, -2), torch.stack(detail_rows, -2), torch.stack(coord_rows, -2)
