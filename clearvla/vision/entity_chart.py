"""Current-image pushforward of a local hypothesis measure.

Query indices are not image positions. The observed coordinate distribution is
retained until an actual image read/write; object identity stays on the K axis.
No observation, task condition, persistent state or learnable gate is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

QUERY_CHART = "query_lattice_v1"
CURRENT_IMAGE_CHART = "current_image_support_v1"


def entity_chart_metadata(mode: str) -> dict[str, object]:
    if mode not in {QUERY_CHART, CURRENT_IMAGE_CHART}:
        raise ValueError("unknown entity chart mode")
    return {
        "schema": mode,
        "assignment_axes": "query_camera_y_x_local_hypothesis",
        "reverse_lookup_axes": (
            "current_camera_y_x" if mode == CURRENT_IMAGE_CHART else "query_camera_y_x"
        ),
        "coordinates": "normalized_xy_align_corners",
        "spatial_measure": "full_candidate_bilinear"
        if mode == CURRENT_IMAGE_CHART
        else "query_cell",
        "reconstruction_support": "observed_cells_not_predicted_coverage",
    }


@dataclass(frozen=True)
class CurrentImageSupport:
    """Producer-owned locations and conditional law, with local M/N intact.

    Shapes: coordinates [B,C,Yq,Xq,M,N,2], probability and valid
    [B,C,Yq,Xq,M,N]. All coordinates belong to the CURRENT camera image.
    Inactive payload may be arbitrary and is quarantined before arithmetic.
    A supported local hypothesis has a normalized conditional distribution;
    its prior/validity/real-vs-null mass remain separate upstream quantities.
    """

    coordinates: Tensor
    probability: Tensor
    valid: Tensor
    log_probability: Tensor

    def validate(self, local_validity: Tensor | None = None) -> None:
        if self.coordinates.ndim != 7 or self.coordinates.shape[-1] != 2:
            raise ValueError("current image coordinates must be [B,C,Yq,Xq,M,N,2]")
        prefix = self.coordinates.shape[:-1]
        if self.probability.shape != prefix or self.valid.shape != prefix:
            raise ValueError("current image location/probability/support axes differ")
        if self.valid.dtype != torch.bool or self.probability.dtype != torch.float32:
            raise TypeError(
                "current image support requires Boolean validity and FP32 probabilities"
            )
        if not self.coordinates.is_floating_point():
            raise TypeError("current image coordinates must be floating point")
        if len({self.coordinates.device, self.probability.device, self.valid.device}) != 1:
            raise ValueError("current image support must use one device")
        active = self.valid
        p = self.probability
        if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
            raise ValueError("current image probabilities must be finite and nonnegative")
        if bool((p.masked_select(~active) != 0).any()):
            raise ValueError("unsupported current image candidates must have zero probability")
        coords = self.coordinates.masked_select(active[..., None])
        if not bool(torch.isfinite(coords).all()) or bool((coords.abs() > 1.0 + 1e-6).any()):
            raise ValueError("supported current image coordinates must lie in [-1,1]")
        has_support = active.any(-1)
        mass = p.sum(-1)
        if not torch.allclose(mass, has_support.float(), rtol=2e-5, atol=2e-6):
            raise ValueError("current image conditional probability is not normalized on support")
        log_p = self.log_probability
        if log_p.shape != prefix or log_p.device != p.device or log_p.dtype != torch.float32:
            raise ValueError("current image log probability must be aligned FP32")
        if not bool(torch.isfinite(log_p).all()):
            raise ValueError("current image log probability must be finite")
        expected_p = torch.where(active, log_p.exp(), 0.0)
        if not torch.allclose(p, expected_p, rtol=2e-5, atol=2e-6):
            raise ValueError("current image probability/log probability disagree")
        # Normalization is also checked in log space to retain underflowed mass.
        safe_log = log_p.masked_fill(~active, -torch.inf)
        safe_log = torch.where(has_support[..., None], safe_log, 0.0)
        totals = torch.logsumexp(safe_log, -1)[has_support]
        if not torch.allclose(totals, torch.zeros_like(totals), atol=2e-5, rtol=0):
            raise ValueError("current image log measure is not normalized")
        if local_validity is not None:
            if local_validity.shape != (*prefix[:-1], 1):
                raise ValueError("current image support lost local hypothesis axes")
            if local_validity.device != p.device:
                raise ValueError("current image/local validity devices differ")
            if not bool(torch.isfinite(local_validity).all()):
                raise ValueError("local source validity must be finite")
            if not torch.equal(local_validity[..., 0] > 0, has_support):
                raise ValueError("current image support and local source validity disagree")


def current_image_grid(rows: int, columns: int, *, device: torch.device) -> Tensor:
    """Normalized pixel centers in the SAME align-corners convention as reads."""
    if min(rows, columns) < 1:
        raise ValueError("current image grid dimensions must be positive")
    x = (
        torch.linspace(-1, 1, columns, device=device)
        if columns > 1
        else torch.zeros(1, device=device)
    )
    y = torch.linspace(-1, 1, rows, device=device) if rows > 1 else torch.zeros(1, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1)


def pushforward_to_current_image(
    measure: Tensor, support: CurrentImageSupport, *, rows: int, columns: int
) -> Tensor:
    """Push [B,K,C,Yq,Xq,M] mass to [B,K,C,Y,X] without a barycenter.

    Bilinear weights conserve each local hypothesis's mass, including image
    edges and one-pixel axes. Accumulation stays FP32 under autocast. Raw
    invalid values are removed before floor/index/product operations. This is
    a piecewise-differentiable spatial write, not a newly observed image.
    """
    support.validate()
    shape = support.probability.shape
    if measure.ndim != 6 or measure.shape[0] != shape[0] or measure.shape[2:] != shape[1:-1]:
        raise ValueError("image pushforward expects [B,K,C,Yq,Xq,M] source mass")
    if measure.device != support.coordinates.device:
        raise ValueError("image pushforward requires one device")
    if min(rows, columns) < 1:
        raise ValueError("image pushforward dimensions must be positive")
    valid_local = support.valid.any(-1)[:, None]
    safe_measure = torch.where(
        valid_local, measure.float(), torch.zeros_like(measure, dtype=torch.float32)
    )
    if not bool(torch.isfinite(safe_measure).all()) or bool((safe_measure < 0).any()):
        raise ValueError("supported image pushforward mass must be finite and nonnegative")
    with torch.autocast(device_type=measure.device.type, enabled=False):
        xy = torch.where(support.valid[..., None], support.coordinates.float(), 0.0).clamp(-1, 1)
        x = (xy[..., 0] + 1) * ((columns - 1) / 2)
        y = (xy[..., 1] + 1) * ((rows - 1) / 2)
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        weights = safe_measure[..., None] * support.probability[:, None]
        batch, objects, cameras = measure.shape[:3]
        values = weights.reshape(batch, objects, cameras, -1)
        image = values.new_zeros(batch, objects, cameras, rows * columns)
        for ox, oy, weight in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ):
            index = (y0.long() + oy).clamp_max(rows - 1) * columns + (x0.long() + ox).clamp_max(
                columns - 1
            )
            index = index.reshape(batch, 1, cameras, -1).expand(-1, objects, -1, -1)
            contribution = values * weight.reshape(batch, 1, cameras, -1)
            image = image.scatter_add(-1, index, contribution)
        return image.reshape(batch, objects, cameras, rows, columns)


def current_image_centers(posterior: Tensor) -> Tensor:
    """One geometric summary derived from the exported image posterior."""
    if posterior.ndim != 5:
        raise ValueError("image posterior must be [B,K,C,Y,X]")
    grid = current_image_grid(posterior.shape[-2], posterior.shape[-1], device=posterior.device)
    mass = posterior.float().sum((-2, -1), keepdim=True)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    return (posterior.float()[..., None] * grid).sum((-3, -2)) / safe_mass[..., 0]


@dataclass(frozen=True)
class ImageLogMeasure:
    """FP32 log mass plus explicit support on [B,K,C,Y,X].

    Zero probability after exp may mean numerical underflow, not absence. The
    log measure is authoritative for conditional reads; an empty segment has
    finite zero payload and a false support bit, never a fake uniform image.
    """

    log_mass: Tensor
    supported: Tensor

    def normalized(self, axes: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        if self.log_mass.ndim != 5 or self.supported.shape != self.log_mass.shape:
            raise ValueError("image log measure must keep [B,K,C,Y,X]")
        if self.log_mass.dtype != torch.float32 or self.supported.dtype != torch.bool:
            raise TypeError("image log measure requires FP32 and Boolean support")
        if self.log_mass.device != self.supported.device:
            raise ValueError("image log measure must share one device")
        any_support = self.supported.any(dim=axes, keepdim=True)
        masked = self.log_mass.masked_fill(~self.supported, -torch.inf)
        safe = torch.where(any_support, masked, torch.zeros_like(masked))
        log_probability = safe - torch.logsumexp(safe, dim=axes, keepdim=True)
        log_probability = torch.where(self.supported, log_probability, 0.0)
        probability = torch.where(self.supported, log_probability.exp(), 0.0)
        return probability, log_probability

    def camera_centers(self) -> Tensor:
        probability, _ = self.normalized((-2, -1))
        grid = current_image_grid(
            probability.shape[-2], probability.shape[-1], device=probability.device
        )
        return (probability[..., None] * grid).sum((-3, -2))


def pushforward_log_to_current_image(
    log_measure: Tensor,
    measure_support: Tensor,
    spatial: CurrentImageSupport,
    *,
    rows: int,
    columns: int,
) -> ImageLogMeasure:
    """Segmented log-sum-exp of bilinear source mass, without tiny-mass division.

    Source logs come from the actual G1/G2/binder logits, never log(exp(logit)).
    Segment maxima are numerical shifts only and are detached; derivatives are
    the usual log-sum-exp derivatives. No confidence floor or mass quota.
    """
    spatial.validate()
    shape = spatial.probability.shape
    if (
        log_measure.ndim != 6
        or log_measure.shape[0] != shape[0]
        or log_measure.shape[2:] != shape[1:-1]
    ):
        raise ValueError("log pushforward requires [B,K,C,Yq,Xq,M]")
    if measure_support.shape != log_measure.shape or measure_support.dtype != torch.bool:
        raise ValueError("source log measure needs aligned Boolean support")
    if (
        log_measure.device != spatial.coordinates.device
        or measure_support.device != log_measure.device
    ):
        raise ValueError("log pushforward requires one device")
    if min(rows, columns) < 1:
        raise ValueError("image dimensions must be positive")
    valid_source = measure_support & spatial.valid.any(-1)[:, None]
    if not bool(torch.isfinite(log_measure.masked_select(valid_source)).all()):
        raise ValueError("supported source log mass must be finite")
    with torch.autocast(device_type=log_measure.device.type, enabled=False):
        xy = torch.where(spatial.valid[..., None], spatial.coordinates.float(), 0.0).clamp(-1, 1)
        x, y = (xy[..., 0] + 1) * ((columns - 1) / 2), (xy[..., 1] + 1) * ((rows - 1) / 2)
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        batch, objects, cameras = log_measure.shape[:3]
        safe_source = torch.where(valid_source, log_measure.float(), 0.0)
        base_log = safe_source[..., None] + spatial.log_probability[:, None]
        valid = valid_source[..., None] & spatial.valid[:, None]
        segments = []
        floor = torch.finfo(torch.float32).min
        maximum = log_measure.new_full(
            (batch, objects, cameras, rows * columns), floor, dtype=torch.float32
        )
        for ox, oy, coefficient in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ):
            index = (y0.long() + oy).clamp_max(rows - 1) * columns + (x0.long() + ox).clamp_max(
                columns - 1
            )
            index = index.reshape(batch, 1, cameras, -1).expand(-1, objects, -1, -1)
            active = valid & (coefficient[:, None] > 0)
            safe_coefficient = torch.where(coefficient > 0, coefficient, 1.0)
            score = (base_log + safe_coefficient.log()[:, None]).reshape(
                batch, objects, cameras, -1
            )
            active = active.reshape_as(score)
            segments.append((index, score, active))
            maximum = maximum.scatter_reduce(
                -1,
                index,
                score.detach().masked_fill(~active, floor),
                reduce="amax",
                include_self=True,
            )
        total = torch.zeros_like(maximum)
        for index, score, active in segments:
            origin = maximum.gather(-1, index)
            # Mask BEFORE exponentiation: inactive huge payload cannot overflow.
            centered = torch.where(active, score - origin, 0.0)
            weight = torch.where(active, centered.exp(), 0.0)
            total = total.scatter_add(-1, index, weight)
        supported = total > 0
        log_total = torch.where(supported, total, 1.0).log()
        result = torch.where(supported, maximum + log_total, 0.0)
        return ImageLogMeasure(
            result.reshape(batch, objects, cameras, rows, columns),
            supported.reshape(batch, objects, cameras, rows, columns),
        )


@dataclass(frozen=True)
class ObjectImageReadSource:
    """The actual G3 log read and its spatial chart, before image discretization.

    Reprojecting this measure to a native DINO chart does not upsample a
    barycenter or treat query indices as image coordinates. No exp/log roundtrip.
    """
    log_measure: Tensor             # [B,K,C,Yq,Xq,M]
    supported: Tensor               # Boolean same axes
    spatial: CurrentImageSupport

    def on_image(self, *, rows: int, columns: int) -> ImageLogMeasure:
        return pushforward_log_to_current_image(
            self.log_measure, self.supported, self.spatial, rows=rows, columns=columns
        )

    def permute(self, index: Tensor) -> "ObjectImageReadSource":
        return ObjectImageReadSource(
            self.log_measure[:, index], self.supported[:, index], self.spatial
        )
