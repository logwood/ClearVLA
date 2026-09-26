"""Explicit image lattices for the opt-in canonical geometry path.

Canonical coordinates are pixel centers of the *outer* RGB image: x right,
y down, with the top-left center at (0, 0). A chart describes nominal sample
centers, not a neural receptive field. Visible image area and interpolation
support are deliberately separate. Importing this module changes no policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

GEOMETRY_SCHEMA = "canonical_rgb_lattice_v1"
CANONICAL_FRAME = "outer_rgb_pixel_centers_v1"
CoordinateUnits = Literal["index", "normalized", "canonical"]


def _hw(value: tuple[int, int], label: str) -> tuple[int, int]:
    if len(value) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in value):
        raise ValueError(f"{label} must contain two positive integers, got {value}")
    return int(value[0]), int(value[1])


def _xy_tensor(value: tuple[float, float], like: Tensor) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=like.device)


def _points(value: Tensor) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError("coordinates must have a final x/y dimension of size 2")
    return value.float()


@dataclass(frozen=True)
class ChartSpec:
    """Axis-aligned affine sample lattice in one declared canonical frame.

    ``visible_bounds_xyxy`` is the producer's half-open pixel-area rectangle;
    it is NOT inferred from token centers. All normalized methods use the
    existing align_corners=True convention. Singleton axes are rejected for
    that convention because their physical displacement scale is undefined.
    Index/canonical conversions and sampling still support singleton axes.
    """

    name: str
    shape_hw: tuple[int, int]
    origin_xy: tuple[float, float]
    pitch_xy: tuple[float, float]
    visible_bounds_xyxy: tuple[float, float, float, float]
    canonical_frame: str = CANONICAL_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_hw", _hw(self.shape_hw, "shape_hw"))
        for key, size in (("origin_xy", 2), ("pitch_xy", 2), ("visible_bounds_xyxy", 4)):
            values = tuple(float(v) for v in getattr(self, key))
            if len(values) != size or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{key} must contain {size} finite values")
            object.__setattr__(self, key, values)
        if any(v <= 0 for v in self.pitch_xy):
            raise ValueError("pitch_xy must be strictly positive")
        x0, y0, x1, y1 = self.visible_bounds_xyxy
        if x0 >= x1 or y0 >= y1:
            raise ValueError("visible bounds must have positive area")
        if not self.name or not self.canonical_frame:
            raise ValueError("chart name and canonical_frame are required")

    def index_to_canonical(self, index: Tensor) -> Tensor:
        index = _points(index)
        return index * _xy_tensor(self.pitch_xy, index) + _xy_tensor(self.origin_xy, index)

    def canonical_to_index(self, canonical: Tensor) -> Tensor:
        canonical = _points(canonical)
        return (canonical - _xy_tensor(self.origin_xy, canonical)) / _xy_tensor(self.pitch_xy, canonical)

    def _normalized_scale(self, like: Tensor) -> Tensor:
        h, w = self.shape_hw
        if h == 1 or w == 1:
            raise ValueError("align_corners=True normalized units require both chart axes > 1")
        return _xy_tensor(((w - 1) / 2, (h - 1) / 2), like)

    def normalized_to_canonical(self, normalized: Tensor) -> Tensor:
        normalized = _points(normalized)
        return self.index_to_canonical((normalized + 1) * self._normalized_scale(normalized))

    def canonical_to_normalized(self, canonical: Tensor) -> Tensor:
        index = self.canonical_to_index(canonical)
        return index / self._normalized_scale(index) - 1

    def lattice(self, *, device: torch.device | str | None = None) -> Tensor:
        """Return [H,W,2] canonical positions in FP32."""
        h, w = self.shape_hw
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32), indexing="ij",
        )
        return self.index_to_canonical(torch.stack((xx, yy), dim=-1))

    def visible_support(self, canonical: Tensor) -> Tensor:
        canonical = _points(canonical)
        lo = _xy_tensor(self.visible_bounds_xyxy[:2], canonical)
        hi = _xy_tensor(self.visible_bounds_xyxy[2:], canonical)
        return torch.isfinite(canonical).all(-1) & (canonical >= lo).all(-1) & (canonical < hi).all(-1)

    def center_support(self, canonical: Tensor) -> Tensor:
        """Full bilinear kernel support, including exact first/last centers."""
        index = self.canonical_to_index(canonical)
        h, w = self.shape_hw
        hi = _xy_tensor((w - 1, h - 1), index)
        return torch.isfinite(index).all(-1) & (index >= 0).all(-1) & (index <= hi).all(-1)

    @property
    def center_domain_xyxy(self) -> tuple[float, float, float, float]:
        h, w = self.shape_hw
        return (
            self.origin_xy[0], self.origin_xy[1],
            self.origin_xy[0] + self.pitch_xy[0] * (w - 1),
            self.origin_xy[1] + self.pitch_xy[1] * (h - 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_SCHEMA, "name": self.name,
            "canonical_frame": self.canonical_frame, "shape_hw": list(self.shape_hw),
            "origin_xy": list(self.origin_xy), "pitch_xy": list(self.pitch_xy),
            "visible_bounds_xyxy": list(self.visible_bounds_xyxy),
            "visible_pixel_domain_xyxy": list(self.visible_bounds_xyxy),
            "center_domain_xyxy": list(self.center_domain_xyxy),
            "interpolation_mode": "bilinear",
            "normalized_align_corners": True,
            "align_corners": True,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChartSpec:
        if data.get("schema") != GEOMETRY_SCHEMA or data.get("normalized_align_corners") is not True:
            raise ValueError("unsupported spatial geometry schema or normalized convention")
        return cls(**{key: data[key] for key in (
            "name", "shape_hw", "origin_xy", "pitch_xy", "visible_bounds_xyxy", "canonical_frame",
        )})


def rgb_chart(input_hw: tuple[int, int], name: str = "rgb") -> ChartSpec:
    h, w = _hw(input_hw, "input_hw")
    return ChartSpec(name, (h, w), (0., 0.), (1., 1.), (-.5, -.5, w - .5, h - .5))


def convolution_chart(
    source: ChartSpec, *, name: str, kernel_hw: tuple[int, int],
    stride_hw: tuple[int, int], padding_hw: tuple[int, int] = (0, 0),
    dilation_hw: tuple[int, int] = (1, 1),
) -> ChartSpec:
    kernel_hw, stride_hw = _hw(kernel_hw, "kernel_hw"), _hw(stride_hw, "stride_hw")
    dilation_hw = _hw(dilation_hw, "dilation_hw")
    if len(padding_hw) != 2 or any(int(v) != v or v < 0 for v in padding_hw):
        raise ValueError("padding_hw must contain two nonnegative integers")
    out_hw = tuple((n + 2 * p - d * (k - 1) - 1) // s + 1
                   for n, p, d, k, s in zip(source.shape_hw, padding_hw, dilation_hw, kernel_hw, stride_hw))
    shift_xy = tuple((d * (k - 1) / 2 - p) * pitch
                     for d, k, p, pitch in zip(dilation_hw[::-1], kernel_hw[::-1], padding_hw[::-1], source.pitch_xy))
    return ChartSpec(name, out_hw, tuple(o + s for o, s in zip(source.origin_xy, shift_xy)),
                     tuple(p * s for p, s in zip(source.pitch_xy, stride_hw[::-1])),
                     source.visible_bounds_xyxy, source.canonical_frame)


def resize_chart(source: ChartSpec, shape_hw: tuple[int, int], name: str) -> ChartSpec:
    """Half-pixel resize geometry; interpolation kernel/filter stays external."""
    shape_hw = _hw(shape_hw, "shape_hw")
    ratio_xy = tuple(n / out for n, out in zip(source.shape_hw[::-1], shape_hw[::-1]))
    return ChartSpec(name, shape_hw,
                     tuple(o + (r - 1) * p / 2 for o, p, r in zip(source.origin_xy, source.pitch_xy, ratio_xy)),
                     tuple(p * r for p, r in zip(source.pitch_xy, ratio_xy)),
                     source.visible_bounds_xyxy, source.canonical_frame)


def crop_chart(source: ChartSpec, shape_hw: tuple[int, int], offset_yx: tuple[int, int], name: str) -> ChartSpec:
    shape_hw = _hw(shape_hw, "shape_hw")
    if len(offset_yx) != 2 or any(int(v) != v or v < 0 for v in offset_yx):
        raise ValueError("crop offset_yx must contain two nonnegative integers; padded crops need a separate support owner")
    if any(offset + size > total for offset, size, total in zip(offset_yx, shape_hw, source.shape_hw)):
        raise ValueError("crop extends outside resized image")
    origin = tuple(o + p * i for o, p, i in zip(source.origin_xy, source.pitch_xy, offset_yx[::-1]))
    lo = tuple(max(o - p / 2, lower) for o, p, lower in zip(origin, source.pitch_xy, source.visible_bounds_xyxy[:2]))
    hi = tuple(min(o + p * (n - .5), upper)
               for o, p, n, upper in zip(origin, source.pitch_xy, shape_hw[::-1], source.visible_bounds_xyxy[2:]))
    return ChartSpec(name, shape_hw, origin, source.pitch_xy, (*lo, *hi), source.canonical_frame)


def area_pool_chart(source: ChartSpec, shape_hw: tuple[int, int], name: str) -> ChartSpec:
    """Nominal centers of PyTorch adaptive-average/area bins.

    Non-divisible pooling can have nonuniform bin centers. Such a grid cannot
    be represented by an affine ChartSpec and is rejected instead of being
    silently approximated by an endpoint resize.
    """
    shape_hw = _hw(shape_hw, "shape_hw")
    origins, pitches = [], []
    for n, out, origin, pitch in zip(source.shape_hw[::-1], shape_hw[::-1], source.origin_xy, source.pitch_xy):
        centers = [(math.floor(i * n / out) + math.ceil((i + 1) * n / out) - 1) / 2 for i in range(out)]
        step = centers[1] - centers[0] if out > 1 else float(n)
        if step <= 0 or any(not math.isclose(centers[i], centers[0] + i * step, abs_tol=1e-9) for i in range(out)):
            raise ValueError(f"area pooling {n}->{out} has non-affine centers; explicit per-center geometry is required")
        origins.append(origin + centers[0] * pitch)
        pitches.append(step * pitch)
    return ChartSpec(name, shape_hw, tuple(origins), tuple(pitches), source.visible_bounds_xyxy, source.canonical_frame)


def build_raw_charts(input_hw: tuple[int, int]) -> dict[str, ChartSpec]:
    rgb = rgb_chart(input_hw)
    stem = convolution_chart(rgb, name="raw_stem", kernel_hw=(7, 7), stride_hw=(2, 2), padding_hw=(3, 3))
    high = convolution_chart(stem, name="raw_high", kernel_hw=(3, 3), stride_hw=(2, 2), padding_hw=(1, 1))
    mid = convolution_chart(high, name="raw_mid", kernel_hw=(3, 3), stride_hw=(2, 2), padding_hw=(1, 1))
    descriptor_high = area_pool_chart(rgb, high.shape_hw, "descriptor_high")
    descriptor_mid = area_pool_chart(descriptor_high, mid.shape_hw, "descriptor_mid")
    # Short names are convenient inside the model; the explicit aliases keep
    # serialized geometry metadata readable and match the source-contract
    # vocabulary without introducing a second ChartSpec object.
    return dict(
        rgb=rgb,
        raw_rgb=rgb,
        raw_stem=stem,
        raw_high=high,
        raw_high84=high,
        raw_mid=mid,
        raw_mid42=mid,
        descriptor_high=descriptor_high,
        area_descriptor84=descriptor_high,
        descriptor_mid=descriptor_mid,
        area_descriptor42=descriptor_mid,
    )


def build_dino_charts(
    input_hw: tuple[int, int], *, resized_hw: tuple[int, int], crop_hw: tuple[int, int],
    crop_offset_yx: tuple[int, int], patch_hw: tuple[int, int], pooled_hw: tuple[int, int],
) -> dict[str, ChartSpec]:
    """Build from actual processor dimensions and offsets, never a model name.

    Patch projection is the DINO Conv2d kernel=stride=patch, padding=0. Inputs
    with a remainder are supported: the unread remainder does not become a
    token center. Pixel visibility and token interpolation still differ.
    """
    resized = resize_chart(rgb_chart(input_hw), resized_hw, "processor_resize")
    crop = crop_chart(resized, crop_hw, crop_offset_yx, "processor_crop")
    patch = convolution_chart(crop, name="dino_patch", kernel_hw=patch_hw, stride_hw=patch_hw)
    pooled = area_pool_chart(patch, pooled_hw, "dino_pooled")
    charts: dict[str, ChartSpec] = dict(
        processor_resize=resized,
        processor_crop=crop,
        dino_patch=patch,
        dino16=patch,
        dino_pooled=pooled,
    )
    if tuple(pooled.shape_hw) == (8, 8):
        charts["dino_pool8"] = pooled
    return charts


def _same_frame(source: ChartSpec, target: ChartSpec) -> None:
    if source.canonical_frame != target.canonical_frame:
        raise ValueError("cannot convert between different canonical frames without an explicit transform")


def _to_canonical(points: Tensor, chart: ChartSpec, units: CoordinateUnits) -> Tensor:
    if units == "canonical":
        return _points(points)
    if units == "index":
        return chart.index_to_canonical(points)
    if units == "normalized":
        return chart.normalized_to_canonical(points)
    raise ValueError(f"unknown coordinate units {units!r}")


def _unit_scale(chart: ChartSpec, units: CoordinateUnits, like: Tensor) -> Tensor:
    if units == "canonical":
        return _xy_tensor((1., 1.), like)
    if units == "index":
        return _xy_tensor(chart.pitch_xy, like)
    if units == "normalized":
        return _xy_tensor(chart.pitch_xy, like) * chart._normalized_scale(like)
    raise ValueError(f"unknown coordinate units {units!r}")


def convert_points(points: Tensor, source: ChartSpec, target: ChartSpec, *,
                   source_units: CoordinateUnits = "index", target_units: CoordinateUnits = "index") -> Tensor:
    _same_frame(source, target)
    canonical = _to_canonical(points, source, source_units)
    if target_units == "canonical":
        return canonical
    if target_units == "index":
        return target.canonical_to_index(canonical)
    if target_units == "normalized":
        return target.canonical_to_normalized(canonical)
    raise ValueError(f"unknown coordinate units {target_units!r}")


def convert_vectors(vectors: Tensor, source: ChartSpec, target: ChartSpec, *,
                    source_units: CoordinateUnits = "index", target_units: CoordinateUnits = "index") -> Tensor:
    _same_frame(source, target)
    vectors = _points(vectors)
    return vectors * (_unit_scale(source, source_units, vectors) / _unit_scale(target, target_units, vectors))


def convert_variance(variance: Tensor, source: ChartSpec, target: ChartSpec, *,
                     source_units: CoordinateUnits = "index", target_units: CoordinateUnits = "index") -> Tensor:
    _same_frame(source, target)
    variance = _points(variance)
    scale = _unit_scale(source, source_units, variance) / _unit_scale(target, target_units, variance)
    return variance * scale.square()


def convert_covariance(covariance: Tensor, source: ChartSpec, target: ChartSpec, *,
                       source_units: CoordinateUnits = "index", target_units: CoordinateUnits = "index") -> Tensor:
    _same_frame(source, target)
    if covariance.shape[-2:] != (2, 2):
        raise ValueError("covariance must end in [2,2]")
    covariance = covariance.float()
    scale = _unit_scale(source, source_units, covariance) / _unit_scale(target, target_units, covariance)
    return covariance * scale[:, None] * scale[None, :]


@dataclass(frozen=True)
class SamplingSupport:
    visible: Tensor
    center: Tensor
    kernel_weight: Tensor
    producer_weight: Tensor

    @property
    def complete(self) -> Tensor:
        """Geometric full support plus a fully producer-supported kernel."""
        return self.visible & self.center & (self.producer_weight >= 1 - 1e-6)


@dataclass(frozen=True)
class SampledField:
    values: Tensor
    support: SamplingSupport


def sample_field(field: Tensor, source: ChartSpec, canonical_points: Tensor, *,
                 source_valid: Tensor | None = None,
                 padding_mode: Literal["zeros", "border", "reflection"] = "zeros") -> SampledField:
    """Sample [*batch,C,H,W] at [*batch,*samples,2], preserving all axes.

    Values have shape [*batch,C,*samples]; support has [*batch,*samples].
    ``source_valid`` optionally has [*batch,H,W] and represents producer
    support, independently for every camera/source. Padding affects values
    only: support is always computed with zeros outside the source lattice.
    Nonfinite coordinates have finite zero values and zero support. Geometry,
    values and autograd stay FP32 even inside a mixed-precision context.
    """
    if field.ndim < 3 or tuple(field.shape[-2:]) != source.shape_hw:
        raise ValueError("field must end in [channels, chart_height, chart_width]")
    if not field.is_floating_point():
        raise ValueError("sampled fields must have a floating-point dtype")
    batch_shape = tuple(field.shape[:-3])
    batch_ndim = len(batch_shape)
    points = _points(canonical_points)
    if points.device != field.device or tuple(points.shape[:batch_ndim]) != batch_shape:
        raise ValueError("field and points must have identical leading batch axes and device")
    sample_shape = tuple(points.shape[batch_ndim:-1])
    if not sample_shape or any(n == 0 for n in sample_shape) or any(n == 0 for n in batch_shape):
        raise ValueError("sampling requires nonempty batch/sample axes")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("unsupported padding mode")
    channels, h, w = field.shape[-3:]
    count = math.prod(batch_shape)
    samples = math.prod(sample_shape)
    with torch.autocast(device_type=field.device.type, enabled=False):
        index = source.canonical_to_index(points)
        finite = torch.isfinite(index).all(-1)
        safe_index = torch.where(finite[..., None], index, torch.zeros_like(index))
        # align_corners=False is an internal implementation detail. Conversion
        # from exact lattice indices preserves the public align_corners=True
        # chart convention and also handles singleton source dimensions.
        grid = (2 * (safe_index + .5) / _xy_tensor((w, h), safe_index) - 1).reshape(count, 1, samples, 2)
        flat_field = field.float().reshape(count, channels, h, w)
        values = F.grid_sample(flat_field, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
        values = values.reshape(*batch_shape, channels, *sample_shape)
        finite_values = finite.unsqueeze(batch_ndim)
        values = torch.where(finite_values, values, torch.zeros_like(values))
        ones = torch.ones((count, 1, h, w), device=field.device, dtype=torch.float32)
        kernel = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        kernel = kernel.reshape(*batch_shape, *sample_shape) * finite
        if source_valid is None:
            producer = kernel
        else:
            if tuple(source_valid.shape) != (*batch_shape, h, w) or source_valid.device != field.device:
                raise ValueError("source_valid must have shape [*batch,H,W] on the field device")
            if not source_valid.is_floating_point() and source_valid.dtype != torch.bool:
                raise ValueError("source_valid must be boolean or a floating support weight")
            if not bool(torch.isfinite(source_valid.float()).all().item()) or bool(
                ((source_valid.float() < 0.0) | (source_valid.float() > 1.0)).any().item()
            ):
                raise ValueError("source_valid must contain finite weights in [0,1]")
            valid_field = source_valid.float().reshape(count, 1, h, w)
            producer = F.grid_sample(valid_field, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            producer = producer.reshape(*batch_shape, *sample_shape) * finite
    return SampledField(values, SamplingSupport(source.visible_support(points), source.center_support(points), kernel, producer))


def resample_field(field: Tensor, source: ChartSpec, target: ChartSpec, *,
                   source_valid: Tensor | None = None,
                   padding_mode: Literal["zeros", "border", "reflection"] = "zeros") -> SampledField:
    _same_frame(source, target)
    points = target.lattice(device=field.device).expand(*field.shape[:-3], *target.shape_hw, 2)
    return sample_field(field, source, points, source_valid=source_valid, padding_mode=padding_mode)


def resample_flow(flow: Tensor, source: ChartSpec, target: ChartSpec, *,
                  source_units: CoordinateUnits = "index", target_units: CoordinateUnits = "index",
                  source_valid: Tensor | None = None,
                  padding_mode: Literal["zeros", "border", "reflection"] = "zeros") -> SampledField:
    """Move the source positions AND convert the sampled displacement units.

    Flow has [*batch,2,H,W]. This operation never negates, exchanges temporal
    direction, or adds chart origins to displacement vectors.
    """
    if flow.ndim < 3 or flow.shape[-3] != 2:
        raise ValueError("flow must have shape [*batch,2,H,W]")
    sampled = resample_field(flow, source, target, source_valid=source_valid, padding_mode=padding_mode)
    values = convert_vectors(sampled.values.movedim(-3, -1), source, target,
                             source_units=source_units, target_units=target_units).movedim(-1, -3)
    return SampledField(values, sampled.support)
