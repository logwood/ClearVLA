"""Current-only visual evidence and explicit G1/G2/G3 grounding.

The historical implementation hid the three grounding stages inside a
mutable state object that also carried several retired W implementations.
This module owns only the current observation plane:

* a learnable, recurrent local flow estimator;
* an early structured context mask;
* four continuous local hypotheses per 8x8 cell;
* three observation-only grounding blocks; and
* the high-resolution current feature bank consumed exactly once by P1.

Language, future supports, action targets and noisy actions are deliberately
not accepted by :class:`CurrentObservationCompiler`.  That is the provenance
boundary, rather than a runtime boolean checked deep inside a forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import CurrentObservation
from .routing import (
    rms_floored_l2_normalize,
    smooth_rms_contract,
    variance_floored_centered_norm,
)
from .types import LocalFactSet


def _coordinate_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _source_relative_add(center: Tensor, displacement: Tensor) -> Tensor:
    """Add a bounded displacement without a border division or hard clamp."""

    return center + (1.0 - center.square()).clamp_min(0.0) * torch.tanh(displacement)


def _flow_parameter_to_displacement(parameter: Tensor) -> Tensor:
    """Convert one recurrent flow parameter into a true chart displacement.

    The recurrent updater predicts an unconstrained parameter because its
    correlation sampler must stay inside the normalized image chart.  That
    parameter is *not* itself a displacement: the source-relative border
    factor in :func:`_source_relative_add` is part of the mapping.  Online G
    and Teacher-G consume physical normalized-coordinate deltas, so the
    conversion is performed exactly once at the estimator boundary.
    """

    if parameter.ndim != 4 or int(parameter.shape[1]) != 2:
        raise ValueError("flow parameters must be [B,2,H,W]")
    batch, _, height, width = parameter.shape
    base = _coordinate_grid(
        height,
        width,
        device=parameter.device,
        dtype=torch.float32,
    )[None].expand(batch, -1, -1, -1)
    mapped = _source_relative_add(base, parameter.float().permute(0, 2, 3, 1))
    return (mapped - base).permute(0, 3, 1, 2).contiguous()


def _sample_feature_chart(feature: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Sample ``[B,C,D,H,W]`` at ``[B,C,Y,X,M,2]`` coordinates.

    Cached DINO values are float16 while the formal CUDA autocast policy is
    bfloat16.  ``grid_sample`` cannot resolve those two low-precision inputs
    through autocast.  Spatial interpolation is also a coordinate-sensitive
    boundary, so both operands are promoted once to FP32 and the compact
    sampled candidate bank remains FP32 until its following autocast-aware
    projection.  The full DINO chart is not retained in a new FP32 cache.
    """

    if feature.ndim != 5 or coordinates.ndim != 6:
        raise ValueError("feature sampling requires [B,C,D,H,W] and [B,C,Y,X,M,2]")
    batch, cameras, channels = feature.shape[:3]
    if tuple(coordinates.shape[:2]) != (batch, cameras) or int(coordinates.shape[-1]) != 2:
        raise ValueError("feature chart and coordinate prefixes do not align")
    rows, columns, hypotheses = coordinates.shape[2:5]
    flat_feature = feature.reshape(batch * cameras, channels, *feature.shape[-2:])
    flat_grid = coordinates.reshape(batch * cameras, rows, columns * hypotheses, 2)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        sampled = F.grid_sample(
            flat_feature.float(),
            flat_grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    sampled = sampled.reshape(batch, cameras, channels, rows, columns, hypotheses)
    sampled = sampled.permute(0, 1, 3, 4, 5, 2).contiguous()
    valid = (coordinates.abs() <= 1.0).all(dim=-1, keepdim=True)
    return sampled, valid


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(min(channels // 8, 16), 1)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.block(value)


class RawFeaturePyramid(nn.Module):
    """Shared raw-image encoder with a 1/8 detail chart.

    It is intentionally convolutional rather than a per-patch MLP: the flow
    updater needs a spatially coherent receptive field and P1 needs local
    high-frequency values that were not pooled through the DINO chart.
    """

    def __init__(self, base: int, feature_dim: int) -> None:
        super().__init__()
        mid = max(base * 2, feature_dim // 2)
        self.network = nn.Sequential(
            nn.Conv2d(3, base, 5, stride=2, padding=2, bias=False),
            _ResidualConv(base),
            nn.Conv2d(base, mid, 3, stride=2, padding=1, bias=False),
            _ResidualConv(mid),
            nn.Conv2d(mid, feature_dim, 3, stride=2, padding=1, bias=False),
            _ResidualConv(feature_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class ConvGRUCell(nn.Module):
    def __init__(self, hidden: int, input_dim: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(hidden + input_dim, 2 * hidden, 3, padding=1)
        self.candidate = nn.Conv2d(hidden + input_dim, hidden, 3, padding=1)

    def forward(self, hidden: Tensor, value: Tensor) -> Tensor:
        joined = torch.cat((hidden, value), dim=1)
        reset, update = self.gates(joined).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((reset * hidden, value), dim=1)))
        return (1.0 - update) * hidden + update * candidate


@dataclass(frozen=True)
class PatchFlowField:
    """Learned local flow as true normalized-coordinate displacements.

    ``forward`` is previous-to-current transport indexed on the current
    destination chart.  ``backward`` is current-to-previous transport indexed
    on the previous destination chart.  Storing each direction on its
    destination chart lets a current fact consume ``forward`` directly and
    makes both reconstruction warps explicit rather than silently attaching a
    source-indexed value to the wrong frame.
    """

    forward: Tensor  # [B,C,2,Hf,Wf]
    # Backward flow is a training-only geometry target.  Deployment keeps the
    # complete forward field/status but does not run a second recurrent flow
    # solve merely to construct a value that no online consumer reads.
    backward: Tensor | None  # [B,C,2,Hf,Wf] when geometry supervision is active
    confidence: Tensor  # [B,C,1,Hf,Wf]
    uncertainty: Tensor  # [B,C,1,Hf,Wf]
    occlusion: Tensor  # [B,C,1,Hf,Wf]
    refinement_sequence: tuple[Tensor, ...]

    def validate(self) -> None:
        if self.forward.ndim != 5 or int(self.forward.shape[2]) != 2:
            raise ValueError("patch flow must be [B,C,2,H,W]")
        if self.backward is not None and tuple(self.backward.shape) != tuple(self.forward.shape):
            raise ValueError("forward and backward flow charts must align")
        scalar_shape = (*self.forward.shape[:2], 1, *self.forward.shape[-2:])
        for name in ("confidence", "uncertainty", "occlusion"):
            if tuple(getattr(self, name).shape) != scalar_shape:
                raise ValueError(f"flow {name} chart does not align")
        if not self.refinement_sequence:
            raise ValueError("flow refinement sequence cannot be empty")


class RecurrentLocalFlow(nn.Module):
    """SEA-RAFT-style local recurrent flow without an all-pairs volume.

    A radius-bounded correlation neighbourhood is recomputed around the
    current continuous estimate.  This keeps batch-eight memory bounded while
    retaining iterative, content-conditioned sub-patch refinement.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        iterations: int,
        radius: int,
        uncertainty_floor: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.iterations = int(iterations)
        self.radius = int(radius)
        self.uncertainty_floor = float(uncertainty_floor)
        correlations = (2 * self.radius + 1) ** 2
        hidden = self.feature_dim
        self.initial = nn.Conv2d(2 * feature_dim, hidden, 3, padding=1)
        self.gru = ConvGRUCell(hidden, feature_dim + correlations + 2)
        self.delta = nn.Sequential(
            _ResidualConv(hidden),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        self.status = nn.Sequential(
            _ResidualConv(hidden),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        delta_output = self.delta[-1]
        if not isinstance(delta_output, nn.Conv2d) or delta_output.bias is None:
            raise TypeError("flow delta head must end in a biased convolution")
        nn.init.zeros_(delta_output.weight)
        nn.init.zeros_(delta_output.bias)

    def _sample(self, target: Tensor, flow: Tensor, dx: int = 0, dy: int = 0) -> Tensor:
        batch, _, height, width = target.shape
        base = _coordinate_grid(
            height,
            width,
            device=target.device,
            dtype=torch.float32,
        )[None].expand(batch, -1, -1, -1)
        offset = flow.float().permute(0, 2, 3, 1)
        offset = offset + offset.new_tensor(
            [2.0 * dx / max(width - 1, 1), 2.0 * dy / max(height - 1, 1)]
        )
        grid = _source_relative_add(base, offset)
        return F.grid_sample(
            target,
            grid.to(dtype=target.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _sample_neighbourhood(self, target: Tensor, flow: Tensor) -> Tensor:
        """Sample every local correlation offset in one grid-sample launch."""

        batch, channels, height, width = target.shape
        base = _coordinate_grid(
            height,
            width,
            device=target.device,
            dtype=torch.float32,
        )[None, None].expand(batch, 1, -1, -1, -1)
        axis = torch.arange(
            -self.radius,
            self.radius + 1,
            device=target.device,
            dtype=torch.float32,
        )
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack(
            (
                2.0 * dx / float(max(width - 1, 1)),
                2.0 * dy / float(max(height - 1, 1)),
            ),
            dim=-1,
        ).reshape(1, -1, 1, 1, 2)
        parameter = flow.float().permute(0, 2, 3, 1)[:, None] + offsets
        grid = _source_relative_add(base, parameter)
        neighbours = int(offsets.shape[1])
        sampled = F.grid_sample(
            target,
            grid.reshape(batch, neighbours * height, width, 2).to(dtype=target.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch, channels, neighbours, height, width).transpose(1, 2)

    def _estimate(
        self, source: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        source_unit, _ = rms_floored_l2_normalize(source, 0.05, dim=1)
        target_unit, _ = rms_floored_l2_normalize(target, 0.05, dim=1)
        hidden = torch.tanh(self.initial(torch.cat((source, target), dim=1)))
        flow = source.new_zeros((source.shape[0], 2, *source.shape[-2:]))
        sequence: list[Tensor] = []
        for _ in range(self.iterations):
            sampled = self._sample_neighbourhood(target_unit, flow)
            corr = (source_unit[:, None] * sampled).sum(dim=2)
            matched = self._sample(target, flow)
            hidden = self.gru(hidden, torch.cat((matched - source, corr, flow), dim=1))
            update, _ = smooth_rms_contract(self.delta(hidden), 0.25)
            flow = flow + update
            sequence.append(flow)
        status = self.status(hidden).float()
        confidence = torch.sigmoid(status[:, :1])
        uncertainty = F.softplus(status[:, 1:2]) + self.uncertainty_floor
        occlusion = torch.sigmoid(status[:, 2:3])
        return flow, confidence, uncertainty, occlusion, tuple(sequence)

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        *,
        compute_backward: bool = True,
    ) -> PatchFlowField:
        if tuple(previous.shape) != tuple(current.shape) or previous.ndim != 5:
            raise ValueError("flow images must align as [B,C,F,H,W]")
        batch, cameras, channels, height, width = previous.shape
        flat_previous = previous.reshape(batch * cameras, channels, height, width)
        flat_current = current.reshape(batch * cameras, channels, height, width)
        # The online fact chart is the *current* frame.  Estimate its inverse
        # correspondence (current -> previous), convert the bounded recurrent
        # parameter to an actual displacement, then negate it.  The resulting
        # previous -> current motion is therefore indexed on current cells and
        # can be consumed by G/Teacher-G without a hidden frame-axis mismatch.
        inverse, confidence, uncertainty, occlusion, inverse_sequence = self._estimate(
            flat_current, flat_previous
        )
        forward = -_flow_parameter_to_displacement(inverse)
        sequence = tuple(-_flow_parameter_to_displacement(value) for value in inverse_sequence)
        backward = None
        if compute_backward:
            source_forward, _, _, _, _ = self._estimate(flat_previous, flat_current)
            backward = -_flow_parameter_to_displacement(source_forward)

        def restore(value: Tensor) -> Tensor:
            return value.reshape(batch, cameras, *value.shape[1:])

        field = PatchFlowField(
            forward=restore(forward),
            backward=None if backward is None else restore(backward),
            confidence=restore(confidence),
            uncertainty=restore(uncertainty),
            occlusion=restore(occlusion),
            refinement_sequence=tuple(restore(value) for value in sequence),
        )
        field.validate()
        return field


class GroundingFactBlock(nn.Module):
    """One named observation-only G block."""

    def __init__(self, hidden: int, heads: int, *, cross_width: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross = nn.Linear(cross_width, hidden, bias=False)
        self.cross_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden, bias=False),
        )

    def forward(self, carrier: Tensor, innovation: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.query_norm(carrier)
        attention, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        delta = attention + self.cross_norm(self.cross(innovation))
        delta, _ = smooth_rms_contract(delta, 0.50)
        carrier = carrier + delta
        ffn, _ = smooth_rms_contract(self.ffn(carrier), 0.50)
        return carrier + ffn, delta


@dataclass(frozen=True)
class ObservationEvidence:
    """Current facts plus the sole high-resolution bank available to P1."""

    local_facts: LocalFactSet
    detail_features: Tensor  # [B,C,F,Hd,Wd]
    previous_detail_features: Tensor  # t=-4 [B,C,F,Hd,Wd], training geometry only
    earlier_detail_features: Tensor  # t=-8 [B,C,F,Hd,Wd], training geometry only
    literal_rgb: Tensor  # [B,C,3,R,R]
    previous_literal_rgb: Tensor  # t=-4 [B,C,3,R,R], training geometry only
    earlier_literal_rgb: Tensor  # t=-8 [B,C,3,R,R], training geometry only
    flow: PatchFlowField  # -4 -> 0
    earlier_flow: PatchFlowField  # -8 -> -4
    context_mask: Tensor  # bool [B,C,8,8]

    def validate(self) -> None:
        self.local_facts.validate()
        batch = self.local_facts.batch
        cameras = int(self.local_facts.public_scene_base.shape[1])
        if self.detail_features.ndim != 5 or tuple(self.detail_features.shape[:2]) != (
            batch,
            cameras,
        ):
            raise ValueError("detail features must be [B,C,F,H,W]")
        if tuple(self.previous_detail_features.shape) != tuple(self.detail_features.shape):
            raise ValueError("previous/current detail features must align")
        if tuple(self.earlier_detail_features.shape) != tuple(self.detail_features.shape):
            raise ValueError("causal detail history must align")
        if self.literal_rgb.ndim != 5 or tuple(self.literal_rgb.shape[:2]) != (
            batch,
            cameras,
        ):
            raise ValueError("literal RGB must be [B,C,3,R,R]")
        if tuple(self.previous_literal_rgb.shape) != tuple(self.literal_rgb.shape):
            raise ValueError("previous/current literal RGB charts must align")
        if tuple(self.earlier_literal_rgb.shape) != tuple(self.literal_rgb.shape):
            raise ValueError("causal literal RGB history must align")
        if tuple(self.context_mask.shape) != (batch, cameras, 8, 8):
            raise ValueError("context mask must preserve [B,C,8,8]")
        self.flow.validate()
        self.earlier_flow.validate()


class CurrentObservationCompiler(nn.Module):
    """Compile current RGB/DINO history into local facts and a P1 bank."""

    hypothesis_offset: Tensor

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        dims = config.dimensions
        obs = config.observation
        self.hidden = dims.hidden_size
        self.content_dim = dims.visual_token_dim
        self.route_dim = obs.address_route_dim
        self.grid = obs.grid_size
        self.hypotheses = obs.local_hypotheses
        self.cameras = dims.num_cameras
        self.mask_ratio = float(obs.mask_ratio)
        self.mask_block_size = int(obs.mask_block_size)
        self.motion_mask_fraction = float(obs.motion_mask_fraction)
        self.raw_encoder = RawFeaturePyramid(obs.raw_base_channels, obs.feature_dim)
        self.flow = RecurrentLocalFlow(
            obs.feature_dim,
            iterations=obs.flow_iterations,
            radius=obs.correlation_radius,
            uncertainty_floor=obs.uncertainty_floor,
        )
        self.dino_to_hidden = nn.Linear(self.content_dim, self.hidden, bias=False)
        self.raw_to_hidden = nn.Linear(obs.feature_dim, self.hidden, bias=False)
        self.position = nn.Parameter(
            torch.randn(1, self.cameras, self.grid, self.grid, self.hidden) * 0.02
        )
        self.camera = nn.Parameter(torch.randn(1, self.cameras, 1, 1, self.hidden) * 0.02)
        self.mask_dino = nn.Parameter(torch.zeros(1, 1, 1, 1, self.content_dim))
        self.mask_raw = nn.Parameter(torch.zeros(1, 1, 1, 1, obs.feature_dim))
        self.g1 = GroundingFactBlock(self.hidden, dims.num_heads, cross_width=self.hidden)
        self.g2 = GroundingFactBlock(
            self.hidden,
            dims.num_heads,
            cross_width=3 * self.route_dim,
        )
        self.g3 = GroundingFactBlock(self.hidden, dims.num_heads, cross_width=self.hidden)
        self.semantic_key = nn.Linear(self.content_dim, self.route_dim, bias=False)
        self.appearance_key = nn.Linear(obs.feature_dim, self.route_dim, bias=False)
        self.geometry_key = nn.Sequential(
            nn.Linear(12, 2 * self.route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.route_dim, self.route_dim, bias=False),
        )
        self.typed_query = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden, self.route_dim, bias=False)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.g3_owner_residual = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(self.hidden + self.route_dim, elementwise_affine=False),
                    nn.Linear(self.hidden + self.route_dim, self.hidden, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.hidden, 1, bias=False),
                )
                for name in ("semantic", "appearance", "geometry")
            }
        )
        for verifier in self.g3_owner_residual.values():
            if not isinstance(verifier, nn.Sequential):
                raise TypeError("G3 owner verifier must be sequential")
            output = verifier[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("G3 owner residual must end in a linear projection")
            nn.init.zeros_(output.weight)
        offset = torch.tensor(
            ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
            dtype=torch.float32,
        )
        if self.hypotheses != int(offset.shape[0]):
            raise ValueError("the active local chart requires exactly four hypotheses")
        self.register_buffer("hypothesis_offset", offset, persistent=True)

    def _raw_features(self, raw: Tensor) -> Tensor:
        batch, history, cameras = raw.shape[:3]
        flat = raw.reshape(batch * history * cameras, *raw.shape[3:])
        feature = self.raw_encoder(flat)
        return feature.reshape(batch, history, cameras, *feature.shape[1:])

    def _dino_chart(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 5:
            raise ValueError("causal DINO history must be [B,H,C,P,D]")
        batch, history, cameras, patches, width = tokens.shape
        side = round(math.sqrt(patches))
        if side * side != patches:
            raise ValueError("DINO patch count must form a square chart")
        return tokens.reshape(batch, history, cameras, side, side, width).permute(
            0, 1, 2, 5, 3, 4
        )

    @torch.no_grad()
    def teacher_supports(self, tokens: Tensor) -> Tensor:
        """Pool cached future DINO tokens once into Teacher-G's 8x8 chart."""

        if tokens.ndim != 5:
            raise ValueError("cached future DINO must be [B,F,C,P,D]")
        batch, supports, cameras, patches, width = tokens.shape
        side = round(math.sqrt(int(patches)))
        if side * side != int(patches) or int(width) != self.content_dim:
            raise ValueError("future DINO cache has an invalid patch chart")
        chart = (
            tokens.detach()
            .reshape(
                batch * supports * cameras,
                side,
                side,
                width,
            )
            .permute(0, 3, 1, 2)
        )
        # Cache storage is float16.  Converting the complete native chart to
        # FP32 before pooling created a ~340 MiB batch-eight temporary.  Pool
        # bounded chunks in FP32 into one preallocated result; this preserves
        # the previous target arithmetic and the complete cached input while
        # removing the unnecessary peak allocation.
        pooled = torch.empty(
            chart.shape[0],
            width,
            self.grid,
            self.grid,
            device=chart.device,
            dtype=torch.float32,
        )
        chunk_rows = 16
        for start in range(0, int(chart.shape[0]), chunk_rows):
            stop = min(start + chunk_rows, int(chart.shape[0]))
            pooled[start:stop] = F.adaptive_avg_pool2d(
                chart[start:stop].float(),
                (self.grid, self.grid),
            )
        return pooled.permute(0, 2, 3, 1).reshape(
            batch,
            supports,
            cameras,
            self.grid,
            self.grid,
            width,
        )

    def _coarse(self, value: Tensor) -> Tensor:
        batch, cameras, channels = value.shape[:3]
        pooled = F.adaptive_avg_pool2d(
            value.reshape(batch * cameras, channels, *value.shape[-2:]),
            (self.grid, self.grid),
        )
        return pooled.reshape(batch, cameras, channels, self.grid, self.grid)

    def _training_mask(self, motion: Tensor) -> Tensor:
        """Build an exact-quota observable mask from motion and block noise."""

        if tuple(motion.shape[-2:]) != (self.grid, self.grid):
            raise ValueError("mask motion chart must use the coarse grounding grid")
        batch, cameras = motion.shape[:2]
        count = min(
            max(round(self.mask_ratio * self.grid * self.grid), 1),
            self.grid * self.grid - 1,
        )
        motion_f = motion.detach().float().flatten(-2)
        low = motion_f.amin(dim=-1, keepdim=True)
        high = motion_f.amax(dim=-1, keepdim=True)
        motion_score = (motion_f - low) / (high - low).clamp_min(1e-6)
        block = self.mask_block_size
        coarse_side = math.ceil(self.grid / block)
        random_block = torch.rand(
            batch,
            cameras,
            coarse_side,
            coarse_side,
            device=motion.device,
            dtype=torch.float32,
        )
        random_score = F.interpolate(
            random_block.reshape(batch * cameras, 1, coarse_side, coarse_side),
            size=(self.grid, self.grid),
            mode="nearest",
        ).reshape(batch, cameras, self.grid * self.grid)
        score = (
            self.motion_mask_fraction * motion_score
            + (1.0 - self.motion_mask_fraction) * random_score
        )
        indices = score.topk(count, dim=-1).indices
        mask = torch.zeros_like(score, dtype=torch.bool).scatter(-1, indices, True)
        return mask.reshape(batch, cameras, self.grid, self.grid)

    def forward(
        self,
        observation: CurrentObservation,
        *,
        context_mask: Tensor | None = None,
        training_mask: bool = False,
        geometry_supervision: bool = True,
        collect_diagnostics: bool = False,
    ) -> tuple[ObservationEvidence, dict[str, Tensor]]:
        batch = observation.batch
        raw_features = self._raw_features(observation.raw_rgb)
        if int(raw_features.shape[1]) != 3:
            raise ValueError("observation compiler requires visual history -8/-4/0")
        earlier_raw = raw_features[:, -3]
        current_raw = raw_features[:, -1]
        previous_raw = raw_features[:, -2]
        flow = self.flow(
            previous_raw,
            current_raw,
            compute_backward=geometry_supervision,
        )
        earlier_flow = self.flow(
            earlier_raw,
            previous_raw,
            compute_backward=geometry_supervision,
        )
        dino_history = self._dino_chart(observation.dino_history)
        history = int(dino_history.shape[1])
        flat_dino = dino_history.reshape(
            batch * history,
            self.cameras,
            self.content_dim,
            *dino_history.shape[-2:],
        )
        coarse_dino_history = self._coarse(flat_dino).reshape(
            batch,
            history,
            self.cameras,
            self.content_dim,
            self.grid,
            self.grid,
        )
        coarse_dino_history = coarse_dino_history.permute(0, 1, 2, 4, 5, 3).contiguous()
        coarse_dino = coarse_dino_history[:, -1]
        dino_chart = dino_history[:, -1]
        coarse_raw = self._coarse(current_raw).permute(0, 1, 3, 4, 2).contiguous()
        flow_coarse = self._coarse(
            flow.forward.reshape(batch, self.cameras, 2, *flow.forward.shape[-2:])
        )
        earlier_flow_coarse = self._coarse(
            earlier_flow.forward.reshape(
                batch, self.cameras, 2, *earlier_flow.forward.shape[-2:]
            )
        )
        coarse_base = _coordinate_grid(
            self.grid,
            self.grid,
            device=flow_coarse.device,
            dtype=torch.float32,
        )[None].expand(batch * self.cameras, -1, -1, -1)
        current_to_previous = coarse_base - flow_coarse.reshape(
            batch * self.cameras, 2, self.grid, self.grid
        ).float().permute(0, 2, 3, 1)
        earlier_flow_aligned = F.grid_sample(
            earlier_flow_coarse.reshape(batch * self.cameras, 2, self.grid, self.grid).float(),
            current_to_previous,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(batch, self.cameras, 2, self.grid, self.grid)
        flow_acceleration = flow_coarse.float() - earlier_flow_aligned
        if context_mask is None and training_mask:
            context_mask = self._training_mask(flow_coarse.square().sum(dim=2).sqrt())
        if context_mask is None:
            context_mask = torch.zeros(
                batch,
                self.cameras,
                self.grid,
                self.grid,
                device=coarse_dino.device,
                dtype=torch.bool,
            )
        if tuple(context_mask.shape) != (batch, self.cameras, self.grid, self.grid):
            raise ValueError("context mask must be [B,C,8,8]")
        visible = (~context_mask)[..., None]
        masked_dino = torch.where(
            visible,
            coarse_dino,
            self.mask_dino.to(device=coarse_dino.device, dtype=coarse_dino.dtype),
        )
        masked_raw = torch.where(
            visible,
            coarse_raw,
            self.mask_raw.to(device=coarse_raw.device, dtype=coarse_raw.dtype),
        )

        confidence = self._coarse(flow.confidence).permute(0, 1, 3, 4, 2)
        uncertainty = self._coarse(flow.uncertainty).permute(0, 1, 3, 4, 2)
        occlusion = self._coarse(flow.occlusion).permute(0, 1, 3, 4, 2)
        flow_xy = flow_coarse.permute(0, 1, 3, 4, 2)
        base = _coordinate_grid(
            self.grid,
            self.grid,
            device=coarse_dino.device,
            dtype=torch.float32,
        )[None, None].expand(batch, self.cameras, -1, -1, -1)
        # These are facts in the *current* frame.  ``flow_xy`` is already a
        # previous->current displacement indexed on this current chart.  It
        # must not be added once more to the current address; it remains a
        # separately typed geometry/transport prior for extrapolation.
        current_center = base
        support = (0.04 + 0.12 * torch.tanh(uncertainty.float())).clamp(0.02, 0.20)
        candidate_delta = self.hypothesis_offset.to(device=base.device)[None, None, None, None]
        candidate_delta = candidate_delta * support[..., None, :]
        coordinates = _source_relative_add(current_center[..., None, :], candidate_delta)
        dino_candidates, dino_valid = _sample_feature_chart(dino_chart, coordinates)
        raw_candidates, raw_valid = _sample_feature_chart(current_raw, coordinates)
        validity = dino_valid & raw_valid & visible[..., None, :]

        def align_chart(value: Tensor, grid: Tensor) -> Tensor:
            channels = int(value.shape[-1])
            source = value.permute(0, 1, 4, 2, 3).reshape(
                batch * self.cameras,
                channels,
                self.grid,
                self.grid,
            )
            with torch.autocast(device_type=value.device.type, enabled=False):
                aligned = F.grid_sample(
                    source.float(),
                    grid.float(),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            return aligned.reshape(
                batch,
                self.cameras,
                channels,
                self.grid,
                self.grid,
            ).permute(0, 1, 3, 4, 2)

        previous_to_earlier = coarse_base - earlier_flow_coarse.reshape(
            batch * self.cameras, 2, self.grid, self.grid
        ).float().permute(0, 2, 3, 1)
        aligned_previous = align_chart(coarse_dino_history[:, -2], current_to_previous)
        aligned_earlier_at_previous = align_chart(
            coarse_dino_history[:, -3], previous_to_earlier
        )
        recent_dino_delta = coarse_dino_history[:, -1] - aligned_previous
        earlier_delta_at_previous = (
            coarse_dino_history[:, -2] - aligned_earlier_at_previous
        )
        earlier_dino_delta = align_chart(earlier_delta_at_previous, current_to_previous)
        visual_history_innovation = torch.where(
            visible,
            recent_dino_delta + 0.5 * earlier_dino_delta,
            torch.zeros_like(recent_dino_delta),
        )
        carrier = (
            self.dino_to_hidden(masked_dino)
            + self.dino_to_hidden(visual_history_innovation)
        ) / math.sqrt(2.0)
        carrier = carrier + self.position + self.camera
        flat = carrier.reshape(batch, self.cameras * self.grid * self.grid, self.hidden)
        flat, g1_delta = self.g1(flat, flat)
        carrier = flat.reshape(batch, self.cameras, self.grid, self.grid, self.hidden)
        typed = {
            "semantic": self.semantic_key(dino_candidates),
            "appearance": self.appearance_key(raw_candidates),
        }
        geometry_input = torch.cat(
            (
                coordinates,
                flow_xy[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1),
                earlier_flow_aligned.permute(0, 1, 3, 4, 2)[..., None, :].expand(
                    -1, -1, -1, -1, self.hypotheses, -1
                ),
                flow_acceleration.permute(0, 1, 3, 4, 2)[..., None, :].expand(
                    -1, -1, -1, -1, self.hypotheses, -1
                ),
                support[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1),
                confidence[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1),
                uncertainty[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1),
                occlusion[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1),
            ),
            dim=-1,
        )
        typed["geometry"] = self.geometry_key(geometry_input)
        scale = float(self.route_dim) ** -0.5
        owner: dict[str, Tensor] = {}
        for name, value in typed.items():
            query, _ = variance_floored_centered_norm(self.typed_query[name](carrier), 0.25)
            key, _ = variance_floored_centered_norm(value, 0.25)
            logits = torch.einsum("bcijr,bcijmr->bcijm", query.float(), key.float()) * scale
            distance = (coordinates - current_center[..., None, :]).square().sum(dim=-1)
            logits = logits - 0.5 * distance / support[..., 0, None].square().clamp_min(1e-4)
            logits = logits.masked_fill(~validity[..., 0], -1.0e4)
            probability = torch.softmax(logits, dim=-1) * validity[..., 0].float()
            owner[name] = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0)

        typed_summary = torch.cat(
            tuple(
                torch.einsum("bcijm,bcijmr->bcijr", owner[name], typed[name].float())
                for name in ("semantic", "appearance", "geometry")
            ),
            dim=-1,
        ).to(dtype=carrier.dtype)
        flat, g2_delta = self.g2(
            carrier.reshape(batch, self.cameras * self.grid * self.grid, self.hidden),
            typed_summary.reshape(batch, self.cameras * self.grid * self.grid, -1),
        )
        carrier = flat.reshape(batch, self.cameras, self.grid, self.grid, self.hidden)
        # G3 is a bounded posterior correction, not a second owner head.  Zero
        # initialized residuals recover G2 exactly.
        refined_owner: dict[str, Tensor] = {}
        for name, probability in owner.items():
            g2_query = carrier[..., None, :].expand(-1, -1, -1, -1, self.hypotheses, -1)
            residual_input = torch.cat((g2_query, typed[name]), dim=-1)
            residual = 0.5 * torch.tanh(
                self.g3_owner_residual[name](residual_input).squeeze(-1).float()
            )
            logits = probability.clamp_min(1e-8).log() + residual
            logits = logits.masked_fill(~validity[..., 0], -1.0e4)
            corrected = torch.softmax(logits, dim=-1) * validity[..., 0].float()
            refined_owner[name] = corrected / corrected.sum(dim=-1, keepdim=True).clamp_min(1.0)
        factual_innovation = self.raw_to_hidden(masked_raw)
        flat, g3_delta = self.g3(
            carrier.reshape(batch, self.cameras * self.grid * self.grid, self.hidden),
            factual_innovation.reshape(batch, self.cameras * self.grid * self.grid, self.hidden),
        )
        public = flat.reshape(batch, self.cameras, self.grid, self.grid, self.hidden)
        facts = LocalFactSet(
            public_scene_base=public,
            # This target is current observable evidence, not future teacher
            # data.  It is detached and never used by an online value path;
            # the grounder consumes it only in the reconstruction objective.
            target_dino_content=coarse_dino.detach(),
            cell_observed=visible,
            content_slots=dino_candidates,
            semantic_slots=typed["semantic"],
            appearance_slots=typed["appearance"],
            geometry_slots=typed["geometry"],
            semantic_owner_probs=refined_owner["semantic"].to(dtype=carrier.dtype),
            appearance_owner_probs=refined_owner["appearance"].to(dtype=carrier.dtype),
            geometry_owner_probs=refined_owner["geometry"].to(dtype=carrier.dtype),
            slot_coordinates=coordinates.to(dtype=carrier.dtype),
            slot_support=support[..., 0, None]
            .expand(-1, -1, -1, -1, self.hypotheses)
            .to(dtype=carrier.dtype),
            slot_validity=validity.to(dtype=carrier.dtype),
            slot_transport_prior=flow_xy[..., None, :]
            .expand(-1, -1, -1, -1, self.hypotheses, -1)
            .to(dtype=carrier.dtype),
        )
        evidence = ObservationEvidence(
            local_facts=facts,
            detail_features=current_raw,
            previous_detail_features=previous_raw,
            earlier_detail_features=earlier_raw,
            literal_rgb=observation.raw_rgb[:, -1],
            previous_literal_rgb=observation.raw_rgb[:, -2],
            earlier_literal_rgb=observation.raw_rgb[:, -3],
            flow=flow,
            earlier_flow=earlier_flow,
            context_mask=context_mask,
        )
        evidence.validate()
        if not collect_diagnostics:
            return evidence, {}
        metrics = {
            "observation_flow_rms": flow.forward.detach().float().square().mean().sqrt(),
            "observation_flow_confidence": flow.confidence.detach().float().mean(),
            "observation_flow_uncertainty": flow.uncertainty.detach().float().mean(),
            "observation_flow_occlusion": flow.occlusion.detach().float().mean(),
            "observation_earlier_flow_rms": earlier_flow.forward.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "observation_flow_acceleration_rms": flow_acceleration.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "observation_visual_history_innovation_rms": visual_history_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "observation_context_mask_fraction": context_mask.detach().float().mean(),
            "grounding_g1_innovation_rms": g1_delta.detach().float().square().mean().sqrt(),
            "grounding_g2_innovation_rms": g2_delta.detach().float().square().mean().sqrt(),
            "grounding_g3_innovation_rms": g3_delta.detach().float().square().mean().sqrt(),
        }
        for name, probability in refined_owner.items():
            probability_f = probability.detach().float().clamp_min(1e-8)
            metrics[f"grounding_{name}_owner_entropy"] = (
                -(probability_f * probability_f.log()).sum(dim=-1) / math.log(self.hypotheses)
            ).mean()
            metrics[f"grounding_{name}_owner_max"] = probability_f.amax(dim=-1).mean()
        return evidence, metrics


__all__ = [
    "CurrentObservationCompiler",
    "ObservationEvidence",
    "PatchFlowField",
    "RecurrentLocalFlow",
]
