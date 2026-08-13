"""Exact active V120 P1 factual precision reader, mechanically extracted.

Only the shared factual P1 implementation and its local refiners live here.
The surrounding V120 monolith, W/P2 routes, and probe-only state are not
imported into the capability-named mainline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..v120_core.config import V39PolicyConfig
from ..v120_core.flow_dino_evidence import LateRawDetailEvidence
from ..v120_core.role_delta_attnres import (
    RoleDeltaAttnRes,
    VarianceFlooredCenteredNorm,
    smooth_rms_contract,
)

@dataclass(frozen=True)
class SharedFactualGlimpseBank:
    """One basis-free P1 value read exposed to basis-specific P2 consumers.

    Every tensor keeps the explicit glimpse axis.  In particular, this
    interface has no action-basis axis: expanding it for P2 is a view over
    already-selected facts, never another observation-bank read.
    """

    literal_rgb: Tensor
    learned_detail: Tensor
    coordinates: Tensor
    semantic: Tensor
    appearance: Tensor
    geometry: Tensor
    future_transport: Tensor
    query_key: Tensor

    def validate(
        self,
        *,
        batch: int,
        rows: int,
        glimpses: int,
        micro_cells: int,
        raw_dim: int,
        route_dim: int,
    ) -> None:
        expected_prefix = (batch, rows, glimpses)
        expected = {
            "literal_rgb": (*expected_prefix, micro_cells, 3),
            "learned_detail": (*expected_prefix, micro_cells, raw_dim),
            "coordinates": (*expected_prefix, micro_cells, 2),
            "semantic": (*expected_prefix, route_dim),
            "appearance": (*expected_prefix, route_dim),
            "geometry": (*expected_prefix, route_dim),
            "future_transport": (*expected_prefix, 5),
            "query_key": (*expected_prefix, route_dim),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"shared factual glimpse {name} must be {shape}, "
                    f"got {tuple(value.shape)}"
                )

def _align_milestone_tokens_to_horizon(
    tokens: Tensor, horizon: int, *, boundaries: tuple[int, ...] | None = None
) -> Tensor:
    """Expand one pooled token per action segment onto the action timeline."""

    if tokens.ndim != 3:
        raise ValueError(f"milestone tokens must be [B,K,H], got {tuple(tokens.shape)}")
    horizon = int(horizon)
    steps = int(tokens.shape[1])
    if horizon < 1 or steps < 1 or steps > horizon:
        raise ValueError(
            f"expected 1 <= milestone steps <= horizon, got steps={steps} horizon={horizon}"
        )
    if boundaries is not None:
        boundaries = tuple(int(value) for value in boundaries)
        if len(boundaries) != steps or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("milestone boundaries must be strictly increasing and match tokens")
        if boundaries[-1] != horizon:
            raise ValueError("the final milestone boundary must equal the action horizon")
    rows: list[Tensor] = []
    previous = 0
    for step in range(steps):
        if boundaries is None:
            lo = int(round(step * horizon / float(steps)))
            hi = int(round((step + 1) * horizon / float(steps)))
        else:
            lo = previous
            hi = boundaries[step]
            previous = hi
        hi = max(hi, lo + 1)
        hi = min(hi, horizon)
        rows.append(tokens[:, step : step + 1].expand(-1, hi - lo, -1))
    aligned = torch.cat(rows, dim=1)
    if aligned.shape[1] != horizon:
        raise RuntimeError(
            f"milestone alignment produced {aligned.shape[1]} tokens for horizon={horizon}"
        )
    return aligned

class _CoordinateTypedLocalRefiner(nn.Module):
    """P2 local refiner with values separated from address conditioning.

    RGB and learned detail are the only value sources. Coordinates, DINO,
    appearance, geometry, trajectory and future transport affect queries/keys
    but cannot manufacture a value. Thus an all-zero RGB/detail micro-patch
    produces an exact zero output through ordinary autograd.
    """

    def __init__(
        self,
        *,
        width: int,
        raw_dim: int,
        route_dim: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        width = int(width)
        attention_heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
        self.width = width
        self.rgb_value = nn.Linear(3, width, bias=False)
        self.detail_value = nn.Linear(int(raw_dim), width, bias=False)
        self.coordinate_key = nn.Linear(2, width, bias=False)
        self.query_condition = nn.Linear(int(route_dim), width, bias=False)
        self.semantic_condition = nn.Linear(int(route_dim), width, bias=False)
        self.appearance_condition = nn.Linear(int(route_dim), width, bias=False)
        self.geometry_condition = nn.Linear(int(route_dim), width, bias=False)
        self.future_condition = nn.Linear(5, width, bias=False)
        self.token_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.token_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    width,
                    attention_heads,
                    dropout=0.0,
                    bias=False,
                    batch_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.ffn_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 2 * width, bias=False),
                    nn.GELU(),
                    nn.Linear(2 * width, width, bias=False),
                )
                for _ in range(int(depth))
            ]
        )
        self.read_norm = VarianceFlooredCenteredNorm(0.25)
        self.read_attn = nn.MultiheadAttention(
            width,
            attention_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self,
        *,
        rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
        query: Tensor,
        semantic: Tensor,
        appearance: Tensor,
        geometry: Tensor,
        future_transport: Tensor,
        intervention: str | None = None,
        collect_diagnostics: bool = True,

    ) -> tuple[Tensor, dict[str, Tensor]]:
        if rgb.ndim != 4 or int(rgb.shape[-1]) != 3:
            raise ValueError("typed local RGB must be [N,G,micro,3]")
        if tuple(learned_detail.shape[:-1]) != tuple(rgb.shape[:-1]):
            raise ValueError("typed learned detail does not align with RGB")
        if tuple(coordinates.shape) != (*rgb.shape[:-1], 2):
            raise ValueError("typed local coordinates do not align with RGB")
        batch, glimpses, micro, _ = rgb.shape
        for name, value in (
            ("query", query),
            ("semantic", semantic),
            ("appearance", appearance),
            ("geometry", geometry),
        ):
            if tuple(value.shape[:2]) != (batch, glimpses):
                raise ValueError(f"typed local {name} context is misaligned")
        if tuple(future_transport.shape) != (batch, glimpses, 5):
            raise ValueError("typed future transport must be [N,G,5]")

        tokens = (
            self.rgb_value(rgb) + self.detail_value(learned_detail)
        ) * (2.0**-0.5)
        position = self.coordinate_key(coordinates)
        tokens = tokens.reshape(batch * glimpses, micro, self.width)
        position = position.reshape_as(tokens)
        for norm, attention, ffn_norm, ffn in zip(
            self.token_norms,
            self.token_attn,
            self.ffn_norms,
            self.ffns,
        ):
            normalized = norm(tokens)
            update, _ = attention(
                normalized + position,
                normalized + position,
                normalized,
                need_weights=False,
            )
            tokens = tokens + (2.0**-0.5) * update
            tokens = tokens + (2.0**-0.5) * ffn(ffn_norm(tokens))

        condition = (
            self.query_condition(query)
            + self.semantic_condition(semantic)
            + self.appearance_condition(appearance)
            + self.geometry_condition(geometry)
            + self.future_condition(future_transport)
        ) / math.sqrt(5.0)
        condition = condition.reshape(batch * glimpses, 1, self.width)
        normalized_tokens = self.read_norm(tokens)
        read, _ = self.read_attn(
            condition,
            normalized_tokens + position,
            normalized_tokens,
            need_weights=False,
        )
        output = self.output(read[:, 0]).reshape(batch, glimpses, self.width)
        spatial_variation = tokens.reshape(
            batch, glimpses, micro, self.width
        ).std(dim=2, unbiased=False).mean()
        return output, {
            "flow_jepa_typed_p1_micro_value_rms": (
                tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p1_spatial_variation": spatial_variation.detach(),
            "flow_jepa_typed_p2_output_rms": (
                output.detach().float().square().mean().sqrt()
            ),
        }


class _StructuredOwnershipLocalRefiner(nn.Module):
    """P2 typed local operations over a lossless 3x3 precision read.

    RGB and learned detail remain separate value-token lanes through local
    attention.  Geometry changes spatial keys; policy, semantic, appearance,
    geometry and horizon each perform an independent read.  Their ordinary
    differentiable contributions meet only at the final action-ready fusion.
    With zero RGB/detail every value and output is exactly zero.
    """


    OWNER_NAMES = ("policy", "semantic", "appearance", "geometry", "horizon")

    def __init__(
        self,
        *,
        width: int,
        raw_dim: int,
        route_dim: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        width = int(width)
        heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
        self.width = width
        self.rgb_value = nn.Linear(3, width, bias=False)
        self.detail_value = nn.Linear(int(raw_dim), width, bias=False)
        self.coordinate_key = nn.Linear(2, width, bias=False)
        self.modality_key = nn.Parameter(torch.randn(2, width) * 0.02)
        self.geometry_key = nn.Linear(int(route_dim), width, bias=False)
        self.owner_conditions = nn.ModuleDict(
            {
                "policy": nn.Linear(int(route_dim), width, bias=False),
                "semantic": nn.Linear(int(route_dim), width, bias=False),
                "appearance": nn.Linear(int(route_dim), width, bias=False),
                "geometry": nn.Linear(int(route_dim), width, bias=False),
                "horizon": nn.Linear(5, width, bias=False),
            }
        )
        self.token_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.token_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    width,
                    heads,
                    dropout=0.0,
                    bias=False,
                    batch_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.ffn_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 2 * width, bias=False),
                    nn.GELU(),
                    nn.Linear(2 * width, width, bias=False),
                )
                for _ in range(int(depth))
            ]
        )
        self.read_norm = VarianceFlooredCenteredNorm(0.25)
        self.read_attn = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.owner_outputs = nn.ModuleDict(
            {
                name: nn.Linear(width, width, bias=False)
                for name in self.OWNER_NAMES
            }
        )

    def forward(
        self,
        *,
        rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
        query: Tensor,
        semantic: Tensor,

        appearance: Tensor,
        geometry: Tensor,
        future_transport: Tensor,
        intervention: str | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if rgb.ndim != 4 or int(rgb.shape[-1]) != 3:
            raise ValueError("structured local RGB must be [N,G,micro,3]")
        if tuple(learned_detail.shape[:-1]) != tuple(rgb.shape[:-1]):
            raise ValueError("structured learned detail does not align with RGB")
        if tuple(coordinates.shape) != (*rgb.shape[:-1], 2):
            raise ValueError("structured coordinates do not align with RGB")
        batch, glimpses, micro, _ = rgb.shape
        for name, value in (
            ("query", query),
            ("semantic", semantic),
            ("appearance", appearance),
            ("geometry", geometry),
        ):
            if tuple(value.shape[:2]) != (batch, glimpses):
                raise ValueError(f"structured local {name} context is misaligned")
        if tuple(future_transport.shape) != (batch, glimpses, 5):
            raise ValueError("structured future transport must be [N,G,5]")

        rgb_tokens = self.rgb_value(rgb)
        detail_tokens = self.detail_value(learned_detail)
        tokens = torch.cat((rgb_tokens, detail_tokens), dim=2)
        coordinate_position = self.coordinate_key(coordinates)
        position = torch.cat((coordinate_position, coordinate_position), dim=2)
        modality_position = torch.cat(
            (
                self.modality_key[0].reshape(1, 1, 1, self.width).expand(
                    batch, glimpses, micro, self.width
                ),
                self.modality_key[1].reshape(1, 1, 1, self.width).expand(
                    batch, glimpses, micro, self.width
                ),
            ),
            dim=2,
        )
        position = position + modality_position.to(dtype=position.dtype)
        tokens = tokens.reshape(batch * glimpses, 2 * micro, self.width)
        position = position.reshape_as(tokens)
        for norm, attention, ffn_norm, ffn in zip(
            self.token_norms,
            self.token_attn,
            self.ffn_norms,
            self.ffns,
        ):
            normalized = norm(tokens)
            update, _ = attention(
                normalized + position,
                normalized + position,
                normalized,
                need_weights=False,
            )
            tokens = tokens + (2.0**-0.5) * update
            tokens = tokens + (2.0**-0.5) * ffn(ffn_norm(tokens))

        owner_inputs = {
            "policy": query,
            "semantic": semantic,
            "appearance": appearance,
            "geometry": geometry,
            "horizon": future_transport,
        }
        owner_queries = torch.stack(
            [
                self.owner_conditions[name](owner_inputs[name])
                for name in self.OWNER_NAMES
            ],
            dim=2,
        ).reshape(batch * glimpses, len(self.OWNER_NAMES), self.width)
        geometry_position = self.geometry_key(geometry).reshape(
            batch * glimpses, 1, self.width
        )
        normalized_tokens = self.read_norm(tokens)
        owner_reads, _ = self.read_attn(
            owner_queries,
            normalized_tokens + position + geometry_position,

            normalized_tokens,
            need_weights=False,
        )
        contributions = {
            name: self.owner_outputs[name](owner_reads[:, index])
            for index, name in enumerate(self.OWNER_NAMES)
        }
        output = sum(contributions.values()) / math.sqrt(
            float(len(self.OWNER_NAMES))
        )
        output = output.reshape(batch, glimpses, self.width)
        spatial_variation = tokens.reshape(
            batch, glimpses, 2 * micro, self.width
        ).std(dim=2, unbiased=False).mean()
        metrics = {
            "flow_jepa_typed_p1_micro_value_rms": (
                tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p1_spatial_variation": spatial_variation.detach(),
            "flow_jepa_typed_p2_output_rms": (
                output.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_lane_rms": (
                rgb_tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_lane_rms": (
                detail_tokens.detach().float().square().mean().sqrt()
            ),
        }
        for name, contribution in contributions.items():
            metrics[f"flow_jepa_typed_p2_{name}_contribution_rms"] = (
                contribution.detach().float().square().mean().sqrt()
            )
        return output, metrics


class _FunctionalOwnershipLocalRefiner(nn.Module):
    """P2 local reader with protected policy content and routed typed deltas.

    RGB and learned-detail patches use separate 3x3 attention lanes, sharing
    weights for efficiency but never forming an 18-token information soup.
    Five typed queries read both lanes.  The policy read is always preserved;
    semantic, appearance, geometry and horizon reads are optional innovations
    selected by a low-rank router before being added to that carrier.
    """

    OWNER_NAMES = ("policy", "semantic", "appearance", "geometry", "horizon")
    DELTA_NAMES = ("semantic", "appearance", "geometry", "horizon")

    def __init__(
        self,
        *,
        width: int,
        raw_dim: int,
        route_dim: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        width = int(width)
        heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
        self.width = width
        self.rgb_value = nn.Linear(3, width, bias=False)
        self.detail_value = nn.Linear(int(raw_dim), width, bias=False)
        self.coordinate_key = nn.Linear(2, width, bias=False)
        self.modality_key = nn.Parameter(torch.randn(2, width) * 0.02)
        self.geometry_key = nn.Linear(int(route_dim), width, bias=False)
        self.owner_conditions = nn.ModuleDict(
            {
                "policy": nn.Linear(int(route_dim), width, bias=False),
                "semantic": nn.Linear(int(route_dim), width, bias=False),
                "appearance": nn.Linear(int(route_dim), width, bias=False),
                "geometry": nn.Linear(int(route_dim), width, bias=False),
                "horizon": nn.Linear(5, width, bias=False),
            }
        )
        self.token_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.token_attn = nn.ModuleList(
            [

                nn.MultiheadAttention(
                    width,
                    heads,
                    dropout=0.0,
                    bias=False,
                    batch_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.ffn_norms = nn.ModuleList(
            [VarianceFlooredCenteredNorm(0.25) for _ in range(int(depth))]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 2 * width, bias=False),
                    nn.GELU(),
                    nn.Linear(2 * width, width, bias=False),
                )
                for _ in range(int(depth))
            ]
        )
        self.read_norm = VarianceFlooredCenteredNorm(0.25)
        self.read_attn = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.owner_outputs = nn.ModuleDict(
            {
                name: nn.Linear(width, width, bias=False)
                for name in self.OWNER_NAMES
            }
        )
        for name in self.DELTA_NAMES:
            nn.init.normal_(
                self.owner_outputs[name].weight, mean=0.0, std=1e-2
            )
        self.delta_router = RoleDeltaAttnRes(
            width,
            min(int(route_dim), width),
            max_sources=len(self.DELTA_NAMES),
            max_value_rms=0.50,
            normalization_floor=0.25,
        )

    def forward(
        self,
        *,
        rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
        query: Tensor,
        semantic: Tensor,
        appearance: Tensor,
        geometry: Tensor,
        future_transport: Tensor,
        intervention: str | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if rgb.ndim != 4 or int(rgb.shape[-1]) != 3:
            raise ValueError("functional local RGB must be [N,G,micro,3]")
        if tuple(learned_detail.shape[:-1]) != tuple(rgb.shape[:-1]):
            raise ValueError("functional learned detail does not align with RGB")
        if tuple(coordinates.shape) != (*rgb.shape[:-1], 2):
            raise ValueError("functional coordinates do not align with RGB")
        batch, glimpses, micro, _ = rgb.shape
        for name, value in (
            ("query", query),
            ("semantic", semantic),
            ("appearance", appearance),
            ("geometry", geometry),
        ):
            if tuple(value.shape[:2]) != (batch, glimpses):
                raise ValueError(f"functional local {name} context is misaligned")
        if tuple(future_transport.shape) != (batch, glimpses, 5):
            raise ValueError("functional future transport must be [N,G,5]")

        rgb_tokens = self.rgb_value(rgb)
        detail_tokens = self.detail_value(learned_detail)
        coordinate_position = self.coordinate_key(coordinates)
        tokens = torch.stack((rgb_tokens, detail_tokens), dim=2)
        position = coordinate_position[:, :, None].expand(
            -1, -1, 2, -1, -1
        )
        position = position + self.modality_key.reshape(
            1, 1, 2, 1, self.width
        ).to(dtype=position.dtype)
        flat_tokens = tokens.reshape(
            batch * glimpses * 2, micro, self.width
        )
        flat_position = position.reshape_as(flat_tokens)
        for norm, attention, ffn_norm, ffn in zip(
            self.token_norms,
            self.token_attn,
            self.ffn_norms,
            self.ffns,
        ):
            normalized = norm(flat_tokens)
            update, _ = attention(
                normalized + flat_position,
                normalized + flat_position,
                normalized,
                need_weights=False,
            )
            flat_tokens = flat_tokens + (2.0**-0.5) * update
            flat_tokens = flat_tokens + (2.0**-0.5) * ffn(
                ffn_norm(flat_tokens)
            )

        owner_inputs = {
            "policy": query,
            "semantic": semantic,
            "appearance": appearance,
            "geometry": geometry,
            "horizon": future_transport,
        }
        owner_query_rows = []
        for name in self.OWNER_NAMES:
            row = self.owner_conditions[name](owner_inputs[name])
            if name == "geometry":
                row = row + self.geometry_key(geometry)
            owner_query_rows.append(row)
        owner_queries = torch.stack(
            owner_query_rows,
            dim=2,
        )
        lane_queries = owner_queries[:, :, None].expand(-1, -1, 2, -1, -1)
        lane_queries = lane_queries + self.modality_key.reshape(
            1, 1, 2, 1, self.width
        ).to(dtype=lane_queries.dtype)
        read_keys = self.read_norm(flat_tokens).reshape(
            batch, glimpses, 2, micro, self.width
        )
        read_keys = read_keys + position
        lane_reads, _ = self.read_attn(
            lane_queries.reshape(
                batch * glimpses * 2, len(self.OWNER_NAMES), self.width
            ),
            read_keys.reshape(batch * glimpses * 2, micro, self.width),
            flat_tokens.reshape(batch * glimpses * 2, micro, self.width),
            need_weights=False,
        )
        typed_lane_reads = lane_reads.reshape(
            batch, glimpses, 2, len(self.OWNER_NAMES), self.width
        )
        owner_index = {
            name: index for index, name in enumerate(self.OWNER_NAMES)
        }
        # Information permissions are functional, not labels on five copies
        # of the same read.  Policy preserves both value lanes; semantic reads
        # only the learned-detail lane; appearance owns literal RGB; geometry
        # may compare both coordinate-keyed lanes; horizon receives their
        # contrast so it cannot duplicate the common appearance read.
        typed_reads = {
            "policy": (
                typed_lane_reads[:, :, 0, owner_index["policy"]]
                + typed_lane_reads[:, :, 1, owner_index["policy"]]

            )
            / math.sqrt(2.0),
            "semantic": typed_lane_reads[
                :, :, 1, owner_index["semantic"]
            ],
            "appearance": typed_lane_reads[
                :, :, 0, owner_index["appearance"]
            ],
            "geometry": (
                typed_lane_reads[:, :, 0, owner_index["geometry"]]
                + typed_lane_reads[:, :, 1, owner_index["geometry"]]
            )
            / math.sqrt(2.0),
            "horizon": (
                typed_lane_reads[:, :, 0, owner_index["horizon"]]
                - typed_lane_reads[:, :, 1, owner_index["horizon"]]
            )
            / math.sqrt(2.0),
        }
        contributions = {
            name: self.owner_outputs[name](typed_reads[name])
            for name in self.OWNER_NAMES
        }
        mode = "" if intervention is None else str(intervention)
        for name in self.DELTA_NAMES:
            if mode == f"p2_{name}_zero":
                contributions[name] = torch.zeros_like(contributions[name])
            elif mode == f"p2_{name}_shuffle":
                source = contributions[name]
                contributions[name] = source.roll(
                    shifts=1,
                    dims=0 if int(source.shape[0]) > 1 else -1,
                )
        policy_carrier = contributions["policy"]
        delta_values = torch.stack(
            [contributions[name] for name in self.DELTA_NAMES], dim=-2
        )
        routed_delta, route_metrics = self.delta_router(
            policy_carrier, delta_values
        )
        output = policy_carrier + routed_delta
        spatial_variation = flat_tokens.reshape(
            batch, glimpses, 2, micro, self.width
        ).std(dim=3, unbiased=False).mean()
        metrics = {
            "flow_jepa_typed_p1_micro_value_rms": (
                flat_tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p1_spatial_variation": spatial_variation.detach(),
            "flow_jepa_typed_p2_output_rms": (
                output.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_lane_rms": (
                rgb_tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_lane_rms": (
                detail_tokens.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_policy_carrier_rms": (
                policy_carrier.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_routed_delta_rms": (
                routed_delta.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_functional_routing": output.new_ones(
                (), dtype=torch.float32
            ),
        }
        for name, contribution in contributions.items():
            metrics[f"flow_jepa_typed_p2_{name}_contribution_rms"] = (
                contribution.detach().float().square().mean().sqrt()
            )
        for key, value in route_metrics.items():
            if key == "source_mass":
                for index, name in enumerate(self.DELTA_NAMES):
                    metrics[
                        f"flow_jepa_typed_p2_{name}_route_mass"
                    ] = value[index].detach()
            elif int(value.numel()) == 1:
                metrics[f"flow_jepa_typed_p2_route_{key}"] = value.detach()

        return output, metrics


class _UtilityPrecisionLocalRefiner(_FunctionalOwnershipLocalRefiner):
    """P2 reader with exact RGB/detail base and precision ownership.

    The four factual lanes are outside the optional owner router. Coordinates
    affect queries and keys only, so zero RGB plus zero learned detail still
    gives an exact-zero policy update. This class intentionally reuses the
    V113 parameter layout, allowing a V113 checkpoint to initialize the new
    graph without inventing an unrelated local reader.
    """

    def forward(
        self,
        *,
        rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
        query: Tensor,
        semantic: Tensor,
        appearance: Tensor,
        geometry: Tensor,
        future_transport: Tensor,
        intervention: str | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if rgb.ndim != 4 or int(rgb.shape[-1]) != 3:
            raise ValueError("utility local RGB must be [N,G,micro,3]")
        if tuple(learned_detail.shape[:-1]) != tuple(rgb.shape[:-1]):
            raise ValueError("utility learned detail does not align with RGB")
        if int(learned_detail.shape[-1]) != int(self.detail_value.in_features):
            raise ValueError("utility learned-detail width is invalid")
        if tuple(coordinates.shape) != (*rgb.shape[:-1], 2):
            raise ValueError("utility coordinates do not align with RGB")
        batch, glimpses, micro, _ = rgb.shape
        if micro < 2:
            raise ValueError("utility precision read requires multiple micro cells")
        for name, value in (
            ("query", query),
            ("semantic", semantic),
            ("appearance", appearance),
            ("geometry", geometry),
        ):
            if tuple(value.shape[:2]) != (batch, glimpses):
                raise ValueError(f"utility local {name} context is misaligned")
        if tuple(future_transport.shape) != (batch, glimpses, 5):
            raise ValueError("utility future transport must be [N,G,5]")

        mode = "" if intervention is None else str(intervention)
        # Split and audit in FP32 even under BF16 autocast.  Only the resulting
        # exact factual lanes are converted back for learned projections.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rgb_f = rgb.float()
            detail_f = learned_detail.float()
            rgb_base_f = rgb_f.mean(dim=2, keepdim=True)
            rgb_precision_f = rgb_f - rgb_base_f
            detail_base_f = detail_f.mean(dim=2, keepdim=True)
            detail_precision_f = detail_f - detail_base_f
            if collect_diagnostics:
                rgb_reconstruction_error = (
                    rgb_base_f + rgb_precision_f - rgb_f
                ).detach().abs().amax()
                detail_reconstruction_error = (
                    detail_base_f + detail_precision_f - detail_f
                ).detach().abs().amax()
                rgb_precision_mean_residual = (
                    rgb_precision_f.mean(dim=2).detach().abs().amax()
                )
                detail_precision_mean_residual = (
                    detail_precision_f.mean(dim=2).detach().abs().amax()
                )
            if mode == "p2_rgb_precision_zero":
                rgb_precision_f = torch.zeros_like(rgb_precision_f)
            elif mode == "p2_rgb_precision_spatial_shuffle":
                rgb_precision_f = rgb_precision_f.roll(shifts=1, dims=2)
            if mode == "p2_detail_precision_zero":
                detail_precision_f = torch.zeros_like(detail_precision_f)
            elif mode == "p2_detail_precision_spatial_shuffle":
                detail_precision_f = detail_precision_f.roll(

                    shifts=1, dims=2
                )
        rgb_base_raw = rgb_base_f.to(dtype=rgb.dtype)
        rgb_precision_raw = rgb_precision_f.to(dtype=rgb.dtype)
        detail_base_raw = detail_base_f.to(dtype=learned_detail.dtype)
        detail_precision_raw = detail_precision_f.to(
            dtype=learned_detail.dtype
        )

        rgb_base = self.rgb_value(rgb_base_raw)[:, :, 0]
        detail_base = self.detail_value(detail_base_raw)[:, :, 0]
        rgb_precision = self.rgb_value(rgb_precision_raw)
        detail_precision = self.detail_value(detail_precision_raw)
        coordinate_position = self.coordinate_key(coordinates)
        precision_tokens = torch.stack((rgb_precision, detail_precision), dim=2)
        position = coordinate_position[:, :, None].expand(
            -1, -1, 2, -1, -1
        )
        position = position + self.modality_key.reshape(
            1, 1, 2, 1, self.width
        ).to(dtype=position.dtype)
        flat_tokens = precision_tokens.reshape(
            batch * glimpses * 2, micro, self.width
        )
        flat_position = position.reshape_as(flat_tokens)
        for norm, attention, ffn_norm, ffn in zip(
            self.token_norms,
            self.token_attn,
            self.ffn_norms,
            self.ffns,
        ):
            normalized = norm(flat_tokens)
            precision_update, _ = attention(
                normalized + flat_position,
                normalized + flat_position,
                normalized,
                need_weights=False,
            )
            flat_tokens = flat_tokens + (2.0**-0.5) * precision_update
            flat_tokens = flat_tokens + (2.0**-0.5) * ffn(
                ffn_norm(flat_tokens)
            )

        owner_inputs = {
            "policy": query,
            "semantic": semantic,
            "appearance": appearance,
            "geometry": geometry,
            "horizon": future_transport,
        }
        owner_query_rows: list[Tensor] = []
        for name in self.OWNER_NAMES:
            row = self.owner_conditions[name](owner_inputs[name])
            if name == "geometry":
                row = row + self.geometry_key(geometry)
            owner_query_rows.append(row)
        owner_queries = torch.stack(owner_query_rows, dim=2)
        lane_queries = owner_queries[:, :, None].expand(-1, -1, 2, -1, -1)
        lane_queries = lane_queries + self.modality_key.reshape(
            1, 1, 2, 1, self.width
        ).to(dtype=lane_queries.dtype)
        read_keys = self.read_norm(flat_tokens).reshape(
            batch, glimpses, 2, micro, self.width
        )
        read_keys = read_keys + position
        lane_reads, _ = self.read_attn(
            lane_queries.reshape(
                batch * glimpses * 2, len(self.OWNER_NAMES), self.width
            ),
            read_keys.reshape(batch * glimpses * 2, micro, self.width),
            flat_tokens.reshape(batch * glimpses * 2, micro, self.width),
            need_weights=False,
        )
        typed_lane_reads = lane_reads.reshape(
            batch, glimpses, 2, len(self.OWNER_NAMES), self.width
        )
        owner_index = {
            name: index for index, name in enumerate(self.OWNER_NAMES)
        }
        typed_reads = {

            "policy": (
                typed_lane_reads[:, :, 0, owner_index["policy"]]
                + typed_lane_reads[:, :, 1, owner_index["policy"]]
            )
            / math.sqrt(2.0),
            "semantic": typed_lane_reads[
                :, :, 1, owner_index["semantic"]
            ],
            "appearance": typed_lane_reads[
                :, :, 0, owner_index["appearance"]
            ],
            "geometry": (
                typed_lane_reads[:, :, 0, owner_index["geometry"]]
                + typed_lane_reads[:, :, 1, owner_index["geometry"]]
            )
            / math.sqrt(2.0),
            "horizon": (
                typed_lane_reads[:, :, 0, owner_index["horizon"]]
                - typed_lane_reads[:, :, 1, owner_index["horizon"]]
            )
            / math.sqrt(2.0),
        }

        policy_output = self.owner_outputs["policy"]
        rgb_base_carrier = policy_output(rgb_base)
        detail_base_carrier = policy_output(detail_base)
        rgb_precision_carrier = policy_output(
            typed_lane_reads[:, :, 0, owner_index["policy"]]
        )
        detail_precision_carrier = policy_output(
            typed_lane_reads[:, :, 1, owner_index["policy"]]
        )
        protected_base = (
            rgb_base_carrier + detail_base_carrier
        ) / math.sqrt(2.0)
        protected_precision = (
            rgb_precision_carrier + detail_precision_carrier
        ) / math.sqrt(2.0)
        policy_carrier = protected_base + protected_precision
        contributions = {
            name: self.owner_outputs[name](typed_reads[name])
            for name in self.DELTA_NAMES
        }
        for name in self.DELTA_NAMES:
            if mode == f"p2_{name}_zero":
                contributions[name] = torch.zeros_like(contributions[name])
            elif mode == f"p2_{name}_shuffle":
                source = contributions[name]
                contributions[name] = (
                    source.roll(shifts=1, dims=0)
                    if int(source.shape[0]) > 1
                    else source
                )
        delta_values = torch.stack(
            [contributions[name] for name in self.DELTA_NAMES], dim=-2
        )
        routed_delta, route_metrics = self.delta_router(
            policy_carrier,
            delta_values,
            collect_diagnostics=collect_diagnostics,
        )
        output = policy_carrier + routed_delta
        if not collect_diagnostics:
            return output, {}
        precision_view = flat_tokens.reshape(
            batch, glimpses, 2, micro, self.width
        )
        spatial_variation = precision_view.std(
            dim=3, unbiased=False
        ).mean()
        metrics = {
            "flow_jepa_typed_p1_micro_value_rms": (
                precision_view.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p1_spatial_variation": spatial_variation.detach(),
            "flow_jepa_typed_p1_rgb_reconstruction_error": (
                rgb_reconstruction_error
            ),
            "flow_jepa_typed_p1_detail_reconstruction_error": (
                detail_reconstruction_error

            ),
            "flow_jepa_typed_p1_rgb_precision_mean_residual": (
                rgb_precision_mean_residual
            ),
            "flow_jepa_typed_p1_detail_precision_mean_residual": (
                detail_precision_mean_residual
            ),
            "flow_jepa_typed_p2_output_rms": (
                output.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_base_rms": (
                rgb_base.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_precision_rms": (
                rgb_precision.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_base_rms": (
                detail_base.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_precision_rms": (
                detail_precision.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_policy_carrier_rms": (
                policy_carrier.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_protected_base_rms": (
                protected_base.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_protected_precision_rms": (
                protected_precision.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_base_carrier_rms": (
                rgb_base_carrier.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_base_carrier_rms": (
                detail_base_carrier.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_rgb_precision_carrier_rms": (
                rgb_precision_carrier.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_detail_precision_carrier_rms": (
                detail_precision_carrier.detach()
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "flow_jepa_typed_p2_routed_delta_rms": (
                routed_delta.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_typed_p2_functional_routing": output.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_typed_p2_utility_precision": output.new_ones(
                (), dtype=torch.float32
            ),
        }
        for name, contribution in contributions.items():
            metrics[f"flow_jepa_typed_p2_{name}_contribution_rms"] = (
                contribution.detach().float().square().mean().sqrt()
            )
        for key, value in route_metrics.items():
            if key == "source_mass":
                for index, name in enumerate(self.DELTA_NAMES):
                    metrics[
                        f"flow_jepa_typed_p2_{name}_route_mass"
                    ] = value[index].detach()
            elif int(value.numel()) == 1:
                metrics[f"flow_jepa_typed_p2_route_{key}"] = value.detach()
        return output, metrics


class LateRawDetailPolicyReader(nn.Module):
    """Read flow-addressed raw detail at the world-to-policy boundary.

    Queries are explicitly organized as ``[action horizon, basis]`` and combine
    the current trajectory token with the matching world-horizon summary.
    Selector projections are bias-free and the already policy-dimensional raw
    value residual is consumed directly.  Consequently, zero detail produces
    an exact zero update and there is no learned amplitude gate that can delete

    the route.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        heads = int(config.flow_jepa_raw_reader_heads)
        if hidden % heads:
            raise ValueError("late raw-detail hidden size must be divisible by heads")
        self.config = config
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.fixed_scale = float(
            getattr(config, "flow_jepa_late_policy_detail_scale", 0.25)
        )
        self.soft_address_lattice = bool(
            int(getattr(config, "flow_jepa_soft_address_lattice", 0))
        )
        self.policy_multi_glimpse_address = bool(
            int(getattr(config, "flow_jepa_policy_multi_glimpse_address", 0))
        )
        self.coordinate_typed_raw_detail = bool(
            int(getattr(config, "flow_jepa_coordinate_typed_raw_detail", 0))
        )
        self.structured_ownership = bool(
            int(getattr(config, "flow_jepa_structured_ownership_bottleneck", 0))
        )
        self.pre_value_owner_routing = bool(
            int(getattr(config, "flow_jepa_pre_value_owner_routing", 0))
        )
        self.functional_mainline_routing = bool(
            int(getattr(config, "flow_jepa_functional_mainline_routing", 0))
        )
        self.utility_precision_mainline = bool(
            int(getattr(config, "flow_jepa_utility_precision_mainline", 0))
        )
        self.shared_factual_glimpse_bank = bool(
            int(getattr(config, "flow_jepa_shared_factual_glimpse_bank", 0))
        )
        self.g_aligned_future_effect = bool(
            int(getattr(config, "flow_jepa_g_aligned_future_effect", 0))
        )
        self.differential_intent_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_differential_intent_effect_mainline",
                    0,
                )
            )
        )
        self.grounded_intent_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_grounded_intent_effect_mainline",
                    0,
                )
            )
        )
        self.object_intent_dynamics_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_object_intent_dynamics_mainline",
                    0,
                )
            )
        )
        self.address_query_batch_budget = int(
            getattr(config, "flow_jepa_address_query_batch_budget", 32)
        )
        self.microgrid_tile = int(
            getattr(config, "flow_jepa_microgrid_tile", 3)
        )
        self.p1_mixed_precision = bool(
            int(getattr(config, "flow_jepa_p1_mixed_precision", 0))
        )
        self.checkpoint_min_batch = int(

            getattr(config, "flow_jepa_checkpoint_min_batch", 4)
        )
        self.raw_activation_checkpoint = bool(
            int(getattr(config, "flow_jepa_raw_activation_checkpoint", 1))
        )
        complete_numerics = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        normalization_floor = float(
            getattr(config, "flow_jepa_routing_norm_floor", 0.25)
        )

        def route_norm(width: int) -> nn.Module:
            if complete_numerics:
                return VarianceFlooredCenteredNorm(normalization_floor)
            return nn.LayerNorm(width, elementwise_affine=False)
        # Evaluation-only posterior interventions. Plain Python state keeps
        # probes outside checkpoints and cannot affect training by accident.
        self._address_eval_intervention: str | None = None
        self._address_eval_apply_count = 0
        self._address_eval_metrics: dict[str, float] = {}
        self.phase_query_proj = (
            nn.Linear(hidden, hidden, bias=False)
            if int(getattr(config, "stateless_phase_enabled", 0))
            else None
        )
        self.condition_query_proj = (
            nn.Linear(hidden, hidden, bias=False)
            if (
                int(getattr(config, "stateless_phase_enabled", 0))
                and not self.differential_intent_effect_mainline
            )
            else None
        )
        self.history_query_proj = (
            nn.Linear(hidden, hidden, bias=False)
            if (
                self.functional_mainline_routing
                and not self.differential_intent_effect_mainline
            )
            else None
        )
        self.phase_query_scale = float(
            getattr(config, "stateless_phase_query_scale", 0.10)
        )
        self.query_norm = route_norm(2 * hidden)
        self.key_norm = route_norm(hidden)
        self.query_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        if self.soft_address_lattice:
            route_dim = int(config.flow_jepa_address_route_dim)
            raw_dim = int(config.flow_jepa_raw_base_channels)
            raw_dim = raw_dim + raw_dim // 2
            self.lattice_route_dim = route_dim
            self.lattice_raw_dim = raw_dim
            self.lattice_query_norm = route_norm(2 * hidden)
            self.lattice_query_proj = nn.Linear(
                2 * hidden,
                route_dim * (heads if self.policy_multi_glimpse_address else 1),
                bias=False,
            )
            if self.utility_precision_mainline:
                # P1 is a factual set read: it may see clean horizon/basis
                # identities and W/goal/phase/history context, but never the
                # noisy action sample.  The existing lattice query remains the
                # action-dependent P2 query, preserving four distinct basis
                # consumers without repeating the expensive spatial posterior.
                self.shared_p1_basis_norm = route_norm(hidden)
                self.shared_p1_basis_key = nn.Linear(
                    hidden, route_dim, bias=False
                )
                self.shared_p1_context_norm = route_norm(hidden)
                self.shared_p1_context_query = nn.Linear(
                    hidden, route_dim, bias=False
                )
                self.shared_p1_glimpse_identity = nn.Parameter(
                    torch.randn(heads, route_dim) * 0.02
                )
                if self.shared_factual_glimpse_bank:
                    self.shared_p1_role_query = nn.ModuleDict(

                        {
                            name: nn.Linear(hidden, route_dim, bias=False)
                            for name in (
                                "semantic",
                                "appearance",
                                "geometry",
                                "coverage",
                            )
                        }
                    )
                    # These are soft initial preferences, not fixed ownership
                    # masks. All three typed owners remain reachable from every
                    # glimpse and ordinary action gradients may change them.
                    self.shared_p1_owner_mix_logits = nn.Parameter(
                        torch.tensor(
                            (
                                (2.0, 0.0, 0.0),
                                (0.0, 2.0, 0.0),
                                (0.0, 0.0, 2.0),
                                (0.0, 0.0, 0.0),
                            )
                        )
                    )
                    self.shared_p2_glimpse_query = nn.Linear(
                        route_dim, route_dim, bias=False
                    )
                    self.shared_p2_glimpse_key = nn.Linear(
                        route_dim, route_dim, bias=False
                    )
                else:
                    self.shared_p1_role_query = None
                    self.register_parameter(
                        "shared_p1_owner_mix_logits", None
                    )
                    self.shared_p2_glimpse_query = None
                    self.shared_p2_glimpse_key = None
            else:
                self.shared_p1_basis_norm = None
                self.shared_p1_basis_key = None
                self.shared_p1_context_norm = None
                self.shared_p1_context_query = None
                self.register_parameter("shared_p1_glimpse_identity", None)
                self.shared_p1_role_query = None
                self.register_parameter("shared_p1_owner_mix_logits", None)
                self.shared_p2_glimpse_query = None
                self.shared_p2_glimpse_key = None
            self.lattice_key_norm = route_norm(route_dim)
            # The observation bank is compiled before the world stack and is
            # therefore safe to cache across ODE steps.  World organization is
            # query-side state: project each W chart cell into the address
            # routing space instead of averaging xy before the precision read.
            # This changes only selector logits; raw precision values remain
            # observation-owned and cannot be rewritten by the world path.
            self.lattice_world_norm = route_norm(hidden)
            self.lattice_world_key_proj = nn.Linear(
                hidden, route_dim, bias=False
            )
            nn.init.normal_(
                self.lattice_world_key_proj.weight,
                mean=0.0,
                std=3e-2,
            )
            if self.coordinate_typed_raw_detail:
                self.lattice_value_out = nn.Identity()
            elif self.policy_multi_glimpse_address:
                self.lattice_value_out = nn.ModuleList(
                    [
                        nn.Sequential(
                            route_norm(raw_dim),
                            nn.Linear(raw_dim, self.head_dim, bias=False),
                            nn.GELU(),
                            nn.Linear(self.head_dim, self.head_dim, bias=False),
                        )
                        for _ in range(heads)
                    ]
                )
            else:
                self.lattice_value_out = nn.Sequential(
                    route_norm(raw_dim),
                    nn.Linear(raw_dim, hidden, bias=False),

                    nn.GELU(),
                    nn.Linear(hidden, hidden, bias=False),
                )
            self.lattice_fine_evidence_scale = nn.Parameter(torch.tensor(0.25))
            if self.coordinate_typed_raw_detail:
                if not self.policy_multi_glimpse_address:
                    raise ValueError(
                        "coordinate-typed P1/P2 requires multi-glimpse addressing"
                    )
                self.raw_micro_grid = int(config.flow_jepa_raw_micro_grid)
                fine_side = 2 * int(config.flow_jepa_raw_reader_radius) + 1
                fine_axis = torch.linspace(-1.0, 1.0, fine_side)
                fine_y, fine_x = torch.meshgrid(
                    fine_axis, fine_axis, indexing="ij"
                )
                fine_points = torch.stack(
                    (fine_x.reshape(-1), fine_y.reshape(-1)), dim=-1
                )
                micro_axis = torch.linspace(-1.0, 1.0, self.raw_micro_grid)
                micro_y, micro_x = torch.meshgrid(
                    micro_axis, micro_axis, indexing="ij"
                )
                micro_centers = torch.stack(
                    (micro_x.reshape(-1), micro_y.reshape(-1)), dim=-1
                )
                spacing = 2.0 / float(max(self.raw_micro_grid - 1, 1))
                micro_basis = torch.exp(
                    -0.5
                    * (
                        fine_points[:, None] - micro_centers[None]
                    ).square().sum(dim=-1)
                    / float(max((0.75 * spacing) ** 2, 1e-4))
                )
                # Each fine point contributes across nearby micro cells.  The
                # P1 posterior is later renormalized inside every micro cell.
                micro_basis = micro_basis / micro_basis.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                self.register_buffer(
                    "typed_micro_basis", micro_basis, persistent=False
                )
                self.typed_fine_query = nn.ModuleDict(
                    {
                        name: nn.Linear(route_dim, route_dim, bias=False)
                        for name in ("semantic", "appearance", "geometry")
                    }
                )
                if self.utility_precision_mainline:
                    # Semantic owns the coarse source/slot evidence in V114;
                    # no semantic fine-candidate tensor is materialized.
                    # Retain the serialized V113 parameter but keep it out of
                    # the optimizer instead of presenting a knowingly dead
                    # trainable module.
                    self.typed_fine_query["semantic"].requires_grad_(False)
                self.typed_coarse_query = nn.ModuleDict(
                    {
                        name: nn.Linear(route_dim, route_dim, bias=False)
                        for name in ("semantic", "appearance", "geometry")
                    }
                )
                self.appearance_world_owner_query = (
                    nn.Linear(route_dim, route_dim, bias=False)
                    if self.functional_mainline_routing
                    else None
                )
                if (
                    self.g_aligned_future_effect
                    and self.appearance_world_owner_query is not None
                ):
                    # V115 P1 addresses only protected G3 current facts.
                    # Retain the V113/V114 gateway in serialized ancestry, but
                    # do not give a dead W->P1 scorer an optimizer owner.
                    self.appearance_world_owner_query.requires_grad_(False)
                self.typed_local_refiners = nn.ModuleList(
                    [
                        (
                            _UtilityPrecisionLocalRefiner
                            if self.utility_precision_mainline
                            else _FunctionalOwnershipLocalRefiner
                            if self.functional_mainline_routing

                            else _StructuredOwnershipLocalRefiner
                            if self.structured_ownership
                            else _CoordinateTypedLocalRefiner
                        )(
                            width=self.head_dim,
                            raw_dim=raw_dim,
                            route_dim=route_dim,
                            depth=2,
                        )
                        for _ in range(heads)
                    ]
                )
            else:
                self.raw_micro_grid = 0
                self.register_buffer(
                    "typed_micro_basis", None, persistent=False
                )
                self.typed_fine_query = None
                self.typed_coarse_query = None
                self.appearance_world_owner_query = None
                self.typed_local_refiners = None
            self.query_proj.requires_grad_(False)
            self.key_proj.requires_grad_(False)
        else:
            self.lattice_route_dim = 0
            self.lattice_raw_dim = 0
            self.lattice_query_norm = None
            self.lattice_query_proj = None
            self.lattice_key_norm = None
            self.lattice_world_norm = None
            self.lattice_world_key_proj = None
            self.lattice_value_out = None
            self.shared_p1_basis_norm = None
            self.shared_p1_basis_key = None
            self.shared_p1_context_norm = None
            self.shared_p1_context_query = None
            self.register_parameter("shared_p1_glimpse_identity", None)
            self.shared_p1_role_query = None
            self.register_parameter("shared_p1_owner_mix_logits", None)
            self.shared_p2_glimpse_query = None
            self.shared_p2_glimpse_key = None
            self.register_parameter("lattice_fine_evidence_scale", None)
            self.raw_micro_grid = 0
            self.register_buffer("typed_micro_basis", None, persistent=False)
            self.typed_fine_query = None
            self.typed_coarse_query = None
            self.appearance_world_owner_query = None
            self.typed_local_refiners = None

    def _shared_factual_p1_query(
        self,
        *,
        clean_basis_tokens: Tensor,
        factual_condition: Tensor,
        world_horizon: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build four action-invariant factual queries per horizon.

        A learned set read summarizes the clean basis identities.  It is
        deliberately conditioned on W and the non-action phase/goal/history
        context, while the noisy trajectory remains owned by P2.
        """

        if (
            self.shared_p1_basis_norm is None
            or self.shared_p1_basis_key is None
            or self.shared_p1_context_norm is None
            or self.shared_p1_context_query is None
            or self.shared_p1_glimpse_identity is None
        ):
            raise RuntimeError("shared factual P1 query modules are incomplete")
        if clean_basis_tokens.ndim != 4:
            raise ValueError("clean basis tokens must be [B,T,K,H]")
        batch, horizon, basis, hidden = clean_basis_tokens.shape
        if hidden != self.hidden:
            raise ValueError("clean basis token width does not match the reader")
        if tuple(factual_condition.shape) != (batch, horizon, hidden):
            raise ValueError("factual condition must be [B,T,H]")
        if (
            world_horizon.ndim != 4

            or tuple(world_horizon.shape[:2]) != (batch, horizon)
            or int(world_horizon.shape[-1]) != hidden
        ):
            raise ValueError("shared factual W context must be [B,T,C,H]")
        cameras = int(world_horizon.shape[2])
        basis_input = (
            clean_basis_tokens + factual_condition[:, :, None]
        ) / math.sqrt(2.0)
        basis_key = self.shared_p1_basis_key(
            self.shared_p1_basis_norm(basis_input)
        )
        contextual_world = (
            world_horizon + factual_condition[:, :, None]
        ) / math.sqrt(2.0)
        normalized_contextual_world = self.shared_p1_context_norm(
            contextual_world
        )
        public_context_query = self.shared_p1_context_query(
            normalized_contextual_world
        )
        if self.shared_factual_glimpse_bank:
            if self.shared_p1_role_query is None:
                raise RuntimeError("V115 factual role queries are incomplete")
            role_queries = torch.stack(
                tuple(
                    self.shared_p1_role_query[name](
                        normalized_contextual_world
                    )
                    for name in (
                        "semantic",
                        "appearance",
                        "geometry",
                        "coverage",
                    )
                ),
                dim=2,
            )
            glimpse_query = (
                public_context_query[:, :, None] + role_queries
            ) / math.sqrt(2.0)
        else:
            glimpse_query = public_context_query[:, :, None]
        glimpse_query = (
            glimpse_query
            + self.shared_p1_glimpse_identity.reshape(
                1, 1, self.heads, 1, self.lattice_route_dim
            ).to(
                device=public_context_query.device,
                dtype=public_context_query.dtype,
            )
        )
        with torch.autocast(
            device_type=public_context_query.device.type, enabled=False
        ):
            basis_logits = torch.einsum(
                "btgcr,btkr->btgck",
                glimpse_query.float(),
                basis_key.float(),
            ) * (float(self.lattice_route_dim) ** -0.5)
            basis_weights = torch.softmax(basis_logits, dim=-1)
            basis_summary = torch.einsum(
                "btgck,btkr->btgcr",
                basis_weights,
                basis_key.float(),
            )
        factual_query = (
            glimpse_query.float() + basis_summary
        ) / math.sqrt(2.0)
        entropy = -(
            basis_weights.clamp_min(1e-8)
            * basis_weights.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(max(basis, 2)))
        if tuple(factual_query.shape) != (
            batch,
            horizon,
            self.heads,
            cameras,
            self.lattice_route_dim,
        ):
            raise RuntimeError("shared factual P1 query has an invalid layout")

        return (
            factual_query.to(dtype=public_context_query.dtype),
            entropy.mean().detach(),
        )

    def _shared_factual_owner_weights(self) -> Tensor:
        """Return soft owner preferences for the four factual glimpses."""

        if (
            not self.shared_factual_glimpse_bank
            or self.shared_p1_owner_mix_logits is None
        ):
            raise RuntimeError("V115 factual owner preferences are unavailable")
        if tuple(self.shared_p1_owner_mix_logits.shape) != (self.heads, 3):
            raise RuntimeError("V115 factual owner preference layout is invalid")
        return torch.softmax(
            self.shared_p1_owner_mix_logits.float(), dim=-1
        )

    @staticmethod
    def _typed_microgrid_expectation(
        route_weights: Tensor,
        fine_weights: Tensor,
        micro_basis: Tensor,
        literal_rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
        *,
        micro_tile: int = 1,
        mixed_precision_values: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Aggregate one micro cell at a time without a state x micro volume.

        This is the exact V110 posterior factorization used by the original
        implementation, evaluated in a memory-safe contraction order.  The
        fine posterior is normalized independently for every micro cell, then
        values are reduced over candidates and finally over coarse states.
        No candidate, coordinate, value channel, or soft probability is
        removed; only the materialization order changes.
        """

        if fine_weights.ndim != 8 or route_weights.ndim != 7:
            raise ValueError("typed micro read expects fine/state posterior tensors")
        if tuple(route_weights.shape) != tuple(fine_weights.shape[:-1]):
            raise ValueError(
                "typed fine and route posteriors do not align: "
                f"route={tuple(route_weights.shape)} "
                f"fine_prefix={tuple(fine_weights.shape[:-1])}"
            )
        if micro_basis.ndim != 2 or int(micro_basis.shape[0]) != int(
            fine_weights.shape[-1]
        ):
            raise ValueError("typed micro basis does not match fine candidates")
        value_prefix = (
            int(fine_weights.shape[0]),
            int(fine_weights.shape[3]),
            int(fine_weights.shape[4]),
            int(fine_weights.shape[5]),
            int(fine_weights.shape[6]),
            int(fine_weights.shape[7]),
        )
        for name, value in (
            ("literal RGB", literal_rgb),
            ("learned detail", learned_detail),
            ("coordinates", coordinates),
        ):
            if tuple(value.shape[:-1]) != value_prefix:
                raise ValueError(f"typed {name} does not align with fine candidates")

        micro_tile = int(micro_tile)
        if micro_tile < 1:
            raise ValueError("typed micro tile must be positive")
        route_f = route_weights.float()
        fine_f = fine_weights.float()
        basis_f = micro_basis.to(device=fine_f.device, dtype=torch.float32)
        if micro_tile > 1:
            rgb_width = int(literal_rgb.shape[-1])
            detail_width = int(learned_detail.shape[-1])
            rgb_detail = torch.cat((literal_rgb, learned_detail), dim=-1)
            coordinate_f = coordinates.float()

            rgb_rows: list[Tensor] = []
            detail_rows: list[Tensor] = []
            coordinate_rows: list[Tensor] = []
            for micro_start in range(0, int(basis_f.shape[1]), micro_tile):
                micro_stop = min(
                    micro_start + micro_tile, int(basis_f.shape[1])
                )
                local_weight = (
                    fine_f[..., None]
                    * basis_f[:, micro_start:micro_stop]
                )
                local_weight = local_weight / local_weight.sum(
                    dim=-2, keepdim=True
                ).clamp_min(1e-8)
                joint_weight = route_f[..., None, None] * local_weight
                if mixed_precision_values:
                    value_weight = joint_weight.to(dtype=rgb_detail.dtype)
                    rgb_detail_rows = torch.einsum(
                        "bqgcijmku,bcijmkv->bqguv",
                        value_weight,
                        rgb_detail,
                    ).float()
                    coordinate_tile = torch.einsum(
                        "bqgcijmku,bcijmkd->bqgud",
                        joint_weight,
                        coordinate_f,
                    )
                else:
                    combined_value = torch.cat(
                        (rgb_detail.float(), coordinate_f), dim=-1
                    )
                    combined_tile = torch.einsum(
                        "bqgcijmku,bcijmkv->bqguv",
                        joint_weight,
                        combined_value,
                    )
                    rgb_detail_rows = combined_tile[
                        ..., : rgb_width + detail_width
                    ]
                    coordinate_tile = combined_tile[
                        ..., rgb_width + detail_width :
                    ]
                rgb_rows.append(rgb_detail_rows[..., :rgb_width])
                detail_rows.append(
                    rgb_detail_rows[
                        ..., rgb_width : rgb_width + detail_width
                    ]
                )
                coordinate_rows.append(coordinate_tile)
            return (
                torch.cat(rgb_rows, dim=-2),
                torch.cat(detail_rows, dim=-2),
                torch.cat(coordinate_rows, dim=-2),
            )
        rgb_f = literal_rgb.float()
        detail_f = learned_detail.float()
        coordinate_f = coordinates.float()
        rgb_rows: list[Tensor] = []
        detail_rows: list[Tensor] = []
        coordinate_rows: list[Tensor] = []
        for micro_index in range(int(basis_f.shape[1])):
            local_weight = fine_f * basis_f[:, micro_index]
            local_weight = local_weight / local_weight.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            state_rgb = torch.einsum(
                "bqgcijmk,bcijmkv->bqgcijmv", local_weight, rgb_f
            )
            state_detail = torch.einsum(
                "bqgcijmk,bcijmkv->bqgcijmv", local_weight, detail_f
            )
            state_coordinate = torch.einsum(
                "bqgcijmk,bcijmkd->bqgcijmd", local_weight, coordinate_f
            )
            rgb_rows.append(
                torch.einsum("bqgcijm,bqgcijmv->bqgv", route_f, state_rgb)
            )
            detail_rows.append(
                torch.einsum("bqgcijm,bqgcijmv->bqgv", route_f, state_detail)
            )

            coordinate_rows.append(
                torch.einsum(
                    "bqgcijm,bqgcijmd->bqgd", route_f, state_coordinate
                )
            )
        return (
            torch.stack(rgb_rows, dim=-2),
            torch.stack(detail_rows, dim=-2),
            torch.stack(coordinate_rows, dim=-2),
        )

    def _configured_typed_microgrid_expectation(
        self,
        route_weights: Tensor,
        fine_weights: Tensor,
        micro_basis: Tensor,
        literal_rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._typed_microgrid_expectation(
            route_weights,
            fine_weights,
            micro_basis,
            literal_rgb,
            learned_detail,
            coordinates,
            micro_tile=(
                self.microgrid_tile
                if self.utility_precision_mainline
                else 1
            ),
            mixed_precision_values=(
                self.p1_mixed_precision
                if self.utility_precision_mainline
                else False
            ),
        )

    def set_address_eval_intervention(self, mode: str) -> None:
        normalized = str(mode).strip().lower().replace("-", "_")
        allowed = {
            "none",
            "address_posterior_uniform",
            "fine_offset_zero",
            "camera_posterior_uniform",
            "camera_swap",
            "world_query_zero",
            "world_query_spatial_shuffle",
            "future_transport_neutral",
            "future_transport_spatial_shuffle",
            "semantic_owner_zero",
            "semantic_owner_shuffle",
            "appearance_owner_zero",
            "appearance_owner_shuffle",
            "geometry_owner_zero",
            "geometry_owner_shuffle",
            "p1_appearance_gateway_zero",
            "p1_appearance_gateway_spatial_shuffle",
            "p2_semantic_zero",
            "p2_semantic_shuffle",
            "p2_appearance_zero",
            "p2_appearance_shuffle",
            "p2_geometry_zero",
            "p2_geometry_shuffle",
            "p2_horizon_zero",
            "p2_horizon_shuffle",
            "p2_rgb_precision_zero",
            "p2_rgb_precision_spatial_shuffle",
            "p2_detail_precision_zero",
            "p2_detail_precision_spatial_shuffle",
            "p2_basis0_zero",
            "p2_basis0_horizon_shuffle",
            "p2_basis1_zero",
            "p2_basis1_horizon_shuffle",
            "p2_basis2_zero",
            "p2_basis2_horizon_shuffle",
            "p2_basis3_zero",
            "p2_basis3_horizon_shuffle",
        }

        if normalized not in allowed:
            raise ValueError(
                "address intervention must be none/address_posterior_uniform/"
                "fine_offset_zero/camera_posterior_uniform/camera_swap/"
                "world_query_zero/world_query_spatial_shuffle/"
                "future_transport_neutral/future_transport_spatial_shuffle/"
                "semantic_owner_zero/semantic_owner_shuffle/"
                "appearance_owner_zero/appearance_owner_shuffle/"
                "geometry_owner_zero/geometry_owner_shuffle or one "
                "p1_appearance_gateway_zero/spatial_shuffle or one "
                "p2_semantic/appearance/geometry/horizon zero/shuffle mode "
                "or one p2_rgb/detail_precision zero/spatial_shuffle mode "
                "or one p2_basis[0-3] zero/horizon_shuffle mode"
            )
        if self.training:
            raise RuntimeError("address-posterior intervention is evaluation-only")
        if not self.soft_address_lattice:
            raise RuntimeError(
                "address-posterior intervention requires the soft address lattice"
            )
        if normalized.startswith("p1_appearance_gateway_") and not (
            self.functional_mainline_routing
        ):
            raise RuntimeError(
                "P1 appearance-gateway intervention requires functional "
                "mainline routing"
            )
        if normalized.startswith(("p2_rgb_precision_", "p2_detail_precision_")) and not (
            self.utility_precision_mainline
        ):
            raise RuntimeError(
                "P2 precision intervention requires utility/precision mainline"
            )
        if normalized.startswith("p2_basis") and not (
            self.utility_precision_mainline
        ):
            raise RuntimeError(
                "P2 basis intervention requires utility/precision mainline"
            )
        self._address_eval_intervention = normalized
        self._address_eval_apply_count = 0
        self._address_eval_metrics = {}

    def clear_address_eval_intervention(self) -> None:
        self._address_eval_intervention = None
        self._address_eval_apply_count = 0
        self._address_eval_metrics = {}

    def address_eval_intervention_state(
        self,
    ) -> dict[str, str | int | float]:
        return {
            "mode": (
                "disabled"
                if self._address_eval_intervention is None
                else self._address_eval_intervention
            ),
            "apply_count": int(self._address_eval_apply_count),
            **self._address_eval_metrics,
        }

    def _read_soft_address_lattice(
        self,
        query_input: Tensor,
        trajectory: Tensor,
        world_horizon_grid: Tensor,
        detail: LateRawDetailEvidence,
        *,
        clean_basis_tokens: Tensor | None = None,
        factual_condition: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bank = detail.address_bank
        if (
            bank is None
            or self.lattice_query_norm is None
            or self.lattice_query_proj is None
            or self.lattice_key_norm is None
            or self.lattice_world_norm is None
            or self.lattice_world_key_proj is None

            or self.lattice_value_out is None
            or self.lattice_fine_evidence_scale is None
        ):
            raise RuntimeError("soft address lattice reader is incomplete")
        batch, horizon, basis, cameras, _ = query_input.shape
        coarse_keys = bank.coarse_keys
        fine_keys = bank.fine_keys
        fine_values = bank.fine_values
        fine_valid = bank.fine_valid
        progressive = detail.progressive_address
        progressive_coarse_bias: Tensor | None = None
        progressive_fine_bias: Tensor | None = None
        progressive_world_source_bias: Tensor | None = None
        progressive_world_owner_source_bias: dict[str, Tensor] = {}
        progressive_world_appearance_fine_query: Tensor | None = None
        progressive_future_transport: Tensor | None = None
        if progressive is not None:
            if progressive.stage != 3:
                raise RuntimeError(
                    "policy detail read requires the completed G3 selector state"
                )
            progressive_coarse_bias = progressive.canonical_coarse_bias
            progressive_fine_bias = progressive.canonical_fine_bias
            if (
                progressive_coarse_bias is None
                or progressive_fine_bias is None
                or progressive.canonical_slot_keys is None
                or progressive.dynamic_fine_keys is None
                or progressive.dynamic_fine_values is None
                or progressive.dynamic_fine_valid is None
                or (
                    not self.g_aligned_future_effect
                    and progressive.world_source_bias is None
                )
            ):
                raise RuntimeError(
                    "completed G3/W state has no progressive selector priors"
                )
            coarse_keys = progressive.canonical_slot_keys
            fine_keys = progressive.dynamic_fine_keys
            fine_values = progressive.dynamic_fine_values
            fine_valid = progressive.dynamic_fine_valid
            if tuple(progressive_coarse_bias.shape) != tuple(coarse_keys.shape[:-1]):
                raise ValueError("G3 coarse prior does not align with the address bank")
            if tuple(progressive_fine_bias.shape) != tuple(fine_keys.shape[:-1]):
                raise ValueError("G3 fine prior does not align with the address bank")
            anchors = int(self.config.future_anchors)
            grid = int(self.config.future_grid_size)
            slots = int(self.config.flow_jepa_address_slots)
            boundaries = tuple(
                int(value) for value in self.config.flow_jepa_action_offsets
            )
            # V115 makes P1 a current-fact reader.  W source priors and the
            # W-owned appearance query are successor hypotheses, so allowing
            # either to steer the only high-resolution read would make P1
            # future-conditioned and collapse the G -> P1 ownership boundary.
            # The FutureEffectField is still carried in the factual glimpse
            # bank below for P2, but it cannot change P1's address posterior.
            if not self.g_aligned_future_effect:
                world_source_bias = progressive.world_source_bias
                assert world_source_bias is not None
                if tuple(world_source_bias.shape) != (
                    batch,
                    anchors,
                    cameras,
                    grid,
                    grid,
                    slots,
                ):
                    raise ValueError(
                        "W source prior does not align with the G3 address basis"
                    )
                aligned_world_source_bias = world_source_bias[
                    :, : len(boundaries)
                ]
                progressive_world_source_bias = (
                    _align_milestone_tokens_to_horizon(
                        aligned_world_source_bias.permute(
                            0, 2, 3, 4, 5, 1
                        ).reshape(

                            batch * cameras * grid * grid * slots,
                            int(aligned_world_source_bias.shape[1]),
                            1,
                        ),
                        horizon,
                        boundaries=boundaries,
                    ).reshape(
                        batch,
                        cameras,
                        grid,
                        grid,
                        slots,
                        horizon,
                    ).permute(0, 5, 1, 2, 3, 4)
                )
                if self.structured_ownership:
                    for name, owner_source_bias in (
                        ("semantic", progressive.world_semantic_source_bias),
                        ("appearance", progressive.world_appearance_source_bias),
                        ("geometry", progressive.world_geometry_source_bias),
                    ):
                        if owner_source_bias is None:
                            raise RuntimeError(
                                "completed V111 state has no W "
                                f"{name} source sidecar"
                            )
                        aligned_owner_bias = owner_source_bias[
                            :, : len(boundaries)
                        ]
                        progressive_world_owner_source_bias[name] = (
                            _align_milestone_tokens_to_horizon(
                                aligned_owner_bias.permute(
                                    0, 2, 3, 4, 5, 1
                                ).reshape(
                                    batch * cameras * grid * grid * slots,
                                    int(aligned_owner_bias.shape[1]),
                                    1,
                                ),
                                horizon,
                                boundaries=boundaries,
                            ).reshape(
                                batch,
                                cameras,
                                grid,
                                grid,
                                slots,
                                horizon,
                            ).permute(0, 5, 1, 2, 3, 4)
                        )
                if self.pre_value_owner_routing:
                    appearance_fine_query = (
                        progressive.world_appearance_fine_query
                    )
                    if appearance_fine_query is None:
                        raise RuntimeError(
                            "completed V112 state has no W appearance fine query"
                        )
                    aligned_fine_query = appearance_fine_query[
                        :, : len(boundaries)
                    ]
                    progressive_world_appearance_fine_query = (
                        _align_milestone_tokens_to_horizon(
                            aligned_fine_query.permute(
                                0, 2, 3, 4, 5, 1, 6
                            ).reshape(
                                batch * cameras * grid * grid * slots,
                                int(aligned_fine_query.shape[1]),
                                self.lattice_route_dim,
                            ),
                            horizon,
                            boundaries=boundaries,
                        ).reshape(
                            batch,
                            cameras,
                            grid,
                            grid,
                            slots,
                            horizon,
                            self.lattice_route_dim,
                        ).permute(0, 5, 1, 2, 3, 4, 6)

                    )
            if self.coordinate_typed_raw_detail:
                current_typed_required = (
                    progressive.dynamic_semantic_keys,
                    progressive.dynamic_appearance_keys,
                    progressive.dynamic_geometry_keys,
                    progressive.dynamic_literal_rgb,
                    progressive.dynamic_fine_coordinates,
                    progressive.canonical_semantic_keys,
                    progressive.canonical_appearance_keys,
                    progressive.canonical_geometry_keys,
                )
                if not all(
                    torch.is_tensor(value) for value in current_typed_required
                ):
                    raise RuntimeError(
                        "completed G3 state has no typed current evidence"
                    )
                if (
                    self.g_aligned_future_effect
                    and not self.differential_intent_effect_mainline
                    and not self.grounded_intent_effect_mainline
                    and not self.object_intent_dynamics_mainline
                ):
                    effect_field = progressive.world_future_effect_field
                    if (
                        effect_field is None
                        or progressive.rectified_centers is None
                    ):
                        raise RuntimeError(
                            "V115 P2 requires the completed FutureEffectField"
                        )
                    effect_field.validate()
                    effect_scale = (
                        effect_field.transport_covariance[..., :2]
                        .float()
                        .mean(dim=-1, keepdim=True)
                        .clamp_min(1e-4)
                        .sqrt()
                    )
                    effect_centers = (
                        progressive.rectified_centers[:, None].float()
                        + effect_field.transport_mean.float()
                    ).clamp(-1.0, 1.0)
                    future_transport_anchors = torch.cat(
                        (
                            effect_centers,
                            effect_scale,
                            effect_field.visibility.float(),
                            effect_field.uncertainty.float(),
                        ),
                        dim=-1,
                    )[:, : len(boundaries)]
                elif (
                    self.differential_intent_effect_mainline
                    or self.grounded_intent_effect_mainline
                    or self.object_intent_dynamics_mainline
                ):
                    # The explicit 3-2-3 paths give W effects their only policy
                    # ingress at P2. P1 carries current G3 geometry as factual
                    # metadata and never requests, copies, or reads W-owned
                    # successor state.
                    current_centers = progressive.rectified_centers
                    current_support = progressive.rectified_support
                    if current_centers is None or current_support is None:
                        raise RuntimeError(
                            "differential P1 has no current G3 geometry"
                        )
                    current_support = current_support.float().clamp_min(1e-4)
                    if current_support.ndim == current_centers.ndim - 1:
                        current_support = current_support.unsqueeze(-1)
                    current_transport = torch.cat(
                        (
                            current_centers.float(),
                            current_support,
                            torch.ones_like(current_support),
                            current_support,
                        ),
                        dim=-1,
                    )

                    future_transport_anchors = current_transport[:, None].expand(
                        -1,
                        len(boundaries),
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                    )
                else:
                    legacy_future_required = (
                        progressive.world_future_centers,
                        progressive.world_future_scale,
                        progressive.world_future_visibility,
                        progressive.world_future_uncertainty,
                    )
                    if not all(
                        torch.is_tensor(value)
                        for value in legacy_future_required
                    ):
                        raise RuntimeError(
                            "completed V110 state has no typed future evidence"
                        )
                    future_transport_anchors = torch.cat(
                        legacy_future_required,
                        dim=-1,
                    )[:, : len(boundaries)]
                transport_width = int(future_transport_anchors.shape[-1])
                progressive_future_transport = _align_milestone_tokens_to_horizon(
                    future_transport_anchors.permute(
                        0, 2, 3, 4, 5, 1, 6
                    ).reshape(
                        batch * cameras * grid * grid * slots,
                        int(future_transport_anchors.shape[1]),
                        transport_width,
                    ),
                    horizon,
                    boundaries=boundaries,
                ).reshape(
                    batch,
                    cameras,
                    grid,
                    grid,
                    slots,
                    horizon,
                    transport_width,
                ).permute(0, 5, 1, 2, 3, 4, 6)
        intervention = self._address_eval_intervention
        collect_diagnostics = bool(
            collect_diagnostics or intervention is not None
        )
        if intervention is not None and self.training:
            raise RuntimeError("address-posterior intervention is evaluation-only")
        if intervention == "camera_swap":
            if int(coarse_keys.shape[1]) <= 1:
                raise RuntimeError("camera swap requires at least two camera charts")
            original_fine_values = fine_values
            coarse_keys = coarse_keys.roll(shifts=1, dims=1)
            fine_keys = fine_keys.roll(shifts=1, dims=1)
            fine_values = fine_values.roll(shifts=1, dims=1)
            fine_valid = fine_valid.roll(shifts=1, dims=1)
            if progressive_coarse_bias is not None:
                progressive_coarse_bias = progressive_coarse_bias.roll(
                    shifts=1, dims=1
                )
            if progressive_fine_bias is not None:
                progressive_fine_bias = progressive_fine_bias.roll(
                    shifts=1, dims=1
                )
            if progressive_world_source_bias is not None:
                progressive_world_source_bias = progressive_world_source_bias.roll(
                    shifts=1, dims=2
                )
            progressive_world_owner_source_bias = {
                name: value.roll(shifts=1, dims=2)
                for name, value in progressive_world_owner_source_bias.items()
            }
            if progressive_world_appearance_fine_query is not None:
                progressive_world_appearance_fine_query = (
                    progressive_world_appearance_fine_query.roll(

                        shifts=1,
                        dims=2,
                    )
                )
            self._address_eval_metrics["camera_bank_value_delta_norm"] = float(
                (fine_values - original_fine_values)
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
        if coarse_keys.ndim != 6 or fine_keys.ndim != 7 or fine_values.ndim != 7:
            raise ValueError("soft address lattice bank has invalid rank")
        grid = int(self.config.future_grid_size)
        slots = int(self.config.flow_jepa_address_slots)
        candidates = int(fine_keys.shape[-2])
        expected_coarse = (
            batch,
            cameras,
            grid,
            grid,
            slots,
            self.lattice_route_dim,
        )
        if tuple(coarse_keys.shape) != expected_coarse:
            raise ValueError(
                "soft address coarse keys must be "
                f"{expected_coarse}, got {tuple(coarse_keys.shape)}"
            )
        if tuple(fine_keys.shape) != (
            *expected_coarse[:-1],
            candidates,
            self.lattice_route_dim,
        ):
            raise ValueError("soft address fine keys do not align with coarse slots")
        if tuple(fine_values.shape) != (
            *expected_coarse[:-1],
            candidates,
            self.lattice_raw_dim,
        ):
            raise ValueError("soft address fine values have an invalid width")
        if tuple(fine_valid.shape) != tuple(fine_keys.shape[:-1]):
            raise ValueError("soft address valid mask does not align with candidates")
        if tuple(world_horizon_grid.shape) != (
            batch,
            horizon,
            cameras,
            grid,
            grid,
            self.hidden,
        ):
            raise ValueError(
                "soft address world chart must preserve "
                f"[B,T,C,G,G,H], got {tuple(world_horizon_grid.shape)}"
            )
        original_world_horizon_grid = world_horizon_grid
        original_progressive_world_source_bias = progressive_world_source_bias
        original_progressive_future_transport = progressive_future_transport
        if intervention == "world_query_zero":
            world_horizon_grid = torch.zeros_like(world_horizon_grid)
            if progressive_world_source_bias is not None:
                progressive_world_source_bias = torch.zeros_like(
                    progressive_world_source_bias
                )
            progressive_world_owner_source_bias = {
                name: torch.zeros_like(value)
                for name, value in progressive_world_owner_source_bias.items()
            }
            if progressive_world_appearance_fine_query is not None:
                progressive_world_appearance_fine_query = torch.zeros_like(
                    progressive_world_appearance_fine_query
                )
            if progressive_future_transport is not None:
                progressive_future_transport = torch.zeros_like(
                    progressive_future_transport
                )
        elif intervention == "world_query_spatial_shuffle":
            world_horizon_grid = world_horizon_grid.roll(
                shifts=(max(grid // 2, 1), max(grid // 3, 1)),

                dims=(3, 4),
            )
            if progressive_world_source_bias is not None:
                progressive_world_source_bias = progressive_world_source_bias.roll(
                    shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                    dims=(3, 4),
                )
            progressive_world_owner_source_bias = {
                name: value.roll(
                    shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                    dims=(3, 4),
                )
                for name, value in progressive_world_owner_source_bias.items()
            }
            if progressive_world_appearance_fine_query is not None:
                progressive_world_appearance_fine_query = (
                    progressive_world_appearance_fine_query.roll(
                        shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                        dims=(3, 4),
                    )
                )
            if progressive_future_transport is not None:
                progressive_future_transport = progressive_future_transport.roll(
                    shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                    dims=(3, 4),
                )
        if intervention is not None:
            world_query_input_delta = (
                world_horizon_grid - original_world_horizon_grid
            ).detach().float().norm(dim=-1).mean()
            world_source_prior_input_delta = (
                (
                    progressive_world_source_bias
                    - original_progressive_world_source_bias
                )
                .detach()
                .float()
                .square()
                .sum(dim=(-3, -2, -1))
                .sqrt()
                .mean()
                if progressive_world_source_bias is not None
                and original_progressive_world_source_bias is not None
                else world_query_input_delta.new_zeros(())
            )
        else:
            world_query_input_delta = trajectory.new_zeros(
                (), dtype=torch.float32
            )
            world_source_prior_input_delta = trajectory.new_zeros(
                (), dtype=torch.float32
            )

        glimpses = self.heads if self.policy_multi_glimpse_address else 1
        p2_query = self.lattice_query_proj(
            self.lattice_query_norm(query_input)
        ).reshape(
            batch,
            horizon * basis,
            cameras,
            glimpses,
            self.lattice_route_dim,
        ).permute(0, 1, 3, 2, 4)
        shared_basis_entropy = trajectory.new_zeros((), dtype=torch.float32)
        address_basis = basis
        if self.utility_precision_mainline:
            if clean_basis_tokens is None or factual_condition is None:
                raise ValueError(
                    "utility/precision P1 requires clean basis tokens and "
                    "a factual condition"
                )
            query, shared_basis_entropy = self._shared_factual_p1_query(
                clean_basis_tokens=clean_basis_tokens,
                factual_condition=factual_condition,
                world_horizon=world_horizon_grid.mean(dim=(3, 4)),
            )
            address_basis = 1
        else:
            query = p2_query
        world_route = self.lattice_world_key_proj(

            self.lattice_world_norm(world_horizon_grid)
        )
        if not self.utility_precision_mainline:
            world_route = world_route[:, :, None].expand(
                -1, -1, basis, -1, -1, -1, -1
            ).reshape(
                batch,
                horizon * basis,
                cameras,
                grid,
                grid,
                self.lattice_route_dim,
            )
        coarse_key = (
            None
            if self.coordinate_typed_raw_detail
            else self.lattice_key_norm(coarse_keys)
        )
        fine_key = (
            None
            if self.coordinate_typed_raw_detail
            else self.lattice_key_norm(fine_keys)
        )
        typed_fine_keys: dict[str, Tensor] = {}
        typed_coarse_keys: dict[str, Tensor] = {}
        typed_literal_rgb: Tensor | None = None
        typed_coordinates: Tensor | None = None
        if self.coordinate_typed_raw_detail:
            if (
                progressive is None
                or self.typed_fine_query is None
                or self.typed_coarse_query is None
                or self.typed_local_refiners is None
                or self.typed_micro_basis is None
            ):
                raise RuntimeError("typed P1/P2 reader is incomplete")
            for name, fine_value, coarse_value in (
                (
                    "semantic",
                    progressive.dynamic_semantic_keys,
                    progressive.canonical_semantic_keys,
                ),
                (
                    "appearance",
                    progressive.dynamic_appearance_keys,
                    progressive.canonical_appearance_keys,
                ),
                (
                    "geometry",
                    progressive.dynamic_geometry_keys,
                    progressive.canonical_geometry_keys,
                ),
            ):
                assert fine_value is not None and coarse_value is not None
                typed_fine_keys[name] = self.lattice_key_norm(fine_value)
                typed_coarse_keys[name] = self.lattice_key_norm(coarse_value)
            typed_literal_rgb = progressive.dynamic_literal_rgb
            typed_coordinates = progressive.dynamic_fine_coordinates
            if typed_literal_rgb is None or typed_coordinates is None:
                raise RuntimeError("typed P1 has no literal RGB/current coordinates")
            if progressive_future_transport is None:
                raise RuntimeError("typed P2 has no future transport distribution")
            if intervention == "camera_swap":
                typed_fine_keys = {
                    name: value.roll(shifts=1, dims=1)
                    for name, value in typed_fine_keys.items()
                }
                typed_coarse_keys = {
                    name: value.roll(shifts=1, dims=1)
                    for name, value in typed_coarse_keys.items()
                }
                typed_literal_rgb = typed_literal_rgb.roll(shifts=1, dims=1)
                typed_coordinates = typed_coordinates.roll(shifts=1, dims=1)
                progressive_future_transport = progressive_future_transport.roll(
                    shifts=1, dims=2
                )
            elif intervention in {
                "semantic_owner_zero",
                "appearance_owner_zero",
                "geometry_owner_zero",

                "semantic_owner_shuffle",
                "appearance_owner_shuffle",
                "geometry_owner_shuffle",
            }:
                owner_name, operation = intervention.rsplit("_owner_", 1)
                if owner_name not in typed_fine_keys:
                    raise RuntimeError(f"unknown typed P owner {owner_name!r}")
                if operation == "zero":
                    typed_fine_keys[owner_name] = torch.zeros_like(
                        typed_fine_keys[owner_name]
                    )
                    typed_coarse_keys[owner_name] = torch.zeros_like(
                        typed_coarse_keys[owner_name]
                    )
                    if owner_name in progressive_world_owner_source_bias:
                        progressive_world_owner_source_bias[owner_name] = (
                            torch.zeros_like(
                                progressive_world_owner_source_bias[owner_name]
                            )
                        )
                    if (
                        owner_name == "appearance"
                        and progressive_world_appearance_fine_query is not None
                    ):
                        progressive_world_appearance_fine_query = torch.zeros_like(
                            progressive_world_appearance_fine_query
                        )
                else:
                    typed_fine_keys[owner_name] = typed_fine_keys[owner_name].roll(
                        shifts=1, dims=0 if batch > 1 else 2
                    )
                    typed_coarse_keys[owner_name] = typed_coarse_keys[owner_name].roll(
                        shifts=1, dims=0 if batch > 1 else 2
                    )
                    if owner_name in progressive_world_owner_source_bias:
                        progressive_world_owner_source_bias[owner_name] = (
                            progressive_world_owner_source_bias[owner_name].roll(
                                shifts=1, dims=0 if batch > 1 else 3
                            )
                        )
                    if (
                        owner_name == "appearance"
                        and progressive_world_appearance_fine_query is not None
                    ):
                        progressive_world_appearance_fine_query = (
                            progressive_world_appearance_fine_query.roll(
                                shifts=1,
                                dims=0 if batch > 1 else 3,
                            )
                        )
            elif intervention == "future_transport_neutral":
                current_centers = progressive.rectified_centers
                current_support = progressive.rectified_support
                if current_centers is None or current_support is None:
                    raise RuntimeError(
                        "future transport neutralization has no current anchor geometry"
                    )
                neutral = progressive_future_transport.clone()
                neutral[..., :2] = current_centers[:, None].to(
                    dtype=neutral.dtype
                )
                neutral[..., 2] = 1.0
                neutral[..., 3] = 0.5
                neutral[..., 4] = current_support[:, None].to(
                    dtype=neutral.dtype
                ).clamp_min(0.05)
                progressive_future_transport = neutral
            elif intervention == "future_transport_spatial_shuffle":
                progressive_future_transport = progressive_future_transport.roll(
                    shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                    dims=(3, 4),
                )
        future_transport_input_delta = (
            (
                progressive_future_transport
                - original_progressive_future_transport
            )
            .detach()
            .float()
            .norm(dim=-1)

            .mean()
            if intervention is not None
            and progressive_future_transport is not None
            and original_progressive_future_transport is not None
            else world_query_input_delta.new_zeros(())
        )
        valid_any = fine_valid.any(dim=-1)
        state_count = cameras * grid * grid * slots
        chunk = int(self.config.flow_jepa_address_query_chunk)
        output_rows: list[Tensor] = []
        route_entropy_rows: list[Tensor] = []
        route_max_rows: list[Tensor] = []
        fine_entropy_rows: list[Tensor] = []
        fine_max_rows: list[Tensor] = []
        camera_mass_rows: list[Tensor] = []
        slot_mass_rows: list[Tensor] = []
        world_logit_std_rows: list[Tensor] = []
        posterior_signature_rows: list[Tensor] = []
        fine_signature_rows: list[Tensor] = []
        typed_metric_rows: dict[str, list[Tensor]] = {}
        evidence_scale = self.lattice_fine_evidence_scale.float().tanh()
        progressive_world_route_prior: Tensor | None = None
        progressive_world_owner_route_priors: dict[str, Tensor] = {}
        active_world_source_bias = progressive_world_source_bias
        if self.structured_ownership and progressive_world_owner_source_bias:
            active_world_source_bias = (
                progressive_world_owner_source_bias["semantic"]
                + progressive_world_owner_source_bias["geometry"]
            ) / math.sqrt(2.0)
        if active_world_source_bias is not None:
            progressive_world_route_prior = active_world_source_bias[
                :, :, None, None
            ].expand(
                -1,
                -1,
                address_basis,
                glimpses,
                -1,
                -1,
                -1,
                -1,
            ).reshape(
                batch,
                horizon * address_basis,
                glimpses,
                cameras,
                grid,
                grid,
                slots,
            )
        if self.structured_ownership:
            for name, owner_bias in progressive_world_owner_source_bias.items():
                progressive_world_owner_route_priors[name] = owner_bias[
                    :, :, None, None
                ].expand(
                    -1,
                    -1,
                    address_basis,
                    glimpses,
                    -1,
                    -1,
                    -1,
                    -1,
                ).reshape(
                    batch,
                    horizon * address_basis,
                    glimpses,
                    cameras,
                    grid,
                    grid,
                    slots,
                )
        posterior_basis: Tensor | None = None
        fine_basis: Tensor | None = None
        if intervention is not None:
            coordinate_axis = torch.linspace(
                -1.0,
                1.0,
                grid,
                device=query.device,

                dtype=torch.float32,
            )
            coordinate_y, coordinate_x = torch.meshgrid(
                coordinate_axis,
                coordinate_axis,
                indexing="ij",
            )
            camera_axis = torch.linspace(
                -1.0,
                1.0,
                cameras,
                device=query.device,
                dtype=torch.float32,
            )
            slot_axis = torch.linspace(
                -1.0,
                1.0,
                slots,
                device=query.device,
                dtype=torch.float32,
            )
            posterior_basis = torch.stack(
                torch.broadcast_tensors(
                    camera_axis[:, None, None, None],
                    coordinate_x[None, :, :, None],
                    coordinate_y[None, :, :, None],
                    slot_axis[None, None, None, :],
                ),
                dim=-1,
            )
            fine_side = int(round(math.sqrt(float(candidates))))
            if fine_side * fine_side == candidates:
                fine_axis = torch.linspace(
                    -1.0,
                    1.0,
                    fine_side,
                    device=query.device,
                    dtype=torch.float32,
                )
                fine_y, fine_x = torch.meshgrid(
                    fine_axis,
                    fine_axis,
                    indexing="ij",
                )
                fine_basis = torch.stack(
                    (fine_x.reshape(-1), fine_y.reshape(-1)),
                    dim=-1,
                )
            else:
                fine_basis = torch.stack(
                    (
                        torch.linspace(
                            -1.0,
                            1.0,
                            candidates,
                            device=query.device,
                            dtype=torch.float32,
                        ),
                        torch.zeros(
                            candidates,
                            device=query.device,
                            dtype=torch.float32,
                        ),
                    ),
                    dim=-1,
                )
        typed_future_rows: Tensor | None = None
        if progressive_future_transport is not None:
            typed_future_rows = progressive_future_transport[:, :, None, None].expand(
                -1,
                -1,
                address_basis,
                glimpses,
                -1,
                -1,
                -1,
                -1,
                -1,
            ).reshape(
                batch,

                horizon * address_basis,
                glimpses,
                cameras,
                grid,
                grid,
                slots,
                int(progressive_future_transport.shape[-1]),
            )
        appearance_fine_query_rows: Tensor | None = None
        if progressive_world_appearance_fine_query is not None:
            appearance_fine_query_rows = (
                progressive_world_appearance_fine_query[
                    :, :, None, None
                ].expand(
                    -1,
                    -1,
                    address_basis,
                    glimpses,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ).reshape(
                    batch,
                    horizon * address_basis,
                    glimpses,
                    cameras,
                    grid,
                    grid,
                    slots,
                    self.lattice_route_dim,
                )
            )
        if self.utility_precision_mainline:
            chunk = min(
                int(query.shape[1]),
                max(self.address_query_batch_budget // max(batch, 1), 1),
            )
        activation_checkpoint_active = bool(
            self.raw_activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and (
                not self.utility_precision_mainline
                or batch >= self.checkpoint_min_batch
            )
        )
        p2_query_structured = (
            p2_query.reshape(
                batch,
                horizon,
                basis,
                glimpses,
                cameras,
                self.lattice_route_dim,
            )
            if self.utility_precision_mainline
            else None
        )
        if p2_query_structured is not None and intervention is not None:
            for basis_index in range(basis):
                if intervention == f"p2_basis{basis_index}_zero":
                    p2_query_structured = p2_query_structured.clone()
                    p2_query_structured[:, :, basis_index] = 0
                    break
                if (
                    intervention
                    == f"p2_basis{basis_index}_horizon_shuffle"
                ):
                    p2_query_structured = p2_query_structured.clone()
                    p2_query_structured[:, :, basis_index] = (
                        p2_query_structured[:, :, basis_index].roll(
                            shifts=1, dims=1
                        )
                    )
                    break
        for start in range(0, int(query.shape[1]), chunk):
            stop = min(start + chunk, int(query.shape[1]))
            query_row = query[:, start:stop]

            typed_fine_query_rows: dict[str, Tensor] = {}
            typed_coarse_query_rows: dict[str, Tensor] = {}
            functional_appearance_query_row: Tensor | None = None
            owner_fine_weights: dict[str, Tensor] = {}
            owner_route_weights: dict[str, Tensor] = {}
            if typed_fine_keys:
                if self.typed_fine_query is None or self.typed_coarse_query is None:
                    raise RuntimeError("typed P query projections are missing")
                # Projection layers belong to the active autocast domain.  The
                # explicit FP32 block below starts only after learned modules,
                # so Float32 parameters never receive an uncast BF16 tensor.
                typed_fine_query_rows = {
                    name: self.typed_fine_query[name](query_row)
                    for name in typed_fine_keys
                    if not (
                        self.utility_precision_mainline
                        and name == "semantic"
                    )
                }
                typed_coarse_query_rows = {
                    name: self.typed_coarse_query[name](query_row)
                    for name in typed_coarse_keys
                }
                if (
                    self.functional_mainline_routing
                    and not self.g_aligned_future_effect
                ):
                    if (
                        self.appearance_world_owner_query is None
                        or appearance_fine_query_rows is None
                    ):
                        raise RuntimeError(
                            "functional P1 has no W-owned appearance gateway"
                        )
                    functional_appearance_query_row = (
                        self.appearance_world_owner_query(
                            appearance_fine_query_rows[:, start:stop].to(
                                dtype=query_row.dtype
                            )
                        )
                    )
            with torch.autocast(device_type=query.device.type, enabled=False):
                query_f = query_row.float()
                if typed_fine_keys:
                    typed_fine_logit_rms: dict[str, Tensor] = {}
                    typed_fine_logits: dict[str, Tensor] = {}
                    for name, typed_key in typed_fine_keys.items():
                        if (
                            self.utility_precision_mainline
                            and name == "semantic"
                        ):
                            # Semantic evidence owns coarse source/slot
                            # selection.  V114 fine offsets are verified by
                            # appearance, geometry and transport.  Computing a
                            # complete semantic candidate tensor here was dead
                            # work: it was never consumed by the joint
                            # posterior, yet retained one of P1's largest
                            # activation families.
                            continue
                        if (
                            self.functional_mainline_routing
                            and not self.g_aligned_future_effect
                            and name == "appearance"
                        ):
                            continue
                        typed_logit = torch.einsum(
                            "bqgcr,bcijmkr->bqgcijmk",
                            typed_fine_query_rows[name].float(),
                            typed_key.float(),
                        ) * (float(self.lattice_route_dim) ** -0.5)
                        if collect_diagnostics:
                            typed_fine_logit_rms[name] = (
                                typed_logit.detach().square().mean().sqrt()
                            )
                        typed_fine_logits[name] = typed_logit
                    if not typed_fine_logits:
                        raise RuntimeError("typed P1 produced no fine logits")
                    appearance_candidate_logit: Tensor | None = None
                    if (
                        self.pre_value_owner_routing

                        and not self.g_aligned_future_effect
                    ):
                        if appearance_fine_query_rows is None:
                            raise RuntimeError(
                                "pre-value P1 has no source-aligned W "
                                "appearance verifier query"
                            )
                        if self.functional_mainline_routing:
                            if functional_appearance_query_row is None:
                                raise RuntimeError(
                                    "functional P1 appearance gateway was not built"
                                )
                            policy_appearance = typed_fine_query_rows[
                                "appearance"
                            ].float()[:, :, :, :, None, None, None, :]
                            world_appearance = (
                                functional_appearance_query_row.float()
                            )
                            original_world_appearance = world_appearance
                            if intervention == "p1_appearance_gateway_zero":
                                world_appearance = torch.zeros_like(
                                    world_appearance
                                )
                            elif (
                                intervention
                                == "p1_appearance_gateway_spatial_shuffle"
                            ):
                                world_appearance = world_appearance.roll(
                                    shifts=(
                                        max(grid // 2, 1),
                                        max(grid // 3, 1),
                                    ),
                                    dims=(4, 5),
                                )
                            # One mandatory W-owned verifier query.  The policy
                            # query can modulate it but cannot independently
                            # score candidates when the W appearance state is
                            # zero.
                            composed_appearance = world_appearance + (
                                policy_appearance
                                * torch.tanh(world_appearance)
                            )
                            composed_appearance, _ = smooth_rms_contract(
                                composed_appearance, 0.75
                            )
                            if intervention in {
                                "p1_appearance_gateway_zero",
                                "p1_appearance_gateway_spatial_shuffle",
                            }:
                                baseline_composed_appearance = (
                                    original_world_appearance
                                    + policy_appearance
                                    * torch.tanh(original_world_appearance)
                                )
                                baseline_composed_appearance, _ = (
                                    smooth_rms_contract(
                                        baseline_composed_appearance,
                                        0.75,
                                    )
                                )
                                typed_metric_rows.setdefault(
                                    "flow_jepa_typed_p1_appearance_gateway_"
                                    "intervention_delta_norm",
                                    [],
                                ).append(
                                    (
                                        composed_appearance
                                        - baseline_composed_appearance
                                    )
                                    .detach()
                                    .float()
                                    .norm(dim=-1)
                                    .mean()
                                )
                            appearance_candidate_logit = torch.einsum(
                                "bqgcijmr,bcijmkr->bqgcijmk",
                                composed_appearance,
                                typed_fine_keys["appearance"].float(),
                            ) * (float(self.lattice_route_dim) ** -0.5)
                            typed_fine_logits[

                                "appearance"
                            ] = appearance_candidate_logit
                            if collect_diagnostics:
                                typed_fine_logit_rms[
                                    "appearance"
                                ] = (
                                    appearance_candidate_logit.detach()
                                    .square()
                                    .mean()
                                    .sqrt()
                                )
                                typed_metric_rows.setdefault(
                                    "flow_jepa_typed_p1_appearance_gateway_query_rms",
                                    [],
                                ).append(
                                    composed_appearance.detach()
                                    .square()
                                    .mean()
                                    .sqrt()
                                )
                        else:
                            appearance_candidate_logit = torch.einsum(
                                "bqgcijmr,bcijmkr->bqgcijmk",
                                appearance_fine_query_rows[
                                    :, start:stop
                                ].float(),
                                typed_fine_keys["appearance"].float(),
                            ) * (float(self.lattice_route_dim) ** -0.5)
                    if self.structured_ownership:
                        # P1 factorizes coarse source ownership from precise
                        # offset verification.  Appearance and geometry own
                        # the fine offset; semantic evidence is retained for
                        # its context read but cannot collapse local detail.
                        appearance_logit = typed_fine_logits["appearance"]
                        if (
                            appearance_candidate_logit is not None
                            and not self.functional_mainline_routing
                        ):
                            appearance_logit = (
                                appearance_logit
                                + appearance_candidate_logit
                            ) / math.sqrt(2.0)
                        if self.shared_factual_glimpse_bank:
                            owner_mix = self._shared_factual_owner_weights()
                            fine_mix = owner_mix[:, 1:3]
                            fine_mix = fine_mix / fine_mix.sum(
                                dim=-1, keepdim=True
                            ).clamp_min(1e-8)
                            fine_stack = torch.stack(
                                (
                                    appearance_logit,
                                    typed_fine_logits["geometry"],
                                ),
                                dim=-1,
                            )
                            fine_logits = (
                                fine_stack
                                * fine_mix.reshape(
                                    1,
                                    1,
                                    self.heads,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    2,
                                )
                            ).sum(dim=-1) * math.sqrt(2.0)
                        else:
                            fine_logits = (
                                appearance_logit
                                + typed_fine_logits["geometry"]
                            ) / math.sqrt(2.0)
                    else:
                        fine_logits = sum(typed_fine_logits.values()) / math.sqrt(3.0)
                    if typed_future_rows is None or typed_coordinates is None:
                        raise RuntimeError(
                            "typed P1 fine routing has no future transport geometry"
                        )

                    transport = typed_future_rows[:, start:stop].float()
                    transport_center = transport[..., :2].unsqueeze(-2)
                    # [B,Q,G,C,i,j,slot,K,2].  Current observed coordinates
                    # remain the value anchors; the W prediction only supplies
                    # a bounded soft likelihood that they remain relevant at
                    # this horizon.
                    current_coordinate = typed_coordinates.float()[
                        :, None, None
                    ]
                    transport_scale = transport[..., 2:3].unsqueeze(-2)
                    transport_visibility = transport[..., 3:4].unsqueeze(-2)
                    transport_uncertainty = transport[..., 4:5].unsqueeze(-2)
                    transport_width = (
                        0.05 + transport_scale * transport_uncertainty
                    ).clamp(0.05, 1.0)
                    transport_distance = (
                        (current_coordinate - transport_center)
                        / transport_width
                    ).square().sum(dim=-1)
                    transport_fine_logit = 0.5 * (
                        (-0.5 * transport_distance).clamp_min(-4.0)
                        + 0.25
                        * (2.0 * transport_visibility[..., 0] - 1.0)
                    )
                    if self.g_aligned_future_effect:
                        # The V115 successor field is a P2 operand, not a P1
                        # address prior.  Retain the selected transport context
                        # in SharedFactualGlimpseBank, while making the factual
                        # address posterior exactly independent of W.
                        transport_fine_logit = fine_logits.new_zeros(())
                    else:
                        fine_logits = fine_logits + transport_fine_logit
                    appearance_pre_value_prior: Tensor | None = None
                    if (
                        self.pre_value_owner_routing
                        and not self.g_aligned_future_effect
                    ):
                        if "appearance" not in progressive_world_owner_route_priors:
                            raise RuntimeError(
                                "pre-value P1 has no W appearance source prior"
                            )
                        # This is a fine-factor term in the single joint
                        # source/slot/candidate posterior.  It is broadcast
                        # across local candidates, so it changes the joint
                        # source/slot mass through ``fine_evidence`` without
                        # pretending that a slot posterior identifies one
                        # exact raw pixel.
                        appearance_pre_value_prior = (
                            progressive_world_owner_route_priors["appearance"][
                                :, start:stop
                            ].float()
                        )
                        if not self.functional_mainline_routing:
                            fine_logits = (
                                fine_logits
                                + appearance_pre_value_prior[..., None]
                            )
                    if (
                        self.structured_ownership
                        and not self.utility_precision_mainline
                    ):
                        owner_fine_logits = {
                            "semantic": typed_fine_logits["semantic"],
                            "appearance": (
                                typed_fine_logits["appearance"]
                                + (
                                    appearance_candidate_logit
                                    if (
                                        appearance_candidate_logit is not None
                                        and not self.functional_mainline_routing
                                    )
                                    else 0.0
                                )
                                + transport_fine_logit
                                + (
                                    appearance_pre_value_prior[..., None]
                                    if (
                                        appearance_pre_value_prior is not None
                                        and not self.functional_mainline_routing
                                    )

                                    else 0.0
                                )
                            ),
                            "geometry": (
                                typed_fine_logits["geometry"]
                                + transport_fine_logit
                            ),
                        }
                    else:
                        owner_fine_logits = {}
                    if self.utility_precision_mainline:
                        # ``fine_logits`` is now the sole joint posterior
                        # operand.  Release the appearance/geometry component
                        # tensor references before softmax/backward; addition
                        # needs no saved value tensor for its gradient.
                        typed_fine_logits.clear()
                        del appearance_logit
                    if collect_diagnostics:
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_future_transport_logit_rms", []
                        ).append(
                            transport_fine_logit.detach().square().mean().sqrt()
                        )
                    if (
                        collect_diagnostics
                        and appearance_pre_value_prior is not None
                    ):
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_appearance_pre_value_prior_rms",
                            [],
                        ).append(
                            appearance_pre_value_prior.detach()
                            .square()
                            .mean()
                            .sqrt()
                        )
                    if (
                        collect_diagnostics
                        and appearance_candidate_logit is not None
                    ):
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_world_appearance_candidate_logit_rms",
                            [],
                        ).append(
                            appearance_candidate_logit.detach()
                            .square()
                            .mean()
                            .sqrt()
                        )
                else:
                    typed_fine_logit_rms = {}
                    typed_fine_logits = {}
                    owner_fine_logits = {}
                    assert fine_key is not None
                    fine_logits = torch.einsum(
                        "bqgcr,bcijmkr->bqgcijmk",
                        query_f,
                        fine_key.float(),
                    ) * (float(self.lattice_route_dim) ** -0.5)
                if progressive_fine_bias is not None:
                    fine_logits = (
                        fine_logits
                        + progressive_fine_bias[:, None, None].float()
                    )
                candidate_mask = fine_valid[:, None, None]
                fine_logits = fine_logits.masked_fill(
                    ~candidate_mask, torch.finfo(fine_logits.dtype).min
                )
                safe_fine_logits = torch.where(
                    valid_any[:, None, None, :, :, :, :, None],
                    fine_logits,
                    torch.zeros_like(fine_logits),
                )
                fine_weights = torch.softmax(safe_fine_logits, dim=-1)
                fine_weights = (
                    fine_weights * candidate_mask.float()
                )
                fine_weights = fine_weights / fine_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)

                if (
                    self.structured_ownership
                    and not self.utility_precision_mainline
                    and owner_fine_logits
                ):
                    for name, owner_logits in owner_fine_logits.items():
                        owner_logits = owner_logits.masked_fill(
                            ~candidate_mask,
                            torch.finfo(owner_logits.dtype).min,
                        )
                        safe_owner_logits = torch.where(
                            valid_any[:, None, None, :, :, :, :, None],
                            owner_logits,
                            torch.zeros_like(owner_logits),
                        )
                        owner_weights = torch.softmax(
                            safe_owner_logits, dim=-1
                        ) * candidate_mask.float()
                        owner_fine_weights[name] = owner_weights / owner_weights.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-8)
                baseline_fine_weights = fine_weights
                if intervention == "fine_offset_zero":
                    center = candidates // 2
                    center_weights = torch.zeros_like(fine_weights)
                    center_weights[..., center] = 1.0
                    center_valid = candidate_mask[..., center : center + 1]
                    fine_weights = torch.where(
                        center_valid,
                        center_weights,
                        baseline_fine_weights,
                    )
                local_values = (
                    None
                    if typed_fine_keys
                    else torch.einsum(
                        "bqgcijmk,bcijmkr->bqgcijmr",
                        fine_weights,
                        fine_values.float(),
                    )
                )
                valid_count = candidate_mask.float().sum(dim=-1).clamp_min(1.0)
                fine_evidence = torch.logsumexp(
                    safe_fine_logits, dim=-1
                ) - valid_count.log()
                fine_evidence = torch.where(
                    valid_any[:, None, None],
                    fine_evidence,
                    fine_evidence.new_full((), -1e4),
                )
                if typed_coarse_keys:
                    typed_route_logit_rms: dict[str, Tensor] = {}
                    typed_route_logits: dict[str, Tensor] = {}
                    for name, typed_key in typed_coarse_keys.items():
                        typed_logit = torch.einsum(
                            "bqgcr,bcijmr->bqgcijm",
                            typed_coarse_query_rows[name].float(),
                            typed_key.float(),
                        ) * (float(self.lattice_route_dim) ** -0.5)
                        if collect_diagnostics:
                            typed_route_logit_rms[name] = (
                                typed_logit.detach().square().mean().sqrt()
                            )
                        typed_route_logits[name] = typed_logit
                    if not typed_route_logits:
                        raise RuntimeError("typed P1 produced no coarse logits")
                    if self.structured_ownership:
                        # Semantic selects the action-relevant source while
                        # geometry constrains its spatial validity. Appearance
                        # remains a local verifier and is marginalized through
                        # fine evidence below instead of voting twice.
                        if self.shared_factual_glimpse_bank:
                            owner_mix = self._shared_factual_owner_weights()
                            route_stack = torch.stack(
                                tuple(
                                    typed_route_logits[name]
                                    for name in (
                                        "semantic",
                                        "appearance",
                                        "geometry",

                                    )
                                ),
                                dim=-1,
                            )
                            route_logits = (
                                route_stack
                                * owner_mix.reshape(
                                    1, 1, self.heads, 1, 1, 1, 1, 3
                                )
                            ).sum(dim=-1) * math.sqrt(3.0)
                        else:
                            route_logits = (
                                typed_route_logits["semantic"]
                                + typed_route_logits["geometry"]
                            ) / math.sqrt(2.0)
                    else:
                        route_logits = sum(typed_route_logits.values()) / math.sqrt(3.0)
                else:
                    typed_route_logit_rms = {}
                    typed_route_logits = {}
                    assert coarse_key is not None
                    route_logits = torch.einsum(
                        "bqgcr,bcijmr->bqgcijm",
                        query_f,
                        coarse_key.float(),
                    ) * (float(self.lattice_route_dim) ** -0.5)
                if progressive_coarse_bias is not None:
                    route_logits = (
                        route_logits
                        + progressive_coarse_bias[:, None, None].float()
                    )
                if progressive_world_route_prior is not None:
                    route_logits = route_logits + progressive_world_route_prior[
                        :, start:stop
                    ].float()
                # The chart is the protected completed-G3 snapshot under V115
                # (legacy versions retain their W-organized chart).  Add its
                # current-fact compatibility once per camera/xy cell and let
                # the existing selector choose slot and sub-cell offset.
                world_logits = torch.einsum(
                    "bqgcr,bqcijr->bqgcij",
                    query_f,
                    world_route[:, start:stop].float(),
                ) * (float(self.lattice_route_dim) ** -0.5)
                route_logits = route_logits + world_logits[..., None]
                if (
                    self.functional_mainline_routing
                    and appearance_pre_value_prior is not None
                ):
                    # Source ownership is a coarse W prior.  Keep it outside
                    # the trainable local-evidence scale so the appearance
                    # owner cannot be silently attenuated with P1 detail.
                    route_logits = route_logits + appearance_pre_value_prior
                route_logits = route_logits + evidence_scale * fine_evidence
                route_logits = route_logits.masked_fill(
                    ~valid_any[:, None, None], torch.finfo(route_logits.dtype).min
                )
                route_logits_flat = route_logits.reshape(
                    batch, stop - start, glimpses, state_count
                )
                route_weights = torch.softmax(route_logits_flat, dim=-1).reshape(
                    batch,
                    stop - start,
                    glimpses,
                    cameras,
                    grid,
                    grid,
                    slots,
                )
                if (
                    self.structured_ownership
                    and not self.utility_precision_mainline
                    and typed_route_logits
                ):
                    for name, owner_logits in typed_route_logits.items():
                        if progressive_coarse_bias is not None:
                            owner_logits = (
                                owner_logits
                                + progressive_coarse_bias[:, None, None].float()
                            )

                        if name in progressive_world_owner_route_priors:
                            owner_logits = owner_logits + (
                                progressive_world_owner_route_priors[name][
                                    :, start:stop
                                ].float()
                            )
                        owner_logits = (
                            owner_logits
                            + world_logits[..., None]
                            + evidence_scale * fine_evidence
                        )
                        owner_logits = owner_logits.masked_fill(
                            ~valid_any[:, None, None],
                            torch.finfo(owner_logits.dtype).min,
                        )
                        owner_route_weights[name] = torch.softmax(
                            owner_logits.reshape(
                                batch, stop - start, glimpses, state_count
                            ),
                            dim=-1,
                        ).reshape(
                            batch,
                            stop - start,
                            glimpses,
                            cameras,
                            grid,
                            grid,
                            slots,
                        )
                baseline_route_weights = route_weights
                if intervention == "address_posterior_uniform":
                    valid_states = valid_any[:, None, None].float()
                    uniform_route_weights = valid_states / valid_states.sum(
                        dim=(3, 4, 5, 6), keepdim=True
                    ).clamp_min(1.0)
                    # ``valid_states`` intentionally has singleton query and
                    # glimpse axes.  Ordinary posterior reductions can
                    # broadcast those axes, but the typed microgrid contract
                    # requires the actual per-query posterior layout to match
                    # ``fine_weights`` exactly.  Materialize only the logical
                    # expanded view; no probability values are duplicated.
                    route_weights = uniform_route_weights.expand_as(
                        baseline_route_weights
                    )
                elif intervention == "camera_posterior_uniform":
                    camera_logits = route_logits.reshape(
                        batch,
                        stop - start,
                        glimpses,
                        cameras,
                        grid * grid * slots,
                    )
                    camera_valid = valid_any.reshape(
                        batch,
                        cameras,
                        grid * grid * slots,
                    )[:, None, None]
                    safe_camera_logits = camera_logits.masked_fill(
                        ~camera_valid,
                        torch.finfo(camera_logits.dtype).min,
                    )
                    within_camera = torch.softmax(
                        safe_camera_logits,
                        dim=-1,
                    )
                    valid_camera = camera_valid.any(dim=-1).float()
                    equal_camera = valid_camera / valid_camera.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                    route_weights = (
                        within_camera * equal_camera[..., None]
                    ).reshape(
                        batch,
                        stop - start,
                        glimpses,
                        cameras,
                        grid,
                        grid,
                        slots,
                    )

                if intervention in {
                    "address_posterior_uniform",
                    "camera_posterior_uniform",
                } and owner_route_weights:
                    owner_route_weights = {
                        name: route_weights for name in owner_route_weights
                    }
                if intervention == "fine_offset_zero" and owner_fine_weights:
                    owner_fine_weights = {
                        name: fine_weights for name in owner_fine_weights
                    }
                if typed_fine_keys:
                    assert typed_literal_rgb is not None
                    assert typed_coordinates is not None
                    assert self.typed_micro_basis is not None
                    assert typed_future_rows is not None
                    micro_inputs = (
                        route_weights,
                        fine_weights,
                        self.typed_micro_basis,
                        typed_literal_rgb,
                        fine_values,
                        typed_coordinates,
                    )
                    if activation_checkpoint_active:
                        (
                            typed_rgb_micro,
                            typed_detail_micro,
                            typed_coordinate_micro,
                        ) = checkpoint(
                            self._configured_typed_microgrid_expectation,
                            *micro_inputs,
                            use_reentrant=False,
                        )
                    else:
                        (
                            typed_rgb_micro,
                            typed_detail_micro,
                            typed_coordinate_micro,
                        ) = self._configured_typed_microgrid_expectation(
                            *micro_inputs
                        )
                    typed_contexts = {
                        name: torch.einsum(
                            "bqgcijm,bqgcijmk,bcijmkr->bqgr",
                            (
                                route_weights
                                if self.utility_precision_mainline
                                else owner_route_weights.get(name, route_weights)
                            ),
                            (
                                fine_weights
                                if self.utility_precision_mainline
                                else owner_fine_weights.get(name, fine_weights)
                            ),
                            typed_key.float(),
                        )
                        for name, typed_key in typed_fine_keys.items()
                    }
                    typed_future_context = torch.einsum(
                        "bqgcijm,bqgcijmv->bqgv",
                        route_weights,
                        typed_future_rows[:, start:stop].float(),
                    )
                    shared_glimpse_bank: (
                        SharedFactualGlimpseBank | None
                    ) = None
                    if self.shared_factual_glimpse_bank:
                        shared_glimpse_bank = SharedFactualGlimpseBank(
                            literal_rgb=typed_rgb_micro,
                            learned_detail=typed_detail_micro,
                            coordinates=typed_coordinate_micro,
                            semantic=typed_contexts["semantic"],
                            appearance=typed_contexts["appearance"],
                            geometry=typed_contexts["geometry"],
                            future_transport=typed_future_context,
                            query_key=query_row.mean(dim=3),
                        )
                        shared_glimpse_bank.validate(
                            batch=batch,

                            rows=stop - start,
                            glimpses=glimpses,
                            micro_cells=self.raw_micro_grid**2,
                            raw_dim=self.lattice_raw_dim,
                            route_dim=self.lattice_route_dim,
                        )
                    raw_context = None
                    if (
                        collect_diagnostics
                        and self.structured_ownership
                        and owner_fine_weights
                    ):
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_semantic_appearance_fine_l1", []
                        ).append(
                            0.5
                            * (
                                owner_fine_weights["semantic"]
                                - owner_fine_weights["appearance"]
                            ).abs().sum(dim=-1).mean().detach()
                        )
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_semantic_appearance_route_l1", []
                        ).append(
                            0.5
                            * (
                                owner_route_weights["semantic"]
                                - owner_route_weights["appearance"]
                            ).abs().flatten(3).sum(dim=-1).mean().detach()
                        )
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_appearance_geometry_route_l1", []
                        ).append(
                            0.5
                            * (
                                owner_route_weights["appearance"]
                                - owner_route_weights["geometry"]
                            ).abs().flatten(3).sum(dim=-1).mean().detach()
                        )
                        typed_metric_rows.setdefault(
                            "flow_jepa_typed_p1_appearance_geometry_fine_l1", []
                        ).append(
                            0.5
                            * (
                                owner_fine_weights["appearance"]
                                - owner_fine_weights["geometry"]
                            ).abs().sum(dim=-1).mean().detach()
                        )
                else:
                    shared_glimpse_bank = None
                    typed_rgb_micro = None
                    typed_detail_micro = None
                    typed_coordinate_micro = None
                    typed_contexts = {}
                    typed_future_context = None
                    assert local_values is not None
                    raw_context = torch.einsum(
                        "bqgcijm,bqgcijmr->bqgr",
                        route_weights,
                        local_values,
                    )
                if collect_diagnostics:
                    route_entropy = -(
                        route_weights.clamp_min(1e-8)
                        * route_weights.clamp_min(1e-8).log()
                    ).sum(dim=(3, 4, 5, 6)) / math.log(
                        float(max(state_count, 2))
                    )
                    fine_entropy = -(
                        fine_weights.clamp_min(1e-8)
                        * fine_weights.clamp_min(1e-8).log()
                    ).sum(dim=-1) / math.log(float(max(candidates, 2)))
                    weighted_fine_entropy = (
                        fine_entropy * route_weights
                    ).sum(dim=(3, 4, 5, 6))
                    weighted_fine_max = (
                        fine_weights.max(dim=-1).values * route_weights
                    ).sum(dim=(3, 4, 5, 6))
                    camera_mass = route_weights.sum(dim=(4, 5, 6))
                    slot_mass = route_weights.sum(dim=(3, 4, 5))

                else:
                    metric_shape = (
                        batch,
                        stop - start,
                        glimpses,
                    )
                    route_entropy = route_weights.new_zeros(metric_shape)
                    weighted_fine_entropy = route_weights.new_zeros(
                        metric_shape
                    )
                    weighted_fine_max = route_weights.new_zeros(metric_shape)
                    camera_mass = route_weights.new_zeros(
                        *metric_shape, cameras
                    )
                    slot_mass = route_weights.new_zeros(
                        *metric_shape, slots
                    )
                if posterior_basis is not None and fine_basis is not None:
                    posterior_signature = torch.einsum(
                        "bqgcijm,cijmf->bqgf",
                        route_weights,
                        posterior_basis,
                    )
                    fine_expected = torch.einsum(
                        "bqgcijmk,kf->bqgcijmf",
                        fine_weights,
                        fine_basis,
                    )
                    fine_signature = torch.einsum(
                        "bqgcijm,bqgcijmf->bqgf",
                        route_weights,
                        fine_expected,
                    )
                if intervention in {
                    "address_posterior_uniform",
                    "camera_posterior_uniform",
                }:
                    posterior_delta = (
                        route_weights - baseline_route_weights
                    ).abs().sum(dim=(3, 4, 5, 6)).mean()
                    self._address_eval_metrics[
                        "address_posterior_l1_delta"
                    ] = float(posterior_delta.detach().cpu())
                if intervention == "fine_offset_zero":
                    fine_delta = (
                        fine_weights - baseline_fine_weights
                    ).abs().sum(dim=-1)
                    self._address_eval_metrics[
                        "fine_posterior_l1_delta"
                    ] = float(
                        (fine_delta * route_weights)
                        .sum(dim=(3, 4, 5, 6))
                        .mean()
                        .detach()
                        .cpu()
                    )
            if typed_fine_keys:
                assert typed_rgb_micro is not None
                assert typed_detail_micro is not None
                assert typed_coordinate_micro is not None
                assert typed_future_context is not None
                assert self.typed_local_refiners is not None
                factual_rgb = (
                    shared_glimpse_bank.literal_rgb
                    if shared_glimpse_bank is not None
                    else typed_rgb_micro
                )
                factual_detail = (
                    shared_glimpse_bank.learned_detail
                    if shared_glimpse_bank is not None
                    else typed_detail_micro
                )
                factual_coordinates = (
                    shared_glimpse_bank.coordinates
                    if shared_glimpse_bank is not None
                    else typed_coordinate_micro
                )
                factual_contexts = (
                    {
                        "semantic": shared_glimpse_bank.semantic,

                        "appearance": shared_glimpse_bank.appearance,
                        "geometry": shared_glimpse_bank.geometry,
                    }
                    if shared_glimpse_bank is not None
                    else typed_contexts
                )
                factual_transport = (
                    shared_glimpse_bank.future_transport
                    if shared_glimpse_bank is not None
                    else typed_future_context
                )
                typed_query_context = (
                    p2_query_structured[:, start:stop].mean(dim=4)
                    if p2_query_structured is not None
                    else query_row.mean(dim=3)
                )
                head_rows: list[Tensor] = []
                chunk_metrics: dict[str, list[Tensor]] = {}
                p2_basis = basis if self.utility_precision_mainline else 1
                flat_rows = batch * (stop - start) * p2_basis
                for glimpse_index, refiner in enumerate(self.typed_local_refiners):
                    if self.utility_precision_mainline:
                        rgb_row = factual_rgb[
                            :, :, glimpse_index
                        ][:, :, None].expand(
                            -1, -1, basis, -1, -1
                        )
                        detail_row = factual_detail[
                            :, :, glimpse_index
                        ][:, :, None].expand(
                            -1, -1, basis, -1, -1
                        )
                        coordinate_row = factual_coordinates[
                            :, :, glimpse_index
                        ][:, :, None].expand(
                            -1, -1, basis, -1, -1
                        )
                        semantic_row = factual_contexts["semantic"][
                            :, :, glimpse_index
                        ][:, :, None].expand(-1, -1, basis, -1)
                        appearance_row = factual_contexts["appearance"][
                            :, :, glimpse_index
                        ][:, :, None].expand(-1, -1, basis, -1)
                        geometry_row = factual_contexts["geometry"][
                            :, :, glimpse_index
                        ][:, :, None].expand(-1, -1, basis, -1)
                        future_row = factual_transport[
                            :, :, glimpse_index
                        ][:, :, None].expand(-1, -1, basis, -1)
                        query_context_row = typed_query_context[
                            :, :, :, glimpse_index
                        ]
                    else:
                        rgb_row = factual_rgb[:, :, glimpse_index]
                        detail_row = factual_detail[:, :, glimpse_index]
                        coordinate_row = factual_coordinates[
                            :, :, glimpse_index
                        ]
                        semantic_row = factual_contexts["semantic"][
                            :, :, glimpse_index
                        ]
                        appearance_row = factual_contexts["appearance"][
                            :, :, glimpse_index
                        ]
                        geometry_row = factual_contexts["geometry"][
                            :, :, glimpse_index
                        ]
                        future_row = factual_transport[
                            :, :, glimpse_index
                        ]
                        query_context_row = typed_query_context[
                            :, :, glimpse_index
                        ]
                    refiner_kwargs: dict[str, bool] = {}
                    if self.utility_precision_mainline:
                        refiner_kwargs["collect_diagnostics"] = (
                            collect_diagnostics
                        )
                    refined, local_metrics = refiner(
                        rgb=rgb_row.reshape(

                            flat_rows, 1, self.raw_micro_grid**2, 3
                        ).to(dtype=query_input.dtype),
                        learned_detail=detail_row.reshape(
                            flat_rows,
                            1,
                            self.raw_micro_grid**2,
                            self.lattice_raw_dim,
                        ).to(dtype=query_input.dtype),
                        coordinates=coordinate_row.reshape(
                            flat_rows, 1, self.raw_micro_grid**2, 2
                        ).to(dtype=query_input.dtype),
                        query=query_context_row.reshape(
                            flat_rows, 1, self.lattice_route_dim
                        ),
                        semantic=semantic_row.reshape(
                            flat_rows, 1, self.lattice_route_dim
                        ).to(
                            dtype=query_input.dtype
                        ),
                        appearance=appearance_row.reshape(
                            flat_rows, 1, self.lattice_route_dim
                        ).to(
                            dtype=query_input.dtype
                        ),
                        geometry=geometry_row.reshape(
                            flat_rows, 1, self.lattice_route_dim
                        ).to(
                            dtype=query_input.dtype
                        ),
                        future_transport=future_row.reshape(
                            flat_rows, 1, 5
                        ).to(dtype=query_input.dtype),
                        intervention=(
                            self._address_eval_intervention
                            if self.functional_mainline_routing
                            else None
                        ),
                        **refiner_kwargs,
                    )
                    head_rows.append(
                        refined[:, 0].reshape(
                            batch,
                            stop - start,
                            p2_basis,
                            self.head_dim,
                        )
                    )
                    for name, value in local_metrics.items():
                        chunk_metrics.setdefault(name, []).append(value)
                if self.shared_factual_glimpse_bank:
                    if (
                        shared_glimpse_bank is None
                        or self.shared_p2_glimpse_query is None
                        or self.shared_p2_glimpse_key is None
                    ):
                        raise RuntimeError(
                            "V115 basis-specific factual cross-read is incomplete"
                        )
                    factual_values = torch.stack(head_rows, dim=3)
                    cross_query = self.shared_p2_glimpse_query(
                        typed_query_context
                    )
                    cross_key = self.shared_p2_glimpse_key(
                        shared_glimpse_bank.query_key
                    )
                    with torch.autocast(
                        device_type=cross_query.device.type, enabled=False
                    ):
                        cross_logits = torch.einsum(
                            "bqkgr,bqsr->bqkgs",
                            cross_query.float(),
                            cross_key.float(),
                        ) * (float(self.lattice_route_dim) ** -0.5)
                        cross_weights = torch.softmax(
                            cross_logits, dim=-1
                        )
                        crossed_values = torch.einsum(
                            "bqkgs,bqksd->bqkgd",
                            cross_weights,
                            factual_values.float(),

                        )
                    typed_output = crossed_values.to(
                        dtype=factual_values.dtype
                    ).flatten(start_dim=-2)
                    if collect_diagnostics:
                        cross_entropy = -(
                            cross_weights.clamp_min(1e-8)
                            * cross_weights.clamp_min(1e-8).log()
                        ).sum(dim=-1) / math.log(float(max(glimpses, 2)))
                        typed_metric_rows.setdefault(
                            "flow_jepa_p2_factual_cross_entropy", []
                        ).append(cross_entropy.mean().detach())
                        typed_metric_rows.setdefault(
                            "flow_jepa_p2_factual_cross_max", []
                        ).append(
                            cross_weights.max(dim=-1).values.mean().detach()
                        )
                        typed_metric_rows.setdefault(
                            "flow_jepa_p2_factual_cross_basis_variation", []
                        ).append(
                            (
                                cross_weights
                                - cross_weights.mean(dim=2, keepdim=True)
                            )
                            .abs()
                            .mean()
                            .detach()
                        )
                        owner_mix = self._shared_factual_owner_weights()
                        for owner_index, owner_name in enumerate(
                            ("semantic", "appearance", "geometry")
                        ):
                            typed_metric_rows.setdefault(
                                "flow_jepa_p1_factual_"
                                f"{owner_name}_owner_mass",
                                [],
                            ).append(
                                owner_mix[:, owner_index].mean().detach()
                            )
                else:
                    typed_output = torch.cat(head_rows, dim=-1)
                output_rows.append(
                    typed_output
                    if self.utility_precision_mainline
                    else typed_output[:, :, 0]
                )
                for name, values in chunk_metrics.items():
                    typed_metric_rows.setdefault(name, []).append(
                        torch.stack(values).mean()
                    )
                for name, value in typed_fine_logit_rms.items():
                    typed_metric_rows.setdefault(
                        f"flow_jepa_typed_p1_{name}_fine_logit_rms", []
                    ).append(value)
                for name, value in typed_route_logit_rms.items():
                    typed_metric_rows.setdefault(
                        f"flow_jepa_typed_p1_{name}_route_logit_rms", []
                    ).append(value)
            else:
                assert raw_context is not None
                raw_context_model = raw_context.to(dtype=query_input.dtype)
            if not typed_fine_keys and self.policy_multi_glimpse_address:
                if not isinstance(self.lattice_value_out, nn.ModuleList):
                    raise RuntimeError(
                        "multi-glimpse policy reader is missing per-glimpse value heads"
                    )
                output_rows.append(
                    torch.cat(
                        [
                            head(raw_context_model[:, :, index])
                            for index, head in enumerate(self.lattice_value_out)
                        ],
                        dim=-1,
                    )
                )
            elif not typed_fine_keys:
                if not isinstance(self.lattice_value_out, nn.Sequential):
                    raise RuntimeError(
                        "single-glimpse policy reader has an invalid value head"
                    )

                output_rows.append(
                    self.lattice_value_out(raw_context_model[:, :, 0])
                )
            route_entropy_rows.append(route_entropy)
            route_max_rows.append(
                route_weights.flatten(3).max(dim=-1).values
                if collect_diagnostics
                else route_entropy.new_zeros(route_entropy.shape)
            )
            fine_entropy_rows.append(weighted_fine_entropy)
            fine_max_rows.append(weighted_fine_max)
            camera_mass_rows.append(camera_mass)
            slot_mass_rows.append(slot_mass)
            world_logit_std_rows.append(
                world_logits.flatten(3).std(dim=-1, unbiased=False)
                if collect_diagnostics
                else route_entropy.new_zeros(route_entropy.shape)
            )
            if posterior_basis is not None and fine_basis is not None:
                posterior_signature_rows.append(posterior_signature)
                fine_signature_rows.append(fine_signature)

        context = torch.cat(output_rows, dim=1).reshape_as(trajectory)
        update = context * self.fixed_scale
        updated = trajectory + update
        if not collect_diagnostics:
            return updated, {
                "flow_jepa_p1_query_rows": trajectory.new_tensor(
                    float(horizon * address_basis), dtype=torch.float32
                ),
                "flow_jepa_p2_query_rows": trajectory.new_tensor(
                    float(horizon * basis), dtype=torch.float32
                ),
                "flow_jepa_p1_query_chunk": trajectory.new_tensor(
                    float(chunk), dtype=torch.float32
                ),
                "flow_jepa_p1_shared_factual": trajectory.new_tensor(
                    float(self.utility_precision_mainline),
                    dtype=torch.float32,
                ),
                "flow_jepa_shared_factual_glimpse_bank": trajectory.new_tensor(
                    float(self.shared_factual_glimpse_bank),
                    dtype=torch.float32,
                ),
            }
        route_entropy = torch.cat(route_entropy_rows, dim=1)
        route_max = torch.cat(route_max_rows, dim=1)
        fine_entropy = torch.cat(fine_entropy_rows, dim=1)
        fine_max = torch.cat(fine_max_rows, dim=1)
        camera_mass = torch.cat(camera_mass_rows, dim=1)
        slot_mass = torch.cat(slot_mass_rows, dim=1)
        world_logit_std = torch.cat(world_logit_std_rows, dim=1)
        posterior_signature = (
            torch.cat(posterior_signature_rows, dim=1).mean(dim=(0, 1, 2))
            if posterior_signature_rows
            else None
        )
        fine_signature = (
            torch.cat(fine_signature_rows, dim=1).mean(dim=(0, 1, 2))
            if fine_signature_rows
            else None
        )
        camera_entropy = -(
            camera_mass.clamp_min(1e-8)
            * camera_mass.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(max(cameras, 2)))
        slot_entropy_raw = -(
            slot_mass.clamp_min(1e-8)
            * slot_mass.clamp_min(1e-8).log()
        ).sum(dim=-1)
        slot_entropy = slot_entropy_raw / math.log(float(max(slots, 2)))
        slot_effective_count = slot_entropy_raw.exp()
        slot_query_variation = slot_mass.std(dim=1, unbiased=False).mean()
        if glimpses > 1:
            glimpse_route_distance = (
                route_entropy.new_tensor(0.0)
                if glimpses < 2
                else slot_mass.std(dim=2, unbiased=False).mean()
            )
            glimpse_effective_count = torch.exp(

                -(
                    slot_mass.mean(dim=(0, 1)).clamp_min(1e-8)
                    * slot_mass.mean(dim=(0, 1)).clamp_min(1e-8).log()
                ).sum(dim=-1)
            ).mean()
        else:
            glimpse_route_distance = route_entropy.new_zeros(())
            glimpse_effective_count = route_entropy.new_ones(())
        trajectory_norm = trajectory.detach().float().norm(dim=-1).mean()
        update_norm = update.detach().float().norm(dim=-1).mean()
        if intervention is not None:
            if posterior_signature is None or fine_signature is None:
                raise RuntimeError("address intervention did not capture its signatures")
            self._address_eval_apply_count += 1
            self._address_eval_metrics["intervention_code"] = {
                "none": 0.0,
                "address_posterior_uniform": 1.0,
                "fine_offset_zero": 2.0,
                "camera_posterior_uniform": 3.0,
                "camera_swap": 4.0,
                "world_query_zero": 5.0,
                "world_query_spatial_shuffle": 6.0,
                "future_transport_neutral": 7.0,
                "future_transport_spatial_shuffle": 8.0,
                "semantic_owner_zero": 9.0,
                "semantic_owner_shuffle": 10.0,
                "appearance_owner_zero": 11.0,
                "appearance_owner_shuffle": 12.0,
                "geometry_owner_zero": 13.0,
                "geometry_owner_shuffle": 14.0,
                "p1_appearance_gateway_zero": 15.0,
                "p1_appearance_gateway_spatial_shuffle": 16.0,
                "p2_semantic_zero": 17.0,
                "p2_semantic_shuffle": 18.0,
                "p2_appearance_zero": 19.0,
                "p2_appearance_shuffle": 20.0,
                "p2_geometry_zero": 21.0,
                "p2_geometry_shuffle": 22.0,
                "p2_horizon_zero": 23.0,
                "p2_horizon_shuffle": 24.0,
                "p2_rgb_precision_zero": 25.0,
                "p2_rgb_precision_spatial_shuffle": 26.0,
                "p2_detail_precision_zero": 27.0,
                "p2_detail_precision_spatial_shuffle": 28.0,
                "p2_basis0_zero": 29.0,
                "p2_basis0_horizon_shuffle": 30.0,
                "p2_basis1_zero": 31.0,
                "p2_basis1_horizon_shuffle": 32.0,
                "p2_basis2_zero": 33.0,
                "p2_basis2_horizon_shuffle": 34.0,
                "p2_basis3_zero": 35.0,
                "p2_basis3_horizon_shuffle": 36.0,
            }[intervention]
            self._address_eval_metrics["world_query_input_delta_norm"] = float(
                world_query_input_delta.cpu()
            )
            self._address_eval_metrics[
                "world_source_prior_input_delta_norm"
            ] = float(world_source_prior_input_delta.cpu())
            self._address_eval_metrics[
                "future_transport_input_delta_norm"
            ] = float(future_transport_input_delta.cpu())
            gateway_delta_rows = typed_metric_rows.get(
                "flow_jepa_typed_p1_appearance_gateway_"
                "intervention_delta_norm",
                [],
            )
            if gateway_delta_rows:
                self._address_eval_metrics[
                    "flow_jepa_typed_p1_appearance_gateway_"
                    "intervention_delta_norm"
                ] = float(
                    torch.stack(gateway_delta_rows).mean().detach().cpu()
                )
            for index, value in enumerate(posterior_signature):
                self._address_eval_metrics[
                    f"address_posterior_signature_{index}"
                ] = float(value.detach().cpu())
            for index, value in enumerate(fine_signature):
                self._address_eval_metrics[

                    f"fine_posterior_signature_{index}"
                ] = float(value.detach().cpu())
            hidden_axis = torch.arange(
                int(update.shape[-1]),
                device=update.device,
                dtype=torch.float32,
            )
            hidden_basis = (
                torch.sin((hidden_axis + 1.0) * 0.37),
                torch.cos((hidden_axis + 1.0) * 0.61),
                torch.sin((hidden_axis + 1.0) * 1.13),
                torch.cos((hidden_axis + 1.0) * 1.71),
            )
            update_f = update.detach().float()
            for index, basis_row in enumerate(hidden_basis):
                self._address_eval_metrics[
                    f"detail_update_signature_{index}"
                ] = float((update_f * basis_row).mean().cpu())
            self._address_eval_metrics[
                "flow_jepa_address_policy_slot_entropy"
            ] = float(slot_entropy.detach().mean().cpu())
            self._address_eval_metrics[
                "flow_jepa_address_policy_slot_effective_count"
            ] = float(slot_effective_count.detach().mean().cpu())
            self._address_eval_metrics[
                "flow_jepa_address_policy_slot_max"
            ] = float(slot_mass.detach().amax(dim=-1).mean().cpu())
            self._address_eval_metrics[
                "flow_jepa_address_policy_slot_query_variation"
            ] = float(slot_query_variation.detach().cpu())
        metrics = {
            "flow_jepa_late_detail_attention_entropy": route_entropy.mean().detach(),
            "flow_jepa_late_detail_attention_max": route_max.mean().detach(),
            "flow_jepa_late_detail_update_norm": update_norm,
            "flow_jepa_late_detail_trajectory_ratio": (
                update_norm / trajectory_norm.clamp_min(1e-6)
            ),
            "flow_jepa_late_detail_fixed_scale": trajectory.new_tensor(
                self.fixed_scale, dtype=torch.float32
            ),
            "flow_jepa_late_detail_token_count": trajectory.new_tensor(
                float(state_count * candidates), dtype=torch.float32
            ),
            "flow_jepa_address_policy_entropy": route_entropy.mean().detach(),
            "flow_jepa_address_policy_max": route_max.mean().detach(),
            "flow_jepa_address_fine_entropy": fine_entropy.mean().detach(),
            "flow_jepa_address_fine_max": fine_max.mean().detach(),
            "flow_jepa_address_camera_entropy": camera_entropy.mean().detach(),
            "flow_jepa_address_camera_max": camera_mass.max(
                dim=-1
            ).values.mean().detach(),
            "flow_jepa_address_policy_slot_entropy": (
                slot_entropy.mean().detach()
            ),
            "flow_jepa_address_policy_slot_effective_count": (
                slot_effective_count.mean().detach()
            ),
            "flow_jepa_address_policy_slot_max": (
                slot_mass.max(dim=-1).values.mean().detach()
            ),
            "flow_jepa_address_policy_slot_query_variation": (
                slot_query_variation.detach()
            ),
            "flow_jepa_address_fine_evidence_scale": evidence_scale.detach(),
            "flow_jepa_address_world_spatial_logit_std": (
                world_logit_std.mean().detach()
            ),
            "flow_jepa_address_policy_glimpse_count": trajectory.new_tensor(
                float(glimpses), dtype=torch.float32
            ),
            "flow_jepa_p1_query_rows": trajectory.new_tensor(
                float(horizon * address_basis), dtype=torch.float32
            ),
            "flow_jepa_p2_query_rows": trajectory.new_tensor(
                float(horizon * basis), dtype=torch.float32
            ),
            "flow_jepa_p1_query_chunk": trajectory.new_tensor(
                float(chunk), dtype=torch.float32
            ),
            "flow_jepa_p1_shared_factual": trajectory.new_tensor(

                float(self.utility_precision_mainline), dtype=torch.float32
            ),
            "flow_jepa_shared_factual_glimpse_bank": trajectory.new_tensor(
                float(self.shared_factual_glimpse_bank), dtype=torch.float32
            ),
            "flow_jepa_p1_clean_basis_entropy": shared_basis_entropy,
            "flow_jepa_policy_multi_glimpse_address": trajectory.new_tensor(
                float(self.policy_multi_glimpse_address), dtype=torch.float32
            ),
            "flow_jepa_address_policy_glimpse_route_variation": (
                glimpse_route_distance.detach()
            ),
            "flow_jepa_address_policy_glimpse_slot_effective_count": (
                glimpse_effective_count.detach()
            ),
        }
        if self.coordinate_typed_raw_detail:
            metrics.update(
                {
                    "flow_jepa_coordinate_typed_raw_detail": trajectory.new_ones(
                        (), dtype=torch.float32
                    ),
                    "flow_jepa_structured_ownership_bottleneck": (
                        trajectory.new_tensor(
                            float(self.structured_ownership), dtype=torch.float32
                        )
                    ),
                    "flow_jepa_pre_value_owner_routing": (
                        trajectory.new_tensor(
                            float(self.pre_value_owner_routing),
                            dtype=torch.float32,
                        )
                    ),
                    "flow_jepa_typed_p1_micro_grid": trajectory.new_tensor(
                        float(self.raw_micro_grid), dtype=torch.float32
                    ),
                    "flow_jepa_typed_p1_micro_token_count": trajectory.new_tensor(
                        float(self.raw_micro_grid**2), dtype=torch.float32
                    ),
                    "flow_jepa_typed_p1_activation_checkpoint": (
                        trajectory.new_tensor(
                            float(self.raw_activation_checkpoint),
                            dtype=torch.float32,
                        )
                    ),
                    "flow_jepa_typed_p1_activation_checkpoint_active": (
                        trajectory.new_tensor(
                            float(activation_checkpoint_active),
                            dtype=torch.float32,
                        )
                    ),
                    "flow_jepa_address_query_chunk_actual": (
                        trajectory.new_tensor(
                            float(chunk),
                            dtype=torch.float32,
                        )
                    ),
                    **{
                        name: torch.stack(values).mean()
                        for name, values in typed_metric_rows.items()
                        if values
                    },
                }
            )
        if progressive is not None:
            assert progressive_coarse_bias is not None
            assert progressive_fine_bias is not None
            metrics.update(
                {
                    "flow_jepa_progressive_policy_prior_active": (
                        trajectory.new_ones((), dtype=torch.float32)
                    ),
                    "flow_jepa_progressive_policy_coarse_prior_rms": (
                        progressive_coarse_bias.detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    "flow_jepa_progressive_policy_fine_prior_rms": (

                        progressive_fine_bias.detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    "flow_jepa_progressive_policy_world_prior_rms": (
                        progressive_world_source_bias.detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                        if progressive_world_source_bias is not None
                        else trajectory.new_zeros((), dtype=torch.float32)
                    ),
                    "flow_jepa_p1_g3_only_factual_address": (
                        trajectory.new_tensor(
                            float(self.g_aligned_future_effect),
                            dtype=torch.float32,
                        )
                    ),
                }
            )
        return updated, metrics

    def forward(
        self,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor,
        detail: LateRawDetailEvidence,
        phase_context: Tensor | None = None,
        condition_query_context: Tensor | None = None,
        history_query_context: Tensor | None = None,
        clean_basis_tokens: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cfg = self.config
        batch = int(trajectory_tokens.shape[0])
        horizon = int(cfg.action_horizon)
        basis = int(cfg.action_basis_tokens)
        expected_trajectory = horizon * basis
        if tuple(trajectory_tokens.shape) != (
            batch,
            expected_trajectory,
            self.hidden,
        ):
            raise ValueError(
                "late raw-detail trajectory must be "
                f"[B,{expected_trajectory},{self.hidden}]"
            )
        selector = detail.selector_tokens
        values = detail.value_tokens
        if (
            selector.ndim != 3
            or tuple(selector.shape) != tuple(values.shape)
            or int(selector.shape[0]) != batch
            or int(selector.shape[-1]) != self.hidden
        ):
            raise ValueError(
                "late raw-detail selector/value must align as [B,N,H]"
            )
        trajectory = trajectory_tokens.reshape(
            batch, horizon, basis, self.hidden
        )
        boundaries = (
            tuple(int(value) for value in cfg.flow_jepa_action_offsets)
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else None
        )
        if self.functional_mainline_routing:
            zero_context = trajectory.new_zeros(batch, horizon, self.hidden)
        else:
            zero_context = trajectory.new_zeros(batch, self.hidden)
        phase_query_delta = zero_context
        condition_query_delta = zero_context
        history_query_delta = zero_context
        if self.phase_query_proj is not None:
            expected_context = (
                (batch, int(cfg.future_anchors), self.hidden)
                if self.functional_mainline_routing

                else (batch, self.hidden)
            )
            if phase_context is None or tuple(phase_context.shape) != expected_context:
                raise ValueError(
                    "stateless phase detail query has the wrong context schema"
                )
            phase_input = phase_context.to(
                device=trajectory.device, dtype=trajectory.dtype
            )
            if self.functional_mainline_routing:
                phase_input = _align_milestone_tokens_to_horizon(
                    phase_input[:, : len(boundaries)],
                    horizon,
                    boundaries=boundaries,
                )
            phase_query_delta = self.phase_query_scale * self.phase_query_proj(
                phase_input
            )
            if self.differential_intent_effect_mainline:
                if (
                    condition_query_context is not None
                    or history_query_context is not None
                    or self.condition_query_proj is not None
                    or self.history_query_proj is not None
                ):
                    raise ValueError(
                        "differential P1 accepts only the canonical "
                        "IntentWindowView context"
                    )
            else:
                if (
                    self.condition_query_proj is None
                    or condition_query_context is None
                    or tuple(condition_query_context.shape) != expected_context
                ):
                    raise ValueError(
                        "goal detail query has the wrong context schema"
                    )
                condition_input = condition_query_context.to(
                    device=trajectory.device, dtype=trajectory.dtype
                )
                if self.functional_mainline_routing:
                    condition_input = _align_milestone_tokens_to_horizon(
                        condition_input[:, : len(boundaries)],
                        horizon,
                        boundaries=boundaries,
                    )
                condition_query_delta = (
                    self.phase_query_scale
                    * self.condition_query_proj(
                        condition_input
                    )
                )
                if self.functional_mainline_routing:
                    if (
                        self.history_query_proj is None
                        or history_query_context is None
                        or tuple(history_query_context.shape) != expected_context
                    ):
                        raise ValueError(
                            "history detail query has the wrong context schema"
                        )
                    history_input = _align_milestone_tokens_to_horizon(
                        history_query_context[
                            :, : len(boundaries)
                        ].to(device=trajectory.device, dtype=trajectory.dtype),
                        horizon,
                        boundaries=boundaries,
                    )
                    history_query_delta = (
                        self.phase_query_scale
                        * self.history_query_proj(history_input)
                    )
        elif phase_context is not None:
            raise ValueError("phase_context was supplied while phase routing is disabled")
        elif condition_query_context is not None:
            raise ValueError(
                "condition query context was supplied while phase routing is disabled"
            )
        elif history_query_context is not None:

            raise ValueError(
                "history query context was supplied while phase routing is disabled"
            )
        cameras = int(cfg.num_cameras)
        grid = int(cfg.future_grid_size)
        anchors = int(cfg.future_anchors)
        expected_detail = cameras * grid * grid
        expected_rollout = anchors * expected_detail
        if int(rollout_tokens.shape[1]) != expected_rollout:
            raise ValueError(
                "late raw-detail world tokens must preserve "
                f"anchor*camera*grid^2={expected_rollout}"
            )
        # Keep camera and xy ownership through the complete soft-lattice read.
        # The global query context still uses an anchor/camera summary, while
        # the selector logits also receive the aligned W chart cell.  Only the
        # legacy V102 reader discards xy before address selection.
        world_anchor_grid = rollout_tokens.reshape(
            batch,
            anchors,
            cameras,
            grid,
            grid,
            self.hidden,
        )
        world_anchor_camera = world_anchor_grid.mean(dim=(3, 4))
        aligned_world_anchor_camera = (
            world_anchor_camera[:, : len(boundaries)]
            if boundaries is not None
            else world_anchor_camera
        )
        world_horizon = _align_milestone_tokens_to_horizon(
            aligned_world_anchor_camera.permute(0, 2, 1, 3).reshape(
                batch * cameras,
                int(aligned_world_anchor_camera.shape[1]),
                self.hidden,
            ),
            horizon,
            boundaries=boundaries,
        ).reshape(batch, cameras, horizon, self.hidden).permute(0, 2, 1, 3)
        aligned_world_anchor_grid = (
            world_anchor_grid[:, : len(boundaries)]
            if boundaries is not None
            else world_anchor_grid
        )
        world_horizon_grid = _align_milestone_tokens_to_horizon(
            aligned_world_anchor_grid.permute(0, 2, 3, 4, 1, 5).reshape(
                batch * cameras * grid * grid,
                int(aligned_world_anchor_grid.shape[1]),
                self.hidden,
            ),
            horizon,
            boundaries=boundaries,
        ).reshape(
            batch,
            cameras,
            grid,
            grid,
            horizon,
            self.hidden,
        ).permute(0, 4, 1, 2, 3, 5)
        if self.functional_mainline_routing:
            trajectory_query = (
                trajectory
                + phase_query_delta[:, :, None]
                + condition_query_delta[:, :, None]
                + history_query_delta[:, :, None]
            )
        else:
            trajectory_query = (
                trajectory
                + phase_query_delta[:, None, None]
                + condition_query_delta[:, None, None]
            )
        factual_condition = (
            phase_query_delta + condition_query_delta + history_query_delta
            if self.functional_mainline_routing
            else trajectory.new_zeros(batch, horizon, self.hidden)
        )
        if self.utility_precision_mainline:

            if clean_basis_tokens is None or tuple(clean_basis_tokens.shape) != (
                batch,
                horizon,
                basis,
                self.hidden,
            ):
                raise ValueError(
                    "utility/precision reader requires clean basis tokens "
                    f"[B,{horizon},{basis},{self.hidden}]"
                )
            clean_basis_tokens = clean_basis_tokens.to(
                device=trajectory.device, dtype=trajectory.dtype
            )
        elif clean_basis_tokens is not None:
            raise ValueError(
                "clean basis tokens were supplied while utility P1 is disabled"
            )
        trajectory_by_camera = trajectory_query[:, :, :, None].expand(
            -1, -1, -1, cameras, -1
        )
        world = world_horizon[:, :, None].expand(-1, -1, basis, -1, -1)
        query_input = torch.cat((trajectory_by_camera, world), dim=-1)
        if detail.address_bank is not None:
            if not self.soft_address_lattice:
                raise RuntimeError(
                    "soft address bank was supplied to the legacy detail reader"
                )
            updated, metrics = self._read_soft_address_lattice(
                query_input,
                trajectory,
                world_horizon_grid,
                detail,
                clean_basis_tokens=clean_basis_tokens,
                factual_condition=factual_condition,
                collect_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                metrics["flow_jepa_phase_detail_query_norm"] = (
                    phase_query_delta.detach().float().norm(dim=-1).mean()
                )
                metrics["flow_jepa_condition_detail_query_norm"] = (
                    condition_query_delta.detach().float().norm(dim=-1).mean()
                )
                metrics["flow_jepa_history_detail_query_norm"] = (
                    history_query_delta.detach().float().norm(dim=-1).mean()
                )
            return updated.reshape_as(trajectory_tokens), metrics
        if self.soft_address_lattice:
            raise RuntimeError("soft address lattice reader received no address bank")
        if int(selector.shape[1]) != expected_detail:
            raise ValueError(
                "late raw-detail tokens must preserve camera*grid^2="
                f"{expected_detail}, got {selector.shape[1]}"
            )
        query = self.query_proj(self.query_norm(query_input)).reshape(
            batch, horizon, basis, cameras, self.heads, self.head_dim
        )
        detail_per_camera = grid * grid
        key = self.key_proj(self.key_norm(selector)).reshape(
            batch, cameras, detail_per_camera, self.heads, self.head_dim
        )
        logits = torch.einsum(
            "btkchd,bcnhd->btkchn", query.float(), key.float()
        ) * (float(self.head_dim) ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        value_heads = values.float().reshape(
            batch, cameras, detail_per_camera, self.heads, self.head_dim
        )
        camera_context = torch.einsum(
            "btkchn,bcnhd->btkchd", weights, value_heads
        ).reshape(batch, horizon, basis, cameras, self.hidden)
        context = camera_context.sum(dim=3) / math.sqrt(float(cameras))
        update = context.to(dtype=trajectory_tokens.dtype) * self.fixed_scale
        updated = trajectory + update
        normalized_entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(max(detail_per_camera, 2)))
        trajectory_norm = trajectory.detach().float().norm(dim=-1).mean()
        update_norm = update.detach().float().norm(dim=-1).mean()
        return updated.reshape_as(trajectory_tokens), {

            "flow_jepa_late_detail_attention_entropy": normalized_entropy.mean().detach(),
            "flow_jepa_late_detail_attention_max": weights.max(dim=-1).values.mean().detach(),
            "flow_jepa_late_detail_update_norm": update_norm,
            "flow_jepa_late_detail_trajectory_ratio": (
                update_norm / trajectory_norm.clamp_min(1e-6)
            ),
            "flow_jepa_late_detail_fixed_scale": trajectory_tokens.new_tensor(
                self.fixed_scale, dtype=torch.float32
            ),
            "flow_jepa_late_detail_token_count": trajectory_tokens.new_tensor(
                float(selector.shape[1]), dtype=torch.float32
            ),
            "flow_jepa_phase_detail_query_norm": (
                phase_query_delta.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_condition_detail_query_norm": (
                condition_query_delta.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_history_detail_query_norm": (
                history_query_delta.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_differential_p1_direct_condition_bypass": (
                torch.maximum(
                    condition_query_delta.detach().float().abs().amax(),
                    history_query_delta.detach().float().abs().amax(),
                )
                if self.differential_intent_effect_mainline
                else trajectory_tokens.new_zeros((), dtype=torch.float32)
            ),
        }

__all__ = ["LateRawDetailPolicyReader"]
