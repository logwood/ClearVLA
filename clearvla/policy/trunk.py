"""Current staged world/action trunk and layer contracts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .codec import ParsevalGripperTemporalFrame
from .config import V39PolicyConfig
from .contracts import scaled_contract_view as _scaled_contract_view
from .decoder import HierarchicalMMDiTActionDecoder
from .flow_dino_evidence import (
    FlowDINOEvidenceEncoder,
    FlowDINOEvidencePack,
    LateRawDetailEvidence,
    ProgressiveGroundingAddressState,
)
from .goal_conditioning import GoalTokenResampler, StatelessPhaseAdapter
from .legacy import (
    AdaptiveRecurrentCVAEActionDecoder,
    HierarchicalLatentMainActionDecoder,
    LatentCVAEActionDecoder,
    LayeredV37StyleResidualActionFlowDenoiser,
    V37StyleResidualActionFlowDenoiser,
)
from .primitives import TimeEmbedding
from .role_delta_attnres import (
    AffineVarianceFlooredCenteredNorm,
    PolicyRoleDeltaBank,
    RoleDeltaAttnRes,
    VarianceFlooredCenteredNorm,
)
from .time_domain_mmdit import EvidenceLatentMMDiTActionDecoder
from .trunk_primitives import (
    CanvasPhysicalVelocityHead,
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
    RolloutActionResidualHead,
    RolloutTargetCodec,
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
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


def _rollout_tokens_to_action_horizon(tokens: Tensor, config: V39PolicyConfig) -> Tensor:
    """Pool rollout spatial tokens per anchor, then align anchors to action time."""

    if tokens.ndim != 3:
        raise ValueError(f"rollout tokens must be [B,F*G,H], got {tuple(tokens.shape)}")
    grid = int(config.num_cameras) * int(config.future_grid_size) * int(config.future_grid_size)
    expected = int(config.future_anchors) * grid
    if int(tokens.shape[1]) != expected:
        raise ValueError(
            f"rollout token count must be future_anchors*grid={expected}, got {tokens.shape[1]}"
        )
    milestones = tokens.reshape(
        tokens.shape[0], int(config.future_anchors), grid, tokens.shape[-1]
    ).mean(dim=2)
    if int(getattr(config, "flow_jepa_enabled", 0)):
        boundaries = tuple(int(value) for value in config.flow_jepa_action_offsets)
        milestones = milestones[:, : len(boundaries)]
    else:
        boundaries = None
    return _align_milestone_tokens_to_horizon(
        milestones, int(config.action_horizon), boundaries=boundaries
    )


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
            if int(getattr(config, "stateless_phase_enabled", 0))
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
                self.typed_coarse_query = nn.ModuleDict(
                    {
                        name: nn.Linear(route_dim, route_dim, bias=False)
                        for name in ("semantic", "appearance", "geometry")
                    }
                )
                self.typed_local_refiners = nn.ModuleList(
                    [
                        (
                            _StructuredOwnershipLocalRefiner
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
            self.register_parameter("lattice_fine_evidence_scale", None)
            self.raw_micro_grid = 0
            self.register_buffer("typed_micro_basis", None, persistent=False)
            self.typed_fine_query = None
            self.typed_coarse_query = None
            self.typed_local_refiners = None

    @staticmethod
    def _typed_microgrid_expectation(
        route_weights: Tensor,
        fine_weights: Tensor,
        micro_basis: Tensor,
        literal_rgb: Tensor,
        learned_detail: Tensor,
        coordinates: Tensor,
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
            raise ValueError("typed fine and route posteriors do not align")
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

        route_f = route_weights.float()
        fine_f = fine_weights.float()
        basis_f = micro_basis.to(device=fine_f.device, dtype=torch.float32)
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
        }
        if normalized not in allowed:
            raise ValueError(
                "address intervention must be none/address_posterior_uniform/"
                "fine_offset_zero/camera_posterior_uniform/camera_swap/"
                "world_query_zero/world_query_spatial_shuffle/"
                "future_transport_neutral/future_transport_spatial_shuffle/"
                "semantic_owner_zero/semantic_owner_shuffle/"
                "appearance_owner_zero/appearance_owner_shuffle/"
                "geometry_owner_zero/geometry_owner_shuffle"
            )
        if self.training:
            raise RuntimeError("address-posterior intervention is evaluation-only")
        if not self.soft_address_lattice:
            raise RuntimeError(
                "address-posterior intervention requires the soft address lattice"
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
                or progressive.world_source_bias is None
            ):
                raise RuntimeError(
                    "completed W state has no progressive selector priors"
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
            world_source_bias = progressive.world_source_bias
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
            boundaries = tuple(
                int(value) for value in self.config.flow_jepa_action_offsets
            )
            aligned_world_source_bias = world_source_bias[:, : len(boundaries)]
            progressive_world_source_bias = _align_milestone_tokens_to_horizon(
                aligned_world_source_bias.permute(0, 2, 3, 4, 5, 1).reshape(
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
            if self.structured_ownership:
                for name, owner_source_bias in (
                    ("semantic", progressive.world_semantic_source_bias),
                    ("appearance", progressive.world_appearance_source_bias),
                    ("geometry", progressive.world_geometry_source_bias),
                ):
                    if owner_source_bias is None:
                        raise RuntimeError(
                            f"completed V111 state has no W {name} source sidecar"
                        )
                    aligned_owner_bias = owner_source_bias[:, : len(boundaries)]
                    progressive_world_owner_source_bias[name] = (
                        _align_milestone_tokens_to_horizon(
                            aligned_owner_bias.permute(0, 2, 3, 4, 5, 1).reshape(
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
            if self.coordinate_typed_raw_detail:
                typed_required = (
                    progressive.dynamic_semantic_keys,
                    progressive.dynamic_appearance_keys,
                    progressive.dynamic_geometry_keys,
                    progressive.dynamic_literal_rgb,
                    progressive.dynamic_fine_coordinates,
                    progressive.canonical_semantic_keys,
                    progressive.canonical_appearance_keys,
                    progressive.canonical_geometry_keys,
                    progressive.world_future_centers,
                    progressive.world_future_scale,
                    progressive.world_future_visibility,
                    progressive.world_future_uncertainty,
                )
                if not all(torch.is_tensor(value) for value in typed_required):
                    raise RuntimeError(
                        "completed V110 state has no typed current/future evidence"
                    )
                future_transport_anchors = torch.cat(
                    (
                        progressive.world_future_centers,
                        progressive.world_future_scale,
                        progressive.world_future_visibility,
                        progressive.world_future_uncertainty,
                    ),
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
            if progressive_future_transport is not None:
                progressive_future_transport = progressive_future_transport.roll(
                    shifts=(max(grid // 2, 1), max(grid // 3, 1)),
                    dims=(3, 4),
                )
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

        glimpses = self.heads if self.policy_multi_glimpse_address else 1
        query = self.lattice_query_proj(
            self.lattice_query_norm(query_input)
        ).reshape(
            batch,
            horizon * basis,
            cameras,
            glimpses,
            self.lattice_route_dim,
        ).permute(0, 1, 3, 2, 4)
        world_route = self.lattice_world_key_proj(
            self.lattice_world_norm(world_horizon_grid)
        )
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
        coarse_key = self.lattice_key_norm(coarse_keys)
        fine_key = self.lattice_key_norm(fine_keys)
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
            if progressive_future_transport is not None
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
                basis,
                glimpses,
                -1,
                -1,
                -1,
                -1,
            ).reshape(
                batch,
                horizon * basis,
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
                    basis,
                    glimpses,
                    -1,
                    -1,
                    -1,
                    -1,
                ).reshape(
                    batch,
                    horizon * basis,
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
                basis,
                glimpses,
                -1,
                -1,
                -1,
                -1,
                -1,
            ).reshape(
                batch,
                horizon * basis,
                glimpses,
                cameras,
                grid,
                grid,
                slots,
                int(progressive_future_transport.shape[-1]),
            )
        for start in range(0, int(query.shape[1]), chunk):
            stop = min(start + chunk, int(query.shape[1]))
            query_row = query[:, start:stop]
            typed_fine_query_rows: dict[str, Tensor] = {}
            typed_coarse_query_rows: dict[str, Tensor] = {}
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
                }
                typed_coarse_query_rows = {
                    name: self.typed_coarse_query[name](query_row)
                    for name in typed_coarse_keys
                }
            with torch.autocast(device_type=query.device.type, enabled=False):
                query_f = query_row.float()
                if typed_fine_keys:
                    typed_fine_logit_rms: dict[str, Tensor] = {}
                    typed_fine_logits: dict[str, Tensor] = {}
                    for name, typed_key in typed_fine_keys.items():
                        typed_logit = torch.einsum(
                            "bqgcr,bcijmkr->bqgcijmk",
                            typed_fine_query_rows[name].float(),
                            typed_key.float(),
                        ) * (float(self.lattice_route_dim) ** -0.5)
                        typed_fine_logit_rms[name] = (
                            typed_logit.detach().square().mean().sqrt()
                        )
                        typed_fine_logits[name] = typed_logit
                    if not typed_fine_logits:
                        raise RuntimeError("typed P1 produced no fine logits")
                    if self.structured_ownership:
                        # P1 factorizes coarse source ownership from precise
                        # offset verification.  Appearance and geometry own
                        # the fine offset; semantic evidence is retained for
                        # its context read but cannot collapse local detail.
                        fine_logits = (
                            typed_fine_logits["appearance"]
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
                    fine_logits = fine_logits + transport_fine_logit
                    if self.structured_ownership:
                        owner_fine_logits = {
                            "semantic": typed_fine_logits["semantic"],
                            "appearance": (
                                typed_fine_logits["appearance"]
                                + transport_fine_logit
                            ),
                            "geometry": (
                                typed_fine_logits["geometry"]
                                + transport_fine_logit
                            ),
                        }
                    else:
                        owner_fine_logits = {}
                    typed_metric_rows.setdefault(
                        "flow_jepa_typed_p1_future_transport_logit_rms", []
                    ).append(
                        transport_fine_logit.detach().square().mean().sqrt()
                    )
                else:
                    typed_fine_logit_rms = {}
                    typed_fine_logits = {}
                    owner_fine_logits = {}
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
                if self.structured_ownership and owner_fine_logits:
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
                        route_logits = (
                            typed_route_logits["semantic"]
                            + typed_route_logits["geometry"]
                        ) / math.sqrt(2.0)
                    else:
                        route_logits = sum(typed_route_logits.values()) / math.sqrt(3.0)
                else:
                    typed_route_logit_rms = {}
                    typed_route_logits = {}
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
                # W1-W3 organize coarse spatial relevance.  Add their local
                # compatibility once per camera/xy cell and let the existing
                # coarse/fine selector choose slot and sub-cell offset.  No
                # world value enters the raw value lane.
                world_logits = torch.einsum(
                    "bqgcr,bqcijr->bqgcij",
                    query_f,
                    world_route[:, start:stop].float(),
                ) * (float(self.lattice_route_dim) ** -0.5)
                route_logits = route_logits + world_logits[..., None]
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
                if self.structured_ownership and typed_route_logits:
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
                    route_weights = valid_states / valid_states.sum(
                        dim=(3, 4, 5, 6), keepdim=True
                    ).clamp_min(1.0)
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
                    if (
                        self.raw_activation_checkpoint
                        and self.training
                        and torch.is_grad_enabled()
                    ):
                        (
                            typed_rgb_micro,
                            typed_detail_micro,
                            typed_coordinate_micro,
                        ) = checkpoint(
                            self._typed_microgrid_expectation,
                            *micro_inputs,
                            use_reentrant=False,
                        )
                    else:
                        (
                            typed_rgb_micro,
                            typed_detail_micro,
                            typed_coordinate_micro,
                        ) = self._typed_microgrid_expectation(*micro_inputs)
                    typed_contexts = {
                        name: torch.einsum(
                            "bqgcijm,bqgcijmk,bcijmkr->bqgr",
                            owner_route_weights.get(name, route_weights),
                            owner_fine_weights.get(name, fine_weights),
                            typed_key.float(),
                        )
                        for name, typed_key in typed_fine_keys.items()
                    }
                    typed_future_context = torch.einsum(
                        "bqgcijm,bqgcijmv->bqgv",
                        route_weights,
                        typed_future_rows[:, start:stop].float(),
                    )
                    raw_context = None
                    if self.structured_ownership and owner_fine_weights:
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
                route_entropy = -(
                    route_weights.clamp_min(1e-8)
                    * route_weights.clamp_min(1e-8).log()
                ).sum(dim=(3, 4, 5, 6)) / math.log(float(max(state_count, 2)))
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
                typed_query_context = query_row.mean(dim=3)
                head_rows: list[Tensor] = []
                chunk_metrics: dict[str, list[Tensor]] = {}
                flat_rows = batch * (stop - start)
                for glimpse_index, refiner in enumerate(self.typed_local_refiners):
                    refined, local_metrics = refiner(
                        rgb=typed_rgb_micro[:, :, glimpse_index].reshape(
                            flat_rows, 1, self.raw_micro_grid**2, 3
                        ).to(dtype=query_input.dtype),
                        learned_detail=typed_detail_micro[
                            :, :, glimpse_index
                        ].reshape(
                            flat_rows,
                            1,
                            self.raw_micro_grid**2,
                            self.lattice_raw_dim,
                        ).to(dtype=query_input.dtype),
                        coordinates=typed_coordinate_micro[
                            :, :, glimpse_index
                        ].reshape(
                            flat_rows, 1, self.raw_micro_grid**2, 2
                        ).to(dtype=query_input.dtype),
                        query=typed_query_context[:, :, glimpse_index].reshape(
                            flat_rows, 1, self.lattice_route_dim
                        ),
                        semantic=typed_contexts["semantic"][
                            :, :, glimpse_index
                        ].reshape(flat_rows, 1, self.lattice_route_dim).to(
                            dtype=query_input.dtype
                        ),
                        appearance=typed_contexts["appearance"][
                            :, :, glimpse_index
                        ].reshape(flat_rows, 1, self.lattice_route_dim).to(
                            dtype=query_input.dtype
                        ),
                        geometry=typed_contexts["geometry"][
                            :, :, glimpse_index
                        ].reshape(flat_rows, 1, self.lattice_route_dim).to(
                            dtype=query_input.dtype
                        ),
                        future_transport=typed_future_context[
                            :, :, glimpse_index
                        ].reshape(flat_rows, 1, 5).to(dtype=query_input.dtype),
                    )
                    head_rows.append(
                        refined[:, 0].reshape(batch, stop - start, self.head_dim)
                    )
                    for name, value in local_metrics.items():
                        chunk_metrics.setdefault(name, []).append(value)
                output_rows.append(torch.cat(head_rows, dim=-1))
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
            route_max_rows.append(route_weights.flatten(3).max(dim=-1).values)
            fine_entropy_rows.append(weighted_fine_entropy)
            fine_max_rows.append(weighted_fine_max)
            camera_mass_rows.append(camera_mass)
            slot_mass_rows.append(slot_mass)
            world_logit_std_rows.append(
                world_logits.flatten(3).std(dim=-1, unbiased=False)
            )
            if posterior_basis is not None and fine_basis is not None:
                posterior_signature_rows.append(posterior_signature)
                fine_signature_rows.append(fine_signature)

        context = torch.cat(output_rows, dim=1).reshape_as(trajectory)
        update = context * self.fixed_scale
        updated = trajectory + update
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
        phase_query_delta = trajectory.new_zeros(batch, self.hidden)
        condition_query_delta = trajectory.new_zeros(batch, self.hidden)
        if self.phase_query_proj is not None:
            if (
                phase_context is None
                or tuple(phase_context.shape) != (batch, self.hidden)
            ):
                raise ValueError(
                    "stateless phase detail query requires [B,H] phase_context"
                )
            phase_query_delta = self.phase_query_scale * self.phase_query_proj(
                phase_context.to(device=trajectory.device, dtype=trajectory.dtype)
            )
            if (
                self.condition_query_proj is None
                or condition_query_context is None
                or tuple(condition_query_context.shape) != (batch, self.hidden)
            ):
                raise ValueError(
                    "goal/history detail query requires [B,H] condition context"
                )
            condition_query_delta = (
                self.phase_query_scale
                * self.condition_query_proj(
                    condition_query_context.to(
                        device=trajectory.device, dtype=trajectory.dtype
                    )
                )
            )
        elif phase_context is not None:
            raise ValueError("phase_context was supplied while phase routing is disabled")
        elif condition_query_context is not None:
            raise ValueError(
                "condition query context was supplied while phase routing is disabled"
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
        boundaries = (
            tuple(int(value) for value in cfg.flow_jepa_action_offsets)
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else None
        )
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
        trajectory_query = (
            trajectory
            + phase_query_delta[:, None, None]
            + condition_query_delta[:, None, None]
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
            )
            metrics["flow_jepa_phase_detail_query_norm"] = (
                phase_query_delta.detach().float().norm(dim=-1).mean()
            )
            metrics["flow_jepa_condition_detail_query_norm"] = (
                condition_query_delta.detach().float().norm(dim=-1).mean()
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
        }


class MidcutContractHeads(nn.Module):
    """Intentionally weak readouts from the DiT midpoint.

    The heads are deliberately no stronger than LayerNorm + Linear.  If these
    heads cannot read motion/event/future information, the information is not
    sufficiently explicit at the mid-cut latent.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.action_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.physical_action_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.rollout_effect_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.rollout_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.transition_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.future_gain = nn.Parameter(
            torch.tensor(float(config.midcut_future_gain_init), dtype=torch.float32)
        )
        # Start action/event readouts small but not exactly zero.  A fully
        # zero final Linear makes the first backward step update only the head
        # itself and gives essentially no gradient to the upstream latent.
        # Small random init keeps the head weak while allowing the contract
        # loss to shape the DiT canvas from the beginning.
        for module in (self.action_head[-1], self.event_head[-1], self.motion_head[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        for module in (
            self.rollout_effect_head[-1],
            self.rollout_delta_head[-1],
            self.transition_head[-1],
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def trajectory_pooled(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(
            b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size
        )
        return grouped.mean(dim=2)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        cfg = self.config
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.trajectory_pooled(trajectory)
        gain = self.future_gain.to(device=canvas.device, dtype=canvas.dtype)
        effect = self.rollout_effect_head(rollout) * gain
        delta = self.rollout_delta_head(rollout) * gain
        event_context = _rollout_tokens_to_action_horizon(delta, cfg)
        transition_base = delta.mean(dim=1, keepdim=True)
        transition = self.transition_head(transition_base).expand(-1, cfg.action_horizon, -1)
        return {
            "midcut_canvas_tokens": canvas,
            "midcut_trajectory_tokens": trajectory,
            "midcut_rollout_tokens": rollout,
            "midcut_register_tokens": registers,
            "midcut_state_tokens": canvas[:, slices["state"]],
            "midcut_state_history_tokens": canvas[:, slices["state_history"]],
            "midcut_executed_tokens": canvas[:, slices["executed"]],
            "midcut_proposal_tokens": canvas[:, slices["proposal"]],
            "midcut_trajectory_pooled": trajectory_pooled,
            "midcut_pred_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_direct_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_rollout_residual_velocity": torch.zeros(
                trajectory_pooled.shape[0],
                cfg.action_horizon,
                cfg.physical_action_dim,
                device=trajectory_pooled.device,
                dtype=trajectory_pooled.dtype,
            ),
            "midcut_rollout_alpha": torch.zeros(
                1,
                cfg.action_horizon,
                1,
                device=trajectory_pooled.device,
                dtype=trajectory_pooled.dtype,
            ),
            "midcut_rollout_effect_pred": effect,
            "midcut_rollout_delta_pred": delta,
            "midcut_rollout_base_effect_pred": torch.zeros_like(effect),
            "midcut_event_logits": self.event_head(event_context),
            "midcut_motion_logits": self.motion_head(trajectory_pooled).squeeze(-1),
            "midcut_transition_latent": transition,
            "midcut_rollout_delta_norm": delta.detach().float().norm(dim=-1).mean(),
            "midcut_rollout_effect_norm": effect.detach().float().norm(dim=-1).mean(),
            "midcut_future_gain": gain.detach().float().abs(),
        }


class LayerContractAdapterHeads(nn.Module):
    """Tiny per-layer adapter contract for V39.1.

    It first applies a small bottleneck residual adapter, then reuses the same
    deliberately weak readout family as the mid-cut contract.  The adapter keeps
    the probe local and cheap; the heads stay too weak to manufacture motion or
    contact structure after the trunk.
    """

    def __init__(self, config: V39PolicyConfig, *, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        h = int(config.hidden_size)
        b = int(config.layer_contract_adapter_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, b),
            nn.GELU(),
            nn.Linear(b, h),
        )
        nn.init.normal_(self.adapter[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter[-1].bias)
        self.readout = MidcutContractHeads(config)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        scale = torch.as_tensor(
            float(self.config.layer_contract_residual_scale),
            device=canvas.device,
            dtype=canvas.dtype,
        )
        adapted = canvas + scale * self.adapter(canvas)
        mid = self.readout(adapted, slices)
        out: dict[str, Tensor] = {
            key[len("midcut_") :]: value for key, value in mid.items() if key.startswith("midcut_")
        }
        out["layer_index"] = torch.as_tensor(
            self.layer_index, device=canvas.device, dtype=torch.long
        )
        return out


class SharedLayerFlowActionProbe(nn.Module):
    """Shared lightweight flow-matching action probe for V39.2.

    Each per-layer adapter first predicts a world/future latent.  This probe then
    reads only the layer-local latent summaries plus the current noisy physical
    action and flow time.  The parameters are shared across layers so lower loss
    identifies a better latent layer rather than a stronger per-layer action
    decoder.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_fm_probe_hidden)
        self.noisy_proj = nn.Linear(ph, h)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.time = TimeEmbedding(h)
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, mid),
            nn.SiLU(),
            nn.Linear(mid, ph),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        *,
        trajectory_pooled: Tensor,
        rollout_effect_pred: Tensor,
        rollout_delta_pred: Tensor,
        noisy_physical: Tensor,
        time: Tensor,
    ) -> Tensor:
        if noisy_physical.shape[:2] != trajectory_pooled.shape[:2]:
            raise ValueError(
                f"noisy_physical and trajectory_pooled horizon mismatch: "
                f"{tuple(noisy_physical.shape)} vs {tuple(trajectory_pooled.shape)}"
            )
        latent_summary = torch.cat(
            [rollout_effect_pred.mean(dim=1), rollout_delta_pred.mean(dim=1)],
            dim=-1,
        )
        latent_bias = self.latent_proj(latent_summary).to(dtype=trajectory_pooled.dtype)[:, None, :]
        t = self.time(time.to(dtype=trajectory_pooled.dtype)).to(dtype=trajectory_pooled.dtype)[
            :, None, :
        ]
        x = (
            self.noisy_proj(noisy_physical.to(dtype=trajectory_pooled.dtype))
            + trajectory_pooled
            + latent_bias
            + t
        )
        return self.net(x)


class LayerRoleScheduler(nn.Module):
    """Deterministic layer-role schedule for V40 latent/causal contracts.

    Lower layers are expected to expose action-sensitive local transition deltas;
    upper layers are expected to expose stable world/future latents.  The schedule
    returns scalar gains used both for prediction mixing and for diagnostics.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self, layer_index: int | Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        count = max(int(self.config.depth) - 1, 1)
        if torch.is_tensor(layer_index):
            idx = layer_index.to(device=device, dtype=dtype)
        else:
            idx = torch.as_tensor(float(layer_index), device=device, dtype=dtype)
        progress = (idx / float(count)).clamp(0.0, 1.0)
        c_low = float(self.config.layer_low_causal_weight)
        c_high = float(self.config.layer_high_causal_weight)
        l_low = float(self.config.layer_low_latent_weight)
        l_high = float(self.config.layer_high_latent_weight)
        causal = c_low + (c_high - c_low) * progress
        latent = l_low + (l_high - l_low) * progress
        return causal, latent


class UnifiedInterventionBlock(nn.Module):
    """One light state-action interaction block for V40.1.

    The block is deliberately not a second DiT.  It performs one cross-attention
    step from grid-local intervention state into compact context tokens, followed
    by a small FFN.  Setting ``layer_causal_feedback_depth=0`` bypasses these
    blocks and leaves the FiLM-gated delta path as the main transition operator.
    """

    def __init__(self, hidden: int, heads: int, mid: int) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden)
        self.kn = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, mid), nn.SiLU(), nn.Linear(mid, hidden))
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        update, _ = self.cross(
            self.qn(state), self.kn(context), self.kn(context), need_weights=False
        )
        state = state + update
        state = state + self.ffn(self.fn(state)).to(dtype=state.dtype)
        return state


class RecurrentMilestoneConsequenceCell(nn.Module):
    """V40.1 unified intervention-latent encoder.

    Public name is preserved for checkpoint/CLI compatibility, but the object is
    no longer a separate action-only consequence head.  It is a single
    intervention-latent head that jointly encodes:

    * layer-local rollout/world tokens;
    * current state token and state-history tokens;
    * executed-action history tokens;
    * optional trajectory/proposal canvas tokens;
    * candidate future action segments.

    It emits an action-conditioned residual latent.  The residual is supervised
    by future-latent targets, while action and state counterfactual views test
    whether the same unified head really depends on both the intervention and
    the originating state/frame context.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_consequence_hidden)
        self.gripper_frame = (
            ParsevalGripperTemporalFrame(config.action_horizon, config.gripper_field_dim)
            if str(getattr(config, "gripper_field_mode", "legacy_handcrafted"))
            == "parseval_temporal"
            else None
        )
        semantic_ph = 2 * int(config.arm_dim) + 1 if self.gripper_frame is not None else ph
        self.action_summary_dim = semantic_ph * 5 + 4
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_summary_dim),
            nn.Linear(self.action_summary_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, h),
        )
        self.step_embed = nn.Embedding(int(config.layer_consequence_steps), h)
        self.layer_embed = nn.Embedding(int(config.depth), h)
        self.memory_tokens = nn.Parameter(
            torch.randn(1, int(config.layer_causal_memory_tokens), h) * 0.02
        )
        self.context_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.action_film = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, 2 * h)
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, mid), nn.SiLU(), nn.Linear(mid, 1)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h)
        )
        self.neutral_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h)
        )
        self.policy_effect_proj = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h)
        )
        self.interaction_blocks = nn.ModuleList(
            [
                UnifiedInterventionBlock(h, int(config.num_heads), mid)
                for _ in range(int(config.layer_causal_feedback_depth))
            ]
        )
        self.effect_norm = nn.LayerNorm(h)
        self.effect_gain = nn.Parameter(
            torch.tensor(float(config.layer_consequence_initial_gain), dtype=torch.float32)
        )
        self.delta_scale = nn.Parameter(
            torch.tensor(float(config.layer_consequence_delta_scale), dtype=torch.float32)
        )
        for module in (
            self.action_encoder[-1],
            self.context_proj[-1],
            self.action_film[-1],
            self.context_gate[-1],
            self.delta_head[-1],
            self.neutral_head[-1],
            self.policy_effect_proj[-1],
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def _segment_action(self, action_physical: Tensor) -> Tensor:
        cfg = self.config
        k = int(cfg.layer_consequence_steps)
        if self.gripper_frame is not None:
            ad = int(cfg.arm_dim)
            gripper_field = action_physical[..., 2 * ad :]
            action_physical = torch.cat(
                [action_physical[..., : 2 * ad], self.gripper_frame.synthesis(gripper_field)],
                dim=-1,
            )
        b, horizon, ph = action_physical.shape
        if horizon <= 0:
            raise ValueError("action_physical horizon must be positive")
        boundaries = (
            tuple(int(value) for value in cfg.flow_jepa_action_offsets)
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else None
        )
        if boundaries is not None and (
            len(boundaries) != k or boundaries[-1] != int(horizon)
        ):
            raise ValueError(
                "Flow-DINO action segments must match window offsets and end at action_horizon"
            )
        rows: list[Tensor] = []
        previous = 0
        for step in range(k):
            if boundaries is None:
                lo = int(round(step * horizon / float(k)))
                hi = int(round((step + 1) * horizon / float(k)))
            else:
                lo = previous
                hi = boundaries[step]
                previous = hi
            hi = max(hi, lo + 1)
            hi = min(hi, horizon)
            seg = action_physical[:, lo:hi]
            mean = seg.mean(dim=1)
            first = seg[:, 0]
            last = seg[:, -1]
            delta = last - first
            std = seg.float().std(dim=1, unbiased=False).to(dtype=action_physical.dtype)
            ad = int(getattr(cfg, "arm_dim", max((ph - 2) // 2, 0)))
            if ad > 0 and 2 * ad + 2 == ph:
                # action_physical is [arm_abs, arm_delta, gripper_value, gripper_delta].
                grip_value = 2 * ad
                grip_mean = seg[..., grip_value].mean(dim=1, keepdim=True)
                grip_delta = (
                    last[:, grip_value : grip_value + 1] - first[:, grip_value : grip_value + 1]
                )
                arm = seg[..., : 2 * ad]
            else:
                g = int(cfg.gripper_dim_index)
                if g < 0:
                    g += ph
                g = min(max(g, 0), ph - 1)
                grip_mean = seg[..., g].mean(dim=1, keepdim=True)
                grip_delta = last[:, g : g + 1] - first[:, g : g + 1]
                arm = (
                    torch.cat([seg[..., :g], seg[..., g + 1 :]], dim=-1) if ph > 1 else seg[..., :0]
                )
            arm_norm = (
                arm.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype)
                if arm.numel()
                else torch.zeros(b, 1, device=action_physical.device, dtype=action_physical.dtype)
            )
            action_norm = (
                seg.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype)
            )
            rows.append(
                torch.cat(
                    [mean, first, last, delta, std, grip_mean, grip_delta, arm_norm, action_norm],
                    dim=-1,
                )
            )
        return torch.stack(rows, dim=1)

    def _compact_tokens(self, x: Tensor | None, *, max_tokens: int = 8) -> Tensor | None:
        if x is None:
            return None
        if x.ndim != 3:
            raise ValueError(f"context tokens must be [B,N,H], got {tuple(x.shape)}")
        if x.shape[1] <= max_tokens:
            return x
        # Uniform deterministic subsampling keeps the head lightweight while
        # still excluding more than a single frame/state token in counterfactuals.
        idx = torch.linspace(0, x.shape[1] - 1, steps=max_tokens, device=x.device).round().long()
        return x.index_select(1, idx)

    def _context_bank(
        self,
        *,
        base_tokens: Tensor,
        state_tokens: Tensor | None,
        state_history_tokens: Tensor | None,
        executed_tokens: Tensor | None,
        trajectory_tokens: Tensor | None,
        proposal_tokens: Tensor | None,
        action_token: Tensor,
        layer_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = base_tokens.shape[0]
        mem = self.memory_tokens.to(device=base_tokens.device, dtype=base_tokens.dtype).expand(
            b, -1, -1
        )
        parts = [
            base_tokens,
            self._compact_tokens(state_tokens, max_tokens=2),
            self._compact_tokens(state_history_tokens, max_tokens=4),
            self._compact_tokens(executed_tokens, max_tokens=4),
            self._compact_tokens(proposal_tokens, max_tokens=4),
            self._compact_tokens(trajectory_tokens, max_tokens=8),
            action_token[:, None, :],
            layer_token[:, None, :],
            mem,
        ]
        kept = [p for p in parts if p is not None]
        bank = self.context_proj(torch.cat(kept, dim=1)).to(dtype=base_tokens.dtype)
        # Pool each semantic group before averaging groups.  This prevents the
        # spatial rollout grid from numerically overwhelming the much shorter
        # state/history groups and keeps explicit context active even when the
        # optional cross-attention feedback depth is zero.
        grouped = torch.stack([part.mean(dim=1) for part in kept], dim=1)
        summary = self.context_proj(grouped).mean(dim=1).to(dtype=base_tokens.dtype)
        return bank, summary

    def _align_milestone_tokens_to_horizon(self, tokens: Tensor, horizon: int) -> Tensor:
        boundaries = (
            tuple(int(value) for value in self.config.flow_jepa_effective_window_offsets)
            if int(getattr(self.config, "flow_jepa_enabled", 0))
            else None
        )
        return _align_milestone_tokens_to_horizon(
            tokens, horizon, boundaries=boundaries
        )

    def forward(
        self,
        *,
        rollout_tokens: Tensor,
        action_physical: Tensor,
        state_tokens: Tensor | None = None,
        state_history_tokens: Tensor | None = None,
        executed_tokens: Tensor | None = None,
        trajectory_tokens: Tensor | None = None,
        proposal_tokens: Tensor | None = None,
        layer_index: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        b = int(rollout_tokens.shape[0])
        k = int(cfg.layer_consequence_steps)
        grid = int(cfg.num_cameras) * int(cfg.future_grid_size) * int(cfg.future_grid_size)
        h = int(cfg.hidden_size)
        if rollout_tokens.shape[1] != int(cfg.future_token_count):
            raise ValueError(
                f"rollout_tokens must have future_token_count={cfg.future_token_count}, got {rollout_tokens.shape[1]}"
            )
        grouped = rollout_tokens.reshape(b, int(cfg.future_anchors), grid, h)
        action_segments = self._segment_action(
            action_physical.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype)
        )
        action_embed = self.action_encoder(action_segments).to(dtype=rollout_tokens.dtype)
        step_ids = torch.arange(k, device=rollout_tokens.device)
        step_embed = self.step_embed(step_ids).to(dtype=rollout_tokens.dtype)
        if layer_index is None:
            layer_id = torch.zeros((), device=rollout_tokens.device, dtype=torch.long)
        elif torch.is_tensor(layer_index):
            layer_id = layer_index.to(device=rollout_tokens.device, dtype=torch.long).clamp(
                0, int(cfg.depth) - 1
            )
        else:
            layer_id = torch.as_tensor(
                int(layer_index), device=rollout_tokens.device, dtype=torch.long
            ).clamp(0, int(cfg.depth) - 1)
        layer_token = self.layer_embed(layer_id)[None].expand(b, -1).to(dtype=rollout_tokens.dtype)
        scale = self.delta_scale.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        gain = self.effect_gain.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        effect_state = torch.zeros(
            b, grid, h, device=rollout_tokens.device, dtype=rollout_tokens.dtype
        )
        preds: list[Tensor] = []
        deltas: list[Tensor] = []
        gates: list[Tensor] = []
        policy_tokens: list[Tensor] = []
        neutral_tokens: list[Tensor] = []
        intervene_tokens: list[Tensor] = []
        for step in range(k):
            # Validation requires one intervention step per future anchor, so
            # predictions and targets share the same temporal indexing.
            anchor = step
            base = grouped[:, anchor]
            a = action_embed[:, step] + step_embed[step][None] + layer_token
            context, context_summary = self._context_bank(
                base_tokens=base,
                state_tokens=state_tokens,
                state_history_tokens=state_history_tokens,
                executed_tokens=executed_tokens,
                trajectory_tokens=trajectory_tokens,
                proposal_tokens=proposal_tokens,
                action_token=a,
                layer_token=layer_token,
            )
            neutral = base + self.neutral_head(base).to(dtype=rollout_tokens.dtype)
            intervention = neutral + effect_state
            for block in self.interaction_blocks:
                intervention = block(intervention, context)
            joint_condition = a + context_summary
            gamma_beta = self.action_film(joint_condition).to(dtype=rollout_tokens.dtype)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            modulated = intervention * (1.0 + gamma[:, None, :]) + beta[:, None, :]
            gate_in = torch.cat(
                [modulated, joint_condition[:, None, :].expand(-1, grid, -1)], dim=-1
            )
            gate = torch.sigmoid(self.context_gate(gate_in).to(dtype=rollout_tokens.dtype))
            raw_delta = torch.tanh(self.delta_head(modulated).to(dtype=rollout_tokens.dtype))
            # V40.1 keeps the local/cumulative contract closed, but restores the
            # normalized increment used by the earlier K4/A6 branch.  The
            # unnormalized gated delta is often too small for action-shuffle
            # contrast to see; LayerNorm provides a per-token direction
            # amplifier.  Crucially, the *same* increment is logged/supervised as
            # milestone_step_delta_pred and accumulated into rollout_effect_pred,
            # so delta matching and cumulative rollout remain mathematically
            # consistent.
            local_delta = scale * gate * raw_delta
            step_delta = gain * self.effect_norm(local_delta).to(dtype=rollout_tokens.dtype)
            effect_state = effect_state + step_delta
            z_intervene = neutral + effect_state
            preds.append(effect_state)
            deltas.append(step_delta)
            gates.append(gate)
            policy_tokens.append(
                self.policy_effect_proj(z_intervene).to(dtype=rollout_tokens.dtype)
            )
            neutral_tokens.append(neutral)
            intervene_tokens.append(z_intervene)
        pred = torch.stack(preds, dim=1)
        delta_stack = torch.stack(deltas, dim=1)
        gate_stack = torch.stack(gates, dim=1)
        policy_stack = torch.stack(policy_tokens, dim=1)
        neutral_stack = torch.stack(neutral_tokens, dim=1)
        intervene_stack = torch.stack(intervene_tokens, dim=1)
        flat_pred = pred.reshape(b, k * grid, h)
        flat_delta = delta_stack.reshape(b, k * grid, h)
        flat_policy = policy_stack.reshape(b, k * grid, h)
        time_policy = _align_milestone_tokens_to_horizon(
            policy_stack.mean(dim=2),
            int(cfg.action_horizon),
            boundaries=(
                tuple(int(value) for value in cfg.flow_jepa_action_offsets)
                if int(getattr(cfg, "flow_jepa_enabled", 0))
                else None
            ),
        )
        return {
            "milestone_rollout_effect_pred": flat_pred,
            "milestone_rollout_delta_pred": flat_pred,
            "milestone_step_delta_pred": flat_delta,
            "milestone_policy_effect_tokens": flat_policy,
            "milestone_policy_time_tokens": time_policy,
            "milestone_neutral_latent_pred": neutral_stack.reshape(b, k * grid, h),
            "milestone_intervention_latent_pred": intervene_stack.reshape(b, k * grid, h),
            "milestone_gate_mean": gate_stack.detach().float().mean(),
            "milestone_step_delta_norm": delta_stack.detach().float().norm(dim=-1).mean(),
            "milestone_effect_norm": pred.detach().float().norm(dim=-1).mean(),
            "milestone_effect_std": pred.detach().float().std(unbiased=False),
            "milestone_effect_gain": gain.detach().float().abs(),
        }


def _zeros_like_scalar(reference: Tensor) -> Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


class TemporalMidcutWorldActionDiT(nn.Module):
    """V38 DiT split into a mid-cut contract trunk and a policy tail."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Evaluation-only causal probe state.  This is deliberately neither a
        # parameter nor a buffer, so it never enters checkpoints or training
        # configuration.  The probe intervenes at ownership boundaries rather
        # than scaling gradients or changing the learned forward contract.
        self._action_path_eval_intervention: str | None = None
        self._action_path_eval_apply_count = 0
        self._action_path_eval_metrics: dict[str, float] = {}
        h = int(config.hidden_size)
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        self.visual_memory = DenseVisualMemory(config)
        self.rollout_codec = RolloutTargetCodec(config)
        self.flow_dino_evidence = (
            FlowDINOEvidenceEncoder(config) if int(getattr(config, "flow_jepa_enabled", 0)) else None
        )
        self.goal_resampler = (
            GoalTokenResampler(
                language_dim=int(config.goal_language_dim),
                hidden=h,
                goal_tokens=int(config.goal_token_count),
                heads=int(config.num_heads),
                depth=int(config.goal_resampler_depth),
                expansion=float(config.ffn_expansion),
            )
            if int(getattr(config, "goal_conditioning_enabled", 0))
            else None
        )
        self.stateless_phase_adapter = (
            StatelessPhaseAdapter(
                h,
                int(getattr(config, "stateless_phase_count", 4)),
            )
            if int(getattr(config, "stateless_phase_enabled", 0))
            else None
        )
        self.phase_world_query_proj = (
            nn.Linear(h, h, bias=False)
            if (
                self.stateless_phase_adapter is not None
                and int(getattr(config, "role_attnres_world_to_policy", 0))
            )
            else None
        )
        self.condition_world_query_proj = (
            nn.Linear(h, h, bias=False)
            if (
                self.stateless_phase_adapter is not None
                and int(getattr(config, "role_attnres_world_to_policy", 0))
            )
            else None
        )
        self.phase_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks))
                ]
            )
            if (
                self.stateless_phase_adapter is not None
                and int(getattr(config, "flow_jepa_role_hierarchy", 0))
            )
            else None
        )
        self.condition_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks))
                ]
            )
            if self.phase_world_block_query_proj is not None
            else None
        )
        if self.flow_dino_evidence is not None:
            # The new path owns both online visual compilation and future-query
            # initialization.  Keep legacy modules in the state dict for old
            # checkpoints, but do not allocate gradients for unused outputs.
            self.visual_memory.requires_grad_(False)
            self.rollout_codec.requires_grad_(False)
        self.seed = UnifiedCanvasSeed(config)
        self.time = TimeEmbedding(h)
        self.content_mod = nn.Sequential(
            (
                AffineVarianceFlooredCenteredNorm(
                    2 * h,
                    float(
                        getattr(
                            config, "flow_jepa_routing_norm_floor", 0.25
                        )
                    ),
                    affine_maximum=4.0,
                )
                if self.complete_numerical_contract
                else nn.LayerNorm(2 * h)
            ),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        nn.init.normal_(self.content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.content_mod[-1].bias)
        self.content_mod_scale = nn.Parameter(torch.tensor(0.10))
        if int(getattr(config, "flow_jepa_role_hierarchy", 0)):
            block_roles = (
                ["grounding"] * int(config.flow_jepa_grounding_blocks)
                + ["world"] * int(config.flow_jepa_world_blocks)
                + ["policy"] * int(config.flow_jepa_policy_blocks)
            )
        else:
            block_roles = ["shared"] * int(config.depth)
        if len(block_roles) != int(config.depth):
            raise ValueError("DiT block-role schedule must match configured depth")
        self.block_roles = tuple(block_roles)
        self.blocks = nn.ModuleList(
            [
                TemporalDynamicsBoundDiTBlock(config, role=role)
                for role in self.block_roles
            ]
        )
        role_route_dim = int(getattr(config, "role_attnres_key_dim", 32))
        role_value_rms = (
            float(getattr(config, "role_attnres_max_value_rms", 1.0))
            if int(getattr(config, "role_residual_amplitude_contract", 0))
            else None
        )
        role_norm_floor = (
            float(getattr(config, "flow_jepa_routing_norm_floor", 0.25))
            if int(getattr(config, "flow_jepa_variance_safe_routing", 0))
            else None
        )
        self.ground_to_world_attnres = (
            RoleDeltaAttnRes(
                h,
                role_route_dim,
                max_sources=int(config.flow_jepa_grounding_blocks),
                max_value_rms=role_value_rms,
                normalization_floor=role_norm_floor,
            )
            if int(getattr(config, "role_attnres_ground_to_world", 0))
            else None
        )
        action_anchor_count = (
            len(tuple(int(value) for value in config.flow_jepa_action_offsets))
            if int(getattr(config, "flow_jepa_enabled", 0))
            else int(config.future_anchors)
        )
        self.world_to_policy_far_anchor_count = max(
            int(config.future_anchors) - int(action_anchor_count),
            0,
        )
        self.interval_stage_typed_value = bool(
            int(getattr(config, "flow_jepa_interval_stage_typed_value", 0))
        )
        self.world_to_policy_attnres = (
            RoleDeltaAttnRes(
                h,
                role_route_dim,
                max_sources=(
                    (
                        int(config.flow_jepa_world_blocks)
                        + 1
                        + int(self.interval_stage_typed_value)
                    )
                    * int(config.num_cameras)
                    * (1 + int(self.world_to_policy_far_anchor_count))
                ),
                max_value_rms=role_value_rms,
                normalization_floor=role_norm_floor,
            )
            if int(getattr(config, "role_attnres_world_to_policy", 0))
            else None
        )
        self.late_raw_detail_reader = (
            LateRawDetailPolicyReader(config)
            if int(getattr(config, "flow_jepa_late_policy_detail", 0))
            else None
        )
        final_decoder = str(getattr(config, "final_action_decoder", "legacy"))
        self.terminal_policy_layer_contracts_only = bool(
            final_decoder == "evidence_latent_mmdit_action"
            and int(getattr(config, "flow_jepa_role_hierarchy", 0))
            and int(getattr(config, "flow_jepa_strict_role_visual_path", 0))
        )
        self.midcut_norm = nn.LayerNorm(h)
        self.midcut_heads = MidcutContractHeads(config)
        if int(config.layer_contract_adapters):
            self.layer_contract_heads = nn.ModuleList(
                [LayerContractAdapterHeads(config, layer_index=i) for i in range(int(config.depth))]
            )
        else:
            self.layer_contract_heads = nn.ModuleList()
        self.layer_fm_probe = (
            SharedLayerFlowActionProbe(config) if int(config.layer_shared_fm_probe) else None
        )
        self.layer_role_scheduler = LayerRoleScheduler(config)
        self.layer_consequence_cell = (
            RecurrentMilestoneConsequenceCell(config)
            if int(config.layer_recurrent_consequence)
            else None
        )
        if self.terminal_policy_layer_contracts_only:
            # V103 is a single deployable path. The historical mid-cut probe
            # and G/W layer readouts are not auxiliary objectives and do not
            # feed the final Evidence-MMDiT. Only the two terminal P adapters
            # remain as layer evidence. Within them, keep just the final
            # event/delta readout when no recurrent consequence cell owns
            # event evidence; all other legacy probe heads stay frozen.
            self.midcut_norm.requires_grad_(False)
            self.midcut_heads.requires_grad_(False)
            policy_start = int(config.depth) - int(config.flow_jepa_policy_blocks)
            for layer_index, head in enumerate(self.layer_contract_heads):
                if layer_index < policy_start:
                    head.requires_grad_(False)
                    continue
                head.readout.requires_grad_(False)
            if (
                self.layer_consequence_cell is None
                and len(self.layer_contract_heads) > 0
            ):
                final_readout = self.layer_contract_heads[-1].readout
                final_readout.future_gain.requires_grad_(True)
                final_readout.rollout_delta_head.requires_grad_(True)
                final_readout.event_head.requires_grad_(True)
        self.final_norm = (
            AffineVarianceFlooredCenteredNorm(
                h,
                float(getattr(config, "flow_jepa_routing_norm_floor", 0.25)),
                affine_maximum=4.0,
            )
            if self.complete_numerical_contract
            else nn.LayerNorm(h)
        )
        self.direct_physical_head = CanvasPhysicalVelocityHead(config)
        self.rollout_residual_head = RolloutActionResidualHead(config)
        self.controlled_dynamics = ControlledResidualLatentDynamics(config)
        self.event_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.hierarchical_mmdit_action_decoder: HierarchicalMMDiTActionDecoder | None = None
        self.evidence_latent_mmdit_action_decoder: EvidenceLatentMMDiTActionDecoder | None = None
        if final_decoder == "residual_action_flow":
            self.residual_action_flow_denoiser = V37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "layered_residual_action_flow":
            self.residual_action_flow_denoiser = LayeredV37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_main_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = HierarchicalLatentMainActionDecoder(config)
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = LatentCVAEActionDecoder(config)
        elif final_decoder == "adaptive_recurrent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = AdaptiveRecurrentCVAEActionDecoder(config)
        elif final_decoder == "hierarchical_mmdit_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
            self.hierarchical_mmdit_action_decoder = HierarchicalMMDiTActionDecoder(config)
        elif final_decoder == "evidence_latent_mmdit_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
            self.hierarchical_mmdit_action_decoder = None
            self.evidence_latent_mmdit_action_decoder = EvidenceLatentMMDiTActionDecoder(config)
        else:
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        if (
            self.latent_cvae_action_decoder is not None
            or self.latent_main_action_decoder is not None
            or self.hierarchical_mmdit_action_decoder is not None
            or self.evidence_latent_mmdit_action_decoder is not None
        ):
            # These readers belong to the legacy action tower. Keep the modules
            # for checkpoint compatibility and the parameter-free pooled()
            # helper, but do not allocate gradients/optimizer state for outputs
            # that the complete latent decoder never consumes.
            self.direct_physical_head.requires_grad_(False)
            self.rollout_residual_head.requires_grad_(False)
            self.motion_probe.requires_grad_(False)
        if (
            self.terminal_policy_layer_contracts_only
            and self.layer_consequence_cell is None
        ):
            # The final terminal P readout supplies event evidence directly;
            # the generic fallback probe is unreachable in this contract.
            self.event_probe.requires_grad_(False)

    def _mod_embed(
        self,
        canvas: Tensor,
        visual_memory: Tensor,
        time_emb: Tensor,
        slices: dict[str, slice],
        *,
        role: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Modulation is shared by every canvas role.  Letting action/stage/
        # window tokens enter this global mean would bypass the directed
        # attention mask on the next block (window -> modulation -> stage).
        # Compile it only from deploy-safe observed context and registers.
        strict_policy = bool(
            int(getattr(self.config, "flow_jepa_strict_role_visual_path", 0))
            and str(role) == "policy"
        )
        if self.flow_dino_evidence is None:
            canvas_summary = canvas.mean(dim=1)
            modulation_source = visual_memory.mean(dim=1)
        else:
            clean_names = [
                "task",
                "state",
                "state_history",
                "executed",
                "registers",
            ]
            if str(role) != "grounding":
                clean_names.append("proposal")
            if strict_policy:
                # Policy modulation may read the world chart produced by the
                # upstream grounding/world blocks, but not the original DINO or
                # raw visual bank.  Otherwise visual cross-attention is merely
                # hidden inside AdaLN modulation.
                clean_names.append("rollout")
            clean_canvas = torch.cat(
                [canvas[:, slices[name]] for name in clean_names],
                dim=1,
            )
            canvas_summary = clean_canvas.mean(dim=1)
            modulation_source = (
                canvas[:, slices["rollout"]].mean(dim=1)
                if strict_policy
                else visual_memory.mean(dim=1)
            )
        summary = torch.cat([canvas_summary, modulation_source], dim=-1)
        content_delta = self.content_mod(summary) * self.content_mod_scale.to(
            device=canvas.device, dtype=canvas.dtype
        )
        return time_emb + content_delta, content_delta, time_emb

    def encode_visual_context(
        self, visual: Tensor, *, raw_visual: Tensor | None = None
    ) -> FlowDINOEvidencePack | None:
        """Compile online visual evidence once for real/counterfactual passes."""

        if self.flow_dino_evidence is None:
            return None
        return self.flow_dino_evidence(visual, raw_visual=raw_visual)

    def set_action_path_eval_intervention(self, mode: str) -> None:
        """Select a transient V101 ownership-boundary intervention.

        ``world_residual_*`` is applied after the final world block and before
        policy blocks.  It preserves the fixed grounding output at every
        anchor/camera/spatial slot and changes only the residual written by the
        world blocks.  Anchor-only and spatial-only modes separate temporal
        organization from xy organization. ``policy_*`` is applied only to the
        final policy workspace entering the native action decoder, leaving
        rollout/evidence inputs intact.
        """

        normalized = str(mode).strip().lower().replace("-", "_")
        allowed = {
            "none",
            "world_residual_zero",
            "world_residual_anchor_shuffle",
            "world_residual_spatial_shuffle",
            "world_residual_spatiotemporal_shuffle",
            "policy_zero",
            "policy_temporal_shuffle",
            "phase_zero",
            "phase_batch_shuffle",
            "condition_query_zero",
            "horizon_address_zero",
            "horizon_address_shuffle",
            "address_g1_zero",
            "address_g1_shuffle",
            "address_g2_zero",
            "address_g2_shuffle",
            "address_g3_zero",
            "address_g3_shuffle",
            "interval_stage_zero",
            "interval_stage_shuffle",
            "grounding_entry_zero",
            "grounding_entry_shuffle",
            "world_to_policy_zero",
            "world_to_policy_shuffle",
            "w2p_far_context_zero",
            "w2p_far_context_shuffle",
            "bottom_far_rollout_zero",
            "bottom_far_rollout_shuffle",
            "all_far_context_zero",
            "all_far_context_shuffle",
            "protected_detail_zero",
            "protected_detail_shuffle",
        }
        for prefix, count in (
            ("g", int(getattr(self.config, "flow_jepa_grounding_blocks", 0))),
            ("w", int(getattr(self.config, "flow_jepa_world_blocks", 0))),
            ("p", int(getattr(self.config, "flow_jepa_policy_blocks", 0))),
        ):
            for index in range(1, count + 1):
                allowed.add(f"{prefix}{index}_zero")
                allowed.add(f"{prefix}{index}_shuffle")
        if normalized not in allowed:
            raise ValueError(
                "action-path intervention must be one of "
                "none/world_residual_zero/world_residual_anchor_shuffle/"
                "world_residual_spatial_shuffle/"
                "world_residual_spatiotemporal_shuffle/"
                "policy_zero/policy_temporal_shuffle/phase_zero/"
                "phase_batch_shuffle/condition_query_zero/horizon_address_zero/"
                "horizon_address_shuffle/address_g1..g3_zero/shuffle or one typed "
                "g1..g3/grounding_entry/w1..w3/world_to_policy/p1/p2/"
                "interval_stage/w2p_far_context/bottom_far_rollout/all_far_context/"
                "protected_detail zero/shuffle mode"
            )
        if self.training:
            raise RuntimeError("action-path intervention is evaluation-only")
        if not (
            int(getattr(self.config, "flow_jepa_role_hierarchy", 0))
            and int(getattr(self.config, "flow_jepa_strict_role_visual_path", 0))
        ):
            raise RuntimeError(
                "action-path intervention requires the strict Flow-JEPA role hierarchy"
            )
        self._action_path_eval_intervention = normalized
        self._action_path_eval_apply_count = 0
        self._action_path_eval_metrics = {}

    def clear_action_path_eval_intervention(self) -> None:
        self._action_path_eval_intervention = None
        self._action_path_eval_apply_count = 0
        self._action_path_eval_metrics = {}

    def action_path_eval_intervention_state(
        self,
    ) -> dict[str, str | int | float]:
        return {
            "mode": (
                "disabled"
                if self._action_path_eval_intervention is None
                else self._action_path_eval_intervention
            ),
            "apply_count": int(self._action_path_eval_apply_count),
            **self._action_path_eval_metrics,
        }

    def _record_action_path_route_metrics(
        self, *metric_sources: dict[str, Tensor] | None
    ) -> None:
        if self._action_path_eval_intervention is None:
            return
        for source in metric_sources:
            if source is None:
                continue
            for key, value in source.items():
                if not (
                    isinstance(value, Tensor)
                    and int(value.numel()) == 1
                    and key.startswith(
                        (
                            "attnres_",
                            "evidence_policy_delta_attnres_",
                            "evidence_protected_detail_basis_",
                        )
                    )
                ):
                    continue
                self._action_path_eval_metrics[key] = float(
                    value.detach().float().cpu()
                )

    def _intervene_query_contexts(
        self,
        phase_context: Tensor | None,
        condition_query_context: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        mode = self._action_path_eval_intervention
        if mode not in {
            "phase_zero",
            "phase_batch_shuffle",
            "condition_query_zero",
        }:
            return phase_context, condition_query_context
        if phase_context is None or condition_query_context is None:
            raise RuntimeError("query-context intervention has no active phase adapter")
        self._action_path_eval_apply_count += 1
        if mode == "phase_zero":
            self._action_path_eval_metrics["phase_context_delta_norm"] = float(
                phase_context.detach().float().norm(dim=-1).mean().cpu()
            )
            return torch.zeros_like(phase_context), condition_query_context
        if mode == "condition_query_zero":
            self._action_path_eval_metrics[
                "condition_query_context_delta_norm"
            ] = float(
                condition_query_context.detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
            return phase_context, torch.zeros_like(condition_query_context)
        if int(phase_context.shape[0]) > 1:
            intervened_phase = phase_context.roll(shifts=1, dims=0)
        else:
            # A one-sample smoke still receives a deterministic mismatch
            # instead of silently becoming an identity intervention.
            intervened_phase = phase_context.roll(
                shifts=max(int(phase_context.shape[-1]) // 2, 1),
                dims=-1,
            )
        self._action_path_eval_metrics["phase_context_delta_norm"] = float(
            (intervened_phase - phase_context)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return intervened_phase, condition_query_context

    def _intervene_named_role_values(
        self,
        values: list[Tensor],
        source_names: tuple[str, ...],
    ) -> list[Tensor]:
        mode = self._action_path_eval_intervention
        suffix = (
            "_zero"
            if mode is not None and mode.endswith("_zero")
            else "_shuffle"
            if mode is not None and mode.endswith("_shuffle")
            else None
        )
        if suffix is None:
            return values
        assert mode is not None
        target = mode[: -len(suffix)]
        # The interval-stage intervention is applied to the spatial W write
        # before its typed xy-mean is constructed.  Reapplying it here would
        # shuffle that one source twice and break coarse/typed consistency.
        if target == "interval_stage":
            return values
        if target not in source_names:
            return values
        self._action_path_eval_apply_count += 1
        index = source_names.index(target)
        updated = list(values)
        original = updated[index]
        if suffix == "_zero":
            intervened = torch.zeros_like(original)
        elif int(original.shape[0]) > 1:
            intervened = original.roll(shifts=1, dims=0)
        else:
            intervened = original.roll(
                shifts=max(int(original.shape[1]) // 2, 1),
                dims=1,
            )
        updated[index] = intervened
        self._action_path_eval_metrics[f"{target}_delta_norm"] = float(
            (intervened - original)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return updated

    def _intervene_world_rollout(
        self,
        rollout: Tensor,
        *,
        world_entry_rollout: Tensor,
    ) -> Tensor:
        mode = self._action_path_eval_intervention
        if mode not in {
            "world_residual_zero",
            "world_residual_anchor_shuffle",
            "world_residual_spatial_shuffle",
            "world_residual_spatiotemporal_shuffle",
        }:
            return rollout
        if tuple(world_entry_rollout.shape) != tuple(rollout.shape):
            raise RuntimeError(
                "world-entry and world-output rollout tensors must have identical shapes"
            )
        self._action_path_eval_apply_count += 1
        # Keep the grounding output and every slot's anchor/camera/spatial
        # identity fixed.  Only the update written by the world blocks is
        # removed or deliberately attached to the wrong anchor/spatial slot.
        world_residual = rollout - world_entry_rollout
        if mode == "world_residual_zero":
            self._action_path_eval_metrics["world_residual_delta_norm"] = float(
                world_residual.detach().float().norm(dim=-1).mean().cpu()
            )
            return world_entry_rollout
        cfg = self.config
        anchors = int(cfg.future_anchors)
        cameras = int(cfg.num_cameras)
        grid = int(cfg.future_grid_size)
        expected = anchors * cameras * grid * grid
        if int(rollout.shape[1]) != expected:
            raise RuntimeError(
                "world intervention expected "
                f"{expected} rollout tokens, got {int(rollout.shape[1])}"
            )
        grouped = world_residual.reshape(
            int(rollout.shape[0]), anchors, cameras, grid, grid, int(rollout.shape[-1])
        )
        # Camera identity remains fixed.  The residual is misaligned while the
        # grounding/position seed at every destination slot remains untouched.
        if mode in {
            "world_residual_anchor_shuffle",
            "world_residual_spatiotemporal_shuffle",
        }:
            grouped = grouped.roll(shifts=1, dims=1)
        if mode in {
            "world_residual_spatial_shuffle",
            "world_residual_spatiotemporal_shuffle",
        }:
            grouped = grouped.roll(shifts=max(grid // 2, 1), dims=3)
            grouped = grouped.roll(shifts=max(grid // 3, 1), dims=4)
        intervened = world_entry_rollout + grouped.reshape_as(rollout)
        self._action_path_eval_metrics["world_residual_delta_norm"] = float(
            (intervened - rollout)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return intervened

    def _intervene_policy_workspace(self, workspace: Tensor) -> Tensor:
        mode = self._action_path_eval_intervention
        if mode not in {"policy_zero", "policy_temporal_shuffle"}:
            return workspace
        self._action_path_eval_apply_count += 1
        if mode == "policy_zero":
            self._action_path_eval_metrics["policy_workspace_delta_norm"] = float(
                workspace.detach().float().norm(dim=-1).mean().cpu()
            )
            return torch.zeros_like(workspace)
        cfg = self.config
        horizon = int(cfg.action_horizon)
        basis = int(cfg.action_basis_tokens)
        expected = horizon * basis
        if int(workspace.shape[1]) != expected:
            raise RuntimeError(
                "policy intervention expected "
                f"{expected} workspace tokens, got {int(workspace.shape[1])}"
            )
        grouped = workspace.reshape(
            int(workspace.shape[0]), horizon, basis, int(workspace.shape[-1])
        )
        # Preserve values and basis identity but attach them to the wrong
        # action horizon.  This directly tests whether temporal workspace
        # correspondence, rather than mere non-zero energy, matters.
        grouped = grouped.roll(shifts=max(horizon // 2, 1), dims=1)
        intervened = grouped.reshape_as(workspace)
        self._action_path_eval_metrics["policy_workspace_delta_norm"] = float(
            (intervened - workspace)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return intervened

    def _intervene_interval_stage_rollout(
        self,
        base_rollout: Tensor,
        refined_rollout: Tensor,
    ) -> Tensor:
        """Remove or mismatch only the V106 W->P bounded interval write."""

        mode = self._action_path_eval_intervention
        if mode not in {"interval_stage_zero", "interval_stage_shuffle"}:
            return refined_rollout
        if tuple(base_rollout.shape) != tuple(refined_rollout.shape):
            raise RuntimeError(
                "interval-stage intervention requires aligned base/refined rollouts"
            )
        self._action_path_eval_apply_count += 1
        stage_write = refined_rollout - base_rollout
        if mode == "interval_stage_zero":
            intervened_write = torch.zeros_like(stage_write)
        elif int(stage_write.shape[0]) > 1:
            intervened_write = stage_write.roll(shifts=1, dims=0)
        else:
            cfg = self.config
            grouped = stage_write.reshape(
                1,
                int(cfg.future_anchors),
                int(cfg.num_cameras),
                int(cfg.future_grid_size),
                int(cfg.future_grid_size),
                int(stage_write.shape[-1]),
            )
            intervened_write = grouped.roll(shifts=1, dims=1).reshape_as(
                stage_write
            )
        self._action_path_eval_metrics[
            "interval_stage_intervention_delta_norm"
        ] = float(
            (intervened_write - stage_write)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return base_rollout + intervened_write

    def _intervene_online_horizon_address(
        self,
        base_rollout: Tensor,
        refined_rollout: Tensor,
    ) -> Tensor:
        """Remove or episode-mismatch only the owned V108 address write."""

        mode = self._action_path_eval_intervention
        if mode not in {"horizon_address_zero", "horizon_address_shuffle"}:
            return refined_rollout
        if tuple(base_rollout.shape) != tuple(refined_rollout.shape):
            raise RuntimeError(
                "horizon-address intervention requires aligned base/refined rollouts"
            )
        self._action_path_eval_apply_count += 1
        address_write = refined_rollout - base_rollout
        if mode == "horizon_address_zero":
            intervened_write = torch.zeros_like(address_write)
        elif int(address_write.shape[0]) > 1:
            intervened_write = address_write.roll(shifts=1, dims=0)
        else:
            cfg = self.config
            grouped = address_write.reshape(
                1,
                int(cfg.future_anchors),
                int(cfg.num_cameras),
                int(cfg.future_grid_size),
                int(cfg.future_grid_size),
                int(address_write.shape[-1]),
            )
            intervened_write = grouped.roll(shifts=1, dims=1).reshape_as(
                address_write
            )
        self._action_path_eval_metrics[
            "horizon_address_intervention_delta_norm"
        ] = float(
            (intervened_write - address_write)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return base_rollout + intervened_write

    @staticmethod
    def _role_route_metrics(
        prefix: str,
        metrics: dict[str, Tensor],
        source_names: tuple[str, ...],
    ) -> dict[str, Tensor]:
        out = {
            f"attnres_{prefix}_{name}": value
            for name, value in metrics.items()
            if name != "source_mass"
        }
        source_mass = metrics.get("source_mass")
        if isinstance(source_mass, Tensor):
            if int(source_mass.numel()) != len(source_names):
                raise RuntimeError("role-route source metrics lost their schema")
            for index, name in enumerate(source_names):
                out[f"attnres_{prefix}_source_mass_{name}"] = source_mass[index]
        return out

    def _apply_ground_to_world_bridge(
        self,
        canvas: Tensor,
        slices: dict[str, slice],
        grounding_deltas: list[Tensor],
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.ground_to_world_attnres is None:
            raise RuntimeError("ground-to-world bridge is disabled")
        expected = int(self.config.flow_jepa_grounding_blocks)
        if len(grounding_deltas) != expected:
            raise RuntimeError(
                f"ground-to-world bridge expected {expected} deltas, "
                f"got {len(grounding_deltas)}"
            )
        cfg = self.config
        rollout_region = slices["rollout"]
        rollout = canvas[:, rollout_region]
        batch = int(rollout.shape[0])
        anchors = int(cfg.future_anchors)
        cameras = int(cfg.num_cameras)
        grid = int(cfg.future_grid_size)
        hidden = int(rollout.shape[-1])
        query = rollout.reshape(
            batch, anchors, cameras, grid, grid, hidden
        ).mean(dim=(3, 4))
        source_names = tuple(f"g{index + 1}" for index in range(expected))
        grounding_deltas = self._intervene_named_role_values(
            grounding_deltas, source_names
        )
        values = torch.stack(grounding_deltas, dim=-2)
        routed, route_metrics = self.ground_to_world_attnres(query, values)
        scale = routed.new_tensor(
            float(getattr(cfg, "role_attnres_ground_to_world_scale", 0.10))
        )
        structured_update = scale * routed
        expanded_update = (
            structured_update[:, :, :, None, None]
            .expand(-1, -1, -1, grid, grid, -1)
            .reshape_as(rollout)
        )
        updated_rollout = rollout + expanded_update
        canvas = torch.cat(
            (
                canvas[:, : int(rollout_region.start)],
                updated_rollout,
                canvas[:, int(rollout_region.stop) :],
            ),
            dim=1,
        )
        metrics = self._role_route_metrics(
            "ground_to_world",
            route_metrics,
            source_names,
        )
        metrics["attnres_ground_to_world_anchor_route_std"] = route_metrics[
            "query_axis_1_route_std"
        ]
        metrics["attnres_ground_to_world_camera_route_std"] = route_metrics[
            "query_axis_2_route_std"
        ]
        metrics["attnres_ground_to_world_fixed_scale"] = scale.detach().float()
        metrics["attnres_ground_to_world_structured_update_norm"] = (
            structured_update.detach().float().norm(dim=-1).mean()
        )
        metrics["attnres_ground_to_world_approved_value_norm"] = (
            routed.detach().float().norm(dim=-1).mean()
        )
        # The carrier write keeps its conservative fixed step, but the typed
        # value handed to the next ownership boundary must not accumulate that
        # scale again. Otherwise G evidence receives G->W, W->P, and P->bottom
        # multipliers while a P delta receives only the final multiplier.
        return canvas, routed, metrics

    def _align_anchor_camera_to_horizon(self, value: Tensor) -> Tensor:
        cfg = self.config
        if value.ndim != 4:
            raise ValueError("role delta must retain [B,anchor,camera,H]")
        batch, anchors, cameras, hidden = value.shape
        boundaries = (
            tuple(int(item) for item in cfg.flow_jepa_action_offsets)
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else None
        )
        selected = value[:, : len(boundaries)] if boundaries is not None else value
        return _align_milestone_tokens_to_horizon(
            selected.permute(0, 2, 1, 3).reshape(
                int(batch) * int(cameras),
                int(selected.shape[1]),
                int(hidden),
            ),
            int(cfg.action_horizon),
            boundaries=boundaries,
        ).reshape(
            int(batch), int(cameras), int(cfg.action_horizon), int(hidden)
        ).permute(0, 2, 1, 3)

    def _far_anchor_camera_context(self, value: Tensor) -> Tensor:
        """Keep non-action anchors as context without assigning action time.

        The action-aligned prefix (4/12/24 in V103) is handled by
        ``_align_anchor_camera_to_horizon``. Later anchors (currently +48)
        remain separate ``[B,far_anchor,camera,H]`` values. They may condition
        every action query downstream, but are never relabelled as a step
        inside the 24-step deploy horizon.
        """

        if value.ndim != 4:
            raise ValueError("role delta must retain [B,anchor,camera,H]")
        cfg = self.config
        action_anchor_count = (
            len(tuple(int(item) for item in cfg.flow_jepa_action_offsets))
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else int(value.shape[1])
        )
        if int(value.shape[1]) < int(action_anchor_count):
            raise ValueError(
                "role delta has fewer anchors than the action-aligned prefix"
            )
        return value[:, int(action_anchor_count) :]

    def _intervene_far_anchor_context(
        self,
        far_values: list[Tensor],
    ) -> list[Tensor]:
        mode = self._action_path_eval_intervention
        if mode not in {
            "w2p_far_context_zero",
            "w2p_far_context_shuffle",
            "all_far_context_zero",
            "all_far_context_shuffle",
        }:
            return far_values
        if len(far_values) <= 0 or int(far_values[0].shape[1]) <= 0:
            raise RuntimeError(
                "far-context intervention requires at least one non-action anchor"
            )
        self._action_path_eval_apply_count += 1
        updated: list[Tensor] = []
        deltas: list[Tensor] = []
        for original in far_values:
            if mode in {"w2p_far_context_zero", "all_far_context_zero"}:
                intervened = torch.zeros_like(original)
            elif int(original.shape[0]) > 1:
                intervened = original.roll(shifts=1, dims=0)
            else:
                intervened = original.roll(
                    shifts=max(int(original.shape[-1]) // 2, 1),
                    dims=-1,
                )
            updated.append(intervened)
            deltas.append(
                (intervened - original).detach().float().norm(dim=-1).mean()
            )
        self._action_path_eval_metrics["w2p_far_context_delta_norm"] = float(
            torch.stack(deltas).mean().cpu()
        )
        return updated

    def _intervene_bottom_far_rollout(self, rollout: Tensor) -> Tensor:
        """Intervene on +48 only at the bottom Evidence-MMDiT rollout input."""

        mode = self._action_path_eval_intervention
        if mode not in {
            "bottom_far_rollout_zero",
            "bottom_far_rollout_shuffle",
            "all_far_context_zero",
            "all_far_context_shuffle",
        }:
            return rollout
        cfg = self.config
        anchors = int(cfg.future_anchors)
        cameras = int(cfg.num_cameras)
        grid = int(cfg.future_grid_size)
        hidden = int(rollout.shape[-1])
        expected = anchors * cameras * grid * grid
        if rollout.ndim != 3 or int(rollout.shape[1]) != expected:
            raise RuntimeError(
                "bottom far-rollout intervention lost the rollout chart schema"
            )
        action_anchor_count = (
            len(tuple(int(item) for item in cfg.flow_jepa_action_offsets))
            if int(getattr(cfg, "flow_jepa_enabled", 0))
            else anchors
        )
        if int(action_anchor_count) >= anchors:
            raise RuntimeError(
                "bottom far-rollout intervention requires a non-action anchor"
            )
        grouped = rollout.reshape(
            int(rollout.shape[0]),
            anchors,
            cameras,
            grid,
            grid,
            hidden,
        )
        local = grouped[:, :action_anchor_count]
        original_far = grouped[:, action_anchor_count:]
        self._action_path_eval_apply_count += 1
        if mode in {"bottom_far_rollout_zero", "all_far_context_zero"}:
            intervened_far = torch.zeros_like(original_far)
        elif int(original_far.shape[0]) > 1:
            intervened_far = original_far.roll(shifts=1, dims=0)
        else:
            intervened_far = original_far.roll(
                shifts=max(int(original_far.shape[-1]) // 2, 1),
                dims=-1,
            )
        self._action_path_eval_metrics["bottom_far_rollout_delta_norm"] = float(
            (intervened_far - original_far)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return torch.cat((local, intervened_far), dim=1).reshape_as(rollout)

    def _world_to_policy_source_candidates(
        self,
        value: Tensor,
        far_value: Tensor,
        source_name: str,
    ) -> tuple[Tensor, tuple[str, ...]]:
        """Expand one typed world delta into local and far context candidates."""

        cfg = self.config
        horizon = int(cfg.action_horizon)
        cameras = int(cfg.num_cameras)
        local = self._align_anchor_camera_to_horizon(value)
        if int(local.shape[2]) != cameras:
            raise ValueError("world-to-policy local camera axis is invalid")
        candidates = [local[:, :, camera] for camera in range(cameras)]
        names = [
            f"{source_name}_camera{camera}" for camera in range(cameras)
        ]
        if far_value.ndim != 4:
            raise ValueError(
                "far world-to-policy values must be [B,far_anchor,camera,H]"
            )
        if (
            int(far_value.shape[0]) != int(value.shape[0])
            or int(far_value.shape[2]) != cameras
            or int(far_value.shape[3]) != int(value.shape[3])
        ):
            raise ValueError("far world-to-policy values do not match local values")
        for far_index in range(int(far_value.shape[1])):
            for camera in range(cameras):
                candidates.append(
                    far_value[:, far_index, camera][:, None].expand(
                        -1, horizon, -1
                    )
                )
                names.append(
                    f"{source_name}_far{far_index + 1}_camera{camera}"
                )
        return torch.stack(candidates, dim=2), tuple(names)

    def _apply_world_to_policy_bridge(
        self,
        canvas: Tensor,
        slices: dict[str, slice],
        world_deltas: list[Tensor],
        source_names: tuple[str, ...],
        phase_context: Tensor | None = None,
        condition_query_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.world_to_policy_attnres is None:
            raise RuntimeError("world-to-policy bridge is disabled")
        if len(world_deltas) <= 0 or len(world_deltas) != len(source_names):
            raise RuntimeError("world-to-policy delta bank is empty or mislabelled")
        cfg = self.config
        horizon = int(cfg.action_horizon)
        basis = int(cfg.action_basis_tokens)
        cameras = int(cfg.num_cameras)
        hidden = int(canvas.shape[-1])
        trajectory_region = slices["trajectory"]
        trajectory = canvas[:, trajectory_region].reshape(
            int(canvas.shape[0]), horizon, basis, hidden
        )
        rollout = canvas[:, slices["rollout"]].reshape(
            int(canvas.shape[0]),
            int(cfg.future_anchors),
            cameras,
            int(cfg.future_grid_size),
            int(cfg.future_grid_size),
            hidden,
        ).mean(dim=(3, 4))
        world_query = self._align_anchor_camera_to_horizon(rollout).mean(dim=2)
        query = trajectory + world_query[:, :, None]
        phase_query_delta = trajectory.new_zeros(
            int(canvas.shape[0]), hidden
        )
        condition_query_delta = trajectory.new_zeros(
            int(canvas.shape[0]), hidden
        )
        if self.phase_world_query_proj is not None:
            if (
                phase_context is None
                or tuple(phase_context.shape) != (int(canvas.shape[0]), hidden)
            ):
                raise ValueError(
                    "stateless world-to-policy route requires [B,H] phase context"
                )
            phase_query_delta = float(
                getattr(cfg, "stateless_phase_query_scale", 0.10)
            ) * self.phase_world_query_proj(
                phase_context.to(device=query.device, dtype=query.dtype)
            )
            query = query + phase_query_delta[:, None, None]
            if (
                self.condition_world_query_proj is None
                or condition_query_context is None
                or tuple(condition_query_context.shape)
                != (int(canvas.shape[0]), hidden)
            ):
                raise ValueError(
                    "goal/history world route requires [B,H] condition context"
                )
            condition_query_delta = float(
                getattr(cfg, "stateless_phase_query_scale", 0.10)
            ) * self.condition_world_query_proj(
                condition_query_context.to(
                    device=query.device, dtype=query.dtype
                )
            )
            query = query + condition_query_delta[:, None, None]
        elif phase_context is not None:
            raise ValueError(
                "phase context was supplied while world phase routing is disabled"
            )
        elif condition_query_context is not None:
            raise ValueError(
                "condition context was supplied while world phase routing is disabled"
            )
        world_deltas = self._intervene_named_role_values(
            world_deltas, source_names
        )
        far_values = self._intervene_far_anchor_context(
            [
                self._far_anchor_camera_context(value)
                for value in world_deltas
            ]
        )
        candidate_banks: list[Tensor] = []
        expanded_names: list[str] = []
        for value, far_value, source_name in zip(
            world_deltas,
            far_values,
            source_names,
            strict=True,
        ):
            source_candidates, candidate_names = (
                self._world_to_policy_source_candidates(
                    value,
                    far_value,
                    source_name,
                )
            )
            candidate_banks.append(source_candidates)
            expanded_names.extend(candidate_names)
        # [B,T,source*(local_camera+far_anchor*camera),H]. Far candidates
        # are horizon-constant context; only the query supplies action time.
        values = torch.cat(candidate_banks, dim=2)
        values = values[:, :, None].expand(-1, -1, basis, -1, -1)
        routed, route_metrics = self.world_to_policy_attnres(query, values)
        scale = routed.new_tensor(
            float(getattr(cfg, "role_attnres_world_to_policy_scale", 0.10))
        )
        structured_update = scale * routed
        updated_trajectory = trajectory + structured_update
        canvas = torch.cat(
            (
                canvas[:, : int(trajectory_region.start)],
                updated_trajectory.reshape_as(canvas[:, trajectory_region]),
                canvas[:, int(trajectory_region.stop) :],
            ),
            dim=1,
        )
        metrics = self._role_route_metrics(
            "world_to_policy", route_metrics, tuple(expanded_names)
        )
        interval_source_indices = [
            index
            for index, name in enumerate(expanded_names)
            if name.startswith("interval_stage_")
        ]
        if interval_source_indices:
            source_mass = route_metrics.get("source_mass")
            if not isinstance(source_mass, Tensor):
                raise RuntimeError(
                    "typed interval-stage route did not expose source mass"
                )
            metrics[
                "attnres_world_to_policy_interval_stage_source_mass"
            ] = source_mass[interval_source_indices].mean()
        metrics["attnres_world_to_policy_horizon_route_std"] = route_metrics[
            "query_axis_1_route_std"
        ]
        metrics["attnres_world_to_policy_basis_route_std"] = route_metrics[
            "query_axis_2_route_std"
        ]
        metrics["attnres_world_to_policy_far_anchor_count"] = (
            route_metrics["update_rms"].new_tensor(
                float(self.world_to_policy_far_anchor_count)
            )
        )
        far_context_norms = [
            value.detach().float().norm(dim=-1).mean()
            for value in far_values
            if int(value.shape[1]) > 0
        ]
        metrics["attnres_world_to_policy_far_context_norm"] = (
            torch.stack(far_context_norms).mean()
            if far_context_norms
            else route_metrics["update_rms"].new_zeros(())
        )
        metrics["attnres_world_to_policy_fixed_scale"] = scale.detach().float()
        metrics["attnres_world_to_policy_structured_update_norm"] = (
            structured_update.detach().float().norm(dim=-1).mean()
        )
        metrics["attnres_world_to_policy_approved_value_norm"] = (
            routed.detach().float().norm(dim=-1).mean()
        )
        metrics["attnres_world_to_policy_phase_query_norm"] = (
            phase_query_delta.detach().float().norm(dim=-1).mean()
        )
        metrics["attnres_world_to_policy_condition_query_norm"] = (
            condition_query_delta.detach().float().norm(dim=-1).mean()
        )
        # As at G->W, the shared trajectory carrier receives a bounded step,
        # while the bottom typed bank receives the routed evidence itself.
        # The single P->MMDiT scale is therefore shared by G/W/P values instead
        # of being multiplied once per ownership boundary.
        return canvas, routed, metrics

    def _intervene_policy_delta_bank(
        self, bank: PolicyRoleDeltaBank
    ) -> PolicyRoleDeltaBank:
        mode = self._action_path_eval_intervention
        source_zero_modes = {
            f"{name}_zero": name for name in bank.source_names
        }
        source_shuffle_modes = {
            f"{name}_shuffle": name for name in bank.source_names
        }
        if mode not in {
            "policy_zero",
            "policy_temporal_shuffle",
            "protected_detail_zero",
            "protected_detail_shuffle",
            *source_zero_modes,
            *source_shuffle_modes,
        }:
            return bank
        self._action_path_eval_apply_count += 1
        if mode == "policy_zero":
            bank_delta = bank.values.detach().float().norm(dim=-1).mean()
            if bank.protected_detail is not None:
                bank_delta = bank_delta + bank.protected_detail.detach().float().norm(
                    dim=-1
                ).mean()
            self._action_path_eval_metrics["policy_bank_delta_norm"] = float(
                bank_delta.cpu()
            )
            return PolicyRoleDeltaBank(
                values=torch.zeros_like(bank.values),
                source_names=bank.source_names,
                source_depths=bank.source_depths,
                protected_detail=(
                    None
                    if bank.protected_detail is None
                    else torch.zeros_like(bank.protected_detail)
                ),
            )
        if mode in {"protected_detail_zero", "protected_detail_shuffle"}:
            if bank.protected_detail is None:
                raise RuntimeError("protected-detail intervention has no detail value")
            if mode == "protected_detail_zero":
                intervened_detail = torch.zeros_like(bank.protected_detail)
            elif int(bank.protected_detail.shape[0]) > 1:
                intervened_detail = bank.protected_detail.roll(shifts=1, dims=0)
            else:
                intervened_detail = bank.protected_detail.roll(
                    shifts=max(int(self.config.action_horizon) // 2, 1),
                    dims=1,
                )
            self._action_path_eval_metrics[
                "protected_detail_delta_norm"
            ] = float(
                (intervened_detail - bank.protected_detail)
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
            return PolicyRoleDeltaBank(
                values=bank.values,
                source_names=bank.source_names,
                source_depths=bank.source_depths,
                protected_detail=intervened_detail,
            )
        if mode in source_zero_modes or mode in source_shuffle_modes:
            source = (
                source_zero_modes[mode]
                if mode in source_zero_modes
                else source_shuffle_modes[mode]
            )
            source_index = bank.source_names.index(source)
            values = bank.values.clone()
            original = values[:, source_index]
            if mode in source_zero_modes:
                intervened = torch.zeros_like(original)
            elif int(original.shape[0]) > 1:
                intervened = original.roll(shifts=1, dims=0)
            else:
                intervened = original.roll(
                    shifts=max(int(self.config.action_horizon) // 2, 1),
                    dims=1,
                )
            values[:, source_index] = intervened
            self._action_path_eval_metrics[f"{source}_delta_norm"] = float(
                (intervened - original)
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
            return PolicyRoleDeltaBank(
                values=values,
                source_names=bank.source_names,
                source_depths=bank.source_depths,
                protected_detail=bank.protected_detail,
            )
        shift = max(int(self.config.action_horizon) // 2, 1)
        intervened_values = bank.values.roll(shifts=shift, dims=2)
        intervened_detail = (
            None
            if bank.protected_detail is None
            else bank.protected_detail.roll(shifts=shift, dims=1)
        )
        self._action_path_eval_metrics["policy_bank_delta_norm"] = float(
            (intervened_values - bank.values)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return PolicyRoleDeltaBank(
            values=intervened_values,
            source_names=bank.source_names,
            source_depths=bank.source_depths,
            protected_detail=intervened_detail,
        )

    def _promote_midcut(
        self,
        mid: dict[str, Tensor],
        *,
        gates: dict[str, Tensor],
        content_norm: Tensor,
        time_norm: Tensor,
    ) -> dict[str, Tensor]:
        pred = mid["midcut_pred_physical_velocity"]
        effect = mid["midcut_rollout_effect_pred"]
        delta = mid["midcut_rollout_delta_pred"]
        z = _zeros_like_scalar(pred)
        out = {
            **mid,
            "canvas_tokens": mid["midcut_canvas_tokens"],
            "trajectory_tokens": mid["midcut_trajectory_tokens"],
            "rollout_tokens": mid["midcut_rollout_tokens"],
            "register_tokens": mid["midcut_register_tokens"],
            "direct_physical_velocity": mid["midcut_direct_physical_velocity"],
            "rollout_residual_velocity": mid["midcut_rollout_residual_velocity"],
            "rollout_alpha": mid["midcut_rollout_alpha"],
            "pred_physical_velocity": pred,
            "rollout_effect_pred": effect,
            "rollout_base_effect_pred": mid["midcut_rollout_base_effect_pred"],
            "rollout_delta_pred": delta,
            "future_latent_pred": effect,
            "action_effect_pred": effect,
            "event_logits": mid["midcut_event_logits"],
            "motion_logits": mid["midcut_motion_logits"],
            "transition_latent": mid["midcut_transition_latent"],
            "rollout_coeff_abs_mean": z,
            "rollout_neutral_coeff_abs_mean": z,
            "rollout_centered_coeff_abs_mean": z,
            "rollout_basis_norm": z,
            "rollout_delta_norm": mid["midcut_rollout_delta_norm"],
            "rollout_base_norm": z,
            "rollout_delta_gain": mid["midcut_future_gain"],
            "gate_self": gates.get("gate_self", z),
            "gate_visual": gates.get("gate_visual", z),
            "gate_stage": gates.get("gate_stage", z),
            "gate_stage_to_window": gates.get("gate_stage_to_window", z),
            "stage_to_window_update_norm": gates.get("stage_to_window_update_norm", z),
            "gate_rollout": gates.get("gate_rollout", z),
            "gate_ffn": gates.get("gate_ffn", z),
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.ones((), device=pred.device, dtype=pred.dtype),
        }
        return out

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
        *,
        executed_memory: Tensor | None = None,
        goal_language_tokens: Tensor | None = None,
        goal_language_mask: Tensor | None = None,
        goal_condition_keep: Tensor | None = None,
        action_history_condition_keep: Tensor | None = None,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
        collect_diagnostics: bool = True,
        visual_context: FlowDINOEvidencePack | None = None,
        raw_visual: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if proposal_keep is None:
            proposal_keep = torch.ones(
                noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype
            )
        if consequence_physical is None:
            consequence_physical = noisy_physical
        else:
            consequence_physical = consequence_physical.to(
                device=noisy_physical.device, dtype=noisy_physical.dtype
            )
        if self.flow_dino_evidence is not None:
            if visual_context is None:
                visual_context = self.flow_dino_evidence(visual, raw_visual=raw_visual)
            visual_memory = visual_context.selector_tokens
            visual_value_memory = visual_context.value_tokens
            stage_init = (
                visual_context.stage_query
                if int(visual_context.stage_query.shape[1]) > 0
                else None
            )
            rollout_init = visual_context.future_queries
        else:
            if visual_context is not None:
                raise ValueError("visual_context was provided while Flow-DINO JEPA is disabled")
            visual_memory = self.visual_memory(visual)
            visual_value_memory = visual_memory
            stage_init = None
            rollout_init = self.rollout_codec.rollout_init(visual)
        if self.goal_resampler is None:
            if goal_language_tokens is not None or goal_language_mask is not None:
                raise ValueError(
                    "language condition was supplied while goal conditioning is disabled"
                )
            goal_tokens = None
        else:
            if goal_language_tokens is None or goal_language_mask is None:
                raise ValueError(
                    "goal conditioning requires language tokens and an attention mask"
                )
            goal_tokens = self.goal_resampler(
                goal_language_tokens.to(device=noisy_physical.device, dtype=noisy_physical.dtype),
                goal_language_mask.to(device=noisy_physical.device, dtype=torch.bool),
            )
        batch = int(noisy_physical.shape[0])
        if (
            int(getattr(cfg, "goal_condition_exact_null", 0))
            and goal_condition_keep is None
        ):
            goal_condition_keep = torch.ones(
                batch, device=noisy_physical.device, dtype=noisy_physical.dtype
            )
        if (
            int(getattr(cfg, "action_history_condition_exact_null", 0))
            and action_history_condition_keep is None
        ):
            action_history_condition_keep = torch.ones(
                batch, device=noisy_physical.device, dtype=noisy_physical.dtype
            )
        canvas, slices = self.seed(
            noisy_physical=noisy_physical,
            state=state,
            state_history=state_history,
            executed_history=executed_history,
            executed_memory=executed_memory,
            proposal_tokens=proposal_tokens,
            proposal_keep=proposal_keep,
            rollout_init=rollout_init,
            stage_init=stage_init,
            goal_tokens=goal_tokens,
            goal_condition_keep=goal_condition_keep,
            action_history_condition_keep=action_history_condition_keep,
        )
        phase_context: Tensor | None = None
        condition_query_context: Tensor | None = None
        phase_metrics: dict[str, Tensor] = {}
        if self.stateless_phase_adapter is not None:
            (
                phase_context,
                condition_query_context,
                phase_metrics,
            ) = self.stateless_phase_adapter(
                goal_tokens=canvas[:, slices["task"]],
                history_tokens=canvas[:, slices["executed"]],
                state_tokens=canvas[:, slices["state"]],
                visual_tokens=visual_value_memory,
            )
            phase_context, condition_query_context = (
                self._intervene_query_contexts(
                    phase_context,
                    condition_query_context,
                )
            )
        # Ownership snapshots are taken before any canvas self-attention.  The
        # final state/trajectory slices are contextual mixtures and can carry
        # noisy-action content, so using them as evidence recreates the exact
        # action -> evidence -> action echo this decoder is meant to remove.
        owned_state_memory = [
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
        ]
        owned_trajectory_memory = canvas[:, slices["proposal"]]
        owned_intent_memory = {
            "task": canvas[:, slices["task"]],
            "state": canvas[:, slices["state"]],
            "state_history": canvas[:, slices["state_history"]],
            "executed": canvas[:, slices["executed"]],
            "proposal": canvas[:, slices["proposal"]],
            "visual": (
                visual_value_memory
                if visual_context is not None
                else canvas[:, slices["rollout"]].mean(dim=1, keepdim=True)
            ),
        }
        strict_role_visual_path = bool(
            int(getattr(cfg, "flow_jepa_strict_role_visual_path", 0))
        )
        if strict_role_visual_path:
            # Raw visual evidence has one owner: the grounding/world route.
            # Clean task/state/action intent remains available to the decoder.
            owned_intent_memory.pop("visual", None)
        rollout_seed = canvas[:, slices["rollout"]].detach()
        trajectory_seed = canvas[:, slices["trajectory"]].detach()
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        content_norm_rows: list[Tensor] = []
        time_norm_rows: list[Tensor] = []
        midcut: dict[str, Tensor] | None = None
        layer_contracts: list[dict[str, Tensor]] = []
        raw_refinement_metrics: dict[str, Tensor] = {}
        late_detail_metrics: dict[str, Tensor] = {}
        late_raw_detail: LateRawDetailEvidence | None = None
        world_entry_rollout: Tensor | None = None
        world_detail_entry_rollout: Tensor | None = None
        grounding_role_deltas: list[Tensor] = []
        world_role_deltas: list[Tensor] = []
        policy_role_deltas: list[Tensor] = []
        policy_role_depths: list[int] = []
        role_delta_metrics: dict[str, Tensor] = {}
        approved_ground_to_world: Tensor | None = None
        approved_world_to_policy: Tensor | None = None
        protected_policy_detail: Tensor | None = None
        interval_stage_prediction: Tensor | None = None
        interval_stage_input_rollout: Tensor | None = None
        interval_stage_role_delta: Tensor | None = None
        online_horizon_address = bool(
            self.flow_dino_evidence is not None
            and self.flow_dino_evidence.online_horizon_address_enabled
            and not self.flow_dino_evidence.progressive_grounding_address_enabled
        )
        progressive_grounding_address = bool(
            self.flow_dino_evidence is not None
            and self.flow_dino_evidence.progressive_grounding_address_enabled
        )
        progressive_address_state: ProgressiveGroundingAddressState | None = None
        online_horizon_address_applied = False
        future_address_metrics: dict[str, Tensor] = {}
        horizon_boundary_metrics: dict[str, Tensor] = {}
        horizon_address_base_metric: Tensor | None = None
        horizon_address_write_metric: Tensor | None = None

        def _record_horizon_boundary(label: str, value: Tensor) -> None:
            # The address write itself is unconditional under the V108 flag.
            # Boundary reductions are audit-only and must not add work to the
            # diagnostics-disabled deployment path.
            if not (
                (online_horizon_address or progressive_grounding_address)
                and collect_diagnostics
            ):
                return
            expected = (
                int(cfg.future_anchors)
                * int(cfg.num_cameras)
                * int(cfg.future_grid_size)
                * int(cfg.future_grid_size)
            )
            if value.ndim != 3 or int(value.shape[1]) != expected:
                raise RuntimeError(
                    f"online horizon boundary {label!r} lost rollout geometry"
                )
            grouped = value.detach().float().reshape(
                int(value.shape[0]),
                int(cfg.future_anchors),
                int(cfg.num_cameras),
                int(cfg.future_grid_size),
                int(cfg.future_grid_size),
                int(value.shape[-1]),
            )
            horizon = grouped.mean(dim=(2, 3, 4))
            prefix = f"flow_jepa_online_address_boundary_{label}"
            horizon_boundary_metrics[f"{prefix}_rms"] = grouped.square().mean().sqrt()
            if int(horizon.shape[1]) > 1:
                horizon_boundary_metrics[f"{prefix}_adjacent_cosine"] = (
                    F.cosine_similarity(horizon[:, 1:], horizon[:, :-1], dim=-1).mean()
                )
            else:
                horizon_boundary_metrics[f"{prefix}_adjacent_cosine"] = grouped.new_zeros(())
            if (
                horizon_address_base_metric is not None
                and horizon_address_write_metric is not None
            ):
                cumulative = value.detach().float() - horizon_address_base_metric
                write = horizon_address_write_metric
                cumulative_flat = cumulative.flatten(1)
                write_flat = write.flatten(1)
                horizon_boundary_metrics[
                    f"{prefix}_cumulative_address_cosine"
                ] = F.cosine_similarity(
                    cumulative_flat,
                    write_flat,
                    dim=-1,
                ).mean()
                horizon_boundary_metrics[
                    f"{prefix}_cumulative_address_projection"
                ] = (
                    (cumulative_flat * write_flat).sum(dim=-1)
                    / write_flat.square().sum(dim=-1).clamp_min(1e-8)
                ).mean()

        collect_role_deltas = bool(int(getattr(cfg, "role_attnres_enabled", 0)))
        if progressive_grounding_address:
            if visual_context is None or self.flow_dino_evidence is None:
                raise RuntimeError(
                    "progressive grounding address requires Flow-DINO context"
                )
            (
                visual_memory,
                visual_value_memory,
                raw_refinement_metrics,
                late_raw_detail,
            ) = self.flow_dino_evidence.refine_raw_evidence(
                visual_context,
                canvas,
                slices,
                return_late_detail=True,
            )
            if late_raw_detail is None or late_raw_detail.address_bank is None:
                raise RuntimeError(
                    "progressive grounding address did not compile its pre-G bank"
                )
            progressive_address_state = (
                self.flow_dino_evidence.begin_progressive_grounding_address(
                    late_raw_detail.address_bank
                )
            )
        _record_horizon_boundary("seed", canvas[:, slices["rollout"]])
        # The latent-main decoder is the final action path, so inference/eval
        # must still materialize layer contracts even when callers disable
        # auxiliary contract evaluation for speed.  We do not add extra losses;
        # we only expose the latents needed by the action decoder.
        final_decoder = str(getattr(cfg, "final_action_decoder", "legacy"))
        force_layer_contracts = bool(enable_final_action_decoder) and (
            final_decoder == "latent_main_action"
            or (
                final_decoder in {"latent_cvae_action", "adaptive_recurrent_cvae_action"}
                and bool(int(getattr(cfg, "latent_cvae_layer_memory", 1)))
            )
            or final_decoder == "hierarchical_mmdit_action"
            or final_decoder == "evidence_latent_mmdit_action"
        )
        effective_layer_contracts = bool(enable_layer_contracts) or force_layer_contracts
        cut = int(cfg.midcut_layer)
        contract_grad_scale = float(getattr(cfg, "layer_contract_grad_scale", 1.0))
        for index, block in enumerate(self.blocks, start=1):
            grounding_boundary = int(
                getattr(cfg, "flow_jepa_grounding_blocks", 0)
            )
            world_boundary = grounding_boundary + int(
                getattr(cfg, "flow_jepa_world_blocks", 0)
            )
            if (
                self.ground_to_world_attnres is not None
                and index == grounding_boundary + 1
            ):
                (
                    canvas,
                    approved_ground_to_world,
                    bridge_metrics,
                ) = self._apply_ground_to_world_bridge(
                    canvas,
                    slices,
                    grounding_role_deltas,
                )
                role_delta_metrics.update(bridge_metrics)
            if online_horizon_address and index == grounding_boundary + 1:
                if self.flow_dino_evidence is None:
                    raise RuntimeError("online horizon address has no Flow-DINO owner")
                if late_raw_detail is None or late_raw_detail.address_bank is None:
                    raise RuntimeError(
                        "online horizon address did not receive the G3 observation bank"
                    )
                rollout_region = slices["rollout"]
                address_base = canvas[:, rollout_region]
                (
                    address_refined,
                    future_address_metrics,
                ) = self.flow_dino_evidence.organize_horizon_address(
                    address_base,
                    late_raw_detail.address_bank,
                )
                address_refined = self._intervene_online_horizon_address(
                    address_base,
                    address_refined,
                )
                horizon_address_base_metric = address_base.detach().float()
                horizon_address_write_metric = (
                    address_refined.detach().float() - horizon_address_base_metric
                )
                future_address_metrics[
                    "flow_jepa_online_horizon_address_write_rms"
                ] = horizon_address_write_metric.square().mean().sqrt()
                canvas = torch.cat(
                    (
                        canvas[:, : int(rollout_region.start)],
                        address_refined,
                        canvas[:, int(rollout_region.stop) :],
                    ),
                    dim=1,
                )
                online_horizon_address_applied = True
                _record_horizon_boundary(
                    "post_address",
                    canvas[:, rollout_region],
                )
            if (
                self.flow_dino_evidence is not None
                and self.flow_dino_evidence.interval_stage_enabled
                and index == world_boundary + 1
            ):
                rollout_region = slices["rollout"]
                interval_stage_input_rollout = canvas[:, rollout_region]
                (
                    interval_stage_rollout,
                    interval_stage_prediction,
                    interval_stage_metrics,
                ) = self.flow_dino_evidence.organize_interval_stage(
                    interval_stage_input_rollout
                )
                if interval_stage_prediction is None:
                    raise RuntimeError(
                        "active interval-stage organizer returned no prediction"
                    )
                interval_stage_rollout = (
                    self._intervene_interval_stage_rollout(
                        interval_stage_input_rollout,
                        interval_stage_rollout,
                    )
                )
                if self.interval_stage_typed_value:
                    interval_stage_write = (
                        interval_stage_rollout - interval_stage_input_rollout
                    )
                    interval_stage_role_delta = interval_stage_write.reshape(
                        int(interval_stage_write.shape[0]),
                        int(cfg.future_anchors),
                        int(cfg.num_cameras),
                        int(cfg.future_grid_size),
                        int(cfg.future_grid_size),
                        int(interval_stage_write.shape[-1]),
                    ).mean(dim=(3, 4))
                    role_delta_metrics[
                        "attnres_observed_interval_stage_delta_norm"
                    ] = (
                        interval_stage_role_delta.detach()
                        .float()
                        .norm(dim=-1)
                        .mean()
                    )
                    role_delta_metrics[
                        "flow_jepa_interval_stage_typed_value"
                    ] = interval_stage_write.new_ones((), dtype=torch.float32)
                canvas = torch.cat(
                    (
                        canvas[:, : int(rollout_region.start)],
                        interval_stage_rollout,
                        canvas[:, int(rollout_region.stop) :],
                    ),
                    dim=1,
                )
                role_delta_metrics.update(interval_stage_metrics)
                _record_horizon_boundary(
                    "post_interval",
                    canvas[:, rollout_region],
                )
            if (
                self.world_to_policy_attnres is not None
                and index == world_boundary + 1
            ):
                world_bridge_values = list(world_role_deltas)
                world_bridge_names = tuple(
                    f"w{depth + 1}" for depth in range(len(world_role_deltas))
                )
                if approved_ground_to_world is not None:
                    world_bridge_values.insert(0, approved_ground_to_world)
                    world_bridge_names = ("grounding_entry",) + world_bridge_names
                if self.interval_stage_typed_value:
                    if interval_stage_role_delta is None:
                        raise RuntimeError(
                            "typed interval-stage value was not built at W->P"
                        )
                    world_bridge_values.append(interval_stage_role_delta)
                    world_bridge_names = world_bridge_names + ("interval_stage",)
                (
                    canvas,
                    approved_world_to_policy,
                    bridge_metrics,
                ) = self._apply_world_to_policy_bridge(
                    canvas,
                    slices,
                    world_bridge_values,
                    world_bridge_names,
                    phase_context=phase_context,
                    condition_query_context=condition_query_context,
                )
                role_delta_metrics.update(bridge_metrics)
            if (
                self.late_raw_detail_reader is not None
                and index == world_boundary + 1
            ):
                if late_raw_detail is None:
                    raise RuntimeError(
                        "late raw detail was not compiled at the grounding boundary"
                    )
                if world_detail_entry_rollout is None:
                    raise RuntimeError(
                        "late-detail world path did not capture its entry rollout"
                    )
                rollout_region = slices["rollout"]
                current_rollout = canvas[:, rollout_region]
                if progressive_grounding_address:
                    if (
                        self.flow_dino_evidence is None
                        or progressive_address_state is None
                    ):
                        raise RuntimeError(
                            "W->P progressive read lost its G3 address state"
                        )
                    (
                        relevance_logits,
                        progressive_horizon_metrics,
                    ) = self.flow_dino_evidence.score_progressive_horizon_posterior(
                        current_rollout,
                        progressive_address_state,
                    )
                    future_address_metrics = {
                        **progressive_horizon_metrics,
                        "flow_jepa_horizon_address_logits": relevance_logits,
                    }
                with torch.no_grad():
                    world_metric_rollout = (
                        current_rollout
                        if interval_stage_input_rollout is None
                        else interval_stage_input_rollout
                    )
                    world_residual = (
                        world_metric_rollout.detach() - world_detail_entry_rollout
                    )
                    grouped_world_residual = world_residual.reshape(
                        world_residual.shape[0],
                        int(cfg.future_anchors),
                        int(cfg.num_cameras),
                        int(cfg.future_grid_size),
                        int(cfg.future_grid_size),
                        world_residual.shape[-1],
                    )
                    spatial_mean = grouped_world_residual.mean(
                        dim=(3, 4), keepdim=True
                    )
                    late_detail_metrics[
                        "flow_jepa_world_spatial_residual_norm"
                    ] = (
                        grouped_world_residual - spatial_mean
                    ).float().norm(dim=-1).mean()
                    late_detail_metrics[
                        "flow_jepa_world_anchor_camera_residual_norm"
                    ] = spatial_mean.float().norm(dim=-1).mean()
                    late_detail_metrics[
                        "flow_jepa_world_anchor_write_only"
                    ] = current_rollout.new_tensor(
                        float(
                            int(
                                getattr(
                                    cfg,
                                    "flow_jepa_world_anchor_write_only",
                                    0,
                                )
                            )
                        ),
                        dtype=torch.float32,
                    )
                trajectory_before_detail = canvas[:, slices["trajectory"]]
                updated_trajectory, reader_metrics = self.late_raw_detail_reader(
                    trajectory_before_detail,
                    current_rollout,
                    late_raw_detail,
                    phase_context=phase_context,
                    condition_query_context=condition_query_context,
                )
                if collect_role_deltas:
                    protected_policy_detail = (
                        updated_trajectory - trajectory_before_detail
                    ).reshape(
                        int(updated_trajectory.shape[0]),
                        int(cfg.action_horizon),
                        int(cfg.action_basis_tokens),
                        int(updated_trajectory.shape[-1]),
                    )
                late_detail_metrics.update(reader_metrics)
                trajectory_region = slices["trajectory"]
                canvas = torch.cat(
                    (
                        canvas[:, : int(trajectory_region.start)],
                        updated_trajectory,
                        canvas[:, int(trajectory_region.stop) :],
                    ),
                    dim=1,
                )
            role = self.block_roles[index - 1]
            rollout_before_block = (
                canvas[:, slices["rollout"]]
                if collect_role_deltas and role in {"grounding", "world"}
                else None
            )
            trajectory_before_block = (
                canvas[:, slices["trajectory"]]
                if collect_role_deltas and role == "policy"
                else None
            )
            mod_emb, content_delta, time_row = self._mod_embed(
                canvas,
                visual_memory,
                time_emb,
                slices,
                role=role,
            )
            if role == "world" and self.phase_world_block_query_proj is not None:
                if (
                    self.condition_world_block_query_proj is None
                    or phase_context is None
                    or condition_query_context is None
                ):
                    raise RuntimeError(
                        "phase-conditioned world block has no query contexts"
                    )
                world_depth = index - grounding_boundary - 1
                if not 0 <= world_depth < len(
                    self.phase_world_block_query_proj
                ):
                    raise RuntimeError(
                        "world block depth is outside its phase-query bank"
                    )
                query_scale = float(
                    getattr(cfg, "stateless_phase_query_scale", 0.10)
                )
                phase_world_delta = query_scale * (
                    self.phase_world_block_query_proj[world_depth](
                        phase_context.to(
                            device=mod_emb.device,
                            dtype=mod_emb.dtype,
                        )
                    )
                    + self.condition_world_block_query_proj[world_depth](
                        condition_query_context.to(
                            device=mod_emb.device,
                            dtype=mod_emb.dtype,
                        )
                    )
                )
                mod_emb = mod_emb + phase_world_delta
                role_delta_metrics[
                    f"flow_jepa_world_block_query_delta_norm_w{world_depth + 1}"
                ] = (
                    phase_world_delta.detach().float().norm(dim=-1).mean()
                )
            content_norm_rows.append(content_delta.float().norm(dim=-1).mean())
            time_norm_rows.append(time_row.float().norm(dim=-1).mean())
            canvas, gates = block(
                canvas,
                visual_memory,
                mod_emb,
                slices,
                visual_value_memory=visual_value_memory,
            )
            gate_rows.append(gates)
            if rollout_before_block is not None:
                rollout_delta = canvas[:, slices["rollout"]] - rollout_before_block
                structured_rollout_delta = rollout_delta.reshape(
                    int(rollout_delta.shape[0]),
                    int(cfg.future_anchors),
                    int(cfg.num_cameras),
                    int(cfg.future_grid_size),
                    int(cfg.future_grid_size),
                    int(rollout_delta.shape[-1]),
                ).mean(dim=(3, 4))
                if role == "grounding":
                    grounding_role_deltas.append(structured_rollout_delta)
                    depth_index = len(grounding_role_deltas)
                    role_delta_metrics[
                        f"attnres_observed_grounding_delta_norm_g{depth_index}"
                    ] = structured_rollout_delta.detach().float().norm(dim=-1).mean()
                else:
                    world_role_deltas.append(structured_rollout_delta)
                    depth_index = len(world_role_deltas)
                    role_delta_metrics[
                        f"attnres_observed_world_delta_norm_w{depth_index}"
                    ] = structured_rollout_delta.detach().float().norm(dim=-1).mean()
                    with torch.no_grad():
                        grouped = rollout_delta.reshape(
                            int(rollout_delta.shape[0]),
                            int(cfg.future_anchors),
                            int(cfg.num_cameras),
                            int(cfg.future_grid_size),
                            int(cfg.future_grid_size),
                            int(rollout_delta.shape[-1]),
                        )
                        spatial_residual = grouped - grouped.mean(
                            dim=(3, 4), keepdim=True
                        )
                        role_delta_metrics[
                            f"attnres_observed_world_xy_update_norm_w{depth_index}"
                        ] = spatial_residual.float().norm(dim=-1).mean()
            if trajectory_before_block is not None:
                trajectory_delta = (
                    canvas[:, slices["trajectory"]] - trajectory_before_block
                ).reshape(
                    int(canvas.shape[0]),
                    int(cfg.action_horizon),
                    int(cfg.action_basis_tokens),
                    int(canvas.shape[-1]),
                )
                policy_role_deltas.append(trajectory_delta)
                policy_role_depths.append(index - 1)
                depth_index = len(policy_role_deltas)
                role_delta_metrics[
                    f"attnres_observed_policy_delta_norm_p{depth_index}"
                ] = trajectory_delta.detach().float().norm(dim=-1).mean()
            if progressive_grounding_address and role == "grounding":
                if (
                    self.flow_dino_evidence is None
                    or progressive_address_state is None
                    or late_raw_detail is None
                ):
                    raise RuntimeError(
                        "progressive grounding stage lost its pre-G address state"
                    )
                grounding_stage = index
                progressive_address_state = (
                    self.flow_dino_evidence.update_progressive_grounding_address(
                        progressive_address_state,
                        canvas[:, slices["rollout"]],
                        stage=grounding_stage,
                        intervention=self._action_path_eval_intervention,
                    )
                )
                if progressive_address_state.metrics is not None:
                    raw_refinement_metrics.update(
                        progressive_address_state.metrics
                    )
                intervention_name = f"address_g{grounding_stage}"
                if self._action_path_eval_intervention in {
                    f"{intervention_name}_zero",
                    f"{intervention_name}_shuffle",
                }:
                    self._action_path_eval_apply_count += 1
                    self._action_path_eval_metrics[
                        f"{intervention_name}_applied"
                    ] = 1.0
                if grounding_stage == grounding_boundary:
                    summary = (
                        progressive_address_state.canonical_summary_tokens
                    )
                    if summary is None:
                        raise RuntimeError(
                            "G3 did not compile its selector summary"
                        )
                    late_raw_detail.progressive_address = (
                        progressive_address_state
                    )
                    # The handoff carries keys/geometry only.  Fine raw values
                    # remain solely in the observation bank until W->P.
                    visual_memory = torch.cat(
                        (visual_memory, summary), dim=1
                    )
                    visual_value_memory = torch.cat(
                        (visual_value_memory, summary), dim=1
                    )
                    raw_refinement_metrics[
                        "flow_jepa_progressive_g3_summary_token_count"
                    ] = summary.new_tensor(
                        float(summary.shape[1]), dtype=torch.float32
                    )
                    owner_sidecar_keys = (
                        progressive_address_state.canonical_semantic_keys,
                        progressive_address_state.canonical_appearance_keys,
                        progressive_address_state.canonical_geometry_keys,
                    )
                    owner_sidecar_token_count = sum(
                        0
                        if value is None
                        else int(
                            value.reshape(
                                int(value.shape[0]), -1, value.shape[-1]
                            ).shape[1]
                        )
                        for value in owner_sidecar_keys
                    )
                    raw_refinement_metrics[
                        "flow_jepa_progressive_g3_owner_sidecar_token_count"
                    ] = summary.new_tensor(
                        float(owner_sidecar_token_count),
                        dtype=torch.float32,
                    )
            if (
                not progressive_grounding_address
                and
                visual_context is not None
                and (
                    visual_context.raw_context is not None
                    or visual_context.late_raw_detail is not None
                )
                and index == int(getattr(cfg, "flow_jepa_grounding_blocks", 3))
            ):
                if self.flow_dino_evidence is None:
                    raise RuntimeError("raw visual context has no owning Flow-DINO encoder")
                (
                    visual_memory,
                    visual_value_memory,
                    raw_refinement_metrics,
                    late_raw_detail,
                ) = (
                    self.flow_dino_evidence.refine_raw_evidence(
                        visual_context,
                        canvas,
                        slices,
                        return_late_detail=True,
                    )
                )
                # Refined raw evidence is still observation-owned and may be
                # read directly by the single final action decoder.  It never
                # receives an action-writing head of its own.
                if not strict_role_visual_path:
                    owned_intent_memory["visual"] = visual_value_memory
            if (
                index == grounding_boundary
                and self.late_raw_detail_reader is not None
            ):
                world_detail_entry_rollout = canvas[
                    :, slices["rollout"]
                ].detach()
            if (
                strict_role_visual_path
                and index == grounding_boundary
                and self._action_path_eval_intervention
                in {
                    "world_residual_zero",
                    "world_residual_anchor_shuffle",
                    "world_residual_spatial_shuffle",
                    "world_residual_spatiotemporal_shuffle",
                }
            ):
                # This fixed slot-aligned seed contains the grounding output
                # and its positional identity, but none of the world-block
                # update that the probe is intended to test.
                world_entry_rollout = canvas[:, slices["rollout"]].detach()
            if (
                strict_role_visual_path
                and index == world_boundary
                and self._action_path_eval_intervention
                in {
                    "world_residual_zero",
                    "world_residual_anchor_shuffle",
                    "world_residual_spatial_shuffle",
                    "world_residual_spatiotemporal_shuffle",
                }
            ):
                if world_entry_rollout is None:
                    raise RuntimeError(
                        "world residual intervention did not capture the grounding boundary"
                    )
                rollout_region = slices["rollout"]
                intervened_rollout = self._intervene_world_rollout(
                    canvas[:, rollout_region],
                    world_entry_rollout=world_entry_rollout,
                )
                canvas = torch.cat(
                    (
                        canvas[:, : int(rollout_region.start)],
                        intervened_rollout,
                        canvas[:, int(rollout_region.stop) :],
                    ),
                    dim=1,
                )
            if online_horizon_address or progressive_grounding_address:
                if index == grounding_boundary:
                    _record_horizon_boundary(
                        "post_g3",
                        canvas[:, slices["rollout"]],
                    )
                elif grounding_boundary < index <= world_boundary:
                    _record_horizon_boundary(
                        f"post_w{index - grounding_boundary}",
                        canvas[:, slices["rollout"]],
                    )
            contract_layer_active = (
                not self.terminal_policy_layer_contracts_only
                or index
                > int(cfg.flow_jepa_grounding_blocks)
                + int(cfg.flow_jepa_world_blocks)
            )
            if (
                effective_layer_contracts
                and contract_layer_active
                and len(self.layer_contract_heads) > 0
            ):
                contract_canvas = _scaled_contract_view(canvas, contract_grad_scale)
                layer_entry = self.layer_contract_heads[index - 1](contract_canvas, slices)
                if self.layer_consequence_cell is not None:
                    # V40: split the layer contract into an explicit world-latent
                    # object and an action-causal object.  Lower layers lean on
                    # the causal branch; upper layers lean on the latent branch.
                    # We keep the old direct outputs for forensics only.
                    latent_effect = layer_entry["rollout_effect_pred"]
                    latent_delta = layer_entry["rollout_delta_pred"]
                    cons = self.layer_consequence_cell(
                        rollout_tokens=layer_entry["rollout_tokens"],
                        action_physical=consequence_physical,
                        state_tokens=layer_entry.get("state_tokens"),
                        state_history_tokens=layer_entry.get("state_history_tokens"),
                        executed_tokens=layer_entry.get("executed_tokens"),
                        trajectory_tokens=layer_entry.get("trajectory_tokens"),
                        proposal_tokens=layer_entry.get("proposal_tokens"),
                        layer_index=index - 1,
                    )
                    causal_gain, latent_gain = self.layer_role_scheduler(
                        index - 1,
                        device=latent_effect.device,
                        dtype=latent_effect.dtype,
                    )
                    causal_effect = cons["milestone_rollout_effect_pred"]
                    causal_delta = cons["milestone_rollout_delta_pred"]
                    layer_entry["latent_rollout_effect_pred"] = latent_effect
                    layer_entry["latent_rollout_delta_pred"] = latent_delta
                    layer_entry["causal_rollout_effect_pred"] = causal_effect
                    layer_entry["causal_rollout_delta_pred"] = causal_delta
                    layer_entry["direct_rollout_effect_pred"] = latent_effect
                    layer_entry["direct_rollout_delta_pred"] = latent_delta
                    # V40.1: one unified intervention-latent head is the
                    # supervised object.  The weak direct latent readout remains
                    # only for forensics; it is no longer mixed into the main
                    # rollout prediction where it can blur causal semantics.
                    layer_entry["rollout_effect_pred"] = causal_effect
                    layer_entry["rollout_delta_pred"] = causal_delta
                    layer_entry["policy_effect_tokens"] = cons["milestone_policy_effect_tokens"]
                    layer_entry["policy_effect_time_tokens"] = cons["milestone_policy_time_tokens"]
                    layer_entry["milestone_step_delta_pred"] = cons["milestone_step_delta_pred"]
                    layer_entry["unified_intervention_latent_pred"] = cons[
                        "milestone_intervention_latent_pred"
                    ]
                    layer_entry["neutral_latent_pred"] = cons["milestone_neutral_latent_pred"]
                    layer_entry["layer_causal_gain"] = causal_gain.detach().float()
                    layer_entry["layer_latent_gain"] = latent_gain.detach().float()
                    if bool(enable_layer_contracts) and int(
                        getattr(cfg, "layer_zero_base_diagnostic", 0)
                    ):
                        # Loss-free shortcut probe.  If zeroing the rollout
                        # tokens barely moves the consequence output, the cell
                        # is probably relying on action features instead of the
                        # state/rollout context.
                        with torch.no_grad():
                            cons_zero = self.layer_consequence_cell(
                                rollout_tokens=torch.zeros_like(layer_entry["rollout_tokens"]),
                                action_physical=consequence_physical,
                                state_tokens=layer_entry.get("state_tokens"),
                                state_history_tokens=layer_entry.get("state_history_tokens"),
                                executed_tokens=layer_entry.get("executed_tokens"),
                                trajectory_tokens=layer_entry.get("trajectory_tokens"),
                                proposal_tokens=layer_entry.get("proposal_tokens"),
                                layer_index=index - 1,
                            )
                            base_eff = cons["milestone_rollout_effect_pred"].detach().float()
                            zero_eff = cons_zero["milestone_rollout_effect_pred"].float()
                            zero_shift = (base_eff - zero_eff).norm(dim=-1).mean() / base_eff.norm(
                                dim=-1
                            ).mean().clamp_min(1e-6)
                        layer_entry["consequence_zero_base_shift"] = zero_shift
                    if (
                        bool(enable_layer_contracts)
                        and int(getattr(cfg, "layer_state_counterfactual", 0))
                        and int(layer_entry["rollout_tokens"].shape[0]) > 1
                    ):
                        flat_state = layer_entry["rollout_tokens"].detach().float().flatten(1)
                        dist_state = torch.cdist(flat_state, flat_state, p=2)
                        eye_state = torch.eye(
                            dist_state.shape[0], device=dist_state.device, dtype=torch.bool
                        )
                        dist_state = dist_state.masked_fill(eye_state, -1.0)
                        state_perm = dist_state.argmax(dim=1)
                        cons_state = self.layer_consequence_cell(
                            rollout_tokens=layer_entry["rollout_tokens"][state_perm],
                            action_physical=consequence_physical,
                            state_tokens=None
                            if layer_entry.get("state_tokens") is None
                            else layer_entry["state_tokens"][state_perm],
                            state_history_tokens=None
                            if layer_entry.get("state_history_tokens") is None
                            else layer_entry["state_history_tokens"][state_perm],
                            executed_tokens=None
                            if layer_entry.get("executed_tokens") is None
                            else layer_entry["executed_tokens"][state_perm],
                            trajectory_tokens=None
                            if layer_entry.get("trajectory_tokens") is None
                            else layer_entry["trajectory_tokens"][state_perm],
                            proposal_tokens=None
                            if layer_entry.get("proposal_tokens") is None
                            else layer_entry["proposal_tokens"][state_perm],
                            layer_index=index - 1,
                        )
                        layer_entry["rollout_effect_pred_shuffle_state"] = cons_state[
                            "milestone_rollout_effect_pred"
                        ]
                        layer_entry["rollout_delta_pred_shuffle_state"] = cons_state[
                            "milestone_rollout_delta_pred"
                        ]
                        layer_entry["milestone_step_delta_pred_shuffle_state"] = cons_state[
                            "milestone_step_delta_pred"
                        ]
                        layer_entry["policy_effect_tokens_shuffle_state"] = cons_state[
                            "milestone_policy_effect_tokens"
                        ]
                    if int(getattr(cfg, "layer_causal_event_from_effect", 1)):
                        event_src = cons["milestone_policy_time_tokens"]
                        layer_entry["event_logits"] = self.event_probe(event_src)
                    for key in (
                        "milestone_gate_mean",
                        "milestone_step_delta_norm",
                        "milestone_effect_norm",
                        "milestone_effect_std",
                        "milestone_effect_gain",
                    ):
                        layer_entry[key] = cons[key]
                if self.layer_fm_probe is not None:
                    probe_velocity = self.layer_fm_probe(
                        trajectory_pooled=layer_entry["trajectory_pooled"],
                        rollout_effect_pred=layer_entry["rollout_effect_pred"],
                        rollout_delta_pred=layer_entry["rollout_delta_pred"],
                        noisy_physical=noisy_physical,
                        time=time,
                    )
                    # In V39.2/V39.3 the action-flow probe is downstream of
                    # the layer latent.  It replaces the per-layer direct
                    # action head for contract losses, while remaining shared
                    # across all layers.
                    layer_entry["pred_physical_velocity"] = probe_velocity
                    layer_entry["direct_physical_velocity"] = probe_velocity
                    layer_entry["layer_fm_probe_velocity"] = probe_velocity
                layer_contracts.append(layer_entry)
            if index == cut:
                mid_canvas = self.midcut_norm(canvas)
                midcut = self.midcut_heads(mid_canvas, slices)
                if stop_at_midcut:
                    content_norm = (
                        torch.stack(content_norm_rows).mean()
                        if content_norm_rows
                        else _zeros_like_scalar(canvas)
                    )
                    time_norm = (
                        torch.stack(time_norm_rows).mean()
                        if time_norm_rows
                        else _zeros_like_scalar(canvas)
                    )
                    gate_mean = {
                        key: torch.stack([row[key] for row in gate_rows]).mean()
                        for key in (
                            "gate_self",
                            "gate_visual",
                            "gate_stage",
                            "gate_stage_to_window",
                            "stage_to_window_update_norm",
                            "gate_rollout",
                            "gate_ffn",
                            "residual_contract_enabled",
                            "residual_contract_max_rms",
                            "residual_contract_after_gate",
                            "residual_raw_rms",
                            "residual_proposed_rms",
                            "residual_bounded_rms",
                            "residual_written_rms",
                            "residual_compression",
                            "normalization_contract_enabled",
                        )
                    }
                    gate_mean["normalization_denominator_min"] = torch.stack(
                        [
                            row["normalization_denominator_min"]
                            for row in gate_rows
                        ]
                    ).amin()
                    gate_mean["normalization_gain_max"] = torch.stack(
                        [row["normalization_gain_max"] for row in gate_rows]
                    ).amax()
                    promoted = self._promote_midcut(
                        midcut, gates=gate_mean, content_norm=content_norm, time_norm=time_norm
                    )
                    if layer_contracts:
                        promoted["layer_contracts"] = layer_contracts
                    return promoted
        if midcut is None:
            # Defensive fallback; validate() should prevent this.
            midcut = self.midcut_heads(self.midcut_norm(canvas), slices)
        if isinstance(
            self.final_norm, AffineVarianceFlooredCenteredNorm
        ):
            (
                canvas,
                terminal_norm_denominator,
                terminal_norm_gain,
            ) = self.final_norm.forward_with_denominator(canvas)
            terminal_norm_denominator = (
                terminal_norm_denominator.detach().float().amin()
            )
            terminal_norm_gain = terminal_norm_gain.detach().float()
        else:
            canvas = self.final_norm(canvas)
            terminal_norm_denominator = canvas.new_ones(
                (), dtype=torch.float32
            )
            terminal_norm_gain = canvas.new_ones((), dtype=torch.float32)
        trajectory = canvas[:, slices["trajectory"]]
        stage_tokens = canvas[:, slices["stage"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.direct_physical_head.pooled(trajectory)
        typed_policy_delta_bank: PolicyRoleDeltaBank | None = None
        if int(getattr(cfg, "flow_jepa_role_hierarchy", 0)):
            normalized_trajectory_seed = self.final_norm(
                trajectory_seed.to(device=trajectory.device, dtype=trajectory.dtype)
            )
            policy_workspace_tokens = trajectory - normalized_trajectory_seed
            if int(getattr(cfg, "role_attnres_policy_to_mmdit", 0)):
                approved_values: list[Tensor] = []
                approved_names: list[str] = []
                approved_depths: list[int] = []
                if approved_world_to_policy is not None:
                    approved_values.append(approved_world_to_policy)
                    approved_names.append("world_to_policy")
                    approved_depths.append(
                        int(cfg.flow_jepa_grounding_blocks)
                        + int(cfg.flow_jepa_world_blocks)
                        - 1
                    )
                approved_values.extend(policy_role_deltas)
                approved_names.extend(
                    f"p{index + 1}" for index in range(len(policy_role_deltas))
                )
                approved_depths.extend(policy_role_depths)
                if not approved_values:
                    raise RuntimeError(
                        "typed policy-to-MMDiT bridge has no policy-approved deltas"
                    )
                typed_policy_delta_bank = PolicyRoleDeltaBank(
                    values=torch.stack(approved_values, dim=1),
                    source_names=tuple(approved_names),
                    source_depths=tuple(approved_depths),
                    protected_detail=protected_policy_detail,
                )
                typed_policy_delta_bank.validate(
                    hidden_size=int(cfg.hidden_size),
                    horizon=int(cfg.action_horizon),
                )
                typed_policy_delta_bank = self._intervene_policy_delta_bank(
                    typed_policy_delta_bank
                )
            else:
                policy_workspace_tokens = self._intervene_policy_workspace(
                    policy_workspace_tokens
                )
            if strict_role_visual_path:
                final_visual_selector = None
                final_visual_values = None
                final_visual_bias = None
            else:
                final_visual_selector = visual_memory
                final_visual_values = visual_value_memory
                final_visual_bias = torch.zeros(
                    int(visual_memory.shape[1]),
                    device=visual_memory.device,
                    dtype=torch.float32,
                )
        else:
            policy_workspace_tokens = owned_trajectory_memory
            final_visual_selector = (
                None if visual_context is None else visual_context.selector_tokens
            )
            final_visual_values = (
                None if visual_context is None else visual_context.value_tokens
            )
            final_visual_bias = None if visual_context is None else visual_context.key_bias
        context_kv = torch.cat(
            [
                canvas[:, slices["task"]],
                canvas[:, slices["state"]],
                canvas[:, slices["state_history"]],
                canvas[:, slices["executed"]],
                canvas[:, slices["proposal"]],
            ],
            dim=1,
        )
        if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
            dynamics = self.controlled_dynamics(
                rollout_init.to(device=rollout.device, dtype=rollout.dtype),
                context_kv,
                action_tokens=trajectory,
                transition_tokens=rollout,
            )
        else:
            # Preserve the exact learned-base path for historical checkpoints.
            dynamics = self.controlled_dynamics(
                rollout,
                context_kv,
                action_tokens=trajectory,
            )
        controlled_delta = dynamics["rollout_delta_pred"]
        rollout_effect_pred = dynamics["rollout_effect_pred"]
        event_context = _rollout_tokens_to_action_horizon(controlled_delta, cfg)
        decoder_mode = str(getattr(cfg, "final_action_decoder", "legacy"))
        direct_velocity: Tensor | None = None
        rollout_residual_velocity: Tensor | None = None
        rollout_alpha: Tensor | None = None
        legacy_velocity: Tensor | None = None
        pred_physical_velocity: Tensor
        legacy_event_logits: Tensor
        legacy_motion_logits: Tensor
        residual_action_flow: dict[str, Tensor] | None = None
        latent_main_action: dict[str, Tensor] | None = None
        latent_cvae_action: dict[str, Tensor] | None = None
        hierarchical_mmdit_action: dict[str, Tensor] | None = None
        evidence_latent_mmdit_action: dict[str, Tensor] | None = None
        decoder_layer_contracts = layer_contracts
        if strict_role_visual_path:
            policy_blocks = int(getattr(cfg, "flow_jepa_policy_blocks", 0))
            if policy_blocks < 1 or len(layer_contracts) < policy_blocks:
                raise RuntimeError(
                    "strict role visual path requires terminal policy layer contracts"
                )
            decoder_layer_contracts = layer_contracts[-policy_blocks:]
        if not enable_final_action_decoder:
            # Counterfactual rollout branches consume only dynamics and layer
            # contracts. Running the final CVAE/MMDiT tower here duplicated a
            # full prior decode whose action output was immediately discarded.
            pred_physical_velocity = torch.zeros_like(noisy_physical)
            legacy_event_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1]), 3
            )
            legacy_motion_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1])
            )
        elif self.evidence_latent_mmdit_action_decoder is not None:
            transition_detach = bool(int(getattr(cfg, "latent_cvae_transition_detach", 0)))

            def _evidence_transition_source(value: Tensor) -> Tensor:
                return value.detach() if transition_detach else value

            if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                transition_memory = [
                    _evidence_transition_source(controlled_delta),
                    _evidence_transition_source(event_context),
                ]
            else:
                transition_memory = [
                    _evidence_transition_source(controlled_delta),
                    _evidence_transition_source(rollout_effect_pred),
                    _evidence_transition_source(event_context),
                ]
            event_evidence = None
            if layer_contracts:
                candidate = layer_contracts[-1].get("event_logits")
                if (
                    isinstance(candidate, Tensor)
                    and candidate.ndim == 3
                    and int(candidate.shape[-1]) == 3
                ):
                    event_evidence = candidate
            if event_evidence is None:
                event_evidence = self.event_probe(event_context)
            decoder_rollout = self._intervene_bottom_far_rollout(rollout)
            evidence_latent_mmdit_action = self.evidence_latent_mmdit_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=owned_trajectory_memory,
                trajectory_workspace_tokens=owned_trajectory_memory,
                policy_action_tokens=(
                    policy_workspace_tokens
                    if (
                        int(getattr(cfg, "flow_jepa_role_hierarchy", 0))
                        and typed_policy_delta_bank is None
                    )
                    else None
                ),
                policy_role_delta_bank=typed_policy_delta_bank,
                rollout_tokens=decoder_rollout,
                transition_memory=transition_memory,
                event_evidence=event_evidence,
                state_memory=owned_state_memory,
                layer_contracts=decoder_layer_contracts,
                intent_memory=owned_intent_memory,
                visual_selector_tokens=final_visual_selector,
                visual_value_tokens=final_visual_values,
                visual_key_bias=final_visual_bias,
                collect_diagnostics=collect_diagnostics,
                evidence_scale=float(getattr(cfg, "latent_cvae_mmdit_evidence_scale", 1.0)),
                noisy_scale=float(getattr(cfg, "latent_cvae_mmdit_noisy_scale", 1.0)),
            )
            pred_physical_velocity = evidence_latent_mmdit_action["pred_velocity"]
            legacy_event_logits = evidence_latent_mmdit_action["event_logits"]
            legacy_motion_logits = evidence_latent_mmdit_action["motion_logits"]
        elif self.hierarchical_mmdit_action_decoder is not None:
            if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                transition_memory = [controlled_delta]
            else:
                transition_memory = [controlled_delta, rollout_effect_pred]
            event_evidence = None
            if layer_contracts:
                candidate = layer_contracts[-1].get("event_logits")
                if (
                    isinstance(candidate, Tensor)
                    and candidate.ndim == 3
                    and int(candidate.shape[-1]) == 3
                ):
                    event_evidence = candidate
            if event_evidence is None:
                event_evidence = self.event_probe(event_context)
            hierarchical_mmdit_action = self.hierarchical_mmdit_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=owned_trajectory_memory,
                trajectory_workspace_tokens=owned_trajectory_memory,
                rollout_tokens=rollout,
                transition_memory=transition_memory,
                event_evidence=event_evidence,
                state_memory=owned_state_memory,
                intent_memory=owned_intent_memory,
                layer_contracts=layer_contracts,
                collect_diagnostics=collect_diagnostics,
            )
            pred_physical_velocity = hierarchical_mmdit_action["pred_velocity"]
            legacy_event_logits = hierarchical_mmdit_action["event_logits"]
            legacy_motion_logits = hierarchical_mmdit_action["motion_logits"]
        elif self.latent_cvae_action_decoder is not None:
            context_memory = (
                [
                    canvas[:, slices["state"]],
                    canvas[:, slices["state_history"]],
                    canvas[:, slices["executed"]],
                    canvas[:, slices["proposal"]],
                ]
                if int(getattr(cfg, "latent_cvae_context_memory", 0))
                else None
            )
            # Rollout has its own full-resolution workspace source. Transition
            # memory therefore carries only explicit consequence semantics and
            # does not duplicate the same rollout grid through a pooled path.
            if int(getattr(cfg, "latent_cvae_transition_memory", 1)):
                if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                    # effect == delta under a fixed-zero base. Feeding both would
                    # duplicate one condition under two semantic names.
                    transition_memory = [controlled_delta, event_context]
                else:
                    transition_memory = [controlled_delta, rollout_effect_pred, event_context]
            else:
                transition_memory = None
            latent_cvae_action = self.latent_cvae_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                trajectory_workspace_tokens=trajectory,
                rollout_tokens=rollout,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory
                if int(getattr(cfg, "latent_cvae_visual_memory", 0))
                else None,
                layer_contracts=layer_contracts,
                target_physical=cvae_target_physical,
            )
            pred_physical_velocity = latent_cvae_action["pred_velocity"]
            legacy_event_logits = latent_cvae_action["event_logits"]
            legacy_motion_logits = latent_cvae_action["motion_logits"]
        elif self.latent_main_action_decoder is not None:
            context_memory = (
                context_kv if int(getattr(cfg, "latent_action_context_memory", 0)) else None
            )
            transition_parts = [rollout, controlled_delta, event_context]
            if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                transition_parts.insert(2, rollout_effect_pred)
            transition_memory = (
                torch.cat(transition_parts, dim=1)
                if int(getattr(cfg, "latent_action_transition_memory", 1))
                else None
            )
            latent_main_action = self.latent_main_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory
                if int(getattr(cfg, "latent_action_visual_memory", 0))
                else None,
                layer_contracts=layer_contracts,
            )
            pred_physical_velocity = latent_main_action["pred_velocity"]
            legacy_event_logits = latent_main_action["event_logits"]
            legacy_motion_logits = latent_main_action["motion_logits"]
        else:
            # Legacy action readers are needed only by legacy/residual decoder
            # modes. CVAE/MMDiT is a complete final path, so computing a second
            # rollout-to-action tower there wastes memory and creates misleading
            # anchor diagnostics for a path that deployment never uses.
            direct_velocity = self.direct_physical_head(trajectory)
            rollout_residual_velocity, rollout_alpha = self.rollout_residual_head(
                trajectory_pooled, controlled_delta
            )
            legacy_velocity = direct_velocity + rollout_residual_velocity
            pred_physical_velocity = legacy_velocity
            legacy_event_logits = self.event_probe(event_context)
            legacy_motion_logits = self.motion_probe(trajectory_pooled.detach()).squeeze(-1)
        if (
            self.latent_cvae_action_decoder is None
            and self.latent_main_action_decoder is None
            and self.hierarchical_mmdit_action_decoder is None
            and self.evidence_latent_mmdit_action_decoder is None
            and self.residual_action_flow_denoiser is not None
        ):
            assert legacy_velocity is not None
            if decoder_mode == "layered_residual_action_flow":
                context_memory = (
                    torch.cat([context_kv, registers], dim=1)
                    if int(getattr(cfg, "action_flow_residual_context_memory", 1))
                    else context_kv
                )
                transition_parts = [rollout, controlled_delta, event_context]
                if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                    transition_parts.insert(2, rollout_effect_pred)
                transition_memory = (
                    torch.cat(transition_parts, dim=1)
                    if int(getattr(cfg, "action_flow_residual_transition_memory", 1))
                    else None
                )
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    context_memory=context_memory,
                    transition_memory=transition_memory,
                    visual_memory=visual_memory
                    if int(getattr(cfg, "action_flow_residual_visual_memory", 1))
                    else None,
                    layer_contracts=layer_contracts,
                )
            else:
                memory_parts: list[Tensor] = []
                if int(getattr(cfg, "action_flow_residual_context_memory", 1)):
                    memory_parts.append(context_kv)
                    memory_parts.append(registers)
                if int(getattr(cfg, "action_flow_residual_transition_memory", 1)):
                    memory_parts.extend(
                        [rollout, controlled_delta, rollout_effect_pred, event_context]
                    )
                if int(getattr(cfg, "action_flow_residual_visual_memory", 1)):
                    memory_parts.append(visual_memory)
                if int(getattr(cfg, "action_flow_residual_layer_memory", 1)) and layer_contracts:
                    last_layer = layer_contracts[-1]
                    for key in (
                        "policy_effect_time_tokens",
                        "policy_effect_tokens",
                        "rollout_effect_pred",
                        "rollout_delta_pred",
                    ):
                        value = last_layer.get(key)
                        if (
                            isinstance(value, Tensor)
                            and value.ndim == 3
                            and value.shape[-1] == cfg.hidden_size
                        ):
                            memory_parts.append(value)
                residual_memory = torch.cat(memory_parts, dim=1) if memory_parts else context_kv
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    memory=residual_memory,
                )
            pred_physical_velocity = legacy_velocity + residual_action_flow["residual_velocity"]
            legacy_event_logits = legacy_event_logits + residual_action_flow["event_delta_logits"]
            legacy_motion_logits = (
                legacy_motion_logits + residual_action_flow["motion_delta_logits"]
            )
        if not collect_diagnostics:
            self._record_action_path_route_metrics(
                role_delta_metrics,
                evidence_latent_mmdit_action,
            )
            minimal = {
                "pred_physical_velocity": pred_physical_velocity,
                "event_logits": legacy_event_logits,
                "motion_logits": legacy_motion_logits,
            }
            if (
                hierarchical_mmdit_action is not None
                and "pred_velocity_coefficients" in hierarchical_mmdit_action
            ):
                minimal["pred_velocity_coefficients"] = hierarchical_mmdit_action[
                    "pred_velocity_coefficients"
                ]
            return minimal
        gate_mean = {
            key: torch.stack([row[key] for row in gate_rows]).mean()
            if gate_rows
            else _zeros_like_scalar(canvas)
            for key in (
                "gate_self",
                "gate_visual",
                "gate_stage",
                "gate_stage_to_window",
                "stage_to_window_update_norm",
                "gate_rollout",
                "gate_ffn",
                "residual_contract_enabled",
                "residual_contract_max_rms",
                "residual_contract_after_gate",
                "residual_raw_rms",
                "residual_proposed_rms",
                "residual_bounded_rms",
                "residual_written_rms",
                "residual_compression",
                "normalization_contract_enabled",
            )
        }
        if gate_rows:
            gate_mean["normalization_denominator_min"] = torch.stack(
                [row["normalization_denominator_min"] for row in gate_rows]
            ).amin()
            gate_mean["normalization_gain_max"] = torch.stack(
                [row["normalization_gain_max"] for row in gate_rows]
            ).amax()
        else:
            gate_mean["normalization_denominator_min"] = canvas.new_ones(
                (), dtype=torch.float32
            )
            gate_mean["normalization_gain_max"] = canvas.new_ones(
                (), dtype=torch.float32
            )
        content_norm = (
            torch.stack(content_norm_rows).mean()
            if content_norm_rows
            else _zeros_like_scalar(canvas)
        )
        time_norm = (
            torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
        )
        with torch.no_grad():
            rollout_seed_final = self.final_norm(
                rollout_seed.to(device=rollout.device, dtype=rollout.dtype)
            )
            rollout_deep_update_norm = (
                (rollout.detach() - rollout_seed_final).float().norm(dim=-1).mean()
            )
        out = {
            **midcut,
            "layer_contracts": layer_contracts,
            "canvas_tokens": canvas,
            "trajectory_tokens": trajectory,
            "stage_tokens": stage_tokens,
            "rollout_tokens": rollout,
            "register_tokens": registers,
            "rollout_deep_update_norm": rollout_deep_update_norm,
            "rollout_deep_token_norm": rollout.detach().float().norm(dim=-1).mean(),
            "pred_physical_velocity": pred_physical_velocity,
            "action_flow_residual_velocity": (
                torch.zeros_like(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow["residual_velocity"]
            ),
            "action_flow_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow["residual_norm"]
            ),
            "action_flow_raw_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow["raw_residual_norm"]
            ),
            "action_flow_residual_alpha_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow["alpha_mean"]
            ),
            "action_flow_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow.get(
                    "stage_router_entropy", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "action_flow_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None
                else residual_action_flow.get(
                    "stage_router_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "stage_router_entropy", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "stage_router_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "layer_memory_count", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_temporal_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "temporal_action_update_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_temporal_near_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "temporal_near_depth", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_action_temporal_mid_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None
                else latent_main_action.get(
                    "temporal_mid_depth", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_kl": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get("cvae_kl", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_prior_std", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_post_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_post_std", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_z_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_condition_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_condition_scan_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_condition_scan_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_condition_lateral_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_condition_lateral_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_layer_summary_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_layer_summary_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_transition_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_transition_condition_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_transition_source_raw_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_transition_source_raw_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_rollout_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_rollout_token_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_rollout_token_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_rollout_token_count", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_consequence_scale_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_consequence_scale_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_consequence_gate_preference": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_consequence_gate_preference", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_consequence_mix_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_consequence_mix_ratio", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_posterior_used": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_posterior_used", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "layer_memory_count", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_prior_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_prior_z_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_post_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_post_z_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mu_gap": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mu_gap", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_prior_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_prior_pred_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_post_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_post_pred_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_post_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_post_gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_refine_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_refine_update_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_noisy_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_noisy_gate_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_noisy_branch_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_noisy_branch_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_noisy_branch_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_noisy_branch_ratio", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_route_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_route_entropy", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_route_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_route_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_route_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_route_effective_slots",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_progress_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_entropy", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_progress_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_progress_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_effective_slots",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_progress_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_continue_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_continue_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_prefix_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_prefix_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_progress_seed_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_seed_entropy",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_progress_seed_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_seed_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_progress_seed_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_seed_effective_slots",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_progress_seed_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_progress_seed_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_route_temperature_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_route_temperature_mean",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_semantic_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_semantic_bias_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_output_adapter_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_output_adapter_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_function_delta_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_function_delta_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_base_highfreq_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_base_highfreq_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_refine_step_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_refine_step_bias_norm",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_capsule_layer_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_capsule_layer_entropy",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_capsule_layer_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_capsule_layer_max", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_capsule_layer_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_capsule_layer_effective_slots",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_condition_strength_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_condition_strength_mean",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_condition_strength_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_condition_strength_std",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_condition_strength_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_condition_strength_max",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_condition_strength_min": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_condition_strength_min",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_condition_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_condition_residual_norm",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_context_direction_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_context_direction_norm",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_micro_step_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_step_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_step_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_step_std", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_progress_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_progress_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_kp_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_kp_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_kd_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_kd_mean", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_feedforward_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_feedforward_norm",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_micro_feedback_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_feedback_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_damping_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_damping_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_function_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_function_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_control_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_control_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_update_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_heun_error": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_heun_error", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_micro_refine_block_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_micro_refine_block_norm",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_adaptive_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_regularizer", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_adaptive_route_entropy_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_adaptive_route_entropy_regularizer",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_mmdit_action_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_update_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_cond_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_cond_update_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_action_cond_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_cond_attention", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_action_noisy_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_noisy_attention", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_action_rollout_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_rollout_attention",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_mmdit_action_rollout_enrichment": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_rollout_enrichment",
                    _zeros_like_scalar(pred_physical_velocity),
                )
            ),
            "latent_cvae_mmdit_action_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_action_token_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_condition_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_condition_token_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "latent_cvae_mmdit_noisy_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None
                else latent_cvae_action.get(
                    "cvae_mmdit_noisy_token_norm", _zeros_like_scalar(pred_physical_velocity)
                )
            ),
            "rollout_effect_pred": rollout_effect_pred,
            "rollout_base_effect_pred": dynamics["rollout_base_effect_pred"],
            "rollout_delta_pred": controlled_delta,
            "rollout_coeff_abs_mean": dynamics["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": dynamics["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": dynamics["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": dynamics["rollout_basis_norm"],
            "rollout_delta_norm": dynamics["rollout_delta_norm"],
            "rollout_base_norm": dynamics["rollout_base_norm"],
            "rollout_decomposition_expansion_ratio": dynamics[
                "rollout_decomposition_expansion_ratio"
            ],
            "rollout_base_is_fixed_zero": dynamics["rollout_base_is_fixed_zero"],
            "rollout_delta_gain": dynamics["rollout_delta_gain"],
            "future_latent_pred": rollout_effect_pred,
            "action_effect_pred": rollout_effect_pred,
            "event_logits": legacy_event_logits,
            "motion_logits": legacy_motion_logits,
            "transition_latent": (
                event_context
                if hierarchical_mmdit_action is None
                else hierarchical_mmdit_action["transition_latent"]
            ),
            "gate_self": gate_mean["gate_self"],
            "gate_visual": gate_mean["gate_visual"],
            "gate_stage": gate_mean["gate_stage"],
            "gate_stage_to_window": gate_mean["gate_stage_to_window"],
            "stage_to_window_update_norm": gate_mean["stage_to_window_update_norm"],
            "gate_rollout": gate_mean["gate_rollout"],
            "gate_ffn": gate_mean["gate_ffn"],
            "role_residual_contract_enabled": gate_mean[
                "residual_contract_enabled"
            ],
            "role_residual_contract_max_rms": gate_mean[
                "residual_contract_max_rms"
            ],
            "role_residual_contract_after_gate": gate_mean[
                "residual_contract_after_gate"
            ],
            "role_residual_raw_rms": gate_mean["residual_raw_rms"],
            "role_residual_proposed_rms": gate_mean[
                "residual_proposed_rms"
            ],
            "role_residual_bounded_rms": gate_mean[
                "residual_bounded_rms"
            ],
            "role_residual_written_rms": gate_mean[
                "residual_written_rms"
            ],
            "role_residual_compression": gate_mean[
                "residual_compression"
            ],
            "role_normalization_contract_enabled": gate_mean[
                "normalization_contract_enabled"
            ],
            "role_normalization_denominator_min": torch.minimum(
                gate_mean["normalization_denominator_min"],
                terminal_norm_denominator,
            ),
            "role_normalization_gain_max": torch.maximum(
                gate_mean["normalization_gain_max"],
                terminal_norm_gain,
            ),
            "terminal_normalization_denominator_min": (
                terminal_norm_denominator
            ),
            "terminal_normalization_gain_max": terminal_norm_gain,
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.zeros((), device=canvas.device, dtype=canvas.dtype),
        }
        role_depths = {"grounding": 0, "world": 0, "policy": 0, "shared": 0}
        role_written_rows: dict[str, list[Tensor]] = {
            "grounding": [],
            "world": [],
            "policy": [],
            "shared": [],
        }
        for role_name, row in zip(self.block_roles, gate_rows, strict=True):
            role_depths[role_name] += 1
            role_label = (
                f"{role_name[0]}{role_depths[role_name]}"
                if role_name != "shared"
                else f"s{role_depths[role_name]}"
            )
            for sublayer in (
                "self",
                "visual",
                "stage",
                "stage_to_window",
                "rollout",
                "ffn",
            ):
                for statistic in (
                    "raw_rms",
                    "proposed_rms",
                    "bounded_rms",
                    "written_rms",
                    "compression",
                ):
                    source_key = f"residual_{sublayer}_{statistic}"
                    if source_key in row:
                        out[
                            f"role_residual_{role_label}_{sublayer}_{statistic}"
                        ] = row[source_key]
            if "residual_written_rms" in row:
                role_written_rows[role_name].append(row["residual_written_rms"])
        for role_name in ("grounding", "world", "policy"):
            if role_written_rows[role_name]:
                out[
                    f"role_residual_{role_name}_written_rms_max"
                ] = torch.stack(role_written_rows[role_name]).amax()
                out[
                    f"role_residual_{role_name}_written_rms_mean"
                ] = torch.stack(role_written_rows[role_name]).mean()
        if goal_tokens is not None:
            goal_metric = owned_intent_memory["task"].detach().float()
            out["flow_jepa_goal_condition_norm"] = goal_metric.norm(dim=-1).mean()
            out["flow_jepa_goal_token_count"] = goal_metric.new_tensor(
                float(goal_metric.shape[1])
            )
            if int(goal_metric.shape[1]) > 1:
                normalized_goal = F.normalize(goal_metric, dim=-1)
                similarity = normalized_goal @ normalized_goal.transpose(1, 2)
                pair_mask = ~torch.eye(
                    int(goal_metric.shape[1]),
                    device=goal_metric.device,
                    dtype=torch.bool,
                )
                out["flow_jepa_goal_pair_cosine"] = similarity[:, pair_mask].mean()
        if executed_memory is not None:
            action_metric = owned_intent_memory["executed"].detach().float()
            out["flow_jepa_action_condition_norm"] = action_metric.norm(dim=-1).mean()
            out["flow_jepa_action_memory_token_count"] = action_metric.new_tensor(
                float(action_metric.shape[1])
            )
            if goal_tokens is not None:
                out["flow_jepa_goal_action_cosine"] = (
                    F.normalize(goal_metric.mean(dim=1), dim=-1)
                    * F.normalize(action_metric.mean(dim=1), dim=-1)
                ).sum(dim=-1).mean()
        if legacy_velocity is not None:
            assert direct_velocity is not None
            assert rollout_residual_velocity is not None
            assert rollout_alpha is not None
            out.update(
                {
                    "direct_physical_velocity": direct_velocity,
                    "rollout_residual_velocity": rollout_residual_velocity,
                    "legacy_physical_velocity": legacy_velocity,
                    "rollout_alpha": rollout_alpha,
                }
            )
        if latent_cvae_action is not None:
            for key, value in latent_cvae_action.items():
                if key.startswith("cvae_") and isinstance(value, Tensor):
                    out.setdefault(f"latent_{key}", value)
        if latent_cvae_action is not None and "post_pred_velocity" in latent_cvae_action:
            out.update(
                {
                    "post_pred_velocity": latent_cvae_action["post_pred_velocity"],
                    "post_event_logits": latent_cvae_action.get(
                        "post_event_logits", legacy_event_logits
                    ),
                    "post_motion_logits": latent_cvae_action.get(
                        "post_motion_logits", legacy_motion_logits
                    ),
                }
            )
        if latent_cvae_action is not None:
            for key in (
                "cvae_adaptive_micro_controller_norm",
                "cvae_adaptive_micro_pred_velocity",
                "cvae_adaptive_micro_event_logits",
                "cvae_adaptive_micro_supervision_logits",
            ):
                if key in latent_cvae_action:
                    out[f"latent_{key}"] = latent_cvae_action[key]
        if hierarchical_mmdit_action is not None:
            for key in tuple(out):
                if key.startswith("latent_cvae_"):
                    out.pop(key)
            if "pred_velocity_coefficients" in hierarchical_mmdit_action:
                out["pred_velocity_coefficients"] = hierarchical_mmdit_action[
                    "pred_velocity_coefficients"
                ]
            for key, value in hierarchical_mmdit_action.items():
                if not isinstance(value, Tensor):
                    continue
                if key.startswith(
                    (
                        "intent_",
                        "owned_",
                        "hierarchical_mmdit_",
                        "refinement_probe_",
                        "refinement_shadow_probe_",
                    )
                ):
                    out[key] = value
        if evidence_latent_mmdit_action is not None:
            for key, value in evidence_latent_mmdit_action.items():
                if isinstance(value, Tensor) and key not in {
                    "pred_velocity",
                    "event_logits",
                    "motion_logits",
                }:
                    out[key] = value
        if visual_context is not None:
            if self.flow_dino_evidence is None:
                raise RuntimeError("Flow-DINO visual context has no owning encoder")
            address_bank = (
                None
                if late_raw_detail is None
                else late_raw_detail.address_bank
            )
            if online_horizon_address:
                if not online_horizon_address_applied:
                    raise RuntimeError(
                        "online horizon address was not applied before the action path"
                    )
                # V108 predicts from the same final carrier consumed by the
                # deployed action path.  The observation bank was read once at
                # G3 -> W1 and is deliberately not revisited here.
                future_prediction = self.flow_dino_evidence.predict_future(rollout)
            elif progressive_grounding_address:
                if progressive_address_state is None:
                    raise RuntimeError(
                        "progressive horizon posterior lost the G3 state"
                    )
                if "flow_jepa_horizon_address_logits" not in future_address_metrics:
                    raise RuntimeError(
                        "progressive horizon posterior was not formed at W->P"
                    )
                # V109 predicts from the same final carrier as deployment.  Its
                # teacher-facing relevance and the source prior consumed by P
                # were formed together at W->P, before any P block could alter
                # the W-owned selector state.
                future_prediction = self.flow_dino_evidence.predict_future(
                    rollout
                )
            else:
                (
                    future_prediction,
                    future_address_metrics,
                ) = self.flow_dino_evidence.predict_future_with_address(
                    rollout,
                    address_bank,
                    enable_address=bool(self.training or collect_diagnostics),
                )
            out["flow_jepa_future_pred"] = future_prediction
            out.update(future_address_metrics)
            out.update(horizon_boundary_metrics)
            if self.flow_dino_evidence.predictive_change_contract:
                # Keep the historical key as the supervised prediction tensor
                # for caller compatibility, while the explicit alias makes its
                # changed semantics impossible for the loss code to miss.
                out["flow_jepa_future_delta_pred"] = future_prediction
                out["flow_jepa_future_delta_prediction_norm"] = (
                    future_prediction.detach().float().norm(dim=-1).mean()
                )
            out["flow_jepa_future_target_mask"] = visual_context.future_target_mask
            out["flow_jepa_future_offsets"] = tuple(
                int(value) for value in self.flow_dino_evidence.window_offsets
            )
            if interval_stage_prediction is not None:
                out["flow_jepa_interval_progress_pred"] = (
                    interval_stage_prediction
                )
                out["flow_jepa_interval_stage_enabled"] = (
                    interval_stage_prediction.new_ones((), dtype=torch.float32)
                )
                out["flow_jepa_variance_safe_routing"] = (
                    interval_stage_prediction.new_tensor(
                        float(
                            bool(
                                int(
                                    getattr(
                                        cfg,
                                        "flow_jepa_variance_safe_routing",
                                        0,
                                    )
                                )
                            )
                        ),
                        dtype=torch.float32,
                    )
                )
                out["flow_jepa_interval_stage_windows"] = tuple(
                    tuple(int(value) for value in window)
                    for window in cfg.flow_jepa_interval_windows
                )
            if self.flow_dino_evidence.late_bottleneck:
                if int(stage_tokens.shape[1]) != 0:
                    raise RuntimeError("late-bottleneck canvas unexpectedly materialized stage tokens")
                grouped_future = rollout.detach().float().reshape(
                    rollout.shape[0],
                    int(cfg.future_anchors),
                    -1,
                    rollout.shape[-1],
                ).mean(dim=2)
                adjacent = F.cosine_similarity(
                    grouped_future[:, 1:], grouped_future[:, :-1], dim=-1
                )
                out["flow_jepa_horizon_adjacent_cosine"] = adjacent.mean()
                out["flow_jepa_far_horizon_norm"] = grouped_future[:, -1].norm(dim=-1).mean()
            else:
                if int(stage_tokens.shape[1]) != 1:
                    raise RuntimeError(
                        "hierarchical Flow-DINO canvas did not preserve one stage token"
                    )
                out["flow_jepa_stage_pred"] = self.flow_dino_evidence.predict_stage(stage_tokens)
                stage_f = stage_tokens.detach().float()
                window_f = rollout.detach().float().mean(dim=1, keepdim=True)
                out["flow_jepa_stage_token_norm"] = stage_f.norm(dim=-1).mean()
                out["flow_jepa_stage_window_cosine"] = (
                    F.normalize(stage_f, dim=-1) * F.normalize(window_f, dim=-1)
                ).sum(dim=-1).mean()
                out["flow_jepa_stage_dynamics_gate"] = gate_mean["gate_stage"].detach().float()
                out["flow_jepa_stage_to_window_gate"] = gate_mean[
                    "gate_stage_to_window"
                ].detach().float()
                out["flow_jepa_stage_to_window_update_norm"] = gate_mean[
                    "stage_to_window_update_norm"
                ].detach().float()
            for key, value in visual_context.losses.items():
                out[key] = value
            for key, value in visual_context.metrics.items():
                out[key] = value
            for key, value in raw_refinement_metrics.items():
                out[key] = value
            for key, value in late_detail_metrics.items():
                out[key] = value
            out["flow_jepa_policy_modulation_visual_free"] = torch.as_tensor(
                float(strict_role_visual_path),
                device=canvas.device,
            )
            out["flow_jepa_world_anchor_write_only"] = torch.as_tensor(
                float(
                    bool(
                        int(
                            getattr(
                                cfg,
                                "flow_jepa_world_anchor_write_only",
                                0,
                            )
                        )
                    )
                ),
                device=canvas.device,
            )
        for key, value in role_delta_metrics.items():
            out[key] = value
        for key, value in phase_metrics.items():
            out[key] = value
        # The frozen model-path probe must observe the actual soft routing
        # used by the deployed forward. Capture only scalar factual state;
        # this evaluation-only branch never modifies the action graph.
        self._record_action_path_route_metrics(out)
        return out

    @torch.no_grad()
    def target_rollout_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        return self.rollout_codec.target_effect(visual, target_visual)

    @torch.no_grad()
    def flow_jepa_teacher_target(
        self, target_visual: Tensor, current_visual: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.flow_dino_evidence is None:
            raise RuntimeError("Flow-DINO JEPA teacher requested while the feature is disabled")
        return self.flow_dino_evidence.teacher_target(target_visual, current_visual)

    @torch.no_grad()
    def flow_jepa_interval_teacher_targets(
        self,
        target_visual: Tensor,
        current_visual: Tensor,
    ) -> dict[str, Tensor]:
        if self.flow_dino_evidence is None:
            raise RuntimeError(
                "interval teacher requested while Flow-DINO is disabled"
            )
        return self.flow_dino_evidence.teacher_interval_targets(
            target_visual,
            current_visual,
        )
