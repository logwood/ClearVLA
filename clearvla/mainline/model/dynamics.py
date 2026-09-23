"""Schema25-R1 W1/W2 typed four-interval object dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..executed_world import ExecutedWorldPrediction
from ..future_time import LEGACY_FUTURE_TIME, resolve_future_time
from ..world_control import (
    KNOWN_PREFIX_WORLD_CONTROL,
    LEGACY_WORLD_CONTROL,
    CandidateControlDomain,
)
from ..world_robot import NO_ROBOT_WORLD, OBSERVED_ROBOT_VIEWS
from .robot_relation import RobotObjectRelation, RobotObjectRelationEncoder
from .routing import (
    VarianceFlooredCenteredNorm,
    register_gradient_axis_rms_metrics,
    register_gradient_rms_metric,
    smooth_rms_contract,
)
from .types import (
    ExecutedActionSequenceCondition,
    FutureObjectDynamics,
    ObjectFactSet,
    ObjectWorldBelief,
    PhysicalActionCondition,
    PhysicalActionSequenceCondition,
    RecordedActionSequenceCondition,
    WorldActionCondition,
)


@dataclass(frozen=True)
class ObjectW1WorkingState:
    """Private W carrier; only the final decoded field reaches P."""

    near: Tensor  # completed generic near condition [B,2,K,H]
    far_base: Tensor  # unprocessed generic far condition [B,2,K,H]
    common_typed: Tensor  # completed once by W1 [B,K,3,H]
    near_interval_innovation: Tensor  # completed by W1 [B,2,K,3,H]
    far_interval_innovation: Tensor  # raw W2-owned input [B,2,K,3,H]
    control_domain: CandidateControlDomain | None = None
    robot_relation: RobotObjectRelation | None = None
    time_grid_mode: str = LEGACY_FUTURE_TIME


class _ObjectIntervalBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        typed_normalization_floor: float,
    ) -> None:
        super().__init__()
        # The ordinary norms are the inherited public/generic W path.  Typed
        # S values carry relevance amplitude, so their attention path uses a
        # separate parameter-free variance floor instead of erasing that
        # amplitude through unit-variance normalization.
        self.object_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.typed_object_norm = VarianceFlooredCenteredNorm(
            typed_normalization_floor
        )
        self.object_attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.interval_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.typed_interval_norm = VarianceFlooredCenteredNorm(
            typed_normalization_floor
        )
        self.interval_attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.typed_ffn_norm = VarianceFlooredCenteredNorm(
            typed_normalization_floor
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    @overload
    def forward(
        self,
        value: Tensor,
        *,
        causal_interval: bool,
        typed: Literal[False] = False,
        collect_diagnostics: bool = False,
        object_support: Tensor | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        value: Tensor,
        *,
        causal_interval: bool,
        typed: Literal[True],
        collect_diagnostics: bool = False,
        object_support: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

    def forward(
        self,
        value: Tensor,
        *,
        causal_interval: bool,
        typed: bool = False,
        collect_diagnostics: bool = False,
        object_support: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if typed:
            return self.forward_typed(
                value,
                causal_interval=causal_interval,
                collect_diagnostics=collect_diagnostics,
                object_support=object_support,
            )
        batch, intervals, objects, hidden = value.shape
        support: Tensor | None = None
        key_padding_mask: Tensor | None = None
        if object_support is not None:
            if object_support.ndim != 2 or tuple(object_support.shape) != (
                batch,
                objects,
            ):
                raise ValueError("W object support must be [B,K]")
            support = object_support.to(device=value.device, dtype=torch.bool)
            # Quarantine invalid K rows before normalization and before the
            # object-axis reduction.  A dummy zero key is unmasked for an
            # all-invalid batch row because PyTorch MHA otherwise returns NaN.
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )
            key_padding_mask = (~support)[:, None, :].expand(
                -1, intervals, -1
            ).reshape(batch * intervals, objects)
            all_invalid = (~support.any(dim=-1)).repeat_interleave(intervals)
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_invalid
        object_view = value.reshape(batch * intervals, objects, hidden)
        normalized = self.object_norm(object_view)
        update, _ = self.object_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape_as(value)
        if support is not None:
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )
        interval_view = value.transpose(1, 2).reshape(batch * objects, intervals, hidden)
        normalized = self.interval_norm(interval_view)
        mask = (
            torch.triu(
                torch.ones(intervals, intervals, device=value.device, dtype=torch.bool),
                diagonal=1,
            )
            if causal_interval
            else None
        )
        update, _ = self.interval_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape(batch, objects, intervals, hidden).transpose(1, 2)
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        value = value + ffn
        if support is not None:
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )
        return value

    @staticmethod
    def _normalization_statistics(
        value: Tensor,
        normalized: Tensor,
        denominator: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        input_rms = value.detach().float().square().mean(dim=-1).sqrt()
        output_rms = normalized.detach().float().square().mean(dim=-1).sqrt()
        ratio = torch.where(
            input_rms > 0.0,
            output_rms / input_rms.clamp_min(1.0e-12),
            input_rms.new_zeros(input_rms.shape),
        )
        denominator_f = denominator.detach().float()
        return (
            denominator_f.amin(),
            denominator_f.reciprocal().amax(),
            ratio.amax(),
        )

    @staticmethod
    def _merge_normalization_statistics(
        rows: tuple[tuple[Tensor, Tensor, Tensor], ...],
    ) -> dict[str, Tensor]:
        if not rows:
            return {}
        return {
            "denominator_min": torch.stack([row[0] for row in rows]).amin(),
            "gain_max": torch.stack([row[1] for row in rows]).amax(),
            "output_input_rms_ratio_max": torch.stack(
                [row[2] for row in rows]
            ).amax(),
        }

    def forward_typed(
        self,
        value: Tensor,
        *,
        causal_interval: bool,
        collect_diagnostics: bool,
        object_support: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run one typed W block without changing the generic W operator."""

        batch, intervals, objects, hidden = value.shape
        support: Tensor | None = None
        key_padding_mask: Tensor | None = None
        if object_support is not None:
            if object_support.ndim != 2 or tuple(object_support.shape) != (
                batch,
                objects,
            ):
                raise ValueError("W object support must be [B,K]")
            support = object_support.to(device=value.device, dtype=torch.bool)
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )
            key_padding_mask = (~support)[:, None, :].expand(
                -1, intervals, -1
            ).reshape(batch * intervals, objects)
            all_invalid = (~support.any(dim=-1)).repeat_interleave(intervals)
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_invalid
        statistics: list[tuple[Tensor, Tensor, Tensor]] = []

        object_view = value.reshape(batch * intervals, objects, hidden)
        normalized, denominator = self.typed_object_norm.forward_with_denominator(
            object_view
        )
        if collect_diagnostics:
            statistics.append(
                self._normalization_statistics(
                    object_view,
                    normalized,
                    denominator,
                )
            )
        update, _ = self.object_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape_as(value)
        if support is not None:
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )

        interval_view = value.transpose(1, 2).reshape(
            batch * objects, intervals, hidden
        )
        normalized, denominator = self.typed_interval_norm.forward_with_denominator(
            interval_view
        )
        if collect_diagnostics:
            statistics.append(
                self._normalization_statistics(
                    interval_view,
                    normalized,
                    denominator,
                )
            )
        mask = (
            torch.triu(
                torch.ones(
                    intervals,
                    intervals,
                    device=value.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            if causal_interval
            else None
        )
        update, _ = self.interval_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape(
            batch, objects, intervals, hidden
        ).transpose(1, 2)

        normalized, denominator = self.typed_ffn_norm.forward_with_denominator(
            value
        )
        if collect_diagnostics:
            statistics.append(
                self._normalization_statistics(
                    value,
                    normalized,
                    denominator,
                )
            )
        ffn = self.ffn[3](self.ffn[2](self.ffn[1](normalized)))
        ffn, _ = smooth_rms_contract(ffn, 0.35)
        value = value + ffn
        if support is not None:
            value = torch.where(
                support[:, None, :, None],
                value,
                torch.zeros_like(value),
            )
        return value, self._merge_normalization_statistics(
            tuple(statistics)
        )


class ObjectFutureDynamicsCompiler(nn.Module):
    """W1 owns common/near; W2 reads W1 and writes far only."""

    TYPE_NAMES = ("semantic", "appearance", "geometry")
    CAMERA_CONDITION_MODES = ("motion_prior_only", "coordinate_role_v1")
    ACTION_CONDITION_MODES = ("interval_mean_v1", "sequence_prefix_v1")
    SEQUENCE_PREFIX_ENDPOINTS = (8, 16, 24, 24)

    @staticmethod
    def _safe_object_validity(
        facts: ObjectFactSet | ObjectWorldBelief,
    ) -> Tensor:
        """Return finite producer-owned object support in FP32."""

        return torch.nan_to_num(
            facts.validity.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)

    @staticmethod
    def _supported_object_values(
        value: Tensor,
        validity: Tensor,
        *,
        object_dim: int = 1,
    ) -> Tensor:
        """Mask raw object rows before projections/normalization."""

        support = validity[..., 0] > 0.0
        axis = int(object_dim)
        if axis < 0:
            axis += value.ndim
        if not 0 <= axis < value.ndim:
            raise ValueError("W object support axis is out of range")
        if support.ndim != 2:
            raise ValueError("W object validity must be [B,K,1]")
        shape = [1] * value.ndim
        shape[0] = int(support.shape[0])
        shape[axis] = int(support.shape[1])
        mask = support.reshape(shape)
        return torch.where(mask, value, torch.zeros_like(value))

    @staticmethod
    def _safe_camera_validity(
        facts: ObjectFactSet | ObjectWorldBelief,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return finite producer-owned camera validity.

        Object validity is applied at the field boundary separately.  Keeping
        the camera lane independent preserves its established fractional
        scaling contract (for example, a half-valid camera remains exactly a
        half-strength geometry carrier even when the object read is supplied
        by a diagnostic fixture).
        """

        camera = torch.nan_to_num(
            facts.camera_validity.to(dtype=dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return camera

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        action_dim: int = 7,
        heads: int,
        normalization_floor: float = 0.25,
        camera_names: tuple[str, ...] | None = None,
        camera_condition_mode: str = "motion_prior_only",
        action_condition_mode: str = "interval_mean_v1",
        control_mode: str = LEGACY_WORLD_CONTROL,
        future_time_grid_mode: str = LEGACY_FUTURE_TIME,
        robot_condition_mode: str = NO_ROBOT_WORLD,
        state_dim: int | None = None,
        state_feature_mode: str = "native_affine_v1",
    ) -> None:
        super().__init__()
        self.time_grid = resolve_future_time(future_time_grid_mode)
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.content_dim = int(content_dim)
        self.normalization_floor = float(normalization_floor)
        if self.normalization_floor <= 0.0:
            raise ValueError("W typed normalization floor must be positive")
        self.camera_condition_mode = str(camera_condition_mode)
        if self.camera_condition_mode not in self.CAMERA_CONDITION_MODES:
            raise ValueError(
                "W camera condition mode must be motion_prior_only or "
                "coordinate_role_v1"
            )
        self.camera_names = tuple(str(name) for name in (camera_names or ()))
        if self.camera_condition_mode == "coordinate_role_v1":
            if not self.camera_names or len(set(self.camera_names)) != len(
                self.camera_names
            ):
                raise ValueError(
                    "coordinate_role_v1 requires non-empty unique camera names"
                )
            if any(not name for name in self.camera_names):
                raise ValueError("W camera role names must be non-empty")
        self.control_mode = str(control_mode)
        if self.control_mode not in {LEGACY_WORLD_CONTROL, KNOWN_PREFIX_WORLD_CONTROL}:
            raise ValueError("unknown W control domain mode")
        if self.control_mode == KNOWN_PREFIX_WORLD_CONTROL and action_condition_mode != "sequence_prefix_v1":
            raise ValueError("known-prefix world requires sequence physical controls")
        self.action_condition_mode = str(action_condition_mode)
        if self.action_condition_mode not in self.ACTION_CONDITION_MODES:
            raise ValueError(
                "W action condition mode must be interval_mean_v1 or "
                "sequence_prefix_v1"
            )
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.object_transport_prior = nn.Linear(2, hidden, bias=False)
        # Retire the old goal attention without shifting any retained W/P2/
        # bottom initialization.  The replacement physical projection uses a
        # seed-derived sidecar stream, while the main construction stream
        # advances exactly as it did through the historical MHA.
        removed_goal_read = nn.MultiheadAttention(
            hidden,
            heads,
            bias=False,
            dropout=0.0,
            batch_first=True,
        )
        del removed_goal_read
        retained_rng_state = torch.get_rng_state()
        sidecar_generator = torch.Generator(device="cpu")
        sidecar_generator.manual_seed(
            (int(torch.initial_seed()) ^ 0x5343483238) % (2**63 - 1)
        )
        # This is the sole W action ingress.  Its input is the lossless
        # [absolute, adjacent-delta] view of the normalized seven-dimensional
        # physical proposal; no coarse hidden coordinate or goal/S carrier is
        # accepted by the compiler API.
        try:
            torch.set_rng_state(sidecar_generator.get_state())
            self.physical_action_condition = nn.Linear(
                2 * int(action_dim), hidden, bias=False
            )
        finally:
            torch.set_rng_state(retained_rng_state)
        # The sequence mode reuses the trained physical row projection above.
        # Only temporal order/time encoding is new.  A private generator keeps
        # these opt-in draws from shifting any retained parameter initialization.
        self.sequence_time_condition: nn.Linear | None = None
        self.sequence_action_recurrence: nn.GRUCell | None = None
        if self.action_condition_mode == "sequence_prefix_v1":
            retained_rng_state = torch.get_rng_state()
            sequence_generator = torch.Generator(device="cpu")
            sequence_generator.manual_seed(
                (int(torch.initial_seed()) ^ 0x535145513234) % (2**63 - 1)
            )
            try:
                torch.set_rng_state(sequence_generator.get_state())
                self.sequence_time_condition = nn.Linear(1, hidden, bias=False)
                self.sequence_action_recurrence = nn.GRUCell(hidden, hidden)
            finally:
                torch.set_rng_state(retained_rng_state)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, 1, hidden) * 0.02)
        self.interval_clock_code: Tensor | None
        self.register_buffer("interval_clock_code", self.time_grid.interval_encoding(hidden)[None, :, None] if self.time_grid.aligned else None)
        self.w1 = _ObjectIntervalBlock(
            hidden,
            heads,
            typed_normalization_floor=self.normalization_floor,
        )
        self.w2 = _ObjectIntervalBlock(
            hidden,
            heads,
            typed_normalization_floor=self.normalization_floor,
        )
        self.w2_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.w1_memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.typed_w2_query_norm = VarianceFlooredCenteredNorm(
            self.normalization_floor
        )
        self.typed_w1_memory_norm = VarianceFlooredCenteredNorm(
            self.normalization_floor
        )
        self.w1_to_w2 = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.delta_head = nn.Linear(hidden, content_dim, bias=False)
        self.transport_head = nn.Linear(hidden, 2, bias=False)
        self.covariance_head = nn.Linear(hidden, 3)
        # Consume the historical construction draws while retiring these
        # heads from registration, serialization and execution.
        removed_status_heads = (
            nn.Linear(hidden, 1),
            nn.Linear(hidden, 1),
            nn.Linear(hidden, 1),
        )
        del removed_status_heads
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.covariance_head.weight)
        with torch.no_grad():
            self.covariance_head.bias.copy_(
                torch.tensor(
                    (-3.0, -3.0, 0.0),
                    dtype=self.covariance_head.bias.dtype,
                )
            )
        # This opt-in branch is a component initialization, not an exact
        # resume from a motion-prior-only checkpoint.  Preserve the inherited
        # construction RNG and initialize the new condition to an exact
        # identity so the first forward matches the accepted W graph.
        self.camera_coordinate_role_condition: nn.Linear | None = None
        if self.camera_condition_mode == "coordinate_role_v1":
            retained_rng_state = torch.get_rng_state()
            try:
                self.camera_coordinate_role_condition = nn.Linear(
                    2 + len(self.camera_names),
                    hidden,
                    bias=False,
                )
            finally:
                torch.set_rng_state(retained_rng_state)
            nn.init.zeros_(self.camera_coordinate_role_condition.weight)
            stable_roles = {
                name: index for index, name in enumerate(sorted(self.camera_names))
            }
            role_basis = torch.zeros(
                len(self.camera_names),
                len(self.camera_names),
                dtype=torch.float32,
            )
            for camera_index, name in enumerate(self.camera_names):
                role_basis[camera_index, stable_roles[name]] = 1.0
        else:
            role_basis = torch.empty(0, 0, dtype=torch.float32)
        self.register_buffer(
            "_camera_role_basis",
            role_basis,
            persistent=False,
        )
        self.robot_condition_mode = str(robot_condition_mode)
        self.robot_relation_encoder: RobotObjectRelationEncoder | None = None
        if self.robot_condition_mode == OBSERVED_ROBOT_VIEWS:
            if state_dim is None or self.camera_condition_mode != "coordinate_role_v1":
                raise ValueError("robot-object W requires explicit state width and named view conditions")
            retained_rng_state = torch.get_rng_state()
            try:
                self.robot_relation_encoder = RobotObjectRelationEncoder(
                    state_dim=state_dim, state_mode=state_feature_mode,
                    content_dim=content_dim, hidden=hidden, camera_names=self.camera_names,
                )
            finally:
                torch.set_rng_state(retained_rng_state)
        elif self.robot_condition_mode != NO_ROBOT_WORLD:
            raise ValueError("unknown W robot observation mode")

    def _robot_relation(self, facts: ObjectFactSet | ObjectWorldBelief) -> RobotObjectRelation | None:
        if self.robot_relation_encoder is None:
            if isinstance(facts, ObjectWorldBelief) and facts.robot_observation is not None:
                raise ValueError("legacy W cannot silently consume an undeclared robot observation")
            return None
        if not isinstance(facts, ObjectWorldBelief):
            raise ValueError("robot-object W requires a compact belief with current robot observation")
        return self.robot_relation_encoder(facts)


    @staticmethod
    def _zero_preserving_condition(
        value: Tensor,
        condition: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Condition an existing owner without creating one from exact zero."""

        if tuple(value.shape) != tuple(condition.shape):
            raise ValueError("zero-preserving W condition axes do not align")
        modulation = value.float() * torch.tanh(condition.float())
        return (value.float() + modulation).to(dtype=value.dtype), modulation

    @staticmethod
    def _appearance_condition_semantic(
        semantic: Tensor,
        appearance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Use appearance as semantic evidence, never as a status value."""

        if tuple(semantic.shape) != tuple(appearance.shape):
            raise ValueError("W semantic and appearance owner axes do not align")
        return ObjectFutureDynamicsCompiler._zero_preserving_condition(
            semantic,
            appearance,
        )

    @staticmethod
    def _run_typed_block(
        block: _ObjectIntervalBlock,
        value: Tensor,
        *,
        causal_interval: bool,
        collect_diagnostics: bool,
        object_support: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if value.ndim != 5 or int(value.shape[3]) != 3:
            raise ValueError("typed W state must be [B,I,K,3,H]")
        batch, intervals, objects, types, hidden = value.shape
        typed_support: Tensor | None = None
        if object_support is not None:
            if object_support.ndim != 2 or tuple(object_support.shape) != (
                batch,
                objects,
            ):
                raise ValueError("typed W object support must be [B,K]")
            typed_support = object_support[:, None, :].expand(
                -1, types, -1
            ).reshape(batch * types, objects)
        typed_batch = value.permute(0, 3, 1, 2, 4).reshape(
            batch * types, intervals, objects, hidden
        )
        result = block(
            typed_batch,
            causal_interval=causal_interval,
            typed=True,
            collect_diagnostics=collect_diagnostics,
            object_support=typed_support,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("typed W block did not return its diagnostic boundary")
        typed_batch, metrics = result
        return (
            typed_batch.reshape(
                batch, types, intervals, objects, hidden
            ).permute(0, 2, 3, 1, 4),
            metrics,
        )

    @staticmethod
    def _merge_normalization_metrics(
        *rows: dict[str, Tensor],
        prefix: str,
    ) -> dict[str, Tensor]:
        active = tuple(row for row in rows if row)
        if not active:
            return {}
        return {
            f"{prefix}_denominator_min": torch.stack(
                [row["denominator_min"] for row in active]
            ).amin(),
            f"{prefix}_gain_max": torch.stack(
                [row["gain_max"] for row in active]
            ).amax(),
            f"{prefix}_output_input_rms_ratio_max": torch.stack(
                [row["output_input_rms_ratio_max"] for row in active]
            ).amax(),
        }

    @staticmethod
    def _normalization_metrics(
        value: Tensor,
        normalized: Tensor,
        denominator: Tensor,
    ) -> dict[str, Tensor]:
        rows = _ObjectIntervalBlock._normalization_statistics(
            value,
            normalized,
            denominator,
        )
        return {
            "denominator_min": rows[0],
            "gain_max": rows[1],
            "output_input_rms_ratio_max": rows[2],
        }

    def _base(
        self,
        facts: ObjectFactSet | ObjectWorldBelief,
        action: WorldActionCondition | RecordedActionSequenceCondition,
        *,
        collect_diagnostics: bool,
        robot_relation: RobotObjectRelation | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Build a goal-invariant object transition from one physical action."""

        facts.validate()
        action_dim = int(self.physical_action_condition.in_features // 2)
        if self.action_condition_mode == "interval_mean_v1":
            if not isinstance(action, PhysicalActionCondition):
                raise TypeError("interval-mean W requires PhysicalActionCondition")
            action.validate(action_dim=action_dim)
        else:
            if not isinstance(action, (PhysicalActionSequenceCondition, RecordedActionSequenceCondition)):
                raise TypeError(
                    "sequence-prefix W requires PhysicalActionSequenceCondition"
                )
            action.validate(action_dim=action_dim)
            if isinstance(action, RecordedActionSequenceCondition) and action.time_grid_mode != self.time_grid.mode:
                raise ValueError("W supervision action uses a different time grid")
        validity = self._safe_object_validity(facts).to(device=facts.content.device)
        safe_content = self._supported_object_values(facts.content, validity)
        objects = self.object_content(safe_content)
        if self.robot_relation_encoder is not None:
            relation = self._robot_relation(facts) if robot_relation is None else robot_relation
            assert relation is not None and isinstance(facts, ObjectWorldBelief)
            relation.validate_source(facts)
            transport_prior = relation.pooled.to(dtype=objects.dtype)
        else:
            safe_transport_prior = self._supported_object_values(
                (facts.transport_prior if facts.latest_flow_steps is None else facts.transport_rate).to(device=facts.content.device),
                validity,
            )
            transport_prior = self.object_transport_prior(safe_transport_prior.to(dtype=facts.content.dtype))
        gradient_metrics: dict[str, Tensor] = {}
        if isinstance(action, (PhysicalActionSequenceCondition, RecordedActionSequenceCondition)):
            if (
                self.sequence_time_condition is None
                or self.sequence_action_recurrence is None
            ):
                raise RuntimeError("sequence-prefix W encoder is not constructed")
            action_source = action.physical_fingerprint.to(dtype=objects.dtype)
            if collect_diagnostics:
                action_source = action_source.reshape_as(action_source)
                register_gradient_rms_metric(
                    action_source,
                    gradient_metrics,
                    "gradient_tensor_w_physical_action_sequence_rms",
                )
            row_carrier, _ = smooth_rms_contract(
                self.physical_action_condition(action_source),
                0.35,
            )
            row_time = (
                action.row_end.to(device=objects.device, dtype=objects.dtype)
                / float(action.control_time_scale if isinstance(action, RecordedActionSequenceCondition) else action.horizon)
            )
            time_carrier = self.sequence_time_condition(row_time)
            recurrent_input, _ = smooth_rms_contract(
                row_carrier + time_carrier,
                0.35,
            )
            recurrent_state = torch.zeros_like(recurrent_input[:, 0])
            prefix_states: list[Tensor] = []
            for row_index in range(action.horizon):
                next_state = self.sequence_action_recurrence(
                    recurrent_input[:, row_index],
                    recurrent_state,
                )
                recurrent_state = (
                    torch.where(action.observed[:, row_index, None], next_state, recurrent_state)
                    if isinstance(action, RecordedActionSequenceCondition) else next_state
                )
                prefix_states.append(recurrent_state)
            action_carrier = torch.stack(
                [
                    prefix_states[prefix_end - 1]
                    for prefix_end in (
                        self.time_grid.endpoints
                        if isinstance(action, RecordedActionSequenceCondition)
                        else tuple(min(end, action.horizon) for end in self.time_grid.endpoints)
                    )
                ],
                dim=1,
            )
            action_carrier, _ = smooth_rms_contract(action_carrier, 0.35)
        else:
            action_source = action.fingerprint.to(dtype=objects.dtype)
            if collect_diagnostics:
                action_source = action_source.reshape_as(action_source)
                register_gradient_rms_metric(
                    action_source,
                    gradient_metrics,
                    "gradient_tensor_w_physical_action_condition_rms",
                )
            action_carrier, _ = smooth_rms_contract(
                self.physical_action_condition(action_source),
                0.35,
            )
        if collect_diagnostics:
            # Keep the first trainable W carrier visible in the recovery
            # diagnostics.  In the single-pass training path this tensor is
            # attached to the coarse action condition again.
            register_gradient_rms_metric(
                action_carrier,
                gradient_metrics,
                "gradient_tensor_w_physical_action_carrier_rms",
            )
        interval_identity = self.interval_identity
        if self.time_grid.aligned:
            assert self.interval_clock_code is not None
            interval_identity = interval_identity + self.interval_clock_code.to(interval_identity)
        interval = action_carrier + interval_identity.to(
            device=objects.device,
            dtype=objects.dtype,
        )[:, :, 0]
        base = objects[:, None] + transport_prior[:, None] + interval[:, :, None]

        # W owns physical evolution for every current object.  Goal relevance
        # belongs to P2's evaluator, so typed W owners come directly from G's
        # goal-free facts.  The interval coordinate is a zero-mean,
        # zero-preserving modulation of the same physical owner; it cannot
        # create a typed value from a missing fact.
        typed_sources = tuple(
            self._supported_object_values(source, validity)
            * validity.to(dtype=source.dtype)
            for source in (facts.semantic, facts.appearance, facts.geometry)
        )
        typed_source_views: list[Tensor] = []
        if collect_diagnostics:
            for source, name in zip(typed_sources, self.TYPE_NAMES, strict=True):
                view = source.reshape_as(source)
                register_gradient_rms_metric(
                    view,
                    gradient_metrics,
                    f"gradient_tensor_w_{name}_fact_ingress_rms",
                )
                typed_source_views.append(view)
        else:
            typed_source_views.extend(typed_sources)
        common_components: list[Tensor] = []
        interval_components: list[Tensor] = []
        for source, projection in zip(
            typed_source_views,
            (self.object_semantic, self.object_appearance, self.object_geometry),
            strict=True,
        ):
            common, _ = smooth_rms_contract(
                projection(source),
                0.35,
            )
            interval_raw = common[:, None] * torch.tanh(
                action_carrier[:, :, None]
            )
            innovation, _ = smooth_rms_contract(
                interval_raw,
                0.35,
            )
            common_components.append(common)
            interval_components.append(innovation)
        typed_common = torch.stack(common_components, dim=2)
        typed_interval = torch.stack(interval_components, dim=3)
        if not collect_diagnostics:
            return base, typed_common, typed_interval, {}
        metrics: dict[str, Tensor] = {
            **gradient_metrics,
            "object_w_goal_direct_ingress": interval.new_zeros((), dtype=torch.float32),
            "object_w_coarse_hidden_direct_ingress": interval.new_zeros(
                (), dtype=torch.float32
            ),
            "object_w_physical_action_carrier_rms": action_carrier.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_typed_common_input_rms": typed_common.detach().float().square().mean().sqrt(),
            "object_w_typed_interval_input_rms": typed_interval.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        if isinstance(action, (PhysicalActionSequenceCondition, RecordedActionSequenceCondition)):
            metrics.update(
                {
                    "object_w_physical_action_sequence_source_rms": action.source_action.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_physical_action_sequence_value_rms": action.canonical_value.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_physical_action_sequence_delta_rms": action.canonical_delta.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_physical_action_sequence_variation": action.canonical_value.detach()
                    .float()
                    .std(dim=1, unbiased=False)
                    .mean(),
                }
            )
        else:
            metrics.update(
                {
                    "object_w_physical_action_condition_rms": action.interval_action.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_physical_action_delta_rms": action.interval_delta.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_physical_action_interval_variation": action.interval_action.detach()
                    .float()
                    .std(dim=1, unbiased=False)
                    .mean(),
                }
            )
        for type_index, name in enumerate(self.TYPE_NAMES):
            common = typed_common[..., type_index, :].detach().float()
            innovation = typed_interval[..., type_index, :].detach().float()
            metrics[f"object_w_{name}_common_contribution_rms"] = common.square().mean().sqrt()
            metrics[f"object_w_{name}_interval_contribution_rms"] = (
                innovation.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_fact_input_rms"] = typed_source_views[
                type_index
            ].detach().float().square().mean().sqrt()
        return base, typed_common, typed_interval, metrics

    def _camera_geometry_carrier(
        self,
        facts: ObjectFactSet | ObjectWorldBelief,
        typed_geometry: Tensor,
        *, robot_relation: RobotObjectRelation | None = None,
    ) -> Tensor:
        """Condition geometry independently on each observed camera motion."""

        if typed_geometry.ndim not in (3, 4) or int(typed_geometry.shape[-1]) != self.hidden:
            raise ValueError("typed camera geometry must be [B,K,H] or [B,I,K,H]")
        validity = self._safe_object_validity(facts).to(device=typed_geometry.device)
        safe_geometry = self._supported_object_values(
            typed_geometry,
            validity,
            object_dim=2 if typed_geometry.ndim == 4 else 1,
        )
        camera_validity = self._safe_camera_validity(
            facts,
            dtype=torch.float32,
        ).to(device=typed_geometry.device)
        # An invalid object can still carry a producer-side camera-valid bit in
        # diagnostic fixtures.  Quarantine its camera transport before the
        # zero-preserving modulation; otherwise ``0 * tanh(NaN)`` would make an
        # otherwise empty geometry carrier non-finite.
        object_camera_support = (
            (camera_validity > 0.0)
            & (validity[:, :, None, :] > 0.0)
        )
        camera_motion = facts.camera_transport_prior if facts.latest_flow_steps is None else facts.camera_transport_rate
        safe_camera_transport = torch.where(
            object_camera_support,
            camera_motion.to(device=typed_geometry.device),
            torch.zeros_like(
                camera_motion.to(device=typed_geometry.device)
            ),
        )
        camera_context = self.object_transport_prior(
            safe_camera_transport.to(dtype=typed_geometry.dtype)
        )
        camera_context = camera_context + self._camera_coordinate_role_context(
            facts,
            object_camera_support=object_camera_support,
            dtype=typed_geometry.dtype,
            device=typed_geometry.device,
        )
        if self.robot_relation_encoder is not None:
            relation = self._robot_relation(facts) if robot_relation is None else robot_relation
            assert relation is not None and isinstance(facts, ObjectWorldBelief)
            relation.validate_source(facts)
            camera_context = camera_context + relation.per_view.to(dtype=camera_context.dtype)
        cameras = int(camera_context.shape[2])
        if typed_geometry.ndim == 3:
            if tuple(typed_geometry.shape[:2]) != tuple(camera_context.shape[:2]):
                raise ValueError("typed common geometry lost object identity")
            carrier = safe_geometry[:, :, None].expand(-1, -1, cameras, -1)
            condition = camera_context
            availability = camera_validity
        else:
            if tuple(typed_geometry.shape[:1] + typed_geometry.shape[2:3]) != tuple(
                camera_context.shape[:2]
            ):
                raise ValueError("typed interval geometry lost object identity")
            carrier = safe_geometry[:, :, :, None].expand(-1, -1, -1, cameras, -1)
            condition = camera_context[:, None].expand(-1, int(typed_geometry.shape[1]), -1, -1, -1)
            availability = camera_validity[:, None]
        conditioned, _ = self._zero_preserving_condition(carrier, condition)
        return conditioned * availability.to(dtype=conditioned.dtype)

    def _camera_coordinate_role_context(
        self,
        facts: ObjectFactSet | ObjectWorldBelief,
        *,
        object_camera_support: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Return a declared-role, current-coordinate W condition.

        Coordinates remain image-plane facts; the stable role basis only
        distinguishes their declared camera charts.  The shared projection
        creates no transport value by itself because its output can only
        modulate an existing producer-owned geometry carrier.
        """

        batch, objects, cameras = object_camera_support.shape[:3]
        if self.camera_condition_mode == "motion_prior_only":
            return torch.zeros(
                batch,
                objects,
                cameras,
                self.hidden,
                device=device,
                dtype=dtype,
            )
        if cameras != len(self.camera_names):
            raise ValueError(
                "W camera axis does not match the declared camera role order"
            )
        projection = self.camera_coordinate_role_condition
        if projection is None:
            raise RuntimeError("coordinate_role_v1 lost its condition projection")
        coordinates = facts.camera_coordinates.to(device=device)
        safe_coordinates = torch.where(
            object_camera_support,
            coordinates,
            torch.zeros_like(coordinates),
        )
        role = self._camera_role_basis.to(device=device, dtype=dtype)
        role = role[None, None].expand(batch, objects, -1, -1)
        condition_input = torch.cat(
            (safe_coordinates.to(dtype=dtype), role),
            dim=-1,
        )
        context = projection(condition_input)
        return torch.where(
            object_camera_support,
            context,
            torch.zeros_like(context),
        )

    def _field_with_diagnostics(
        self,
        *,
        facts: ObjectFactSet | ObjectWorldBelief,
        typed_common: Tensor,
        typed_interval_innovation: Tensor,
        diagnostic_prefix: str | None = None,
        robot_relation: RobotObjectRelation | None = None,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        if typed_common.ndim != 4 or tuple(typed_common.shape[1:3]) != (
            facts.objects,
            3,
        ):
            raise ValueError("W typed common must be [B,K,3,H]")
        if typed_interval_innovation.ndim != 5 or tuple(typed_interval_innovation.shape[2:4]) != (
            facts.objects,
            3,
        ):
            raise ValueError("W typed innovation must be [B,I,K,3,H]")
        if (
            int(typed_common.shape[-1]) != self.hidden
            or int(typed_interval_innovation.shape[-1]) != self.hidden
        ):
            raise ValueError("W typed owner hidden width is invalid")

        validity = self._safe_object_validity(facts).to(device=typed_common.device)
        safe_typed_common = self._supported_object_values(typed_common, validity)
        safe_typed_interval = self._supported_object_values(
            typed_interval_innovation,
            validity,
            object_dim=2,
        )
        semantic_common_state, _ = self._appearance_condition_semantic(
            safe_typed_common[..., 0, :],
            safe_typed_common[..., 1, :],
        )
        semantic_interval_state, _ = self._appearance_condition_semantic(
            safe_typed_interval[..., 0, :],
            safe_typed_interval[..., 1, :],
        )
        semantic_common = self.delta_head(semantic_common_state)
        semantic_innovation = self.delta_head(semantic_interval_state)
        object_availability = validity[:, None].to(dtype=semantic_innovation.dtype)
        semantic_delta = (semantic_common[:, None] + semantic_innovation) * object_availability

        geometry_common = self._camera_geometry_carrier(
            facts,
            safe_typed_common[..., 2, :],
            robot_relation=robot_relation,
        )
        geometry_innovation = self._camera_geometry_carrier(
            facts,
            safe_typed_interval[..., 2, :],
            robot_relation=robot_relation,
        )
        transport_common_pre_tanh = self.transport_head(geometry_common).float()
        transport_innovation_pre_tanh = self.transport_head(geometry_innovation).float()
        transport_common = 0.50 * torch.tanh(transport_common_pre_tanh)
        transport_innovation = 0.50 * torch.tanh(transport_innovation_pre_tanh)
        base_camera_availability = self._safe_camera_validity(
            facts,
            dtype=torch.float32,
        ).to(device=typed_interval_innovation.device)
        camera_availability = base_camera_availability[:, None] * validity[:, None, :, None, :].to(
            device=typed_interval_innovation.device,
            dtype=torch.float32,
        )
        transport = (transport_common[:, None] + transport_innovation) * camera_availability
        transport = transport.to(dtype=typed_interval_innovation.dtype)

        full_geometry = self._camera_geometry_carrier(
            facts,
            safe_typed_common[:, None, ..., 2, :]
            + safe_typed_interval[..., 2, :],
            robot_relation=robot_relation,
        )
        covariance_raw = self.covariance_head(full_geometry).float()
        covariance_xx = torch.nn.functional.softplus(covariance_raw[..., 0])
        covariance_yy = torch.nn.functional.softplus(covariance_raw[..., 1])
        correlation = torch.tanh(covariance_raw[..., 2])
        covariance_xy = correlation * torch.sqrt(covariance_xx * covariance_yy)
        covariance = (
            torch.stack(
                (covariance_xx, covariance_xy, covariance_yy),
                dim=-1,
            )
            * camera_availability
        )

        current_reference = self._supported_object_values(
            facts.content,
            validity,
        ).detach().to(dtype=semantic_delta.dtype)
        base_camera_availability = self._safe_camera_validity(
            facts,
            dtype=torch.float32,
        ).to(device=semantic_delta.device)
        object_camera_availability = base_camera_availability * validity[:, :, None, :].to(
            device=semantic_delta.device,
            dtype=torch.float32,
        )
        camera_coordinates = facts.camera_coordinates.to(device=semantic_delta.device)
        safe_camera_coordinates = torch.where(
            object_camera_availability > 0.0,
            camera_coordinates,
            torch.zeros_like(camera_coordinates),
        )
        field = FutureObjectDynamics(
            camera_names=self.camera_names,
            time_grid_mode=self.time_grid.mode,
            current_reference=current_reference,
            successor_content=current_reference[:, None] + semantic_delta,
            semantic_delta=semantic_delta,
            transport_mean=transport,
            transport_covariance=covariance,
            chart_availability=validity,
            log_chart_availability=torch.nan_to_num(
                facts.log_validity.float(), nan=0.0, posinf=0.0, neginf=0.0
            ),
            camera_coordinates=safe_camera_coordinates.to(dtype=semantic_delta.dtype),
            camera_chart_availability=self._safe_camera_validity(facts),
            log_camera_chart_availability=torch.nan_to_num(
                facts.log_camera_validity.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )
        if diagnostic_prefix is None:
            return field, {}
        coordinate_role_context = self._camera_coordinate_role_context(
            facts,
            object_camera_support=(object_camera_availability > 0.0),
            dtype=semantic_delta.dtype,
            device=semantic_delta.device,
        )
        camera_condition_weight_rms = semantic_delta.new_zeros(
            (), dtype=torch.float32
        )
        if self.camera_coordinate_role_condition is not None:
            camera_condition_weight_rms = (
                self.camera_coordinate_role_condition.weight.detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
        saturation = torch.cat(
            (
                transport_common_pre_tanh.detach().reshape(-1),
                transport_innovation_pre_tanh.detach().reshape(-1),
            )
        )
        return field, {
            "object_w_transport_head_weight_rms": self.transport_head.weight.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_camera_coordinate_role_mode_active": semantic_delta.new_tensor(
                float(self.camera_condition_mode == "coordinate_role_v1"),
                dtype=torch.float32,
            ),
            "object_w_camera_coordinate_role_weight_rms": (
                camera_condition_weight_rms
            ),
            f"{diagnostic_prefix}_camera_coordinate_role_condition_rms": (
                coordinate_role_context.detach().float().square().mean().sqrt()
            ),
            f"{diagnostic_prefix}_transport_common_pre_tanh_rms": (
                transport_common_pre_tanh.detach().square().mean().sqrt()
            ),
            f"{diagnostic_prefix}_transport_innovation_pre_tanh_rms": (
                transport_innovation_pre_tanh.detach().square().mean().sqrt()
            ),
            f"{diagnostic_prefix}_transport_head_saturation_fraction": (
                torch.tanh(saturation).abs() >= 0.95
            )
            .float()
            .mean(),
        }

    def _field(
        self,
        *,
        facts: ObjectFactSet | ObjectWorldBelief,
        typed_common: Tensor,
        typed_interval_innovation: Tensor,
    ) -> FutureObjectDynamics:
        """Preserve the original field-only boundary for structural callers."""

        field, _ = self._field_with_diagnostics(
            facts=facts,
            typed_common=typed_common,
            typed_interval_innovation=typed_interval_innovation,
            diagnostic_prefix=None,
        )
        return field

    @staticmethod
    def _typed_state_metrics(value: Tensor, *, prefix: str) -> dict[str, Tensor]:
        if value.ndim != 5 or int(value.shape[3]) != 3:
            raise ValueError("typed W diagnostics require [B,I,K,3,H]")
        metrics: dict[str, Tensor] = {}
        for type_index, name in enumerate(ObjectFutureDynamicsCompiler.TYPE_NAMES):
            typed = value[..., type_index, :].detach().float()
            metrics[f"{prefix}_{name}_state_rms"] = typed.square().mean().sqrt()
            metrics[f"{prefix}_{name}_state_interval_variation"] = typed.std(
                dim=1, unbiased=False
            ).mean()
            metrics[f"{prefix}_{name}_state_object_variation"] = typed.std(
                dim=2, unbiased=False
            ).mean()
        return metrics

    def predict_executed_endpoint(self, *, facts: ObjectWorldBelief,
                                  action: ExecutedActionSequenceCondition) -> ExecutedWorldPrediction:
        """Current-weight retrodiction; only existing first four-step endpoint."""
        if not isinstance(action, ExecutedActionSequenceCondition) or not self.time_grid.aligned:
            raise ValueError("executed W endpoint requires its own aligned control record")
        action.validate(action_dim=self.action_dim)
        complete=action.observed[:,:4].all(1)
        empty=~action.observed[:,:4].any(1)
        if bool(action.observed[:,4:].any()) or not bool((complete|empty).all()):
            raise ValueError("executed endpoint requires exactly four controls or an absent transition")
        _,state,_=self.forward_w1(facts=facts,action=action,collect_diagnostics=False)
        field,_=self._field_with_diagnostics(facts=facts,typed_common=state.common_typed,
            typed_interval_innovation=state.near_interval_innovation,
            diagnostic_prefix=None,robot_relation=state.robot_relation)
        field.validate(expected_intervals=2)
        return ExecutedWorldPrediction(field.semantic_delta[:,0],field.transport_mean[:,0],
                                       field.transport_covariance[:,0],action,complete)

    def forward_w1(
        self,
        *,
        facts: ObjectFactSet | ObjectWorldBelief,
        action: WorldActionCondition | RecordedActionSequenceCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics | None, ObjectW1WorkingState, dict[str, Tensor]]:
        relation = self._robot_relation(facts)
        base, raw_common, raw_interval, base_metrics = self._base(
            facts, action, collect_diagnostics=collect_diagnostics, robot_relation=relation,
        )
        object_support = self._safe_object_validity(facts).detach()[..., 0] > 0.0
        near = self.w1(
            base[:, :2],
            causal_interval=True,
            object_support=object_support,
        )
        common_batch, common_norm_metrics = self._run_typed_block(
            self.w1,
            raw_common[:, None],
            causal_interval=False,
            collect_diagnostics=collect_diagnostics,
            object_support=object_support,
        )
        common_before = common_batch[:, 0]
        if isinstance(action, (PhysicalActionSequenceCondition, RecordedActionSequenceCondition)):
            # A shared action-dependent common would let W1 interval 1's
            # <=16-row prefix leak back into interval 0.  Keep common factual
            # in sequence mode; the per-interval typed path remains action
            # conditioned and causal below.
            common = common_before
            common_modulation = torch.zeros_like(common_before)
        else:
            common, common_modulation = self._zero_preserving_condition(
                common_before,
                near.mean(dim=1)[:, :, None, :].expand_as(common_before),
            )
        near_context = (common[:, None] + near[:, :, :, None, :]).expand_as(raw_interval[:, :2])
        near_input, near_condition_modulation = self._zero_preserving_condition(
            raw_interval[:, :2],
            near_context,
        )
        near_typed, near_norm_metrics = self._run_typed_block(
            self.w1,
            near_input,
            causal_interval=True,
            collect_diagnostics=collect_diagnostics,
            object_support=object_support,
        )
        field: FutureObjectDynamics | None = None
        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            field, field_metrics = self._field_with_diagnostics(
                facts=facts,
                typed_common=common,
                typed_interval_innovation=near_typed,
                diagnostic_prefix="object_w1",
                robot_relation=relation,
            )
            field.validate(expected_intervals=2)
            metrics.update(self._metrics(field, prefix="object_w1"))
            metrics.update(field_metrics)
            metrics.update(self._typed_state_metrics(near_typed, prefix="object_w1"))
            metrics.update(
                self._merge_normalization_metrics(
                    common_norm_metrics,
                    near_norm_metrics,
                    prefix="object_w1_typed_norm",
                )
            )
            metrics.update(
                {
                    "object_w1_common_processing_delta_rms": (
                        common_before.detach().float() - raw_common.detach().float()
                    )
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_common_generic_condition_rms": common_modulation.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_near_condition_rms": near_condition_modulation.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                }
            )
        metrics.update(base_metrics)
        return (
            field,
            ObjectW1WorkingState(
                time_grid_mode=self.time_grid.mode,
                robot_relation=relation,
                near=near,
                far_base=base[:, 2:],
                common_typed=common,
                near_interval_innovation=near_typed,
                far_interval_innovation=raw_interval[:, 2:],
                control_domain=(
                    CandidateControlDomain(action.horizon, interval_bounds=self.time_grid.bounds)
                    if self.control_mode == KNOWN_PREFIX_WORLD_CONTROL
                    and isinstance(action, PhysicalActionSequenceCondition) else None
                ),
            ),
            metrics,
        )

    def forward_w2(
        self,
        *,
        facts: ObjectFactSet | ObjectWorldBelief,
        w1_state: ObjectW1WorkingState,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        if w1_state.time_grid_mode != self.time_grid.mode:
            raise ValueError("W2 cannot consume W1 from a different physical time grid")
        if tuple(w1_state.near.shape[1:3]) != (2, facts.objects) or tuple(
            w1_state.far_base.shape[1:3]
        ) != (2, facts.objects):
            raise ValueError("W2 requires both completed W1 intervals")
        if tuple(w1_state.near_interval_innovation.shape[1:4]) != (2, facts.objects, 3) or tuple(
            w1_state.far_interval_innovation.shape[1:4]
        ) != (2, facts.objects, 3):
            raise ValueError("W2 typed innovation lost interval/object/type identity")
        batch, _, objects, hidden = w1_state.near.shape
        far_query = w1_state.far_base.transpose(1, 2).reshape(batch * objects, 2, hidden)
        near_memory = w1_state.near.transpose(1, 2).reshape(batch * objects, 2, hidden)
        normalized_near = self.w1_memory_norm(near_memory)
        near_update, _ = self.w1_to_w2(
            self.w2_query_norm(far_query),
            normalized_near,
            normalized_near,
            need_weights=False,
        )
        near_update, _ = smooth_rms_contract(near_update, 0.35)
        far = (far_query + near_update).reshape(batch, objects, 2, hidden).transpose(1, 2)
        object_support = self._safe_object_validity(facts).detach()[..., 0] > 0.0
        far = self.w2(
            far,
            causal_interval=True,
            object_support=object_support,
        )

        typed_far_query = w1_state.far_interval_innovation.permute(0, 2, 3, 1, 4).reshape(
            batch * objects * 3, 2, hidden
        )
        typed_memory = (
            torch.cat(
                (
                    w1_state.common_typed[:, None],
                    w1_state.near_interval_innovation,
                ),
                dim=1,
            )
            .permute(0, 2, 3, 1, 4)
            .reshape(batch * objects * 3, 3, hidden)
        )
        normalized_typed_query, typed_query_denominator = (
            self.typed_w2_query_norm.forward_with_denominator(typed_far_query)
        )
        normalized_memory, typed_memory_denominator = (
            self.typed_w1_memory_norm.forward_with_denominator(typed_memory)
        )
        typed_read, _ = self.w1_to_w2(
            normalized_typed_query,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        typed_read, _ = smooth_rms_contract(typed_read, 0.35)
        typed_context = typed_read.reshape(batch, objects, 3, 2, hidden).permute(0, 3, 1, 2, 4)
        far_context = typed_context + far[:, :, :, None, :]
        far_input, far_condition_modulation = self._zero_preserving_condition(
            w1_state.far_interval_innovation,
            far_context,
        )
        far_typed, far_norm_metrics = self._run_typed_block(
            self.w2,
            far_input,
            causal_interval=True,
            collect_diagnostics=collect_diagnostics,
            object_support=object_support,
        )
        completed_innovation = torch.cat(
            (w1_state.near_interval_innovation, far_typed),
            dim=1,
        )
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            register_gradient_axis_rms_metrics(
                w1_state.common_typed,
                gradient_metrics,
                (
                    "gradient_tensor_w2_semantic_common_rms",
                    "gradient_tensor_w2_appearance_common_rms",
                    "gradient_tensor_w2_geometry_common_rms",
                ),
                dim=-2,
            )
            register_gradient_axis_rms_metrics(
                completed_innovation,
                gradient_metrics,
                (
                    "gradient_tensor_w2_semantic_interval_rms",
                    "gradient_tensor_w2_appearance_interval_rms",
                    "gradient_tensor_w2_geometry_interval_rms",
                ),
                dim=-2,
            )
        field, field_metrics = self._field_with_diagnostics(
            facts=facts,
            typed_common=w1_state.common_typed,
            typed_interval_innovation=completed_innovation,
            diagnostic_prefix="object_w2" if collect_diagnostics else None,
            robot_relation=w1_state.robot_relation,
        )
        if w1_state.control_domain is not None:
            field = replace(field, control_domain=w1_state.control_domain)
        field.validate()
        if not collect_diagnostics:
            return field, {}
        metrics = self._metrics(field, prefix="object_w2")
        metrics.update(field_metrics)
        metrics.update(gradient_metrics)
        metrics.update(self._typed_state_metrics(completed_innovation, prefix="object_w2"))
        metrics.update(
            self._merge_normalization_metrics(
                self._normalization_metrics(
                    typed_far_query,
                    normalized_typed_query,
                    typed_query_denominator,
                ),
                self._normalization_metrics(
                    typed_memory,
                    normalized_memory,
                    typed_memory_denominator,
                ),
                far_norm_metrics,
                prefix="object_w2_typed_norm",
            )
        )
        metrics["object_w_far_condition_rms"] = (
            far_condition_modulation.detach().float().square().mean().sqrt()
        )
        metrics["object_w_typed_common_state_rms"] = (
            w1_state.common_typed.detach().float().square().mean().sqrt()
        )
        return field, metrics

    @staticmethod
    def _metrics(field: FutureObjectDynamics, *, prefix: str) -> dict[str, Tensor]:
        delta = field.semantic_delta.detach().float()
        adjacent = (
            F.cosine_similarity(
                delta[:, 1:].flatten(2),
                delta[:, :-1].flatten(2),
                dim=-1,
                eps=1.0e-4,
            ).mean()
            if field.intervals > 1
            else delta.new_zeros(())
        )
        object_similarity = F.normalize(delta, dim=-1, eps=1.0e-4)
        object_similarity = torch.einsum("bikd,bijd->bikj", object_similarity, object_similarity)
        objects = int(delta.shape[2])
        mask = ~torch.eye(objects, device=delta.device, dtype=torch.bool)
        pair = (
            object_similarity.masked_select(mask[None, None]).mean()
            if objects > 1
            else delta.new_zeros(())
        )
        covariance = field.transport_covariance.detach().float()
        determinant = covariance[..., 0] * covariance[..., 2] - covariance[..., 1].square()
        return {
            f"{prefix}_semantic_delta_rms": delta.square().mean().sqrt(),
            f"{prefix}_interval_adjacent_cosine": adjacent,
            f"{prefix}_object_pair_cosine": pair,
            f"{prefix}_transport_rms": field.transport_mean.detach().float().square().mean().sqrt(),
            f"{prefix}_covariance_determinant_min": determinant.amin(),
            f"{prefix}_camera_chart_availability": field.camera_chart_availability.detach()
            .float()
            .mean(),
        }


__all__ = ["ObjectFutureDynamicsCompiler", "ObjectW1WorkingState"]
