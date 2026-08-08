"""Current staged world/action trunk and layer contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .codec import ParsevalGripperTemporalFrame
from .config import V39PolicyConfig
from .contracts import scaled_contract_view as _scaled_contract_view
from .decoder import HierarchicalMMDiTActionDecoder
from .differential_intent_effect import (
    ConsequenceAwarePlanState,
    ConsequencePlanOrganizer,
    DifferentialFutureEffectReader,
    DifferentialPolicyPlanBank,
    DifferentialPolicyPlanCompiler,
    DifferentialStatelessIntentController,
    DifferentialWindowEffectBank,
    IntentStateBank,
)
from .flow_dino_evidence import (
    FlowDINOEvidenceEncoder,
    FlowDINOEvidencePack,
    FutureEffectField,
    LateRawDetailEvidence,
    ProgressiveGroundingAddressState,
    WindowEffectBank,
)
from .goal_conditioning import (
    GoalPhaseState,
    GoalTokenResampler,
    StatelessGoalPhaseMachine,
    StatelessHorizonConditionAdapter,
    StatelessIntentController,
    StatelessIntentState,
    StatelessPhaseAdapter,
)
from .grounded_intent_effect import (
    GROUNDING_MANIFEST,
    ConsequenceConditionedPolicyPlanCompiler,
    StatelessIntentOrganizer,
    ZeroPreservingConsequenceOrganizer,
)
from .grounded_intent_effect import (
    BoundedFutureEffectReader as GroundedFutureEffectReader,
)
from .grounded_intent_effect import (
    ConsequencePlanState as GroundedConsequencePlanState,
)
from .grounded_intent_effect import (
    FutureEffectField as GroundedFutureEffectField,
)
from .grounded_intent_effect import (
    PolicyPlanDeltaBank as GroundedPolicyPlanDeltaBank,
)
from .grounded_intent_effect import (
    StatelessIntentState as GroundedIntentState,
)
from .legacy import (
    AdaptiveRecurrentCVAEActionDecoder,
    HierarchicalLatentMainActionDecoder,
    LatentCVAEActionDecoder,
    LayeredV37StyleResidualActionFlowDenoiser,
    V37StyleResidualActionFlowDenoiser,
)
from .object_intent_dynamics_323 import (
    ARCHITECTURE_MANIFEST as OBJECT_INTENT_DYNAMICS_MANIFEST,
)
from .object_intent_dynamics_323 import (
    CAPABILITY_SCHEMA as OBJECT_INTENT_DYNAMICS_SCHEMA,
)
from .object_intent_dynamics_323 import (
    CoarseActionIntent,
    CoarseActionIntentState,
    DenseObjectGrounder,
    FutureObjectDynamics,
    FuturePlanRecognition,
    FuturePlanRecognizer,
    ObjectConsequenceState,
    ObjectFactSet,
    ObjectFactualDock,
    ObjectFutureDynamicsCompiler,
    ObjectFutureEffectReader,
    ObjectFutureTeacher,
    ObjectIntentState,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ObjectTopTrainingTargets,
    ObjectW1WorkingState,
    StatelessObjectIntentOrganizer,
    ZeroPreservingObjectConsequence,
)
from .primitives import TimeEmbedding
from .role_delta_attnres import (
    AffineVarianceFlooredCenteredNorm,
    PolicyRoleDeltaBank,
    RoleDeltaAttnRes,
    VarianceFlooredCenteredNorm,
    smooth_rms_contract,
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


@dataclass(frozen=True)
class LateRawDetailReadResult:
    """Backward-compatible P1 result with an optional typed object dock."""

    trajectory: Tensor
    metrics: dict[str, Tensor]
    object_dock: ObjectFactualDock | None = None

    def __iter__(self):
        # Historical callers unpacked ``(trajectory, metrics)``.  Preserve
        # that interface while the active object mainline consumes the typed
        # dock explicitly rather than smuggling it through a metrics dict.
        yield self.trajectory
        yield self.metrics


@dataclass(frozen=True)
class V115StaticEvidenceCache:
    """Observation/clean-intent evidence reused inside one deployment sample.

    The cache ends immediately before the first dynamic policy block.  It
    never supplies a cached noisy-action trajectory: ``pre_policy_canvas`` is
    read only outside the trajectory slice, while ``policy_ingress_delta`` is
    the action-independent G/W-to-P factual write added to the current ODE
    trajectory.  Training never constructs this object.
    """

    pre_policy_canvas: Tensor
    midcut_static_canvas: Tensor | None
    visual_memory: Tensor
    visual_value_memory: Tensor
    goal_tokens: Tensor
    world_detail_entry_rollout: Tensor
    late_raw_detail: LateRawDetailEvidence
    progressive_address_state: ProgressiveGroundingAddressState
    goal_phase_state: (
        GoalPhaseState
        | StatelessIntentState
        | IntentStateBank
        | GroundedIntentState
        | ObjectIntentState
    )
    phase_context: Tensor
    goal_context: Tensor | None
    history_context: Tensor | None
    interval_stage_prediction: Tensor | None
    policy_ingress_delta: Tensor
    protected_policy_detail: Tensor
    phase_metrics: dict[str, Tensor]
    raw_refinement_metrics: dict[str, Tensor]
    late_detail_metrics: dict[str, Tensor]
    role_delta_metrics: dict[str, Tensor]
    future_address_metrics: dict[str, Tensor]
    horizon_boundary_metrics: dict[str, Tensor]
    gate_rows: tuple[dict[str, Tensor], ...]
    gate_row_roles: tuple[str, ...]
    content_norm_rows: tuple[Tensor, ...]
    time_norm_rows: tuple[Tensor, ...]
    # Capability-named object top.  These are online current-input products;
    # no future teacher or training recognizer is ever cached for deployment.
    object_facts: ObjectFactSet | None = None
    object_factual_dock: ObjectFactualDock | None = None
    object_intent_state: ObjectIntentState | None = None
    object_coarse_action: CoarseActionIntentState | None = None
    object_future_dynamics: FutureObjectDynamics | None = None
    object_top_metrics: dict[str, Tensor] | None = None

    def validate(
        self,
        *,
        canvas: Tensor,
        slices: dict[str, slice],
    ) -> None:
        if tuple(self.pre_policy_canvas.shape) != tuple(canvas.shape):
            raise ValueError(
                "V115 static cache canvas does not match this deployment batch"
            )
        if (
            self.midcut_static_canvas is not None
            and tuple(self.midcut_static_canvas.shape)
            != tuple(canvas.shape)
        ):
            raise ValueError(
                "V115 static cache midcut canvas shape mismatch"
            )
        trajectory = canvas[:, slices["trajectory"]]
        if tuple(self.policy_ingress_delta.shape) != tuple(trajectory.shape):
            raise ValueError(
                "V115 static cache ingress delta does not match trajectory"
            )
        if self.object_factual_dock is not None:
            self.object_factual_dock.validate()
        batch = int(canvas.shape[0])
        if int(self.goal_tokens.shape[0]) != batch:
            raise ValueError("V115 static cache goal batch mismatch")
        if self.pre_policy_canvas.device != canvas.device:
            raise ValueError("V115 static cache device mismatch")
        if tuple(self.world_detail_entry_rollout.shape) != tuple(
            canvas[:, slices["rollout"]].shape
        ):
            raise ValueError(
                "V115 static cache protected G3 chart shape mismatch"
            )
        if isinstance(self.goal_phase_state, ObjectIntentState):
            if (
                self.object_facts is None
                or self.object_coarse_action is None
                or self.object_future_dynamics is None
            ):
                raise ValueError(
                    "object static cache has no completed object/S/W state"
                )
            self.object_facts.validate()
            self.object_future_dynamics.validate()
            effect_field = None
        elif isinstance(self.goal_phase_state, GroundedIntentState):
            grounded_effect = (
                self.progressive_address_state.world_grounded_effect_field
            )
            grounded_w1 = (
                self.progressive_address_state.world_grounded_effect_w1_field
            )
            if grounded_effect is None or grounded_w1 is None:
                raise ValueError(
                    "grounded static cache has no completed W1/W2 effect field"
                )
            grounded_w1.validate(expected_intervals=2)
            grounded_effect.validate()
        elif isinstance(self.goal_phase_state, IntentStateBank):
            differential_effect = (
                self.progressive_address_state.world_differential_effect_field
            )
            differential_w1 = (
                self.progressive_address_state.world_differential_effect_w1_field
            )
            if differential_effect is None or differential_w1 is None:
                raise ValueError(
                    "differential static cache has no completed W1/W2 effect bank"
                )
            differential_w1.validate(expected_slots=2)
            differential_effect.validate(expected_slots=3)
        else:
            effect_field = (
                self.progressive_address_state.world_future_effect_field
            )
            if effect_field is None:
                raise ValueError(
                    "V115 static cache has no completed FutureEffectField"
                )
            effect_field.validate()
        if (
            not isinstance(
                self.goal_phase_state,
                (IntentStateBank, GroundedIntentState, ObjectIntentState),
            )
            and effect_field.current_content is not None
        ):
            w1_field = (
                self.progressive_address_state.world_future_effect_w1_field
            )
            if w1_field is None:
                raise ValueError(
                    "V116 static cache has no supervised W1 FutureEffectField"
                )
            w1_field.validate()
            if w1_field.current_content is None:
                raise ValueError(
                    "V116 static cache W1 field uses the legacy carrier"
                )
            if self.goal_phase_state.terminal_probability is None:
                raise ValueError(
                    "V116 static cache has no separate terminal evidence"
                )
        if isinstance(self.goal_phase_state, GroundedIntentState):
            self.goal_phase_state.validate(
                batch=batch,
                hidden=int(canvas.shape[-1]),
                horizon=int(self.goal_phase_state.temporal_control.shape[1]),
            )
        elif isinstance(self.goal_phase_state, ObjectIntentState):
            self.goal_phase_state.validate(
                horizon=int(self.goal_phase_state.temporal_queries.shape[1]),
                hidden=int(canvas.shape[-1]),
            )
        else:
            self.goal_phase_state.validate(
                batch=batch,
                program_states=int(self.goal_phase_state.goal_program.shape[1]),
                intervals=int(self.goal_phase_state.interval_selector.shape[1]),
                hidden=int(canvas.shape[-1]),
            )
        if self.object_intent_state is not None:
            if (
                self.object_facts is None
                or self.object_coarse_action is None
                or self.object_future_dynamics is None
            ):
                raise ValueError(
                    "object static cache lost facts, coarse intent, W1 or W2"
                )
            self.object_facts.validate()
            self.object_intent_state.validate(
                horizon=int(self.object_intent_state.temporal_queries.shape[1]),
                hidden=int(canvas.shape[-1]),
            )
            self.object_future_dynamics.validate()


@dataclass(frozen=True)
class PolicyPlanDeltaBank:
    """P3 typed plan lanes with explicit provenance and no visual bank."""

    protected_base: Tensor
    precision: Tensor
    effect: Tensor
    temporal: Tensor
    terminal: Tensor | None = None
    execution_terminal: ExecutionTerminalEvidence | None = None

    @property
    def source_names(self) -> tuple[str, ...]:
        names = [
            "p3_precision",
            "p3_effect",
            "p3_temporal",
        ]
        if self.terminal is not None:
            names.append("p3_terminal")
        return tuple(names)

    def validate(
        self,
        *,
        batch: int,
        horizon: int,
        basis: int,
        hidden: int,
    ) -> None:
        expected = (batch, horizon, basis, hidden)
        for name in (
            "protected_base",
            "precision",
            "effect",
            "temporal",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"policy-plan {name} must be {expected}, "
                    f"got {tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"policy-plan {name} is non-finite")
        if self.terminal is not None:
            if tuple(self.terminal.shape) != expected:
                raise ValueError("policy-plan terminal must be [B,T,K,H]")
            if not bool(torch.isfinite(self.terminal).all()):
                raise ValueError("policy-plan terminal is non-finite")
        if self.execution_terminal is not None:
            self.execution_terminal.validate(batch=batch)

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        values = [self.precision, self.effect, self.temporal]
        if self.terminal is not None:
            values.append(self.terminal)
        return PolicyRoleDeltaBank(
            values=torch.stack(values, dim=1),
            source_names=self.source_names,
            source_depths=(int(source_depth),) * len(values),
            # Factual base is mandatory and bypasses the optional/null delta
            # router in the bottom decoder. It is still softly read across
            # action bases by the existing no-null protected-detail reader.
            protected_detail=self.protected_base,
        )


@dataclass(frozen=True)
class ExecutionTerminalEvidence:
    """Completion evidence owned only by the execution controller."""

    probability: Tensor
    uncertainty: Tensor

    def validate(self, *, batch: int) -> None:
        for name, value in (
            ("probability", self.probability),
            ("uncertainty", self.uncertainty),
        ):
            if tuple(value.shape) != (batch, 1):
                raise ValueError(f"execution terminal {name} must be [B,1]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"execution terminal {name} is non-finite")


class StructuredFutureEffectReader(nn.Module):
    """V117 P2-owned zero-preserving read over all window/spatial effects."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.hidden = int(config.hidden_size)
        self.horizon = int(config.action_horizon)
        self.basis = int(config.action_basis_tokens)
        self.slots = int(getattr(config, "flow_jepa_future_slots", 3))
        if self.slots != 3:
            raise ValueError("V117 structured effect reader requires three slots")
        self.basis_identity = nn.Parameter(
            torch.randn(1, 1, self.basis, self.hidden) * 0.02
        )
        self.query = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, self.hidden, bias=False),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, self.hidden, bias=False),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(self.hidden + 8, elementwise_affine=False),
            nn.Linear(self.hidden + 8, self.hidden, bias=False),
        )
        nn.init.normal_(self.value[-1].weight, mean=0.0, std=3e-3)

    def forward(
        self,
        query_tokens: Tensor,
        effect: FutureEffectField,
        *,
        window_selector: Tensor | None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        effect.validate()
        if effect.current_content is None or effect.successor_content is None:
            raise RuntimeError("P2 effect reader requires supervised content")
        expected = (
            int(query_tokens.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(query_tokens.shape) != expected:
            raise ValueError("P2 effect query must be [B,T,K,H]")
        if int(effect.semantic_delta.shape[1]) != self.slots:
            raise ValueError("P2 effect field does not contain near/mid/late")
        if not isinstance(effect, WindowEffectBank) or effect.slot_valid is None:
            raise TypeError("V117 P2 requires a typed WindowEffectBank")
        if not bool((effect.slot_valid.detach().float() > 0.5).all()):
            raise RuntimeError("V117 P2 cannot read an incomplete W1 effect bank")
        batch = int(query_tokens.shape[0])
        if window_selector is None:
            selector = query_tokens.new_full(
                (batch, self.slots), 1.0 / float(self.slots)
            )
        else:
            if tuple(window_selector.shape) != (batch, self.slots):
                raise ValueError("P2 window selector must be [B,3]")
            selector = window_selector.to(
                device=query_tokens.device, dtype=query_tokens.dtype
            )
        # A positive floor keeps every supervised slot available without a
        # route quota or hard gate. The selector changes logits, never values.
        selector = 0.05 + 0.95 * selector.float().clamp(0.0, 1.0)
        selector = selector / selector.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        current = effect.current_content
        successor = effect.successor_content
        geometry = torch.cat(
            (
                effect.transport_mean,
                effect.transport_covariance,
                effect.persistence,
                effect.visibility,
                effect.uncertainty,
            ),
            dim=-1,
        )
        key = self.key(0.5 * (current + successor)).reshape(
            batch, self.slots, -1, self.hidden
        )
        value = self.value(
            torch.cat((effect.semantic_delta, geometry), dim=-1)
        ).reshape(batch, self.slots, -1, self.hidden)
        basis = self.basis_identity.to(
            device=query_tokens.device, dtype=query_tokens.dtype
        ).expand(batch, self.horizon, -1, -1)
        query = self.query(query_tokens + basis)
        logits = torch.einsum(
            "btkh,bsnh->btksn", query.float(), key.float()
        ) / math.sqrt(float(self.hidden))
        slot_basis = torch.eye(
            self.slots, device=query.device, dtype=torch.float32
        )[None]
        temporal_prior = F.interpolate(
            slot_basis,
            size=self.horizon,
            mode="linear",
            align_corners=True,
        )[0].transpose(0, 1)
        temporal_prior = 0.05 + 0.95 * temporal_prior
        temporal_prior = temporal_prior / temporal_prior.sum(
            dim=-1, keepdim=True
        )
        slot_log_prior = (
            selector[:, None, None, :, None].clamp_min(1e-8).log()
            + temporal_prior[None, :, None, :, None].clamp_min(1e-8).log()
        )
        posterior = torch.softmax(
            (logits + slot_log_prior).flatten(-2), dim=-1
        ).reshape_as(logits).to(dtype=value.dtype)
        read = torch.einsum("btksn,bsnh->btkh", posterior, value)
        if not collect_diagnostics:
            return read, {}
        spatial_count = int(value.shape[2])
        entropy = -(
            posterior.float().clamp_min(1e-8)
            * posterior.float().clamp_min(1e-8).log()
        ).sum(dim=(-2, -1)) / math.log(
            float(max(self.slots * spatial_count, 2))
        )
        slot_mass = posterior.detach().float().sum(dim=-1).mean(
            dim=(0, 1, 2)
        )
        metrics = {
            "flow_jepa_p2_structured_effect_read_rms": (
                read.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_p2_structured_effect_entropy": entropy.detach().mean(),
            "flow_jepa_p2_structured_effect_slot_variation": (
                slot_mass.std(unbiased=False)
            ),
        }
        for index, name in enumerate(("near", "mid", "late")):
            metrics[f"flow_jepa_p2_effect_{name}_mass"] = slot_mass[index]
        return read, metrics


class PolicyPlanCompiler(nn.Module):
    """Compile P1/P2 facts and online consequences into typed bottom lanes.

    The compiler has no access to RGB/DINO banks or noisy trajectory values.
    Each lane receives a different legal operand set, so lane naming cannot
    collapse into four aliases over one hidden carrier.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        basis = int(config.action_basis_tokens)
        self.hidden = hidden
        self.basis = basis
        self.horizon = int(config.action_horizon)
        self.intervals = int(config.future_anchors)
        self.effect_slots = int(
            getattr(config, "flow_jepa_future_slots", self.intervals)
        )
        self.effect_read_in_p2 = bool(
            int(getattr(config, "flow_jepa_effect_read_in_p2", 0))
        )
        self.supervised_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_supervised_effect_mainline",
                    0,
                )
            )
        )
        self.basis_identity = nn.Parameter(
            torch.randn(1, 1, basis, hidden) * 0.02
        )
        if self.supervised_effect_mainline and not self.effect_read_in_p2:
            self.effect_geometry = None
            self.effect_innovation = None
            self.structured_effect_key = nn.Sequential(
                nn.LayerNorm(hidden, elementwise_affine=False),
                nn.Linear(hidden, hidden, bias=False),
            )
            self.structured_effect_value = nn.Sequential(
                nn.LayerNorm(hidden + 8, elementwise_affine=False),
                nn.Linear(hidden + 8, hidden, bias=False),
            )
            self.structured_effect_query = nn.Sequential(
                nn.LayerNorm(hidden, elementwise_affine=False),
                nn.Linear(hidden, hidden, bias=False),
            )
            nn.init.normal_(
                self.structured_effect_value[-1].weight,
                mean=0.0,
                std=3e-3,
            )
        elif not self.supervised_effect_mainline:
            self.effect_geometry = nn.Sequential(
                nn.LayerNorm(8, elementwise_affine=False),
                nn.Linear(8, hidden, bias=False),
            )
            self.effect_innovation = nn.Sequential(
                nn.LayerNorm(
                    int(config.flow_jepa_address_route_dim),
                    elementwise_affine=False,
                ),
                nn.Linear(
                    int(config.flow_jepa_address_route_dim),
                    hidden,
                    bias=False,
                ),
            )
            self.structured_effect_key = None
            self.structured_effect_value = None
            self.structured_effect_query = None
        else:
            self.effect_geometry = None
            self.effect_innovation = None
            self.structured_effect_key = None
            self.structured_effect_value = None
            self.structured_effect_query = None
        self.precision_lane = self._lane(3 * hidden, hidden)
        self.effect_lane = self._lane(
            hidden if self.supervised_effect_mainline else 4 * hidden,
            hidden,
        )
        self.temporal_lane = self._lane(4 * hidden, hidden)
        self.terminal_uncertainty = (
            None
            if self.supervised_effect_mainline
            else nn.Linear(1, hidden, bias=False)
        )
        self.terminal_lane = (
            None
            if self.supervised_effect_mainline
            else self._lane(5 * hidden, hidden)
        )
        initialized_lanes = [
            self.precision_lane,
            self.effect_lane,
            self.temporal_lane,
        ]
        if self.terminal_lane is not None:
            initialized_lanes.append(self.terminal_lane)
        for lane in initialized_lanes:
            nn.init.normal_(lane[-1].weight, mean=0.0, std=3e-3)

    @staticmethod
    def _lane(input_width: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_width, elementwise_affine=False),
            nn.Linear(input_width, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def _interval_to_action(self, value: Tensor) -> Tensor:
        if (
            value.ndim != 3
            or int(value.shape[1]) != self.intervals
            or int(value.shape[2]) != self.hidden
        ):
            raise ValueError("P3 interval context must be [B,A,H]")
        return F.interpolate(
            value.float().transpose(1, 2),
            size=self.horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=value.dtype)

    def _expand_global_condition(
        self,
        value: Tensor,
        *,
        batch: int,
        name: str,
    ) -> Tensor:
        """Broadcast one [B,H] condition over action time and basis axes."""

        if tuple(value.shape) != (batch, self.hidden):
            raise ValueError(f"P3 {name} must be [B,H]")
        return value[:, None, None, :].expand(
            batch,
            self.horizon,
            self.basis,
            self.hidden,
        )

    def _effect_summary(
        self,
        state: ProgressiveGroundingAddressState,
    ) -> Tensor:
        effect = state.world_future_effect_field
        if effect is None:
            raise RuntimeError("P3 has no supervised FutureEffectField")
        effect.validate()
        if effect.state_innovation is None:
            raise RuntimeError("legacy P3 effect summary has no state innovation")
        if self.effect_innovation is None or self.effect_geometry is None:
            raise RuntimeError("legacy P3 effect modules are missing")
        reliability = (
            effect.persistence.float()
            * effect.visibility.float()
            / (1.0 + effect.uncertainty.float())
        )
        reduce_dims = (2, 3, 4, 5)
        denominator = reliability.sum(
            dim=reduce_dims
        ).clamp_min(1e-6)

        def weighted(value: Tensor) -> Tensor:
            return (
                value.float() * reliability
            ).sum(dim=reduce_dims) / denominator

        semantic = weighted(effect.semantic_delta)
        innovation = self.effect_innovation(
            weighted(effect.state_innovation).to(
                dtype=effect.semantic_delta.dtype
            )
        )
        geometry = torch.cat(
            (
                weighted(effect.transport_mean),
                weighted(effect.transport_covariance),
                weighted(effect.persistence),
                weighted(effect.visibility),
                weighted(effect.uncertainty),
            ),
            dim=-1,
        )
        geometry = self.effect_geometry(
            geometry.to(dtype=effect.semantic_delta.dtype)
        )
        return (semantic + innovation + geometry) / math.sqrt(3.0)

    def _structured_effect_read(
        self,
        query_tokens: Tensor,
        state: ProgressiveGroundingAddressState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Cross-read the complete spatial effect field without value aliases."""

        effect = state.world_future_effect_field
        if effect is None:
            raise RuntimeError("V116 P2 has no FutureEffectField")
        effect.validate()
        if (
            effect.current_content is None
            or effect.successor_content is None
            or self.structured_effect_key is None
            or self.structured_effect_value is None
            or self.structured_effect_query is None
        ):
            raise RuntimeError("V116 P2 effect interface is incomplete")
        if tuple(query_tokens.shape[1:]) != (
            self.horizon,
            self.basis,
            self.hidden,
        ):
            raise ValueError("V116 P2 effect query must be [B,T,K,H]")
        batch = int(query_tokens.shape[0])
        query = self.structured_effect_query(query_tokens)
        interval_basis = torch.eye(
            self.intervals,
            device=query.device,
            dtype=torch.float32,
        )[None]
        interval_weight = F.interpolate(
            interval_basis,
            size=self.horizon,
            mode="linear",
            align_corners=True,
        )[0].transpose(0, 1)
        result = torch.zeros_like(query)
        entropy_rows: list[Tensor] = []
        read_rows: list[Tensor] = []
        for interval in range(self.intervals):
            current = effect.current_content[:, interval]
            successor = effect.successor_content[:, interval]
            geometry = torch.cat(
                (
                    effect.transport_mean[:, interval],
                    effect.transport_covariance[:, interval],
                    effect.persistence[:, interval],
                    effect.visibility[:, interval],
                    effect.uncertainty[:, interval],
                ),
                dim=-1,
            )
            # Keys may describe current/successor identity. Values contain
            # only the supervised effect variables; query/basis never enter
            # the value path.
            key = self.structured_effect_key(
                0.5 * (current + successor)
            ).reshape(batch, -1, self.hidden)
            value_input = torch.cat(
                (effect.semantic_delta[:, interval], geometry), dim=-1
            )
            value = self.structured_effect_value(value_input).reshape(
                batch, -1, self.hidden
            )
            logits = torch.einsum(
                "btkh,bnh->btkn", query.float(), key.float()
            ) / math.sqrt(float(self.hidden))
            posterior = torch.softmax(logits, dim=-1).to(dtype=value.dtype)
            read = torch.einsum("btkn,bnh->btkh", posterior, value)
            result = result + interval_weight[:, interval].to(
                dtype=read.dtype
            )[None, :, None, None] * read
            entropy_rows.append(
                -(
                    posterior.float().clamp_min(1e-8)
                    * posterior.float().clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                / math.log(float(max(int(value.shape[1]), 2)))
            )
            read_rows.append(read.detach().float().square().mean().sqrt())
        metrics = {
            "flow_jepa_p2_structured_effect_read_rms": (
                result.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_p2_structured_effect_entropy": torch.stack(
                entropy_rows
            ).mean(),
            "flow_jepa_p2_structured_effect_interval_rms_variation": (
                torch.stack(read_rows).std(unbiased=False)
            ),
        }
        return result, metrics

    def forward(
        self,
        *,
        p1_delta: Tensor,
        p2_delta: Tensor,
        protected_detail: Tensor,
        progressive_state: ProgressiveGroundingAddressState,
        goal_phase: GoalPhaseState | StatelessIntentState | IntentStateBank,
        p2_effect: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[PolicyPlanDeltaBank, dict[str, Tensor]]:
        batch = int(p1_delta.shape[0])
        expected = (
            batch,
            self.horizon,
            self.basis,
            self.hidden,
        )
        for name, value in (
            ("P1 delta", p1_delta),
            ("P2 delta", p2_delta),
            ("protected detail", protected_detail),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"P3 {name} must be [B,T,K,H]")
        goal_phase.validate(
            batch=batch,
            program_states=int(goal_phase.goal_program.shape[1]),
            intervals=self.intervals,
            hidden=self.hidden,
        )
        basis = self.basis_identity.to(
            device=p1_delta.device, dtype=p1_delta.dtype
        ).expand(batch, self.horizon, -1, -1)
        structured_effect_metrics: dict[str, Tensor] = {}
        if self.effect_read_in_p2:
            if p2_effect is None or tuple(p2_effect.shape) != expected:
                raise ValueError("V117 P3 requires the P2-owned effect read")
            effect_time = p2_effect
        elif self.supervised_effect_mainline:
            effect_time, structured_effect_metrics = (
                self._structured_effect_read(
                    p2_delta + basis,
                    progressive_state,
                )
            )
        else:
            effect_time = self._interval_to_action(
                self._effect_summary(progressive_state)
            )[:, :, None].expand(-1, -1, self.basis, -1)
        phase_time = (
            goal_phase.temporal_control
            if isinstance(goal_phase, (StatelessIntentState, IntentStateBank))
            else self._interval_to_action(goal_phase.interval_selector)
        )[:, :, None].expand(-1, -1, self.basis, -1)
        history_time = self._interval_to_action(
            goal_phase.history_context
        )[:, :, None].expand(-1, -1, self.basis, -1)
        goal_time = None
        active = None
        next_goal = None
        remaining = None
        uncertainty = None
        if not self.supervised_effect_mainline:
            goal_time = self._interval_to_action(
                goal_phase.goal_context
            )[:, :, None].expand(-1, -1, self.basis, -1)
            active = self._expand_global_condition(
                goal_phase.active_goal,
                batch=batch,
                name="active goal",
            )
            next_goal = self._expand_global_condition(
                goal_phase.next_goal,
                batch=batch,
                name="next goal",
            )
            remaining = self._expand_global_condition(
                goal_phase.remaining_goal,
                batch=batch,
                name="remaining goal",
            )
            if self.terminal_uncertainty is None:
                raise RuntimeError("legacy P3 terminal projection is missing")
            uncertainty = self._expand_global_condition(
                self.terminal_uncertainty(goal_phase.phase_uncertainty),
                batch=batch,
                name="phase uncertainty",
            )

        raw_lanes = {
            "precision": self.precision_lane(
                torch.cat((protected_detail, p1_delta, basis), dim=-1)
            ),
            "effect": self.effect_lane(
                effect_time
                if self.supervised_effect_mainline
                else torch.cat(
                    (effect_time, p2_delta, goal_time, basis), dim=-1
                )
            ),
            "temporal": self.temporal_lane(
                torch.cat((p2_delta, phase_time, history_time, basis), dim=-1)
            ),
        }
        if not self.supervised_effect_mainline:
            if (
                self.terminal_lane is None
                or uncertainty is None
                or active is None
                or next_goal is None
                or remaining is None
            ):
                raise RuntimeError("legacy P3 terminal lane is missing")
            raw_lanes["terminal"] = self.terminal_lane(
                torch.cat(
                    (
                        active,
                        next_goal,
                        remaining,
                        uncertainty,
                        basis,
                    ),
                    dim=-1,
                )
            )
        lane_bounds = {
            "precision": 0.35,
            "effect": 0.35,
            "temporal": 0.30,
            # Terminal/exit evidence is deliberately the smallest optional
            # write: an early exit is costlier than continuing a plan, so it
            # must not dominate the shared action carrier by amplitude alone.
            "terminal": 0.12,
        }
        bounded = {
            name: smooth_rms_contract(
                value, lane_bounds[name]
            )[0]
            for name, value in raw_lanes.items()
        }
        execution_terminal = None
        if self.supervised_effect_mainline:
            if goal_phase.terminal_probability is None:
                raise RuntimeError(
                    "V116 P3 requires separate terminal completion evidence"
                )
            execution_terminal = ExecutionTerminalEvidence(
                probability=goal_phase.terminal_probability,
                uncertainty=goal_phase.phase_uncertainty,
            )
        bank = PolicyPlanDeltaBank(
            protected_base=protected_detail,
            precision=bounded["precision"],
            effect=bounded["effect"],
            temporal=bounded["temporal"],
            terminal=bounded.get("terminal"),
            execution_terminal=execution_terminal,
        )
        bank.validate(
            batch=batch,
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return bank, {}
        metrics = {
            "flow_jepa_policy_plan_compiler_active": (
                p1_delta.new_ones((), dtype=torch.float32)
            ),
            "flow_jepa_supervised_effect_mainline_active": (
                p1_delta.new_tensor(
                    float(self.supervised_effect_mainline),
                    dtype=torch.float32,
                )
            ),
            "flow_jepa_policy_plan_protected_base_rms": (
                protected_detail.detach().float().square().mean().sqrt()
            ),
        }
        metrics.update(structured_effect_metrics)
        if execution_terminal is not None:
            metrics.update(
                {
                    "flow_jepa_execution_terminal_probability": (
                        execution_terminal.probability.detach().float().mean()
                    ),
                    "flow_jepa_execution_terminal_uncertainty": (
                        execution_terminal.uncertainty.detach().float().mean()
                    ),
                }
            )
        for name in bank.source_names:
            short = name.removeprefix("p3_")
            value = getattr(bank, short)
            metrics[f"flow_jepa_policy_plan_{short}_rms"] = (
                value.detach().float().square().mean().sqrt()
            )
        lane_values = [bank.precision, bank.effect, bank.temporal]
        if bank.terminal is not None:
            lane_values.append(bank.terminal)
        lanes = torch.stack(lane_values, dim=1).detach().float()
        normalized = F.normalize(lanes, dim=-1, eps=1e-6)
        similarity = torch.einsum(
            "bstkh,butkh->bsu", normalized, normalized
        ) / float(self.horizon * self.basis)
        lane_count = len(lane_values)
        off_diagonal = ~torch.eye(
            lane_count, device=similarity.device, dtype=torch.bool
        )[None]
        metrics["flow_jepa_policy_plan_lane_cosine"] = (
            similarity.masked_select(off_diagonal).mean()
        )
        metrics["flow_jepa_policy_plan_lane_variation"] = (
            lanes.std(dim=1, unbiased=False).mean()
        )
        return bank, metrics


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
                and not self.object_intent_dynamics_mainline
            )
            else None
        )
        self.history_query_proj = (
            nn.Linear(hidden, hidden, bias=False)
            if (
                self.functional_mainline_routing
                and not self.differential_intent_effect_mainline
                and not self.object_intent_dynamics_mainline
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
                if self.object_intent_dynamics_mainline:
                    if not self.shared_factual_glimpse_bank:
                        raise ValueError(
                            "object factual docking requires the shared P1 glimpse bank"
                        )
                    object_feature_width = raw_dim + 5 + 3 * route_dim
                    self.object_dock_value_heads = nn.ModuleList(
                        [
                            nn.Sequential(
                                route_norm(object_feature_width),
                                nn.Linear(
                                    object_feature_width,
                                    2 * self.head_dim,
                                    bias=False,
                                ),
                                nn.SiLU(),
                                nn.Linear(
                                    2 * self.head_dim,
                                    self.head_dim,
                                    bias=False,
                                ),
                            )
                            for _ in range(heads)
                        ]
                    )
                else:
                    self.object_dock_value_heads = None
            else:
                self.raw_micro_grid = 0
                self.register_buffer(
                    "typed_micro_basis", None, persistent=False
                )
                self.typed_fine_query = None
                self.typed_coarse_query = None
                self.appearance_world_owner_query = None
                self.typed_local_refiners = None
                self.object_dock_value_heads = None
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
            self.object_dock_value_heads = None

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
        object_facts: ObjectFactSet | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], ObjectFactualDock | None]:
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
        object_support: Tensor | None = None
        object_typed_support: dict[str, Tensor] = {}
        object_prior: Tensor | None = None
        object_null_support: Tensor | None = None
        object_null_prior: Tensor | None = None
        if self.object_intent_dynamics_mainline:
            if object_facts is None:
                raise ValueError(
                    "object P1 requires the completed global-object facts"
                )
            object_facts.validate()
            expected_assignment = (
                batch,
                object_facts.objects,
                cameras,
                grid,
                grid,
                slots,
            )
            if tuple(object_facts.candidate_assignment.shape) != expected_assignment:
                raise ValueError(
                    "object candidate assignment does not align with the P1 chart"
                )
            assignment = object_facts.candidate_assignment.float()
            assignment_total = assignment.flatten(2).sum(dim=-1)
            object_support = assignment / assignment_total[
                :, :, None, None, None, None
            ].clamp_min(1e-8)
            for name, typed_assignment in (
                ("semantic", object_facts.semantic_candidate_assignment),
                ("appearance", object_facts.appearance_candidate_assignment),
                ("geometry", object_facts.geometry_candidate_assignment),
            ):
                typed_assignment = typed_assignment.float()
                typed_total = typed_assignment.flatten(2).sum(dim=-1)
                object_typed_support[name] = typed_assignment / typed_total[
                    :, :, None, None, None, None
                ].clamp_min(1e-8)
            null_assignment = object_facts.null_assignment.float()
            if tuple(null_assignment.shape) != (
                batch,
                cameras,
                grid,
                grid,
                slots,
            ):
                raise ValueError(
                    "object null assignment does not align with the P1 chart"
                )
            null_total = null_assignment.flatten(1).sum(dim=-1, keepdim=True)
            object_null_support = null_assignment / null_total[
                :, :, None, None, None
            ].clamp_min(1e-8)
            partition = assignment_total.sum(dim=-1, keepdim=True) + null_total
            object_prior = assignment_total / partition.clamp_min(1e-8)
            object_null_prior = null_total / partition.clamp_min(1e-8)
        elif object_facts is not None:
            raise ValueError(
                "global-object facts were supplied outside the object mainline"
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
        object_dock_fact_rows: list[Tensor] = []
        object_dock_posterior_rows: list[Tensor] = []
        object_dock_null_rows: list[Tensor] = []
        object_dock_chart_rows: list[Tensor] = []
        object_dock_coordinate_rows: list[Tensor] = []
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
            object_conditional_route: Tensor | None = None
            object_typed_routes: dict[str, Tensor] = {}
            object_posterior_row: Tensor | None = None
            object_null_posterior_row: Tensor | None = None
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
                if self.object_intent_dynamics_mainline:
                    if (
                        object_support is None
                        or object_prior is None
                        or object_null_support is None
                        or object_null_prior is None
                        or object_facts is None
                    ):
                        raise RuntimeError(
                            "object P1 lost its global-to-local support dock"
                        )
                    objects = object_facts.objects
                    support_log = object_support.clamp_min(1e-12).log()
                    object_route_logit = (
                        route_logits[:, :, :, None]
                        + support_log[:, None, None]
                    )
                    object_route_flat = object_route_logit.flatten(4)
                    object_conditional_route = torch.softmax(
                        object_route_flat, dim=-1
                    ).reshape(
                        batch,
                        stop - start,
                        glimpses,
                        objects,
                        cameras,
                        grid,
                        grid,
                        slots,
                    )
                    object_compatibility = torch.logsumexp(
                        object_route_flat, dim=-1
                    ) + object_prior.clamp_min(1e-8).log()[:, None, None]
                    null_route_flat = (
                        route_logits
                        + object_null_support.clamp_min(1e-12).log()[
                            :, None, None
                        ]
                    ).flatten(3)
                    null_compatibility = torch.logsumexp(
                        null_route_flat, dim=-1
                    ) + object_null_prior.clamp_min(1e-8).log()[:, None]
                    object_with_null = torch.softmax(
                        torch.cat(
                            (object_compatibility, null_compatibility[..., None]),
                            dim=-1,
                        ),
                        dim=-1,
                    )
                    object_posterior_row = object_with_null[..., :-1]
                    object_null_posterior_row = object_with_null[..., -1:]
                    # P1 is a current-fact read and therefore conditions on a
                    # non-null object.  The original null probability remains
                    # explicit for P2; it is not discarded or converted into a
                    # fake background object.
                    conditional_object_mass = object_posterior_row / (
                        1.0 - object_null_posterior_row
                    ).clamp_min(1e-6)
                    route_weights = torch.einsum(
                        "bqgk,bqgkcijm->bqgcijm",
                        conditional_object_mass,
                        object_conditional_route,
                    )
                    for name, typed_support in object_typed_support.items():
                        typed_route_logit = (
                            route_logits[:, :, :, None]
                            + typed_support.clamp_min(1e-12).log()[
                                :, None, None
                            ]
                        )
                        object_typed_routes[name] = torch.softmax(
                            typed_route_logit.flatten(4), dim=-1
                        ).reshape_as(object_conditional_route)
                else:
                    route_weights = torch.softmax(
                        route_logits_flat, dim=-1
                    ).reshape(
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
                    object_source_value: Tensor | None = None
                    object_source_chart: Tensor | None = None
                    object_source_coordinate: Tensor | None = None
                    if self.object_intent_dynamics_mainline:
                        if (
                            object_conditional_route is None
                            or object_posterior_row is None
                            or object_null_posterior_row is None
                            or self.object_dock_value_heads is None
                            or set(object_typed_routes)
                            != {"semantic", "appearance", "geometry"}
                        ):
                            raise RuntimeError(
                                "object P1 did not construct its typed factual dock"
                            )
                        # Reuse the exact fine posterior already paid for by
                        # P1.  This creates genuine K-specific current facts;
                        # no pooled P1 hidden is expanded back into an object
                        # axis.  The expensive 3x3 micro refiners below still
                        # execute only once on the selected aggregate.
                        state_rgb = torch.einsum(
                            "bqgcijmn,bcijmnv->bqgcijmv",
                            fine_weights,
                            typed_literal_rgb.float(),
                        )
                        state_detail = torch.einsum(
                            "bqgcijmn,bcijmnv->bqgcijmv",
                            fine_weights,
                            fine_values.float(),
                        )
                        state_coordinate = torch.einsum(
                            "bqgcijmn,bcijmnd->bqgcijmd",
                            fine_weights,
                            typed_coordinates.float(),
                        )
                        object_rgb = torch.einsum(
                            "bqgkcijm,bqgcijmv->bqgkv",
                            object_conditional_route,
                            state_rgb,
                        )
                        object_detail = torch.einsum(
                            "bqgkcijm,bqgcijmv->bqgkv",
                            object_conditional_route,
                            state_detail,
                        )
                        object_source_coordinate_global = torch.einsum(
                            "bqgkcijm,bqgcijmd->bqgkd",
                            object_conditional_route,
                            state_coordinate,
                        )
                        object_camera_mass = object_conditional_route.sum(
                            dim=(-3, -2, -1)
                        )
                        object_source_coordinate = torch.einsum(
                            "bqgkcijm,bqgcijmd->bqgkcd",
                            object_conditional_route,
                            state_coordinate,
                        ) / object_camera_mass[..., None].clamp_min(1e-6)
                        object_typed_contexts: dict[str, Tensor] = {}
                        for name, typed_key in typed_fine_keys.items():
                            state_key = torch.einsum(
                                "bqgcijmn,bcijmnr->bqgcijmr",
                                fine_weights,
                                typed_key.float(),
                            )
                            object_typed_contexts[name] = torch.einsum(
                                "bqgkcijm,bqgcijmr->bqgkr",
                                object_typed_routes[name],
                                state_key,
                            )
                        object_feature = torch.cat(
                            (
                                object_rgb,
                                object_detail,
                                object_source_coordinate_global,
                                object_typed_contexts["semantic"],
                                object_typed_contexts["appearance"],
                                object_typed_contexts["geometry"],
                            ),
                            dim=-1,
                        ).float()
                        object_source_value = torch.stack(
                            tuple(
                                head(object_feature[:, :, glimpse_index]).to(
                                    dtype=query_row.dtype
                                )
                                for glimpse_index, head in enumerate(
                                    self.object_dock_value_heads
                                )
                            ),
                            dim=2,
                        )
                        object_source_chart = object_conditional_route.sum(dim=-1)
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
                    if self.object_intent_dynamics_mainline:
                        if (
                            object_source_value is None
                            or object_source_chart is None
                            or object_source_coordinate is None
                            or object_posterior_row is None
                            or object_null_posterior_row is None
                        ):
                            raise RuntimeError(
                                "object factual values were lost before the P1 dock"
                            )
                        crossed_object_value = torch.einsum(
                            "bqags,bqskd->bqagkd",
                            cross_weights,
                            object_source_value.float(),
                        )
                        fact_by_object = crossed_object_value.permute(
                            0, 1, 2, 4, 3, 5
                        ).flatten(start_dim=-2)
                        fact_by_object, _ = smooth_rms_contract(
                            fact_by_object.to(dtype=typed_output.dtype), 0.35
                        )
                        crossed_object_posterior = torch.einsum(
                            "bqags,bqsk->bqagk",
                            cross_weights,
                            object_posterior_row,
                        )
                        crossed_null_posterior = torch.einsum(
                            "bqags,bqsk->bqagk",
                            cross_weights,
                            object_null_posterior_row,
                        )
                        crossed_chart = torch.einsum(
                            "bqags,bqskcyx->bqagkcyx",
                            cross_weights,
                            object_source_chart,
                        )
                        crossed_coordinate = torch.einsum(
                            "bqags,bqskcd->bqagkcd",
                            cross_weights,
                            object_source_coordinate,
                        )
                        object_dock_fact_rows.append(fact_by_object)
                        object_dock_posterior_rows.append(
                            crossed_object_posterior.mean(dim=3)
                        )
                        object_dock_null_rows.append(
                            crossed_null_posterior.mean(dim=3)
                        )
                        object_dock_chart_rows.append(
                            crossed_chart.mean(dim=3)
                        )
                        object_dock_coordinate_rows.append(
                            crossed_coordinate.mean(dim=3)
                        )
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
        object_dock: ObjectFactualDock | None = None
        if self.object_intent_dynamics_mainline:
            if not all(
                (
                    object_dock_fact_rows,
                    object_dock_posterior_rows,
                    object_dock_null_rows,
                    object_dock_chart_rows,
                    object_dock_coordinate_rows,
                )
            ):
                raise RuntimeError("object P1 produced no factual dock rows")
            dock_object_posterior = torch.cat(
                object_dock_posterior_rows, dim=1
            ).float()
            dock_null_posterior = torch.cat(
                object_dock_null_rows, dim=1
            ).float()
            dock_mass = (
                dock_object_posterior.sum(dim=-1, keepdim=True)
                + dock_null_posterior
            ).clamp_min(1e-8)
            # Cross-glimpse attention may execute in BF16 under the outer
            # training autocast.  Renormalize the K+null probability once in
            # FP32 at the typed dock instead of accepting a ~1/256 mass drift.
            dock_object_posterior = dock_object_posterior / dock_mass
            dock_null_posterior = dock_null_posterior / dock_mass
            object_dock = ObjectFactualDock(
                fact_by_object=(
                    torch.cat(object_dock_fact_rows, dim=1) * self.fixed_scale
                ),
                object_posterior=dock_object_posterior,
                null_posterior=dock_null_posterior,
                chart_posterior=torch.cat(object_dock_chart_rows, dim=1),
                camera_coordinates=torch.cat(
                    object_dock_coordinate_rows, dim=1
                ),
                aggregate_fact=update,
            )
            object_dock.validate()
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
            }, object_dock
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
        if object_dock is not None:
            metrics.update(
                {
                    "object_p1_dock_object_mass": (
                        object_dock.object_posterior.detach().float().sum(dim=-1).mean()
                    ),
                    "object_p1_dock_null_mass": (
                        object_dock.null_posterior.detach().float().mean()
                    ),
                    "object_p1_dock_object_variation": (
                        object_dock.fact_by_object.detach().float().std(
                            dim=3, unbiased=False
                        ).mean()
                    ),
                    "object_p1_dock_chart_entropy": (
                        -(
                            object_dock.chart_posterior.detach().float()
                            .flatten(-3)
                            .clamp_min(1e-8)
                            * object_dock.chart_posterior.detach().float()
                            .flatten(-3)
                            .clamp_min(1e-8)
                            .log()
                        ).sum(dim=-1)
                        / math.log(
                            float(
                                max(
                                    int(
                                        object_dock.chart_posterior.shape[-3]
                                        * object_dock.chart_posterior.shape[-2]
                                        * object_dock.chart_posterior.shape[-1]
                                    ),
                                    2,
                                )
                            )
                        )
                    ).mean(),
                }
            )
        return updated, metrics, object_dock

    def forward(
        self,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor,
        detail: LateRawDetailEvidence,
        phase_context: Tensor | None = None,
        condition_query_context: Tensor | None = None,
        history_query_context: Tensor | None = None,
        clean_basis_tokens: Tensor | None = None,
        object_facts: ObjectFactSet | None = None,
        collect_diagnostics: bool = True,
    ) -> LateRawDetailReadResult:
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
            if (
                self.differential_intent_effect_mainline
                or self.object_intent_dynamics_mainline
            ):
                if (
                    condition_query_context is not None
                    or history_query_context is not None
                    or self.condition_query_proj is not None
                    or self.history_query_proj is not None
                ):
                    raise ValueError(
                        "explicit P1 accepts only its canonical intent context"
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
            updated, metrics, object_dock = self._read_soft_address_lattice(
                query_input,
                trajectory,
                world_horizon_grid,
                detail,
                clean_basis_tokens=clean_basis_tokens,
                factual_condition=factual_condition,
                object_facts=object_facts,
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
            return LateRawDetailReadResult(
                trajectory=updated.reshape_as(trajectory_tokens),
                metrics=metrics,
                object_dock=object_dock,
            )
        if self.soft_address_lattice:
            raise RuntimeError("soft address lattice reader received no address bank")
        if object_facts is not None:
            raise ValueError("object factual docking requires the soft address lattice")
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
        return LateRawDetailReadResult(
            trajectory=updated.reshape_as(trajectory_tokens),
            metrics={
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
            },
            object_dock=None,
        )


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
        self.functional_mainline_routing = bool(
            int(getattr(config, "flow_jepa_functional_mainline_routing", 0))
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
            if (
                int(getattr(config, "stateless_phase_enabled", 0))
                and not self.functional_mainline_routing
            )
            else None
        )
        self.stateless_intent_controller = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_stateless_intent_controller",
                    0,
                )
            )
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
        self.explicit_object_top = bool(
            self.grounded_intent_effect_mainline
            or self.object_intent_dynamics_mainline
        )
        if self.object_intent_dynamics_mainline:
            OBJECT_INTENT_DYNAMICS_MANIFEST.validate()
            if self.goal_resampler is not None:
                # Full T5 tokens land directly in the capability-owned S.
                # The ancestral task resampler remains serializable only.
                self.goal_resampler.requires_grad_(False)
            self.object_grounder = DenseObjectGrounder(
                hidden=h,
                content_dim=int(config.visual_token_dim),
                route_dim=int(config.flow_jepa_address_route_dim),
                objects=4,
                iterations=3,
            )
            self.object_intent_organizer = StatelessObjectIntentOrganizer(
                hidden=h,
                goal_dim=int(config.goal_language_dim),
                state_dim=int(config.state_dim),
                action_dim=int(config.action_dim),
                content_dim=int(config.visual_token_dim),
                route_dim=int(config.flow_jepa_address_route_dim),
                horizon=int(config.action_horizon),
                heads=int(config.num_heads),
            )
            self.object_future_teacher = ObjectFutureTeacher(
                content_dim=int(config.visual_token_dim),
                key_dim=64,
            )
            self.object_plan_recognizer = FuturePlanRecognizer(
                hidden=h,
                action_dim=int(config.action_dim),
                state_dim=int(config.state_dim),
                content_dim=int(config.visual_token_dim),
                heads=int(config.num_heads),
            )
            self.object_coarse_action = CoarseActionIntent(
                hidden=h,
                action_dim=int(config.action_dim),
                heads=int(config.num_heads),
            )
            self.object_future_compiler = ObjectFutureDynamicsCompiler(
                hidden=h,
                content_dim=int(config.visual_token_dim),
                route_dim=int(config.flow_jepa_address_route_dim),
                heads=int(config.num_heads),
            )
        else:
            self.object_grounder = None
            self.object_intent_organizer = None
            self.object_future_teacher = None
            self.object_plan_recognizer = None
            self.object_coarse_action = None
            self.object_future_compiler = None
        if self.object_intent_dynamics_mainline:
            # S is owned by the capability package below; the ancestry flag is
            # retained only because the shared 3-2-3 foundation validates it.
            self.stateless_goal_phase_machine = None
        elif self.grounded_intent_effect_mainline:
            GROUNDING_MANIFEST.validate()
            self.stateless_goal_phase_machine = StatelessIntentOrganizer(
                hidden=h,
                state_dim=int(config.state_dim),
                action_dim=int(config.action_dim),
                fact_dim=int(config.flow_jepa_address_route_dim),
                action_horizon=int(config.action_horizon),
                goal_dim=int(config.goal_language_dim),
                heads=int(config.num_heads),
            )
        elif self.differential_intent_effect_mainline:
            self.stateless_goal_phase_machine = (
                DifferentialStatelessIntentController(
                    h,
                    int(getattr(config, "stateless_phase_count", 4)),
                    int(config.future_anchors),
                    int(config.action_horizon),
                    state_dim=int(config.state_dim),
                    action_dim=int(config.action_dim),
                    heads=int(config.num_heads),
                )
            )
        elif self.stateless_intent_controller:
            self.stateless_goal_phase_machine = StatelessIntentController(
                h,
                int(getattr(config, "stateless_phase_count", 4)),
                int(config.future_anchors),
                int(config.action_horizon),
                state_dim=int(config.state_dim),
                action_dim=int(config.action_dim),
                control_hidden=256,
                heads=4,
                windows=int(getattr(config, "flow_jepa_future_slots", 3)),
            )
        elif int(
            getattr(
                config,
                "flow_jepa_stateless_goal_phase_machine",
                0,
            )
        ):
            self.stateless_goal_phase_machine = StatelessGoalPhaseMachine(
                h,
                int(getattr(config, "stateless_phase_count", 4)),
                int(config.future_anchors),
                int(config.num_heads),
                state_dim=int(config.state_dim),
                action_dim=int(config.action_dim),
                separate_terminal=bool(
                    int(
                        getattr(
                            config,
                            "flow_jepa_supervised_effect_mainline",
                            0,
                        )
                    )
                ),
            )
        else:
            self.stateless_goal_phase_machine = None
        self.stateless_horizon_adapter = (
            StatelessHorizonConditionAdapter(
                h,
                int(config.future_anchors),
                int(config.num_heads),
            )
            if (
                self.functional_mainline_routing
                and self.stateless_goal_phase_machine is None
                and not self.object_intent_dynamics_mainline
            )
            else None
        )
        self.phase_world_query_proj = (
            nn.Linear(h, h, bias=False)
            if (
                (
                    self.stateless_phase_adapter is not None
                    or self.stateless_horizon_adapter is not None
                    or self.stateless_goal_phase_machine is not None
                )
                and int(getattr(config, "role_attnres_world_to_policy", 0))
                and not self.explicit_object_top
            )
            else None
        )
        self.condition_world_query_proj = (
            nn.Linear(h, h, bias=False)
            if (
                (
                    self.stateless_phase_adapter is not None
                    or self.stateless_horizon_adapter is not None
                    or self.stateless_goal_phase_machine is not None
                )
                and int(getattr(config, "role_attnres_world_to_policy", 0))
                and not (
                    self.differential_intent_effect_mainline
                    or self.explicit_object_top
                )
            )
            else None
        )
        self.history_world_query_proj = (
            nn.Linear(h, h, bias=False)
            if (
                (
                    self.stateless_horizon_adapter is not None
                    or self.stateless_goal_phase_machine is not None
                )
                and int(getattr(config, "role_attnres_world_to_policy", 0))
                and not (
                    self.differential_intent_effect_mainline
                    or self.explicit_object_top
                )
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
        self.horizon_phase_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks) + 1)
                ]
            )
            if (
                self.functional_mainline_routing
                and not (
                    self.differential_intent_effect_mainline
                    or self.explicit_object_top
                )
            )
            else None
        )
        self.horizon_goal_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks) + 1)
                ]
            )
            if (
                self.functional_mainline_routing
                and not (
                    self.differential_intent_effect_mainline
                    or self.explicit_object_top
                )
            )
            else None
        )
        self.horizon_history_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks) + 1)
                ]
            )
            if (
                self.functional_mainline_routing
                and not (
                    self.differential_intent_effect_mainline
                    or self.explicit_object_top
                )
            )
            else None
        )
        v115_typed_horizon_context = bool(
            int(getattr(config, "flow_jepa_policy_plan_compiler", 0))
            and not (
                self.differential_intent_effect_mainline
                or self.explicit_object_top
            )
        )
        self.supervised_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_supervised_effect_mainline",
                    0,
                )
            )
        )
        self.horizon_proposal_world_block_query_proj = (
            nn.ModuleList(
                [
                    nn.Linear(h, h, bias=False)
                    for _ in range(int(config.flow_jepa_world_blocks) + 1)
                ]
            )
            if (
                self.supervised_effect_mainline
                and not self.explicit_object_top
            )
            else None
        )
        self.grounded_clean_proposal_proj = (
            nn.Linear(h, h, bias=False)
            if self.grounded_intent_effect_mainline
            else None
        )
        self.horizon_typed_context_router = (
            nn.ModuleList(
                [
                    RoleDeltaAttnRes(
                        h,
                        int(getattr(config, "role_attnres_key_dim", 32)),
                        max_sources=(
                            4 if self.supervised_effect_mainline else 3
                        ),
                        include_null=True,
                        max_value_rms=0.35,
                        normalization_floor=float(
                            getattr(
                                config,
                                "flow_jepa_routing_norm_floor",
                                0.25,
                            )
                        ),
                    )
                    for _ in range(
                        int(config.flow_jepa_world_blocks) + 1
                    )
                ]
            )
            if v115_typed_horizon_context
            else None
        )
        if v115_typed_horizon_context:
            self.horizon_typed_context_query = nn.Parameter(
                torch.randn(
                    int(config.flow_jepa_world_blocks) + 1,
                    int(config.future_anchors),
                    h,
                )
                * 0.02
            )
        else:
            self.register_parameter(
                "horizon_typed_context_query", None
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
            if (
                int(getattr(config, "role_attnres_ground_to_world", 0))
                and not self.explicit_object_top
            )
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
            if (
                int(getattr(config, "role_attnres_world_to_policy", 0))
                and not self.explicit_object_top
            )
            else None
        )
        self.late_raw_detail_reader = (
            LateRawDetailPolicyReader(config)
            if int(getattr(config, "flow_jepa_late_policy_detail", 0))
            else None
        )
        if self.object_intent_dynamics_mainline:
            self.policy_plan_compiler = ObjectPolicyPlanCompiler(
                hidden=h,
                horizon=int(config.action_horizon),
                basis=int(config.action_basis_tokens),
            )
            self.consequence_plan_organizer = ZeroPreservingObjectConsequence(h)
        elif self.grounded_intent_effect_mainline:
            self.policy_plan_compiler = (
                ConsequenceConditionedPolicyPlanCompiler(
                    hidden=h,
                    horizon=int(config.action_horizon),
                    basis=int(config.action_basis_tokens),
                )
            )
            self.consequence_plan_organizer = (
                ZeroPreservingConsequenceOrganizer(h)
            )
        elif self.differential_intent_effect_mainline:
            self.policy_plan_compiler = DifferentialPolicyPlanCompiler(
                hidden=h,
                horizon=int(config.action_horizon),
                basis=int(config.action_basis_tokens),
            )
            self.consequence_plan_organizer = ConsequencePlanOrganizer(h)
        else:
            self.policy_plan_compiler = (
                PolicyPlanCompiler(config)
                if int(
                    getattr(config, "flow_jepa_policy_plan_compiler", 0)
                )
                else None
            )
            self.consequence_plan_organizer = None
        self.effect_read_in_p2 = bool(
            int(getattr(config, "flow_jepa_effect_read_in_p2", 0))
            or self.explicit_object_top
        )
        if self.object_intent_dynamics_mainline:
            self.p2_effect_reader = ObjectFutureEffectReader(
                hidden=h,
                content_dim=int(config.visual_token_dim),
            )
        elif self.grounded_intent_effect_mainline:
            self.p2_effect_reader = GroundedFutureEffectReader(
                hidden=h,
                horizon=int(config.action_horizon),
                basis=int(config.action_basis_tokens),
                effect_dim=int(config.visual_token_dim),
            )
        elif self.differential_intent_effect_mainline:
            self.p2_effect_reader = DifferentialFutureEffectReader(
                hidden=h,
                horizon=int(config.action_horizon),
                basis=int(config.action_basis_tokens),
            )
        else:
            self.p2_effect_reader = (
                StructuredFutureEffectReader(config)
                if self.effect_read_in_p2
                else None
            )
        if self.policy_plan_compiler is not None:
            if not self.block_roles or self.block_roles[-1] != "policy":
                raise RuntimeError(
                    "policy plan compiler must replace the final policy block"
                )
            # Preserve the serialized eight-block skeleton while removing the
            # ordinary P3 computation and optimizer ownership.
            self.blocks[-1].requires_grad_(False)
        if self.explicit_object_top:
            grounding_blocks = int(config.flow_jepa_grounding_blocks)
            world_blocks = int(config.flow_jepa_world_blocks)
            for block in self.blocks[
                grounding_blocks : grounding_blocks + world_blocks
            ]:
                block.requires_grad_(False)
            # Grounded P2 is the bounded effect read plus the algebraic
            # consequence update.  Keeping the inherited generic P2 trainable
            # would recreate an unsupervised free organizer beside it.
            self.blocks[-2].requires_grad_(False)
        final_decoder = str(getattr(config, "final_action_decoder", "legacy"))
        self.terminal_policy_layer_contracts_only = bool(
            final_decoder == "evidence_latent_mmdit_action"
            and int(getattr(config, "flow_jepa_role_hierarchy", 0))
            and int(getattr(config, "flow_jepa_strict_role_visual_path", 0))
            and not self.object_intent_dynamics_mainline
        )
        self.midcut_norm = nn.LayerNorm(h)
        self.midcut_heads = MidcutContractHeads(config)
        if (
            int(config.layer_contract_adapters)
            and not self.object_intent_dynamics_mainline
        ):
            self.layer_contract_heads = nn.ModuleList(
                [LayerContractAdapterHeads(config, layer_index=i) for i in range(int(config.depth))]
            )
        else:
            self.layer_contract_heads = nn.ModuleList()
        self.layer_fm_probe = (
            SharedLayerFlowActionProbe(config)
            if (
                int(config.layer_shared_fm_probe)
                and not self.object_intent_dynamics_mainline
            )
            else None
        )
        self.layer_role_scheduler = LayerRoleScheduler(config)
        self.layer_consequence_cell = (
            RecurrentMilestoneConsequenceCell(config)
            if (
                int(config.layer_recurrent_consequence)
                and not self.object_intent_dynamics_mainline
            )
            else None
        )
        if self.object_intent_dynamics_mainline:
            # The object capability has one explicit top-to-bottom ingress:
            # ObjectPolicyPlanDeltaBank.  Historical mid-cut/layer adapters
            # neither own an active loss nor feed the decoder, so allocating
            # trainable parameters for them would create dead optimizer state.
            self.midcut_norm.requires_grad_(False)
            self.midcut_heads.requires_grad_(False)
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
        if (
            self.policy_plan_compiler is not None
            and len(self.layer_contract_heads) > 0
        ):
            # P3 is represented by PolicyPlanDeltaBank, not by a duplicate
            # generic layer-contract readout over the unchanged P2 canvas.
            self.layer_contract_heads[-1].requires_grad_(False)
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
        explicit_policy_handoff = bool(
            strict_policy
            and self.policy_plan_compiler is not None
        )
        if self.flow_dino_evidence is None:
            canvas_summary = canvas.mean(dim=1)
            modulation_source = visual_memory.mean(dim=1)
        else:
            grounded_fact_only = bool(
                self.explicit_object_top and str(role) == "grounding"
            )
            grounded_policy_explicit_only = bool(
                self.explicit_object_top and str(role) == "policy"
            )
            clean_names = (
                ["trajectory"]
                if grounded_policy_explicit_only
                else (
                    ["state", "registers"]
                    if grounded_fact_only
                    else [
                        "task",
                        "state",
                        "state_history",
                        "executed",
                        "registers",
                    ]
                )
            )
            if (
                str(role) != "grounding"
                and not grounded_policy_explicit_only
            ):
                clean_names.append("proposal")
            if strict_policy and not explicit_policy_handoff:
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
            if explicit_policy_handoff:
                # V115 P reads G/W only through the explicit bridge and the
                # supervised FutureEffectField.  Reusing the accumulated W
                # rollout here would recreate the removed world-residual bypass
                # through a global AdaLN side channel.
                modulation_source = canvas_summary
            else:
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
            "address_g3_slot_permute",
            "address_g3_slot_mean",
            "interval_stage_zero",
            "interval_stage_shuffle",
            "future_effect_zero",
            "future_effect_spatial_shuffle",
            "future_effect_current_zero",
            "future_effect_current_spatial_shuffle",
            "future_effect_semantic_zero",
            "future_effect_semantic_spatial_shuffle",
            "future_effect_transport_zero",
            "future_effect_transport_spatial_shuffle",
            "future_effect_reliability_zero",
            "future_effect_reliability_spatial_shuffle",
            "future_effect_reliability_one",
            "grounding_entry_zero",
            "grounding_entry_shuffle",
            "functional_owner_boundary_zero",
            "functional_owner_boundary_shuffle",
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
            "goal_context_zero",
            "history_context_zero",
            "intent_window_selector_uniform",
            "intent_window_selector_shuffle",
            "intent_temporal_zero",
            "intent_temporal_shuffle",
            "intent_state_zero",
            "intent_state_shuffle",
            "intent_window_near_zero",
            "intent_window_near_shuffle",
            "intent_window_mid_zero",
            "intent_window_mid_shuffle",
            "intent_window_late_zero",
            "intent_window_late_shuffle",
            "future_effect_near_zero",
            "future_effect_near_shuffle",
            "future_effect_mid_zero",
            "future_effect_mid_shuffle",
            "future_effect_late_zero",
            "future_effect_late_shuffle",
            "future_effect_far_zero",
            "future_effect_far_shuffle",
            "future_effect_h4_8_zero",
            "future_effect_h4_8_shuffle",
            "future_effect_h8_16_zero",
            "future_effect_h8_16_shuffle",
            "future_effect_h16_32_zero",
            "future_effect_h16_32_shuffle",
            "future_effect_h32_48_zero",
            "future_effect_h32_48_shuffle",
            "intent_goal_set_zero",
            "intent_goal_set_shuffle",
            "intent_achieved_zero",
            "intent_achieved_shuffle",
            "intent_remaining_zero",
            "intent_remaining_shuffle",
            "intent_interval_h4_8_zero",
            "intent_interval_h4_8_shuffle",
            "intent_interval_h8_16_zero",
            "intent_interval_h8_16_shuffle",
            "intent_interval_h16_32_zero",
            "intent_interval_h16_32_shuffle",
            "intent_interval_h32_48_zero",
            "intent_interval_h32_48_shuffle",
        }
        for prefix, count in (
            ("g", int(getattr(self.config, "flow_jepa_grounding_blocks", 0))),
            ("w", int(getattr(self.config, "flow_jepa_world_blocks", 0))),
            ("p", int(getattr(self.config, "flow_jepa_policy_blocks", 0))),
        ):
            for index in range(1, count + 1):
                allowed.add(f"{prefix}{index}_zero")
                allowed.add(f"{prefix}{index}_shuffle")
        if self.policy_plan_compiler is not None:
            if self.object_intent_dynamics_mainline:
                lanes = [
                    "p3_precision",
                    "p3_temporal",
                    "p3_state_change",
                ]
            elif (
                self.differential_intent_effect_mainline
                or self.grounded_intent_effect_mainline
            ):
                lanes = ["p3_precision", "p3_temporal"]
            else:
                lanes = ["p3_precision", "p3_effect", "p3_temporal"]
                if not self.supervised_effect_mainline:
                    lanes.append("p3_terminal")
            for lane in lanes:
                allowed.add(f"{lane}_zero")
                allowed.add(f"{lane}_shuffle")
        if self.functional_mainline_routing:
            for depth in range(int(self.config.flow_jepa_world_blocks) + 1):
                allowed.add(f"functional_w{depth}_route_zero")
                allowed.add(f"functional_w{depth}_route_shuffle")
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
                "interval_stage/future_effect/w2p_far_context/"
                "bottom_far_rollout/all_far_context/protected_detail "
                "zero/shuffle mode"
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
                    and (
                        key.startswith(
                            (
                                "attnres_",
                                "evidence_policy_delta_attnres_",
                                "evidence_protected_detail_basis_",
                                "grounded_p2_effect_value_",
                                "grounded_p2_effect_reliability_",
                            )
                        )
                        or (
                            key.startswith("flow_jepa_functional_w")
                            and key.endswith(
                                "_route_intervention_delta_norm"
                            )
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

    def _intervene_horizon_query_contexts(
        self,
        phase_context: Tensor,
        goal_context: Tensor,
        history_context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        phase_context, goal_context = self._intervene_query_contexts(
            phase_context, goal_context
        )
        if phase_context is None or goal_context is None:
            raise RuntimeError("horizon context intervention removed a context")
        if self._action_path_eval_intervention == "goal_context_zero":
            self._action_path_eval_apply_count += 1
            goal_context = torch.zeros_like(goal_context)
        if self._action_path_eval_intervention == "condition_query_zero":
            history_context = torch.zeros_like(history_context)
        elif self._action_path_eval_intervention == "history_context_zero":
            self._action_path_eval_apply_count += 1
            history_context = torch.zeros_like(history_context)
        return phase_context, goal_context, history_context

    def _intervene_intent_state(
        self,
        state: GroundedIntentState | StatelessIntentState | IntentStateBank,
    ) -> GroundedIntentState | StatelessIntentState | IntentStateBank:
        """Probe V117's selector and temporal outputs without touching S inputs."""

        mode = self._action_path_eval_intervention
        if isinstance(state, GroundedIntentState):
            state.validate(
                batch=int(state.interval_intents.shape[0]),
                hidden=int(state.interval_intents.shape[-1]),
                horizon=int(state.temporal_control.shape[1]),
            )

            def shuffled(value: Tensor, *, structured_dim: int = 1) -> Tensor:
                if int(value.shape[0]) > 1:
                    return value.roll(shifts=1, dims=0)
                if value.ndim > structured_dim and int(value.shape[structured_dim]) > 1:
                    return value.roll(shifts=1, dims=structured_dim)
                return value.roll(
                    shifts=max(int(value.shape[-1]) // 2, 1),
                    dims=-1,
                )

            protected_goal_tokens = state.protected_goal_tokens
            achieved_evidence = state.achieved_evidence
            remaining_goal = state.remaining_goal
            interval_intents = state.interval_intents
            temporal_control = state.temporal_control
            completion_evidence = state.completion_evidence
            completion_probability = state.completion_probability
            completion_uncertainty = state.completion_uncertainty
            if mode == "intent_state_zero":
                protected_goal_tokens = torch.zeros_like(protected_goal_tokens)
                achieved_evidence = torch.zeros_like(achieved_evidence)
                remaining_goal = torch.zeros_like(remaining_goal)
                interval_intents = torch.zeros_like(interval_intents)
                temporal_control = torch.zeros_like(temporal_control)
                completion_evidence = torch.zeros_like(completion_evidence)
                completion_probability = torch.zeros_like(completion_probability)
                completion_uncertainty = torch.zeros_like(completion_uncertainty)
            elif mode == "intent_state_shuffle":
                protected_goal_tokens = shuffled(protected_goal_tokens)
                achieved_evidence = shuffled(achieved_evidence)
                remaining_goal = shuffled(remaining_goal)
                interval_intents = shuffled(interval_intents)
                temporal_control = shuffled(temporal_control)
                completion_evidence = shuffled(completion_evidence)
                completion_probability = shuffled(completion_probability)
                completion_uncertainty = shuffled(completion_uncertainty)
            elif mode in {"intent_goal_set_zero", "intent_goal_set_shuffle"}:
                protected_goal_tokens = (
                    torch.zeros_like(protected_goal_tokens)
                    if mode.endswith("_zero")
                    else shuffled(protected_goal_tokens)
                )
            elif mode in {"intent_achieved_zero", "intent_achieved_shuffle"}:
                achieved_evidence = (
                    torch.zeros_like(achieved_evidence)
                    if mode.endswith("_zero")
                    else shuffled(achieved_evidence)
                )
            elif mode in {"intent_remaining_zero", "intent_remaining_shuffle"}:
                remaining_goal = (
                    torch.zeros_like(remaining_goal)
                    if mode.endswith("_zero")
                    else shuffled(remaining_goal)
                )
            elif mode in {
                "intent_interval_h4_8_zero",
                "intent_interval_h4_8_shuffle",
                "intent_interval_h8_16_zero",
                "intent_interval_h8_16_shuffle",
                "intent_interval_h16_32_zero",
                "intent_interval_h16_32_shuffle",
                "intent_interval_h32_48_zero",
                "intent_interval_h32_48_shuffle",
            }:
                interval_index = {
                    "intent_interval_h4_8_zero": 0,
                    "intent_interval_h4_8_shuffle": 0,
                    "intent_interval_h8_16_zero": 1,
                    "intent_interval_h8_16_shuffle": 1,
                    "intent_interval_h16_32_zero": 2,
                    "intent_interval_h16_32_shuffle": 2,
                    "intent_interval_h32_48_zero": 3,
                    "intent_interval_h32_48_shuffle": 3,
                }[mode]
                interval_intents = interval_intents.clone()
                row = interval_intents[:, interval_index]
                interval_intents[:, interval_index] = (
                    torch.zeros_like(row)
                    if mode.endswith("_zero")
                    else (
                        row.roll(shifts=1, dims=0)
                        if int(row.shape[0]) > 1
                        else state.interval_intents[
                            :,
                            (interval_index + 1) % 4,
                        ]
                    )
                )
            elif mode == "intent_temporal_zero":
                temporal_control = torch.zeros_like(temporal_control)
            elif mode == "intent_temporal_shuffle":
                temporal_control = shuffled(temporal_control)
            else:
                return state
            intervened = replace(
                state,
                protected_goal_tokens=protected_goal_tokens,
                achieved_evidence=achieved_evidence,
                remaining_goal=remaining_goal,
                interval_intents=interval_intents,
                temporal_control=temporal_control,
                completion_evidence=completion_evidence,
                completion_probability=completion_probability,
                completion_uncertainty=completion_uncertainty,
            )
            intervened.validate(
                batch=int(interval_intents.shape[0]),
                hidden=int(interval_intents.shape[-1]),
                horizon=int(temporal_control.shape[1]),
            )
            self._action_path_eval_apply_count += 1
            component_deltas = (
                protected_goal_tokens.float()
                - state.protected_goal_tokens.detach().float(),
                achieved_evidence.float()
                - state.achieved_evidence.detach().float(),
                remaining_goal.float() - state.remaining_goal.detach().float(),
                interval_intents.float()
                - state.interval_intents.detach().float(),
                temporal_control.float()
                - state.temporal_control.detach().float(),
                completion_evidence.float()
                - state.completion_evidence.detach().float(),
            )
            self._action_path_eval_metrics[
                "grounded_intent_boundary_delta_norm"
            ] = float(
                torch.stack(
                    tuple(
                        delta.detach().square().mean().sqrt()
                        for delta in component_deltas
                    )
                )
                .mean()
                .cpu()
            )
            return intervened
        if isinstance(state, IntentStateBank):
            window_tokens = state.window_view.tokens
            temporal = state.temporal_control
            if mode == "intent_state_zero":
                window_tokens = torch.zeros_like(window_tokens)
            elif mode == "intent_state_shuffle":
                window_tokens = (
                    window_tokens.roll(shifts=1, dims=0)
                    if int(window_tokens.shape[0]) > 1
                    else window_tokens.roll(shifts=1, dims=1)
                )
            elif mode in {
                "intent_window_near_zero",
                "intent_window_mid_zero",
                "intent_window_late_zero",
            }:
                index = {
                    "intent_window_near_zero": 0,
                    "intent_window_mid_zero": 1,
                    "intent_window_late_zero": 2,
                }[mode]
                window_tokens = window_tokens.clone()
                window_tokens[:, index] = 0
            elif mode in {
                "intent_window_near_shuffle",
                "intent_window_mid_shuffle",
                "intent_window_late_shuffle",
            }:
                index = {
                    "intent_window_near_shuffle": 0,
                    "intent_window_mid_shuffle": 1,
                    "intent_window_late_shuffle": 2,
                }[mode]
                window_tokens = window_tokens.clone()
                row = window_tokens[:, index]
                window_tokens[:, index] = (
                    row.roll(shifts=1, dims=0)
                    if int(row.shape[0]) > 1
                    else window_tokens[:, (index + 1) % 3]
                )
            elif mode == "intent_temporal_zero":
                temporal = torch.zeros_like(temporal)
            elif mode == "intent_temporal_shuffle":
                temporal = (
                    temporal.roll(shifts=1, dims=0)
                    if int(temporal.shape[0]) > 1
                    else temporal.roll(
                        shifts=max(int(temporal.shape[1]) // 2, 1),
                        dims=1,
                    )
                )
            else:
                return state
            if mode.startswith("intent_state"):
                temporal = F.interpolate(
                    window_tokens.float().transpose(1, 2),
                    size=int(temporal.shape[1]),
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2).to(dtype=window_tokens.dtype)
            self._action_path_eval_apply_count += 1
            self._action_path_eval_metrics[
                "intent_window_state_delta_norm"
            ] = float(
                (window_tokens - state.window_view.tokens)
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
            self._action_path_eval_metrics["intent_temporal_delta_norm"] = float(
                (temporal - state.temporal_control)
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
                .cpu()
            )
            return replace(
                state,
                window_view=replace(
                    state.window_view,
                    tokens=window_tokens,
                    predictive_effect=(
                        torch.zeros_like(state.window_view.predictive_effect)
                        if mode == "intent_state_zero"
                        else state.window_view.predictive_effect
                    ),
                ),
                temporal_control=temporal,
            )
        selector = state.window_selector
        temporal = state.temporal_control
        if mode == "intent_window_selector_uniform":
            selector = torch.full_like(selector, 1.0 / float(selector.shape[1]))
        elif mode == "intent_window_selector_shuffle":
            selector = (
                selector.roll(shifts=1, dims=0)
                if int(selector.shape[0]) > 1
                else selector.roll(shifts=1, dims=1)
            )
        elif mode == "intent_temporal_zero":
            temporal = torch.zeros_like(temporal)
        elif mode == "intent_temporal_shuffle":
            temporal = (
                temporal.roll(shifts=1, dims=0)
                if int(temporal.shape[0]) > 1
                else temporal.roll(
                    shifts=max(int(temporal.shape[1]) // 2, 1), dims=1
                )
            )
        else:
            return state
        self._action_path_eval_apply_count += 1
        self._action_path_eval_metrics["intent_window_selector_delta_norm"] = float(
            (selector - state.window_selector)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        self._action_path_eval_metrics["intent_temporal_delta_norm"] = float(
            (temporal - state.temporal_control)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return replace(
            state,
            window_selector=selector,
            temporal_control=temporal,
        )

    def _functional_world_horizon_context(
        self,
        *,
        depth: int,
        phase_context: Tensor | None,
        goal_context: Tensor | None,
        history_context: Tensor | None,
        proposal_context: Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        if not self.functional_mainline_routing:
            return None, {}
        if self.grounded_intent_effect_mainline:
            if int(depth) != 0:
                raise ValueError(
                    "grounded W accepts the clean proposal only at its entry "
                    "boundary; W1/W2 read the typed intent state directly"
                )
            if any(
                value is not None
                for value in (phase_context, goal_context, history_context)
            ):
                raise ValueError(
                    "grounded W entry forbids generic phase/goal/history aliases"
                )
            if (
                self.grounded_clean_proposal_proj is None
                or proposal_context is None
                or proposal_context.ndim != 3
                or int(proposal_context.shape[-1])
                != int(self.config.hidden_size)
            ):
                raise ValueError(
                    "grounded W entry requires one clean [B,T,H] proposal"
                )
            proposal = F.interpolate(
                proposal_context.float().transpose(1, 2),
                size=4,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2).to(device=device, dtype=dtype)
            proposal, contract = smooth_rms_contract(
                self.grounded_clean_proposal_proj(proposal),
                0.35,
            )
            metrics: dict[str, Tensor] = {}
            if collect_diagnostics:
                metrics = {
                    "grounded_w_clean_proposal_rms": proposal.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "grounded_w_clean_proposal_contract_min": contract.detach()
                    .float()
                    .amin(),
                    "grounded_w_generic_condition_bypass": proposal.new_zeros(
                        (), dtype=torch.float32
                    ),
                }
            return proposal, metrics
        if self.differential_intent_effect_mainline:
            # ``IntentWindowView`` is the differential graph's only learned
            # intent carrier and enters W1/W2 at the effect compiler.  Routing
            # language/history innovations here as well would recreate the
            # parallel condition soup that this graph removes.  The clean
            # proposal remains a distinct, typed W operand.
            if any(
                value is not None
                for value in (phase_context, goal_context, history_context)
            ):
                raise ValueError(
                    "differential W routing forbids direct phase/goal/history "
                    "contexts outside IntentWindowView"
                )
            if (
                self.horizon_proposal_world_block_query_proj is None
                or proposal_context is None
                or proposal_context.ndim != 3
                or int(proposal_context.shape[-1])
                != int(self.config.hidden_size)
            ):
                raise ValueError(
                    "differential W routing requires one clean [B,T,H] "
                    "proposal operand"
                )
            if not 0 <= int(depth) < len(
                self.horizon_proposal_world_block_query_proj
            ):
                raise ValueError("differential W proposal depth is out of range")
            proposal_context = F.interpolate(
                proposal_context.float().transpose(1, 2),
                size=int(self.config.future_anchors),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2).to(device=device, dtype=dtype)
            context = self.horizon_proposal_world_block_query_proj[int(depth)](
                proposal_context
            )
            scale = float(
                getattr(self.config, "stateless_phase_query_scale", 0.10)
            )
            context, _ = smooth_rms_contract(scale * context, 0.35)
            metrics: dict[str, Tensor] = {}
            if collect_diagnostics:
                metrics[
                    f"flow_jepa_w{int(depth)}_clean_proposal_context_rms"
                ] = context.detach().float().square().mean().sqrt()
                metrics[
                    f"flow_jepa_w{int(depth)}_direct_intent_bypass"
                ] = context.new_zeros((), dtype=torch.float32)
            return context, metrics
        banks = (
            self.horizon_phase_world_block_query_proj,
            self.horizon_goal_world_block_query_proj,
            self.horizon_history_world_block_query_proj,
        )
        if any(bank is None for bank in banks):
            raise RuntimeError("functional horizon projection banks are incomplete")
        expected = (
            int(phase_context.shape[0]) if phase_context is not None else -1,
            int(self.config.future_anchors),
            int(self.config.hidden_size),
        )
        for name, context in (
            ("phase", phase_context),
            ("goal", goal_context),
            ("history", history_context),
        ):
            if context is None or tuple(context.shape) != expected:
                raise ValueError(
                    f"functional W {name} context must be [B,anchor,H]"
                )
        if not 0 <= int(depth) < len(banks[0]):
            raise ValueError("functional W context depth is out of range")
        assert phase_context is not None
        assert goal_context is not None
        assert history_context is not None
        if self.supervised_effect_mainline:
            if (
                self.horizon_proposal_world_block_query_proj is None
                or proposal_context is None
                or proposal_context.ndim != 3
                or int(proposal_context.shape[0]) != int(expected[0])
                or int(proposal_context.shape[-1])
                != int(self.config.hidden_size)
            ):
                raise ValueError(
                    "V116 W routing requires clean [B,T,H] proposal context"
                )
            proposal_context = F.interpolate(
                proposal_context.float().transpose(1, 2),
                size=int(self.config.future_anchors),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2).to(device=device, dtype=dtype)
        scale = float(
            getattr(self.config, "stateless_phase_query_scale", 0.10)
        )
        typed_rows = [
                banks[0][int(depth)](
                    phase_context.to(device=device, dtype=dtype)
                ),
                banks[1][int(depth)](
                    goal_context.to(device=device, dtype=dtype)
                ),
                banks[2][int(depth)](
                    history_context.to(device=device, dtype=dtype)
                ),
        ]
        source_names = ["phase", "goal", "history"]
        if self.supervised_effect_mainline:
            assert self.horizon_proposal_world_block_query_proj is not None
            assert proposal_context is not None
            typed_rows.append(
                self.horizon_proposal_world_block_query_proj[int(depth)](
                    proposal_context
                )
            )
            source_names.append("proposal")
        typed_values = torch.stack(typed_rows, dim=-2)
        route_metrics: dict[str, Tensor] = {}
        if self.horizon_typed_context_router is not None:
            if self.horizon_typed_context_query is None:
                raise RuntimeError(
                    "typed horizon router has no ordered query bank"
                )
            query = self.horizon_typed_context_query[int(depth)].to(
                device=device, dtype=dtype
            )[None].expand(int(expected[0]), -1, -1)
            context, raw_metrics = self.horizon_typed_context_router[
                int(depth)
            ](
                query,
                typed_values,
                collect_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                prefix = f"flow_jepa_w{int(depth)}_typed_condition"
                route_metrics = {
                    f"{prefix}_{name}": value
                    for name, value in raw_metrics.items()
                    if name != "source_mass"
                }
                source_mass = raw_metrics.get("source_mass")
                if (
                    isinstance(source_mass, Tensor)
                    and int(source_mass.numel()) == len(source_names)
                ):
                    for source_index, source_name in enumerate(
                        source_names
                    ):
                        route_metrics[
                            f"{prefix}_{source_name}_mass"
                        ] = source_mass[source_index]
        else:
            # Exact ancestral behavior for V113/V114 flags-off graphs.
            context = typed_values.sum(dim=-2) / math.sqrt(
                float(len(source_names))
            )
        context = scale * context
        context, _ = smooth_rms_contract(context, 0.35)
        return context, route_metrics

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
        if self.policy_plan_compiler is not None:
            # V115 deliberately has no whole-world-residual action ingress.
            # Keep measuring the internal W working delta, but make this legacy
            # action probe an explicit identity instead of perturbing an
            # unconsumed tensor and relying on masked-kernel numerical equality.
            world_residual = rollout - world_entry_rollout
            self._action_path_eval_metrics[
                "world_residual_delta_norm"
            ] = float(
                world_residual.detach().float().norm(dim=-1).mean().cpu()
            )
            self._action_path_eval_metrics[
                "world_residual_action_ingress_absent"
            ] = 1.0
            return rollout
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

    def _intervene_future_effect_field(
        self,
        state: ProgressiveGroundingAddressState,
    ) -> None:
        mode = self._action_path_eval_intervention
        if mode not in {
            "future_effect_zero",
            "future_effect_spatial_shuffle",
            "future_effect_current_zero",
            "future_effect_current_spatial_shuffle",
            "future_effect_semantic_zero",
            "future_effect_semantic_spatial_shuffle",
            "future_effect_transport_zero",
            "future_effect_transport_spatial_shuffle",
            "future_effect_reliability_zero",
            "future_effect_reliability_spatial_shuffle",
            "future_effect_reliability_one",
            "future_effect_near_zero",
            "future_effect_mid_zero",
            "future_effect_late_zero",
            "future_effect_near_shuffle",
            "future_effect_mid_shuffle",
            "future_effect_late_shuffle",
            "future_effect_far_zero",
            "future_effect_far_shuffle",
            "future_effect_h4_8_zero",
            "future_effect_h4_8_shuffle",
            "future_effect_h8_16_zero",
            "future_effect_h8_16_shuffle",
            "future_effect_h16_32_zero",
            "future_effect_h16_32_shuffle",
            "future_effect_h32_48_zero",
            "future_effect_h32_48_shuffle",
        }:
            return
        grounded_field = getattr(
            state,
            "world_grounded_effect_field",
            None,
        )
        if grounded_field is not None:
            grounded_field.validate()
            self._action_path_eval_apply_count += 1
            zero_mode = mode.endswith("_zero")

            def changed(value: Tensor) -> Tensor:
                if zero_mode:
                    return torch.zeros_like(value)
                return value.roll(
                    shifts=(
                        max(int(value.shape[3]) // 2, 1),
                        max(int(value.shape[4]) // 3, 1),
                    ),
                    dims=(3, 4),
                )

            def current_changed(value: Tensor) -> Tensor:
                if zero_mode:
                    return torch.zeros_like(value)
                return value.roll(
                    shifts=(
                        max(int(value.shape[2]) // 2, 1),
                        max(int(value.shape[3]) // 3, 1),
                    ),
                    dims=(2, 3),
                )

            current_reference = grounded_field.current_reference
            semantic_delta = grounded_field.semantic_delta
            transport_delta = grounded_field.transport_delta
            covariance_delta = grounded_field.covariance_delta
            visibility_change = grounded_field.visibility_change
            persistence_change = grounded_field.persistence_change
            reliability = grounded_field.reliability
            uncertainty = grounded_field.uncertainty
            if mode in {"future_effect_zero", "future_effect_spatial_shuffle"}:
                semantic_delta = changed(semantic_delta)
                transport_delta = changed(transport_delta)
                covariance_delta = changed(covariance_delta)
                visibility_change = changed(visibility_change)
                persistence_change = changed(persistence_change)
                reliability = changed(reliability)
                uncertainty = changed(uncertainty)
            elif mode.startswith("future_effect_current_"):
                current_reference = current_changed(current_reference)
            elif mode.startswith("future_effect_semantic_"):
                semantic_delta = changed(semantic_delta)
            elif mode.startswith("future_effect_transport_"):
                transport_delta = changed(transport_delta)
                covariance_delta = changed(covariance_delta)
            elif mode == "future_effect_reliability_one":
                # Audit-only counterfactual: bypass the predicted reliability
                # attenuation while keeping content, geometry, validity and
                # uncertainty bit-identical.  This is intentionally not a
                # training-path switch.
                reliability = torch.ones_like(reliability)
            elif mode.startswith("future_effect_reliability_"):
                reliability = changed(reliability)
                uncertainty = changed(uncertainty)
            else:
                interval_index = {
                    "future_effect_near_zero": 0,
                    "future_effect_near_shuffle": 0,
                    "future_effect_mid_zero": 1,
                    "future_effect_mid_shuffle": 1,
                    "future_effect_late_zero": 2,
                    "future_effect_late_shuffle": 2,
                    "future_effect_far_zero": 3,
                    "future_effect_far_shuffle": 3,
                    "future_effect_h4_8_zero": 0,
                    "future_effect_h4_8_shuffle": 0,
                    "future_effect_h8_16_zero": 1,
                    "future_effect_h8_16_shuffle": 1,
                    "future_effect_h16_32_zero": 2,
                    "future_effect_h16_32_shuffle": 2,
                    "future_effect_h32_48_zero": 3,
                    "future_effect_h32_48_shuffle": 3,
                }[mode]

                def replace_interval(value: Tensor) -> Tensor:
                    output = value.clone()
                    output[:, interval_index] = changed(
                        value[:, interval_index : interval_index + 1]
                    )[:, 0]
                    return output

                semantic_delta = replace_interval(semantic_delta)
                transport_delta = replace_interval(transport_delta)
                covariance_delta = replace_interval(covariance_delta)
                visibility_change = replace_interval(visibility_change)
                persistence_change = replace_interval(persistence_change)
                reliability = replace_interval(reliability)
                uncertainty = replace_interval(uncertainty)
            intervened = replace(
                grounded_field,
                current_reference=current_reference,
                semantic_delta=semantic_delta,
                transport_delta=transport_delta,
                covariance_delta=covariance_delta,
                visibility_change=visibility_change,
                persistence_change=persistence_change,
                reliability=reliability,
                uncertainty=uncertainty,
            )
            intervened.validate()
            state.world_grounded_effect_field = intervened
            component_deltas = (
                intervened.current_reference.detach().float()
                - grounded_field.current_reference.detach().float(),
                intervened.semantic_delta.detach().float()
                - grounded_field.semantic_delta.detach().float(),
                intervened.transport_delta.detach().float()
                - grounded_field.transport_delta.detach().float(),
                intervened.covariance_delta.detach().float()
                - grounded_field.covariance_delta.detach().float(),
                intervened.visibility_change.detach().float()
                - grounded_field.visibility_change.detach().float(),
                intervened.persistence_change.detach().float()
                - grounded_field.persistence_change.detach().float(),
                intervened.reliability.detach().float()
                - grounded_field.reliability.detach().float(),
            )
            self._action_path_eval_metrics[
                "future_effect_boundary_delta_norm"
            ] = float(
                torch.stack(
                    tuple(
                        delta.square().mean().sqrt()
                        for delta in component_deltas
                    )
                )
                .mean()
                .cpu()
            )
            return
        differential_field = getattr(
            state,
            "world_differential_effect_field",
            None,
        )
        if differential_field is not None:
            if mode == "future_effect_reliability_one":
                raise RuntimeError(
                    "reliability-one intervention requires the grounded "
                    "FutureEffectField"
                )
            differential_field.validate(expected_slots=3)
            self._action_path_eval_apply_count += 1
            zero_mode = mode.endswith("_zero")

            def effect_changed(value: Tensor) -> Tensor:
                if zero_mode:
                    return torch.zeros_like(value)
                return value.roll(
                    shifts=(
                        max(int(value.shape[3]) // 2, 1),
                        max(int(value.shape[4]) // 3, 1),
                    ),
                    dims=(3, 4),
                )

            def current_changed(value: Tensor) -> Tensor:
                if zero_mode:
                    return torch.zeros_like(value)
                return value.roll(
                    shifts=(
                        max(int(value.shape[2]) // 2, 1),
                        max(int(value.shape[3]) // 3, 1),
                    ),
                    dims=(2, 3),
                )

            current_reference = differential_field.current_reference
            semantic_delta = differential_field.semantic_delta
            transport_mean = differential_field.transport_mean
            transport_covariance = differential_field.transport_covariance
            persistence = differential_field.persistence
            visibility = differential_field.visibility
            uncertainty = differential_field.uncertainty
            if mode in {"future_effect_zero", "future_effect_spatial_shuffle"}:
                semantic_delta = effect_changed(semantic_delta)
                transport_mean = effect_changed(transport_mean)
                transport_covariance = effect_changed(transport_covariance)
                persistence = effect_changed(persistence)
                visibility = effect_changed(visibility)
                uncertainty = effect_changed(uncertainty)
            elif mode.startswith("future_effect_current_"):
                current_reference = current_changed(current_reference)
            elif mode.startswith("future_effect_semantic_"):
                semantic_delta = effect_changed(semantic_delta)
            elif mode.startswith("future_effect_transport_"):
                transport_mean = effect_changed(transport_mean)
                transport_covariance = effect_changed(transport_covariance)
            elif mode.startswith("future_effect_reliability_"):
                persistence = effect_changed(persistence)
                visibility = effect_changed(visibility)
                uncertainty = effect_changed(uncertainty)
            else:
                slot_index = {
                    "future_effect_near_zero": 0,
                    "future_effect_mid_zero": 1,
                    "future_effect_late_zero": 2,
                    "future_effect_near_shuffle": 0,
                    "future_effect_mid_shuffle": 1,
                    "future_effect_late_shuffle": 2,
                }[mode]

                def replace_slot(value: Tensor) -> Tensor:
                    output = value.clone()
                    output[:, slot_index] = effect_changed(
                        value[:, slot_index : slot_index + 1]
                    )[:, 0]
                    return output

                semantic_delta = replace_slot(semantic_delta)
                transport_mean = replace_slot(transport_mean)
                transport_covariance = replace_slot(transport_covariance)
                persistence = replace_slot(persistence)
                visibility = replace_slot(visibility)
                uncertainty = replace_slot(uncertainty)
            intervened = replace(
                differential_field,
                current_reference=current_reference,
                semantic_delta=semantic_delta,
                transport_mean=transport_mean,
                transport_covariance=transport_covariance,
                persistence=persistence,
                visibility=visibility,
                uncertainty=uncertainty,
            )
            intervened.validate(expected_slots=3)
            state.world_differential_effect_field = intervened
            component_deltas = (
                intervened.current_reference.detach().float()
                - differential_field.current_reference.detach().float(),
                intervened.semantic_delta.detach().float()
                - differential_field.semantic_delta.detach().float(),
                intervened.transport_mean.detach().float()
                - differential_field.transport_mean.detach().float(),
                intervened.transport_covariance.detach().float()
                - differential_field.transport_covariance.detach().float(),
                intervened.persistence.detach().float()
                - differential_field.persistence.detach().float(),
                intervened.visibility.detach().float()
                - differential_field.visibility.detach().float(),
                intervened.uncertainty.detach().float()
                - differential_field.uncertainty.detach().float(),
            )
            self._action_path_eval_metrics[
                "future_effect_boundary_delta_norm"
            ] = float(
                torch.stack(
                    tuple(
                        delta.square().mean().sqrt()
                        for delta in component_deltas
                    )
                ).mean().cpu()
            )
            return
        field = state.world_future_effect_field
        if field is None:
            raise RuntimeError(
                "FutureEffect intervention reached no online effect field"
            )
        field.validate()
        if mode == "future_effect_reliability_one":
            raise RuntimeError(
                "reliability-one intervention requires the grounded "
                "FutureEffectField"
            )
        self._action_path_eval_apply_count += 1

        zero_mode = mode.endswith("_zero")

        def changed(value: Tensor) -> Tensor:
            if zero_mode:
                return torch.zeros_like(value)
            return value.roll(
                shifts=(
                    max(int(value.shape[3]) // 2, 1),
                    max(int(value.shape[4]) // 3, 1),
                ),
                dims=(3, 4),
            )

        supervised = field.current_content is not None
        semantic_delta = field.semantic_delta
        transport_mean = field.transport_mean
        transport_covariance = field.transport_covariance
        persistence = field.persistence
        visibility = field.visibility
        uncertainty = field.uncertainty
        state_innovation = field.state_innovation
        current_content = field.current_content
        successor_content = field.successor_content
        if mode in {"future_effect_zero", "future_effect_spatial_shuffle"}:
            semantic_delta = changed(semantic_delta)
            transport_mean = changed(transport_mean)
            transport_covariance = changed(transport_covariance)
            persistence = changed(persistence)
            visibility = changed(visibility)
            uncertainty = changed(uncertainty)
            state_innovation = (
                changed(state_innovation)
                if state_innovation is not None
                else None
            )
            current_content = (
                changed(current_content)
                if current_content is not None
                else None
            )
            successor_content = (
                changed(successor_content)
                if successor_content is not None
                else None
            )
        elif mode.startswith("future_effect_current_"):
            if not supervised or current_content is None:
                raise RuntimeError(
                    "current-content intervention requires V116 FutureEffect"
                )
            current_content = changed(current_content)
            successor_content = current_content + semantic_delta
        elif mode.startswith("future_effect_semantic_"):
            semantic_delta = changed(semantic_delta)
            if supervised:
                assert current_content is not None
                successor_content = current_content + semantic_delta
        elif mode.startswith("future_effect_transport_"):
            transport_mean = changed(transport_mean)
            transport_covariance = changed(transport_covariance)
        elif mode.startswith("future_effect_reliability_"):
            persistence = changed(persistence)
            visibility = changed(visibility)
            uncertainty = changed(uncertainty)

        effect_type = WindowEffectBank if isinstance(field, WindowEffectBank) else FutureEffectField
        effect_kwargs: dict[str, Tensor | None] = {
            "semantic_delta": semantic_delta,
            "transport_mean": transport_mean,
            "transport_covariance": transport_covariance,
            "persistence": persistence,
            "visibility": visibility,
            "uncertainty": uncertainty,
            "state_innovation": state_innovation,
            "current_content": current_content,
            "successor_content": successor_content,
        }
        if isinstance(field, WindowEffectBank):
            effect_kwargs["slot_valid"] = field.slot_valid
        intervened = effect_type(
            **effect_kwargs,
        )
        intervened.validate()
        state.world_future_effect_field = intervened
        component_deltas = [
            (
                intervened.semantic_delta.detach().float()
                - field.semantic_delta.detach().float()
            ),
            (
                intervened.transport_mean.detach().float()
                - field.transport_mean.detach().float()
            ),
            (
                intervened.transport_covariance.detach().float()
                - field.transport_covariance.detach().float()
            ),
            (
                intervened.persistence.detach().float()
                - field.persistence.detach().float()
            ),
            (
                intervened.visibility.detach().float()
                - field.visibility.detach().float()
            ),
            (
                intervened.uncertainty.detach().float()
                - field.uncertainty.detach().float()
            ),
        ]
        for before, after in (
            (field.state_innovation, intervened.state_innovation),
            (field.current_content, intervened.current_content),
            (field.successor_content, intervened.successor_content),
        ):
            if before is not None and after is not None:
                component_deltas.append(
                    after.detach().float() - before.detach().float()
                )
        self._action_path_eval_metrics[
            "future_effect_boundary_delta_norm"
        ] = float(
            torch.stack(
                tuple(
                    delta.square().mean().sqrt()
                    for delta in component_deltas
                )
            )
            .mean()
            .cpu()
        )

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
        *,
        collect_diagnostics: bool = True,
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
        routed, route_metrics = self.ground_to_world_attnres(
            query,
            values,
            collect_diagnostics=collect_diagnostics,
        )
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
        if not collect_diagnostics:
            return canvas, routed, {}
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
        history_query_context: Tensor | None = None,
        *,
        collect_diagnostics: bool = True,
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
        batch = int(canvas.shape[0])
        context_shape = (
            (batch, horizon, hidden)
            if self.functional_mainline_routing
            else (batch, hidden)
        )
        phase_query_delta = trajectory.new_zeros(*context_shape)
        condition_query_delta = trajectory.new_zeros(*context_shape)
        history_query_delta = trajectory.new_zeros(*context_shape)
        if self.phase_world_query_proj is not None:
            expected_context = (
                (batch, int(cfg.future_anchors), hidden)
                if self.functional_mainline_routing
                else (batch, hidden)
            )
            if (
                phase_context is None
                or tuple(phase_context.shape) != expected_context
            ):
                raise ValueError(
                    "stateless world-to-policy phase context has the wrong schema"
                )
            phase_input = phase_context.to(
                device=query.device, dtype=query.dtype
            )
            if self.functional_mainline_routing:
                phase_input = self._align_anchor_camera_to_horizon(
                    phase_input[:, :, None]
                )[:, :, 0]
            phase_query_delta = float(
                getattr(cfg, "stateless_phase_query_scale", 0.10)
            ) * self.phase_world_query_proj(
                phase_input
            )
            query = query + (
                phase_query_delta[:, :, None]
                if self.functional_mainline_routing
                else phase_query_delta[:, None, None]
            )
            if self.differential_intent_effect_mainline:
                if (
                    condition_query_context is not None
                    or history_query_context is not None
                    or self.condition_world_query_proj is not None
                    or self.history_world_query_proj is not None
                ):
                    raise ValueError(
                        "differential G-to-P bridge accepts only the canonical "
                        "IntentWindowView context"
                    )
            else:
                if (
                    self.condition_world_query_proj is None
                    or condition_query_context is None
                    or tuple(condition_query_context.shape)
                    != expected_context
                ):
                    raise ValueError(
                        "goal world route has the wrong context schema"
                    )
                condition_input = condition_query_context.to(
                    device=query.device, dtype=query.dtype
                )
                if self.functional_mainline_routing:
                    condition_input = self._align_anchor_camera_to_horizon(
                        condition_input[:, :, None]
                    )[:, :, 0]
                condition_query_delta = float(
                    getattr(cfg, "stateless_phase_query_scale", 0.10)
                ) * self.condition_world_query_proj(
                    condition_input
                )
                query = query + (
                    condition_query_delta[:, :, None]
                    if self.functional_mainline_routing
                    else condition_query_delta[:, None, None]
                )
                if self.functional_mainline_routing:
                    if (
                        self.history_world_query_proj is None
                        or history_query_context is None
                        or tuple(history_query_context.shape) != expected_context
                    ):
                        raise ValueError(
                            "history world route has the wrong context schema"
                        )
                    history_input = self._align_anchor_camera_to_horizon(
                        history_query_context[
                            :, :, None
                        ].to(device=query.device, dtype=query.dtype)
                    )[:, :, 0]
                    history_query_delta = float(
                        getattr(cfg, "stateless_phase_query_scale", 0.10)
                    ) * self.history_world_query_proj(history_input)
                    query = query + history_query_delta[:, :, None]
        elif phase_context is not None:
            raise ValueError(
                "phase context was supplied while world phase routing is disabled"
            )
        elif condition_query_context is not None:
            raise ValueError(
                "condition context was supplied while world phase routing is disabled"
            )
        elif history_query_context is not None:
            raise ValueError(
                "history context was supplied while world phase routing is disabled"
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
        routed, route_metrics = self.world_to_policy_attnres(
            query,
            values,
            collect_diagnostics=collect_diagnostics,
        )
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
        if not collect_diagnostics:
            return canvas, routed, {}
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
        metrics["attnres_world_to_policy_history_query_norm"] = (
            history_query_delta.detach().float().norm(dim=-1).mean()
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
            # Snapshot before the indexed write below.  A view into ``values``
            # would be mutated by that write and make every intervention
            # delta metric spuriously report exactly zero.
            original = values[:, source_index].clone()
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
        collect_audit_metrics: bool = True,
        visual_context: FlowDINOEvidencePack | None = None,
        raw_visual: Tensor | None = None,
        future_training_pack: dict[str, Tensor] | None = None,
        allow_future_training_evidence: bool = False,
        v115_static_evidence_cache: V115StaticEvidenceCache | None = None,
        build_v115_static_evidence_cache: bool = False,
    ) -> dict[str, Tensor]:
        cfg = self.config
        # The normal deployment path keeps audit reductions off.  A transient
        # frozen model-path intervention, including the explicit ``none``
        # replay, is itself a diagnostic request and must retain the factual
        # boundary metrics needed to verify what was (or was not) changed.
        collect_audit_metrics = bool(
            collect_audit_metrics
            or self._action_path_eval_intervention is not None
        )
        if future_training_pack is not None and not allow_future_training_evidence:
            raise ValueError(
                "future training evidence requires the explicit teacher-forced "
                "forward boundary"
            )
        if v115_static_evidence_cache is not None and (
            self.training or self.policy_plan_compiler is None
        ):
            raise ValueError(
                "V115 static evidence cache is deployment-only and requires "
                "the V115 policy plan compiler"
            )
        if build_v115_static_evidence_cache and (
            self.training or self.policy_plan_compiler is None
        ):
            raise ValueError(
                "building V115 static evidence is deployment-only and "
                "requires the V115 policy plan compiler"
            )
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
            if v115_static_evidence_cache is not None:
                goal_tokens = v115_static_evidence_cache.goal_tokens
            elif self.object_intent_dynamics_mainline:
                # S consumes the complete T5 sequence below.  These zeros only
                # preserve the historical canvas layout; strict G/P masks and
                # the bottom typed interface make them semantically inert.
                goal_tokens = noisy_physical.new_zeros(
                    int(noisy_physical.shape[0]),
                    int(cfg.goal_token_count),
                    int(cfg.hidden_size),
                )
            else:
                goal_tokens = self.goal_resampler(
                    goal_language_tokens.to(
                        device=noisy_physical.device,
                        dtype=noisy_physical.dtype,
                    ),
                    goal_language_mask.to(
                        device=noisy_physical.device,
                        dtype=torch.bool,
                    ),
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
        history_query_context: Tensor | None = None
        goal_phase_state: (
            GoalPhaseState
            | StatelessIntentState
            | IntentStateBank
            | GroundedIntentState
            | ObjectIntentState
            | None
        ) = None
        phase_metrics: dict[str, Tensor] = {}
        v115_reuse_static = v115_static_evidence_cache is not None
        if self.stateless_horizon_adapter is not None:
            (
                phase_context,
                condition_query_context,
                history_query_context,
                phase_metrics,
            ) = self.stateless_horizon_adapter(
                goal_tokens=canvas[:, slices["task"]],
                history_tokens=canvas[:, slices["executed"]],
                state_tokens=canvas[:, slices["state"]],
                visual_tokens=visual_value_memory,
                collect_diagnostics=collect_audit_metrics,
            )
            (
                phase_context,
                condition_query_context,
                history_query_context,
            ) = self._intervene_horizon_query_contexts(
                phase_context,
                condition_query_context,
                history_query_context,
            )
        elif self.stateless_phase_adapter is not None:
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
        if v115_reuse_static:
            assert v115_static_evidence_cache is not None
            v115_static_evidence_cache.validate(
                canvas=canvas,
                slices=slices,
            )
            goal_phase_state = (
                v115_static_evidence_cache.goal_phase_state
            )
            phase_context = v115_static_evidence_cache.phase_context
            condition_query_context = (
                v115_static_evidence_cache.goal_context
            )
            history_query_context = (
                v115_static_evidence_cache.history_context
            )
            phase_metrics = dict(
                v115_static_evidence_cache.phase_metrics
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
        if self.object_intent_dynamics_mainline:
            # The typed P3 bank carries goal/effect/temporal information.  The
            # generic decoder intent bank receives only current observable
            # state and the last executed action, so it cannot bypass S/W/P.
            owned_state_memory = [
                canvas[:, slices["state"]],
                canvas[:, slices["state_history"]][:, -1:],
            ]
            owned_intent_memory = {
                "state": canvas[:, slices["state"]],
                "executed": canvas[:, slices["executed"]][:, -1:],
            }
            # The decoder already receives the diffusion state through its
            # native action lift and P through the typed bank.  A proposal
            # token here would be a second clean-action shortcut around W/P.
            owned_trajectory_memory = torch.zeros_like(
                canvas[:, slices["proposal"]]
            )
        # The V115 phase replay consumes the clean, pre-attention language and
        # observable history snapshots.  It is intentionally built only after
        # G3 completes, but it must not inherit any later noisy-action carrier
        # or mutable W/P canvas value.
        if self.stateless_goal_phase_machine is not None:
            if goal_tokens is None:
                raise RuntimeError(
                    "V115 goal-phase machine requires resampled T5 goal tokens"
                )
            goal_phase_goal_tokens = goal_tokens
            if goal_condition_keep is not None:
                goal_phase_goal_tokens = (
                    goal_phase_goal_tokens
                    * goal_condition_keep.to(
                        device=goal_phase_goal_tokens.device,
                        dtype=goal_phase_goal_tokens.dtype,
                    )[:, None, None]
                )
        else:
            goal_phase_goal_tokens = canvas[:, slices["task"]]
        grounded_goal_language_tokens: Tensor | None = None
        grounded_goal_language_mask: Tensor | None = None
        if self.explicit_object_top:
            if goal_language_tokens is None or goal_language_mask is None:
                raise RuntimeError(
                    "grounded intent organization requires the complete T5 "
                    "token sequence and mask"
                )
            grounded_goal_language_tokens = goal_language_tokens.to(
                device=noisy_physical.device,
                dtype=noisy_physical.dtype,
            )
            grounded_goal_language_mask = goal_language_mask.to(
                device=noisy_physical.device,
                dtype=torch.bool,
            )
            if goal_condition_keep is not None:
                grounded_goal_language_tokens = (
                    grounded_goal_language_tokens
                    * goal_condition_keep.to(
                        device=grounded_goal_language_tokens.device,
                        dtype=grounded_goal_language_tokens.dtype,
                    )[:, None, None]
                )
        goal_phase_state_history_tokens = torch.cat(
            (
                state_history,
                state[:, None],
            ),
            dim=1,
        )
        goal_phase_action_history_tokens = executed_history
        if action_history_condition_keep is not None:
            goal_phase_action_history_tokens = (
                goal_phase_action_history_tokens
                * action_history_condition_keep.to(
                    device=goal_phase_action_history_tokens.device,
                    dtype=goal_phase_action_history_tokens.dtype,
                )[:, None, None]
            )
        strict_role_visual_path = bool(
            int(getattr(cfg, "flow_jepa_strict_role_visual_path", 0))
        )
        if strict_role_visual_path:
            # Raw visual evidence has one owner: the grounding/world route.
            # Clean task/state/action intent remains available to the decoder.
            owned_intent_memory.pop("visual", None)
        rollout_seed = canvas[:, slices["rollout"]].detach()
        # Snapshot the action query before any P write, but keep ordinary
        # autograd to its owning seed projection.  Value isolation requires a
        # provenance boundary, not a gradient stop: P1/W still cannot enter
        # through this tensor because it is captured before either is written.
        trajectory_seed = canvas[:, slices["trajectory"]]
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        gate_row_roles: list[str] = []
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
        policy_plan_delta_bank: (
            PolicyPlanDeltaBank
            | DifferentialPolicyPlanBank
            | GroundedPolicyPlanDeltaBank
            | ObjectPolicyPlanDeltaBank
            | None
        ) = None
        consequence_plan_state: (
            ConsequenceAwarePlanState
            | GroundedConsequencePlanState
            | ObjectConsequenceState
            | None
        ) = None
        object_facts: ObjectFactSet | None = None
        object_factual_dock: ObjectFactualDock | None = None
        object_intent_state: ObjectIntentState | None = None
        object_coarse_action: CoarseActionIntentState | None = None
        object_plan_recognition: FuturePlanRecognition | None = None
        object_teacher_dynamics: FutureObjectDynamics | None = None
        object_w1_dynamics: FutureObjectDynamics | None = None
        object_w1_hidden: ObjectW1WorkingState | None = None
        object_future_dynamics: FutureObjectDynamics | None = None
        object_training_targets: ObjectTopTrainingTargets | None = None
        object_top_metrics: dict[str, Tensor] = {}
        p2_structured_effect_read: Tensor | None = None
        role_delta_metrics: dict[str, Tensor] = {}
        approved_ground_to_world: Tensor | None = None
        approved_world_to_policy: Tensor | None = None
        protected_policy_detail: Tensor | None = None
        interval_stage_prediction: Tensor | None = None
        interval_stage_input_rollout: Tensor | None = None
        interval_stage_role_delta: Tensor | None = None
        functional_owner_boundary_role_delta: Tensor | None = None
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
        v115_pre_policy_canvas: Tensor | None = None
        v115_midcut_static_canvas: Tensor | None = None

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
        if v115_reuse_static:
            assert v115_static_evidence_cache is not None
            if not progressive_grounding_address:
                raise RuntimeError(
                    "V115 static evidence requires the progressive G path"
                )
            visual_memory = v115_static_evidence_cache.visual_memory
            visual_value_memory = (
                v115_static_evidence_cache.visual_value_memory
            )
            late_raw_detail = v115_static_evidence_cache.late_raw_detail
            progressive_address_state = (
                v115_static_evidence_cache.progressive_address_state
            )
            world_detail_entry_rollout = (
                v115_static_evidence_cache.world_detail_entry_rollout
            )
            raw_refinement_metrics = dict(
                v115_static_evidence_cache.raw_refinement_metrics
            )
            raw_refinement_metrics[
                "flow_jepa_v115_static_evidence_built"
            ] = canvas.new_zeros((), dtype=torch.float32)
            raw_refinement_metrics[
                "flow_jepa_v115_static_evidence_reused"
            ] = canvas.new_ones((), dtype=torch.float32)
            late_detail_metrics = dict(
                v115_static_evidence_cache.late_detail_metrics
            )
            role_delta_metrics = dict(
                v115_static_evidence_cache.role_delta_metrics
            )
            future_address_metrics = dict(
                v115_static_evidence_cache.future_address_metrics
            )
            horizon_boundary_metrics = dict(
                v115_static_evidence_cache.horizon_boundary_metrics
            )
            gate_rows = [
                dict(row) for row in v115_static_evidence_cache.gate_rows
            ]
            gate_row_roles = list(
                v115_static_evidence_cache.gate_row_roles
            )
            content_norm_rows = list(
                v115_static_evidence_cache.content_norm_rows
            )
            time_norm_rows = list(
                v115_static_evidence_cache.time_norm_rows
            )
            interval_stage_prediction = (
                v115_static_evidence_cache.interval_stage_prediction
            )
            object_facts = v115_static_evidence_cache.object_facts
            object_factual_dock = (
                v115_static_evidence_cache.object_factual_dock
            )
            object_intent_state = (
                v115_static_evidence_cache.object_intent_state
            )
            object_coarse_action = (
                v115_static_evidence_cache.object_coarse_action
            )
            object_w1_hidden = None
            object_future_dynamics = (
                v115_static_evidence_cache.object_future_dynamics
            )
            object_top_metrics = dict(
                v115_static_evidence_cache.object_top_metrics or {}
            )
            protected_policy_detail = (
                v115_static_evidence_cache.protected_policy_detail
            )
            v115_pre_policy_canvas = (
                v115_static_evidence_cache.pre_policy_canvas
            )
            v115_midcut_static_canvas = (
                v115_static_evidence_cache.midcut_static_canvas
            )
            trajectory_region = slices["trajectory"]
            canvas = torch.cat(
                (
                    v115_pre_policy_canvas[
                        :, : int(trajectory_region.start)
                    ],
                    canvas[:, trajectory_region],
                    v115_pre_policy_canvas[
                        :, int(trajectory_region.stop) :
                    ],
                ),
                dim=1,
            )
        elif progressive_grounding_address:
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
        if not v115_reuse_static:
            _record_horizon_boundary(
                "seed", canvas[:, slices["rollout"]]
            )
        # Legacy latent decoders need layer contracts at inference.  The
        # object capability does not: its sole consequence ingress is the
        # typed P3 bank, and constructing the old contracts would be an
        # unconsumed per-block tower in both training and five-step deploy.
        final_decoder = str(getattr(cfg, "final_action_decoder", "legacy"))
        legacy_layer_contract_path = not self.object_intent_dynamics_mainline
        force_layer_contracts = (
            legacy_layer_contract_path
            and bool(enable_final_action_decoder)
            and (
                final_decoder == "latent_main_action"
                or (
                    final_decoder
                    in {"latent_cvae_action", "adaptive_recurrent_cvae_action"}
                    and bool(int(getattr(cfg, "latent_cvae_layer_memory", 1)))
                )
                or final_decoder == "hierarchical_mmdit_action"
                or final_decoder == "evidence_latent_mmdit_action"
            )
        )
        effective_layer_contracts = legacy_layer_contract_path and (
            bool(enable_layer_contracts) or force_layer_contracts
        )
        cut = int(cfg.midcut_layer)
        contract_grad_scale = float(getattr(cfg, "layer_contract_grad_scale", 1.0))
        for index, block in enumerate(self.blocks, start=1):
            grounding_boundary = int(
                getattr(cfg, "flow_jepa_grounding_blocks", 0)
            )
            world_boundary = grounding_boundary + int(
                getattr(cfg, "flow_jepa_world_blocks", 0)
            )
            if v115_reuse_static and index <= world_boundary:
                # G1-G3/W1-W2 write only observation/clean-intent regions in
                # V115. Their completed state was restored above. The midcut
                # readout can still depend on the current noisy trajectory, so
                # keep that small readout dynamic if its cut lies here.
                if (
                    index == int(cfg.midcut_layer)
                    and not self.object_intent_dynamics_mainline
                ):
                    if v115_midcut_static_canvas is None:
                        raise RuntimeError(
                            "V115 static cache lost its dynamic midcut boundary"
                        )
                    trajectory_region = slices["trajectory"]
                    midcut_canvas = torch.cat(
                        (
                            v115_midcut_static_canvas[
                                :, : int(trajectory_region.start)
                            ],
                            canvas[:, trajectory_region],
                            v115_midcut_static_canvas[
                                :, int(trajectory_region.stop) :
                            ],
                        ),
                        dim=1,
                    )
                    mid_canvas = self.midcut_norm(midcut_canvas)
                    midcut = self.midcut_heads(mid_canvas, slices)
                continue
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
                    collect_diagnostics=collect_audit_metrics,
                )
                role_delta_metrics.update(bridge_metrics)
            if (
                self.policy_plan_compiler is not None
                and not v115_reuse_static
                and index == world_boundary + 1
            ):
                if v115_pre_policy_canvas is not None:
                    raise RuntimeError(
                        "V115 captured its pre-policy static canvas twice"
                    )
                v115_pre_policy_canvas = canvas
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
                and not v115_reuse_static
                and not self.object_intent_dynamics_mainline
            ):
                rollout_region = slices["rollout"]
                interval_stage_input_rollout = canvas[:, rollout_region]
                if self.functional_mainline_routing:
                    if progressive_address_state is None:
                        raise RuntimeError(
                            "functional interval supervision lost its online W state"
                        )
                    interval_stage_prediction = (
                        self.flow_dino_evidence.progressive_interval_prediction(
                            progressive_address_state
                        )
                    )
                    interval_stage_rollout = interval_stage_input_rollout
                    carrier_rms = (
                        interval_stage_input_rollout.detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                    )
                    online_write_rms = (
                        functional_owner_boundary_role_delta.detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                        if functional_owner_boundary_role_delta is not None
                        else interval_stage_prediction.new_zeros(
                            (), dtype=torch.float32
                        )
                    )
                    interval_stage_metrics = {
                        "flow_jepa_interval_stage_active": (
                            interval_stage_prediction.new_ones(
                                (), dtype=torch.float32
                            )
                        ),
                        "flow_jepa_interval_stage_online_w_candidate": (
                            interval_stage_prediction.new_ones(
                                (), dtype=torch.float32
                            )
                        ),
                        "flow_jepa_interval_stage_written_delta_rms": (
                            online_write_rms
                        ),
                        "flow_jepa_interval_stage_carrier_ratio": (
                            online_write_rms / carrier_rms.clamp_min(1e-8)
                        ),
                    }
                else:
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
                        if collect_audit_metrics:
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
                            ] = interval_stage_write.new_ones(
                                (), dtype=torch.float32
                            )
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
                and not v115_reuse_static
                and not self.grounded_intent_effect_mainline
                and not self.object_intent_dynamics_mainline
            ):
                if self.policy_plan_compiler is not None:
                    # Accumulated W hidden deltas are working state, not the
                    # teacher-aligned effect interface. Current G grounding is
                    # a legal factual P input; V117 forbids the old selected
                    # final-W hidden innovation because WindowEffectBank is the
                    # sole W->P carrier.
                    world_bridge_values = []
                    world_bridge_names = ()
                else:
                    world_bridge_values = list(world_role_deltas)
                    world_bridge_names = tuple(
                        f"w{depth + 1}"
                        for depth in range(len(world_role_deltas))
                    )
                if approved_ground_to_world is not None:
                    world_bridge_values.insert(0, approved_ground_to_world)
                    world_bridge_names = ("grounding_entry",) + world_bridge_names
                if self.functional_mainline_routing:
                    if self.p2_effect_reader is None:
                        if functional_owner_boundary_role_delta is None:
                            raise RuntimeError(
                                "functional final-W-to-P owner candidate was not built"
                            )
                        world_bridge_values.append(
                            functional_owner_boundary_role_delta
                        )
                        world_bridge_names = world_bridge_names + (
                            "functional_owner_boundary",
                        )
                elif self.interval_stage_typed_value:
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
                    condition_query_context=(
                        None
                        if self.differential_intent_effect_mainline
                        else condition_query_context
                    ),
                    history_query_context=(
                        None
                        if self.differential_intent_effect_mainline
                        else history_query_context
                    ),
                    collect_diagnostics=collect_audit_metrics,
                )
                role_delta_metrics.update(bridge_metrics)
            if (
                self.late_raw_detail_reader is not None
                and index == world_boundary + 1
                and not v115_reuse_static
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
                policy_factual_rollout = (
                    world_detail_entry_rollout
                    if self.policy_plan_compiler is not None
                    else current_rollout
                )
                if progressive_grounding_address:
                    if (
                        self.flow_dino_evidence is None
                        or progressive_address_state is None
                    ):
                        raise RuntimeError(
                            "W->P progressive read lost its G3 address state"
                        )
                    if self.policy_plan_compiler is not None:
                        # V115 supersedes the quadratic legacy W target/source
                        # posterior. P1 is addressed from completed G3 facts,
                        # while P2 consumes the supervised FutureEffectField.
                        # Rebuilding the old all-target/all-source tensor here
                        # would be both a dead auxiliary head and one of the
                        # dominant avoidable activation families.
                        future_address_metrics = {
                            "flow_jepa_v115_legacy_w_posterior_skipped": (
                                policy_factual_rollout.new_ones(
                                    (), dtype=torch.float32
                                )
                            )
                        }
                    else:
                        (
                            relevance_logits,
                            progressive_horizon_metrics,
                        ) = self.flow_dino_evidence.score_progressive_horizon_posterior(
                            policy_factual_rollout,
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
                reader_result = self.late_raw_detail_reader(
                    trajectory_before_detail,
                    policy_factual_rollout,
                    late_raw_detail,
                    phase_context=phase_context,
                    condition_query_context=(
                        None
                        if (
                            self.differential_intent_effect_mainline
                            or self.object_intent_dynamics_mainline
                        )
                        else condition_query_context
                    ),
                    history_query_context=(
                        None
                        if (
                            self.differential_intent_effect_mainline
                            or self.object_intent_dynamics_mainline
                        )
                        else history_query_context
                    ),
                    clean_basis_tokens=(
                        self.seed.clean_action_basis_tokens(
                            int(trajectory_before_detail.shape[0]),
                            device=trajectory_before_detail.device,
                            dtype=trajectory_before_detail.dtype,
                        )
                        if self.late_raw_detail_reader.utility_precision_mainline
                        else None
                    ),
                    object_facts=(
                        object_facts
                        if self.object_intent_dynamics_mainline
                        else None
                    ),
                    collect_diagnostics=collect_audit_metrics,
                )
                updated_trajectory = reader_result.trajectory
                reader_metrics = reader_result.metrics
                if self.object_intent_dynamics_mainline:
                    object_factual_dock = reader_result.object_dock
                    if object_factual_dock is None:
                        raise RuntimeError(
                            "object P1 did not return the Object-Chart dock"
                        )
                if collect_role_deltas or self.explicit_object_top:
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
            if (
                self.late_raw_detail_reader is not None
                and index == world_boundary + 1
                and v115_reuse_static
            ):
                assert v115_static_evidence_cache is not None
                trajectory_region = slices["trajectory"]
                cached_ingress = (
                    v115_static_evidence_cache.policy_ingress_delta.to(
                        device=canvas.device,
                        dtype=canvas.dtype,
                    )
                )
                canvas = torch.cat(
                    (
                        canvas[:, : int(trajectory_region.start)],
                        canvas[:, trajectory_region] + cached_ingress,
                        canvas[:, int(trajectory_region.stop) :],
                    ),
                    dim=1,
                )
            if (
                self.policy_plan_compiler is not None
                and build_v115_static_evidence_cache
                and not self.training
                and not v115_reuse_static
                and index == world_boundary + 1
            ):
                required_static = (
                    v115_pre_policy_canvas,
                    goal_tokens,
                    late_raw_detail,
                    progressive_address_state,
                    goal_phase_state,
                    phase_context,
                    protected_policy_detail,
                    world_detail_entry_rollout,
                )
                if any(
                    value is None
                    for value in required_static
                ):
                    raise RuntimeError(
                        "V115 deployment cache lost a static ownership boundary"
                    )
                assert v115_pre_policy_canvas is not None
                assert goal_tokens is not None
                assert late_raw_detail is not None
                assert progressive_address_state is not None
                assert goal_phase_state is not None
                assert phase_context is not None
                assert protected_policy_detail is not None
                assert world_detail_entry_rollout is not None
                if not self.object_intent_dynamics_mainline and (
                    condition_query_context is None
                    or history_query_context is None
                ):
                    raise RuntimeError(
                        "legacy V115 cache lost goal/history query contexts"
                    )
                if (
                    not self.object_intent_dynamics_mainline
                    and interval_stage_prediction is None
                ):
                    raise RuntimeError(
                        "legacy V115 cache lost interval-stage prediction"
                    )
                if self.object_intent_dynamics_mainline and any(
                    value is None
                    for value in (
                        object_facts,
                        object_factual_dock,
                        object_intent_state,
                        object_coarse_action,
                        object_w1_hidden,
                        object_future_dynamics,
                    )
                ):
                    raise RuntimeError(
                        "object deployment cache lost its completed G/S/W state"
                    )
                raw_refinement_metrics[
                    "flow_jepa_v115_static_evidence_built"
                ] = canvas.new_ones((), dtype=torch.float32)
                raw_refinement_metrics[
                    "flow_jepa_v115_static_evidence_reused"
                ] = canvas.new_zeros((), dtype=torch.float32)
                trajectory_region = slices["trajectory"]
                v115_static_evidence_cache = V115StaticEvidenceCache(
                    pre_policy_canvas=v115_pre_policy_canvas,
                    midcut_static_canvas=v115_midcut_static_canvas,
                    visual_memory=visual_memory,
                    visual_value_memory=visual_value_memory,
                    goal_tokens=goal_tokens,
                    world_detail_entry_rollout=(
                        world_detail_entry_rollout
                    ),
                    late_raw_detail=late_raw_detail,
                    progressive_address_state=progressive_address_state,
                    goal_phase_state=goal_phase_state,
                    phase_context=phase_context,
                    goal_context=condition_query_context,
                    history_context=history_query_context,
                    interval_stage_prediction=interval_stage_prediction,
                    policy_ingress_delta=(
                        canvas[:, trajectory_region]
                        - v115_pre_policy_canvas[:, trajectory_region]
                    ),
                    protected_policy_detail=protected_policy_detail,
                    phase_metrics=dict(phase_metrics),
                    raw_refinement_metrics=dict(
                        raw_refinement_metrics
                    ),
                    late_detail_metrics=dict(late_detail_metrics),
                    role_delta_metrics=dict(role_delta_metrics),
                    future_address_metrics=dict(
                        future_address_metrics
                    ),
                    horizon_boundary_metrics=dict(
                        horizon_boundary_metrics
                    ),
                    gate_rows=tuple(dict(row) for row in gate_rows),
                    gate_row_roles=tuple(gate_row_roles),
                    content_norm_rows=tuple(content_norm_rows),
                    time_norm_rows=tuple(time_norm_rows),
                    object_facts=object_facts,
                    object_factual_dock=object_factual_dock,
                    object_intent_state=object_intent_state,
                    object_coarse_action=object_coarse_action,
                    object_future_dynamics=object_future_dynamics,
                    object_top_metrics=dict(object_top_metrics),
                )
                v115_static_evidence_cache.validate(
                    canvas=canvas,
                    slices=slices,
                )
            role = self.block_roles[index - 1]
            is_policy_plan_compiler_layer = bool(
                self.policy_plan_compiler is not None
                and index == len(self.blocks)
            )
            is_v117_p2 = bool(
                self.p2_effect_reader is not None
                and role == "policy"
                and index == world_boundary + 2
            )
            is_grounded_explicit_p2 = bool(
                is_v117_p2 and self.explicit_object_top
            )
            if is_v117_p2:
                if (
                    progressive_address_state is None
                    or goal_phase_state is None
                    or p2_structured_effect_read is not None
                ):
                    raise RuntimeError(
                        "V117 P2 lost its effect bank/intent state or read twice"
                    )
                trajectory_region = slices["trajectory"]
                # P2's action operand is the current diffusion query, not
                # the trajectory after P1 has already written factual
                # evidence into it.  P1 reaches P2 only through the typed
                # ObjectFactualDock below; otherwise the pooled P1 carrier
                # is a second, object-agnostic route around that dock.
                p2_query = trajectory_seed.to(
                    device=canvas.device,
                    dtype=canvas.dtype,
                ).reshape(
                    int(canvas.shape[0]),
                    int(cfg.action_horizon),
                    int(cfg.action_basis_tokens),
                    int(canvas.shape[-1]),
                )
                if self.object_intent_dynamics_mainline:
                    if (
                        not isinstance(goal_phase_state, ObjectIntentState)
                        or not isinstance(
                            self.p2_effect_reader,
                            ObjectFutureEffectReader,
                        )
                        or object_future_dynamics is None
                        or object_factual_dock is None
                    ):
                        raise RuntimeError(
                            "object P2 lost its P1 dock, intent, or completed W dynamics"
                        )
                    raw_p2_effect, effect_metrics = self.p2_effect_reader(
                        p2_query,
                        object_future_dynamics,
                        goal_phase_state,
                        object_factual_dock,
                        collect_diagnostics=collect_audit_metrics,
                    )
                elif self.grounded_intent_effect_mainline:
                    if (
                        not isinstance(goal_phase_state, GroundedIntentState)
                        or not isinstance(
                            self.p2_effect_reader,
                            GroundedFutureEffectReader,
                        )
                    ):
                        raise RuntimeError(
                            "grounded P2 lost its canonical intent state"
                        )
                    effect_field = (
                        progressive_address_state.world_grounded_effect_field
                    )
                    if effect_field is None:
                        raise RuntimeError(
                            "grounded P2 has no completed four-interval effect"
                        )
                    raw_p2_effect, effect_metrics = self.p2_effect_reader(
                        p2_query,
                        effect_field,
                        goal_phase_state,
                        collect_diagnostics=collect_audit_metrics,
                    )
                elif self.differential_intent_effect_mainline:
                    if not isinstance(goal_phase_state, IntentStateBank):
                        raise RuntimeError(
                            "differential P2 lost the canonical intent bank"
                        )
                    effect_field = (
                        progressive_address_state.world_differential_effect_field
                    )
                    if effect_field is None:
                        raise RuntimeError(
                            "differential P2 has no completed effect bank"
                        )
                    raw_p2_effect, effect_metrics = self.p2_effect_reader(
                        p2_query,
                        effect_field,
                        goal_phase_state.window_view,
                        collect_diagnostics=collect_audit_metrics,
                    )
                else:
                    effect_field = (
                        progressive_address_state.world_grounded_effect_field
                        if self.grounded_intent_effect_mainline
                        else (
                            progressive_address_state.world_differential_effect_field
                            if self.differential_intent_effect_mainline
                            else progressive_address_state.world_future_effect_field
                        )
                    )
                    if effect_field is None:
                        raise RuntimeError(
                            "V117 P2 has no completed WindowEffectBank"
                        )
                    window_selector = (
                        goal_phase_state.window_selector
                        if isinstance(goal_phase_state, StatelessIntentState)
                        else None
                    )
                    raw_p2_effect, effect_metrics = self.p2_effect_reader(
                        p2_query,
                        effect_field,
                        window_selector=window_selector,
                        collect_diagnostics=collect_audit_metrics,
                    )
                p2_structured_effect_read, effect_contract = smooth_rms_contract(
                    raw_p2_effect, 0.35
                )
                if self.object_intent_dynamics_mainline:
                    if (
                        not isinstance(
                            self.consequence_plan_organizer,
                            ZeroPreservingObjectConsequence,
                        )
                        or protected_policy_detail is None
                        or len(policy_role_deltas) != 1
                    ):
                        raise RuntimeError(
                            "object P2 requires exactly the completed P1 fact"
                        )
                    p1_fact = protected_policy_detail + policy_role_deltas[0]
                    (
                        consequence_plan_state,
                        consequence_metrics,
                    ) = self.consequence_plan_organizer(
                        factual_base=p1_fact,
                        effect=p2_structured_effect_read,
                        collect_diagnostics=collect_audit_metrics,
                    )
                    p2_write = (
                        consequence_plan_state.effect
                        + consequence_plan_state.interaction
                    )
                    role_delta_metrics.update(consequence_metrics)
                elif self.grounded_intent_effect_mainline:
                    if (
                        not isinstance(
                            self.consequence_plan_organizer,
                            ZeroPreservingConsequenceOrganizer,
                        )
                        or protected_policy_detail is None
                        or len(policy_role_deltas) != 1
                    ):
                        raise RuntimeError(
                            "grounded P2 requires the completed P1 factual "
                            "innovation and zero-preserving organizer"
                        )
                    p1_fact = (
                        protected_policy_detail + policy_role_deltas[0]
                    )
                    (
                        consequence_plan_state,
                        consequence_metrics,
                    ) = self.consequence_plan_organizer(
                        factual_base=p1_fact,
                        effect_read=p2_structured_effect_read,
                    )
                    p2_write = (
                        consequence_plan_state.effect
                        + consequence_plan_state.interaction
                    )
                    role_delta_metrics.update(consequence_metrics)
                else:
                    p2_write = p2_structured_effect_read
                canvas = torch.cat(
                    (
                        canvas[:, : int(trajectory_region.start)],
                        canvas[:, trajectory_region]
                        + p2_write.reshape(
                            int(canvas.shape[0]), -1, int(canvas.shape[-1])
                        ),
                        canvas[:, int(trajectory_region.stop) :],
                    ),
                    dim=1,
                )
                role_delta_metrics.update(effect_metrics)
                if collect_audit_metrics:
                    role_delta_metrics[
                        "flow_jepa_p2_structured_effect_contract_min"
                    ] = effect_contract.detach().float().amin()
                    if self.object_intent_dynamics_mainline:
                        role_delta_metrics["object_p2_contract_min"] = (
                            effect_contract.detach().float().amin()
                        )
                        role_delta_metrics["object_p2_effect_rms"] = (
                            p2_structured_effect_read.detach()
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                        )
                if self.explicit_object_top:
                    policy_role_deltas.append(p2_write)
                    policy_role_depths.append(index - 1)
                    if collect_audit_metrics:
                        role_delta_metrics[
                            "attnres_observed_policy_delta_norm_p2"
                        ] = p2_write.detach().float().norm(dim=-1).mean()
            rollout_before_block = (
                canvas[:, slices["rollout"]]
                if (
                    collect_role_deltas
                    and role in {"grounding", "world"}
                    and not (
                        self.explicit_object_top
                        and role == "world"
                    )
                )
                else None
            )
            trajectory_before_block = (
                canvas[:, slices["trajectory"]]
                if (
                    collect_role_deltas
                    and role == "policy"
                    and not is_policy_plan_compiler_layer
                    and not is_grounded_explicit_p2
                )
                else None
            )
            skip_grounded_owned_block = bool(
                self.explicit_object_top
                and (role == "world" or is_grounded_explicit_p2)
            )
            if is_policy_plan_compiler_layer or skip_grounded_owned_block:
                # P3 has an explicit typed operand contract below.  Do not
                # compute the generic block modulation or visual/canvas reads
                # merely to discard them.
                mod_emb = time_emb
                content_delta = canvas.new_zeros(
                    int(canvas.shape[0]), int(canvas.shape[-1])
                )
                time_row = content_delta
            else:
                # V115 G/W own observable facts and clean-proposal
                # consequences.  Neither tensor is a diffusion state, so
                # changing the ODE time must not silently rewrite them at every
                # deployment step.  P1/P2 and the bottom action tower remain
                # time-conditioned.
                role_time_emb = (
                    torch.zeros_like(time_emb)
                    if self.policy_plan_compiler is not None
                    and role in {"grounding", "world"}
                    else time_emb
                )
                mod_emb, content_delta, time_row = self._mod_embed(
                    canvas,
                    visual_memory,
                    role_time_emb,
                    slices,
                    role=role,
                )
            functional_horizon_context: Tensor | None = None
            rollout_query_context: Tensor | None = None
            if (
                role == "world"
                and self.functional_mainline_routing
                and not self.grounded_intent_effect_mainline
                and not self.object_intent_dynamics_mainline
            ):
                world_depth = index - grounding_boundary - 1
                (
                    functional_horizon_context,
                    typed_condition_metrics,
                ) = self._functional_world_horizon_context(
                    depth=world_depth + 1,
                    phase_context=(
                        None
                        if self.differential_intent_effect_mainline
                        else phase_context
                    ),
                    goal_context=(
                        None
                        if self.differential_intent_effect_mainline
                        else condition_query_context
                    ),
                    history_context=(
                        None
                        if self.differential_intent_effect_mainline
                        else history_query_context
                    ),
                    proposal_context=owned_trajectory_memory,
                    device=mod_emb.device,
                    dtype=mod_emb.dtype,
                    collect_diagnostics=collect_audit_metrics,
                )
                role_delta_metrics.update(typed_condition_metrics)
                if functional_horizon_context is None:
                    raise RuntimeError(
                        "functional W block lost its horizon selector context"
                    )
                # Keep the selector compact.  The world block broadcasts it
                # while its rollout view still has explicit anchor/camera/xy
                # axes, avoiding one persistent full-chart context tensor per
                # W depth.
                rollout_query_context = functional_horizon_context
                if collect_audit_metrics:
                    role_delta_metrics[
                        f"flow_jepa_world_block_horizon_context_norm_w{world_depth + 1}"
                    ] = (
                        functional_horizon_context.detach()
                        .float()
                        .norm(dim=-1)
                        .mean()
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
                if collect_audit_metrics:
                    role_delta_metrics[
                        f"flow_jepa_world_block_query_delta_norm_w{world_depth + 1}"
                    ] = (
                        phase_world_delta.detach().float().norm(dim=-1).mean()
                    )
            # Gate/residual/numerical rows are audit-only.  The action and
            # representation losses require ``collect_diagnostics`` for their
            # tensors, but do not consume these scalar reductions.
            collect_block_metrics = bool(
                (collect_audit_metrics or stop_at_midcut)
                and not is_policy_plan_compiler_layer
                and not skip_grounded_owned_block
            )
            if collect_block_metrics:
                content_norm_rows.append(
                    content_delta.float().norm(dim=-1).mean()
                )
                time_norm_rows.append(
                    time_row.float().norm(dim=-1).mean()
                )
            if is_policy_plan_compiler_layer:
                if (
                    self.policy_plan_compiler is None
                    or protected_policy_detail is None
                    or progressive_address_state is None
                    or goal_phase_state is None
                ):
                    raise RuntimeError(
                        "P3 compiler lost precision, future-effect, or "
                        "goal-phase ownership"
                    )
                if len(policy_role_deltas) != 2:
                    raise RuntimeError(
                        "P3 compiler requires exactly the real P1/P2 "
                        "trajectory innovations"
                    )
                if self.object_intent_dynamics_mainline:
                    if (
                        not isinstance(
                            self.policy_plan_compiler,
                            ObjectPolicyPlanCompiler,
                        )
                        or not isinstance(
                            consequence_plan_state,
                            ObjectConsequenceState,
                        )
                        or not isinstance(goal_phase_state, ObjectIntentState)
                        or object_factual_dock is None
                    ):
                        raise RuntimeError(
                            "object P3 lost consequence or stateless intent"
                        )
                    # Keep the action operand independent of the P1/P2 writes.
                    # P3 receives unresolved K-specific P1 detail through the
                    # typed dock and the P2 innovation through consequence.
                    action_query = trajectory_seed.to(
                        device=canvas.device,
                        dtype=canvas.dtype,
                    ).reshape(
                        int(canvas.shape[0]),
                        int(cfg.action_horizon),
                        int(cfg.action_basis_tokens),
                        int(canvas.shape[-1]),
                    )
                    (
                        policy_plan_delta_bank,
                        plan_metrics,
                    ) = self.policy_plan_compiler(
                        factual_dock=object_factual_dock,
                        consequence=consequence_plan_state,
                        intent=goal_phase_state,
                        action_query=action_query,
                        collect_diagnostics=collect_audit_metrics,
                    )
                elif self.grounded_intent_effect_mainline:
                    if (
                        not isinstance(
                            self.policy_plan_compiler,
                            ConsequenceConditionedPolicyPlanCompiler,
                        )
                        or not isinstance(
                            consequence_plan_state,
                            GroundedConsequencePlanState,
                        )
                        or not isinstance(
                            goal_phase_state,
                            GroundedIntentState,
                        )
                        or p2_structured_effect_read is None
                    ):
                        raise RuntimeError(
                            "grounded P3 lost consequence, intent, or P2 "
                            "effect ownership"
                        )
                    action_query = canvas[:, slices["trajectory"]].reshape(
                        int(canvas.shape[0]),
                        int(cfg.action_horizon),
                        int(cfg.action_basis_tokens),
                        int(canvas.shape[-1]),
                    )
                    (
                        policy_plan_delta_bank,
                        plan_metrics,
                    ) = self.policy_plan_compiler(
                        p1_delta=policy_role_deltas[0],
                        protected_detail=protected_policy_detail,
                        consequence=consequence_plan_state,
                        intent=goal_phase_state,
                        action_query=action_query,
                    )
                elif self.differential_intent_effect_mainline:
                    if (
                        self.consequence_plan_organizer is None
                        or not isinstance(goal_phase_state, IntentStateBank)
                        or p2_structured_effect_read is None
                    ):
                        raise RuntimeError(
                            "differential P3 lost consequence, intent, or "
                            "P2 effect ownership"
                        )
                    (
                        consequence_plan_state,
                        consequence_metrics,
                    ) = self.consequence_plan_organizer(
                        factual_base=protected_policy_detail,
                        effect_read=p2_structured_effect_read,
                        p2_delta=policy_role_deltas[1],
                    )
                    (
                        policy_plan_delta_bank,
                        plan_metrics,
                    ) = self.policy_plan_compiler(
                        p1_delta=policy_role_deltas[0],
                        protected_detail=protected_policy_detail,
                        consequence=consequence_plan_state,
                        intent=goal_phase_state,
                    )
                    plan_metrics.update(consequence_metrics)
                else:
                    (
                        policy_plan_delta_bank,
                        plan_metrics,
                    ) = self.policy_plan_compiler(
                        p1_delta=policy_role_deltas[0],
                        p2_delta=policy_role_deltas[1],
                        protected_detail=protected_policy_detail,
                        progressive_state=progressive_address_state,
                        goal_phase=goal_phase_state,
                        p2_effect=p2_structured_effect_read,
                        collect_diagnostics=collect_audit_metrics,
                    )
                role_delta_metrics.update(plan_metrics)
                gates = {}
            elif skip_grounded_owned_block:
                gates = {}
            else:
                canvas, gates = block(
                    canvas,
                    visual_memory,
                    mod_emb,
                    slices,
                    visual_value_memory=visual_value_memory,
                    rollout_query_context=rollout_query_context,
                    collect_diagnostics=collect_block_metrics,
                )
                if collect_block_metrics:
                    gate_rows.append(gates)
                    gate_row_roles.append(role)
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
                    if collect_audit_metrics:
                        role_delta_metrics[
                            f"attnres_observed_grounding_delta_norm_g{depth_index}"
                        ] = (
                            structured_rollout_delta.detach()
                            .float()
                            .norm(dim=-1)
                            .mean()
                        )
                else:
                    world_role_deltas.append(structured_rollout_delta)
                    depth_index = len(world_role_deltas)
                    if collect_audit_metrics:
                        role_delta_metrics[
                            f"attnres_observed_world_delta_norm_w{depth_index}"
                        ] = (
                            structured_rollout_delta.detach()
                            .float()
                            .norm(dim=-1)
                            .mean()
                        )
                    if collect_audit_metrics:
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
            if self.object_intent_dynamics_mainline and role == "world":
                if (
                    self.object_future_compiler is None
                    or object_facts is None
                    or object_intent_state is None
                    or object_coarse_action is None
                ):
                    raise RuntimeError(
                        "object W lost facts, S, coarse action, or compiler"
                    )
                object_world_depth = index - grounding_boundary
                if object_world_depth == 1:
                    (
                        object_w1_dynamics,
                        object_w1_hidden,
                        object_w1_metrics,
                    ) = self.object_future_compiler.forward_w1(
                        facts=object_facts,
                        intent=object_intent_state,
                        action=object_coarse_action,
                        collect_diagnostics=collect_audit_metrics,
                    )
                    object_top_metrics.update(object_w1_metrics)
                elif object_world_depth == int(cfg.flow_jepa_world_blocks):
                    if object_w1_hidden is None:
                        raise RuntimeError("object W2 has no causal W1 state")
                    (
                        object_future_dynamics,
                        object_w2_metrics,
                    ) = self.object_future_compiler.forward_w2(
                        facts=object_facts,
                        intent=object_intent_state,
                        action=object_coarse_action,
                        w1_state=object_w1_hidden,
                        collect_diagnostics=collect_audit_metrics,
                    )
                    object_top_metrics.update(object_w2_metrics)
                else:
                    raise RuntimeError(
                        f"object W depth must be 1 or 2, got {object_world_depth}"
                    )
            if (
                progressive_grounding_address
                and role == "world"
                and self.flow_dino_evidence is not None
                and self.flow_dino_evidence.pre_value_owner_routing_enabled
                and not self.object_intent_dynamics_mainline
            ):
                if progressive_address_state is None:
                    raise RuntimeError(
                        "world owner transition lost its progressive G3 state"
                )
                owner_depth = index - grounding_boundary
                rollout_region = slices["rollout"]
                owner_base = canvas[:, rollout_region]
                (
                    owner_refined,
                    owner_metrics,
                ) = self.flow_dino_evidence.advance_progressive_world_owner_state(
                    owner_base,
                    progressive_address_state,
                    depth=owner_depth,
                    intervention=self._action_path_eval_intervention,
                    horizon_query_context=functional_horizon_context,
                    intent_window_view=(
                        goal_phase_state.window_view
                        if isinstance(goal_phase_state, IntentStateBank)
                        else None
                    ),
                    grounded_intent_state=(
                        goal_phase_state
                        if isinstance(goal_phase_state, GroundedIntentState)
                        else None
                    ),
                    collect_diagnostics=collect_audit_metrics,
                )
                if self._action_path_eval_intervention in {
                    f"functional_w{owner_depth}_route_zero",
                    f"functional_w{owner_depth}_route_shuffle",
                }:
                    self._action_path_eval_apply_count += 1
                if (
                    self.policy_plan_compiler is not None
                    and owner_depth == int(cfg.flow_jepa_world_blocks)
                ):
                    self._intervene_future_effect_field(
                        progressive_address_state
                    )
                if (
                    self.functional_mainline_routing
                    and owner_depth == int(cfg.flow_jepa_world_blocks)
                ):
                    owner_boundary_write = owner_refined - owner_base
                    functional_owner_boundary_role_delta = (
                        owner_boundary_write.reshape(
                            int(owner_boundary_write.shape[0]),
                            int(cfg.future_anchors),
                            int(cfg.num_cameras),
                            int(cfg.future_grid_size),
                            int(cfg.future_grid_size),
                            int(owner_boundary_write.shape[-1]),
                        ).mean(dim=(3, 4))
                    )
                    if collect_audit_metrics:
                        role_delta_metrics[
                            "attnres_observed_functional_owner_boundary_delta_norm"
                        ] = (
                            functional_owner_boundary_role_delta.detach()
                            .float()
                            .norm(dim=-1)
                            .mean()
                        )
                    if self._action_path_eval_intervention in {
                        "interval_stage_zero",
                        "interval_stage_shuffle",
                    }:
                        self._action_path_eval_apply_count += 1
                canvas = torch.cat(
                    (
                        canvas[:, : int(rollout_region.start)],
                        owner_refined,
                        canvas[:, int(rollout_region.stop) :],
                    ),
                    dim=1,
                )
                raw_refinement_metrics.update(owner_metrics)
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
                if collect_audit_metrics:
                    role_delta_metrics[
                        f"attnres_observed_policy_delta_norm_p{depth_index}"
                    ] = (
                        trajectory_delta.detach().float().norm(dim=-1).mean()
                    )
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
                        collect_diagnostics=collect_audit_metrics,
                    )
                )
                if progressive_address_state.metrics is not None:
                    raw_refinement_metrics.update(
                        progressive_address_state.metrics
                    )
                intervention_name = f"address_g{grounding_stage}"
                stage_address_intervention = (
                    self._action_path_eval_intervention in {
                        f"{intervention_name}_zero",
                        f"{intervention_name}_shuffle",
                    }
                    or (
                        grounding_stage == 3
                        and self._action_path_eval_intervention
                        in {
                            "address_g3_slot_permute",
                            "address_g3_slot_mean",
                        }
                    )
                )
                if stage_address_intervention:
                    self._action_path_eval_apply_count += 1
                    self._action_path_eval_metrics[
                        f"{intervention_name}_applied"
                    ] = 1.0
                    if self._action_path_eval_intervention in {
                        "address_g3_slot_permute",
                        "address_g3_slot_mean",
                    }:
                        slot_delta = (
                            None
                            if progressive_address_state.metrics is None
                            else progressive_address_state.metrics.get(
                                "grounded_g3_slot_intervention_delta_norm"
                            )
                        )
                        public_delta = (
                            None
                            if progressive_address_state.metrics is None
                            else progressive_address_state.metrics.get(
                                "grounded_g3_slot_intervention_public_base_delta_norm"
                            )
                        )
                        if slot_delta is None or public_delta is None:
                            raise RuntimeError(
                                "grounded G3 slot intervention did not report "
                                "its explicit boundary"
                            )
                        self._action_path_eval_metrics[
                            "grounded_g3_slot_intervention_delta_norm"
                        ] = float(slot_delta.detach().float().cpu())
                        self._action_path_eval_metrics[
                            "grounded_g3_slot_intervention_public_base_delta_norm"
                        ] = float(public_delta.detach().float().cpu())
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
                    if self.object_intent_dynamics_mainline:
                        if (
                            self.object_grounder is None
                            or self.object_intent_organizer is None
                            or self.object_coarse_action is None
                            or self.object_plan_recognizer is None
                            or self.object_future_teacher is None
                            or progressive_address_state.grounded_fact_set is None
                            or grounded_goal_language_tokens is None
                            or grounded_goal_language_mask is None
                        ):
                            raise RuntimeError(
                                "object G3/S lost its local facts, language, or owner modules"
                            )
                        object_facts, object_ground_metrics = self.object_grounder(
                            progressive_address_state.grounded_fact_set
                        )
                        object_top_metrics.update(object_ground_metrics)
                        (
                            object_intent_state,
                            object_intent_metrics,
                        ) = self.object_intent_organizer(
                            goal_tokens=grounded_goal_language_tokens,
                            goal_mask=grounded_goal_language_mask,
                            state_history=state_history,
                            state=state,
                            executed_history=goal_phase_action_history_tokens,
                            facts=object_facts,
                            collect_diagnostics=collect_audit_metrics,
                        )
                        object_top_metrics.update(object_intent_metrics)
                        future_action = None
                        future_state = None
                        if future_training_pack is not None:
                            future_action = future_training_pack.get("future_action")
                            future_state = future_training_pack.get("future_state")
                            target_visual = future_training_pack.get("target_visual")
                            future_offsets = future_training_pack.get("future_offsets")
                            if target_visual is not None or future_offsets is not None:
                                if target_visual is None or future_offsets is None:
                                    raise ValueError(
                                        "object teacher requires target_visual and future_offsets together"
                                    )
                                teacher_visual = (
                                    target_visual[:, :, -1]
                                    if target_visual.ndim == 6
                                    else target_visual
                                )
                                future_supports = (
                                    self.flow_dino_evidence.object_teacher_supports(
                                        teacher_visual
                                    )
                                )
                                (
                                    object_teacher_dynamics,
                                    object_teacher_metrics,
                                ) = self.object_future_teacher(
                                    facts=object_facts,
                                    future_supports=future_supports,
                                    future_offsets=future_offsets,
                                )
                                object_top_metrics.update(object_teacher_metrics)
                        if (future_action is None) != (future_state is None):
                            raise ValueError(
                                "future action and state must be supplied together"
                            )
                        if future_action is not None and future_state is not None:
                            object_plan_recognition = self.object_plan_recognizer(
                                future_action=future_action,
                                future_state=future_state,
                                teacher=object_teacher_dynamics,
                            )
                            action_match = F.smooth_l1_loss(
                                object_intent_state.interval_action_innovations.float(),
                                object_plan_recognition.action_targets.float(),
                            )
                            state_match = F.smooth_l1_loss(
                                object_intent_state.interval_state_innovations.float(),
                                object_plan_recognition.state_targets.float(),
                            )

                            def object_match(
                                online: Tensor, target: Tensor
                            ) -> Tensor:
                                row = F.smooth_l1_loss(
                                    online.float(),
                                    target.float(),
                                    reduction="none",
                                ).mean(dim=-1, keepdim=True)
                                weight = (
                                    object_plan_recognition.object_validity
                                    .detach().float()
                                )
                                return (row * weight).sum() / weight.sum().clamp_min(1.0)

                            object_key_match = object_match(
                                object_intent_state.interval_object_keys,
                                object_plan_recognition.object_key_targets,
                            )
                            object_value_match = object_match(
                                object_intent_state.interval_object_values,
                                object_plan_recognition.object_value_targets,
                            )
                            online_intent_loss = 0.25 * (
                                action_match
                                + state_match
                                + object_key_match
                                + object_value_match
                            )
                            plan_recognition_loss = (
                                object_plan_recognition.reconstruction_loss
                            )
                        else:
                            online_intent_loss = summary.new_zeros(())
                            plan_recognition_loss = summary.new_zeros(())
                        object_coarse_action = self.object_coarse_action(
                            object_intent_state,
                            future_action=future_action,
                        )
                        object_training_targets = ObjectTopTrainingTargets(
                            teacher_dynamics=object_teacher_dynamics,
                            plan_recognition=object_plan_recognition,
                            online_intent_loss=online_intent_loss,
                            plan_recognition_loss=plan_recognition_loss,
                            coarse_action_loss=object_coarse_action.loss,
                            object_reconstruction_loss=(
                                object_facts.reconstruction_error
                            ),
                        )
                        object_top_metrics.update(
                            {
                                "object_intent_online_match_loss": online_intent_loss.detach(),
                                "object_intent_action_match_loss": (
                                    action_match.detach()
                                    if object_plan_recognition is not None
                                    else online_intent_loss.detach()
                                ),
                                "object_intent_state_match_loss": (
                                    state_match.detach()
                                    if object_plan_recognition is not None
                                    else online_intent_loss.detach()
                                ),
                                "object_intent_object_key_match_loss": (
                                    object_key_match.detach()
                                    if object_plan_recognition is not None
                                    else online_intent_loss.detach()
                                ),
                                "object_intent_object_value_match_loss": (
                                    object_value_match.detach()
                                    if object_plan_recognition is not None
                                    else online_intent_loss.detach()
                                ),
                                "object_plan_recognition_loss": plan_recognition_loss.detach(),
                                "object_coarse_action_loss": object_coarse_action.loss.detach(),
                                "object_coarse_action_rms": object_coarse_action.action_prediction.detach().float().square().mean().sqrt(),
                            }
                        )
                        if collect_audit_metrics:
                            coarse_innovation = (
                                object_coarse_action.innovations.detach().float()
                            )
                            coarse_prediction = (
                                object_coarse_action.action_prediction.detach().float()
                            )
                            object_top_metrics.update(
                                {
                                    "object_coarse_action_innovation_rms": coarse_innovation.square().mean().sqrt(),
                                    "object_coarse_action_interval_variation": coarse_innovation.std(
                                        dim=1, unbiased=False
                                    ).mean(),
                                    "object_coarse_action_adjacent_cosine": F.cosine_similarity(
                                        coarse_innovation[:, 1:],
                                        coarse_innovation[:, :-1],
                                        dim=-1,
                                        eps=1e-4,
                                    ).mean(),
                                    "object_coarse_action_prediction_interval_variation": coarse_prediction.std(
                                        dim=1, unbiased=False
                                    ).mean(),
                                }
                            )
                            if object_coarse_action.target is not None:
                                coarse_target = (
                                    object_coarse_action.target.detach().float()
                                )
                                object_top_metrics[
                                    "object_coarse_action_target_normalized_error"
                                ] = (
                                    (coarse_prediction - coarse_target)
                                    .square()
                                    .mean()
                                    .sqrt()
                                    / coarse_target.square()
                                    .mean()
                                    .sqrt()
                                    .clamp_min(1e-3)
                                )
                        goal_phase_state = object_intent_state
                        phase_context = object_intent_state.interval_queries
                        # Goal, history and typed objects already have one
                        # canonical composition inside S.interval_queries.
                        # Re-exporting mean-goal and last-history aliases gave
                        # the current-only P1 path two correlated extra inputs.
                        condition_query_context = None
                        history_query_context = None
                    if self.stateless_goal_phase_machine is not None:
                        if any(
                            value is not None
                            for value in (
                                phase_context,
                                condition_query_context,
                                history_query_context,
                                goal_phase_state,
                            )
                        ):
                            raise RuntimeError(
                                "V115 goal-phase state was built more than once"
                            )
                        if self.grounded_intent_effect_mainline:
                            if (
                                not isinstance(
                                    self.stateless_goal_phase_machine,
                                    StatelessIntentOrganizer,
                                )
                                or progressive_address_state.grounded_fact_set
                                is None
                                or grounded_goal_language_tokens is None
                                or grounded_goal_language_mask is None
                            ):
                                raise RuntimeError(
                                    "grounded S lost its complete T5 sequence "
                                    "or completed G3 fact set"
                                )
                            (
                                goal_phase_state,
                                phase_metrics,
                            ) = self.stateless_goal_phase_machine(
                                goal_tokens=grounded_goal_language_tokens,
                                goal_mask=grounded_goal_language_mask,
                                state_history_tokens=(
                                    goal_phase_state_history_tokens
                                ),
                                action_history_tokens=(
                                    goal_phase_action_history_tokens
                                ),
                                facts=(
                                    progressive_address_state.grounded_fact_set
                                ),
                                collect_diagnostics=collect_audit_metrics,
                            )
                            phase_context = goal_phase_state.interval_intents
                            condition_query_context = (
                                goal_phase_state.remaining_goal[:, None]
                                .expand(-1, 4, -1)
                            )
                            history_query_context = (
                                goal_phase_state.achieved_evidence[:, None]
                                .expand(-1, 4, -1)
                            )
                            goal_phase_state = self._intervene_intent_state(
                                goal_phase_state
                            )
                            phase_context = goal_phase_state.interval_intents
                            condition_query_context = (
                                goal_phase_state.remaining_goal[:, None]
                                .expand(-1, 4, -1)
                            )
                            history_query_context = (
                                goal_phase_state.achieved_evidence[:, None]
                                .expand(-1, 4, -1)
                            )
                        else:
                            (
                                goal_phase_state,
                                phase_metrics,
                            ) = self.stateless_goal_phase_machine(
                                goal_tokens=goal_phase_goal_tokens,
                                state_history_tokens=(
                                    goal_phase_state_history_tokens
                                ),
                                history_tokens=(
                                    goal_phase_action_history_tokens
                                ),
                                grounding_tokens=summary,
                                collect_diagnostics=collect_audit_metrics,
                            )
                            if isinstance(
                                goal_phase_state,
                                (StatelessIntentState, IntentStateBank),
                            ):
                                goal_phase_state = self._intervene_intent_state(
                                    goal_phase_state
                                )
                            phase_context = goal_phase_state.interval_selector
                            condition_query_context = (
                                goal_phase_state.goal_context
                            )
                            history_query_context = (
                                goal_phase_state.history_context
                            )
                            (
                                phase_context,
                                condition_query_context,
                                history_query_context,
                            ) = self._intervene_horizon_query_contexts(
                                phase_context,
                                condition_query_context,
                                history_query_context,
                            )
                    if (
                        self.policy_plan_compiler is not None
                        and self.late_raw_detail_reader is not None
                    ):
                        # Snapshot the canonical current-fact chart at the
                        # literal G3 boundary.  The depth-0 owner transition
                        # below is already Goal/Phase-conditioned W-entry
                        # working state and must not be relabelled as P1's
                        # protected observation base.
                        world_detail_entry_rollout = canvas[
                            :, slices["rollout"]
                        ]
                    if (
                        self.flow_dino_evidence.pre_value_owner_routing_enabled
                        and not self.object_intent_dynamics_mainline
                    ):
                        rollout_region = slices["rollout"]
                        (
                            entry_horizon_context,
                            typed_condition_metrics,
                        ) = self._functional_world_horizon_context(
                            depth=0,
                            phase_context=(
                                None
                                if (
                                    self.differential_intent_effect_mainline
                                    or self.grounded_intent_effect_mainline
                                )
                                else phase_context
                            ),
                            goal_context=(
                                None
                                if (
                                    self.differential_intent_effect_mainline
                                    or self.grounded_intent_effect_mainline
                                )
                                else condition_query_context
                            ),
                            history_context=(
                                None
                                if (
                                    self.differential_intent_effect_mainline
                                    or self.grounded_intent_effect_mainline
                                )
                                else history_query_context
                            ),
                            proposal_context=owned_trajectory_memory,
                            device=canvas.device,
                            dtype=canvas.dtype,
                            collect_diagnostics=collect_audit_metrics,
                        )
                        role_delta_metrics.update(
                            typed_condition_metrics
                        )
                        (
                            owner_refined,
                            owner_metrics,
                        ) = self.flow_dino_evidence.advance_progressive_world_owner_state(
                            canvas[:, rollout_region],
                            progressive_address_state,
                            depth=0,
                            intervention=self._action_path_eval_intervention,
                            horizon_query_context=entry_horizon_context,
                            intent_window_view=(
                                goal_phase_state.window_view
                                if isinstance(goal_phase_state, IntentStateBank)
                                else None
                            ),
                            grounded_intent_state=(
                                goal_phase_state
                                if isinstance(
                                    goal_phase_state,
                                    GroundedIntentState,
                                )
                                else None
                            ),
                            collect_diagnostics=collect_audit_metrics,
                        )
                        if self._action_path_eval_intervention in {
                            "functional_w0_route_zero",
                            "functional_w0_route_shuffle",
                        }:
                            self._action_path_eval_apply_count += 1
                        canvas = torch.cat(
                            (
                                canvas[:, : int(rollout_region.start)],
                                owner_refined,
                                canvas[:, int(rollout_region.stop) :],
                            ),
                            dim=1,
                        )
                        raw_refinement_metrics.update(owner_metrics)
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
                and (
                    self.policy_plan_compiler is None
                    or world_detail_entry_rollout is None
                )
            ):
                world_detail_entry_rollout = canvas[
                    :, slices["rollout"]
                ]
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
                (
                    not self.terminal_policy_layer_contracts_only
                    or index
                    > int(cfg.flow_jepa_grounding_blocks)
                    + int(cfg.flow_jepa_world_blocks)
                )
                and not is_policy_plan_compiler_layer
            )
            if (
                effective_layer_contracts
                and contract_layer_active
                and len(self.layer_contract_heads) > 0
            ):
                contract_canvas = _scaled_contract_view(canvas, contract_grad_scale)
                if (
                    self.policy_plan_compiler is not None
                    and role == "policy"
                ):
                    if world_detail_entry_rollout is None:
                        raise RuntimeError(
                            "V115 policy contract lost its protected G3 chart"
                        )
                    rollout_region = slices["rollout"]
                    protected_contract_rollout = _scaled_contract_view(
                        world_detail_entry_rollout,
                        contract_grad_scale,
                    )
                    contract_canvas = torch.cat(
                        (
                            contract_canvas[
                                :, : int(rollout_region.start)
                            ],
                            protected_contract_rollout,
                            contract_canvas[
                                :, int(rollout_region.stop) :
                            ],
                        ),
                        dim=1,
                    )
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
                if (
                    self.policy_plan_compiler is not None
                    and not v115_reuse_static
                    and index <= world_boundary
                    and not self.object_intent_dynamics_mainline
                ):
                    v115_midcut_static_canvas = canvas
                if self.object_intent_dynamics_mainline:
                    if stop_at_midcut:
                        raise RuntimeError(
                            "object-intent capability has no legacy mid-cut "
                            "training boundary"
                        )
                    # This capability owns no mid-cut objective or decoder
                    # input.  Keep the module fields only for old config/state
                    # compatibility; do not execute their dead readout graph.
                    midcut = {}
                else:
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
            midcut = (
                {}
                if self.object_intent_dynamics_mainline
                else self.midcut_heads(self.midcut_norm(canvas), slices)
            )
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
        if self.policy_plan_compiler is not None:
            if world_detail_entry_rollout is None:
                raise RuntimeError(
                    "V115 bottom path lost its protected G3 factual chart"
                )
            # The legacy bottom decoder still accepts a rollout-shaped factual
            # memory.  Under V115 that memory is the protected G3 base, never
            # the accumulated W working carrier. Under V117 W reaches the
            # bottom only through WindowEffectBank -> P2 -> P3; the inherited
            # final-W hidden innovation is deliberately not bridged.
            rollout = world_detail_entry_rollout
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.direct_physical_head.pooled(trajectory)
        typed_policy_delta_bank: PolicyRoleDeltaBank | None = None
        if int(getattr(cfg, "flow_jepa_role_hierarchy", 0)):
            normalized_trajectory_seed = self.final_norm(
                trajectory_seed.to(device=trajectory.device, dtype=trajectory.dtype)
            )
            policy_workspace_tokens = trajectory - normalized_trajectory_seed
            if int(getattr(cfg, "role_attnres_policy_to_mmdit", 0)):
                if self.policy_plan_compiler is not None:
                    if policy_plan_delta_bank is None:
                        raise RuntimeError(
                            "V115 bottom path did not receive the P3 plan bank"
                        )
                    typed_policy_delta_bank = (
                        policy_plan_delta_bank.as_policy_role_bank(
                            source_depth=len(self.blocks) - 1
                        )
                    )
                else:
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
                        f"p{index + 1}"
                        for index in range(len(policy_role_deltas))
                    )
                    approved_depths.extend(policy_role_depths)
                    if not approved_values:
                        raise RuntimeError(
                            "typed policy-to-MMDiT bridge has no "
                            "policy-approved deltas"
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
        if self.object_intent_dynamics_mainline:
            if not isinstance(policy_plan_delta_bank, ObjectPolicyPlanDeltaBank):
                raise RuntimeError(
                    "object bottom dynamics has no completed P3 plan bank"
                )
            context_kv = torch.cat(
                [
                    canvas[:, slices["state"]],
                    canvas[:, slices["state_history"]][:, -1:],
                    canvas[:, slices["executed"]][:, -1:],
                ],
                dim=1,
            )
        else:
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
        controlled_action_tokens = (
            self.final_norm(
                trajectory_seed.to(device=trajectory.device, dtype=trajectory.dtype)
            )
            if self.object_intent_dynamics_mainline
            else trajectory
        )
        if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
            dynamics = self.controlled_dynamics(
                rollout_init.to(device=rollout.device, dtype=rollout.dtype),
                context_kv,
                action_tokens=controlled_action_tokens,
                transition_tokens=rollout,
            )
        else:
            # Preserve the exact learned-base path for historical checkpoints.
            dynamics = self.controlled_dynamics(
                rollout,
                context_kv,
                action_tokens=controlled_action_tokens,
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
        # Object P reaches the bottom exclusively through the typed P3 bank.
        # The capability does not construct historical layer contracts at all:
        # they would be an unsupervised second consequence ingress and pure
        # training/deployment overhead.
        decoder_layer_contracts = (
            [] if self.object_intent_dynamics_mainline else layer_contracts
        )
        if strict_role_visual_path and not self.object_intent_dynamics_mainline:
            policy_blocks = int(getattr(cfg, "flow_jepa_policy_blocks", 0))
            generic_policy_contracts = policy_blocks - int(
                self.policy_plan_compiler is not None
            )
            if (
                generic_policy_contracts < 1
                or len(layer_contracts) < generic_policy_contracts
            ):
                raise RuntimeError(
                    "strict role visual path requires terminal policy layer contracts"
                )
            decoder_layer_contracts = layer_contracts[
                -generic_policy_contracts:
            ]
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
            if decoder_layer_contracts:
                candidate = decoder_layer_contracts[-1].get("event_logits")
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
                execution_terminal_probability=(
                    policy_plan_delta_bank.execution_terminal.probability
                    if (
                        policy_plan_delta_bank is not None
                        and not isinstance(
                            policy_plan_delta_bank,
                            ObjectPolicyPlanDeltaBank,
                        )
                        and policy_plan_delta_bank.execution_terminal is not None
                    )
                    else None
                ),
                execution_terminal_uncertainty=(
                    policy_plan_delta_bank.execution_terminal.uncertainty
                    if (
                        policy_plan_delta_bank is not None
                        and not isinstance(
                            policy_plan_delta_bank,
                            ObjectPolicyPlanDeltaBank,
                        )
                        and policy_plan_delta_bank.execution_terminal is not None
                    )
                    else None
                ),
                rollout_tokens=decoder_rollout,
                transition_memory=transition_memory,
                event_evidence=event_evidence,
                state_memory=owned_state_memory,
                layer_contracts=decoder_layer_contracts,
                intent_memory=owned_intent_memory,
                visual_selector_tokens=final_visual_selector,
                visual_value_tokens=final_visual_values,
                visual_key_bias=final_visual_bias,
                collect_diagnostics=bool(
                    collect_diagnostics
                    or self._action_path_eval_intervention is not None
                ),
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
            if decoder_layer_contracts:
                candidate = decoder_layer_contracts[-1].get("event_logits")
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
                layer_contracts=decoder_layer_contracts,
                collect_diagnostics=bool(
                    collect_diagnostics
                    or self._action_path_eval_intervention is not None
                ),
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
                layer_contracts=decoder_layer_contracts,
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
                layer_contracts=decoder_layer_contracts,
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
                    layer_contracts=decoder_layer_contracts,
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
                if (
                    int(getattr(cfg, "action_flow_residual_layer_memory", 1))
                    and decoder_layer_contracts
                ):
                    last_layer = decoder_layer_contracts[-1]
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
                raw_refinement_metrics,
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
            if v115_static_evidence_cache is not None:
                minimal["_v115_static_evidence_cache"] = (
                    v115_static_evidence_cache  # type: ignore[assignment]
                )
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
        if gate_rows:
            for role_name, row in zip(
                gate_row_roles, gate_rows, strict=True
            ):
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
                    role_written_rows[role_name].append(
                        row["residual_written_rms"]
                    )
        for role_name in ("grounding", "world", "policy"):
            if role_written_rows[role_name]:
                out[
                    f"role_residual_{role_name}_written_rms_max"
                ] = torch.stack(role_written_rows[role_name]).amax()
                out[
                    f"role_residual_{role_name}_written_rms_mean"
                ] = torch.stack(role_written_rows[role_name]).mean()
        if goal_tokens is not None and collect_audit_metrics:
            if self.object_intent_dynamics_mainline:
                if not isinstance(object_intent_state, ObjectIntentState):
                    raise RuntimeError(
                        "object goal diagnostics lost the completed S state"
                    )
                # The bottom task shortcut is intentionally absent.  Audit
                # the protected S boundary instead of reintroducing a task
                # token solely to satisfy historical logging code.
                goal_metric = (
                    object_intent_state.protected_goal_set.detach().float()
                )
            else:
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
        if executed_memory is not None and collect_audit_metrics:
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
            if self.object_intent_dynamics_mainline:
                # ``FutureObjectDynamics`` is the only supervised future
                # representation in this capability.  Do not project it back
                # into the historical slot-reduced JEPA tensor merely to
                # satisfy an ancestry output key: that would recreate the
                # spatial/object averaging bypass the object graph removes.
                future_prediction = None
            elif bool(
                int(
                    getattr(
                        cfg,
                        "flow_jepa_g_aligned_future_effect",
                        0,
                    )
                )
            ):
                if interval_stage_prediction is None:
                    raise RuntimeError(
                        "V115 JEPA prediction lost its supervised "
                        "FutureEffectField"
                    )
                # This is the slot-reduced semantic delta decoded from the
                # exact FutureEffectField consumed by P2 and compiled by P3.
                # P1 remains the protected current-detail owner. Do not train
                # an independent rollout-side future head.
                future_prediction = interval_stage_prediction
            elif online_horizon_address:
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
            if future_prediction is not None:
                out["flow_jepa_future_pred"] = future_prediction
                out.update(future_address_metrics)
                out.update(horizon_boundary_metrics)
                if self.flow_dino_evidence.predictive_change_contract:
                    # Keep the historical key as the supervised prediction tensor
                    # for caller compatibility, while the explicit alias makes its
                    # changed semantics impossible for the loss code to miss.
                    out["flow_jepa_future_delta_pred"] = future_prediction
                    if collect_audit_metrics:
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
                if bool(
                    int(
                        getattr(
                            cfg,
                            "flow_jepa_g_aligned_future_effect",
                            0,
                        )
                    )
                ):
                    if progressive_address_state is None:
                        raise RuntimeError(
                            "V115 output lost the progressive address state"
                        )
                    effect_field = (
                        progressive_address_state.world_grounded_effect_field
                        if self.grounded_intent_effect_mainline
                        else (
                            progressive_address_state.world_differential_effect_field
                            if self.differential_intent_effect_mainline
                            else progressive_address_state.world_future_effect_field
                        )
                    )
                    if effect_field is None:
                        raise RuntimeError(
                            "V115 output has no FutureEffectField"
                        )
                    effect_field.validate()
                    if isinstance(effect_field, GroundedFutureEffectField):
                        out.update(
                            {
                                "grounded_intent_effect_active": (
                                    effect_field.semantic_delta.new_ones(
                                        (), dtype=torch.float32
                                    )
                                ),
                                "flow_jepa_future_effect_semantic_pred_slots": (
                                    effect_field.semantic_delta
                                ),
                                "flow_jepa_future_effect_transport_pred_slots": (
                                    effect_field.transport_delta
                                ),
                                "flow_jepa_future_effect_transport_covariance_pred_slots": (
                                    effect_field.covariance_delta
                                ),
                                "flow_jepa_future_effect_persistence_pred_slots": (
                                    effect_field.persistence_change
                                ),
                                "flow_jepa_future_effect_visibility_pred_slots": (
                                    effect_field.visibility_change
                                ),
                                "flow_jepa_future_effect_uncertainty_pred_slots": (
                                    effect_field.uncertainty
                                ),
                                "flow_jepa_future_effect_reliability_pred_slots": (
                                    effect_field.reliability
                                ),
                                "flow_jepa_future_effect_current_reference": (
                                    effect_field.current_reference
                                ),
                                "flow_jepa_future_effect_successor_pred_slots": (
                                    effect_field.successor_content
                                ),
                                "flow_jepa_future_effect_slot_valid": (
                                    effect_field.validity
                                ),
                            }
                        )
                    else:
                        out.update(
                            {
                                "flow_jepa_future_effect_semantic_pred_slots": (
                                    effect_field.semantic_delta
                                ),
                                "flow_jepa_future_effect_transport_pred_slots": (
                                    effect_field.transport_mean
                                ),
                                "flow_jepa_future_effect_transport_covariance_pred_slots": (
                                    effect_field.transport_covariance
                                ),
                                "flow_jepa_future_effect_persistence_pred_slots": (
                                    effect_field.persistence
                                ),
                                "flow_jepa_future_effect_visibility_pred_slots": (
                                    effect_field.visibility
                                ),
                                "flow_jepa_future_effect_uncertainty_pred_slots": (
                                    effect_field.uncertainty
                                ),
                            }
                        )
                    if isinstance(effect_field, GroundedFutureEffectField):
                        pass
                    elif isinstance(effect_field, DifferentialWindowEffectBank):
                        out["flow_jepa_future_effect_current_reference"] = (
                            effect_field.current_reference
                        )
                        out[
                            "flow_jepa_future_effect_successor_pred_slots"
                        ] = effect_field.successor_content
                        out["flow_jepa_future_effect_slot_valid"] = (
                            effect_field.slot_valid
                        )
                    else:
                        if effect_field.state_innovation is not None:
                            out[
                                "flow_jepa_future_effect_state_innovation_slots"
                            ] = effect_field.state_innovation
                        if effect_field.current_content is not None:
                            out[
                                "flow_jepa_future_effect_current_pred_slots"
                            ] = effect_field.current_content
                        if effect_field.successor_content is not None:
                            out[
                                "flow_jepa_future_effect_successor_pred_slots"
                            ] = effect_field.successor_content
                        if isinstance(effect_field, WindowEffectBank):
                            out["flow_jepa_future_effect_slot_valid"] = (
                                effect_field.slot_valid
                                if effect_field.slot_valid is not None
                                else effect_field.semantic_delta.new_ones(3)
                            )
                    w1_field = (
                        progressive_address_state.world_grounded_effect_w1_field
                        if self.grounded_intent_effect_mainline
                        else (
                            progressive_address_state.world_differential_effect_w1_field
                            if self.differential_intent_effect_mainline
                            else progressive_address_state.world_future_effect_w1_field
                        )
                    )
                    if w1_field is not None:
                        if isinstance(w1_field, GroundedFutureEffectField):
                            w1_field.validate(expected_intervals=2)
                            out.update(
                                {
                                    "flow_jepa_future_effect_w1_successor_pred_slots": (
                                        w1_field.successor_content
                                    ),
                                    "flow_jepa_future_effect_w1_semantic_pred_slots": (
                                        w1_field.semantic_delta
                                    ),
                                    "flow_jepa_future_effect_w1_transport_pred_slots": (
                                        w1_field.transport_delta
                                    ),
                                    "flow_jepa_future_effect_w1_transport_covariance_pred_slots": (
                                        w1_field.covariance_delta
                                    ),
                                    "flow_jepa_future_effect_w1_persistence_pred_slots": (
                                        w1_field.persistence_change
                                    ),
                                    "flow_jepa_future_effect_w1_visibility_pred_slots": (
                                        w1_field.visibility_change
                                    ),
                                    "flow_jepa_future_effect_w1_uncertainty_pred_slots": (
                                        w1_field.uncertainty
                                    ),
                                    "flow_jepa_future_effect_w1_reliability_pred_slots": (
                                        w1_field.reliability
                                    ),
                                    "flow_jepa_future_effect_w1_current_reference": (
                                        w1_field.current_reference
                                    ),
                                    "flow_jepa_future_effect_w1_slot_valid": (
                                        w1_field.validity
                                    ),
                                }
                            )
                        else:
                            w1_field.validate()
                            out.update(
                                {
                                    "flow_jepa_future_effect_w1_successor_pred_slots": (
                                        w1_field.successor_content
                                    ),
                                    "flow_jepa_future_effect_w1_semantic_pred_slots": (
                                        w1_field.semantic_delta
                                    ),
                                    "flow_jepa_future_effect_w1_transport_pred_slots": (
                                        w1_field.transport_mean
                                    ),
                                    "flow_jepa_future_effect_w1_transport_covariance_pred_slots": (
                                        w1_field.transport_covariance
                                    ),
                                    "flow_jepa_future_effect_w1_persistence_pred_slots": (
                                        w1_field.persistence
                                    ),
                                    "flow_jepa_future_effect_w1_visibility_pred_slots": (
                                        w1_field.visibility
                                    ),
                                    "flow_jepa_future_effect_w1_uncertainty_pred_slots": (
                                        w1_field.uncertainty
                                    ),
                                }
                            )
                        if isinstance(w1_field, GroundedFutureEffectField):
                            pass
                        elif isinstance(
                            w1_field,
                            DifferentialWindowEffectBank,
                        ):
                            out[
                                "flow_jepa_future_effect_w1_current_reference"
                            ] = w1_field.current_reference
                            out["flow_jepa_future_effect_w1_slot_valid"] = (
                                w1_field.slot_valid
                            )
                        else:
                            out[
                                "flow_jepa_future_effect_w1_current_pred_slots"
                            ] = w1_field.current_content
                        if isinstance(w1_field, WindowEffectBank):
                            out["flow_jepa_future_effect_w1_slot_valid"] = (
                                w1_field.slot_valid
                                if w1_field.slot_valid is not None
                                else w1_field.semantic_delta.new_ones(3)
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
                    raise RuntimeError(
                        "late-bottleneck canvas unexpectedly materialized "
                        "stage tokens"
                    )
                if collect_audit_metrics:
                    grouped_future = rollout.detach().float().reshape(
                        rollout.shape[0],
                        int(cfg.future_anchors),
                        -1,
                        rollout.shape[-1],
                    ).mean(dim=2)
                    adjacent = F.cosine_similarity(
                        grouped_future[:, 1:],
                        grouped_future[:, :-1],
                        dim=-1,
                    )
                    out["flow_jepa_horizon_adjacent_cosine"] = adjacent.mean()
                    out["flow_jepa_far_horizon_norm"] = (
                        grouped_future[:, -1].norm(dim=-1).mean()
                    )
            else:
                if int(stage_tokens.shape[1]) != 1:
                    raise RuntimeError(
                        "hierarchical Flow-DINO canvas did not preserve one stage token"
                    )
                out["flow_jepa_stage_pred"] = self.flow_dino_evidence.predict_stage(stage_tokens)
                if collect_audit_metrics:
                    stage_f = stage_tokens.detach().float()
                    window_f = rollout.detach().float().mean(
                        dim=1, keepdim=True
                    )
                    out["flow_jepa_stage_token_norm"] = (
                        stage_f.norm(dim=-1).mean()
                    )
                    out["flow_jepa_stage_window_cosine"] = (
                        F.normalize(stage_f, dim=-1)
                        * F.normalize(window_f, dim=-1)
                    ).sum(dim=-1).mean()
                    out["flow_jepa_stage_dynamics_gate"] = gate_mean[
                        "gate_stage"
                    ].detach().float()
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
        if isinstance(goal_phase_state, ObjectIntentState):
            if (
                object_facts is None
                or object_coarse_action is None
                or object_future_dynamics is None
            ):
                raise RuntimeError("object output lost completed G/S/W state")
            out.update(
                {
                    "object_intent_dynamics_active": canvas.new_ones(
                        (), dtype=torch.float32
                    ),
                    "object_intent_schema": canvas.new_tensor(
                        float(OBJECT_INTENT_DYNAMICS_SCHEMA), dtype=torch.float32
                    ),
                    "object_fact_content": object_facts.content,
                    "object_fact_semantic": object_facts.semantic,
                    "object_fact_appearance": object_facts.appearance,
                    "object_fact_geometry": object_facts.geometry,
                    "object_fact_camera_coordinates": object_facts.camera_coordinates,
                    "object_fact_camera_transport_prior": object_facts.camera_transport_prior,
                    "object_fact_camera_validity": object_facts.camera_validity,
                    "object_fact_existence": object_facts.existence,
                    "object_fact_validity": object_facts.validity,
                    "object_fact_to_chart": object_facts.object_to_chart,
                    "object_intent_goal_set": goal_phase_state.protected_goal_set,
                    "object_intent_interval_queries": goal_phase_state.interval_queries,
                    "object_intent_interval_action_queries": goal_phase_state.interval_action_queries,
                    "object_intent_interval_action_innovations": goal_phase_state.interval_action_innovations,
                    "object_intent_interval_state_queries": goal_phase_state.interval_state_queries,
                    "object_intent_interval_state_innovations": goal_phase_state.interval_state_innovations,
                    "object_intent_interval_object_keys": goal_phase_state.interval_object_keys,
                    "object_intent_interval_object_values": goal_phase_state.interval_object_values,
                    "object_intent_temporal_queries": goal_phase_state.temporal_queries,
                    "object_intent_temporal_innovations": goal_phase_state.temporal_innovations,
                    "object_intent_state_change_evidence": goal_phase_state.state_change_evidence,
                    "object_coarse_action_prediction": object_coarse_action.action_prediction,
                    "object_future_current_reference": object_future_dynamics.current_reference,
                    "object_future_successor_prediction": object_future_dynamics.successor_content,
                    "object_future_semantic_prediction": object_future_dynamics.semantic_delta,
                    "object_future_transport_prediction": object_future_dynamics.transport_mean,
                    "object_future_covariance_prediction": object_future_dynamics.transport_covariance,
                    "object_future_visibility_prediction": object_future_dynamics.visibility,
                    "object_future_persistence_prediction": object_future_dynamics.persistence,
                    "object_future_uncertainty_prediction": object_future_dynamics.uncertainty,
                    "object_future_validity_prediction": object_future_dynamics.validity,
                }
            )
            if object_w1_dynamics is not None:
                out["object_w1_semantic_prediction"] = (
                    object_w1_dynamics.semantic_delta
                )
            if object_teacher_dynamics is not None:
                out.update(
                    {
                        "object_future_semantic_target": object_teacher_dynamics.semantic_delta,
                        "object_future_successor_target": object_teacher_dynamics.successor_content,
                        "object_future_transport_target": object_teacher_dynamics.transport_mean,
                        "object_future_covariance_target": object_teacher_dynamics.transport_covariance,
                        "object_future_visibility_target": object_teacher_dynamics.visibility,
                        "object_future_persistence_target": object_teacher_dynamics.persistence,
                        "object_future_uncertainty_target": object_teacher_dynamics.uncertainty,
                        "object_future_validity_target": object_teacher_dynamics.validity,
                    }
                )
            if object_training_targets is not None:
                out.update(
                    {
                        "object_intent_online_match_loss_raw": object_training_targets.online_intent_loss,
                        "object_plan_recognition_loss_raw": object_training_targets.plan_recognition_loss,
                        "object_coarse_action_loss_raw": object_training_targets.coarse_action_loss,
                        "object_reconstruction_loss_raw": object_training_targets.object_reconstruction_loss,
                    }
                )
            for key, value in object_top_metrics.items():
                out[key] = value
        elif isinstance(goal_phase_state, GroundedIntentState):
            out.update(
                {
                    "grounded_intent_protected_goal_tokens": (
                        goal_phase_state.protected_goal_tokens
                    ),
                    "grounded_intent_achieved_evidence": (
                        goal_phase_state.achieved_evidence
                    ),
                    "grounded_intent_remaining_goal": (
                        goal_phase_state.remaining_goal
                    ),
                    "grounded_intent_interval_intents": (
                        goal_phase_state.interval_intents
                    ),
                    "grounded_intent_temporal_control": (
                        goal_phase_state.temporal_control
                    ),
                    "grounded_intent_completion_evidence": (
                        goal_phase_state.completion_evidence
                    ),
                    "grounded_intent_completion_probability": (
                        goal_phase_state.completion_probability
                    ),
                    "grounded_intent_completion_uncertainty": (
                        goal_phase_state.completion_uncertainty
                    ),
                    "grounded_intent_goal_attention": (
                        goal_phase_state.goal_attention
                    ),
                    "grounded_intent_interval_source_attention": (
                        goal_phase_state.interval_source_attention
                    ),
                }
            )
        elif goal_phase_state is not None:
            out.update(
                {
                    "flow_jepa_goal_phase_active_goal": (
                        goal_phase_state.active_goal
                    ),
                    "flow_jepa_goal_phase_next_goal": (
                        goal_phase_state.next_goal
                    ),
                    "flow_jepa_goal_phase_remaining_goal": (
                        goal_phase_state.remaining_goal
                    ),
                    "flow_jepa_goal_phase_belief": (
                        goal_phase_state.phase_belief
                    ),
                    "flow_jepa_goal_phase_uncertainty": (
                        goal_phase_state.phase_uncertainty
                    ),
                    "flow_jepa_goal_phase_interval_selector": (
                        goal_phase_state.interval_selector
                    ),
                }
            )
            if goal_phase_state.terminal_probability is not None:
                out["flow_jepa_goal_phase_terminal_probability"] = (
                    goal_phase_state.terminal_probability
                )
            if isinstance(
                goal_phase_state,
                (StatelessIntentState, IntentStateBank),
            ):
                out["flow_jepa_intent_window_selector"] = (
                    goal_phase_state.window_selector
                )
                out["flow_jepa_intent_temporal_control"] = (
                    goal_phase_state.temporal_control
                )
                # Keep the sample axis for detached runtime comparison with
                # factual frame position. The scalar diagnostic below may
                # overwrite the historical mean-valued key.
                out["flow_jepa_intent_progress_coordinate_per_sample"] = (
                    goal_phase_state.progress_coordinate
                )
                out["flow_jepa_intent_progress_coordinate"] = (
                    goal_phase_state.progress_coordinate
                )
                if isinstance(goal_phase_state, IntentStateBank):
                    out["flow_jepa_intent_state_bank"] = (
                        goal_phase_state.intent_state
                    )
                    out["flow_jepa_intent_window_tokens"] = (
                        goal_phase_state.window_view.tokens
                    )
                    out["flow_jepa_intent_predictive_effect"] = (
                        goal_phase_state.window_view.predictive_effect
                    )
        if policy_plan_delta_bank is not None:
            out["flow_jepa_policy_plan_protected_base"] = (
                policy_plan_delta_bank.protected_base
            )
            out["flow_jepa_policy_plan_precision"] = (
                policy_plan_delta_bank.precision
            )
            out["flow_jepa_policy_plan_temporal"] = (
                policy_plan_delta_bank.temporal
            )
            if isinstance(policy_plan_delta_bank, ObjectPolicyPlanDeltaBank):
                if not isinstance(
                    consequence_plan_state,
                    ObjectConsequenceState,
                ):
                    raise RuntimeError(
                        "object output lost its consequence plan state"
                    )
                out.update(
                    {
                        "object_consequence_factual_base": consequence_plan_state.factual_base,
                        "object_consequence_effect": consequence_plan_state.effect,
                        "object_consequence_interaction": consequence_plan_state.interaction,
                        "object_consequence_protected": consequence_plan_state.protected_consequence,
                        "object_policy_plan_state_change": policy_plan_delta_bank.state_change,
                    }
                )
            elif isinstance(policy_plan_delta_bank, GroundedPolicyPlanDeltaBank):
                if not isinstance(
                    consequence_plan_state,
                    GroundedConsequencePlanState,
                ):
                    raise RuntimeError(
                        "grounded output lost its consequence plan state"
                    )
                out["grounded_consequence_factual_base"] = (
                    consequence_plan_state.factual_base
                )
                out["grounded_consequence_effect"] = (
                    consequence_plan_state.effect
                )
                out["grounded_consequence_interaction"] = (
                    consequence_plan_state.interaction
                )
                out["grounded_consequence_protected"] = (
                    consequence_plan_state.protected_consequence
                )
            elif isinstance(policy_plan_delta_bank, DifferentialPolicyPlanBank):
                if consequence_plan_state is None:
                    raise RuntimeError(
                        "differential output lost its consequence plan state"
                    )
                out["flow_jepa_consequence_factual_base"] = (
                    consequence_plan_state.factual_base
                )
                out["flow_jepa_consequence_effect_base"] = (
                    consequence_plan_state.effect_base
                )
                out["flow_jepa_consequence_organized_delta"] = (
                    consequence_plan_state.organized_delta
                )
            else:
                out["flow_jepa_policy_plan_effect"] = (
                    policy_plan_delta_bank.effect
                )
                if policy_plan_delta_bank.terminal is not None:
                    out["flow_jepa_policy_plan_terminal"] = (
                        policy_plan_delta_bank.terminal
                    )
            if (
                not isinstance(
                    policy_plan_delta_bank,
                    ObjectPolicyPlanDeltaBank,
                )
                and policy_plan_delta_bank.execution_terminal is not None
            ):
                out["flow_jepa_execution_terminal_evidence"] = (
                    policy_plan_delta_bank.execution_terminal.probability
                )
        if collect_audit_metrics:
            for key, value in phase_metrics.items():
                out[key] = value
        else:
            for key in (
                "gate_self",
                "gate_visual",
                "gate_stage",
                "gate_stage_to_window",
                "stage_to_window_update_norm",
                "gate_rollout",
                "gate_ffn",
                "role_residual_contract_enabled",
                "role_residual_contract_max_rms",
                "role_residual_contract_after_gate",
                "role_residual_raw_rms",
                "role_residual_proposed_rms",
                "role_residual_bounded_rms",
                "role_residual_written_rms",
                "role_residual_compression",
                "role_normalization_contract_enabled",
                "role_normalization_denominator_min",
                "role_normalization_gain_max",
                "terminal_normalization_denominator_min",
                "terminal_normalization_gain_max",
                "mod_content_norm",
                "mod_time_norm",
                "mod_content_to_time",
            ):
                out.pop(key, None)
        # The frozen model-path probe must observe the actual soft routing
        # used by the deployed forward. Capture only scalar factual state;
        # this evaluation-only branch never modifies the action graph.
        self._record_action_path_route_metrics(out)
        if v115_static_evidence_cache is not None:
            out["_v115_static_evidence_cache"] = (
                v115_static_evidence_cache  # type: ignore[assignment]
            )
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
        current_state: ProgressiveGroundingAddressState | None = None,
    ) -> dict[str, Tensor]:
        if self.flow_dino_evidence is None:
            raise RuntimeError(
                "interval teacher requested while Flow-DINO is disabled"
            )
        return self.flow_dino_evidence.teacher_interval_targets(
            target_visual,
            current_visual,
            current_state,
        )
