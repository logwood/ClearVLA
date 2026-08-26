"""Typed boundaries for the capability-named mainline 3-2-3 graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..manifest import INTERVALS

INTERVAL_NAMES = ("h4_8", "h8_16", "h16_32", "h32_48")
INTERVAL_BOUNDS = INTERVALS


def _shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class LocalFactSet:
    """Pre-G/G2 local evidence with camera, cell, type and M axes intact.

    ``M`` is a local hypothesis axis, not a persistent object identity.  This
    type replaces the accidental dependency on the historical
    ``grounded_intent_effect.py`` monolith while preserving the exact tensor
    boundary consumed by :class:`DenseObjectGrounder`.
    """

    public_scene_base: Tensor  # [B,C,Y,X,H]
    target_dino_content: Tensor  # detached full-current target [B,C,Y,X,D]
    cell_observed: Tensor  # bool [B,C,Y,X,1]
    content_slots: Tensor  # [B,C,Y,X,M,D]
    semantic_slots: Tensor  # [B,C,Y,X,M,R]
    appearance_slots: Tensor
    geometry_slots: Tensor
    semantic_owner_probs: Tensor  # [B,C,Y,X,M]
    appearance_owner_probs: Tensor
    geometry_owner_probs: Tensor
    slot_coordinates: Tensor  # [B,C,Y,X,M,2]
    slot_support: Tensor  # [B,C,Y,X,M]
    slot_validity: Tensor  # [B,C,Y,X,M,1]
    slot_transport_prior: Tensor | None = None  # [B,C,Y,X,M,2]
    semantic_owner_log_probs: Tensor | None = None  # finite FP32 [B,C,Y,X,M]
    appearance_owner_log_probs: Tensor | None = None
    geometry_owner_log_probs: Tensor | None = None

    @property
    def batch(self) -> int:
        return int(self.content_slots.shape[0])

    @property
    def local_hypotheses(self) -> int:
        return int(self.content_slots.shape[-2])

    @property
    def route_dim(self) -> int:
        return int(self.semantic_slots.shape[-1])

    @property
    def content_dim(self) -> int:
        return int(self.content_slots.shape[-1])

    def validate(self) -> None:
        if self.public_scene_base.ndim != 5:
            raise ValueError("local public scene base must be [B,C,Y,X,H]")
        if self.content_slots.ndim != 6 or self.semantic_slots.ndim != 6:
            raise ValueError("local content/typed slots must be [B,C,Y,X,M,*]")
        prefix = tuple(self.content_slots.shape[:-1])
        if tuple(self.semantic_slots.shape[:-1]) != prefix:
            raise ValueError("local semantic and content slot axes do not align")
        for name in ("appearance_slots", "geometry_slots"):
            value = getattr(self, name)
            if tuple(value.shape) != tuple(self.semantic_slots.shape):
                raise ValueError(f"local {name} is not aligned to semantic slots")
        for name in (
            "semantic_owner_probs",
            "appearance_owner_probs",
            "geometry_owner_probs",
            "slot_support",
        ):
            if tuple(getattr(self, name).shape) != prefix:
                raise ValueError(f"local {name} lost the M hypothesis axis")
        _shape(self.slot_coordinates, (*prefix, 2), "local slot coordinates")
        _shape(self.slot_validity, (*prefix, 1), "local slot validity")
        if self.slot_transport_prior is not None:
            _shape(
                self.slot_transport_prior,
                (*prefix, 2),
                "local slot transport prior",
            )
        owner_logs = (
            self.semantic_owner_log_probs,
            self.appearance_owner_log_probs,
            self.geometry_owner_log_probs,
        )
        if any(value is not None for value in owner_logs) and any(
            value is None for value in owner_logs
        ):
            raise ValueError("local typed owner logs must be supplied together")
        for name in (
            "semantic_owner_log_probs",
            "appearance_owner_log_probs",
            "geometry_owner_log_probs",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            _shape(value, prefix, f"local {name.replace('_', ' ')}")
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise TypeError(f"local {name} must be finite FP32")
        if tuple(self.public_scene_base.shape[:4]) != prefix[:4]:
            raise ValueError("local public scene base lost camera/spatial identity")
        _shape(
            self.target_dino_content,
            (*prefix[:4], self.content_dim),
            "local full-current DINO target",
        )
        _shape(self.cell_observed, (*prefix[:4], 1), "local observed-cell mask")
        if self.cell_observed.dtype != torch.bool:
            raise TypeError("local observed-cell mask must be boolean")


@dataclass(frozen=True)
class DenseFactChart:
    """Lossless current chart before global object binding.

    The local hypothesis axis from the progressive address path is retained in
    ``candidate_*``.  ``dino_content`` is the dense per-cell target used by
    the object reconstruction objective; it is never reconstructed from a
    spatial mean.
    """

    public_scene_base: Tensor  # [B,C,Y,X,H]
    dino_content: Tensor  # [B,C,Y,X,D]
    cell_observed: Tensor  # bool [B,C,Y,X,1]
    candidate_content: Tensor  # [B,C,Y,X,M,D]
    candidate_semantic: Tensor  # [B,C,Y,X,M,R]
    candidate_appearance: Tensor
    candidate_geometry: Tensor
    candidate_coordinates: Tensor  # [B,C,Y,X,M,2]
    candidate_support: Tensor  # [B,C,Y,X,M]
    candidate_validity: Tensor  # [B,C,Y,X,M,1]
    candidate_owner_prior: Tensor  # [B,C,Y,X,M]
    candidate_owner_log_prior: Tensor  # producer-owned finite FP32
    candidate_semantic_prior: Tensor  # typed conditional local priors
    candidate_appearance_prior: Tensor
    candidate_geometry_prior: Tensor
    candidate_semantic_log_prior: Tensor  # producer-owned finite FP32
    candidate_appearance_log_prior: Tensor
    candidate_geometry_log_prior: Tensor
    candidate_transport_prior: Tensor  # [B,C,Y,X,M,2]

    def validate(self) -> None:
        if self.public_scene_base.ndim != 5 or self.dino_content.ndim != 5:
            raise ValueError("dense fact charts must retain [B,C,Y,X,*]")
        if self.candidate_content.ndim != 6:
            raise ValueError("dense candidates must retain [B,C,Y,X,M,*]")
        prefix = tuple(self.candidate_content.shape[:-1])
        for name in ("candidate_semantic", "candidate_appearance", "candidate_geometry"):
            value = getattr(self, name)
            if tuple(value.shape[:-1]) != prefix:
                raise ValueError(f"{name} lost the local candidate axis")
        if tuple(self.candidate_coordinates.shape) != (*prefix, 2):
            raise ValueError("candidate coordinates are misaligned")
        if tuple(self.candidate_support.shape) != prefix:
            raise ValueError("candidate support is misaligned")
        if tuple(self.candidate_validity.shape) != (*prefix, 1):
            raise ValueError("candidate validity is misaligned")
        if tuple(self.candidate_owner_prior.shape) != prefix:
            raise ValueError("candidate owner prior is misaligned")
        if self.candidate_owner_prior.dtype != torch.float32:
            raise TypeError("candidate owner prior must remain FP32")
        if tuple(self.candidate_owner_log_prior.shape) != prefix:
            raise ValueError("candidate owner log prior is misaligned")
        if self.candidate_owner_log_prior.dtype != torch.float32 or not bool(
            torch.isfinite(self.candidate_owner_log_prior).all()
        ):
            raise TypeError("candidate owner log prior must be finite FP32")
        for name in (
            "candidate_semantic_prior",
            "candidate_appearance_prior",
            "candidate_geometry_prior",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != prefix:
                raise ValueError(f"{name} is misaligned")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must remain FP32")
        for name in (
            "candidate_semantic_log_prior",
            "candidate_appearance_log_prior",
            "candidate_geometry_log_prior",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != prefix:
                raise ValueError(f"{name} is misaligned")
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise TypeError(f"{name} must be finite FP32")
        if tuple(self.candidate_transport_prior.shape) != (*prefix, 2):
            raise ValueError("candidate transport prior is misaligned")
        chart_prefix = prefix[:4]
        if tuple(self.public_scene_base.shape[:4]) != chart_prefix:
            raise ValueError("public chart lost camera/spatial identity")
        if tuple(self.dino_content.shape[:4]) != chart_prefix:
            raise ValueError("DINO chart lost camera/spatial identity")
        _shape(self.cell_observed, (*chart_prefix, 1), "dense observed-cell mask")
        if self.cell_observed.dtype != torch.bool:
            raise TypeError("dense observed-cell mask must be boolean")


@dataclass(frozen=True)
class ObjectFactSet:
    """Four global objects with a soft, reversible chart correspondence."""

    dense_chart: DenseFactChart
    content: Tensor  # [B,K,D]
    semantic: Tensor  # [B,K,R]
    appearance: Tensor
    geometry: Tensor
    camera_coordinates: Tensor  # [B,K,C,2]
    camera_transport_prior: Tensor  # [B,K,C,2]
    camera_support: Tensor  # [B,K,C,1]
    camera_validity: Tensor  # [B,K,C,1]
    log_camera_validity: Tensor  # producer-owned finite FP32 [B,K,C,1]
    support: Tensor  # [B,K,1]
    # Read-conditioned object-vs-null confidence.  This is deliberately not
    # the fraction of total chart area allocated to an object.
    existence: Tensor  # [B,K,1]
    # Physical support of the object's own read.  Unlike existence this is a
    # legal-source mask, not a learned confidence or allocation prior.
    validity: Tensor  # [B,K,1]
    log_validity: Tensor  # producer-owned finite FP32 [B,K,1]
    object_to_chart: Tensor  # read posterior [B,K,C,Y,X]
    candidate_assignment: Tensor  # joint local-prior competition mass [B,K,C,Y,X,M]
    # One physical K+null assignment owns object identity.  These three
    # posteriors are bounded verification reads inside that physical support;
    # they may refine which evidence explains an object but cannot create a
    # second object identity or move mass to null.
    semantic_candidate_assignment: Tensor  # [B,K,C,Y,X,M]
    appearance_candidate_assignment: Tensor  # [B,K,C,Y,X,M]
    geometry_candidate_assignment: Tensor  # [B,K,C,Y,X,M]
    null_assignment: Tensor  # joint local-prior null mass [B,C,Y,X,M]
    reconstructed_dino: Tensor  # [B,C,Y,X,D]
    reconstruction_error: Tensor  # scalar, not weighted here

    @property
    def batch(self) -> int:
        return int(self.content.shape[0])

    @property
    def objects(self) -> int:
        return int(self.content.shape[1])

    @property
    def coordinates(self) -> Tensor:
        """V120 object coordinate reduced only over physical camera support."""

        weight = self.camera_validity.float() * self.camera_support.float()
        return (
            self.camera_coordinates.float() * weight
        ).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

    @property
    def transport_prior(self) -> Tensor:
        """V120 object transport prior reduced only over valid cameras."""

        weight = self.camera_validity.float() * self.camera_support.float()
        return (
            self.camera_transport_prior.float() * weight
        ).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

    def validate(self) -> None:
        self.dense_chart.validate()
        if self.content.ndim != 3:
            raise ValueError("object content must be [B,K,D]")
        batch, objects = self.content.shape[:2]
        for name in ("semantic", "appearance", "geometry"):
            value = getattr(self, name)
            if tuple(value.shape[:2]) != (batch, objects) or value.ndim != 3:
                raise ValueError(f"object {name} must be [B,K,*]")
        cameras = int(self.dense_chart.dino_content.shape[1])
        _shape(
            self.camera_coordinates,
            (batch, objects, cameras, 2),
            "object camera coordinates",
        )
        _shape(
            self.camera_transport_prior,
            (batch, objects, cameras, 2),
            "object camera transport prior",
        )
        _shape(
            self.camera_support,
            (batch, objects, cameras, 1),
            "object camera support",
        )
        _shape(
            self.camera_validity,
            (batch, objects, cameras, 1),
            "object camera validity",
        )
        _shape(
            self.log_camera_validity,
            (batch, objects, cameras, 1),
            "object camera log validity",
        )
        if self.camera_validity.dtype != torch.float32:
            raise TypeError("object camera validity must remain FP32")
        if self.log_camera_validity.dtype != torch.float32:
            raise TypeError("object camera log validity must remain FP32")
        _shape(self.support, (batch, objects, 1), "object support")
        _shape(self.existence, (batch, objects, 1), "object existence")
        _shape(self.validity, (batch, objects, 1), "object validity")
        _shape(self.log_validity, (batch, objects, 1), "object log validity")
        if self.validity.dtype != torch.float32:
            raise TypeError("object validity must remain FP32")
        if self.log_validity.dtype != torch.float32:
            raise TypeError("object log validity must remain FP32")
        chart = self.dense_chart.dino_content
        expected_chart = (batch, objects, *chart.shape[1:4])
        _shape(self.object_to_chart, expected_chart, "object-to-chart posterior")
        candidates = self.dense_chart.candidate_content
        _shape(
            self.candidate_assignment,
            (batch, objects, *candidates.shape[1:5]),
            "candidate assignment",
        )
        for name in (
            "semantic_candidate_assignment",
            "appearance_candidate_assignment",
            "geometry_candidate_assignment",
        ):
            _shape(
                getattr(self, name),
                (batch, objects, *candidates.shape[1:5]),
                name.replace("_", " "),
            )
        _shape(self.null_assignment, tuple(candidates.shape[:5]), "null assignment")
        _shape(self.reconstructed_dino, tuple(chart.shape), "reconstructed DINO")
        if self.reconstruction_error.ndim != 0:
            raise ValueError("object reconstruction error must be scalar")

    def permute(self, permutation: Tensor) -> "ObjectFactSet":
        """Permutation-equivariant view used by downstream causal audits."""

        if permutation.ndim != 1 or int(permutation.numel()) != self.objects:
            raise ValueError("object permutation must contain every object once")
        index = permutation.to(device=self.content.device, dtype=torch.long)
        return ObjectFactSet(
            dense_chart=self.dense_chart,
            content=self.content[:, index],
            semantic=self.semantic[:, index],
            appearance=self.appearance[:, index],
            geometry=self.geometry[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            camera_transport_prior=self.camera_transport_prior[:, index],
            camera_support=self.camera_support[:, index],
            camera_validity=self.camera_validity[:, index],
            log_camera_validity=self.log_camera_validity[:, index],
            support=self.support[:, index],
            existence=self.existence[:, index],
            validity=self.validity[:, index],
            log_validity=self.log_validity[:, index],
            object_to_chart=self.object_to_chart[:, index],
            candidate_assignment=self.candidate_assignment[:, index],
            semantic_candidate_assignment=(self.semantic_candidate_assignment[:, index]),
            appearance_candidate_assignment=(self.appearance_candidate_assignment[:, index]),
            geometry_candidate_assignment=(self.geometry_candidate_assignment[:, index]),
            null_assignment=self.null_assignment,
            reconstructed_dino=self.reconstructed_dino,
            reconstruction_error=self.reconstruction_error,
        )


@dataclass(frozen=True)
class FactualPrecisionDock:
    """The exact V120 P1 current-fact boundary.

    P1 owns 24 horizon rows and four factual lanes. Object identity is not a
    P1 axis: global K belongs to W/P2 and must not be recreated by expanding a
    pooled factual value.
    """

    protected_detail: Tensor  # [B,24,4,H]

    def validate(self, *, horizon: int = 24, basis: int | None = None) -> None:
        if self.protected_detail.ndim != 4:
            raise ValueError("factual precision detail must be [B,24,4,H]")
        if int(self.protected_detail.shape[1]) != int(horizon):
            raise ValueError("V120 P1 lost its horizon-query axis")
        if int(self.protected_detail.shape[2]) < 1:
            raise ValueError("V120 P1 requires at least one action-basis lane")
        if basis is not None and int(self.protected_detail.shape[2]) != int(basis):
            raise ValueError("V120 P1 action-basis axis does not match the model")


@dataclass(frozen=True)
class P2QueryDock:
    """The only three-owner dynamic P1 boundary consumed by P2."""

    action_query: Tensor  # [B,24,Q,H]
    factual_base: Tensor  # [B,24,Q,H]
    policy_query_residual: Tensor  # [B,24,Q,H]

    def validate(self) -> None:
        expected = tuple(self.action_query.shape)
        if len(expected) != 4:
            raise ValueError("P2 query dock must be [B,T,Q,H]")
        for name in ("factual_base", "policy_query_residual"):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"P2 query {name} must align with action_query")
            if value.device != self.action_query.device:
                raise ValueError(f"P2 query {name} must share action_query device")

    def combined(self) -> Tensor:
        """Form the exact post-P1 query only at P2's real consumer."""

        self.validate()
        return self.action_query + self.factual_base + self.policy_query_residual


@dataclass(frozen=True)
class CompletedP1PolicyState:
    """Live P1 state with observation fact and policy write kept distinct."""

    factual_base: Tensor  # [B,24,Q,H]
    policy_query_residual: Tensor  # [B,24,Q,H]

    def p2_dock(self, action_query: Tensor) -> P2QueryDock:
        dock = P2QueryDock(
            action_query=action_query,
            factual_base=self.factual_base,
            policy_query_residual=self.policy_query_residual,
        )
        dock.validate()
        return dock

    def validate(
        self,
        *,
        horizon: int = 24,
        basis: int | None = None,
        hidden: int | None = None,
    ) -> None:
        expected = tuple(self.factual_base.shape)
        if len(expected) != 4:
            raise ValueError("completed P1 policy state must be [B,T,Q,H]")
        if int(expected[1]) != int(horizon):
            raise ValueError("completed P1 policy state lost its horizon axis")
        if basis is not None and int(expected[2]) != int(basis):
            raise ValueError("completed P1 policy state lost its action-basis axis")
        if hidden is not None and int(expected[3]) != int(hidden):
            raise ValueError("completed P1 policy state has the wrong hidden width")
        if tuple(self.policy_query_residual.shape) != expected:
            raise ValueError(
                "completed P1 policy residual must align with factual_base"
            )
        if self.policy_query_residual.device != self.factual_base.device:
            raise ValueError(
                "completed P1 policy residual must share factual_base device"
            )


@dataclass(frozen=True)
class ActionIntentDock:
    """S-owned inputs that the clean coarse-action compiler may consume."""

    public_interval_carrier: Tensor  # [B,I,H]
    history_memory: Tensor  # [B,L,H]
    public_object_memory: Tensor  # [B,K,H]

    def validate(self, *, hidden: int) -> None:
        batch = int(self.public_interval_carrier.shape[0])
        _shape(
            self.public_interval_carrier,
            (batch, 4, hidden),
            "action-intent public interval carrier",
        )
        if self.history_memory.ndim != 3 or int(self.history_memory.shape[0]) != batch:
            raise ValueError("action-intent history memory must be [B,L,H]")
        if self.public_object_memory.ndim != 3 or int(
            self.public_object_memory.shape[0]
        ) != batch:
            raise ValueError("action-intent object memory must be [B,K,H]")


@dataclass(frozen=True)
class WorldIntentDock:
    """S-owned relevance boundary consumed by W1/W2 exactly once."""

    protected_goal_memory: Tensor  # [B,G,H]
    public_interval_carrier: Tensor  # [B,I,H]
    typed_common_mass: Tensor  # [B,K,type,1]
    typed_common_value: Tensor  # [B,K,type,R]
    typed_interval_residual_mass: Tensor  # [B,I,K,type,1]
    typed_interval_residual_value: Tensor  # [B,I,K,type,R]

    @property
    def typed_relevance_mass(self) -> Tensor:
        """Reconstruct the unchanged Schema25 interval relevance mass."""

        return self.typed_common_mass[:, None] + self.typed_interval_residual_mass

    @property
    def typed_relevance_value(self) -> Tensor:
        """Reconstruct the unchanged Schema25 interval relevance value."""

        return self.typed_common_value[:, None] + self.typed_interval_residual_value

    def validate(self, *, hidden: int) -> None:
        batch = int(self.public_interval_carrier.shape[0])
        _shape(
            self.public_interval_carrier,
            (batch, 4, hidden),
            "world-intent public interval carrier",
        )
        _shape(
            self.protected_goal_memory,
            (batch, 4, hidden),
            "world-intent protected goal memory",
        )
        if self.typed_common_mass.ndim != 4:
            raise ValueError("typed common mass must be [B,K,type,1]")
        if int(self.typed_common_mass.shape[0]) != batch:
            raise ValueError("typed common mass batch does not align")
        if tuple(self.typed_common_mass.shape[-2:]) != (3, 1):
            raise ValueError("typed common mass lost semantic/appearance/geometry")
        objects = int(self.typed_common_mass.shape[1])
        _shape(
            self.typed_interval_residual_mass,
            (batch, 4, objects, 3, 1),
            "typed interval residual mass",
        )
        if self.typed_common_value.ndim != 4 or tuple(
            self.typed_common_value.shape[:3]
        ) != (batch, objects, 3):
            raise ValueError("typed common value lost object/type identity")
        route = int(self.typed_common_value.shape[-1])
        _shape(
            self.typed_interval_residual_value,
            (batch, 4, objects, 3, route),
            "typed interval residual value",
        )


@dataclass(frozen=True)
class FactualIntentDock:
    """Read-only S context for the unchanged V120 P1 factual reader."""

    phase_context: Tensor  # [B,I,H]
    condition_query_context: Tensor  # [B,I,H]
    history_query_context: Tensor  # [B,I,H]

    def validate(self, *, hidden: int) -> None:
        batch = int(self.phase_context.shape[0])
        expected = (batch, 4, hidden)
        _shape(self.phase_context, expected, "factual-intent phase context")
        _shape(
            self.condition_query_context,
            expected,
            "factual-intent condition context",
        )
        _shape(self.history_query_context, expected, "factual-intent history context")


@dataclass(frozen=True)
class PolicyIntentDock:
    """Read-only reduced and typed S context for P2/P3."""

    interval_key: Tensor  # [B,I,H]
    temporal_control: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]
    typed_common_value: Tensor  # [B,K,3,R]
    typed_interval_residual_value: Tensor  # [B,I,K,3,R]

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.interval_key.shape[0])
        _shape(self.interval_key, (batch, 4, hidden), "policy-intent interval key")
        _shape(
            self.temporal_control,
            (batch, horizon, hidden),
            "policy-intent temporal control",
        )
        _shape(
            self.state_change_evidence,
            (batch, hidden),
            "policy-intent state-change evidence",
        )
        if self.typed_common_value.ndim != 4 or tuple(
            self.typed_common_value.shape[:1]
        ) != (batch,):
            raise ValueError("policy-intent typed common value must be [B,K,3,R]")
        objects = int(self.typed_common_value.shape[1])
        if int(self.typed_common_value.shape[2]) != 3:
            raise ValueError("policy-intent typed common value lost type identity")
        route = int(self.typed_common_value.shape[3])
        _shape(
            self.typed_interval_residual_value,
            (batch, 4, objects, 3, route),
            "policy-intent typed interval residual value",
        )


@dataclass(frozen=True)
class ObjectIntentState:
    """Stateless intent bundle with public and object/type-owned outputs."""

    protected_goal_set: Tensor  # [B,4,H]
    history_tokens: Tensor  # [B,L,H]
    object_tokens: Tensor  # [B,K,H]
    public_interval_carrier: Tensor  # [B,4,H]
    policy_interval_context: Tensor  # [B,4,H]
    temporal_queries: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]
    typed_common_mass: Tensor  # [B,K,3,1]
    typed_common_value: Tensor  # [B,K,3,R]
    typed_interval_residual_mass: Tensor  # [B,4,K,3,1]
    typed_interval_residual_value: Tensor  # [B,4,K,3,R]
    typed_policy_components: Tensor  # [B,4,3,H]
    goal_attention: Tensor  # [B,4,Lg]
    interval_goal_attention: Tensor  # [B,4,4]
    interval_history_attention: Tensor  # [B,4,L]
    interval_object_attention: Tensor  # [B,4,K]

    @property
    def interval_queries(self) -> Tensor:
        """Compatibility name for the S-owned consumer-specific context."""

        return self.policy_interval_context

    @property
    def typed_relevance_mass(self) -> Tensor:
        """Compatibility view of the unchanged Schema25 selector mass."""

        return self.typed_common_mass[:, None] + self.typed_interval_residual_mass

    @property
    def typed_relevance_value(self) -> Tensor:
        """Compatibility view of the unchanged Schema25 selected value."""

        return self.typed_common_value[:, None] + self.typed_interval_residual_value

    @property
    def interval_semantic_attention(self) -> Tensor:
        return self.typed_relevance_mass[..., 0, 0]

    @property
    def interval_appearance_attention(self) -> Tensor:
        return self.typed_relevance_mass[..., 1, 0]

    @property
    def interval_geometry_attention(self) -> Tensor:
        return self.typed_relevance_mass[..., 2, 0]

    def action_dock(self) -> ActionIntentDock:
        return ActionIntentDock(
            public_interval_carrier=self.public_interval_carrier,
            history_memory=self.history_tokens,
            public_object_memory=self.object_tokens,
        )

    def world_dock(self) -> WorldIntentDock:
        return WorldIntentDock(
            protected_goal_memory=self.protected_goal_set,
            public_interval_carrier=self.public_interval_carrier,
            typed_common_mass=self.typed_common_mass,
            typed_common_value=self.typed_common_value,
            typed_interval_residual_mass=self.typed_interval_residual_mass,
            typed_interval_residual_value=self.typed_interval_residual_value,
        )

    def factual_dock(self) -> FactualIntentDock:
        batch = int(self.policy_interval_context.shape[0])
        return FactualIntentDock(
            phase_context=self.policy_interval_context,
            condition_query_context=self.protected_goal_set.mean(dim=1)[:, None].expand(
                -1, 4, -1
            ),
            history_query_context=self.history_tokens[:, -1:, :].expand(
                batch, 4, -1
            ),
        )

    def policy_dock(self) -> PolicyIntentDock:
        return PolicyIntentDock(
            interval_key=self.policy_interval_context,
            temporal_control=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            typed_common_value=self.typed_common_value,
            typed_interval_residual_value=self.typed_interval_residual_value,
        )

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.public_interval_carrier.shape[0])
        _shape(self.protected_goal_set, (batch, 4, hidden), "protected goal set")
        _shape(
            self.public_interval_carrier,
            (batch, 4, hidden),
            "public interval carrier",
        )
        _shape(
            self.policy_interval_context,
            (batch, 4, hidden),
            "policy interval context",
        )
        _shape(self.temporal_queries, (batch, horizon, hidden), "temporal queries")
        _shape(self.state_change_evidence, (batch, hidden), "state-change evidence")
        if self.history_tokens.ndim != 3 or int(self.history_tokens.shape[0]) != batch:
            raise ValueError("intent history tokens must be [B,L,H]")
        if self.object_tokens.ndim != 3 or int(self.object_tokens.shape[0]) != batch:
            raise ValueError("object intent public tokens must be [B,K,H]")
        objects = int(self.object_tokens.shape[1])
        _shape(
            self.typed_common_mass,
            (batch, objects, 3, 1),
            "typed common mass",
        )
        _shape(
            self.typed_interval_residual_mass,
            (batch, 4, objects, 3, 1),
            "typed interval residual mass",
        )
        if self.typed_common_value.ndim != 4 or tuple(
            self.typed_common_value.shape[:3]
        ) != (batch, objects, 3):
            raise ValueError("typed common value lost object/type identity")
        route = int(self.typed_common_value.shape[-1])
        _shape(
            self.typed_interval_residual_value,
            (batch, 4, objects, 3, route),
            "typed interval residual value",
        )
        _shape(
            self.typed_policy_components,
            (batch, 4, 3, hidden),
            "typed policy components",
        )
        self.action_dock().validate(hidden=hidden)
        self.world_dock().validate(hidden=hidden)
        self.factual_dock().validate(hidden=hidden)
        self.policy_dock().validate(horizon=horizon, hidden=hidden)

    def permute(self, permutation: Tensor) -> "ObjectIntentState":
        """Return the same intent bundle under a relabeling of global K slots."""

        objects = int(self.object_tokens.shape[1])
        if permutation.ndim != 1 or int(permutation.numel()) != objects:
            raise ValueError("intent permutation must cover every K slot")
        index = permutation.to(
            device=self.object_tokens.device,
            dtype=torch.long,
        )
        return ObjectIntentState(
            protected_goal_set=self.protected_goal_set,
            history_tokens=self.history_tokens,
            object_tokens=self.object_tokens[:, index],
            public_interval_carrier=self.public_interval_carrier,
            policy_interval_context=self.policy_interval_context,
            temporal_queries=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            typed_common_mass=self.typed_common_mass[:, index],
            typed_common_value=self.typed_common_value[:, index],
            typed_interval_residual_mass=self.typed_interval_residual_mass[
                :, :, index
            ],
            typed_interval_residual_value=self.typed_interval_residual_value[
                :, :, index
            ],
            typed_policy_components=self.typed_policy_components,
            goal_attention=self.goal_attention,
            interval_goal_attention=self.interval_goal_attention,
            interval_history_attention=self.interval_history_attention,
            interval_object_attention=self.interval_object_attention[:, :, index],
        )


# The descriptive name is exported without breaking existing import sites that
# still use ObjectIntentState for the same typed runtime container.
StatelessIntentBundle = ObjectIntentState


@dataclass(frozen=True)
class FuturePlanRecognition:
    """Recovered V120 whole-segment target for online interval intent."""

    interval_targets: Tensor  # [B,4,H]
    action_summary: Tensor  # [B,4,A]
    state_summary: Tensor  # [B,4,S]
    effect_summary: Tensor  # [B,4,D]
    reconstruction_loss: Tensor

    def validate(self, *, hidden: int) -> None:
        if self.interval_targets.ndim != 3:
            raise ValueError("recognizer interval target must be [B,4,H]")
        batch = int(self.interval_targets.shape[0])
        _shape(self.interval_targets, (batch, 4, hidden), "interval targets")
        if self.action_summary.ndim != 3 or self.state_summary.ndim != 3:
            raise ValueError("recognizer action/state summaries lost interval axis")
        if self.effect_summary.ndim != 3:
            raise ValueError("recognizer effect summary lost interval axis")
        if self.reconstruction_loss.ndim != 0:
            raise ValueError("recognizer reconstruction loss must be scalar")


@dataclass(frozen=True)
class CoarseActionIntentState:
    tokens: Tensor  # [B,4,H]
    action_prediction: Tensor  # [B,4,A]
    target: Tensor | None
    loss: Tensor


@dataclass(frozen=True)
class HistoryActionProposalState:
    """Auxiliary causal prediction reconstructed from executed-action history.

    The recovered V120 object-policy path supervises this prediction but does
    not feed its tokens into G/S/W/P, controlled transition or the bottom.
    Keeping that distinction explicit prevents the schema-20 proposal alias
    from silently returning through a typed container.
    """

    tokens: Tensor  # [B,T,H]
    action_prediction: Tensor  # [B,T,A]
    history_tokens: Tensor  # [B,summary+recent,H]

    def validate(
        self,
        *,
        horizon: int,
        hidden: int,
        action_dim: int,
        history_tokens: int,
    ) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("history proposal tokens must be [B,T,H]")
        batch = int(self.tokens.shape[0])
        _shape(self.tokens, (batch, horizon, hidden), "history proposal tokens")
        _shape(
            self.action_prediction,
            (batch, horizon, action_dim),
            "history proposal action",
        )
        _shape(
            self.history_tokens,
            (batch, history_tokens, hidden),
            "encoded action history",
        )


@dataclass(frozen=True)
class ControlledTransitionState:
    """Action-centred low-rank transition evidence consumed read-only below P.

    ``selector`` carries the typed W transition state.  ``value`` is the
    real-action coefficient response minus the neutral-context response, so a
    learned action-independent base cannot become a second world residual.
    """

    selector: Tensor  # [B,I*C*8*8,H] -- 512 V120 spatial transition rows
    value: Tensor  # [B,I*C*8*8,H]
    action_coefficients: Tensor  # [B,I*C*8*8,R]
    neutral_coefficients: Tensor  # [B,I*C*8*8,R]

    def validate(self, *, hidden: int) -> None:
        if self.selector.ndim != 3 or tuple(self.selector.shape) != tuple(
            self.value.shape
        ):
            raise ValueError("controlled transition selector/value must align")
        if int(self.selector.shape[-1]) != int(hidden):
            raise ValueError("controlled transition hidden width is invalid")
        expected = (*self.selector.shape[:-1], int(self.action_coefficients.shape[-1]))
        _shape(
            self.action_coefficients,
            expected,
            "controlled action coefficients",
        )
        _shape(
            self.neutral_coefficients,
            expected,
            "controlled neutral coefficients",
        )


@dataclass(frozen=True)
class ControlledTransitionSource:
    """ODE-invariant protected G3 chart for the dynamic transition."""

    selector: Tensor  # [B,4*C*8*8,H]

    def validate(self, *, hidden: int, rows: int = 512) -> None:
        if self.selector.ndim != 3:
            raise ValueError("controlled transition source must be [B,N,H]")
        if tuple(self.selector.shape[1:]) != (int(rows), int(hidden)):
            raise ValueError(
                "controlled transition source must retain every G3 spatial row"
            )


@dataclass(frozen=True)
class FutureObjectDynamics:
    """The only W value object visible to P2."""

    current_reference: Tensor  # [B,K,D]
    successor_content: Tensor  # [B,I,K,D]
    semantic_delta: Tensor  # [B,I,K,D]
    transport_mean: Tensor  # camera-specific [B,I,K,C,2]
    transport_covariance: Tensor  # FP32 PSD xx/xy/yy [B,I,K,C,3]
    chart_availability: Tensor  # current observable object support [B,K,1]
    log_chart_availability: Tensor  # producer-owned finite FP32 [B,K,1]
    camera_coordinates: Tensor  # current real camera charts [B,K,C,2]
    camera_chart_availability: Tensor  # current observable support [B,K,C,1]
    log_camera_chart_availability: Tensor  # producer-owned finite FP32 [B,K,C,1]

    @property
    def intervals(self) -> int:
        return int(self.semantic_delta.shape[1])

    @staticmethod
    def _common(value: Tensor) -> Tensor:
        """Return the one protected effect shared across future intervals."""

        if value.ndim < 3:
            raise ValueError("future effect must retain an interval axis")
        return value.float().mean(dim=1)

    @classmethod
    def _interval_innovation(cls, value: Tensor) -> Tensor:
        """Return the exact interval coordinate around the protected common."""

        common = cls._common(value)
        return value.float() - common[:, None]

    @property
    def semantic_common(self) -> Tensor:
        return self._common(self.semantic_delta)

    @property
    def semantic_interval_innovation(self) -> Tensor:
        return self._interval_innovation(self.semantic_delta)

    @property
    def transport_common(self) -> Tensor:
        return self._common(self.transport_mean)

    @property
    def transport_interval_innovation(self) -> Tensor:
        return self._interval_innovation(self.transport_mean)

    def validate(self, *, expected_intervals: int = 4) -> None:
        if self.current_reference.ndim != 3 or self.semantic_delta.ndim != 4:
            raise ValueError("future dynamics lost object or interval identity")
        batch, intervals, objects, width = self.semantic_delta.shape
        if intervals != int(expected_intervals):
            raise ValueError(
                f"future dynamics requires {expected_intervals} intervals, got {intervals}"
            )
        _shape(self.current_reference, (batch, objects, width), "current object reference")
        _shape(self.successor_content, (batch, intervals, objects, width), "successor content")
        if self.camera_coordinates.ndim != 4:
            raise ValueError("future camera coordinates must be [B,K,C,2]")
        cameras = int(self.camera_coordinates.shape[2])
        _shape(
            self.camera_coordinates,
            (batch, objects, cameras, 2),
            "future camera coordinates",
        )
        _shape(
            self.transport_mean,
            (batch, intervals, objects, cameras, 2),
            "transport mean",
        )
        _shape(
            self.transport_covariance,
            (batch, intervals, objects, cameras, 3),
            "transport covariance",
        )
        if self.transport_covariance.dtype != torch.float32:
            raise TypeError("future transport covariance must remain FP32")
        _shape(
            self.chart_availability,
            (batch, objects, 1),
            "future chart availability",
        )
        if self.chart_availability.dtype != torch.float32:
            raise TypeError("future chart availability must remain FP32")
        _shape(
            self.log_chart_availability,
            (batch, objects, 1),
            "future chart log availability",
        )
        if self.log_chart_availability.dtype != torch.float32:
            raise TypeError("future chart log availability must remain FP32")
        _shape(
            self.camera_chart_availability,
            (batch, objects, cameras, 1),
            "future camera chart availability",
        )
        if self.camera_chart_availability.dtype != torch.float32:
            raise TypeError("future camera chart availability must remain FP32")
        _shape(
            self.log_camera_chart_availability,
            (batch, objects, cameras, 1),
            "future camera chart log availability",
        )
        if self.log_camera_chart_availability.dtype != torch.float32:
            raise TypeError("future camera chart log availability must remain FP32")

    def permute(self, permutation: Tensor) -> "FutureObjectDynamics":
        """Relabel the persistent global-object axis without changing values."""

        objects = int(self.current_reference.shape[1])
        if permutation.ndim != 1 or int(permutation.numel()) != objects:
            raise ValueError("future-dynamics permutation must cover every K slot")
        index = permutation.to(
            device=self.current_reference.device,
            dtype=torch.long,
        )
        return FutureObjectDynamics(
            current_reference=self.current_reference[:, index],
            successor_content=self.successor_content[:, :, index],
            semantic_delta=self.semantic_delta[:, :, index],
            transport_mean=self.transport_mean[:, :, index],
            transport_covariance=self.transport_covariance[:, :, index],
            chart_availability=self.chart_availability[:, index],
            log_chart_availability=self.log_chart_availability[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            camera_chart_availability=self.camera_chart_availability[:, index],
            log_camera_chart_availability=(
                self.log_camera_chart_availability[:, index]
            ),
        )

    @classmethod
    def neutral(cls, facts: ObjectFactSet, *, intervals: int = 4) -> "FutureObjectDynamics":
        facts.validate()
        current = facts.content
        batch, objects, width = current.shape
        cameras = int(facts.camera_coordinates.shape[2])
        zeros = current.new_zeros(batch, intervals, objects, width)
        return cls(
            current_reference=current,
            successor_content=current[:, None].expand(-1, intervals, -1, -1),
            semantic_delta=zeros,
            transport_mean=current.new_zeros(batch, intervals, objects, cameras, 2),
            transport_covariance=torch.zeros(
                batch,
                intervals,
                objects,
                cameras,
                3,
                device=current.device,
                dtype=torch.float32,
            ),
            chart_availability=facts.validity.float(),
            log_chart_availability=facts.log_validity.float(),
            camera_coordinates=facts.camera_coordinates.to(dtype=current.dtype),
            camera_chart_availability=facts.camera_validity.float(),
            log_camera_chart_availability=facts.log_camera_validity.float(),
        )


@dataclass(frozen=True)
class ObjectTopTrainingTargets:
    teacher_dynamics: FutureObjectDynamics | None
    current_loss_support: Tensor  # training-only current facts [B,K,C,1]
    plan_recognition: FuturePlanRecognition | None
    online_intent_loss: Tensor
    plan_recognition_loss: Tensor
    coarse_action_loss: Tensor
    history_proposal_loss: Tensor
    object_reconstruction_loss: Tensor

    @property
    def total_unweighted(self) -> Tensor:
        return (
            self.online_intent_loss
            + self.plan_recognition_loss
            + self.coarse_action_loss
            + self.history_proposal_loss
            + self.object_reconstruction_loss
        )


def normalized_entropy(probability: Tensor, *, dim: int = -1) -> Tensor:
    support = int(probability.shape[dim])
    if support < 2:
        return probability.new_zeros(probability.shape[:-1], dtype=torch.float32)
    value = probability.float().clamp_min(1e-8)
    return -(value * value.log()).sum(dim=dim) / torch.log(value.new_tensor(float(support)))
