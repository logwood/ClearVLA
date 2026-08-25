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

    g3_public_scene_audit: Tensor  # G3-local audit carrier [B,C,Y,X,H]
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
    candidate_semantic_prior: Tensor  # typed conditional local priors
    candidate_appearance_prior: Tensor
    candidate_geometry_prior: Tensor
    candidate_transport_prior: Tensor  # [B,C,Y,X,M,2]

    def validate(self) -> None:
        if self.g3_public_scene_audit.ndim != 5 or self.dino_content.ndim != 5:
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
        for name in (
            "candidate_semantic_prior",
            "candidate_appearance_prior",
            "candidate_geometry_prior",
        ):
            if tuple(getattr(self, name).shape) != prefix:
                raise ValueError(f"{name} is misaligned")
        if tuple(self.candidate_transport_prior.shape) != (*prefix, 2):
            raise ValueError("candidate transport prior is misaligned")
        chart_prefix = prefix[:4]
        if tuple(self.g3_public_scene_audit.shape[:4]) != chart_prefix:
            raise ValueError("G3 audit chart lost camera/spatial identity")
        if tuple(self.dino_content.shape[:4]) != chart_prefix:
            raise ValueError("DINO chart lost camera/spatial identity")
        _shape(self.cell_observed, (*chart_prefix, 1), "dense observed-cell mask")
        if self.cell_observed.dtype != torch.bool:
            raise TypeError("dense observed-cell mask must be boolean")


@dataclass(frozen=True)
class ObjectFactSet:
    """Four global objects with a soft, reversible chart correspondence."""

    dense_chart: DenseFactChart
    # One protected scene-wide content value plus K zero-centred innovations.
    # ``content`` remains the absolute object reference required by Teacher;
    # online S/W consumers must use ``content_innovation`` for the K axis so
    # the public direction is not copied K times and mistaken for ownership.
    public_content: Tensor  # [B,D]
    content: Tensor  # [B,K,D]
    semantic: Tensor  # [B,K,R]
    appearance: Tensor
    geometry: Tensor
    camera_coordinates: Tensor  # [B,K,C,2]
    camera_transport_prior: Tensor  # [B,K,C,2]
    camera_support: Tensor  # [B,K,C,1]
    camera_chart_availability: Tensor  # observable camera/chart support [B,K,C,1]
    # Joint physical assignment mass in each camera, normalized by the
    # camera's observed candidate mass.  This is evidence strength used for
    # camera reduction and audit; unlike ``camera_support`` it is not an
    # object-width statistic, and unlike ``camera_chart_availability`` it is not an
    # independent physical loss-support field.
    camera_evidence_mass: Tensor  # [B,K,C,1]
    support: Tensor  # [B,K,1]
    # Read-conditioned object-vs-null confidence.  This is deliberately not
    # the fraction of total chart area allocated to an object.
    existence: Tensor  # [B,K,1]
    # Physical support of the object's own read.  Unlike existence this is a
    # legal-source mask, not a learned confidence or allocation prior.
    chart_availability: Tensor  # observable object/chart support [B,K,1]
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
    def content_innovation(self) -> Tensor:
        """Object-owned content after removing the single public scene base."""

        return self.content - self.public_content[:, None].to(dtype=self.content.dtype)

    @property
    def coordinates(self) -> Tensor:
        """V120 object coordinate reduced only over physical camera support."""

        weight = self.camera_evidence_mass.float()
        return (
            self.camera_coordinates.float() * weight
        ).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

    @property
    def transport_prior(self) -> Tensor:
        """V120 object transport prior reduced only over valid cameras."""

        weight = self.camera_evidence_mass.float()
        return (
            self.camera_transport_prior.float() * weight
        ).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)

    def validate(self) -> None:
        self.dense_chart.validate()
        if self.content.ndim != 3:
            raise ValueError("object content must be [B,K,D]")
        batch, objects = self.content.shape[:2]
        _shape(
            self.public_content,
            (batch, int(self.content.shape[-1])),
            "public scene content",
        )
        if self.public_content.dtype != self.content.dtype:
            raise TypeError("public and object content must share one model dtype")
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
            self.camera_chart_availability,
            (batch, objects, cameras, 1),
            "object camera chart availability",
        )
        _shape(
            self.camera_evidence_mass,
            (batch, objects, cameras, 1),
            "object camera evidence mass",
        )
        _shape(self.support, (batch, objects, 1), "object support")
        _shape(self.existence, (batch, objects, 1), "object existence")
        _shape(
            self.chart_availability,
            (batch, objects, 1),
            "object chart availability",
        )
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
            public_content=self.public_content,
            content=self.content[:, index],
            semantic=self.semantic[:, index],
            appearance=self.appearance[:, index],
            geometry=self.geometry[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            camera_transport_prior=self.camera_transport_prior[:, index],
            camera_support=self.camera_support[:, index],
            camera_chart_availability=self.camera_chart_availability[:, index],
            camera_evidence_mass=self.camera_evidence_mass[:, index],
            support=self.support[:, index],
            existence=self.existence[:, index],
            chart_availability=self.chart_availability[:, index],
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
    """The only dynamic P1 boundary consumed by P2.

    Keeping the three owners separate until the actual P2 consumer makes the
    dynamic residual a query refinement rather than a second factual value.
    ``combined`` intentionally preserves the established numerical sum.
    """

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
        self.validate()
        return self.action_query + self.factual_base + self.policy_query_residual


@dataclass(frozen=True)
class CompletedP1PolicyState:
    """Live P1 state with factual and P2-query ownership split.

    ``factual_base`` is the cached, observation-owned P1 detail and therefore
    has no noisy-action dependency. ``policy_query_residual`` is the live V120
    policy-block write. It refines P2's effect query and conditionally refines
    P3 precision, while remaining outside the factual/protected consequence.
    The three P2 values are combined only by :class:`P2QueryDock` at P2's real
    consumer boundary.
    """

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
        for name in ("policy_query_residual",):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"completed P1 {name} must align with factual_base")
            if value.device != self.factual_base.device:
                raise ValueError(f"completed P1 {name} must share factual_base device")


@dataclass(frozen=True)
class ActionIntentDock:
    """S-owned inputs that the clean coarse-action compiler may consume."""

    interval_condition_innovation: Tensor  # [B,I,H]
    history_memory: Tensor  # [B,L,H]
    public_scene_memory: Tensor  # [B,1,H]
    object_innovation_memory: Tensor  # [B,K,H]

    def validate(self, *, hidden: int) -> None:
        batch = int(self.interval_condition_innovation.shape[0])
        _shape(
            self.interval_condition_innovation,
            (batch, 4, hidden),
            "action-intent interval condition innovation",
        )
        if (
            self.history_memory.ndim != 3
            or int(self.history_memory.shape[0]) != batch
            or int(self.history_memory.shape[-1]) != hidden
        ):
            raise ValueError("action-intent history memory must be [B,L,H]")
        _shape(
            self.public_scene_memory,
            (batch, 1, hidden),
            "action-intent public scene memory",
        )
        if (
            self.object_innovation_memory.ndim != 3
            or int(self.object_innovation_memory.shape[0]) != batch
            or int(self.object_innovation_memory.shape[-1]) != hidden
        ):
            raise ValueError("action-intent object innovation memory must be [B,K,H]")


@dataclass(frozen=True)
class WorldIntentDock:
    """S-owned common/residual boundary consumed by W1/W2 exactly once."""

    protected_goal_memory: Tensor  # [B,G,H]
    interval_condition_innovation: Tensor  # [B,I,H]
    typed_common_mass: Tensor  # [B,K,type,1]
    typed_common_value: Tensor  # [B,K,type,R]
    typed_interval_residual_mass: Tensor  # [B,I,K,type,1]
    typed_interval_residual_value: Tensor  # [B,I,K,type,R]

    def validate(self, *, hidden: int) -> None:
        batch = int(self.interval_condition_innovation.shape[0])
        _shape(
            self.interval_condition_innovation,
            (batch, 4, hidden),
            "world-intent interval condition innovation",
        )
        _shape(
            self.protected_goal_memory,
            (batch, 4, hidden),
            "world-intent protected goal memory",
        )
        if self.typed_common_mass.ndim != 4:
            raise ValueError("typed common mass must be [B,K,type,1]")
        if tuple(self.typed_common_mass.shape[:1]) != (batch,):
            raise ValueError("typed common mass lost its batch axis")
        if int(self.typed_common_mass.shape[2]) != 3 or int(
            self.typed_common_mass.shape[-1]
        ) != 1:
            raise ValueError("typed common mass lost semantic/appearance/geometry")
        if self.typed_common_value.ndim != 4 or tuple(
            self.typed_common_value.shape[:3]
        ) != tuple(self.typed_common_mass.shape[:3]):
            raise ValueError("typed common value does not align with its mass")
        if self.typed_interval_residual_mass.ndim != 5 or tuple(
            self.typed_interval_residual_mass.shape[:2]
        ) != (batch, 4):
            raise ValueError("typed interval residual mass must be [B,I,K,type,1]")
        if tuple(self.typed_interval_residual_mass.shape[2:4]) != tuple(
            self.typed_common_mass.shape[1:3]
        ) or int(self.typed_interval_residual_mass.shape[-1]) != 1:
            raise ValueError("typed common/residual mass axes do not align")
        if self.typed_interval_residual_value.ndim != 5 or tuple(
            self.typed_interval_residual_value.shape[:4]
        ) != tuple(self.typed_interval_residual_mass.shape[:4]):
            raise ValueError("typed interval residual value does not align with its mass")


@dataclass(frozen=True)
class FactualIntentDock:
    """Owner-preserving S context for the unchanged V120 P1 reader.

    K is conditionally read inside S; P1 receives the resulting interval/type
    context rather than a fixed K mean or an expanded global summary.
    """

    public_interval_context: Tensor  # [B,I,H]
    goal_interval_context: Tensor  # [B,I,H]
    history_interval_context: Tensor  # [B,I,H]
    typed_interval_context: Tensor  # [B,I,3,H]

    def validate(self, *, hidden: int) -> None:
        batch = int(self.public_interval_context.shape[0])
        expected = (batch, 4, hidden)
        _shape(
            self.public_interval_context,
            expected,
            "factual-intent public interval context",
        )
        _shape(
            self.goal_interval_context,
            expected,
            "factual-intent goal interval context",
        )
        _shape(
            self.history_interval_context,
            expected,
            "factual-intent history interval context",
        )
        _shape(
            self.typed_interval_context,
            (batch, 4, 3, hidden),
            "factual-intent typed interval context",
        )


@dataclass(frozen=True)
class PolicyIntentDock:
    """Read-only S context for complementary P2 and unchanged P3."""

    common_key: Tensor  # [B,H]
    interval_residual_key: Tensor  # [B,I,H]
    # Both typed axes retain S/W owner order:
    # semantic / appearance / geometry.  P2 consumer order is different and
    # must use an explicit named mapping rather than matching integer indices.
    typed_common_object_value: Tensor  # [B,K,S-type,R]
    typed_interval_residual_value: Tensor  # [B,I,K,S-type,R]
    temporal_control: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.common_key.shape[0])
        _shape(self.common_key, (batch, hidden), "policy-intent common key")
        _shape(
            self.interval_residual_key,
            (batch, 4, hidden),
            "policy-intent interval residual key",
        )
        if self.typed_common_object_value.ndim != 4 or tuple(
            self.typed_common_object_value.shape[:1]
        ) != (batch,):
            raise ValueError("policy-intent typed common value must be [B,K,type,R]")
        if int(self.typed_common_object_value.shape[2]) != 3:
            raise ValueError("policy-intent typed common value lost its type axis")
        if self.typed_interval_residual_value.ndim != 5 or tuple(
            self.typed_interval_residual_value.shape[:2]
        ) != (batch, 4):
            raise ValueError(
                "policy-intent typed residual must retain interval/object/type axes"
            )
        if tuple(self.typed_interval_residual_value.shape[2:4]) != tuple(
            self.typed_common_object_value.shape[1:3]
        ):
            raise ValueError("policy-intent common/residual object axes do not align")
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


@dataclass(frozen=True)
class ObjectIntentState:
    """Stateless intent bundle with public and object/type-owned outputs."""

    protected_goal_set: Tensor  # [B,4,H]
    history_tokens: Tensor  # [B,L,H]
    public_scene_token: Tensor  # [B,1,H]
    object_tokens: Tensor  # [B,K,H], public-free object innovations
    public_interval_carrier: Tensor  # [B,4,H]
    interval_condition_innovation: Tensor  # [B,4,H]
    public_common_condition: Tensor  # [B,H]
    public_interval_residual_condition: Tensor  # [B,4,H]
    goal_interval_context: Tensor  # [B,4,H]
    history_interval_context: Tensor  # [B,4,H]
    policy_interval_context: Tensor  # [B,4,H]
    policy_interval_innovation: Tensor  # [B,4,H]
    temporal_queries: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]
    typed_common_mass: Tensor  # [B,K,3,1]
    typed_common_value: Tensor  # [B,K,3,R]
    typed_interval_residual_mass: Tensor  # [B,4,K,3,1]
    typed_interval_residual_value: Tensor  # [B,4,K,3,R]
    typed_common_policy_components: Tensor  # [B,3,H]
    typed_interval_residual_policy_components: Tensor  # [B,4,3,H]
    goal_attention: Tensor  # [B,4,Lg]
    interval_goal_attention: Tensor  # [B,4,4]
    interval_history_attention: Tensor  # [B,4,L]
    interval_object_attention: Tensor  # [B,4,1+K], public then K innovations

    @property
    def interval_queries(self) -> Tensor:
        """Compatibility name for the S-owned consumer-specific context."""

        return self.policy_interval_context

    def action_dock(self) -> ActionIntentDock:
        return ActionIntentDock(
            interval_condition_innovation=self.interval_condition_innovation,
            history_memory=self.history_tokens,
            public_scene_memory=self.public_scene_token,
            object_innovation_memory=self.object_tokens,
        )

    def world_dock(self) -> WorldIntentDock:
        return WorldIntentDock(
            protected_goal_memory=self.protected_goal_set,
            interval_condition_innovation=self.interval_condition_innovation,
            typed_common_mass=self.typed_common_mass,
            typed_common_value=self.typed_common_value,
            typed_interval_residual_mass=self.typed_interval_residual_mass,
            typed_interval_residual_value=self.typed_interval_residual_value,
        )

    def factual_dock(self) -> FactualIntentDock:
        return FactualIntentDock(
            public_interval_context=self.interval_condition_innovation,
            goal_interval_context=self.goal_interval_context,
            history_interval_context=self.history_interval_context,
            typed_interval_context=(
                self.typed_common_policy_components[:, None]
                + self.typed_interval_residual_policy_components
            ),
        )

    def policy_dock(self) -> PolicyIntentDock:
        return PolicyIntentDock(
            common_key=self.public_common_condition,
            interval_residual_key=self.public_interval_residual_condition,
            typed_common_object_value=self.typed_common_value,
            typed_interval_residual_value=self.typed_interval_residual_value,
            temporal_control=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
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
            self.interval_condition_innovation,
            (batch, 4, hidden),
            "interval condition innovation",
        )
        _shape(
            self.public_common_condition,
            (batch, hidden),
            "public common condition",
        )
        _shape(
            self.public_interval_residual_condition,
            (batch, 4, hidden),
            "public interval residual condition",
        )
        _shape(
            self.goal_interval_context,
            (batch, 4, hidden),
            "goal interval context",
        )
        _shape(
            self.history_interval_context,
            (batch, 4, hidden),
            "history interval context",
        )
        _shape(
            self.policy_interval_context,
            (batch, 4, hidden),
            "policy interval context",
        )
        _shape(
            self.policy_interval_innovation,
            (batch, 4, hidden),
            "policy interval innovation",
        )
        _shape(self.temporal_queries, (batch, horizon, hidden), "temporal queries")
        _shape(self.state_change_evidence, (batch, hidden), "state-change evidence")
        if self.history_tokens.ndim != 3 or int(self.history_tokens.shape[0]) != batch:
            raise ValueError("intent history tokens must be [B,L,H]")
        if self.object_tokens.ndim != 3 or int(self.object_tokens.shape[0]) != batch:
            raise ValueError("object intent innovation tokens must be [B,K,H]")
        _shape(
            self.public_scene_token,
            (batch, 1, hidden),
            "intent public scene token",
        )
        objects = int(self.object_tokens.shape[1])
        _shape(self.typed_common_mass, (batch, objects, 3, 1), "typed common mass")
        if self.typed_common_value.ndim != 4 or tuple(
            self.typed_common_value.shape[:3]
        ) != (batch, objects, 3):
            raise ValueError("typed common value lost object/type identity")
        _shape(
            self.typed_interval_residual_mass,
            (batch, 4, objects, 3, 1),
            "typed interval residual mass",
        )
        if self.typed_interval_residual_value.ndim != 5 or tuple(
            self.typed_interval_residual_value.shape[:4]
        ) != (batch, 4, objects, 3):
            raise ValueError("typed residual value lost interval/object/type identity")
        _shape(
            self.typed_common_policy_components,
            (batch, 3, hidden),
            "typed common policy components",
        )
        _shape(
            self.typed_interval_residual_policy_components,
            (batch, 4, 3, hidden),
            "typed interval residual policy components",
        )
        if self.goal_attention.ndim != 3 or tuple(
            self.goal_attention.shape[:2]
        ) != (batch, 4):
            raise ValueError("intent goal attention must be [B,4,Lg]")
        _shape(
            self.interval_goal_attention,
            (batch, 4, 4),
            "intent interval-goal attention",
        )
        _shape(
            self.interval_history_attention,
            (batch, 4, int(self.history_tokens.shape[1])),
            "intent interval-history attention",
        )
        _shape(
            self.interval_object_attention,
            (batch, 4, objects + 1),
            "intent public-plus-object attention",
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
            public_scene_token=self.public_scene_token,
            object_tokens=self.object_tokens[:, index],
            public_interval_carrier=self.public_interval_carrier,
            interval_condition_innovation=self.interval_condition_innovation,
            public_common_condition=self.public_common_condition,
            public_interval_residual_condition=(
                self.public_interval_residual_condition
            ),
            goal_interval_context=self.goal_interval_context,
            history_interval_context=self.history_interval_context,
            policy_interval_context=self.policy_interval_context,
            policy_interval_innovation=self.policy_interval_innovation,
            temporal_queries=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            typed_common_mass=self.typed_common_mass[:, index],
            typed_common_value=self.typed_common_value[:, index],
            typed_interval_residual_mass=self.typed_interval_residual_mass[:, :, index],
            typed_interval_residual_value=self.typed_interval_residual_value[:, :, index],
            typed_common_policy_components=self.typed_common_policy_components,
            typed_interval_residual_policy_components=(
                self.typed_interval_residual_policy_components
            ),
            goal_attention=self.goal_attention,
            interval_goal_attention=self.interval_goal_attention,
            interval_history_attention=self.interval_history_attention,
            interval_object_attention=torch.cat(
                (
                    self.interval_object_attention[:, :, :1],
                    self.interval_object_attention[:, :, 1:][:, :, index],
                ),
                dim=2,
            ),
        )


# The descriptive name is exported without breaking existing import sites that
# still use ObjectIntentState for the same typed runtime container.
StatelessIntentBundle = ObjectIntentState


@dataclass(frozen=True)
class IntentStateSupervision:
    """The one auxiliary target owned by S itself.

    Future object effects belong to W and are supervised only at the
    :class:`FutureObjectDynamics` boundary.  S retains a small observable
    state-summary objective; it no longer decodes a second copy of W's
    semantic or geometry fields.
    """

    state_prediction: Tensor  # adjacent interval increment [B,4,S]
    state_target: Tensor  # adjacent interval increment [B,4,S]
    loss: Tensor

    def validate(self) -> None:
        if self.state_prediction.ndim != 3 or tuple(
            self.state_prediction.shape
        ) != tuple(self.state_target.shape):
            raise ValueError("intent state prediction/target must align as [B,4,S]")
        if int(self.state_prediction.shape[1]) != 4:
            raise ValueError("intent state target lost the four interval axis")
        if self.loss.ndim != 0:
            raise ValueError("intent state supervision loss must be scalar")


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
    transport_covariance: Tensor  # PSD xx/xy/yy [B,I,K,C,3]
    chart_availability: Tensor  # observable support [B,K,1]
    # Current object geometry remains a real camera mixture.  Reducing these
    # coordinates to one normalized-image mean creates a point that belongs
    # to no camera and silently penalizes multi-view objects in P2.
    camera_coordinates: Tensor  # [B,K,C,2]
    camera_chart_availability: Tensor  # observable camera/chart support [B,K,C,1]
    camera_weights: Tensor  # [B,K,C,1]

    @property
    def intervals(self) -> int:
        return int(self.semantic_delta.shape[1])

    @staticmethod
    def _common(value: Tensor) -> Tensor:
        """Return the protected effect shared across all future intervals."""

        if value.ndim < 3:
            raise ValueError("future effect must retain an interval axis")
        # Decomposition is one shared physical interface.  Compute it in
        # FP32 even when the online carrier is BF16 so interval centring does
        # not depend on autocast rounding at each downstream consumer.
        return value.float().mean(dim=1)

    @classmethod
    def _residual(cls, value: Tensor) -> Tensor:
        """Return the exact zero-mean interval residual of one effect field."""

        common = cls._common(value)
        return value.float() - common[:, None]

    @property
    def semantic_common(self) -> Tensor:
        return self._common(self.semantic_delta)

    @property
    def semantic_interval_residual(self) -> Tensor:
        return self._residual(self.semantic_delta)

    @property
    def transport_common(self) -> Tensor:
        return self._common(self.transport_mean)

    @property
    def transport_interval_residual(self) -> Tensor:
        return self._residual(self.transport_mean)

    def validate_effect_decomposition(self) -> None:
        """Check the exact common-plus-residual algebra used by S/W/P2."""

        for name in ("semantic_delta", "transport_mean"):
            value = getattr(self, name)
            common = self._common(value)
            residual = self._residual(value)
            if not torch.allclose(
                value.float(),
                common[:, None] + residual,
                atol=2.0e-6,
                rtol=0.0,
            ):
                raise ValueError(f"future {name} common/residual identity failed")
            residual_mean = residual.mean(dim=1)
            if not torch.allclose(
                residual_mean,
                torch.zeros_like(residual_mean),
                atol=2.0e-6,
                rtol=0.0,
            ):
                raise ValueError(f"future {name} residual is not interval-centred")

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
        _shape(
            self.transport_mean,
            (batch, intervals, objects, int(self.camera_coordinates.shape[2]), 2),
            "transport mean",
        )
        _shape(
            self.transport_covariance,
            (batch, intervals, objects, int(self.camera_coordinates.shape[2]), 3),
            "transport covariance",
        )
        if self.transport_covariance.dtype != torch.float32:
            raise ValueError(
                "future transport covariance must retain its FP32 PSD boundary"
            )
        _shape(
            self.chart_availability,
            (batch, objects, 1),
            "chart availability",
        )
        if self.camera_coordinates.ndim != 4 or tuple(
            self.camera_coordinates.shape[:2]
        ) != (batch, objects) or int(self.camera_coordinates.shape[-1]) != 2:
            raise ValueError("future camera coordinates must be [B,K,C,2]")
        _shape(
            self.camera_weights,
            (*self.camera_coordinates.shape[:-1], 1),
            "future camera weights",
        )
        _shape(
            self.camera_chart_availability,
            (*self.camera_coordinates.shape[:-1], 1),
            "future camera chart availability",
        )

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
            camera_coordinates=self.camera_coordinates[:, index],
            camera_chart_availability=self.camera_chart_availability[:, index],
            camera_weights=self.camera_weights[:, index],
        )

    @classmethod
    def neutral(cls, facts: ObjectFactSet, *, intervals: int = 4) -> "FutureObjectDynamics":
        facts.validate()
        current = facts.content
        batch, objects, width = current.shape
        zeros = current.new_zeros(batch, intervals, objects, width)
        return cls(
            current_reference=current,
            successor_content=current[:, None].expand(-1, intervals, -1, -1),
            semantic_delta=zeros,
            transport_mean=current.new_zeros(
                batch,
                intervals,
                objects,
                int(facts.camera_coordinates.shape[2]),
                2,
            ),
            transport_covariance=torch.zeros(
                batch,
                intervals,
                objects,
                int(facts.camera_coordinates.shape[2]),
                3,
                device=current.device,
                dtype=torch.float32,
            ),
            chart_availability=facts.chart_availability,
            camera_coordinates=facts.camera_coordinates.to(dtype=current.dtype),
            camera_chart_availability=facts.camera_chart_availability.to(
                dtype=current.dtype
            ),
            camera_weights=(
                facts.camera_evidence_mass.float()
                * facts.camera_chart_availability.float()
            ).to(dtype=current.dtype),
        )


@dataclass(frozen=True)
class ObjectTopTrainingTargets:
    teacher_dynamics: FutureObjectDynamics | None
    current_loss_support: Tensor  # training-only current facts [B,K,C,1]
    intent_supervision: IntentStateSupervision | None
    public_intent_loss: Tensor
    coarse_action_loss: Tensor
    history_proposal_loss: Tensor
    object_reconstruction_loss: Tensor

    @property
    def total_unweighted(self) -> Tensor:
        return (
            self.public_intent_loss
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
