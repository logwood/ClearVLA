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
    candidate_semantic_prior: Tensor  # typed conditional local priors
    candidate_appearance_prior: Tensor
    candidate_geometry_prior: Tensor
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
    camera_validity: Tensor  # [B,K,C,1]
    # Joint physical assignment mass in each camera, normalized by the
    # camera's observed candidate mass.  This is evidence strength used for
    # camera reduction and audit; unlike ``camera_support`` it is not an
    # object-width statistic, and unlike ``camera_validity`` it is not an
    # independent physical loss-support field.
    camera_evidence_mass: Tensor  # [B,K,C,1]
    support: Tensor  # [B,K,1]
    # Read-conditioned object-vs-null confidence.  This is deliberately not
    # the fraction of total chart area allocated to an object.
    existence: Tensor  # [B,K,1]
    # Physical support of the object's own read.  Unlike existence this is a
    # legal-source mask, not a learned confidence or allocation prior.
    validity: Tensor  # [B,K,1]
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
            self.camera_validity,
            (batch, objects, cameras, 1),
            "object camera validity",
        )
        _shape(
            self.camera_evidence_mass,
            (batch, objects, cameras, 1),
            "object camera evidence mass",
        )
        _shape(self.support, (batch, objects, 1), "object support")
        _shape(self.existence, (batch, objects, 1), "object existence")
        _shape(self.validity, (batch, objects, 1), "object validity")
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
            camera_validity=self.camera_validity[:, index],
            camera_evidence_mass=self.camera_evidence_mass[:, index],
            support=self.support[:, index],
            existence=self.existence[:, index],
            validity=self.validity[:, index],
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
class ActionIntentDock:
    """S-owned inputs that the clean coarse-action compiler may consume."""

    interval_condition_innovation: Tensor  # [B,I,H]
    history_memory: Tensor  # [B,L,H]
    public_scene_memory: Tensor  # [B,1,H]
    object_innovation_memory: Tensor  # [B,K,H]
    typed_interval_object_value: Tensor  # [B,I,K,type,R]

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
        if self.typed_interval_object_value.ndim != 5 or tuple(
            self.typed_interval_object_value.shape[:2]
        ) != (batch, 4):
            raise ValueError(
                "action-intent typed value must retain interval/object/type axes"
            )
        if int(self.typed_interval_object_value.shape[3]) != 3:
            raise ValueError(
                "action-intent typed value lost semantic/appearance/geometry"
            )


@dataclass(frozen=True)
class WorldIntentDock:
    """S-owned relevance boundary consumed by W1/W2 exactly once."""

    protected_goal_memory: Tensor  # [B,G,H]
    interval_condition_innovation: Tensor  # [B,I,H]
    typed_relevance_mass: Tensor  # [B,I,K,type,1]
    typed_relevance_value: Tensor  # [B,I,K,type,R]

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
        if self.typed_relevance_mass.ndim != 5:
            raise ValueError("typed relevance mass must be [B,I,K,type,1]")
        if tuple(self.typed_relevance_mass.shape[:2]) != (batch, 4):
            raise ValueError("typed relevance mass lost its interval axis")
        if int(self.typed_relevance_mass.shape[3]) != 3 or int(
            self.typed_relevance_mass.shape[-1]
        ) != 1:
            raise ValueError("typed relevance mass lost semantic/appearance/geometry")
        if self.typed_relevance_value.ndim != 5 or tuple(
            self.typed_relevance_value.shape[:4]
        ) != tuple(self.typed_relevance_mass.shape[:4]):
            raise ValueError("typed relevance value does not align with its mass")


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
    """Read-only S context for the unchanged P2/P3 compilers."""

    interval_key: Tensor  # [B,I,H]
    typed_interval_object_value: Tensor  # [B,I,K,type,R]
    temporal_control: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.interval_key.shape[0])
        _shape(self.interval_key, (batch, 4, hidden), "policy-intent interval key")
        if self.typed_interval_object_value.ndim != 5 or tuple(
            self.typed_interval_object_value.shape[:2]
        ) != (batch, 4):
            raise ValueError(
                "policy-intent typed value must retain interval/object/type axes"
            )
        if int(self.typed_interval_object_value.shape[3]) != 3:
            raise ValueError(
                "policy-intent typed value lost semantic/appearance/geometry"
            )
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
    policy_interval_context: Tensor  # [B,4,H]
    policy_interval_innovation: Tensor  # [B,4,H]
    temporal_queries: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]
    typed_relevance_mass: Tensor  # [B,4,K,3,1]
    typed_relevance_value: Tensor  # [B,4,K,3,R]
    typed_policy_components: Tensor  # [B,4,3,H]
    goal_attention: Tensor  # [B,4,Lg]
    interval_goal_attention: Tensor  # [B,4,4]
    interval_history_attention: Tensor  # [B,4,L]
    interval_object_attention: Tensor  # [B,4,1+K], public then K innovations

    @property
    def interval_queries(self) -> Tensor:
        """Compatibility name for the S-owned consumer-specific context."""

        return self.policy_interval_context

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
            interval_condition_innovation=self.interval_condition_innovation,
            history_memory=self.history_tokens,
            public_scene_memory=self.public_scene_token,
            object_innovation_memory=self.object_tokens,
            typed_interval_object_value=self.typed_relevance_value,
        )

    def world_dock(self) -> WorldIntentDock:
        return WorldIntentDock(
            protected_goal_memory=self.protected_goal_set,
            interval_condition_innovation=self.interval_condition_innovation,
            typed_relevance_mass=self.typed_relevance_mass,
            typed_relevance_value=self.typed_relevance_value,
        )

    def factual_dock(self) -> FactualIntentDock:
        batch = int(self.policy_interval_context.shape[0])
        return FactualIntentDock(
            phase_context=self.policy_interval_innovation,
            condition_query_context=self.protected_goal_set.mean(dim=1)[:, None].expand(
                -1, 4, -1
            ),
            history_query_context=self.history_tokens[:, -1:, :].expand(
                batch, 4, -1
            ),
        )

    def policy_dock(self) -> PolicyIntentDock:
        return PolicyIntentDock(
            # P2 owns two explicit sources: public observable condition and
            # typed interval/object values.  Feeding the already typed-enriched
            # policy innovation here duplicated the typed route through both
            # public_intent_key and typed_intent_key.
            interval_key=self.interval_condition_innovation,
            typed_interval_object_value=self.typed_relevance_value,
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
        _shape(
            self.typed_relevance_mass,
            (batch, 4, objects, 3, 1),
            "typed relevance mass",
        )
        if self.typed_relevance_value.ndim != 5 or tuple(
            self.typed_relevance_value.shape[:4]
        ) != (batch, 4, objects, 3):
            raise ValueError("typed relevance value lost interval/object/type identity")
        _shape(
            self.typed_policy_components,
            (batch, 4, 3, hidden),
            "typed policy components",
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
            policy_interval_context=self.policy_interval_context,
            policy_interval_innovation=self.policy_interval_innovation,
            temporal_queries=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            typed_relevance_mass=self.typed_relevance_mass[:, :, index],
            typed_relevance_value=self.typed_relevance_value[:, :, index],
            typed_policy_components=self.typed_policy_components,
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
class IntentFutureSupervision:
    """Stable physical targets decoded from the online S boundaries."""

    state_prediction: Tensor  # [B,4,S]
    state_target: Tensor  # [B,4,S]
    semantic_prediction: Tensor  # [B,4,K,D]
    semantic_target: Tensor
    status_prediction: Tensor  # [B,4,K,2]
    status_target: Tensor
    transport_prediction: Tensor  # [B,4,K,2]
    transport_target: Tensor
    public_loss: Tensor
    semantic_loss: Tensor
    status_loss: Tensor
    transport_loss: Tensor
    typed_loss: Tensor

    def validate(self) -> None:
        if self.state_prediction.ndim != 3 or tuple(
            self.state_prediction.shape
        ) != tuple(self.state_target.shape):
            raise ValueError("intent state prediction/target must align as [B,4,S]")
        batch = int(self.state_prediction.shape[0])
        if int(self.state_prediction.shape[1]) != 4:
            raise ValueError("intent state target lost the four interval axis")
        for name in ("semantic", "status", "transport"):
            prediction = getattr(self, f"{name}_prediction")
            target = getattr(self, f"{name}_target")
            if prediction.ndim != 4 or tuple(prediction.shape) != tuple(target.shape):
                raise ValueError(f"intent {name} prediction/target must be [B,4,K,*]")
            if tuple(prediction.shape[:2]) != (batch, 4):
                raise ValueError(f"intent {name} target lost batch/interval identity")
        scalar_losses = (
            self.public_loss,
            self.semantic_loss,
            self.status_loss,
            self.transport_loss,
            self.typed_loss,
        )
        if any(value.ndim != 0 for value in scalar_losses):
            raise ValueError("intent future supervision losses must be scalars")


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
    transport_mean: Tensor  # [B,I,K,2]
    transport_covariance: Tensor  # [B,I,K,3]
    visibility: Tensor  # zero-centred visibility change [B,I,K,1]
    persistence: Tensor  # zero-centred track-persistence change [B,I,K,1]
    uncertainty: Tensor  # [B,I,K,1]
    reliability: Tensor  # calibration only [B,I,K,1]
    current_selector_validity: Tensor  # observed support [B,K,1]
    future_selector_validity: Tensor  # online P2 selector [B,I,K,1]
    object_coordinates: Tensor  # [B,K,2]

    @property
    def intervals(self) -> int:
        return int(self.semantic_delta.shape[1])

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
            (batch, intervals, objects, 2),
            "transport mean",
        )
        _shape(
            self.transport_covariance,
            (batch, intervals, objects, 3),
            "transport covariance",
        )
        for name in ("visibility", "persistence", "uncertainty", "reliability"):
            _shape(getattr(self, name), (batch, intervals, objects, 1), name)
        _shape(
            self.current_selector_validity,
            (batch, objects, 1),
            "current selector validity",
        )
        _shape(
            self.future_selector_validity,
            (batch, intervals, objects, 1),
            "future selector validity",
        )
        _shape(
            self.object_coordinates,
            (batch, objects, 2),
            "future object coordinates",
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
            visibility=self.visibility[:, :, index],
            persistence=self.persistence[:, :, index],
            uncertainty=self.uncertainty[:, :, index],
            reliability=self.reliability[:, :, index],
            current_selector_validity=self.current_selector_validity[:, index],
            future_selector_validity=self.future_selector_validity[:, :, index],
            object_coordinates=self.object_coordinates[:, index],
        )

    @classmethod
    def neutral(cls, facts: ObjectFactSet, *, intervals: int = 4) -> "FutureObjectDynamics":
        facts.validate()
        current = facts.content
        batch, objects, width = current.shape
        zeros = current.new_zeros(batch, intervals, objects, width)
        scalar = current.new_zeros(batch, intervals, objects, 1)
        return cls(
            current_reference=current,
            successor_content=current[:, None].expand(-1, intervals, -1, -1),
            semantic_delta=zeros,
            transport_mean=current.new_zeros(batch, intervals, objects, 2),
            transport_covariance=current.new_zeros(batch, intervals, objects, 3),
            visibility=scalar,
            persistence=scalar,
            uncertainty=scalar,
            reliability=scalar,
            current_selector_validity=(
                facts.validity * facts.existence.detach().clamp(0.0, 1.0)
            ),
            future_selector_validity=(
                facts.validity * facts.existence.detach().clamp(0.0, 1.0)
            )[:, None].expand(-1, intervals, -1, -1),
            object_coordinates=facts.coordinates.to(dtype=current.dtype),
        )


@dataclass(frozen=True)
class ObjectTopTrainingTargets:
    teacher_dynamics: FutureObjectDynamics | None
    current_loss_support: Tensor  # training-only current facts [B,K,C,1]
    intent_supervision: IntentFutureSupervision | None
    public_intent_loss: Tensor
    typed_intent_loss: Tensor
    coarse_action_loss: Tensor
    history_proposal_loss: Tensor
    object_reconstruction_loss: Tensor

    @property
    def total_unweighted(self) -> Tensor:
        return (
            self.public_intent_loss
            + self.typed_intent_loss
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
