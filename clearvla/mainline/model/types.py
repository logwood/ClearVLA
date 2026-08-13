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
    content: Tensor  # [B,K,D]
    semantic: Tensor  # [B,K,R]
    appearance: Tensor
    geometry: Tensor
    camera_coordinates: Tensor  # [B,K,C,2]
    camera_transport_prior: Tensor  # [B,K,C,2]
    camera_support: Tensor  # [B,K,C,1]
    camera_validity: Tensor  # [B,K,C,1]
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
            content=self.content[:, index],
            semantic=self.semantic[:, index],
            appearance=self.appearance[:, index],
            geometry=self.geometry[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            camera_transport_prior=self.camera_transport_prior[:, index],
            camera_support=self.camera_support[:, index],
            camera_validity=self.camera_validity[:, index],
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
class ObjectIntentState:
    """Recovered V120 stateless intent with cumulative typed queries."""

    protected_goal_set: Tensor  # [B,4,H]
    history_tokens: Tensor  # [B,L,H]
    object_tokens: Tensor  # [B,K,H]
    semantic_object_tokens: Tensor  # [B,K,H]
    appearance_object_tokens: Tensor  # [B,K,H]
    geometry_object_tokens: Tensor  # [B,K,H]
    interval_queries: Tensor  # [B,4,H]
    temporal_queries: Tensor  # [B,T,H]
    state_change_evidence: Tensor  # [B,H]
    goal_attention: Tensor  # [B,4,Lg]
    interval_goal_attention: Tensor  # [B,4,4]
    interval_history_attention: Tensor  # [B,4,L]
    interval_object_attention: Tensor  # [B,4,K]
    interval_semantic_attention: Tensor  # [B,4,K]
    interval_appearance_attention: Tensor  # [B,4,K]
    interval_geometry_attention: Tensor  # [B,4,K]

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.interval_queries.shape[0])
        _shape(self.protected_goal_set, (batch, 4, hidden), "protected goal set")
        _shape(self.interval_queries, (batch, 4, hidden), "interval queries")
        _shape(self.temporal_queries, (batch, horizon, hidden), "temporal queries")
        _shape(self.state_change_evidence, (batch, hidden), "state-change evidence")
        if self.object_tokens.ndim != 3:
            raise ValueError("object intent public tokens must be [B,K,H]")
        object_shape = tuple(self.object_tokens.shape)
        for name in (
            "semantic_object_tokens",
            "appearance_object_tokens",
            "geometry_object_tokens",
        ):
            if tuple(getattr(self, name).shape) != object_shape:
                raise ValueError(f"{name} lost the global-object axis")

    def permute(self, permutation: Tensor) -> "ObjectIntentState":
        """Return the same intent state under a relabeling of global K slots.

        Interval, temporal and state-change values have no object axis and
        therefore remain unchanged.  Keeping this operation explicit makes
        causal audits exercise the real G→S→W owner boundary instead of
        rebuilding a synthetic object axis after pooling.
        """

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
            semantic_object_tokens=self.semantic_object_tokens[:, index],
            appearance_object_tokens=self.appearance_object_tokens[:, index],
            geometry_object_tokens=self.geometry_object_tokens[:, index],
            interval_queries=self.interval_queries,
            temporal_queries=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            goal_attention=self.goal_attention,
            interval_goal_attention=self.interval_goal_attention,
            interval_history_attention=self.interval_history_attention,
            interval_object_attention=self.interval_object_attention[:, :, index],
            interval_semantic_attention=self.interval_semantic_attention[:, :, index],
            interval_appearance_attention=self.interval_appearance_attention[:, :, index],
            interval_geometry_attention=self.interval_geometry_attention[:, :, index],
        )


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
    transport_mean: Tensor  # [B,I,K,2]
    transport_covariance: Tensor  # [B,I,K,3]
    visibility: Tensor  # zero-centred visibility change [B,I,K,1]
    persistence: Tensor  # zero-centred track-persistence change [B,I,K,1]
    uncertainty: Tensor  # [B,I,K,1]
    reliability: Tensor  # calibration only [B,I,K,1]
    future_selector_validity: Tensor  # online P2 selector [B,I,K,1]
    future_address: Tensor  # [B,I,K,C,Y,X]
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
        if self.future_address.ndim != 6 or tuple(self.future_address.shape[:3]) != (
            batch,
            intervals,
            objects,
        ):
            raise ValueError("future address must be [B,I,K,C,Y,X]")
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
            future_selector_validity=self.future_selector_validity[:, :, index],
            future_address=self.future_address[:, :, index],
            object_coordinates=self.object_coordinates[:, index],
        )

    @classmethod
    def neutral(cls, facts: ObjectFactSet, *, intervals: int = 4) -> "FutureObjectDynamics":
        facts.validate()
        current = facts.content
        batch, objects, width = current.shape
        zeros = current.new_zeros(batch, intervals, objects, width)
        scalar = current.new_zeros(batch, intervals, objects, 1)
        address = facts.object_to_chart[:, None].expand(-1, intervals, -1, -1, -1, -1)
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
            future_selector_validity=facts.validity[:, None].expand(
                -1, intervals, -1, -1
            ),
            future_address=address,
            object_coordinates=facts.coordinates.to(dtype=current.dtype),
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
