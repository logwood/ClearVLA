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
    typed_consistency_error: Tensor  # scalar, inside the existing G budget

    @property
    def batch(self) -> int:
        return int(self.content.shape[0])

    @property
    def objects(self) -> int:
        return int(self.content.shape[1])

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
        if self.typed_consistency_error.ndim != 0:
            raise ValueError("typed consistency error must be scalar")

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
            typed_consistency_error=self.typed_consistency_error,
        )


@dataclass(frozen=True)
class ObjectFactualDock:
    """The single current-fact bridge from global K objects into P1/P2.

    ``fact_by_object`` is read from the high-resolution current observation
    under the corresponding global-object support.  It is not reconstructed
    by expanding an already pooled P1 value.  ``aggregate_fact`` is the actual
    P1 update delivered to the policy trajectory.  P2 reuses the same K/null
    posterior and chart coordinates instead of inventing another object basis.
    """

    fact_by_object: Tensor  # [B,T,Q,K,H], conditional values
    object_posterior: Tensor  # [B,T,Q,K]
    null_posterior: Tensor  # [B,T,Q,1]
    chart_posterior: Tensor  # [B,T,Q,K,C,Y,X]
    camera_coordinates: Tensor  # [B,T,Q,K,C,2]
    aggregate_fact: Tensor  # [B,T,Q,H]

    @property
    def objects(self) -> int:
        return int(self.fact_by_object.shape[3])

    def validate(self) -> None:
        if self.fact_by_object.ndim != 5:
            raise ValueError("object factual values must be [B,T,Q,K,H]")
        batch, horizon, basis, objects, hidden = self.fact_by_object.shape
        _shape(
            self.object_posterior,
            (batch, horizon, basis, objects),
            "object factual posterior",
        )
        _shape(
            self.null_posterior,
            (batch, horizon, basis, 1),
            "object factual null posterior",
        )
        if self.chart_posterior.ndim != 7 or tuple(self.chart_posterior.shape[:4]) != (
            batch,
            horizon,
            basis,
            objects,
        ):
            raise ValueError("object factual chart posterior must be [B,T,Q,K,C,Y,X]")
        cameras = int(self.chart_posterior.shape[4])
        _shape(
            self.camera_coordinates,
            (batch, horizon, basis, objects, cameras, 2),
            "object factual camera coordinates",
        )
        _shape(
            self.aggregate_fact,
            (batch, horizon, basis, hidden),
            "object factual aggregate",
        )
        # Numerical invariants are checked at construction/preflight.  P2
        # calls this shape validator at every ODE step, so value reductions or
        # Python ``bool`` conversions here would introduce a GPU sync into the
        # deployment hot path.

    def permute(self, permutation: Tensor) -> "ObjectFactualDock":
        if permutation.ndim != 1 or int(permutation.numel()) != self.objects:
            raise ValueError("object factual permutation must cover every K slot")
        index = permutation.to(device=self.fact_by_object.device, dtype=torch.long)
        return ObjectFactualDock(
            fact_by_object=self.fact_by_object[:, :, :, index],
            object_posterior=self.object_posterior[:, :, :, index],
            null_posterior=self.null_posterior,
            chart_posterior=self.chart_posterior[:, :, :, index],
            camera_coordinates=self.camera_coordinates[:, :, :, index],
            aggregate_fact=self.aggregate_fact,
        )


@dataclass(frozen=True)
class ObjectIntentState:
    """Online, stateless intent state; no phase/progress forward variable."""

    interval_queries: Tensor  # [B,4,H]
    interval_action_innovations: Tensor  # [B,4,H], identity-free
    interval_state_innovations: Tensor  # [B,4,H], identity-free
    interval_object_keys: Tensor  # [B,4,K,H]
    interval_object_values: Tensor  # [B,4,K,H]
    temporal_queries: Tensor  # [B,T,H]
    temporal_innovations: Tensor  # [B,T,H], the only temporal value exported to P3
    # A zero-centred value built only from observed state deltas and current
    # transport.  It is not a phase, progress, completion, or terminal score.
    state_change_evidence: Tensor  # [B,H]

    def validate(self, *, horizon: int, hidden: int) -> None:
        batch = int(self.interval_queries.shape[0])
        _shape(self.interval_queries, (batch, 4, hidden), "interval queries")
        _shape(
            self.interval_action_innovations,
            (batch, 4, hidden),
            "interval action innovations",
        )
        _shape(
            self.interval_state_innovations,
            (batch, 4, hidden),
            "interval state innovations",
        )
        _shape(self.temporal_queries, (batch, horizon, hidden), "temporal queries")
        _shape(
            self.temporal_innovations,
            (batch, horizon, hidden),
            "temporal innovations",
        )
        _shape(self.state_change_evidence, (batch, hidden), "state-change evidence")
        if self.interval_object_keys.ndim != 4:
            raise ValueError("interval object keys must preserve [B,I,K,H]")
        objects = int(self.interval_object_keys.shape[2])
        _shape(
            self.interval_object_keys,
            (batch, 4, objects, hidden),
            "interval object keys",
        )
        _shape(
            self.interval_object_values,
            (batch, 4, objects, hidden),
            "interval object values",
        )

    def permute(self, permutation: Tensor) -> "ObjectIntentState":
        """Return the same intent state under a relabeling of global K slots.

        Interval, temporal and state-change values have no object axis and
        therefore remain unchanged.  Keeping this operation explicit makes
        causal audits exercise the real G→S→W owner boundary instead of
        rebuilding a synthetic object axis after pooling.
        """

        objects = int(self.interval_object_keys.shape[2])
        if permutation.ndim != 1 or int(permutation.numel()) != objects:
            raise ValueError("intent permutation must cover every K slot")
        index = permutation.to(
            device=self.interval_object_keys.device,
            dtype=torch.long,
        )
        return ObjectIntentState(
            interval_queries=self.interval_queries,
            interval_action_innovations=self.interval_action_innovations,
            interval_state_innovations=self.interval_state_innovations,
            interval_object_keys=self.interval_object_keys[:, :, index],
            interval_object_values=self.interval_object_values[:, :, index],
            temporal_queries=self.temporal_queries,
            temporal_innovations=self.temporal_innovations,
            state_change_evidence=self.state_change_evidence,
        )


@dataclass(frozen=True)
class FuturePlanRecognition:
    """Factorized training-only targets for the online intent organizer."""

    action_targets: Tensor  # [B,4,H]
    state_targets: Tensor  # [B,4,H]
    object_key_targets: Tensor  # [B,4,K,H]
    object_value_targets: Tensor  # [B,4,K,H]
    action_summary: Tensor  # [B,4,3*A] start/end/change
    state_summary: Tensor  # [B,4,3*S] start/end/change
    effect_summary: Tensor  # [B,4,K,2D+2] stable/delta/transport
    object_validity: Tensor  # [B,4,K,1]
    reconstruction_loss: Tensor

    def validate(self, *, hidden: int) -> None:
        if self.action_targets.ndim != 3:
            raise ValueError("recognizer action target must be [B,4,H]")
        batch = int(self.action_targets.shape[0])
        _shape(self.action_targets, (batch, 4, hidden), "action targets")
        _shape(self.state_targets, (batch, 4, hidden), "state targets")
        if (
            self.object_key_targets.ndim != 4
            or tuple(self.object_key_targets.shape[:2]) != (batch, 4)
            or int(self.object_key_targets.shape[-1]) != hidden
        ):
            raise ValueError("recognizer object keys must be [B,4,K,H]")
        _shape(
            self.object_value_targets,
            tuple(self.object_key_targets.shape),
            "recognizer object values",
        )
        _shape(
            self.object_validity,
            (*self.object_key_targets.shape[:-1], 1),
            "recognizer object validity",
        )
        if self.action_summary.ndim != 3 or self.state_summary.ndim != 3:
            raise ValueError("recognizer action/state summaries lost interval axis")
        if self.effect_summary.ndim != 4:
            raise ValueError("recognizer effect summary lost object axis")
        if self.reconstruction_loss.ndim != 0:
            raise ValueError("recognizer reconstruction loss must be scalar")


@dataclass(frozen=True)
class CoarseActionIntentState:
    tokens: Tensor  # [B,4,H], query plus innovation for diagnostics/reconstruction
    innovations: Tensor  # [B,4,H], the only W-visible value
    action_prediction: Tensor  # [B,4,3*A] start/end/change
    target: Tensor | None
    loss: Tensor


@dataclass(frozen=True)
class HistoryActionProposalState:
    """The preserved clean 24-step proposal from executed-action history.

    This is an online, causal condition.  It may shape P1's factual query and
    the controlled transition, but it is not a second W value or bottom action
    shortcut.
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
class FutureObjectDynamics:
    """The only W value object visible to P2."""

    current_reference: Tensor  # [B,K,D]
    successor_content: Tensor  # [B,I,K,D]
    semantic_delta: Tensor  # [B,I,K,D]
    transport_mean: Tensor  # [B,I,K,C,2]
    transport_covariance: Tensor  # [B,I,K,C,3]
    visibility: Tensor  # zero-centred visibility change [B,I,K,1]
    persistence: Tensor  # zero-centred track-persistence change [B,I,K,1]
    uncertainty: Tensor  # [B,I,K,1]
    reliability: Tensor  # calibration only [B,I,K,1]
    validity: Tensor  # physical per-camera support [B,I,K,C,1]
    future_address: Tensor  # [B,I,K,C,Y,X]
    object_coordinates: Tensor  # [B,K,C,2]

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
        cameras = int(self.future_address.shape[3])
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
        for name in ("visibility", "persistence", "uncertainty", "reliability"):
            _shape(getattr(self, name), (batch, intervals, objects, 1), name)
        _shape(
            self.validity,
            (batch, intervals, objects, cameras, 1),
            "future camera validity",
        )
        _shape(
            self.object_coordinates,
            (batch, objects, cameras, 2),
            "future object camera coordinates",
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
            validity=self.validity[:, :, index],
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
        cameras = int(facts.object_to_chart.shape[2])
        address = facts.object_to_chart[:, None].expand(-1, intervals, -1, -1, -1, -1)
        return cls(
            current_reference=current,
            successor_content=current[:, None].expand(-1, intervals, -1, -1),
            semantic_delta=zeros,
            transport_mean=current.new_zeros(batch, intervals, objects, cameras, 2),
            transport_covariance=current.new_zeros(batch, intervals, objects, cameras, 3),
            visibility=scalar,
            persistence=scalar,
            uncertainty=scalar,
            reliability=scalar,
            validity=facts.camera_validity[:, None].expand(-1, intervals, -1, -1, -1),
            future_address=address,
            object_coordinates=facts.camera_coordinates,
        )


@dataclass(frozen=True)
class ObjectTopTrainingTargets:
    teacher_dynamics: FutureObjectDynamics | None
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
