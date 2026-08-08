"""Typed boundaries for the object/intent/dynamics 3-2-3 graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

CAPABILITY_NAME = "object_intent_dynamics_323"
CAPABILITY_SCHEMA = 2
INTERVAL_NAMES = ("h4_8", "h8_16", "h16_32", "h32_48")
INTERVAL_BOUNDS = ((4, 8), (8, 16), (16, 32), (32, 48))


@dataclass(frozen=True)
class ArchitectureManifest:
    """Small serialized identity; executable types own the real contract."""

    capability: str = CAPABILITY_NAME
    schema: int = CAPABILITY_SCHEMA
    topology: tuple[int, int, int] = (3, 2, 3)
    intervals: tuple[tuple[int, int], ...] = INTERVAL_BOUNDS
    object_slots: int = 4
    language_required: bool = True
    bottom_compatibility: str = "evidence_mmdit_cvae_workspace_v1"

    def validate(self) -> None:
        if self.capability != CAPABILITY_NAME or int(self.schema) != CAPABILITY_SCHEMA:
            raise ValueError("object-intent architecture identity is invalid")
        if tuple(self.topology) != (3, 2, 3):
            raise ValueError("object-intent architecture requires topology 3-2-3")
        if tuple(self.intervals) != INTERVAL_BOUNDS:
            raise ValueError("object-intent architecture requires four canonical intervals")
        if int(self.object_slots) != 4:
            raise ValueError("the first object-intent schema owns four global objects")
        if not self.language_required:
            raise ValueError("formal object-intent training requires language")

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "schema": int(self.schema),
            "topology": tuple(self.topology),
            "intervals": tuple(tuple(row) for row in self.intervals),
            "object_slots": int(self.object_slots),
            "language_required": bool(self.language_required),
            "bottom_compatibility": self.bottom_compatibility,
        }


ARCHITECTURE_MANIFEST = ArchitectureManifest()


def _shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")


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
    candidate_content: Tensor  # [B,C,Y,X,M,D]
    candidate_semantic: Tensor  # [B,C,Y,X,M,R]
    candidate_appearance: Tensor
    candidate_geometry: Tensor
    candidate_coordinates: Tensor  # [B,C,Y,X,M,2]
    candidate_support: Tensor  # [B,C,Y,X,M]
    candidate_validity: Tensor  # [B,C,Y,X,M,1]
    candidate_owner_prior: Tensor  # [B,C,Y,X,M]
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
        if tuple(self.candidate_transport_prior.shape) != (*prefix, 2):
            raise ValueError("candidate transport prior is misaligned")
        chart_prefix = prefix[:4]
        if tuple(self.public_scene_base.shape[:4]) != chart_prefix:
            raise ValueError("public chart lost camera/spatial identity")
        if tuple(self.dino_content.shape[:4]) != chart_prefix:
            raise ValueError("DINO chart lost camera/spatial identity")


@dataclass(frozen=True)
class ObjectFactSet:
    """Four global objects with a soft, reversible chart correspondence."""

    dense_chart: DenseFactChart
    content: Tensor  # [B,K,D]
    semantic: Tensor  # [B,K,R]
    appearance: Tensor
    geometry: Tensor
    coordinates: Tensor  # [B,K,2]
    transport_prior: Tensor  # [B,K,2]
    support: Tensor  # [B,K,1]
    # Read-conditioned object-vs-null confidence.  This is deliberately not
    # the fraction of total chart area allocated to an object.
    existence: Tensor  # [B,K,1]
    # Physical support of the object's own read.  Unlike existence this is a
    # legal-source mask, not a learned confidence or allocation prior.
    validity: Tensor  # [B,K,1]
    object_to_chart: Tensor  # read posterior [B,K,C,Y,X]
    candidate_assignment: Tensor  # joint local-prior competition mass [B,K,C,Y,X,M]
    null_assignment: Tensor  # joint local-prior null mass [B,C,Y,X,M]
    reconstructed_dino: Tensor  # [B,C,Y,X,D]
    reconstruction_error: Tensor  # scalar, not weighted here

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
        _shape(self.coordinates, (batch, objects, 2), "object coordinates")
        _shape(self.transport_prior, (batch, objects, 2), "object transport prior")
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
            coordinates=self.coordinates[:, index],
            transport_prior=self.transport_prior[:, index],
            support=self.support[:, index],
            existence=self.existence[:, index],
            validity=self.validity[:, index],
            object_to_chart=self.object_to_chart[:, index],
            candidate_assignment=self.candidate_assignment[:, index],
            null_assignment=self.null_assignment,
            reconstructed_dino=self.reconstructed_dino,
            reconstruction_error=self.reconstruction_error,
        )


@dataclass(frozen=True)
class ObjectIntentState:
    """Online, stateless intent state; no phase/progress forward variable."""

    protected_goal_set: Tensor  # [B,4,H]
    history_tokens: Tensor  # [B,L,H]
    object_tokens: Tensor  # protected public/content objects [B,K,H]
    semantic_object_tokens: Tensor  # [B,K,H]
    appearance_object_tokens: Tensor  # [B,K,H]
    geometry_object_tokens: Tensor  # [B,K,H]
    interval_queries: Tensor  # [B,4,H]
    temporal_queries: Tensor  # [B,T,H]
    # A zero-centred value built only from observed state deltas and current
    # transport.  It is not a phase, progress, completion, or terminal score.
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


@dataclass(frozen=True)
class FuturePlanRecognition:
    """Training-only posterior target for online interval intent."""

    interval_targets: Tensor  # [B,4,H], detached at online matching boundary
    action_summary: Tensor  # [B,4,A]
    state_summary: Tensor  # [B,4,S]
    effect_summary: Tensor  # [B,4,D]
    reconstruction_loss: Tensor


@dataclass(frozen=True)
class CoarseActionIntentState:
    tokens: Tensor  # [B,4,H]
    action_prediction: Tensor  # [B,4,A]
    target: Tensor | None
    loss: Tensor


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
    validity: Tensor  # [B,I,K,1]
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
        _shape(self.transport_mean, (batch, intervals, objects, 2), "transport mean")
        _shape(self.transport_covariance, (batch, intervals, objects, 3), "transport covariance")
        for name in ("visibility", "persistence", "uncertainty", "validity"):
            _shape(getattr(self, name), (batch, intervals, objects, 1), name)
        if self.future_address.ndim != 6 or tuple(self.future_address.shape[:3]) != (
            batch,
            intervals,
            objects,
        ):
            raise ValueError("future address must be [B,I,K,C,Y,X]")
        _shape(self.object_coordinates, (batch, objects, 2), "future object coordinates")

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
            validity=scalar,
            future_address=address,
            object_coordinates=facts.coordinates,
        )


@dataclass(frozen=True)
class ObjectTopTrainingTargets:
    teacher_dynamics: FutureObjectDynamics | None
    plan_recognition: FuturePlanRecognition | None
    online_intent_loss: Tensor
    plan_recognition_loss: Tensor
    coarse_action_loss: Tensor
    object_reconstruction_loss: Tensor

    @property
    def total_unweighted(self) -> Tensor:
        return (
            self.online_intent_loss
            + self.plan_recognition_loss
            + self.coarse_action_loss
            + self.object_reconstruction_loss
        )


def normalized_entropy(probability: Tensor, *, dim: int = -1) -> Tensor:
    support = int(probability.shape[dim])
    if support < 2:
        return probability.new_zeros(probability.shape[:-1], dtype=torch.float32)
    value = probability.float().clamp_min(1e-8)
    return -(value * value.log()).sum(dim=dim) / torch.log(
        value.new_tensor(float(support))
    )
