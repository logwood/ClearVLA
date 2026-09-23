"""Typed boundaries for the capability-named mainline 3-2-3 graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from clearvla.vision.entity_chart import (
    CURRENT_IMAGE_CHART,
    QUERY_CHART,
    CurrentImageSupport,
    ImageLogMeasure,
    ObjectImageReadSource,
)
from clearvla.vision.entity_history import ObservedEntityHistory
from clearvla.vision.source_time import displacement_rate, validate_reference_steps

from ..annotation_goal import AnnotatedGoalEvidence, AnnotatedGoalValues
from ..future_time import LEGACY_FUTURE_TIME, resolve_future_time
from ..instruction_change import InstructionChangeEvidence
from ..instruction_posterior import InstructionChangeValues
from ..manifest import INTERVALS
from ..operation_expectation import OperationExpectation
from ..transition_condition import SUMMED_TRANSITION, validate_transition_condition_mode
from ..world_control import CandidateControlDomain
from ..world_robot import RobotWorldObservation
from .target_binding import TargetBinding, TargetEvidence

INTERVAL_NAMES = ("h4_8", "h8_16", "h16_32", "h32_48")
INTERVAL_BOUNDS = INTERVALS


def _shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")


def _camera_weighted_mean(
    value: Tensor,
    camera_validity: Tensor,
    camera_support: Tensor,
) -> Tensor:
    """Reduce a camera axis without allowing unsupported values to leak.

    Camera coordinates/transport are producer-owned evidence.  A malformed
    value in a zero-validity camera must not survive as ``NaN * 0`` and poison
    the compact object belief (or its backward graph).  Finite, supported
    values retain the historical weighted-mean arithmetic exactly.
    """

    weight = camera_validity.float() * camera_support.float()
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    value_f = value.float()
    safe_value = torch.where(
        weight > 0.0,
        value_f,
        torch.zeros_like(value_f),
    )
    numerator = (safe_value * weight).sum(dim=2)
    denominator = weight.sum(dim=2).clamp_min(1e-6)
    return numerator / denominator


def intersect_object_camera_validity(
    object_validity: Tensor,
    camera_validity: Tensor,
) -> Tensor:
    """Return the detached legal camera support for an object loss boundary.

    ``camera_validity`` is normally produced by the same K-to-chart read as
    ``object_validity``, so the implication ``camera => object`` already holds
    on the healthy Grounder path.  Keep the implication explicit at the
    training seam as well: a replaced or malformed ``ObjectFactSet`` must not
    make an object-invalid camera row supervise the Teacher/recognizer.
    The result intentionally remains FP32 and detached because it is a
    producer-owned support mask, not a learnable confidence.
    """

    if object_validity.ndim != 3 or tuple(object_validity.shape[-1:]) != (1,):
        raise ValueError("object validity must be [B,K,1]")
    if camera_validity.ndim != 4 or tuple(camera_validity.shape[-1:]) != (1,):
        raise ValueError("camera validity must be [B,K,C,1]")
    if tuple(camera_validity.shape[:2]) != tuple(object_validity.shape[:2]):
        raise ValueError("object and camera validity must share [B,K]")
    object_support = torch.nan_to_num(
        object_validity.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    camera_support = torch.nan_to_num(
        camera_validity.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    return camera_support * object_support[:, :, None, :]


@dataclass(frozen=True)
class FlowStepContext:
    """Numerical context for one bottom velocity evaluation.

    The action bottom predicts an instantaneous physical velocity.  The outer
    integrator owns the Euler update, but a non-uniform grid still needs to be
    visible to the bottom so a checkpoint trained for that contract can model
    interval-dependent numerical error.  This record carries only numerical
    quantities; it is deliberately incapable of carrying language, object
    identity or outlet/task metadata.

    ``normalized_index`` is the left-node ``k/N`` (not an integer index), and
    ``endpoint`` is zero for an integrating node and one for the separate
    non-updating ``t=1`` head read.
    """

    time: Tensor  # [B], continuous flow time t
    step_size: Tensor  # [B], Euler delta-t (zero for endpoint read)
    normalized_index: Tensor  # [B], k/N
    endpoint: Tensor  # [B], 0 or 1

    @property
    def delta_t(self) -> Tensor:
        """Explicit alias used by numerical-solver callers."""

        return self.step_size

    @property
    def step_index(self) -> Tensor:
        """Compatibility alias for the normalized ``k/N`` coordinate."""

        return self.normalized_index

    @classmethod
    def from_step(
        cls,
        time: Tensor,
        *,
        step_size: float | Tensor,
        step_index: float | Tensor,
        endpoint: bool | float | Tensor = False,
    ) -> "FlowStepContext":
        """Create a batch-aligned context from scalar scheduler metadata."""

        if time.ndim != 1:
            raise ValueError("flow-step time must be [B]")
        batch = int(time.shape[0])

        def _batch_value(value: float | Tensor, *, name: str) -> Tensor:
            if isinstance(value, Tensor):
                if value.ndim == 0:
                    return value.to(device=time.device, dtype=time.dtype).expand(batch)
                if tuple(value.shape) != (batch,):
                    raise ValueError(f"flow-step {name} tensor must be [B]")
                return value.to(device=time.device, dtype=time.dtype)
            return time.new_full((batch,), float(value))

        result = cls(
            time=time,
            step_size=_batch_value(step_size, name="step_size"),
            normalized_index=_batch_value(step_index, name="step_index"),
            endpoint=_batch_value(endpoint, name="endpoint"),
        )
        result.validate(batch=batch, device=time.device)
        return result

    def validate(
        self,
        *,
        batch: int | None = None,
        device: torch.device | None = None,
        time: Tensor | None = None,
        strict: bool = True,
    ) -> None:
        """Validate the narrow numerical boundary before entering the bottom.

        ``strict=True`` is the fail-closed constructor/diagnostic check.  The
        sampler already validates its serialized schedule on the host and
        constructs every node from those finite scalar boundaries.  Internal
        calls in the ODE hot path therefore use ``strict=False``: shape,
        dtype and device checks remain active, while tensor-to-Python
        finite/range/equality checks (which synchronize CUDA) are deferred to
        the external boundary.  This keeps numerical validation from becoming
        an accidental per-node performance bottleneck.
        """

        values = {
            "time": self.time,
            "step_size": self.step_size,
            "normalized_index": self.normalized_index,
            "endpoint": self.endpoint,
        }
        for name, value in values.items():
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"flow-step {name} must be a rank-1 tensor")
            if not value.is_floating_point():
                raise TypeError(f"flow-step {name} must be floating point")
        inferred_batch = int(self.time.shape[0])
        expected_batch = inferred_batch if batch is None else int(batch)
        if inferred_batch != expected_batch:
            raise ValueError("flow-step context is not aligned with the bottom batch")
        for name, value in values.items():
            if int(value.shape[0]) != expected_batch:
                raise ValueError(f"flow-step {name} batch is not aligned")
            if device is not None and value.device != device:
                raise ValueError(f"flow-step {name} is on a different device")
        if strict:
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"flow-step {name} contains non-finite values")
        if time is not None:
            if tuple(time.shape) != (expected_batch,):
                raise ValueError("flow-step reference time must be [B]")
            if time.device != self.time.device:
                raise ValueError("flow-step reference time is on a different device")
            if strict:
                # Do not use an approximate comparison here: the context is
                # built from the exact tensor handed to the decoder, and a
                # mismatch is a caller bug that would make the numerical
                # contract ambiguous.
                if not torch.equal(self.time, time):
                    raise ValueError("flow-step context time differs from decoder time")
        if strict:
            if bool((self.step_size < 0.0).any()):
                raise ValueError("flow-step step_size must be non-negative")
            if bool(((self.time < 0.0) | (self.time > 1.0)).any()):
                raise ValueError("flow-step time must lie in [0,1]")
            if bool(((self.normalized_index < 0.0) | (self.normalized_index > 1.0)).any()):
                raise ValueError("flow-step normalized_index must lie in [0,1]")
            if bool(((self.endpoint != 0.0) & (self.endpoint != 1.0)).any()):
                raise ValueError("flow-step endpoint must be binary 0 or 1")

    def embedding(self, hidden: int, *, dtype: torch.dtype | None = None) -> Tensor:
        """Return a fixed, bounded Fourier code for the numerical tuple.

        There are intentionally no parameters or buffers here.  The legacy
        uniform/no-context path therefore keeps its state-dict and exact
        arithmetic unchanged, while a context-aware run can learn to use this
        code through the existing bottom condition projections.
        """

        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError("flow-step embedding width must be positive")
        values = torch.stack(
            (
                self.time.float(),
                self.step_size.float(),
                self.normalized_index.float(),
                self.endpoint.float(),
            ),
            dim=-1,
        )
        # Four scalar lanes, each with sine/cosine pairs.  ceil(hidden/8)
        # keeps the output exactly ``hidden`` after truncation/padding for both
        # the production width and the tiny structural-test widths.
        half = max((hidden + 7) // 8, 1)
        frequency = torch.exp(
            -torch.log(values.new_tensor(10000.0))
            * torch.arange(half, device=values.device, dtype=values.dtype)
            / float(max(half - 1, 1))
        )
        phase = values[..., None] * frequency
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        if int(encoded.shape[-1]) < hidden:
            encoded = F.pad(encoded, (0, hidden - int(encoded.shape[-1])))
        encoded = encoded[..., :hidden]
        # A fixed small scale keeps an untrained custom-schedule diagnostic from
        # overwhelming the learned time condition.  It remains nonzero so the
        # path is observable and trainable through downstream weights.
        encoded = 0.05 * encoded
        return encoded.to(dtype=dtype or self.time.dtype)


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
    latest_flow_steps: Tensor | None = None  # [B], zero means no observed temporal pair
    semantic_owner_log_probs: Tensor | None = None  # finite FP32 [B,C,Y,X,M]
    appearance_owner_log_probs: Tensor | None = None
    geometry_owner_log_probs: Tensor | None = None
    context_slots: Tensor | None = None  # completed current G3 at the same support [B,C,Y,X,M,H]
    current_image_support: CurrentImageSupport | None = None
    observed_history: ObservedEntityHistory | None = None

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
        if self.latest_flow_steps is not None:
            validate_reference_steps(self.latest_flow_steps, batch=self.batch, device=self.content_slots.device)
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
        if self.current_image_support is not None:
            self.current_image_support.validate(self.slot_validity)
        if self.observed_history is not None:
            self.observed_history.validate()
            if self.current_image_support is None:
                raise ValueError("causal entity history needs actual current-image support")
            history_shape = self.observed_history.content.shape
            if (history_shape[0], *history_shape[2:5], history_shape[-1]) != tuple(self.target_dino_content.shape):
                raise ValueError("entity history and current observed chart axes differ")
            history = self.observed_history
            if history.content.device != self.content_slots.device:
                raise ValueError("entity history and current facts must share their source device")
            if self.latest_flow_steps is None or not torch.equal(
                self.latest_flow_steps, history.source_time.pair_steps[:, -1]
            ):
                raise ValueError("entity history and current flow durations disagree")
            if not torch.equal(history.observed[:, -1], self.cell_observed[..., 0]):
                raise ValueError("entity history and current observation support disagree")
            current = history.content[:, -1].to(dtype=self.target_dino_content.dtype)
            if not torch.equal(
                current.masked_select(self.cell_observed),
                self.target_dino_content.masked_select(self.cell_observed),
            ):
                raise ValueError("entity history and current observed content disagree")
        if self.context_slots is not None:
            _shape(self.context_slots, (*prefix, int(self.public_scene_base.shape[-1])), "local completed G3 context")
            if self.context_slots.device != self.content_slots.device:
                raise ValueError("local G3 context must share its candidate device")
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
    candidate_context: Tensor | None = None  # learned completed G3, never a reconstruction target
    current_image_support: CurrentImageSupport | None = None

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
        if self.current_image_support is not None:
            self.current_image_support.validate(self.candidate_validity)
        if self.candidate_context is not None:
            _shape(self.candidate_context, (*prefix, int(self.public_scene_base.shape[-1])), "dense completed G3 context")
            if self.candidate_context.device != self.candidate_content.device:
                raise ValueError("dense G3 context must share candidate device")
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
    latest_flow_steps: Tensor | None = None  # [B], producer time, not model confidence
    object_chart_mode: str = QUERY_CHART
    current_image_measure: ImageLogMeasure | None = None
    current_image_source: ObjectImageReadSource | None = None

    @property
    def batch(self) -> int:
        return int(self.content.shape[0])

    @property
    def objects(self) -> int:
        return int(self.content.shape[1])

    @property
    def coordinates(self) -> Tensor:
        """V120 object coordinate reduced only over physical camera support."""

        return _camera_weighted_mean(
            self.camera_coordinates,
            self.camera_validity,
            self.camera_support,
        )

    @property
    def camera_transport_rate(self) -> Tensor:
        if self.latest_flow_steps is None:
            raise ValueError("observed motion rate requires explicit source duration")
        return displacement_rate(self.camera_transport_prior, self.latest_flow_steps)

    @property
    def transport_rate(self) -> Tensor:
        return _camera_weighted_mean(self.camera_transport_rate, self.camera_validity, self.camera_support)

    @property
    def transport_prior(self) -> Tensor:
        """V120 object transport prior reduced only over valid cameras."""

        return _camera_weighted_mean(
            self.camera_transport_prior,
            self.camera_validity,
            self.camera_support,
        )

    def validate(self) -> None:
        if self.latest_flow_steps is not None:
            validate_reference_steps(self.latest_flow_steps, batch=self.batch, device=self.content.device)
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
        if self.object_chart_mode not in {QUERY_CHART, CURRENT_IMAGE_CHART}:
            raise ValueError("unknown object-to-chart coordinate semantics")
        if (self.object_chart_mode == CURRENT_IMAGE_CHART) != (self.dense_chart.current_image_support is not None):
            raise ValueError("object chart mode and actual support disagree")
        if (self.object_chart_mode == CURRENT_IMAGE_CHART) != (self.current_image_measure is not None):
            raise ValueError("object chart mode and authoritative image measure disagree")
        if self.current_image_measure is not None:
            if self.current_image_measure.log_mass.shape != self.object_to_chart.shape:
                raise ValueError("object image measure has different axes")
        if self.current_image_source is not None:
            src=self.current_image_source
            if src.spatial is not self.dense_chart.current_image_support or self.current_image_measure is None:
                raise ValueError("G3 image source lost the actual spatial owner")
            if src.log_measure.shape != self.candidate_assignment.shape or src.supported.shape != src.log_measure.shape:
                raise ValueError("G3 image source lost actual local candidate axes")
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
            current_image_source=None if self.current_image_source is None else self.current_image_source.permute(index),
            object_chart_mode=self.object_chart_mode,
            current_image_measure=(
                ImageLogMeasure(self.current_image_measure.log_mass[:, index], self.current_image_measure.supported[:, index])
                if self.current_image_measure is not None else None
            ),
            content=self.content[:, index],
            semantic=self.semantic[:, index],
            appearance=self.appearance[:, index],
            geometry=self.geometry[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            latest_flow_steps=self.latest_flow_steps,
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

    def world_belief(self, *, robot_observation: RobotWorldObservation | None = None) -> "ObjectWorldBelief":
        """Export only the current object evidence needed by a W rerun.

        Deployment may perform one outer action-world refinement.  Retaining
        the complete dense chart in that cache would silently turn a static
        object belief into a second training/source graph, so this boundary
        deliberately exports the compact G result consumed by W only.
        """

        self.validate()
        return ObjectWorldBelief(
            robot_observation=robot_observation,
            content=self.content,
            semantic=self.semantic,
            appearance=self.appearance,
            geometry=self.geometry,
            camera_coordinates=self.camera_coordinates,
            latest_flow_steps=self.latest_flow_steps,
            camera_transport_prior=self.camera_transport_prior,
            camera_support=self.camera_support,
            camera_validity=self.camera_validity,
            log_camera_validity=self.log_camera_validity,
            validity=self.validity,
            log_validity=self.log_validity,
        )


@dataclass(frozen=True)
class ObjectWorldBelief:
    """Compact current object belief retained for an outer W refinement.

    This is a single-observation belief view, not a persistent cross-cycle
    tracker.  It intentionally excludes dense source charts, reconstruction
    targets and goal/intent values; those owners remain outside the W rerun.
    """

    content: Tensor  # [B,K,D]
    semantic: Tensor  # [B,K,R]
    appearance: Tensor  # [B,K,R]
    geometry: Tensor  # [B,K,R]
    camera_coordinates: Tensor  # [B,K,C,2]
    camera_transport_prior: Tensor  # [B,K,C,2]
    camera_support: Tensor  # [B,K,C,1]
    camera_validity: Tensor  # [B,K,C,1]
    log_camera_validity: Tensor  # [B,K,C,1]
    validity: Tensor  # [B,K,1]
    log_validity: Tensor  # [B,K,1]
    latest_flow_steps: Tensor | None = None
    robot_observation: RobotWorldObservation | None = None

    @property
    def batch(self) -> int:
        return int(self.content.shape[0])

    @property
    def objects(self) -> int:
        return int(self.content.shape[1])

    @property
    def camera_transport_rate(self) -> Tensor:
        if self.latest_flow_steps is None:
            raise ValueError("observed motion rate requires explicit source duration")
        return displacement_rate(self.camera_transport_prior, self.latest_flow_steps)

    @property
    def transport_rate(self) -> Tensor:
        return _camera_weighted_mean(self.camera_transport_rate, self.camera_validity, self.camera_support)

    @property
    def transport_prior(self) -> Tensor:
        return _camera_weighted_mean(
            self.camera_transport_prior,
            self.camera_validity,
            self.camera_support,
        )

    def validate(self) -> None:
        if self.robot_observation is not None:
            self.robot_observation.validate(batch=self.batch, device=self.content.device)
        if self.latest_flow_steps is not None:
            validate_reference_steps(self.latest_flow_steps, batch=self.batch, device=self.content.device)
        if self.content.ndim != 3:
            raise ValueError("world belief content must be [B,K,D]")
        batch, objects = self.content.shape[:2]
        for name in ("semantic", "appearance", "geometry"):
            value = getattr(self, name)
            if value.ndim != 3 or tuple(value.shape[:2]) != (batch, objects):
                raise ValueError(f"world belief {name} must be [B,K,*]")
        if self.camera_coordinates.ndim != 4:
            raise ValueError("world belief camera coordinates must be [B,K,C,2]")
        cameras = int(self.camera_coordinates.shape[2])
        _shape(
            self.camera_coordinates,
            (batch, objects, cameras, 2),
            "world belief camera coordinates",
        )
        for name in ("camera_transport_prior", "camera_support", "camera_validity", "log_camera_validity"):
            value = getattr(self, name)
            expected_width = 2 if name == "camera_transport_prior" else 1
            _shape(
                value,
                (batch, objects, cameras, expected_width),
                f"world belief {name}",
            )
        _shape(self.validity, (batch, objects, 1), "world belief validity")
        _shape(self.log_validity, (batch, objects, 1), "world belief log validity")
        if self.camera_validity.dtype != torch.float32:
            raise TypeError("world belief camera validity must remain FP32")
        if self.log_camera_validity.dtype != torch.float32:
            raise TypeError("world belief camera log validity must remain FP32")
        if self.validity.dtype != torch.float32:
            raise TypeError("world belief validity must remain FP32")
        if self.log_validity.dtype != torch.float32:
            raise TypeError("world belief log validity must remain FP32")

    def permute(self, permutation: Tensor) -> "ObjectWorldBelief":
        self.validate()
        if permutation.ndim != 1 or int(permutation.numel()) != self.objects:
            raise ValueError("world-belief permutation must contain every object")
        index = permutation.to(device=self.content.device, dtype=torch.long)
        return ObjectWorldBelief(
            robot_observation=self.robot_observation,
            content=self.content[:, index],
            semantic=self.semantic[:, index],
            appearance=self.appearance[:, index],
            geometry=self.geometry[:, index],
            camera_coordinates=self.camera_coordinates[:, index],
            latest_flow_steps=self.latest_flow_steps,
            camera_transport_prior=self.camera_transport_prior[:, index],
            camera_support=self.camera_support[:, index],
            camera_validity=self.camera_validity[:, index],
            log_camera_validity=self.log_camera_validity[:, index],
            validity=self.validity[:, index],
            log_validity=self.log_validity[:, index],
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
    # Producer-owned physical validity is metadata, not a learned selector.
    # Carrying it beside the materialized K memory lets the coarse reader
    # mask padded/invalid rows without reopening ObjectFactSet or creating a
    # second object-binding path.
    public_object_validity: Tensor | None = None  # FP32 [B,K,1]
    history_validity: Tensor | None = None  # bool [B,L], source padding only
    target_binding: TargetBinding | None = None
    target_evidence: TargetEvidence | None = None

    time_grid_mode: str = LEGACY_FUTURE_TIME

    def validate(self, *, hidden: int) -> None:
        resolve_future_time(self.time_grid_mode)
        batch = int(self.public_interval_carrier.shape[0])
        _shape(
            self.public_interval_carrier,
            (batch, 4, hidden),
            "action-intent public interval carrier",
        )
        if self.history_memory.ndim != 3 or int(self.history_memory.shape[0]) != batch:
            raise ValueError("action-intent history memory must be [B,L,H]")
        if self.history_validity is not None:
            if (
                tuple(self.history_validity.shape) != tuple(self.history_memory.shape[:2])
                or self.history_validity.dtype != torch.bool
                or self.history_validity.device != self.history_memory.device
            ):
                raise ValueError("intent history validity must be boolean [B,L] on memory device")
        if self.public_object_memory.ndim != 3 or int(
            self.public_object_memory.shape[0]
        ) != batch:
            raise ValueError("action-intent object memory must be [B,K,H]")
        if self.public_object_validity is not None:
            if self.public_object_validity.ndim != 3 or tuple(
                self.public_object_validity.shape[:2]
            ) != tuple(self.public_object_memory.shape[:2]) or int(
                self.public_object_validity.shape[-1]
            ) != 1:
                raise ValueError("action-intent object validity must be [B,K,1]")
            if self.public_object_validity.dtype != torch.float32:
                raise TypeError("action-intent object validity must remain FP32")
            if self.public_object_validity.device != self.public_object_memory.device:
                raise ValueError(
                    "action-intent object validity must share object-memory device"
                )


@dataclass(frozen=True)
class FactualIntentDock:
    """Read-only S context for the unchanged V120 P1 factual reader."""

    phase_context: Tensor  # [B,I,H]
    condition_query_context: Tensor  # [B,I,H]
    history_query_context: Tensor  # [B,I,H]
    target_binding: TargetBinding | None = None
    target_evidence: TargetEvidence | None = None

    time_grid_mode: str = LEGACY_FUTURE_TIME

    def validate(self, *, hidden: int) -> None:
        resolve_future_time(self.time_grid_mode)
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
    target_object_address_logit: Tensor  # FP32 [B,I,K]
    typed_common_value: Tensor  # [B,K,3,R]
    typed_interval_residual_value: Tensor  # [B,I,K,3,R]
    target_binding: TargetBinding | None = None
    instruction_change: InstructionChangeEvidence | None = None
    instruction_plan_values: InstructionChangeValues | None = None
    annotated_goal: AnnotatedGoalEvidence | None = None
    annotated_goal_values: AnnotatedGoalValues | None = None
    operation_expectation: OperationExpectation | None = None

    time_grid_mode: str = LEGACY_FUTURE_TIME

    def validate(self, *, horizon: int, hidden: int) -> None:
        resolve_future_time(self.time_grid_mode)
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
        if self.annotated_goal is not None:
            self.annotated_goal.validate()
            if self.annotated_goal.instruction_change is not self.instruction_change:
                raise ValueError("goal belongs to another causal observation/reference")
        if self.annotated_goal_values is not None:
            self.annotated_goal_values.validate(hidden=hidden)
            if self.annotated_goal_values.evidence is not self.annotated_goal:
                raise ValueError("prepared goal belongs to another goal evidence")
        if self.operation_expectation is not None:
            self.operation_expectation.validate()
            if self.operation_expectation.binding is not self.target_binding:
                raise ValueError("operation expectation cannot reselect the target")
            if self.operation_expectation.time_grid_mode != self.time_grid_mode:
                raise ValueError("operation expectation/intent time chart mismatch")
        if self.instruction_plan_values is not None:
            self.instruction_plan_values.validate(hidden=hidden)
            if self.instruction_plan_values.evidence is not self.instruction_change:
                raise ValueError("prepared instruction read belongs to another evidence source")
        if self.instruction_change is not None:
            self.instruction_change.validate()
            if self.instruction_change.binding is not self.target_binding:
                raise ValueError("policy instruction change cannot reselect the target")
        if int(self.typed_common_value.shape[2]) != 3:
            raise ValueError("policy-intent typed common value lost type identity")
        _shape(
            self.target_object_address_logit,
            (batch, 4, objects),
            "policy-intent target object address logit",
        )
        if self.target_object_address_logit.dtype != torch.float32:
            raise TypeError("policy-intent target object address logit must remain FP32")
        if self.target_object_address_logit.device != self.interval_key.device:
            raise ValueError(
                "policy-intent target object address logit must share intent device"
            )
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
    target_object_address_logit: Tensor  # FP32 [B,4,K]
    typed_common_mass: Tensor  # [B,K,3,1]
    typed_common_value: Tensor  # [B,K,3,R]
    typed_interval_residual_mass: Tensor  # [B,4,K,3,1]
    typed_interval_residual_value: Tensor  # [B,4,K,3,R]
    typed_policy_components: Tensor  # [B,4,3,H]
    goal_attention: Tensor  # [B,4,Lg]
    interval_goal_attention: Tensor  # [B,4,4]
    interval_history_attention: Tensor  # [B,4,L]
    interval_object_attention: Tensor  # [B,4,K]
    # The object memory is always accompanied by the producer-owned physical
    # validity mask.  It is optional only for old in-memory fixtures that
    # construct this compatibility container by hand.
    object_validity: Tensor | None = None  # FP32 [B,K,1]
    history_validity: Tensor | None = None  # bool [B,L]
    target_binding: TargetBinding | None = None
    target_evidence: TargetEvidence | None = None
    instruction_change: InstructionChangeEvidence | None = None
    instruction_plan_values: InstructionChangeValues | None = None
    annotated_goal: AnnotatedGoalEvidence | None = None
    annotated_goal_values: AnnotatedGoalValues | None = None
    operation_expectation: OperationExpectation | None = None

    time_grid_mode: str = LEGACY_FUTURE_TIME

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
            time_grid_mode=self.time_grid_mode,
            public_interval_carrier=self.public_interval_carrier,
            history_memory=self.history_tokens,
            public_object_memory=self.object_tokens,
            public_object_validity=self.object_validity,
            target_binding=self.target_binding,
            target_evidence=self.target_evidence,
            history_validity=self.history_validity,
        )

    def factual_dock(self) -> FactualIntentDock:
        batch = int(self.policy_interval_context.shape[0])
        return FactualIntentDock(
            time_grid_mode=self.time_grid_mode,
            phase_context=self.policy_interval_context,
            target_binding=self.target_binding,
            target_evidence=self.target_evidence,
            condition_query_context=self.protected_goal_set.mean(dim=1)[:, None].expand(
                -1, 4, -1
            ),
            history_query_context=self.history_tokens[:, -1:, :].expand(
                batch, 4, -1
            ),
        )

    def policy_dock(self) -> PolicyIntentDock:
        return PolicyIntentDock(
            time_grid_mode=self.time_grid_mode,
            interval_key=self.policy_interval_context,
            temporal_control=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            instruction_change=self.instruction_change,
            instruction_plan_values=self.instruction_plan_values,
            operation_expectation=self.operation_expectation,
            annotated_goal=self.annotated_goal,
            annotated_goal_values=self.annotated_goal_values,
            target_object_address_logit=self.target_object_address_logit,
            typed_common_value=self.typed_common_value,
            target_binding=self.target_binding,
            typed_interval_residual_value=self.typed_interval_residual_value,
        )

    def validate(self, *, horizon: int, hidden: int) -> None:
        resolve_future_time(self.time_grid_mode)
        batch = int(self.public_interval_carrier.shape[0])
        _shape(self.protected_goal_set, (batch, 4, hidden), "protected goal set")
        if self.annotated_goal is not None:
            self.annotated_goal.validate()
            if self.annotated_goal.instruction_change is not self.instruction_change:
                raise ValueError("goal belongs to another causal observation/reference")
        if self.annotated_goal_values is not None:
            self.annotated_goal_values.validate(hidden=hidden)
            if self.annotated_goal_values.evidence is not self.annotated_goal:
                raise ValueError("prepared goal belongs to another goal evidence")
        if self.operation_expectation is not None:
            self.operation_expectation.validate()
            if self.operation_expectation.binding is not self.target_binding:
                raise ValueError("operation expectation cannot reselect the target")
            if self.operation_expectation.time_grid_mode != self.time_grid_mode:
                raise ValueError("operation expectation/intent time chart mismatch")
        if self.instruction_plan_values is not None:
            self.instruction_plan_values.validate(hidden=hidden)
            if self.instruction_plan_values.evidence is not self.instruction_change:
                raise ValueError("prepared instruction read belongs to another evidence source")
        if self.instruction_change is not None:
            self.instruction_change.validate()
            if self.instruction_change.binding is not self.target_binding:
                raise ValueError("instruction change must share the current target law")
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
        if self.object_validity is not None:
            _shape(
                self.object_validity,
                (batch, objects, 1),
                "intent object validity",
            )
            if self.object_validity.dtype != torch.float32:
                raise TypeError("intent object validity must remain FP32")
            if self.object_validity.device != self.object_tokens.device:
                raise ValueError(
                    "intent object validity must share object-token device"
                )
        _shape(
            self.target_object_address_logit,
            (batch, 4, objects),
            "target object address logit",
        )
        if self.target_object_address_logit.dtype != torch.float32:
            raise TypeError("target object address logit must remain FP32")
        if self.target_object_address_logit.device != self.object_tokens.device:
            raise ValueError("target object address logit must share object-token device")
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
        if (self.target_binding is None) != (self.target_evidence is None):
            raise ValueError("target binding and current evidence must be provided together")
        if self.target_binding is not None and self.target_evidence is not None:
            self.target_evidence.validate(hidden=hidden)
            self.target_binding.validate(batch=batch, objects=objects, device=self.object_tokens.device)
            if not torch.equal(self.target_binding.supported, self.target_evidence.valid.any(-1)):
                raise ValueError("intent target support disagrees with observed evidence")
        self.action_dock().validate(hidden=hidden)
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
        binding = None if self.target_binding is None else self.target_binding.permute(index)
        change = None
        operation = None
        if self.operation_expectation is not None:
            if binding is None:
                raise ValueError("operation expectation permutation lost shared target")
            operation = self.operation_expectation.permute(index, binding)
        if self.instruction_change is not None:
            if binding is None:
                raise ValueError("instruction change cannot lose target law during permutation")
            change = self.instruction_change.permute(index, binding)
        annotated_goal = None
        if self.annotated_goal is not None:
            if change is None:
                raise ValueError("goal permutation lost instruction evidence")
            annotated_goal = self.annotated_goal.permute(change)
        return ObjectIntentState(
            time_grid_mode=self.time_grid_mode,
            protected_goal_set=self.protected_goal_set,
            history_tokens=self.history_tokens,
            history_validity=self.history_validity,
            object_tokens=self.object_tokens[:, index],
            target_binding=binding,
            instruction_change=change,
            instruction_plan_values=(None if self.instruction_plan_values is None or change is None else
                                     self.instruction_plan_values.permute(index,change)),
            operation_expectation=operation,
            annotated_goal=annotated_goal,
            annotated_goal_values=(None if self.annotated_goal_values is None or annotated_goal is None else
                                   self.annotated_goal_values.permute(index, annotated_goal)),
            target_evidence=None if self.target_evidence is None else self.target_evidence.permute(index),
            public_interval_carrier=self.public_interval_carrier,
            policy_interval_context=self.policy_interval_context,
            temporal_queries=self.temporal_queries,
            state_change_evidence=self.state_change_evidence,
            target_object_address_logit=self.target_object_address_logit[:, :, index],
            typed_common_mass=self.typed_common_mass[:, index],
            typed_common_value=self.typed_common_value[:, index],
            typed_interval_residual_mass=self.typed_interval_residual_mass[:, :, index],
            typed_interval_residual_value=self.typed_interval_residual_value[:, :, index],
            typed_policy_components=self.typed_policy_components,
            goal_attention=self.goal_attention,
            interval_goal_attention=self.interval_goal_attention,
            interval_history_attention=self.interval_history_attention,
            interval_object_attention=self.interval_object_attention[:, :, index],
            object_validity=(
                None if self.object_validity is None else self.object_validity[:, index]
            ),
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
    interval_valid: Tensor | None = None

    time_grid_mode: str = LEGACY_FUTURE_TIME

    def validate(self, *, hidden: int) -> None:
        resolve_future_time(self.time_grid_mode)
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
        if self.interval_valid is not None:
            _shape(self.interval_valid, (batch, 4), "recognizer interval support")
            if (
                self.interval_valid.dtype != torch.bool
                or self.interval_valid.device != self.interval_targets.device
            ):
                raise ValueError("recognizer source support must be boolean on the target device")


@dataclass(frozen=True)
class CoarseActionIntentState:
    tokens: Tensor  # [B,4,H]
    action_prediction: Tensor  # [B,4,A]
    target: Tensor | None
    loss: Tensor


@dataclass(frozen=True)
class PhysicalActionCondition:
    """Canonical physical action condition owned by the proposal/world seam.

    W predicts the four configured future intervals, so its physical action
    ABI uses the same four interval rows rather than inventing a 24-row chart
    whose final two intervals would not cover the W target horizons.  The
    local delta is a deterministic view of the canonical interval action.  An
    outlet-specific subtype may retain a separate source proposal when its
    command semantics require a deterministic conversion before W.  No hidden
    proposal coordinate, goal token or S carrier is representable here.
    """

    interval_action: Tensor  # [B,4,A], canonical normalized physical action
    interval_delta: Tensor  # [B,4,A], deterministic canonical physical delta
    current_action: Tensor  # [B,A], canonical row-zero reconstruction boundary

    @classmethod
    def from_interval_action(
        cls,
        interval_action: Tensor,
        current_action: Tensor,
    ) -> "PhysicalActionCondition":
        if interval_action.ndim != 3 or int(interval_action.shape[1]) != 4:
            raise ValueError("physical action condition must have four interval rows")
        batch, _, action_dim = interval_action.shape
        _shape(
            current_action,
            (batch, action_dim),
            "physical action condition current action",
        )
        boundary = torch.cat(
            (
                current_action[:, None].to(
                    device=interval_action.device,
                    dtype=interval_action.dtype,
                ),
                interval_action[:, :-1],
            ),
            dim=1,
        )
        condition = cls(
            interval_action=interval_action,
            interval_delta=interval_action - boundary,
            current_action=current_action,
        )
        condition.validate(action_dim=action_dim)
        return condition

    @classmethod
    def from_horizon_action(
        cls,
        action: Tensor,
        current_action: Tensor,
    ) -> "PhysicalActionCondition":
        """Build the four-row ABI from a decoded 24-row native action.

        The W intervals are defined in the 48-step future chart while the
        deployed action chart has 24 rows.  The same clipped interval windows
        used by the recovered intent target are therefore used here: later
        windows that extend past row 24 are represented by their final
        available row.  This is deterministic and auditable; it is not a new
        learned extrapolator.
        """

        if action.ndim != 3:
            raise ValueError("horizon action must be [B,T,A]")
        if int(action.shape[1]) != 24:
            raise ValueError("horizon action condition requires 24 rows")
        slices: list[slice] = []
        for lower, upper in INTERVAL_BOUNDS:
            start = min(max(int(lower) - 1, 0), int(action.shape[1]) - 1)
            stop = min(max(int(upper), start + 1), int(action.shape[1]))
            slices.append(slice(start, stop))
        interval_action = torch.stack(
            [action[:, row].mean(dim=1) for row in slices],
            dim=1,
        )
        return cls.from_interval_action(interval_action, current_action)

    @property
    def batch(self) -> int:
        return int(self.interval_action.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.interval_action.shape[-1])

    @property
    def fingerprint(self) -> Tensor:
        """Deterministic, lossless in-graph fingerprint of the physical ABI."""

        return torch.cat((self.interval_action, self.interval_delta), dim=-1)

    @property
    def action_fingerprint(self) -> Tensor:
        """Named alias used by the candidate-world boundary."""

        return self.fingerprint

    @property
    def source_interval_action(self) -> Tensor:
        """Return the unchanged proposal for the established Pen/RDT path.

        This is a property rather than a dataclass field so continuous outlets
        retain the exact historical three-field container ABI.
        """

        return self.interval_action

    def reconstructed_interval_action(self) -> Tensor:
        """Reconstruct interval absolutes from the stored adjacent deltas."""

        boundary = torch.cat(
            (
                self.current_action[:, None].to(
                    device=self.interval_action.device,
                    dtype=self.interval_action.dtype,
                ),
                self.interval_action[:, :-1],
            ),
            dim=1,
        )
        return boundary + self.interval_delta

    def validate(self, *, action_dim: int, intervals: int = 4) -> None:
        if self.interval_action.ndim != 3:
            raise ValueError("physical action condition must be [B,I,A]")
        batch = self.batch
        expected = (batch, int(intervals), int(action_dim))
        _shape(
            self.interval_action,
            expected,
            "physical action condition interval action",
        )
        _shape(
            self.interval_delta,
            expected,
            "physical action condition interval delta",
        )
        _shape(
            self.current_action,
            (batch, int(action_dim)),
            "physical action condition current action",
        )
        if not (
            self.interval_action.device
            == self.interval_delta.device
            == self.current_action.device
        ):
            raise ValueError("physical action condition tensors must share a device")
        if not (
            self.interval_action.is_floating_point()
            and self.interval_delta.is_floating_point()
            and self.current_action.is_floating_point()
        ):
            raise TypeError("physical action condition must be floating point")
        expected_boundary = torch.cat(
            (
                self.current_action[:, None].to(
                    device=self.interval_action.device,
                    dtype=self.interval_action.dtype,
                ),
                self.interval_action[:, :-1],
            ),
            dim=1,
        )
        # This metadata validator is used on live CUDA paths; shape/device are
        # checked here while exact deterministic reconstruction is guarded by
        # focused CPU/CUDA contract tests to avoid a deployment sync.
        if tuple(expected_boundary.shape) != tuple(self.interval_delta.shape):
            raise ValueError("physical action condition delta boundary is invalid")

    def assert_exact_reconstruction(self) -> None:
        """Check the deterministic delta identity on an explicit audit call.

        This is intentionally separate from the hot-path metadata validator:
        comparing CUDA tensor values here would introduce a synchronization on
        every deployment node.
        """

        reconstructed = self.reconstructed_interval_action()
        if not torch.equal(reconstructed, self.interval_action):
            raise ValueError("physical action condition delta reconstruction failed")


@dataclass(frozen=True)
class OutletPhysicalActionCondition(PhysicalActionCondition):
    """Relative-outlet W condition with an auditable source proposal.

    Relative-command outlets convert arm commands before W consumes them, but
    the exact native proposal remains tagged beside the canonical cumulative
    condition. Pen and RDT continue to use the unchanged three-field
    :class:`PhysicalActionCondition` container.
    """

    source_action: Tensor  # [B,4,A], exact outlet-native proposal tensor

    @property
    def source_interval_action(self) -> Tensor:
        return self.source_action

    def validate(self, *, action_dim: int, intervals: int = 4) -> None:
        super().validate(action_dim=action_dim, intervals=intervals)
        expected = (self.batch, int(intervals), int(action_dim))
        _shape(
            self.source_action,
            expected,
            "outlet physical action condition source action",
        )
        if self.source_action.device != self.interval_action.device:
            raise ValueError("outlet source action must share the canonical device")
        if not self.source_action.is_floating_point():
            raise TypeError("outlet source action must be floating point")

    def assert_exact_reconstruction(self) -> None:
        """Audit relative-outlet integration without weakening the shared gate."""

        if not torch.allclose(
            self.reconstructed_interval_action(),
            self.interval_action,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("outlet physical action delta reconstruction failed")


PHYSICAL_ACTION_SEQUENCE_SCHEMA = "physical_action_sequence_prefix_v1"
ABSOLUTE_ACTION_SEQUENCE_CHART = "absolute_action_sequence_v1"
RELATIVE_COMMAND_SEQUENCE_CHART = "relative_command_sequence_v1"
_ACTION_SEQUENCE_CHART_CODES = {
    ABSOLUTE_ACTION_SEQUENCE_CHART: 1.0,
    RELATIVE_COMMAND_SEQUENCE_CHART: 2.0,
}


def physical_action_normalizer_fingerprint(
    offset: Tensor,
    scale: Tensor,
) -> str:
    """Hash exactly the affine chart values used by the outlet adapter."""

    offset_values = tuple(
        float(value)
        for value in offset.detach().float().cpu().reshape(-1).tolist()
    )
    scale_values = tuple(
        float(value)
        for value in scale.detach().float().cpu().reshape(-1).tolist()
    )
    encoded = json.dumps(
        {"offset": offset_values, "scale": scale_values},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PhysicalActionSequenceCondition:
    """Versioned 24-row physical action condition consumed by W.

    ``source_action`` is the normalized outlet-native proposal and remains the
    sole producer-owned value.  ``canonical_value`` and ``canonical_delta``
    are deterministic outlet-adapter views.  For an absolute action chart they
    are the action and its adjacent difference.  For a relative-command chart
    the arm value is the prefix command sum while the arm delta is the centered
    command itself; the gripper lane remains an ordinary value/difference.

    Row time is control-step time, not flow-ODE time.  Version one deliberately
    accepts exactly 24 contiguous known steps and does not imply controls for
    W's 32--48 future interval.
    """

    source_action: Tensor  # [B,24,A], normalized outlet-native proposal
    canonical_value: Tensor  # [B,24,A], adapter-owned physical sequence value
    canonical_delta: Tensor  # [B,24,A], adapter-owned adjacent/command delta
    current_action: Tensor  # [B,A], canonical row-zero boundary
    row_start: Tensor  # [B,24,1], inclusive control-step start
    row_end: Tensor  # [B,24,1], exclusive control-step end
    normalizer_offset: Tensor  # FP32 [B,1,A], normalized value of native zero
    normalizer_scale: Tensor  # FP32 [B,1,A], native-to-normalized affine scale
    outlet_profile: str
    chart_version: str
    normalizer_fingerprint: str
    arm_dim: int
    schema: str = PHYSICAL_ACTION_SEQUENCE_SCHEMA
    known_prefix_end: int = 24

    @classmethod
    def from_absolute_action(
        cls,
        action: Tensor,
        current_action: Tensor,
        *,
        outlet_profile: str,
        normalizer_offset: Tensor,
        normalizer_scale: Tensor,
        normalizer_fingerprint: str | None = None,
    ) -> "PhysicalActionSequenceCondition":
        """Build the exact absolute-chart sequence without interval pooling."""

        if action.ndim != 3 or int(action.shape[1]) != 24:
            raise ValueError("physical action sequence must be [B,24,A]")
        batch, horizon, action_dim = action.shape
        _shape(
            current_action,
            (batch, action_dim),
            "physical action sequence current action",
        )
        boundary = torch.cat(
            (
                current_action[:, None].to(
                    device=action.device,
                    dtype=action.dtype,
                ),
                action[:, :-1],
            ),
            dim=1,
        )
        row_start = torch.arange(
            horizon,
            device=action.device,
            dtype=action.dtype,
        ).reshape(1, horizon, 1).expand(batch, -1, -1)
        offset, scale = cls._expanded_normalizer_chart(
            action,
            normalizer_offset,
            normalizer_scale,
        )
        condition = cls(
            source_action=action,
            canonical_value=action,
            canonical_delta=action - boundary,
            current_action=current_action,
            row_start=row_start,
            row_end=row_start + 1,
            normalizer_offset=offset,
            normalizer_scale=scale,
            outlet_profile=str(outlet_profile),
            chart_version=ABSOLUTE_ACTION_SEQUENCE_CHART,
            normalizer_fingerprint=(
                physical_action_normalizer_fingerprint(offset[0], scale[0])
                if normalizer_fingerprint is None
                else str(normalizer_fingerprint)
            ),
            arm_dim=action_dim - 1,
        )
        condition.validate(action_dim=action_dim)
        return condition

    @classmethod
    def from_relative_command(
        cls,
        action: Tensor,
        current_action: Tensor,
        *,
        outlet_profile: str,
        arm_dim: int,
        normalizer_offset: Tensor,
        normalizer_scale: Tensor,
        normalizer_fingerprint: str | None = None,
    ) -> "PhysicalActionSequenceCondition":
        """Build the deterministic centered-command/cumulative sequence."""

        if action.ndim != 3 or int(action.shape[1]) != 24:
            raise ValueError("physical action sequence must be [B,24,A]")
        batch, horizon, action_dim = action.shape
        if not 0 < int(arm_dim) < action_dim:
            raise ValueError("relative action sequence arm width is invalid")
        _shape(
            current_action,
            (batch, action_dim),
            "physical action sequence current action",
        )
        offset, scale = cls._expanded_normalizer_chart(
            action,
            normalizer_offset,
            normalizer_scale,
        )
        source_current = current_action.to(device=action.device, dtype=action.dtype)
        # Metadata retains the exact FP32 chart; the numerical action view
        # keeps the established arithmetic in the producer's dtype.
        arm_command = action[..., :arm_dim] - offset[..., :arm_dim].to(
            dtype=action.dtype
        )
        canonical_arm = torch.cumsum(arm_command, dim=1)
        gripper = action[..., arm_dim:]
        gripper_boundary = torch.cat(
            (source_current[:, None, arm_dim:], gripper[:, :-1]),
            dim=1,
        )
        canonical_value = torch.cat((canonical_arm, gripper), dim=-1)
        canonical_delta = torch.cat(
            (arm_command, gripper - gripper_boundary),
            dim=-1,
        )
        canonical_current = torch.cat(
            (
                torch.zeros_like(source_current[..., :arm_dim]),
                source_current[..., arm_dim:],
            ),
            dim=-1,
        )
        row_start = torch.arange(
            horizon,
            device=action.device,
            dtype=action.dtype,
        ).reshape(1, horizon, 1).expand(batch, -1, -1)
        condition = cls(
            source_action=action,
            canonical_value=canonical_value,
            canonical_delta=canonical_delta,
            current_action=canonical_current,
            row_start=row_start,
            row_end=row_start + 1,
            normalizer_offset=offset,
            normalizer_scale=scale,
            outlet_profile=str(outlet_profile),
            chart_version=RELATIVE_COMMAND_SEQUENCE_CHART,
            normalizer_fingerprint=(
                physical_action_normalizer_fingerprint(offset[0], scale[0])
                if normalizer_fingerprint is None
                else str(normalizer_fingerprint)
            ),
            arm_dim=int(arm_dim),
        )
        condition.validate(action_dim=action_dim)
        return condition

    @staticmethod
    def _expanded_normalizer_chart(
        action: Tensor,
        offset: Tensor,
        scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, _, action_dim = action.shape

        def expand(value: Tensor, name: str) -> Tensor:
            flattened = value.to(device=action.device, dtype=torch.float32).reshape(
                -1,
                action_dim,
            )
            if int(flattened.shape[0]) not in {1, batch}:
                raise ValueError(f"physical action sequence {name} batch is invalid")
            if int(flattened.shape[0]) == 1:
                flattened = flattened.expand(batch, -1)
            return flattened[:, None]

        return expand(offset, "normalizer offset"), expand(
            scale,
            "normalizer scale",
        )

    @property
    def batch(self) -> int:
        return int(self.source_action.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.source_action.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.source_action.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.source_action.device

    @property
    def physical_fingerprint(self) -> Tensor:
        """The two physical row streams read by W's sequence encoder."""

        return torch.cat((self.canonical_value, self.canonical_delta), dim=-1)

    @property
    def fingerprint(self) -> Tensor:
        """Lossless numeric audit view; metadata stays explicit beside it."""

        chart_code = self.source_action.new_full(
            (self.batch, self.horizon, 1),
            _ACTION_SEQUENCE_CHART_CODES[self.chart_version],
        )
        current_boundary = self.current_action[:, None].to(
            device=self.device,
            dtype=self.source_action.dtype,
        ).expand(-1, self.horizon, -1)
        return torch.cat(
            (
                self.source_action,
                self.canonical_value,
                self.canonical_delta,
                current_boundary,
                self.normalizer_offset.expand(-1, self.horizon, -1),
                self.normalizer_scale.expand(-1, self.horizon, -1),
                self.row_start,
                self.row_end,
                chart_code,
            ),
            dim=-1,
        )

    @property
    def action_fingerprint(self) -> Tensor:
        return self.fingerprint

    @property
    def metadata_identity(self) -> tuple[str, str, str, str, int, int]:
        return (
            self.schema,
            self.outlet_profile,
            self.chart_version,
            self.normalizer_fingerprint,
            int(self.arm_dim),
            int(self.known_prefix_end),
        )

    def validate(self, *, action_dim: int, horizon: int = 24) -> None:
        if self.schema != PHYSICAL_ACTION_SEQUENCE_SCHEMA:
            raise ValueError("unknown physical action sequence schema")
        if self.chart_version not in _ACTION_SEQUENCE_CHART_CODES:
            raise ValueError("unknown physical action sequence chart version")
        if not self.outlet_profile:
            raise ValueError("physical action sequence outlet profile is empty")
        if not self.normalizer_fingerprint:
            raise ValueError("physical action sequence normalizer identity is empty")
        if (
            type(self.known_prefix_end) is not int
            or self.known_prefix_end != 24
            or int(horizon) != 24
        ):
            raise ValueError("sequence-prefix-v1 requires one complete 24-row prefix")
        if type(self.arm_dim) is not int or not 0 < self.arm_dim < int(action_dim):
            raise ValueError("physical action sequence arm width is invalid")
        expected = (self.batch, 24, int(action_dim))
        _shape(self.source_action, expected, "physical action sequence source")
        _shape(self.canonical_value, expected, "physical action sequence value")
        _shape(self.canonical_delta, expected, "physical action sequence delta")
        _shape(
            self.current_action,
            (self.batch, int(action_dim)),
            "physical action sequence current action",
        )
        time_shape = (self.batch, 24, 1)
        _shape(self.row_start, time_shape, "physical action sequence row start")
        _shape(self.row_end, time_shape, "physical action sequence row end")
        normalizer_shape = (self.batch, 1, int(action_dim))
        _shape(
            self.normalizer_offset,
            normalizer_shape,
            "physical action sequence normalizer offset",
        )
        _shape(
            self.normalizer_scale,
            normalizer_shape,
            "physical action sequence normalizer scale",
        )
        tensors = (
            self.source_action,
            self.canonical_value,
            self.canonical_delta,
            self.current_action,
            self.row_start,
            self.row_end,
            self.normalizer_offset,
            self.normalizer_scale,
        )
        if any(value.device != self.source_action.device for value in tensors):
            raise ValueError("physical action sequence tensors must share a device")
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("physical action sequence tensors must be floating point")
        if (
            self.normalizer_offset.dtype != torch.float32
            or self.normalizer_scale.dtype != torch.float32
        ):
            raise TypeError(
                "physical action sequence normalizer metadata must remain FP32"
            )

    def assert_exact_contract(self) -> None:
        """Run synchronization-requiring value checks only in explicit audits."""

        expected_start = torch.arange(
            24,
            device=self.device,
            dtype=self.row_start.dtype,
        ).reshape(1, 24, 1).expand(self.batch, -1, -1)
        if not torch.equal(self.row_start, expected_start):
            raise ValueError("physical action sequence row starts are not contiguous")
        if not torch.equal(self.row_end, expected_start + 1):
            raise ValueError("physical action sequence row ends are not contiguous")
        if not bool(torch.isfinite(self.normalizer_offset).all()) or not bool(
            torch.isfinite(self.normalizer_scale).all()
        ):
            raise ValueError("physical action sequence normalizer is non-finite")
        if bool((self.normalizer_scale <= 0.0).any()):
            raise ValueError("physical action sequence normalizer scale is not positive")
        if not torch.equal(
            self.normalizer_offset,
            self.normalizer_offset[:1].expand_as(self.normalizer_offset),
        ) or not torch.equal(
            self.normalizer_scale,
            self.normalizer_scale[:1].expand_as(self.normalizer_scale),
        ):
            raise ValueError("physical action sequence normalizer differs across batch")
        expected_fingerprint = physical_action_normalizer_fingerprint(
            self.normalizer_offset[0],
            self.normalizer_scale[0],
        )
        if self.normalizer_fingerprint != expected_fingerprint:
            raise ValueError("physical action sequence normalizer identity is inconsistent")
        if self.chart_version == ABSOLUTE_ACTION_SEQUENCE_CHART:
            if not torch.equal(self.source_action, self.canonical_value):
                raise ValueError("absolute action sequence source/value diverged")
            delta_source = self.source_action
            delta = self.canonical_delta
            current = self.current_action
        else:
            arm = self.arm_dim
            expected_arm_delta = (
                self.source_action[..., :arm]
                - self.normalizer_offset[..., :arm].to(dtype=self.source_action.dtype)
            )
            if not torch.equal(
                self.canonical_delta[..., :arm],
                expected_arm_delta,
            ):
                raise ValueError("relative action sequence source/delta diverged")
            if not torch.allclose(
                self.canonical_value[..., :arm],
                torch.cumsum(expected_arm_delta, dim=1),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("relative action sequence cumulative value diverged")
            if not torch.equal(
                self.canonical_value[..., arm:],
                self.source_action[..., arm:],
            ):
                raise ValueError("relative action sequence gripper value diverged")
            if not torch.equal(
                self.current_action[..., :arm],
                torch.zeros_like(self.current_action[..., :arm]),
            ):
                raise ValueError("relative action sequence arm boundary is not zero")
            delta_source = self.source_action[..., arm:]
            delta = self.canonical_delta[..., arm:]
            current = self.current_action[..., arm:]
        boundary = torch.cat(
            (
                current[:, None].to(
                    device=self.device,
                    dtype=delta_source.dtype,
                ),
                delta_source[:, :-1],
            ),
            dim=1,
        )
        # Recompute the producer's subtraction rather than trying to invert a
        # rounded low-precision delta with addition. Both charts retain the
        # same numerical factory operations under FP32, BF16 and FP16.
        if not torch.equal(delta, delta_source - boundary):
            raise ValueError("physical action sequence delta reconstruction failed")


WorldActionCondition = PhysicalActionCondition | PhysicalActionSequenceCondition


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

    ``selector`` is the completed G3 feature chart, NOT a physical W rollout.
    ``value`` is a decoder feature. In summed_legacy_v1 its coefficients subtract
    a learned context response; typed_plan_v1 instead uses a structural zero-
    feature origin, separate source values and typed plan attention. Neither is
    a calibrated no-op counterfactual or measured physical execution error.
    """

    selector: Tensor  # [B,I*C*8*8,H] -- 512 V120 spatial transition rows
    value: Tensor  # [B,I*C*8*8,H]
    action_coefficients: Tensor  # [B,I*C*8*8,R]
    neutral_coefficients: Tensor  # [B,I*C*8*8,R]
    condition_mode: str = SUMMED_TRANSITION

    def validate(self, *, hidden: int) -> None:
        validate_transition_condition_mode(self.condition_mode)
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
    """The only W value object visible to P2.

    In control_aligned_24_v1 these are uniform means at the declared sparse
    successor supports, NOT instantaneous/endpoint states. Transport is image
    displacement from the current reference. Covariance is a mean of within-
    support correspondence covariances, not task risk or total temporal spread.
    """

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
    control_domain: CandidateControlDomain | None = None  # candidate control, not visibility
    camera_names: tuple[str, ...] = ()  # W-owned chart order, not guessed from axis length

    time_grid_mode: str = LEGACY_FUTURE_TIME

    @property
    def intervals(self) -> int:
        return int(self.semantic_delta.shape[1])

    def policy_view(self) -> "FutureObjectDynamics":
        """Exclude uncontrolled extrapolation before any P2 nonlinear read.

        Raw W outputs may remain available for explicitly labelled audits, but
        unknown intervals supply neither keys nor values to action generation.
        Current scene coordinates/availability are NOT changed by this mask.
        """
        domain = self.control_domain
        if domain is None:
            return self
        return replace(
            self,
            successor_content=torch.where(
                domain.mask_for(self.successor_content), self.successor_content,
                self.current_reference[:, None],
            ),
            semantic_delta=domain.quarantine(self.semantic_delta),
            transport_mean=domain.quarantine(self.transport_mean),
            transport_covariance=domain.quarantine(self.transport_covariance),
        )

    def _common(self, value: Tensor) -> Tensor:
        """Common effect over admitted intervals, never over unknown controls."""
        if value.ndim < 3:
            raise ValueError("future effect must retain an interval axis")
        if self.control_domain is not None:
            return self.control_domain.common(value)
        return value.float().mean(dim=1)

    def _interval_innovation(self, value: Tensor) -> Tensor:
        common = self._common(value)
        if self.control_domain is not None:
            safe = self.control_domain.quarantine(value.float())
            return self.control_domain.quarantine(safe - common[:, None])
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
        grid = resolve_future_time(self.time_grid_mode)
        if self.control_domain is not None:
            self.control_domain.validate(intervals=expected_intervals)
            if self.control_domain.interval_bounds != grid.bounds:
                raise ValueError("world prediction and candidate time grid disagree")
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
        if self.camera_names and (len(self.camera_names) != cameras
                or len(set(self.camera_names)) != cameras
                or any(not name for name in self.camera_names)):
            raise ValueError("future named camera charts must be unique and match C")
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
            camera_names=self.camera_names,
            time_grid_mode=self.time_grid_mode,
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
            control_domain=self.control_domain,
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
class RecordedActionSequenceCondition:
    """Canonical chart shared by separately typed recorded-control sources.

    Not a subtype of a deployment action condition. Known rows are a source-
    owned prefix; missing rows are quarantined and do not advance the recurrence.
    Time uses the same 24-control-step scale as the online sequence chart, not
    rescaled 48-step time. Only the outlet adapter constructs canonical values.
    """

    source_action: Tensor
    canonical_value: Tensor
    canonical_delta: Tensor
    observed: Tensor  # bool [B,Tw]; actual controls, not a model prediction
    row_end: Tensor  # [B,Tw,1], control-step endpoints
    outlet_profile: str
    chart_version: str
    normalizer_fingerprint: str
    control_time_scale: int = 24
    schema: str = "observed-world-action-sequence-v1"

    time_grid_mode: str = LEGACY_FUTURE_TIME

    @property
    def batch(self) -> int:
        return int(self.source_action.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.source_action.shape[1])

    @property
    def device(self) -> torch.device:
        return self.source_action.device

    @property
    def physical_fingerprint(self) -> Tensor:
        return torch.cat((self.canonical_value, self.canonical_delta), dim=-1)

    @property
    def interval_observed(self) -> Tensor:
        # A future state depends on ALL earlier controls, not just controls in
        # its interval. An unobserved earlier control cannot be filled by zero.
        return torch.stack([
            self.observed[:, :upper].all(dim=1) for _, upper in resolve_future_time(self.time_grid_mode).bounds
        ], dim=1)

    def validate(self, *, action_dim: int) -> None:
        expected_schema = ("executed-world-action-sequence-v1"
            if isinstance(self, ExecutedActionSequenceCondition) else "observed-world-action-sequence-v1")
        if self.schema != expected_schema:
            raise ValueError("unknown recorded action source schema")
        if self.source_action.ndim != 3:
            raise ValueError("observed world action must be [B,Tw,A]")
        horizon = max(upper for _, upper in resolve_future_time(self.time_grid_mode).bounds)
        _shape(self.source_action, (self.batch, horizon, action_dim), "observed world actions")
        _shape(self.canonical_value, tuple(self.source_action.shape), "observed action values")
        _shape(self.canonical_delta, tuple(self.source_action.shape), "observed action deltas")
        _shape(self.observed, (self.batch, horizon), "observed action support")
        _shape(self.row_end, (self.batch, horizon, 1), "observed action time")
        if self.observed.dtype != torch.bool or self.observed.device != self.device:
            raise ValueError("observed action support must be Boolean on the action device")
        if bool((self.observed[:, 1:] & ~self.observed[:, :-1]).any()):
            raise ValueError("observed actions must form a causal prefix without holes")
        if type(self.control_time_scale) is not int or self.control_time_scale != 24:
            raise ValueError("observed controls must retain the online physical time scale")
        if self.chart_version not in _ACTION_SEQUENCE_CHART_CODES or not self.outlet_profile or not self.normalizer_fingerprint:
            raise ValueError("observed action chart metadata is missing or unknown")
        for value in (self.source_action, self.canonical_value, self.canonical_delta, self.row_end):
            if value.device != self.device or not value.is_floating_point() or value.requires_grad:
                raise ValueError("observed actions must be detached floating labels on one device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("observed action labels must be finite after quarantine")
        for value in (self.source_action, self.canonical_value, self.canonical_delta):
            if bool((value.masked_select(~self.observed[..., None].expand_as(value)) != 0).any()):
                raise ValueError("unobserved action payload must be quarantined")
        expected = torch.arange(1, horizon + 1, device=self.device, dtype=self.row_end.dtype)
        if not torch.equal(self.row_end, expected[None, :, None].expand(self.batch, -1, -1)):
            raise ValueError("observed action time must retain actual contiguous control steps")


@dataclass(frozen=True)
class ObservedActionSequenceCondition(RecordedActionSequenceCondition):
    """Future labels, never acknowledged controls or a deployment candidate."""

@dataclass(frozen=True)
class ExecutedActionSequenceCondition(RecordedActionSequenceCondition):
    """Already executed recorded controls; never future labels or proposals."""
    schema: str = "executed-world-action-sequence-v1"


@dataclass(frozen=True)
class SupervisedWorld:
    """Prediction used by future losses only, never admitted as CandidateWorld."""

    action_condition: ObservedActionSequenceCondition
    dynamics: FutureObjectDynamics

    def validate(self, *, action_dim: int) -> None:
        if not isinstance(self.action_condition, ObservedActionSequenceCondition):
            raise TypeError("supervised world requires observed controls")
        self.action_condition.validate(action_dim=action_dim)
        self.dynamics.validate()
        if self.dynamics.time_grid_mode != self.action_condition.time_grid_mode:
            raise ValueError("supervised world action and prediction time grids differ")
        if self.dynamics.control_domain is not None:
            raise ValueError("supervised world cannot inherit a candidate control domain")
        if self.action_condition.batch != int(self.dynamics.current_reference.shape[0]):
            raise ValueError("supervised world batches differ")
        if self.action_condition.device != self.dynamics.semantic_delta.device:
            raise ValueError("supervised world devices differ")


@dataclass(frozen=True)
class CandidateWorld:
    """An action-tagged W prediction consumed by the consequence evaluator.

    The tag is an object-identity boundary, not a numeric hash.  It makes a
    stale-world mix-up fail before P2 while avoiding a CUDA synchronization on
    every ODE node.  Explicit value equality remains available through the
    action-condition audit method when a probe needs it.
    """

    action_condition: WorldActionCondition
    dynamics: FutureObjectDynamics

    @property
    def action_fingerprint(self) -> Tensor:
        return self.action_condition.action_fingerprint

    def validate(self, *, action_dim: int) -> None:
        if not isinstance(self.action_condition, (PhysicalActionCondition, PhysicalActionSequenceCondition)):
            raise TypeError("candidate world rejects training-only observed controls")
        self.action_condition.validate(action_dim=action_dim)
        self.dynamics.validate()
        domain = self.dynamics.control_domain
        if domain is not None and (
            not isinstance(self.action_condition, PhysicalActionSequenceCondition)
            or domain.known_prefix_steps != self.action_condition.horizon
        ):
            raise ValueError("candidate control domain differs from actual action prefix")
        if self.action_condition.batch != int(self.dynamics.current_reference.shape[0]):
            raise ValueError("candidate world action and dynamics batches do not align")
        condition_device = (
            self.action_condition.interval_action.device
            if isinstance(self.action_condition, PhysicalActionCondition)
            else self.action_condition.device
        )
        if condition_device != self.dynamics.semantic_delta.device:
            raise ValueError("candidate world action and dynamics must share a device")

    def assert_action_identity(
        self,
        action_condition: WorldActionCondition,
    ) -> None:
        """Reject a world paired with another candidate action object."""

        if self.action_condition is not action_condition:
            raise ValueError(
                "candidate world action fingerprint does not match current candidate"
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
    future_interval_valid: Tensor | None = None  # dataset-owned, never a predicted confidence
    supervised_world: SupervisedWorld | None = None  # training plane only
    robot_response_loss: Tensor | None = None  # past-to-current observed response
    annotated_goal_terms: dict[str, Tensor] | None = None
    operation_terms: dict[str, Tensor] | None = None  # supervised expectations, never observed progress

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
