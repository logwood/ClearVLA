"""Reference-conditioned annotated endpoint intent, not a success/progress oracle.

The label type is legal only on the training plane. Its annotation provenance
and endpoint offset are admission metadata, never inputs to the goal predictor.
The online goal is inferred from language and the observed instruction start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .instruction_change import InstructionChangeEvidence
    from .instruction_reference import InstructionReference

ANNOTATED_ENDPOINT_GOAL = "annotated_endpoint_relation_v1"


def annotation_goal_metadata(camera_names: tuple[str, ...]) -> dict[str, object]:
    if (
        not camera_names
        or any(not n for n in camera_names)
        or len(set(camera_names)) != len(camera_names)
    ):
        raise ValueError("annotated goal requires unique named views")
    return {
        "schema": "reference-annotated-endpoint-goal-v1",
        "camera_names": list(camera_names),
        "predictor": "instruction-start-DINO-state-and-language-only-no-current-or-clock",
        "labels": "source-annotation-end-observation-not-window-end-or-absorbing-suffix",
        "measurement": "native-normalized-DINO-cosine-soft-law-uniform-real-cell-prior-plus-fixed-null",
        "prediction": "normalized-end-scene-and-full-anchor-destination-law-with-learned-null",
        "values": "independent-S-P3-content-spatial-joint-expectations-before-reduction",
        "correspondence": "current-G3-to-reference-soft-law-with-real-null-mass",
        "support": "source-label-availability-not-predicted-goal-or-target-confidence",
        "lifecycle": "prepare-once-per-online-encoding-no-extra-G-W-or-ODE-image-read",
        "limits": "annotated-demonstration-result-prior-not-certified-success-3D-or-persistent-ID",
    }


@dataclass(frozen=True)
class AnnotationEndpoint:
    """Independent training-only annotation endpoint, possibly outside W's horizon."""

    dino: Tensor  # [B,C,N,D], normalized ONLY by the target builder
    state: Tensor  # FP32 [B,S], shared observed state feature chart
    visual_observed: Tensor  # bool [B,C,N], real label support
    state_observed: Tensor  # bool [B]
    declared: Tensor  # bool [B], verified endpoint provenance
    source_indices: Tensor  # int64 [B,5]: annotation, context, start, end, center
    offset_steps: Tensor  # int64 [B], endpoint minus current center; label-only

    def validate(self, config: ExperimentConfig, *, batch: int, device: torch.device) -> None:
        d = config.dimensions
        shape = (batch, d.num_cameras, d.patches_per_camera, d.visual_token_dim)
        if self.dino.shape != shape or self.state.shape != (batch, d.state_dim):
            raise ValueError("annotation endpoint must retain its native image/state chart")
        if self.visual_observed.shape != shape[:-1] or self.visual_observed.dtype != torch.bool:
            raise ValueError("annotation endpoint visual support must be bool [B,C,N]")
        for x in (self.state_observed, self.declared):
            if x.shape != (batch,) or x.dtype != torch.bool:
                raise ValueError("annotation endpoint support/declaration must be bool [B]")
        if self.source_indices.shape != (batch, 5) or self.source_indices.dtype != torch.long:
            raise ValueError("annotation endpoint needs exact int64 source provenance")
        if self.offset_steps.shape != (batch,) or self.offset_steps.dtype != torch.long:
            raise ValueError("annotation endpoint offsets must be int64 [B]")
        if not self.dino.is_floating_point() or self.state.dtype != torch.float32:
            raise TypeError("annotation endpoint needs floating DINO and FP32 state")
        if any(
            x.device != device
            for x in (
                self.dino,
                self.state,
                self.visual_observed,
                self.state_observed,
                self.declared,
                self.source_indices,
                self.offset_steps,
            )
        ):
            raise ValueError("annotation endpoint must share the training batch device")
        used = self.visual_observed.flatten(1).any(1) | self.state_observed
        if bool((used & ~self.declared).any()):
            raise ValueError("undeclared annotation endpoint cannot supply labels")
        anno, context, start, end, center = self.source_indices.unbind(-1)
        legal = (anno >= 0) & (context >= 0) & (start >= context) & (end > start)
        legal = legal & (center >= start - context) & (center <= end - context)
        legal = legal & (self.offset_steps == end - context - center)
        if bool((self.declared & ~legal).any()):
            raise ValueError("annotation endpoint source chronology is inconsistent")
        if bool((~self.declared & (self.offset_steps != 0)).any()):
            raise ValueError("unknown annotation endpoint must not invent a future offset")
        for value, support in (
            (self.dino, self.visual_observed[..., None]),
            (self.state, self.state_observed[:, None]),
        ):
            if not bool(torch.isfinite(torch.where(support, value, 0.0)).all()):
                raise ValueError("annotation endpoint has nonfinite observed labels")


@dataclass(frozen=True)
class EndpointGoalPrediction:
    """Start-conditioned expected terminal scene and soft per-start-patch relation."""

    scene: Tensor  # FP32 [B,C,N,D], predicted normalized DINO values
    log_probability: Tensor  # FP32 [B,C,N,N+1], all destinations + null
    robot_delta: Tensor  # FP32 [B,S], expected endpoint minus start features
    coordinates: Tensor  # FP32 [N,2], named image chart, not world coords
    reference: InstructionReference
    camera_names: tuple[str, ...]

    def validate(self) -> None:
        b, c, n, _ = self.reference.dino.shape
        if self.scene.shape != self.reference.dino.shape or self.log_probability.shape != (
            b,
            c,
            n,
            n + 1,
        ):
            raise ValueError("endpoint goal must keep every reference/destination patch")
        if self.robot_delta.shape != self.reference.state.shape or self.coordinates.shape != (n, 2):
            raise ValueError("endpoint goal lost robot or spatial axes")
        if c != len(self.camera_names) or len(set(self.camera_names)) != c:
            raise ValueError("endpoint goal has a different named camera chart")
        for x in (self.scene, self.log_probability, self.robot_delta, self.coordinates):
            if x.dtype != torch.float32 or x.device != self.reference.dino.device:
                raise ValueError("endpoint goal numerical values must be same-device FP32")


@dataclass(frozen=True)
class AnnotatedGoalEvidence:
    prediction: EndpointGoalPrediction
    current_probability: Tensor  # FP32 [B,C,N,N+1], frozen observed soft law
    current_content: Tensor  # FP32 [B,C,N,D], normalized observed current DINO
    current_observed: Tensor  # bool [B,C,N]
    instruction_change: InstructionChangeEvidence

    def validate(self) -> None:
        p = self.prediction
        p.validate()
        ev = self.instruction_change
        if (
            ev.posterior is None
            or ev.reference is not p.reference
            or ev.camera_names != p.camera_names
        ):
            raise ValueError("goal comparison needs its exact causal instruction reference")
        if (
            self.current_probability.shape != p.log_probability.shape
            or self.current_content.shape != p.scene.shape
        ):
            raise ValueError("goal/current measurement native charts differ")
        if (
            self.current_observed.shape != p.scene.shape[:-1]
            or self.current_observed.dtype != torch.bool
        ):
            raise ValueError("goal comparison source support must remain bool [B,C,N]")
        for x in (self.current_probability, self.current_content):
            if x.dtype != torch.float32 or x.device != p.scene.device:
                raise ValueError("goal observations require same-device FP32")
        if self.current_observed.device != p.scene.device:
            raise ValueError("goal source support belongs to another device")

    def permute(self, change: InstructionChangeEvidence) -> AnnotatedGoalEvidence:
        # Reference-patch predictions/measurements have NO current K axis.
        return replace(self, instruction_change=change)


@dataclass(frozen=True)
class AnnotatedGoalValues:
    content: Tensor  # FP32 [B,K,C,H]
    image: Tensor  # FP32 [B,K,C,H]
    joint: Tensor  # FP32 [B,K,C,H]
    robot: Tensor  # FP32 [B,H]
    evidence: AnnotatedGoalEvidence
    reader_identity: int

    def validate(self, *, hidden: int) -> None:
        ev = self.evidence.instruction_change
        shape = (*ev.content_delta.shape[:3], hidden)
        if any(
            x.shape != shape for x in (self.content, self.image, self.joint)
        ) or self.robot.shape != (shape[0], hidden):
            raise ValueError("prepared goal values lost named object/view/robot axes")
        if any(
            x.dtype != torch.float32 or x.device != ev.current_state.device
            for x in (
                self.content,
                self.image,
                self.joint,
                self.robot,
            )
        ):
            raise ValueError("prepared goal values must be same-device FP32")

    def permute(self, index: Tensor, evidence: AnnotatedGoalEvidence) -> AnnotatedGoalValues:
        return replace(
            self,
            content=self.content[:, index],
            image=self.image[:, index],
            joint=self.joint[:, index],
            evidence=evidence,
        )
