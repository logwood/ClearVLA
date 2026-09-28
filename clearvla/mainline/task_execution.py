"""Typed task/scene execution evidence; no action rules or simulator truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .future_time import CONTROL_ALIGNED_FUTURE_TIME

if TYPE_CHECKING:
    from .model.target_binding import TargetBinding
    from .model.types import CandidateWorld

NO_TASK_EXECUTION = "none"
JOINT_TASK_EXECUTION = "joint_object_scene_v1"


@dataclass(frozen=True)
class TaskRelationEvidence:
    """S-owned [B,I,K,C,H] task-conditioned CURRENT relations.

    These are learned features, NOT physical contact, calibrated relative 3D
    coordinates, future labels, or persistent object identities. Binding is the
    original K+null law; every consumer retains it. I is the declared four-part
    control clock. Camera names retain meaning until a named within-K read.
    """

    values: Tensor
    view_observed: Tensor
    binding: TargetBinding
    camera_names: tuple[str, ...]
    current_content: Tensor
    current_state: Tensor
    time_grid_mode: str = CONTROL_ALIGNED_FUTURE_TIME

    def validate(self, *, hidden: int) -> None:
        if self.values.ndim != 5 or self.values.shape[1] != 4 or self.values.shape[-1] != hidden:
            raise ValueError("task relations must preserve [B,4,K,C,H]")
        b, _, k, c, _ = self.values.shape
        if min(b, k, c) < 1 or self.view_observed.shape != (b, k, c):
            raise ValueError("task relation observation support lost B/K/C")
        if (
            self.view_observed.dtype != torch.bool
            or self.view_observed.device != self.values.device
        ):
            raise ValueError("task relation source support must be colocated bool")
        if len(self.camera_names) != c or len(set(self.camera_names)) != c:
            raise ValueError("task relation requires unique named camera roles")
        if self.time_grid_mode != CONTROL_ALIGNED_FUTURE_TIME:
            raise ValueError("task relation requires the 24-step control chart")
        if self.current_content.ndim != 3 or self.current_content.shape[:2] != (b, k):
            raise ValueError("task relations lost current object source")
        if self.current_state.ndim != 2 or self.current_state.shape[0] != b:
            raise ValueError("task relations lost current robot source")
        if (
            self.current_content.device != self.values.device
            or self.current_state.device != self.values.device
        ):
            raise ValueError("task relation source device mismatch")
        self.binding.validate(batch=b, objects=k, device=self.values.device)
        if bool((self.view_observed & ~self.binding.supported[..., None]).any()):
            raise ValueError("task relation cannot invent unsupported objects")


def task_execution_metadata() -> dict[str, object]:
    return {
        "schema": "joint-task-object-scene-execution-v1",
        "relation": "task-times-object-native-spatial-expectation-and-robot-before-role-reduction",
        "spatial": "nonlinear-native-grid-features-before-G3-law-expectation-no-centroid-only-input",
        "binding": "task-object-interaction-K-plus-null-history-query-only-no-reselection",
        "task_value": "S-and-coarse-joint-relation-values-no-additive-history-query-scaffold",
        "comparison": "S-expected-minus-candidate-W-on-common-K-C-control-chart",
        "scene_mass": "(supported-object-minus-target-mass)/declared-K",
        "factual_reader": "task-relation-query-only-current-fact-values",
        "bottom": "compiled-P3-only-no-raw-language-or-object-ID",
        "physical_contact_claim": False,
        "persistent_identity_claim": False,
    }


@dataclass(frozen=True)
class TaskExecutionPlan:
    """P2-compiled target/scene outcome evidence, never raw W input to P3."""

    target: Tensor
    scene: Tensor
    relation: TaskRelationEvidence
    candidate_world: CandidateWorld

    def validate(self, *, hidden: int, horizon: int, basis: int) -> None:
        self.relation.validate(hidden=hidden)
        shape = (self.relation.values.shape[0], horizon, basis, hidden)
        if self.target.shape != shape or self.scene.shape != shape:
            raise ValueError("compiled task execution lost row/basis axes")
        if (
            self.target.device != self.relation.values.device
            or self.scene.device != self.target.device
        ):
            raise ValueError("compiled task execution source device mismatch")
        if self.candidate_world.dynamics.source_content is not self.relation.current_content:
            raise ValueError("compiled task execution belongs to another current world")
