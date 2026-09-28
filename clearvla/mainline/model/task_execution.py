"""Joint task-grounded execution, from S relations to candidate consequence reads.

One shared target law, no hard argmax, no motion/phase/contact rules. This is a
learned information path, not an assurance that a trained robot obeys its task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time
from ..task_execution import TaskRelationEvidence
from .routing import smooth_rms_contract
from .target_binding import BoundTargetRead, TargetBinding, TargetEvidence, masked_probability

if TYPE_CHECKING:
    from ..operation_expectation import OperationExpectation
    from .types import FutureObjectDynamics


class JointTaskRelationEncoder(nn.Module):
    """Build nonlinear task/object/view/robot relations BEFORE I/K/C reduction.

    Native image positions and robot-state features enter separate learned
    projections; image coordinates are never subtracted from a world TCP.
    Language remains S-owned. History is context, not a substitute for task.
    """

    role_basis: Tensor

    def __init__(self, *, hidden: int, state_dim: int, camera_names: tuple[str, ...]) -> None:
        super().__init__()
        self.hidden = hidden
        self.camera_names = tuple(camera_names)
        if hidden < 1 or not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("joint relations require hidden width and named cameras")
        self.attributes = nn.Linear(4 * hidden, hidden, bias=False)
        self.coordinates = nn.Sequential(
            nn.Linear(2, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, hidden, bias=False)
        )
        self.robot = nn.Linear(state_dim, hidden, bias=False)
        self.view = nn.Linear(len(camera_names), hidden, bias=False)
        self.history = nn.Linear(hidden, hidden, bias=False)
        self.relation = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.task = nn.Linear(hidden, hidden, bias=False)
        self.interaction = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def forward(
        self,
        *,
        task_intervals: Tensor,
        attributes: Tensor,
        position_probability: Tensor,
        position_grid: Tensor,
        state: Tensor,
        history_context: Tensor,
        view_observed: Tensor,
        binding: TargetBinding,
        current_content: Tensor,
    ) -> TaskRelationEvidence:
        b, k, sources, h = attributes.shape
        c = len(self.camera_names)
        if (
            sources != 4
            or h != self.hidden
            or position_probability.ndim != 4
            or position_probability.shape[:3] != (b, k, c)
        ):
            raise ValueError("joint relation attributes/location law lost their source chart")
        if position_grid.shape != (position_probability.shape[-1], 2):
            raise ValueError("joint relation needs its native image grid")
        if (
            task_intervals.shape != (b, 4, h)
            or view_observed.shape != (b, k, c)
            or view_observed.dtype != torch.bool
        ):
            raise ValueError("joint relation task/support axes differ")
        if state.shape != (b, self.robot.in_features) or history_context.shape != (b, h):
            raise ValueError("joint relation robot/history chart differs")
        valid = view_observed & binding.supported[..., None]
        dtype = self.task.weight.dtype
        a = torch.where(valid.any(-1)[..., None, None], attributes, 0.0)
        a = self.attributes(a.flatten(-2).to(dtype))[:, :, None].expand(-1, -1, c, -1)
        # Apply nonlinear position features BEFORE expectation. Equal
        # centroids no longer force equal spatial evidence. The learned finite
        # feature map is not an injective or calibrated geometry guarantee.
        p = torch.where(valid[..., None], position_probability.float(), 0.0)
        if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
            raise ValueError("task relation position probability is invalid")
        if not torch.allclose(p.sum(-1), valid.float(), atol=2e-5, rtol=0):
            raise ValueError(
                "task relation position law must be source-normalized within each view"
            )
        features = self.coordinates(position_grid.to(dtype))
        with torch.autocast(device_type=features.device.type, enabled=False):
            xy = torch.einsum("bkcn,nh->bkch", p, features.float()).to(a.dtype)
        robot = self.robot(state.to(dtype))[:, None, None].expand(-1, k, c, -1)
        role = self.view(self.role_basis.to(dtype))[None, None].expand(b, k, -1, -1)
        spatial = self.relation(torch.cat((a, xy, robot, role), -1))
        # Joint values, not another object score. Preserve separate sources
        # until a nonlinear interaction; no learned coordinate-as-meter claim.
        spatial = spatial * (
            1.0 + torch.tanh(self.history(history_context.to(dtype)))[:, None, None]
        )
        task = self.task(task_intervals.to(dtype))[:, :, None, None].expand(-1, -1, k, c, -1)
        rel = spatial[:, None].expand(-1, 4, -1, -1, -1)
        # No relation-only or task-only additive value bypass. This
        # constrains representation ownership, not a robot behavior rule.
        values = self.interaction(
            torch.cat((rel * task, rel * torch.tanh(task), task * torch.tanh(rel)), -1)
        )
        values = torch.where(valid[:, None, :, :, None], values, 0.0)
        result = TaskRelationEvidence(
            values, valid, binding, self.camera_names, current_content, state
        )
        result.validate(hidden=h)
        return result


class ObjectRoleRead(nn.Module):
    """Query-dependent I/C read INSIDE each K, then explicit role-mass read.

    There is no K softmax. Null and missing-view mass cannot be normalized
    away; no query-only/bias value survives a zero source. Same global query
    reads target and scene with different parameters, never in one competition.
    """

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads:
            raise ValueError("object-role attention dimensions must be head aligned")
        self.hidden, self.heads, self.width = hidden, heads, hidden // heads
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, query: Tensor, keys: Tensor, values: Tensor, support: Tensor, mass: Tensor
    ) -> Tensor:
        if keys.shape != values.shape or keys.ndim != 5 or keys.shape[-1] != self.hidden:
            raise ValueError("role read needs matching [B,I,K,C,H] keys and values")
        b, i, k, c, h = keys.shape
        if support.shape != (b, i, k, c) or support.dtype != torch.bool or mass.shape != (b, k):
            raise ValueError("role read support/mass is not the same object chart")
        if query.shape[0] != b or query.shape[-1] != h:
            raise ValueError("role query shape differs")
        safe_k = (
            torch.where(support[..., None], keys, 0.0)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, k, i * c, h)
        )
        safe_v = (
            torch.where(support[..., None], values, 0.0).permute(0, 2, 1, 3, 4).reshape_as(safe_k)
        )
        dtype = self.query.weight.dtype
        q = self.query(self.query_norm(query.to(dtype))).reshape(b, -1, self.heads, self.width)
        kk = self.key(self.key_norm(safe_k.to(dtype))).reshape(b, k, i * c, self.heads, self.width)
        vv = self.value(safe_v.to(dtype)).reshape_as(kk)
        legal = support.permute(0, 2, 1, 3).reshape(b, k, i * c)
        # Probability law really stays FP32 under both CUDA and CPU autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits = torch.einsum("bpad,bklad->bpkal", q.float(), kk.float()) * self.width**-0.5
            p = masked_probability(logits, legal[:, None, :, None].expand_as(logits))
        per_object = torch.einsum("bpkal,bklad->bpkad", p.to(vv.dtype), vv)
        per_object = self.output(per_object.flatten(-2))
        out = (per_object.float() * mass[:, None, :, None]).sum(2)
        return out.to(query.dtype).reshape_as(query)


class TaskRelationRead(nn.Module):
    """Shared target plus separately owned scene relation; no second selector."""

    time_code: Tensor

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.target = ObjectRoleRead(hidden, heads)
        self.scene = ObjectRoleRead(hidden, heads)
        grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
        self.register_buffer("time_code", grid.interval_encoding(hidden), persistent=True)

    def forward(self, query: Tensor, evidence: TaskRelationEvidence) -> Tensor:
        evidence.validate(hidden=self.hidden)
        support = evidence.view_observed[:, None].expand(-1, 4, -1, -1)
        mass = evidence.binding.mass
        # Complement is an expectation weight, not a renormalized categorical
        # non-target law. At null=1, scene remains; no object inherits target ID.
        scene_mass = (evidence.binding.supported.float() - mass) / mass.shape[-1]
        keys = evidence.values + self.time_code.to(evidence.values)[None, :, None, None]
        target = self.target(query, keys, evidence.values, support, mass)
        scene = self.scene(query, keys, evidence.values, support, scene_mass)
        result, _ = smooth_rms_contract(target + scene, 0.35)
        return result


class TaskAwareFactualRead(nn.Module):
    """Task relation changes the P1 query, never the factual value source."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.relation_query = TaskRelationRead(hidden, heads)
        self.current_values = BoundTargetRead(hidden, heads)

    def forward(
        self,
        query: Tensor,
        evidence: TargetEvidence,
        binding: TargetBinding,
        *,
        relation: TaskRelationEvidence,
    ) -> Tensor:
        if relation.binding is not binding:
            raise ValueError("P1 relation cannot substitute target identity")
        task_query = query + self.relation_query(query, relation)
        return self.current_values(task_query, evidence, binding)


class TaskOutcomePlanRead(nn.Module):
    """Compare S intent and W candidate effect BEFORE pooling K or C.

    Differences are prediction features in the same DINO/image change chart,
    not measured physical progress. W remains language-invariant at fixed
    facts/action. Separate target and scene readers preserve side-effect
    information even when target mass is concentrated or null.
    """

    time_code: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, heads: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.hidden, self.camera_names = hidden, tuple(camera_names)
        self.semantic = nn.Linear(content_dim, hidden, bias=False)
        self.image = nn.Linear(2, hidden, bias=False)
        self.relation_key = nn.Linear(hidden, hidden, bias=False)
        self.effect_key = nn.Linear(2 * hidden, hidden, bias=False)
        self.gap_model = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.task_modulation = nn.Linear(hidden, hidden, bias=False)
        self.target_relation = ObjectRoleRead(hidden, heads)
        self.target_gap = ObjectRoleRead(hidden, heads)
        self.scene_gap = ObjectRoleRead(hidden, heads)
        grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
        self.register_buffer("time_code", grid.interval_encoding(hidden), persistent=True)

    def forward(
        self,
        query: Tensor,
        relation: TaskRelationEvidence,
        expected: OperationExpectation,
        predicted: FutureObjectDynamics,
    ) -> tuple[Tensor, Tensor]:
        relation.validate(hidden=self.hidden)
        expected.validate()
        predicted.validate()
        if expected.binding is not relation.binding:
            raise ValueError("task comparison cannot substitute target binding")
        if (
            expected.camera_names != self.camera_names
            or predicted.camera_names != self.camera_names
            or relation.camera_names != self.camera_names
        ):
            raise ValueError("task comparison camera roles differ")
        if (
            expected.time_grid_mode != relation.time_grid_mode
            or predicted.time_grid_mode != relation.time_grid_mode
        ):
            raise ValueError("task comparison time charts differ")
        if (
            expected.semantic_delta.shape != predicted.semantic_delta.shape
            or expected.image_delta.shape != predicted.transport_mean.shape
        ):
            raise ValueError("task comparison requires the same per-object effect charts")
        # The W prediction and task expectation must originate from the exact
        # same G object chart, not equal-shaped K slots from another frame.
        if (
            predicted.source_content is not expected.current_content
            or relation.current_content is not expected.current_content
            or relation.current_state is not expected.current_state
        ):
            raise ValueError("task comparison lost exact current object reference")
        if predicted.control_domain is None:
            raise ValueError("task comparison requires a declared known-action control domain")
        support = relation.view_observed[:, None].expand(-1, 4, -1, -1)
        support = support & expected.view_observed[:, None]
        support = support & (predicted.chart_availability[..., 0] > 0)[:, None, :, None]
        support = support & (predicted.camera_chart_availability[..., 0] > 0)[:, None]
        support = support & predicted.control_domain.mask_for(support)
        dtype = self.semantic.weight.dtype
        sem_valid = support.any(-1)
        expected_sem = torch.where(sem_valid[..., None], expected.semantic_delta, 0.0)
        predicted_sem = torch.where(sem_valid[..., None], predicted.semantic_delta, 0.0)
        expected_img = torch.where(support[..., None], expected.image_delta, 0.0)
        predicted_img = torch.where(support[..., None], predicted.transport_mean, 0.0)
        sem = self.semantic((expected_sem.float() - predicted_sem.float()).to(dtype))
        sem = sem[:, :, :, None].expand(-1, -1, -1, len(self.camera_names), -1)
        image = self.image((expected_img.float() - predicted_img.float()).to(dtype))
        # Keep content and image errors distinct until a nonlinear joint read.
        gap = self.gap_model(torch.cat((sem, image), -1))
        gap = gap * (1.0 + torch.tanh(self.task_modulation(relation.values.to(dtype))))
        gap = torch.where(support[..., None], gap, 0.0)
        key = self.relation_key(relation.values.to(dtype)) + self.effect_key(
            torch.cat((sem, image), -1)
        )
        key = key + self.time_code.to(key)[None, :, None, None]
        mass = relation.binding.mass
        target = self.target_relation(query, key, relation.values, support, mass)
        target = target + self.target_gap(query, key, gap, support, mass)
        scene_mass = (relation.binding.supported.float() - mass) / mass.shape[-1]
        scene = self.scene_gap(query, key, gap, support, scene_mass)
        # Scene comparison disappears exactly when desired/predicted effects
        # agree; target CURRENT relations remain available for approaching.
        return target, scene


class TaskConditionedTargetBinder(nn.Module):
    """Replace, rather than duplicate, the shared target selector.

    History can condition which S task tokens are read, but is not added to
    their value or directly scored as target identity. Every real-object score
    is a learned task/object interaction. A zero task value cannot let robot
    history alone choose a nearer object. This does not certify grounding.
    """

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.objects = nn.Linear(hidden, hidden, bias=False)
        self.history_query = nn.Linear(hidden, hidden, bias=False)
        self.task_value = nn.Linear(hidden, hidden, bias=False)
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.task_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.task_read = nn.MultiheadAttention(
            hidden, heads, batch_first=True, bias=False, dropout=0.0
        )
        self.compatibility = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.null = nn.Linear(hidden, 1)

    def forward(
        self, task: Tensor, objects: Tensor, supported: Tensor, *, history: Tensor
    ) -> TargetBinding:
        if (
            task.ndim != 3
            or objects.ndim != 3
            or task.shape[0] != objects.shape[0]
            or task.shape[-1] != objects.shape[-1]
        ):
            raise ValueError("task-conditioned binding requires named task and object tokens")
        if (
            supported.shape != objects.shape[:2]
            or supported.dtype != torch.bool
            or history.shape != (task.shape[0], task.shape[-1])
        ):
            raise ValueError("task-conditioned binding lost source support/history")
        safe = torch.where(supported[..., None], objects, 0.0)
        obj = self.objects(safe)
        query = self.query_norm(obj + self.history_query(history)[:, None])
        task_value = self.task_value(task)
        read, _ = self.task_read(query, self.task_norm(task_value), task_value, need_weights=False)
        score = self.score(read * torch.tanh(self.compatibility(obj)))[..., 0]
        return TargetBinding.from_logits(score, self.null(task_value.mean(1)), supported)
