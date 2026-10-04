"""Joint task-grounded execution, from S relations to candidate consequence reads.

One shared target law, no hard argmax, no motion/phase/contact rules. This is a
learned information path, not an assurance that a trained robot obeys its task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time
from ..task_execution import (
    JOINT_SPATIAL_TASK_EXECUTION, JOINT_TASK_EXECUTION, JOINT_TASK_EXECUTION_MODES,
    TaskRelationEvidence,
)
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

    def __init__(
        self, *, hidden: int, state_dim: int, camera_names: tuple[str, ...],
        task_execution_mode: str = JOINT_TASK_EXECUTION, spatial_tile_size: int = 16,
    ) -> None:
        super().__init__()
        if task_execution_mode not in JOINT_TASK_EXECUTION_MODES:
            raise ValueError("unknown task relation execution mode")
        if type(spatial_tile_size) is not int or spatial_tile_size < 1:
            raise ValueError("spatial tile size must be a positive integer")
        self.task_execution_mode = task_execution_mode
        self.spatial_tile_size = spatial_tile_size
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
        # Within the existing S owner; v1 creates no parameters or RNG draws.
        self.spatial_context = (
            nn.Linear(4 * hidden, hidden, bias=False)
            if task_execution_mode == JOINT_SPATIAL_TASK_EXECUTION else None
        )

    def _spatial_expectation(
        self, features: Tensor, context: Tensor, probability: Tensor,
    ) -> Tensor:
        """Condition BEFORE marginalizing the original G3 spatial law.

        E_p[SiLU(q + phi(x)) - SiLU(q)] is a learned image/robot relation,
        not pixels minus metres or a calibrated pose. p is neither reselected
        nor renormalized. Zero coordinate features yield zero spatial values.
        Non-reentrant recomputation bounds saved pointwise activations as well
        as forward temporaries. Tile order affects floating rounding only.
        """
        if features.ndim != 2 or features.shape[-1] != self.hidden:
            raise ValueError("spatial basis must be [N,H]")
        if context.ndim != 5 or context.shape[1] != 4 or context.shape[-1] != self.hidden:
            raise ValueError("spatial context must retain [B,4,K,C,H]")
        b, _, k, c, _ = context.shape
        if features.shape[0] < 1 or probability.shape != (b, k, c, features.shape[0]):
            raise ValueError("spatial expectation lost its native source law")
        if features.device != context.device or probability.device != context.device:
            raise ValueError("spatial expectation sources must share a device")

        def integrate(q: Tensor, points: Tensor, mass: Tensor) -> Tensor:
            with torch.autocast(device_type=q.device.type, enabled=False):
                qf = q.float()
                local = F.silu(qf[..., None, :] + points.float()[None, None, None, None])
                local = local - F.silu(qf)[..., None, :]
                return torch.einsum("bkcn,bikcnh->bikch", mass.float(), local)

        with torch.autocast(device_type=context.device.type, enabled=False):
            result = torch.zeros_like(context, dtype=torch.float32)
            for start in range(0, features.shape[0], self.spatial_tile_size):
                stop = min(start + self.spatial_tile_size, features.shape[0])
                points = features[start:stop]
                mass = probability[..., start:stop]
                if torch.is_grad_enabled() and any(x.requires_grad for x in (context, points, mass)):
                    piece = checkpoint(integrate, context, points, mass,
                                       use_reentrant=False, preserve_rng_state=False)
                else:
                    piece = integrate(context, points, mass)
                result = result + piece
        return result

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
        per_view = attributes.ndim == 5
        if attributes.ndim not in (4, 5):
            raise ValueError("relation attributes require [B,K,4,H] or actual [B,K,C,4,H]")
        b, k = attributes.shape[:2]
        sources, h = attributes.shape[-2:]
        c = len(self.camera_names)
        if per_view and attributes.shape[2] != c:
            raise ValueError("relation camera content axis differs")
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
        if per_view:
            a = torch.where(valid[...,None,None], attributes, 0.0)
            a = self.attributes(a.flatten(-2).to(dtype))
        else:
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
        if self.spatial_context is None:
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
        else:
            for value in (position_grid, state, history_context, task_intervals):
                if value.device != attributes.device or not value.is_floating_point():
                    raise ValueError("spatial relation contexts must be colocated floating tensors")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite supported spatial relation context")
            robot = self.robot(state.to(dtype))[:, None, None].expand(-1, k, c, -1)
            role = self.view(self.role_basis.to(dtype))[None, None].expand(b, k, -1, -1)
            task = self.task(task_intervals.to(dtype))[:, :, None, None].expand(-1, -1, k, c, -1)
            obj = a[:, None].expand(-1, 4, -1, -1, -1)
            robot = robot[:, None].expand_as(obj)
            role = role[:, None].expand_as(obj)
            with torch.autocast(device_type=a.device.type, enabled=False):
                features = F.linear(
                    F.silu(F.linear(position_grid.float(), self.coordinates[0].weight.float())),
                    self.coordinates[2].weight.float(),
                )
                context = F.linear(torch.cat((obj, robot, role, task), -1).float(),
                                   self.spatial_context.weight.float())
                xy = self._spatial_expectation(features, context, p)
            rel = self.relation(torch.cat((obj, xy.to(obj.dtype), robot, role), -1))
            rel = rel * (1.0 + torch.tanh(self.history(history_context.to(dtype))))[:, None, None, None]
        # No relation-only or task-only additive value bypass. This
        # constrains representation ownership, not a robot behavior rule.
        values = self.interaction(
            torch.cat((rel * task, rel * torch.tanh(task), task * torch.tanh(rel)), -1)
        )
        values = torch.where(valid[:, None, :, :, None], values, 0.0)
        result = TaskRelationEvidence(
            values, valid, binding, self.camera_names, current_content, state,
            execution_mode=self.task_execution_mode,
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
        self, *, hidden: int, content_dim: int, heads: int, camera_names: tuple[str, ...],
        task_execution_mode: str = JOINT_TASK_EXECUTION,
    ) -> None:
        super().__init__()
        if task_execution_mode not in JOINT_TASK_EXECUTION_MODES:
            raise ValueError("unknown task outcome execution mode")
        self.task_execution_mode = task_execution_mode
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

        self.effect_context_key = (
            nn.Linear(2 * hidden, hidden, bias=False)
            if task_execution_mode == JOINT_SPATIAL_TASK_EXECUTION else None
        )

    def _effect_comparison(self, expected: Tensor, predicted: Tensor) -> tuple[Tensor, Tensor]:
        """Compare effects after a SHARED nonlinear encoding, before I/K/C pooling.

        Both operands are online predictions, not labels or observed progress.
        Equal operands give exactly zero discrepancy; swapping negates it.
        The midpoint is selector-only, never an additive motor/value shortcut.
        Keep projections in FP32 so BF16 does not round large similar operands
        to equality before the discrepancy is formed.
        """
        if self.effect_context_key is None:
            raise ValueError("paired effect comparison belongs to the v2 task graph")
        if expected.shape != predicted.shape or expected.shape[-1] != 2 * self.hidden:
            raise ValueError("paired effects must share their content/image chart")
        if expected.device != predicted.device:
            raise ValueError("paired effects must share a device")
        with torch.autocast(device_type=expected.device.type, enabled=False):
            pair = torch.stack((expected.float(), predicted.float()), -2)
            encoded = F.linear(F.silu(F.linear(pair, self.gap_model[0].weight.float())),
                               self.gap_model[2].weight.float())
            gap = encoded[..., 0, :] - encoded[..., 1, :]
            midpoint = 0.5 * (expected.float() + predicted.float())
            difference = expected.float() - predicted.float()
            key = F.linear(difference, self.effect_key.weight.float())
            key = key + F.linear(midpoint, self.effect_context_key.weight.float())
        return gap, key

    def forward(
        self,
        query: Tensor,
        relation: TaskRelationEvidence,
        expected: OperationExpectation,
        predicted: FutureObjectDynamics,
    ) -> tuple[Tensor, Tensor]:
        relation.validate(hidden=self.hidden)
        if relation.execution_mode != self.task_execution_mode:
            raise ValueError("task comparison cannot mix execution graphs")
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
        if self.effect_context_key is None:
            sem = self.semantic((expected_sem.float() - predicted_sem.float()).to(dtype))
            sem = sem[:, :, :, None].expand(-1, -1, -1, len(self.camera_names), -1)
            image = self.image((expected_img.float() - predicted_img.float()).to(dtype))
            # Keep content and image errors distinct until a nonlinear joint read.
            # Quarantine the common support BEFORE any trainable projection or
            # product. Masking only the final value can still yield NaN * 0 in
            # Linear weight gradients from an unavailable relation/view.
            relation_values = torch.where(support[..., None], relation.values, 0.0)
            gap = self.gap_model(torch.cat((sem, image), -1))
            gap = gap * (1.0 + torch.tanh(self.task_modulation(relation_values.to(dtype))))
            gap = torch.where(support[..., None], gap, 0.0)
            key = self.relation_key(relation_values.to(dtype)) + self.effect_key(
                torch.cat((sem, image), -1)
            )
        else:
            with torch.autocast(device_type=query.device.type, enabled=False):
                semantic_pair = F.linear(torch.stack((expected_sem, predicted_sem), -2).float(),
                                         self.semantic.weight.float())
                semantic_pair = semantic_pair[:, :, :, None].expand(
                    -1, -1, -1, len(self.camera_names), -1, -1)
                image_pair = F.linear(torch.stack((expected_img, predicted_img), -2).float(),
                                      self.image.weight.float())
                effects = torch.cat((semantic_pair, image_pair), -1)
                effects = torch.where(support[..., None, None], effects, 0.0)
                relation_values = torch.where(support[..., None], relation.values, 0.0)
                gap, effect_key = self._effect_comparison(effects[..., 0, :], effects[..., 1, :])
            gap = gap * (1.0 + torch.tanh(self.task_modulation(relation_values.to(dtype))))
            gap = torch.where(support[..., None], gap, 0.0)
            key = self.relation_key(relation_values.to(dtype)) + effect_key
        key = key + self.time_code.to(key)[None, :, None, None]
        mass = relation.binding.mass
        target = self.target_relation(query, key, relation_values, support, mass)
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
        # One shared K+null law. Linear([object, task]) is separable: the
        # task half adds the same scalar to every K and cannot change real-K
        # log odds. Keep this zero-start owner/shape, but feed genuine cross
        # features in forward. The ABI declares the changed weight semantics.
        self.task_object_score = nn.Linear(2 * hidden, 1, bias=False)
        nn.init.zeros_(self.task_object_score.weight)
        self.null = nn.Linear(hidden, 1)

    def forward(
        self, task: Tensor, objects: Tensor, supported: Tensor, *, history: Tensor,
        view_support: Tensor | None = None,
    ) -> TargetBinding:
        if (
            task.ndim != 3
            or objects.ndim not in (3, 4)
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
        per_view = objects.ndim == 4
        object_count = objects.shape[1]
        if per_view:
            if view_support is None or view_support.shape != objects.shape[:3] or view_support.dtype != torch.bool:
                raise ValueError("camera-conditioned binding requires producer view support")
            legal = view_support & supported[...,None]
            safe = torch.where(legal[...,None], objects, 0.0).flatten(1,2)
        else:
            if view_support is not None:
                raise ValueError("pooled binder cannot acquire a fabricated camera axis")
            safe = torch.where(supported[..., None], objects, 0.0)
        obj = self.objects(safe)
        query = self.query_norm(obj + self.history_query(history)[:, None])
        task_value = self.task_value(task)
        read, _ = self.task_read(query, self.task_norm(task_value), task_value, need_weights=False)
        object_score = self.score(read * torch.tanh(self.compatibility(obj)))[..., 0]
        task_context = task_value.mean(1)[:, None].expand(-1, obj.shape[1], -1)
        task_object_score = self.task_object_score(torch.cat(
            (obj * task_context, obj * torch.tanh(task_context)), dim=-1
        ))[..., 0]
        score = object_score + task_object_score
        if per_view:
            score = score.reshape(task.shape[0], object_count, -1).float()
            # Log-mean-exp is permutation equivariant and does not reward an
            # object solely for having more available cameras. No view quota.
            score = torch.logsumexp(torch.where(legal, score, -1e9), -1) - legal.sum(-1).clamp_min(1).float().log()
            supported = supported & legal.any(-1)
            score = torch.where(supported, score, 0.0)
        return TargetBinding.from_logits(score, self.null(task_value.mean(1)), supported)
